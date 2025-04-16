import yaml
import sys
import os 
import io
import pandas as pd
import numpy as np
import torch as th 
import wandb
import pickle
from tqdm import tqdm
import random 
import argparse
from collections import Counter, defaultdict 
from copy import deepcopy
from datetime import datetime

from sklearn.model_selection import KFold

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.data_loader import load_or_cache_data 
from src.utility.ut_general import name_to_model, dict_to_namespace, namespace_to_dict, relabel_train_outcome, recover_original_label, normalizing_factors
from src.utility.ut_model import drop_edges, mask_node_features, add_noisy_node_features
from src.data.scaling import MeanStdScaler
from src.data.KNN_imputer import KNNImputer_with_OneHotEncoding


def preprocess_dataset(fmri_data, config, fmri_outcomes = None, time_string = None): 
    
    # Enable scaling: 
    if config.scaling.enabled: 
        scaler_type = config.scaling.scaler 
        if scaler_type == "MeanStd": 
            scaler = MeanStdScaler()

    if fmri_outcomes is not None:

        train_fmri_sorted = fmri_data.sort_values(by="participant_id").reset_index(drop=True) # sort fmri matrix 
        train_outcome_sorted = fmri_outcomes.sort_values(by="participant_id").reset_index(drop=True) # sort label 

        # Task = four, adhd, or sex 
        train_label = relabel_train_outcome(train_outcome_sorted, task=config.task) 

        # Double check if label and train participant_id matches 
        assert np.array_equal(train_fmri_sorted["participant_id"].values, train_label["participant_id"].values)
        
        label_sorted = train_label["Label"]
        participant_ids = train_fmri_sorted["participant_id"].values

        # Double check: 
        assert np.array_equal(train_fmri_sorted["participant_id"].values, participant_ids)

        # fmri_unscaled_indexed = train_fmri_sorted.set_index("participant_id")
        fmri_unscaled_dropped = train_fmri_sorted.drop(columns = "participant_id")

        # Fit the scaler after balancing 
        if scaler is None: 
            raise ValueError("You need to have scaling") 
        else: 
            scaler_type = config.scaling.scaler
            pickled_scaling_file = os.path.join(config.checkpoint_dir, config.model_name, time_string, f"scaler_{scaler_type}.pth")
            scaler.fit(fmri_unscaled_dropped)
            # Save the scaler information into a pickle file to re-use during testing: 
            with open(pickled_scaling_file, "wb") as f:
                pickle.dump(scaler, f)

        fmri_scaled = scaler.transform(fmri_unscaled_dropped)

        return fmri_scaled, label_sorted, participant_ids
    
    else: 
        participant_ids = fmri_data["participant_id"].values 
        fmri_dropped = fmri_data.drop(columns = "participant_id")
        
        # Scaling the test set to make it distribute the same
        if scaler is None: 
            raise ValueError("You need to have scaling")
        else: 
            scaler_type = config.scaling.scaler
            pickled_scaling_file = os.path.join(config.path_to_checkpoint_folder, f"scaler_{scaler_type}.pth")
            with open(pickled_scaling_file, "rb") as f:
                scaler = pickle.load(f)  
        fmri_scaled = scaler.transform(fmri_dropped)

        return fmri_scaled, None, participant_ids

def make_lovely_matrix(fmri_row, used_to_build_node_feature = False): 
    m = np.zeros((200,200))  

    for col_name, val in fmri_row.items(): 
        if col_name == "participant_id": 
            continue

        node1, node2_temp = col_name.split("throw_")
        node2 = node2_temp.split("thcolum")[0]

        if val > 0: 
            m[int(node1)][int(node2)] = val 
        elif val < 0: 
            if used_to_build_node_feature: 
                m[int(node2)][int(node1)] = val 
            else: 
                m[int(node2)][int(node1)] = abs(val) 

    return m 

def create_graph_lst(fmri, config, participant_ids, label = None): 
    """Make directional graphs stored in a graph lst"""

    graph_lst = [] 
    graph_num, node_num =  len(fmri), 200 # node_num hard-coded here 

    print("Begin loading patient data ... Making directional graphs")
    for i in tqdm(range(graph_num)): 

        matrix = make_lovely_matrix(fmri.iloc[i,:])

        # Use the correlation for each node as the node feature: 
        if config.node_features.correlation_matrix: 
            matrix = make_lovely_matrix(fmri.iloc[i,:], used_to_build_node_feature = True)
            matrix_tensor = th.tensor(matrix).float()

            # Mask node features to prevent overfitting if needed: 
            if config.node_features.masking: 
                mask = th.bernoulli(th.full_like(matrix_tensor, 1 - config.node_features.mask_prob))
                x = matrix_tensor * mask 

            x = matrix_tensor.clone()

        # Use identity matrix as the node features: 
        elif config.node_features.identity: 
            x = th.eye(node_num) 

        # Direction --> sign on the matrix 
        edge_inx = matrix.nonzero()
        edge_inx = [[edge_inx[0][i], edge_inx[1][i]] for i in range(len(edge_inx[0]))]
        edge_attr = th.FloatTensor([matrix[idx[0], idx[1]] for idx in edge_inx])
        x = th.eye(node_num) 

        graph_data = Data(x = x.clone(), # 200 x 200 identity matrix for all node features 
                          edge_index = th.LongTensor(edge_inx).transpose(1,0), 
                          edge_attr = edge_attr.clone().detach(), 
                          y = label[i] if label is not None else None, 
                          participant_id = participant_ids[i])

        graph_lst.append(graph_data) 

    return graph_lst 


def make_lovely_unidirectional_matrix(fmri_row, pos = True): 
    """Helper function to make unidirectional graphs"""
    m = np.zeros((200,200))
    for col_name, val in fmri_row.items(): 
        if col_name == "participant_id": 
            continue
        node1, node2_temp = col_name.split("throw_")
        node2 = node2_temp.split("thcolum")[0]

        if pos: 
            if val > 0: 
                m[int(node1)][int(node2)] = val 
        else: 
            if val < 0: 
                m[int(node1)][int(node2)] = abs(val) 
            
    return m 

def create_unidirectional_graph_lst(fmri, config, participant_ids, label = None): 
    """Make two unidirectional graph (positive and negative) stored in two graph lists"""

    graph_lst_pos, graph_lst_neg = [],[] 
    graph_num, node_num = len(fmri), 200 # 200 is hard-coded 

    # fmri_connect = fmri.drop(columns = "participant_id")
    # cont_matrix = th.tensor(fmri_connect.values).float() 

    print("Begin loading patient data ... Making unidirectional graphs")
    for i in tqdm(range(graph_num)): 
        
        # Use the correlation for each node as the node feature: 
        if config.node_features.correlation_matrix: 
            matrix = make_lovely_unidirectional_matrix(fmri.iloc[i,:])
            matrix_tensor = th.tensor(matrix).float()

            # Mask node features to prevent overfitting if needed: 
            if config.node_features.masking: 
                mask = th.bernoulli(th.full_like(matrix_tensor, 1 - config.node_features.mask_prob))
                x = matrix_tensor * mask 
            x = matrix_tensor.clone()

        # Use identity matrix as the node features: 
        elif config.node_features.identity: 
            x = th.eye(node_num) 
        
        matrix_pos = make_lovely_unidirectional_matrix(fmri.iloc[i,:])
        matrix_neg = make_lovely_unidirectional_matrix(fmri.iloc[i,:], pos = False)

        # create positive graphs 
        edge_inx_pos = matrix_pos.nonzero()
        edge_inx_pos = [[edge_inx_pos[0][i], edge_inx_pos[1][i]] for i in range(len(edge_inx_pos[0]))]
        edge_attr_pos = th.FloatTensor([matrix_pos[idx[0], idx[1]] for idx in edge_inx_pos])

        graph_data_pos = Data(x = x.clone(), 
                          edge_index = th.LongTensor(edge_inx_pos).transpose(1,0), 
                          edge_attr = edge_attr_pos.clone().detach(), 
                          y = label[i] if label is not None else None, 
                          participant_id = participant_ids[i])
        
        # create negative graphs 
        edge_inx_neg = matrix_neg.nonzero()
        edge_inx_neg = [[edge_inx_neg[0][i], edge_inx_neg[1][i]] for i in range(len(edge_inx_neg[0]))]
        edge_attr_neg = th.FloatTensor([matrix_neg[idx[0], idx[1]] for idx in edge_inx_neg])

        graph_data_neg = Data(x = matrix_tensor.clone(), 
                          edge_index = th.LongTensor(edge_inx_neg).transpose(1,0), 
                          edge_attr = edge_attr_neg.clone().detach(), 
                          y = label[i] if label is not None else None, 
                          participant_id = participant_ids[i])

        graph_lst_pos.append(graph_data_pos) 
        graph_lst_neg.append(graph_data_neg) 
    
    return graph_lst_pos, graph_lst_neg

    
def train(model, optimizer, criterion, train_loader = None, unidirectional = False, train_loader_pos = None, train_loader_neg = None): 

    # This below line needs to be checked: 
    if not train_loader and not unidirectional: 
        raise ValueError("Something is wrong with train data loading") 

    model.train() 
    total_loss, total_samples, correct = 0, 0, 0

    if not unidirectional: 
        for data in tqdm(train_loader): 
            optimizer.zero_grad() 
            out = model(data)
            loss = criterion(out, data.y) 
            loss.backward()
            optimizer.step() 
            total_loss += loss.item() 
            predictions = out.argmax(dim=1) 
            correct += (predictions == data.y).sum().item()
            total_samples += data.y.size(0)
            
            print("prediction after training:", predictions)

        train_loss = total_loss / len(train_loader)   
        train_accuracy = (correct / total_samples) * 100  

    else: 
        # tqdm doesn't work here. 
        for data_pos, data_neg in tqdm(zip(train_loader_pos, train_loader_neg)): 
            optimizer.zero_grad() 
            out = model(data_pos, data_neg)
            loss = criterion(out, data_pos.y) # either data_pos or data_neg has the same y  
            loss.backward()
            optimizer.step() 
            total_loss += loss.item() 
            predictions = out.argmax(dim=1) 
            correct += (predictions == data_pos.y).sum().item() # again, either data_pos or neg has the same u 
            total_samples += data_pos.y.size(0)
            print("prediction after training:", predictions)

        train_loss = total_loss / len(train_loader_pos) # train_loader_pos and train_loader_neg has the same length    
        train_accuracy = (correct / total_samples) * 100  

    return train_loss, train_accuracy 

def validate(model, criterion, val_loader = None, unidirectional = False, val_loader_pos = None, val_loader_neg = None): 

    if not val_loader and not unidirectional: 
        raise ValueError("Something is wrong with validation data loading") 

    model.eval()
    total_loss, correct, total_samples = 0, 0, 0 

    if not unidirectional: 
        with th.no_grad():  
            for data in val_loader:
                out = model(data) 
                loss = criterion(out, data.y)
                total_loss += loss.item()
                predictions = out.argmax(dim=1)
                correct += (predictions == data.y).sum().item()            
                total_samples += data.y.size(0)
                print("prediction after validation:", predictions)

        test_loss = total_loss / len(val_loader)
        test_accuracy = (correct / total_samples) * 100

    else: 
        with th.no_grad():  
            for data_pos, data_neg in zip(val_loader_pos, val_loader_neg):
                out = model(data_pos, data_neg) 
                loss = criterion(out, data_pos.y) # either data_pos or data_neg's y should be fine. They are the same
                total_loss += loss.item()
                predictions = out.argmax(dim=1)
                correct += (predictions == data_pos.y).sum().item()            
                total_samples += data_pos.y.size(0)
                print("prediction after validation:", predictions)
        
        test_loss = total_loss / len(val_loader_pos) # again, either val_loader_pos or val_loader_negative is fine 
        test_accuracy = (correct / total_samples) * 100

    return test_loss, test_accuracy

def add_metadata_to_graph_lst(graph_lst, config):
    
    rootfolder = config.root_folder 
    sys.path.append(os.path.join(rootfolder))
    datafolder = os.path.join(rootfolder, "data")
    
    pickle_file = os.path.join(datafolder, "data.pkl") 
    train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)
    
    train_data_dic.update(test_data_dic)
    
    imputer = KNNImputer_with_OneHotEncoding() 

    metadata = imputer.fit_transform(train_data_dic, split="train" if args.train_config else "test")

    def encode_column_into_bins(df, column, bins, labels):
        df['binned'] = pd.cut(df[column], bins=bins, labels=labels, right=True)
        df_encoded = pd.get_dummies(df, columns=['binned'], prefix='', prefix_sep='', dtype=float)
        df_encoded = df_encoded.drop(columns=[column])
        return df_encoded

    # Special case handedness column
    metadata['EHQ_EHQ_Total'] = metadata['EHQ_EHQ_Total'] / 200 + 0.5
    metadata = encode_column_into_bins(metadata, 'ColorVision_CV_Score' , [0, 12, 100], ['Color_Blind', 'Normal_Vision'])
    metadata = encode_column_into_bins(metadata, 'MRI_Track_Age_at_Scan' , [0, 4, 11, 17, 30], ['Infant', 'Child', "Adolescent", "Adult"])

    # Normalize everything to be between 0 and 1
    # See utility/ut_general
    for col in normalizing_factors:
        metadata[col] /= normalizing_factors[col]

    # Remove features in train which never appear
    columns_which_dont_appear_in_train = ["Basic_Demos_Study_Site_5", "PreInt_Demos_Fam_Child_Race_-1", 
                                          "Barratt_Barratt_P1_Edu_-1", "Barratt_Barratt_P1_Occ_-1", 
                                          "Barratt_Barratt_P2_Edu_-1", "Barratt_Barratt_P2_Occ_-1", 
                                          "Infant"]
    for col in columns_which_dont_appear_in_train:
        metadata = metadata.drop(columns=[col])
    
    raw_values = metadata.values
    
    print("Adding metadata to each graph ...")
    for datapoint in tqdm(graph_lst):
        value_index = metadata.index.tolist().index(datapoint.participant_id)
        datapoint.metadata = th.FloatTensor(raw_values[value_index, :])

        # De-bugging:
        # print("De-bugging: Check if the feature is assigned to correct participant_id and labels")
        # print("datapoint.participant_id", datapoint.participant_id)
        # print("metadata.iloc.value.index \n", metadata.iloc[value_index,:])
        # print("datapoint: \n", datapoint)
        # print("datapoint.x: ", datapoint.x[198])
        # print("metadata: \n", datapoint.metadata)
        # print("edge index: \n", datapoint.edge_index)
        # print("edge attr: \n", datapoint.edge_attr)


def balancing_trainning_graph_lst(graph_lst, seed = 3407): 

    print("Start balancing trainning data ...")

    random.seed(seed)
    print("Balancing the trainning set only --> Not the validation set")
    label_counts = Counter([int(graph.y.item()) for graph in graph_lst])
    max_count = max(label_counts.values()) # count for largest class 

    class_to_graphs = defaultdict(list) # store labels and graphs 
    for graph in graph_lst:
        label = int(graph.y.item())
        class_to_graphs[label].append(graph)

    balanced_graph_lst = []

    for label, graphs in class_to_graphs.items():
        # print("label", label)
        # print("graphs", graphs)
        n_to_add = max_count - len(graphs)
        balanced_graph_lst.extend(graphs) # add the current graph to the lst 

        if n_to_add > 0:
            sampled = random.choices(graphs, k=n_to_add)  
            for graph in sampled: 
                g_aug = deepcopy(graph) 
                g_aug = drop_edges(g_aug, drop_prob=0.1)
                graph = mask_node_features(graph, mask_prob=0.1) 
                graph = add_noisy_node_features(graph, noise_level=0.01)
                balanced_graph_lst.append(g_aug)

    # random.shuffle(balanced_graph_lst) 
    return balanced_graph_lst 


def cross_validation(model, train_fmri, train_outcomes, config, time_string): 

    kfold = KFold(n_splits = config.num_folds, shuffle=True, random_state=config.master_seed)
    log_messages = [] 
    lowest_val_loss_global = float("inf")
    
    para_message = (f"""
        Model: {config.model_name},
        Initial Learning Rate: {config.lr},
        {config.num_folds} fold cross validation each with {config.num_epochs} epochs
        Batch size {config.batch_size}, 
        Master seed: {config.master_seed},
        With metadata: {config.add_metadata}, 
        Unidirectional graph: {config.model_params.undirectional_graph}
        """)

    if config.model_name == "DirGNN_GatConv_model": 
        model_specific_message = (f"""
            # DirGNN-GATv2 specific parameters: 
            num_layers_GATv2: {config.model_params.num_layers_GATv2},
            num_layers_DirGNN: {config.model_params.num_layers_DirGNN},
            """)

    print(para_message)
    log_messages.append(para_message)

    for fold, (train_inx, val_inx) in enumerate(kfold.split(train_fmri)): 

        # De-bugging: 
        assert len(set(train_inx) & set(val_inx)) == 0
        assert len(train_inx) + len(val_inx) == len(train_fmri)
        
        lowest_val_loss = float("inf")
        lowest_valloss_epoch = None 
        patience_count = 0 

        # Just use one fold first 
        # if fold > 0: 
        #     print("Let's just break the loop at fold 1")
        #     break 

        # Set up models 
        best_model_buffer = io.BytesIO() 
        model_class = name_to_model.get(config.model_name)  # Safely get class reference
        model = model_class(config)
        optimizer = th.optim.Adam(model.parameters(), lr = config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        loss_weights = th.tensor(config.model_params.loss_weights, dtype = th.float) 
        criterion = th.nn.CrossEntropyLoss(weight=loss_weights, 
                    label_smoothing = config.model_params.label_smoothing) 

        # Prepare dataset: 
        train_fmri_preprocessed, train_label, all_participant_ids = preprocess_dataset(
                fmri_data = train_fmri, 
                config = config, 
                fmri_outcomes = train_outcomes, 
                time_string = time_string) 
        
        # split the train and validation set: 
        train_fmri_unbalanced = train_fmri_preprocessed.loc[train_inx]

        train_participant_ids = all_participant_ids[train_inx]
        train_label_tensor = th.tensor(train_label[train_inx].to_numpy(dtype=np.int16), dtype=th.long) 

        val_fmri = train_fmri_preprocessed.loc[val_inx]
        val_label = train_label[val_inx]
        val_participant_ids = all_participant_ids[val_inx]
        val_label_tensor = th.tensor(val_label.to_numpy(dtype=np.int16), dtype=th.long) 
        
        # create graph lst for train and validation set separately: 
        if not config.model_params.undirectional_graph: # Only GCN model has this, and there is a check to make sure GCN model has not undire. graphs 
            graph_lst_train = create_graph_lst(fmri = train_fmri_unbalanced, 
                                            config = config,
                                            participant_ids = train_participant_ids, 
                                            label = train_label_tensor)
            graph_lst_val =  create_graph_lst(fmri = val_fmri, 
                                        config = config, 
                                        participant_ids = val_participant_ids, 
                                        label = val_label_tensor)
            
            if config.resampling_enabled: 
                graph_lst_train = balancing_trainning_graph_lst(graph_lst_train)
                
            if config.add_metadata:
                add_metadata_to_graph_lst(graph_lst_train, config)
                add_metadata_to_graph_lst(graph_lst_val, config)

            train_loader = DataLoader(graph_lst_train, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(graph_lst_val, batch_size=config.batch_size, shuffle=True)

        else: 
            graph_lst_train_pos, graph_lst_train_neg = create_unidirectional_graph_lst(
                fmri = train_fmri_unbalanced, 
                config = config, 
                participant_ids = train_participant_ids, 
                label = train_label_tensor)
            graph_lst_val_pos, graph_lst_val_neg = create_unidirectional_graph_lst(
                    fmri = val_fmri, 
                    config = config,
                    participant_ids = val_participant_ids, 
                    label = val_label_tensor)
            
            if config.resampling_enabled: 
                graph_lst_train_pos = balancing_trainning_graph_lst(graph_lst_train_pos)
                graph_lst_train_neg = balancing_trainning_graph_lst(graph_lst_train_neg)
                
            if config.add_metadata:
                add_metadata_to_graph_lst(graph_lst_train_pos, config)
                add_metadata_to_graph_lst(graph_lst_val_pos, config)
                add_metadata_to_graph_lst(graph_lst_train_neg, config)
                add_metadata_to_graph_lst(graph_lst_val_neg, config)

            train_loader_pos = DataLoader(graph_lst_train_pos, batch_size=config.batch_size, shuffle=True)
            val_loader_pos = DataLoader(graph_lst_val_pos, batch_size=config.batch_size, shuffle=True)

            train_loader_neg = DataLoader(graph_lst_train_neg, batch_size=config.batch_size, shuffle=True)
            val_loader_neg = DataLoader(graph_lst_val_neg, batch_size=config.batch_size, shuffle=True)
        
        message0 = f"Start training and validating for fold = {fold}"
        log_messages.append(message0)

        print("Finish processing data... Upload to W&B ...")
        if config.wandb.enabled: 
            run = wandb.init(
                project=config.wandb.project,
                group=f"{config.model_name}-{config.task}-{time_string}",
                name=f"Fold {fold}",
                config=config
            ) 

        for epoch in range(config.num_epochs):

            if config.model_name != "GCN_model" or not config.model_params.undirectional_graph: 
                train_loss, train_accuracy = train(model = model, criterion = criterion, optimizer = optimizer, 
                                                   train_loader = train_loader, 
                                                   unidirectional= False, 
                                                   train_loader_pos = None, train_loader_neg = None)

                val_loss, val_accuracy = validate(model = model, criterion = criterion, val_loader = val_loader, 
                                                  unidirectional = False, 
                                                  val_loader_pos = None, val_loader_neg = None)
                
            elif config.model_name == "GCN_model" and config.model_params.undirectional_graph: 
                train_loss, train_accuracy = train(model = model, optimizer = optimizer, criterion = criterion,
                                                    train_loader = None, unidirectional = True, 
                                                    train_loader_pos = train_loader_pos, 
                                                    train_loader_neg = train_loader_neg)
                val_loss, val_accuracy = validate(model = model, criterion = criterion, val_loader = None, 
                                                  unidirectional = True, 
                                                  val_loader_pos = val_loader_pos, 
                                                  val_loader_neg = val_loader_neg)
            else: 
                raise ValueError("Something is wrong with model setting. R u using GCN? R u using unidirectional?")
            
            scheduler.step(val_loss) 
            updated_lr = optimizer.param_groups[0]['lr'] 

            message1 = f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Updated Learning Rate: {updated_lr}"
            message2 = f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%" 
            print(message1)
            print(message2)
            log_messages.append(message1)
            log_messages.append(message2)

            if config.wandb.enabled:
                run.log({"train_acc": train_accuracy,
                        "train_loss": train_loss,
                        "val_acc": val_accuracy, 
                        "val_loss": val_loss, 
                        "current_learning_rate": updated_lr})

            if val_loss <= lowest_val_loss: 
                lowest_val_loss = val_loss
                lowest_valloss_epoch = epoch 
                patience_count = 0 

                # Save the best paras among all k-folds 
                if lowest_val_loss <= lowest_val_loss_global:
                    best_model_buffer.seek(0)  
                    th.save(model.state_dict(), best_model_buffer)

            else: 
                patience_count += 1
            
            if patience_count > config.patience: 
                log_messages.append(f"Validation loss reached a plateau at epoch {lowest_valloss_epoch}, break")
                break
        
        if config.wandb.enabled:
            run.finish()
    
    best_model_buffer.seek(0)
    with open(os.path.join(config.checkpoint_dir, config.model_name, time_string, "checkpoint.pth"), "wb") as f:
        f.write(best_model_buffer.read())
    
    with open(os.path.join(config.checkpoint_dir, config.model_name, time_string, "train_params.yaml"), "w") as f:
        saved_config = namespace_to_dict(config)
        yaml.dump(saved_config, f, default_flow_style=False, sort_keys=False)

    logfile = os.path.join(config.checkpoint_dir, config.model_name, time_string, "log.txt")
    with open(logfile, "w") as f:
        f.write("\n".join(log_messages) + "\n")
        
    return log_messages 

def run_inference(data, path_to_checkpoint_folder):
    with open(os.path.join(path_to_checkpoint_folder, "train_params.yaml"), "r") as file:
        config = yaml.safe_load(file)
        
    config = dict_to_namespace(config)
    model_class = name_to_model.get(config.model_name, "GCN_Model")  # Safely get class reference
    model = model_class(config)
    
    model.load_state_dict(th.load(os.path.join(path_to_checkpoint_folder, "checkpoint.pth")))
    model.eval()
    
    test_loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)
    
    if config.task.lower() == "four": 
        with open(os.path.join(path_to_checkpoint_folder, "final_predictions.csv"), "w") as f:
            f.write("participant_id,ADHD_Outcome,Sex_F\n")
            result_string = ""
            with th.no_grad():  
                for data in tqdm(test_loader):
                    out = model(data) 
                    predictions = out.argmax(dim=1)
                    for i in range(len(predictions)):
                        ADHD_outcome, Sex_F = recover_original_label(predictions[i])
                        result_string += f"{data.participant_id[i]},\t{ADHD_outcome},\t{Sex_F}\n"
            f.write(result_string[:-1]) # Remove last newline
    
    # Two class predictions: 
    else: 
        with open(os.path.join(path_to_checkpoint_folder, f"final_predictions_{config.task}.csv"), "w") as f:
            f.write(f"participant_id,{config.task}\n")
            result_string = ""
            with th.no_grad():  
                for data in tqdm(test_loader):
                    out = model(data) 
                    predictions = out.argmax(dim=1)

                    if config.task == "adhd": 
                        for i in range(len(predictions)):
                            result_string += f"{data.participant_id[i]},\t{ADHD_outcome}\n"
                    else: # config.task == "sex" 
                        for i in range(len(predictions)):
                            result_string += f"{data.participant_id[i]},\t{Sex_F}\n"
                            
            f.write(result_string[:-1]) # Remove last newline


def check_config_files(config): 
    """A group to make sure the configuration files parameters are compatible"""

    if config.task.lower() != "four" and len(config.model_params.loss_weights) != 2: 
        raise ValueError("U idiot, change loss weights or check config.tasks!")
    
    if config.model_name != "GCN_model" and config.model_params.undirectional_graph: 
        raise ValueError("R u sure u want to use unidirectional graphs for non-GCN models?") 
    
    if config.model_name == "GCN_model" and not config.model_params.undirectional_graph: 
        raise ValueError("R u sure you dont want to use unidiretcional graohs for GCN model?")
    
    if config.predictor_paras.norm_enabled: 
        if config.predictor_paras.norm_timing != "before" or config.predictor_paras.norm_timing != "after": 
            raise ValueError("R need to specify a correct normalization timing -> before or after the linear projectors?")
    
    if not (config.node_features.identity or config.node_features.correlation_matrix):
        raise ValueError("C’mon... you gotta use at least one type of node feature :(")

def main(args):

    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    config_path = args.train_config if args.train_config else args.test_config

    if config_path == None:
        raise ValueError("U idiot, specify either a train or test config file")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config = dict_to_namespace(config)

    check_config_files(config)
    
    # Data and function loading: 
    rootfolder = config.root_folder # change this
    sys.path.append(os.path.join(rootfolder))
    # datafolder = os.path.join(rootfolder, "data")
    datafolder = rootfolder
    pickle_file = os.path.join(datafolder, "data.pkl") 
    train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)

    if args.test_config:
        test_fmri = test_data_dic["test_fmri"]
        
        test_data, _, participant_ids = preprocess_dataset(fmri_data = test_fmri, 
                                        config = config, 
                                        fmri_outcomes = None, 
                                        time_string = None) 

        graph_lst_test = create_graph_lst(fmri = test_data, 
                                          config = config, 
                                          participant_ids = participant_ids, 
                                          label = None)

        if config.add_metadata:
            add_metadata_to_graph_lst(graph_lst_test, config)

        print(f"Starting inference...")
        run_inference(graph_lst_test, config.path_to_checkpoint_folder)

    elif args.train_config:

        # Make the output dir first
        os.makedirs(os.path.join(config.checkpoint_dir, config.model_name, time_string), exist_ok=True)
        
        master_seed = config.master_seed
        th.manual_seed(master_seed)
        data_outcome = train_data_dic[f"train_outcome"]
        data_mri = train_data_dic[f"train_fmri"] 

        print(f"Starting {config.num_folds} fold cross-validation...")
        cross_validation(config.model_name, data_mri, data_outcome, config, time_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None)
    parser.add_argument("--test_config", type=str, default=None)
    args = parser.parse_args()
    main(args)