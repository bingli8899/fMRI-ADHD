import argparse
import yaml
import sys
import os 
import io
import pandas as pd
import numpy as np
import torch as th 
import wandb
import pickle

from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 

from tqdm import tqdm
import random 
from imblearn.over_sampling import SMOTE #, ADASYN
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from src.model.GCN import GCN_Model
from src.data.data_loader import load_or_cache_data # load_data 
from src.utility.ut_GNNbranch import relabel_train_outcome, recover_original_label
from src.utility.ut_general import name_to_model, dict_to_namespace, namespace_to_dict
from src.utility.ut_stats import select_top_columns_MutualInfo_4classes
from src.data.scaling import MeanStdScaler
from src.data.KNN_imputer import KNNImputer_with_OneHotEncoding
from src.utility.ut_general import normalizing_factors

from datetime import datetime

def make_lovely_matrix(fmri_row): 
    m = np.zeros((200,200))  
    for col_name, val in fmri_row.items(): 
        if col_name == "participant_id": 
            continue

        node1, node2_temp = col_name.split("throw_")
        node2 = node2_temp.split("thcolum")[0]

        if val > 0: 
            m[int(node1)][int(node2)] = val 
        elif val < 0: 
            m[int(node2)][int(node1)] = abs(val) 

    return m 

# If fmri_outcomes is None, it is the test data
def create_graph_lst(fmri_data, config, fmri_outcomes = None, scaler=None, time_string = None): 

    graph_lst = [] 
    scaler = MeanStdScaler() 

    if fmri_outcomes is not None:

        # sort by participant_id first so later could be re-sampeld 
        train_fmri_sorted = fmri_data.sort_values(by="participant_id")
        train_outcome_sorted = fmri_outcomes.sort_values(by="participant_id") 
        # train_fmri_sorted_connect = train_fmri_sorted.drop(columns = "participant_id")
        train_label = relabel_train_outcome(train_outcome_sorted)
        if (train_fmri_sorted["participant_id"].values != train_outcome_sorted["participant_id"].values).all(): 
            raise ValueError("Oh nooooo! Mismatch in participant id")
        
        # Step 1: Create a mapping from participant_id to unique numbers
        participant_mapping = {pid: idx for idx, pid in enumerate(train_fmri_sorted["participant_id"].unique())}

        # Step 2: Replace participant_id with assigned numbers
        train_fmri_sorted["participant_id_mapped"] = train_fmri_sorted["participant_id"].map(participant_mapping)

        # Drop original participant_id before SMOTE
        train_fmri_sorted = train_fmri_sorted.drop(columns=["participant_id"])

        # Balancing dataset using SMOTE: 
        smote = SMOTE(sampling_strategy="auto", random_state=config.master_seed)
        
        #train_fmri_sorted = train_fmri_sorted.set_index("participant_id")
        #print(train_fmri_sorted.index)
        fmri_balanced_unscaled, label_balanced = smote.fit_resample(train_fmri_sorted, train_label["Label"])
        
        # Step 4: Convert back to DataFrame
        fmri_balanced_df = pd.DataFrame(fmri_balanced_unscaled, columns=train_fmri_sorted.columns)

        # Step 5: Map back participant_id numbers to original IDs
        reverse_mapping = {v: k for k, v in participant_mapping.items()}  # Reverse the mapping
        fmri_balanced_cat_df = pd.concat([fmri_balanced_df, pd.Series(fmri_balanced_df["participant_id_mapped"].map(reverse_mapping), name="participant_id")], axis=1)
        
        train_label_tensor = th.tensor(label_balanced.to_numpy(dtype=np.int16), dtype=th.long)
        
        participant_ids = fmri_balanced_cat_df["participant_id"].values
        fmri_balanced_unscaled_dropped = fmri_balanced_cat_df.drop(columns = ["participant_id", "participant_id_mapped"])

        # Fit the scaler after balancing 
        if scaler is None: 
            raise ValueError("You need to have scaling") 
        else: 
            scaler_type = config.scaling.scaler
            pickled_scaling_file = os.path.join(config.checkpoint_dir, config.model_name, time_string, f"scaler_{scaler_type}.pth")
            scaler.fit(fmri_balanced_unscaled_dropped)
            # Save the scaler information into a pickle file to re-use during testing: 
            with open(pickled_scaling_file, "wb") as f:
                pickle.dump(scaler, f)

        fmri_balanced = scaler.transform(fmri_balanced_unscaled_dropped)

    else: # No balancing for test data

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
        fmri_balanced = scaler.transform(fmri_dropped)

    graph_num, node_num =  len(fmri_balanced), 200 # node_num hard-coded here 

    print("Begin loading patient data ...")
    for i in tqdm(range(graph_num)): 

        matrix = make_lovely_matrix(fmri_balanced.iloc[i,:])

        # Direction --> sign on the matrix 
        edge_inx = matrix.nonzero()
        edge_inx = [[edge_inx[0][i], edge_inx[1][i]] for i in range(len(edge_inx[0]))]
        edge_attr = th.FloatTensor([matrix[idx[0], idx[1]] for idx in edge_inx])
        x = th.eye(node_num) 

        graph_data = Data(x = x, # 200 x 200 identity matrix for all node features 
                          edge_index = th.LongTensor(edge_inx).transpose(1,0), 
                          edge_attr = edge_attr.clone().detach(), 
                          y = train_label_tensor[i] if fmri_outcomes is not None else None, 
                          participant_id = participant_ids[i])

        graph_lst.append(graph_data) 

    return graph_lst 

    
def train(train_loader, model, criterion, optimizer): 

    model.train() 
    total_loss, total_samples, correct = 0, 0, 0

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

    train_accuracy = (correct / total_samples) * 100  

    return total_loss / len(train_loader), train_accuracy 

def validate(val_loader, model, criterion): 

    model.eval()
    total_loss, correct, total_samples = 0, 0, 0 

    with th.no_grad():  
        for data in val_loader:
            out = model(data) 
            loss = criterion(out, data.y)
            total_loss += loss.item()
            predictions = out.argmax(dim=1)
            correct += (predictions == data.y).sum().item()            
            total_samples += data.y.size(0)

    test_loss = total_loss / len(val_loader)
    test_accuracy = (correct / total_samples) * 100

    return test_loss, test_accuracy

def add_metadata_to_graph_lst(graph_lst, config):
    
    rootfolder = config.root_folder 
    sys.path.append(os.path.join(rootfolder))
    datafolder = rootfolder
    
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
    columns_which_dont_appear_in_train = ["Basic_Demos_Study_Site_5", "PreInt_Demos_Fam_Child_Race_-1", "Barratt_Barratt_P1_Edu_-1", "Barratt_Barratt_P1_Occ_-1", "Barratt_Barratt_P2_Edu_-1", "Barratt_Barratt_P2_Occ_-1", "Infant"]
    for col in columns_which_dont_appear_in_train:
        metadata = metadata.drop(columns=[col])
    
    raw_values = metadata.values
    
    print("Adding metadata to each graph ...")
    for datapoint in tqdm(graph_lst):
        value_index = metadata.index.tolist().index(datapoint.participant_id)
        datapoint.metadata = th.FloatTensor(raw_values[value_index, :])

def cross_validation(model, graph_lst, config, time_string): 

    kfold = KFold(n_splits = config.num_folds, shuffle=True, random_state=config.master_seed)
    log_messages = [] 
    lowest_val_loss_global = float("inf")
    
    num_linear_predictors = 1 
    para_message = (f"""
        Model: {config.model_name},
        Number of Layers: {config.num_layers},
        Number of Linear Predictors: {num_linear_predictors},
        Initial Learning Rate: {config.lr},
        Dropout rate: {config.dropout},
        {config.num_folds} fold cross validation each with {config.num_epochs} epochs
        Batch size {config.batch_size}, 
        Master seed: {config.master_seed},
        Stopping patience: {config.patience}
        """
    )
    print(para_message)
    log_messages.append(para_message)

    for fold, (train_inx, val_inx) in enumerate(kfold.split(graph_lst)): 
        
        lowest_val_loss = float("inf")
        lowest_valloss_epoch = None 
        patience_count = 0 

        # Just use one fold first 
        # if fold > 0: 
        #     print("Let's just break the loop at fold 1")
        #     break 
    
        if config.wandb.enabled:
            run = wandb.init(
                project=config.wandb.project,
                group=f"{config.model_name} {time_string}",
                name=f"Fold {fold}",
                config=config
            ) 

        train_data = [graph_lst[i] for i in train_inx]
        val_data = [graph_lst[i] for i in val_inx]
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)
        best_model_buffer = io.BytesIO() 

        model_class = name_to_model.get(config.model_name, "GCN_Model")  # Safely get class reference
        model = model_class(config)
        optimizer = th.optim.Adam(model.parameters(), lr = config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        criterion = th.nn.CrossEntropyLoss() # multi-class

        message0 = f"Start training and validating for fold = {fold}"
        log_messages.append(message0)

        for epoch in range(config.num_epochs):
            train_loss, train_accuracy = train(train_loader, model, criterion, optimizer)
            val_loss, val_accuracy = validate(val_loader, model, criterion) 
            
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

    

def main(args):

    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    
    config_path = args.train_config if args.train_config else args.test_config
    if config_path == None:
        raise ValueError("U idiot, specify either a train or test config file")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    config = dict_to_namespace(config)
    
    # Data and function loading: 
    rootfolder = config.root_folder # change this
    sys.path.append(os.path.join(rootfolder))
    # datafolder = os.path.join(rootfolder, "data")
    datafolder = rootfolder
    pickle_file = os.path.join(datafolder, "data.pkl") 
    train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)

    if args.test_config:
        test_fmri = test_data_dic["test_fmri"]

        if config.scaling.enabled: 
            scaler_type = config.scaling.scaler
            if scaler_type == "MeanStd": 
                scaler = MeanStdScaler()

        graph_lst = create_graph_lst(test_fmri, config, fmri_outcomes = None, scaler = scaler, time_string = time_string)

        if config.add_metadata:
            add_metadata_to_graph_lst(graph_lst, config)

        print(f"Starting inference...")
        run_inference(graph_lst, config.path_to_checkpoint_folder)

    elif args.train_config:

        # Make the output dir first
        os.makedirs(os.path.join(config.checkpoint_dir, config.model_name, time_string), exist_ok=True)
        
        master_seed = config.master_seed
        th.manual_seed(master_seed)
        data_outcome = train_data_dic[f"train_outcome"]
        data_mri = train_data_dic[f"train_fmri"] 

        # Enable scaling: 
        if config.scaling.enabled: 
            scaler_type = config.scaling.scaler 
            if scaler_type == "MeanStd": 
                scaler = MeanStdScaler()

        graph_lst = create_graph_lst(data_mri, config, fmri_outcomes = data_outcome, scaler = scaler, time_string = time_string)
        
        # Add metadata 
        if config.add_metadata:
            add_metadata_to_graph_lst(graph_lst, config)

        print(f"Starting {config.num_folds} fold cross-validation...")
        cross_validation(config.model_name, graph_lst, config, time_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None)
    parser.add_argument("--test_config", type=str, default=None)
    args = parser.parse_args()
    main(args)