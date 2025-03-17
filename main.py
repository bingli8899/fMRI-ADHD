import argparse
import yaml
import sys
import os 
import io
import numpy as np
import torch as th 
import wandb

from sklearn.model_selection import KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 

from tqdm import tqdm
import random 
from imblearn.over_sampling import SMOTE #, ADASYN
from torch.optim.lr_scheduler import ReduceLROnPlateau 

from src.model.SimpleGNN import GCN_Model
from src.data.data_loader import load_data, load_or_cache_data
from src.utility.ut_GNNbranch import relabel_train_outcome
from src.utility.ut_general import name_to_model, dict_to_namespace, namespace_to_dict
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

def create_graph_lst(train_outcome, train_fmri, config): 

    graph_lst = [] 

    # sort by participant_id first so later could be re-sampeld 
    train_fmri_sorted = train_fmri.sort_values(by="participant_id")
    train_outcome_sorted = train_outcome.sort_values(by="participant_id") 
    train_fmri_sorted_connect = train_fmri_sorted.drop(columns = "participant_id")
    train_label = relabel_train_outcome(train_outcome_sorted)

    if (train_fmri_sorted["participant_id"].values != train_outcome_sorted["participant_id"].values).all(): 
        raise ValueError("Oh nooooo! Mismatch in participant id")

    smote = SMOTE(sampling_strategy="auto", random_state=config.master_seed)
    fmri_balanced, label_balanced = smote.fit_resample(train_fmri_sorted_connect, train_label["Label"])
    train_label_tensor = th.tensor(label_balanced.to_numpy(dtype=np.int16), dtype=th.long)

    graph_num, node_num =  train_label_tensor.shape[0], 200 # node_num hard-coded here 

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
                          y = train_label_tensor[i])

        graph_lst.append(graph_data) 

    return graph_lst 


def data_splitting(graph_lst, splitting_threshold = 0.7): 
    random.shuffle(graph_lst)

    if splitting_threshold >= 1 or splitting_threshold <= 0: 
        raise ValueError("What r u doing? 0 < Splitting threshold < 1")
    split_inx = int(len(graph_lst) * splitting_threshold) #8:2 train test splitting 
    train_data = graph_lst[:split_inx]
    test_data = graph_lst[split_inx:]
    return train_data, test_data 

    
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


def cross_validation(model, graph_lst, config): 

    kfold = KFold(n_splits = config.num_folds, shuffle=True, random_state=config.master_seed)
    log_messages = [] 
    
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

        if config.wandb:
            run = wandb.init(
                entity="bli283-university-of-wisconsin-madison",
                project="fmri-adhd",
                config=config) 


        train_data = [graph_lst[i] for i in train_inx]
        val_data = [graph_lst[i] for i in val_inx]
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)
        best_model_buffer = io.BytesIO() 

        model_class = name_to_model.get(config.model_name, "GCN_Model")  # Safely get class reference
        model = model_class(config)
        print(model)
        
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

            if config.wandb:
                run.log({"train_acc": train_accuracy,
                        "train_loss": train_loss,
                        "val_acc": val_accuracy, 
                        "val_loss": val_loss, 
                        "current_learning_rate": updated_lr})

            if val_loss <= lowest_val_loss: 
                lowest_val_loss = val_loss
                lowest_valloss_epoch = epoch 
                patience_count = 0 

                best_model_buffer.seek(0)  
                th.save(model.state_dict(), best_model_buffer)
            else: 
                patience_count += 1
            
            if patience_count > config.patience: 
                log_messages.append(f"Validation loss reached a plateau at epoch {lowest_valloss_epoch}, break")
                break
        
        if config.wandb:
            run.finish()
    
    best_model_buffer.seek(0)
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(os.path.join(config.checkpoint_dir, config.model_name, time_string), exist_ok=True)
    with open(os.path.join(config.checkpoint_dir, config.model_name, time_string, "checkpoint.pth"), "wb") as f:
        f.write(best_model_buffer.read())
    
    with open(os.path.join(config.checkpoint_dir, config.model_name, time_string, "train_params.yaml"), "w") as f:
        saved_config = namespace_to_dict(config)
        yaml.dump(saved_config, f, default_flow_style=False, sort_keys=False)

    logfile = os.path.join(config.checkpoint_dir, config.model_name, time_string, "log.txt")
    with open(logfile, "w") as f:
        f.write("\n".join(log_messages) + "\n")
        
    return log_messages 

def main(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
        

    config = dict_to_namespace(config)
    
    # Data and function loading: 
    rootfolder = config.root_folder # change this
    sys.path.append(os.path.join(rootfolder))
    #datafolder = os.path.join(rootfolder, "data")
    datafolder = rootfolder

    pickle_file = os.path.join(datafolder, "data.pkl") 
    train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)
    train_outcome = train_data_dic["train_outcome"]
    train_fmri = train_data_dic["train_fmri"] 
    test_fmri = test_data_dic["test_fmri"]

    master_seed = config.master_seed
    th.manual_seed(master_seed)

    
    graph_lst = create_graph_lst(train_outcome, train_fmri, config)
    
    print(f"Starting {config.num_folds} fold cross-validation...")
    cross_validation(config.model_name, graph_lst, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    main(args)