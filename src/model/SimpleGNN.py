# Let's use a single GNN without DGL to see the performance first 
# conda activate simpleGNN 

""" Improve this: 
1) Try different pooling 
2) Normalizaton 
3) jk (GCN)
4) 
"""


import sys
import os 
import io
import numpy as np
import torch as th 
import wandb
from sklearn.model_selection import KFold
from torch_geometric.nn import GCNConv, GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch.nn import Dropout, Linear
from tqdm import tqdm
import random 
from imblearn.over_sampling import SMOTE #, ADASYN
from torch.optim.lr_scheduler import ReduceLROnPlateau 


# Data and function loading: 
rootfolder = "/u/b/i/bingl/private/fMRI-AHDH/" # change this
sys.path.append(os.path.join(rootfolder))

from src.data.data_loader import load_data, load_or_cache_data
from src.utility.ut_GNNbranch import relabel_train_outcome, check_label_balance  

datafolder = os.path.join(rootfolder, "data")

pickle_file = os.path.join(datafolder, "data.pkl") 
train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)
train_outcome = train_data_dic["train_outcome"]
train_fmri = train_data_dic["train_fmri"] 
test_fmri = test_data_dic["test_fmri"]

master_seed = 12345
th.manual_seed(master_seed)

train_label = relabel_train_outcome(train_outcome)
check_label_balance(train_outcome) 
# clearly not balanced: 
# {'percentage_2': 0.48, 'percentage_0': 0.21, 'percentage_3': 0.18, 'percentage_1': 0.14} 

def make_lovely_matrix(fmri_row): 
    m = np.zeros((200,200))  
    for col_name, val in fmri_row.items(): 
        if col_name == "participant_id": 
            continue

        node1, node2_temp = col_name.split("throw_")
        node2 = node2_temp.split("thcolum")[0]
        # print(node1, node2) 

        if val > 0: 
            m[int(node1)][int(node2)] = val 
        elif val < 0: 
            m[int(node2)][int(node1)] = abs(val) 

    return m 

def create_graph_lst(train_outcome, train_fmri): 

    graph_lst = [] 
    # x_init = th.rand((200, 200))

    # sort by participant_id first so later could be re-sampeld 
    train_fmri_sorted = train_fmri.sort_values(by="participant_id")
    train_outcome_sorted = train_outcome.sort_values(by="participant_id") 
    train_fmri_sorted_connect = train_fmri_sorted.drop(columns = "participant_id")
    train_label = relabel_train_outcome(train_outcome_sorted)

    if (train_fmri_sorted["participant_id"].values != train_outcome_sorted["participant_id"].values).all(): 
        raise ValueError("Oh nooooo! Mismatch in participant id")

    smote = SMOTE(sampling_strategy="auto", random_state=master_seed)
    fmri_balanced, label_balanced = smote.fit_resample(train_fmri_sorted_connect, train_label["Label"])
    train_label_tensor = th.tensor(label_balanced.to_numpy(dtype=np.int16), dtype=th.long)

    graph_num, node_num =  train_label_tensor.shape[0], 200 # node_num hard-coded here 

    for i in range(graph_num): 

        matrix = make_lovely_matrix(fmri_balanced.iloc[i,:])

        # Direction --> sign on the matrix 
        edge_inx = matrix.nonzero()
        edge_inx = [[edge_inx[0][i], edge_inx[1][i]] for i in range(len(edge_inx[0]))]
        #print(edge_inx.shape)
        #print("edge_index: ", edge_inx)
        edge_attr = th.FloatTensor([matrix[idx[0], idx[1]] for idx in edge_inx])
        #print(edge_attr.shape, edge_inx.shape)
        x = th.eye(node_num) 
        #x = th.rand((200, 200) 

        graph_data = Data(x = x, # 200 x 200 identity matrix for all node features 
                          edge_index = th.LongTensor(edge_inx).transpose(1,0), 
                          edge_attr = edge_attr.clone().detach(), 
                          y = train_label_tensor[i])
        
        # de-bugging: 
        # print(f"for itr{i}, graph object: {graph_data}")
        # print(f"for ith{i}, edge inx shape: {graph_data.edge_index.shape}")
        # print(f"for ith{i}, edge inx: {graph_data.edge_index}")
        # print(f"for ith{i}, edge attr shape: {graph_data.edge_attr.shape}")
        # print(f"for ith{i}, edge attr: {graph_data.edge_attr}")

        graph_lst.append(graph_data) 

    return graph_lst 


def data_splitting(graph_lst, splitting_threshold = 0.7): 
    random.shuffle(graph_lst)
    # print(graph_lst)

    if splitting_threshold >= 1 or splitting_threshold <= 0: 
        raise ValueError("What r u doing? 0 < Splitting threshold < 1")
    split_inx = int(len(graph_lst) * splitting_threshold) #8:2 train test splitting 
    train_data = graph_lst[:split_inx]
    test_data = graph_lst[split_inx:]
    return train_data, test_data 

class GNN(th.nn.Module): 

    def __init__(self, dropout_rate): 
        super(GNN, self).__init__() 
        self.conv1 = GCNConv(in_channels = 200, out_channels = 128)
        self.conv2 = GCNConv(in_channels = 128, out_channels = 64)
        self.conv3 = GCNConv(in_channels = 64, out_channels = 4) 
        self.dropout = th.nn.Dropout(p=dropout_rate) 

    def forward(self, data): 

        x, edge_inx, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch 
        
        # De-bugging steps: 
        # if th.isnan(x).any():
        #     print("Nan in X")
        #     exit(1) 
        # if th.isnan(edge_attr).any():
        #     print("Nan in edge_attr")
        #     exit(1) 
        # if edge_inx.max() > x.shape[0]: 
        #     print("Edge inx is wrong")
        #     exit(1)
 
        x = self.conv1(x, edge_inx, edge_attr) 
        x = th.relu(x) 
        x = self.dropout(x)

        print(f"x after 1st layer: {x}") # de-bugging 

        x = self.conv2(x, edge_inx, edge_attr) 
        x = th.relu(x) 
        x = self.dropout(x) 

        print(f"x after 2nd layer: {x}") # de-bugging 

        x = self.conv3(x, edge_inx, edge_attr)

        # de-bugging: 
        # print(f"batch: {batch}")
        # print(f"x: {x}")
        
        # Try different pool methods: 
        x = global_mean_pool(x, batch)
        # x = global_max_pool(x, batch)
        # x = global_sort_pool(x, batch)
        # x = global_max_pool(x, batch)

        return x
    
def train(train_loader, model, criterion, LinearProjector, optimizer): 

    model.train() 
    total_loss, total_samples, correct = 0, 0, 0

    for data in tqdm(train_loader): 
        optimizer.zero_grad() 
        # out=model(data) # --> GNN 
        
        #print(f"Node feature shape: {data.x.shape}")
        #print(f"Batch feature shape: {data.batch.shape}")

        # GCN: 
        out = model(data.x, data.edge_index, data.edge_attr, batch_size = 8)
        out = global_mean_pool(out, data.batch)
        out = LinearProjector(out)
        # out = LinearProjector2(out)

        # print(out) # debugging 
        # print("Model Output:", out)
        # print("Labels:", data.y)
        # print("data.y Shape:", data.y.shape)  
        # print("Unique Label Values:", data.y.unique())
        # if th.isnan(out).any(): # De-bugging --> If nan value from model output 
        #     print("Nan in model output")
        #     exit(1)

        loss = criterion(out, data.y) 
        # print(f"loss: {loss.item()}") # debugging 

        loss.backward()
        # if th.isnan(loss).any():  # De-bugging --> If nan from loss? 
        #     print("Nan in loss") 
        #     exit(1) 

        # th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() 
        total_loss += loss.item() 

        predictions = out.argmax(dim=1) 

        # de-bugging: 
        # print(f"Training, model output {out}")
        # print(f"Training, Labels: {data.y}")
        # print(f"Training, prediction: {predictions}")

        correct += (predictions == data.y).sum().item()
        total_samples += data.y.size(0)

    train_accuracy = (correct / total_samples) * 100  

    return total_loss / len(train_loader), train_accuracy 

def validate(val_loader, model, criterion, LinearProjector): 

    model.eval()
    total_loss, correct, total_samples = 0, 0, 0 

    with th.no_grad():  
        for data in val_loader:

            # out = model(data) # --> GNN

            # GCN: 
            out = model(data.x, data.edge_index, data.edge_attr, batch_size = 8) 
            out = global_mean_pool(out, data.batch)
            out = LinearProjector(out)
            # out = LinearProjector2(out)

            # de-bugging: 
            # if th.isnan(out).any():
            #     print("NaN in model output")
            #     exit(1)

            loss = criterion(out, data.y)
            total_loss += loss.item()
            predictions = out.argmax(dim=1)

            # de-bugging: 
            # print(f"Validating, model output {out}")
            # print(f"Validating, labels: {data.y}")
            # print(f"Validating, prediction: {predictions}")
            # print(f"Predictions Shape: {predictions.shape}, Labels Shape: {data.y.shape}")

            correct += (predictions == data.y).sum().item()
            
            total_samples += data.y.size(0)

    test_loss = total_loss / len(val_loader)
    test_accuracy = (correct / total_samples) * 100

    return test_loss, test_accuracy


graph_lst = create_graph_lst(train_outcome, train_fmri) 
# train_data, test_data = data_splitting(graph_lst) # splitting threshold = 0.8 
 
# GNN: 
# model = GNN(dropout_rate = 0.1)

# Two linear projector: 
# LinearProjector1 = Linear(200, 128)
# LinearProjector2 = Linear(128, 4)

# train_loader = DataLoader(train_data, batch_size = 8, shuffle = True)
# test_loader = DataLoader(test_data, batch_size=8, shuffle=False) 

def cross_validation(graph_lst, k, initial_lr, num_layers, dropout, batch_size, epochs_num, master_seed, patience): 

    kfold = KFold(n_splits = k, shuffle=True, random_state= master_seed)
    log_messages = [] 

    model_choice = "GCN" # change this later. 
    num_linear_predictors = 1 
    para_message = (f"""
        Model: {model_choice},
        Number of Layers: {num_layers},
        Number of Linear Predictors: {num_linear_predictors},
        Initial Learning Rate: {initial_lr},
        Dropout rate: {dropout},
        Cross validation fold {k} each with {epochs_num} epochs
        Batch size {batch_size}, 
        Master seed: {master_seed},
        Stopping patience: {patience}
        """
        )
    log_messages.append(para_message)

    for fold, (train_inx, val_inx) in enumerate(kfold.split(graph_lst)): 
        
        lowest_val_loss = float("inf")
        lowest_valloss_epoch = None 
        patience_count = 0 

        # Just use one fold first 
        # if fold > 0: 
        #     print("Let's just break the loop at fold 1")
        #     break 

        run = wandb.init(
        entity="bli283-university-of-wisconsin-madison",
        project="fmri-adhd",
        config={
            "initial_learning_rate": initial_lr,
            "architecture": "GCN",
            "dataset": "fmri-only",
            "num_layers": num_layers,  
            "epochs": epochs_num,
            "k-fold": k, 
            "dropout": dropout, 
            "batch_size": batch_size, 
            "master_seed": master_seed, 
            "patience_to_stop": patience, 
        },) 


        train_data = [graph_lst[i] for i in train_inx]
        val_data = [graph_lst[i] for i in val_inx]
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        best_model_buffer = io.BytesIO() 

        model = GCN(in_channels = 200, 
            hidden_channels = 200, 
            num_layers = num_layers, 
            out_channels = 200, 
            norm='batch_norm', 
            act_first=False, # activation after normalization
            dropout = dropout
            ) 

        optimizer = th.optim.Adam(model.parameters(), lr = initial_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        criterion = th.nn.CrossEntropyLoss() # multi-class
        LinearProjector = Linear(200, 4) 

        message0 = f"Start training and validating for fold = {fold}"
        log_messages.append(message0)

        for epoch in range(epochs_num):
            train_loss, train_accuracy = train(train_loader, model, criterion, LinearProjector, optimizer)
            val_loss, val_accuracy = validate(val_loader, model, criterion, LinearProjector) 
            
            scheduler.step(val_loss) 
            updated_lr = optimizer.param_groups[0]['lr'] 

            message1 = f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Updated Learning Rate: {updated_lr}"
            message2 = f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%" 
            print(message1)
            print(message2)
            log_messages.append(message1)
            log_messages.append(message2)

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
            
            if patience_count > patience: 
                log_messages.append(f"Validation loss reached a plateau at epoch {lowest_valloss_epoch}, break")
                break

        run.finish()
    
    best_model_buffer.seek(0)
    with open(f"GCN_layer{num_layers}.pth", "wb") as f:
        f.write(best_model_buffer.read())  

    return log_messages 

def main(): 
    
    k = 5
    initial_lr = 0.001
    num_layers = 3
    dropout = 0.1
    batch_size = 8
    epochs_num = 250
    master_seed = 54321
    patience = 20

    logfile = f"GCN_fold{k}_iniLr{initial_lr}_layers{num_layers}_dropout{dropout}_batch{batch_size}_epochs{epochs_num}_seed{master_seed}_patience{patience}_batchnormalization.log"

    log_messages = cross_validation(graph_lst, k, initial_lr, num_layers, dropout, batch_size, epochs_num, master_seed, patience)

    with open(logfile, "w") as f:
        f.write("\n".join(log_messages) + "\n")  

if __name__ == "__main__":
    main() 





    
