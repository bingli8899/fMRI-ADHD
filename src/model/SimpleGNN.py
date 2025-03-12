# Let's use a single GNN without DGL to see the performance first 
# conda activate simpleGNN 

# This is a simple GNN: 
# 1) Edge attribute is uni-directional 
# 2) Only positive values in the matrix is considered as edge_attr, negative values (deactivation) is not considered. 
# 3) Use identity matrix as node feature 
# 4) Three layers of covolution. I didn't even found any improvement of using three layers compared to two layers. 
# 5) I didn't even include the metadata yet. 

# Notes: 

# With global_mean_pool and without dropping out: 
# 1) With two convolution layers, accuracy after 100 epochs = 76% to 73% (without data dropout)
# 2) With three convolution layers, accuracy after 100 epochs = 100% --> Is this even real? This doesn't make sense to me 
# This suggests overfitting probably --> Then, add dropout (dropout rate = 0.5)

# With global_mean_pool and dropping_out rate at 0.5: 
# With three covolution layers, accuract after 100 epochs = 85% but completely failed on the test dataset :-( 

# Improve this: 
# 1) Sign of activation should be included as direction 
# 2) Node feature is one-hot vector instead of identity matrix so the network could learn some unique feature for each node 
# 3) Fine-tune patameters? 
# 4) Check if there is an balanced labeling 
#   --> if not balanced, then I can repeat some lower frequency labeling (Oversampling). 


import sys
import os 
import numpy as np
import torch as th 

from torch_geometric.nn import GCNConv, GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch.nn import Dropout, Linear
from tqdm import tqdm
import random 
from imblearn.over_sampling import SMOTE, ADASYN

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
    cont_matrix = th.tensor(fmri_balanced.values).float()

    graph_num, node_num = cont_matrix.shape[0], 200 # node_num hard-coded here 

    for i in range(graph_num): 
        matrix = cont_matrix[i].view(100, 199) # ith row in connectivity matrix 
        #edge_inx = 
        edge_inx = (matrix > 0).nonzero(as_tuple = False).t() # uni-directional for now 
        edge_attr = matrix[edge_inx[0], edge_inx[1]] 
        x = th.eye(node_num) 
        #x = th.rand((200, 200))

        graph_data = Data(x = x, # 200 x 200 identity matrix for all node features 
                          edge_index = edge_inx, 
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
    
def train(): 

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
        out = LinearProjector1(out)
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
        print(f"During training, model output {out}")
        print(f"During, training, Labels: {data.y}")
        print(f"During training, prediction: {predictions}")

        correct += (predictions == data.y).sum().item()
        total_samples += data.y.size(0)

    train_accuracy = (correct / total_samples) * 100  

    return total_loss / len(train_loader), train_accuracy 

def test(): 
    model.eval()
    total_loss, correct, total_samples = 0, 0, 0 

    with th.no_grad():  
        for data in test_loader:

            # out = model(data) # --> GNN

            # GCN: 
            out = model(data.x, data.edge_index, data.edge_attr, batch_size = 8) 
            out = global_mean_pool(out, data.batch)
            out = LinearProjector1(out)
            # out = LinearProjector2(out)

            # de-bugging: 
            # if th.isnan(out).any():
            #     print("NaN in model output")
            #     exit(1)

            loss = criterion(out, data.y)
            total_loss += loss.item()
            predictions = out.argmax(dim=1)

            # de-bugging: 
            print(f"During testing, model output {out}")
            print(f"During testing, labels: {data.y}")
            print(f"During testing, prediction: {predictions}")
            # print(f"Predictions Shape: {predictions.shape}, Labels Shape: {data.y.shape}")

            correct += (predictions == data.y).sum().item()
            
            total_samples += data.y.size(0)

    test_loss = total_loss / len(test_loader)
    test_accuracy = (correct / total_samples) * 100

    #model.train() # re-enable dropout 

    return test_loss, test_accuracy


graph_lst = create_graph_lst(train_outcome, train_fmri) 
train_data, test_data = data_splitting(graph_lst) # splitting threshold = 0.8 

# GCN: 
model = GCN(in_channels = 200, 
            hidden_channels = 200, 
            num_layers = 4, 
            out_channels = 200) 

# GNN: 
# model = GNN(dropout_rate = 0.1)

LinearProjector1 = Linear(200, 4)
# LinearProjector2 = Linear(128, 4)


# Have a learning rate scheduler 
# Use valudation loss 
# val_loss = validate() 
# print the learning rate 
# Torch.ReduceLROnPlateau 

optimizer = th.optim.Adam(model.parameters(), lr = 0.0001)

criterion = th.nn.CrossEntropyLoss() # multi-class 
train_loader = DataLoader(train_data, batch_size = 8, shuffle = True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False) 

log_messages = [] 

for epoch in range(400+1):
    train_loss, train_accuracy = train()
    test_loss, test_accuracy = test() 
    if epoch % 2 == 0: 

        message1 = f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%"
        message2 = f"Epoch: {epoch}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%" 

        print(message1)
        print(message2)
        log_messages.append(message1)
        log_messages.append(message2)

with open("training_log_400epochs_TwoLinear.txt", "w") as f:
    f.write("\n".join(log_messages) + "\n") 





    
