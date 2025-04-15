# Metalayer 

import sys
import os 
import numpy as np 
import pandas as pd
import torch
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader 
from torch.nn import Linear, Module
from torch_geometric.nn import MetaLayer, global_mean_pool 
from torch.nn import Linear 
from torch_geometric.utils import scatter
from tqdm import tqdm
import random 
from imblearn.over_sampling import SMOTE
from torch_geometric.data import Data, Batch, DataListLoader

# Check later: 
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
# from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

rootfolder = "/u/b/i/bingl/private/fMRI-AHDH/" # change this
sys.path.append(os.path.join(rootfolder))

from src.data.data_loader import load_or_cache_data
from src.utility.ut_GNNbranch import relabel_train_outcome  
from src.data.KNN_imputer import KNNImputer_with_OneHotEncoding
from src.utility.ut_general import normalizing_factors

datafolder = os.path.join(rootfolder, "data")
pickle_file = os.path.join(datafolder, "data.pkl") 
train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)
train_fmri = train_data_dic["train_fmri"] 
test_fmri = test_data_dic["test_fmri"]

master_seed = 3471
torch.manual_seed(master_seed)

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


def create_graph_lst(fmri_data, fmri_outcomes=None, scaler = None): # change this later 
    graph_lst = []

    if fmri_outcomes is not None:
        train_fmri_sorted = fmri_data.sort_values(by="participant_id")
        train_outcome_sorted = fmri_outcomes.sort_values(by="participant_id")
        train_label = relabel_train_outcome(train_outcome_sorted)

        if (train_fmri_sorted["participant_id"].values != train_outcome_sorted["participant_id"].values).all():
            raise ValueError("Mismatch in participant ID!")

        # Map participant IDs to numeric values
        participant_mapping = {pid: idx for idx, pid in enumerate(train_fmri_sorted["participant_id"].unique())}
        train_fmri_sorted["participant_id_mapped"] = train_fmri_sorted["participant_id"].map(participant_mapping)

        # Drop original ID column before SMOTE
        train_fmri_sorted = train_fmri_sorted.drop(columns=["participant_id"])

        # Apply scaling
        # if scaler is None:
        #     raise ValueError("You need to have scaling")
        # else:
        #     scaler.fit(train_fmri_sorted)
        #     with open(f"scaler.pkl", "wb") as f:
        #         pickle.dump(scaler, f)

        # Balance dataset using SMOTE
        smote = SMOTE(sampling_strategy="auto", random_state=1234)
        fmri_balanced, label_balanced = smote.fit_resample(train_fmri_sorted, train_label["Label"])
        fmri_balanced = pd.DataFrame(fmri_balanced, columns=train_fmri_sorted.columns)

        # Reverse mapping to original participant IDs
        reverse_mapping = {v: k for k, v in participant_mapping.items()}
        fmri_balanced["participant_id"] = fmri_balanced["participant_id_mapped"].map(reverse_mapping)

        train_label_tensor = torch.tensor(label_balanced.to_numpy(dtype=np.int16), dtype=torch.long)
        participant_ids = fmri_balanced["participant_id"].values
        fmri_balanced = fmri_balanced.drop(columns=["participant_id", "participant_id_mapped"])

    else:
        participant_ids = fmri_data["participant_id"].values
        fmri_dropped = fmri_data.drop(columns="participant_id")

        # if scaler is None:
        #     raise ValueError("You need to have scaling")
        # else:
        #     with open("scaler.pkl", "rb") as f:
        #         scaler = pickle.load(f)
        #     fmri_balanced = scaler.transform(fmri_dropped)
        fmri_balanced = fmri_dropped 

    graph_num, node_num = len(fmri_balanced), 200


    for i in tqdm(range(5)):

        matrix = make_lovely_matrix(fmri_balanced.iloc[i,:])

        # Direction --> sign on the matrix 
        edge_inx = matrix.nonzero()
        edge_inx = [[edge_inx[0][i], edge_inx[1][i]] for i in range(len(edge_inx[0]))]
        edge_attr = torch.FloatTensor([matrix[idx[0], idx[1]] for idx in edge_inx])
        x = torch.eye(node_num) 

        graph_data = Data(x = x, # 200 x 200 identity matrix for all node features 
                          edge_index = torch.LongTensor(edge_inx).transpose(1,0), 
                          edge_attr = edge_attr.clone().detach(), 
                          y = train_label_tensor[i] if fmri_outcomes is not None else None, 
                          participant_id = participant_ids[i])
        
        print(f"graphs for {i}")
        print(graph_data)

        graph_lst.append(graph_data) 

    return graph_lst

def add_metadata_to_graph_lst(graph_lst, datafolder):
    
    # rootfolder = config.root_folder 
    # sys.path.append(os.path.join(rootfolder))
    # datafolder = rootfolder
    
    pickle_file = os.path.join(datafolder, "data.pkl") 
    train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)
    
    train_data_dic.update(test_data_dic)
    
    imputer = KNNImputer_with_OneHotEncoding()
    # metadata = imputer.fit_transform(train_data_dic, split="train" if args.train_config else "test")
    metadata = imputer.fit_transform(train_data_dic, split="train") 

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
        datapoint.metadata = torch.FloatTensor(raw_values[value_index, :])
        # datapoint.metadata = torch.FloatTensor(raw_values[value_index, :]).view(1, -1) # change this to u 


class EdgeModel(Module):
    def __init__(self):
        super().__init__()
        self.edge_mlp = Linear(200 + 200 + 1 + 81, 4)

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        print(src.shape, dst.shape, edge_attr.shape, u[batch].shape)
        out = torch.cat([src, dst, edge_attr[:,None], u[batch][:,None]], 1)
        return self.edge_mlp(out)

class NodeModel(Module):
    def __init__(self):
        super().__init__()
        self.node_mlp_1 = Linear(200, 128)
        self.node_mlp_2 = Linear(128, 4)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')
        out = th.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(Module):
    def __init__(self):
        super().__init__()
        self.global_mlp = Linear(81, 4) 

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([
            u,
            scatter(x, batch, dim=0, reduce='mean'),
        ], dim=1)
        return self.global_mlp(out)

fmri_data = train_fmri
fmri_outcome = train_data_dic["train_outcome"] 
graph_lst = create_graph_lst(fmri_data, fmri_outcome, scaler = None)
add_metadata_to_graph_lst(graph_lst, datafolder)

g = graph_lst[0]
x = g.x 
print(f"node feature shape {x.shape}")
edge_index = g.edge_index
print(f"node feature shape {edge_index.shape}")
edge_attr = g.edge_attr 
print(f"edge attr shape {edge_attr.shape}")
u = g.metadata.view(1,-1)
print(f"U shape {u.shape}")

op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
batch = torch.zeros(x.shape[0], dtype=torch.long) 
x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)


batch_size = 2
train_loader = DataListLoader(graph_lst, batch_size=batch_size, shuffle=True)

# def train(train_loader, model, criterion, optimizer): 

#     model.train() 
#     total_loss, total_samples, correct = 0, 0, 0

#     for data in tqdm(train_loader): 
#         optimizer.zero_grad() 
#         out = model(data)
#         loss = criterion(out, data.y) 
#         loss.backward()
#         optimizer.step() 
#         total_loss += loss.item() 
#         predictions = out.argmax(dim=1) 
#         correct += (predictions == data.y).sum().item()
#         total_samples += data.y.size(0)

#     train_accuracy = (correct / total_samples) * 100  

#     return total_loss / len(train_loader), train_accuracy 



  
# # Might need to change later: 
# inputs = 200
# hidden = 128
# outputs = 4

# class EdgeModel(Module):
#     def __init__(self):
#         super().__init__()
#         # self.edge_mlp = Linear(200 + 200 + 1 + 81, 4)

#         self.edge_mlp = Seq(Lin(inputs, hidden), 
#                             BatchNorm1d(hidden),
#                             ReLU(),
#                             Lin(hidden, hidden))

#     def forward(self, src, dst, edge_attr, u, batch):
#         # src, dst: [E, F_x], where E is the number of edges.
#         # edge_attr: [E, F_e]
#         # u: [B, F_u], where B is the number of graphs.
#         # batch: [E] with max entry B - 1.
#         # out = th.cat([src, dst, edge_attr, u[batch]], 1)

#         out = torch.cat([src, dst], 1)
#         return self.edge_mlp(out)

# class NodeModel(Module):
#     def __init__(self):
#         super().__init__()

#         self.node_mlp_1 = Linear(200, 128)
#         self.node_mlp_2 = Linear(128, 4)

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         # x: [N, F_x], where N is the number of nodes.
#         # edge_index: [2, E] with max entry N - 1.
#         # edge_attr: [E, F_e]
#         # u: [B, F_u]
#         # batch: [N] with max entry B - 1.
#         row, col = edge_index
#         out = torch.cat([x[row], edge_attr], dim=1)
#         out = self.node_mlp_1(out)
#         out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
#         out = torch.cat([x, out], dim=1)
#         return self.node_mlp_2(out)

# class GlobalModel(Module):
#     def __init__(self):
#         super().__init__()
#         self.global_mlp = Linear(81, 4) 

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         # x: [N, F_x], where N is the number of nodes.
#         # edge_index: [2, E] with max entry N - 1.
#         # edge_attr: [E, F_e]
#         # u: [B, F_u]
#         # batch: [N] with max entry B - 1.
#         out = th.cat([
#             u,
#             scatter(x, batch, dim=0, reduce='mean'),
#         ], dim=1)
#         return self.global_mlp(out)

  
# # Might need to change later: 
# inputs = 200
# hidden = 128
# outputs = 4

# class EdgeModel(Module):
#     def __init__(self):
#         super().__init__()
#         # self.edge_mlp = Linear(200 + 200 + 1 + 81, 4)

#         self.edge_mlp = Seq(Lin(inputs, hidden), 
#                             BatchNorm1d(hidden),
#                             ReLU(),
#                             Lin(hidden, hidden))

#     def forward(self, src, dst, edge_attr, u, batch):
#         # src, dst: [E, F_x], where E is the number of edges.
#         # edge_attr: [E, F_e]
#         # u: [B, F_u], where B is the number of graphs.
#         # batch: [E] with max entry B - 1.
#         # out = th.cat([src, dst, edge_attr, u[batch]], 1)

#         out = torch.cat([src, dst], 1)
#         return self.edge_mlp(out)

# class NodeModel(Module):
#     def __init__(self):
#         super().__init__()

#         self.node_mlp_1 = Linear(200, 128)
#         self.node_mlp_2 = Linear(128, 4)

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         # x: [N, F_x], where N is the number of nodes.
#         # edge_index: [2, E] with max entry N - 1.
#         # edge_attr: [E, F_e]
#         # u: [B, F_u]
#         # batch: [N] with max entry B - 1.
#         row, col = edge_index
#         out = torch.cat([x[row], edge_attr], dim=1)
#         out = self.node_mlp_1(out)
#         out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
#         out = torch.cat([x, out], dim=1)
#         return self.node_mlp_2(out)

# class GlobalModel(Module):
#     def __init__(self):
#         super().__init__()
#         self.global_mlp = Linear(81, 4) 

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         # x: [N, F_x], where N is the number of nodes.
#         # edge_index: [2, E] with max entry N - 1.
#         # edge_attr: [E, F_e]
#         # u: [B, F_u]
#         # batch: [N] with max entry B - 1.
#         out = th.cat([
#             u,
#             scatter(x, batch, dim=0, reduce='mean'),
#         ], dim=1)
#         return self.global_mlp(out)





