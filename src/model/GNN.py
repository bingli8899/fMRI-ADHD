# Script for graphic neural network (GNN) 
# Installation of dgl --> Follow https://www.dgl.ai/dgl_docs/install/index.html


import sys
import os 
import pandas as pd
import torch as th
import dgl 

from dgl.data import DGLDataset 
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info 

rootfolder = "/u/b/i/bingl/private/fMRI-AHDH" 
sys.path.append(os.path.join(rootfolder))
datafolder = os.path.join(rootfolder, "data")
from src.data.data_loader import load_or_cache_data 
from src.data.make_toy_dataset import make_toy_dataset 

# Load full data:  
pickle_file = os.path.join(datafolder, "data.pkl") 
train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)
# # train_quant = train_data_dic["train_quant"]
# train_outcome = train_data_dic["train_outcome"]
# # train_cate = train_data_dic["train_cate"]
# train_fmri = train_data_dic["train_fmri"] 
# # test_quant = test_data_dic["test_quant"]
# # test_cate = test_data_dic["test_cate"]
# test_fmri = test_data_dic["test_fmri"]

# Use toy dataset to test the data 
train_toy = make_toy_dataset(train_data_dic, output_path = datafolder)
test_toy = make_toy_dataset(test_data_dic, data_type = "test", output_path = datafolder)

train_toy_fmri = train_toy["train_fmri"] 
print(train_toy_fmri.head)
print(train_toy["train_outcome"].head)
print(test_toy["test_fmri"].head)

def process(data_dic, mode="train"): 

    if mode == "train": 
        fmri = data_dic["train_fmri"]
        label = data_dic["train_outcome"] 
    elif mode == "test": 
        fmri = data_dic["test_fmri"]
        label = None
    else: 
        raise ValueError("Wrong mode")

    for ithrow, row in fmri.iterrows(): 
        participant_id = row["participant_id"]
        adjacent_dic = {} 

        for col_name, val in row.items(): 

            print(f"for {ithrow}: {col_name} and {val}")

            if col_name == "participant_id": 
                continue 
            
            node1, node2_temp = col_name.split("throw_")
            node2 = node2_temp.split["thcolum"][0]
            print(node1, node2) 
            
            if val > 0:  # If val > 0, direction node1 (src_node) to node2 (dst_node)
                adjacent_dic[(node1, node2)] = val 
            else: # If val < 0, direction node2 (src_node) to node1 (dst_node)
                adjacent_dic[(node1, node2)] = abs(val) # all val > 1 since sign is presented by direction 

    return adjacent_dic

def make_graphs(adjacent_dic, num_nodes):
    src_nodes = [] # source nodes  
    dst_nodes = [] # destination nodes 
    edge_features = [] # edge feature 

    for (src_node, dst_node), edge_feature in adjacent_dic.items(): 
        src_nodes.append(src_node)
        dst_nodes.append(dst_node)
        edge_features.append(edge_feature) 

    src_tensor = th.tensor(src_nodes, dtype=th.int64)
    dst_tensor = th.tensor(dst_nodes, dtype=th.int64)
    edge_feature_tensor = th.tensor(edge_features, dtype=th.float32)

    graph = dgl.graph((src_tensor, dst_tensor))
    graph.edata["feature"] = edge_feature_tensor

    # Having a one-hot vector for each node instead of a identity matrix 
    # Preventing GNN from learning identical feature representation 
    identity_mtx = torch.eye(num_nodes, dtype=th.float32) # Create one-hot encoding
    ind = th.arange(num_nodes) # dgl.ndata doesn't allow individual assignment so create group ind 
    graph.ndata["feature"] = identity_mtx[ind]

    return graph
 
def __len__(graph): 
    return graph.num_nodes(), graph.num_edges() 


# def make_graph(adjacent_dic): 

def main(): 
    train_toy_1person = make_toy_dataset(train_data_dic, n_subjects = 1, n_regions = 30, output_path = datafolder)
    adj_dict = process(train_toy_1person, mode="train")
    num_nodes = train_toy_fmri.shape[1] - 1  # Exclude participant_id column
    graph = make_graphs(adj_dict, num_nodes)

    print("Graph:", graph)
    print("Number of Nodes:", graph.num_nodes())
    print("Number of Edges:", graph.num_edges())
    print("Node Features:\n", graph.ndata["feature"])
    print("Edge Features:\n", graph.edata["feature"])

main()

# class GraphDataset(DGLDataset): 

#     def __init__(self, data_dic, mode = "train"): 

#         self.data_dic = data_dic 
#         self.mode = mode.lower() 
#         self.graph = [] 
#         self.labels = [] 

#         super().__init__(name=f"toy_fmri_graph_{mode}") 

#     def process(self): 

#         if self.mode == "train": 
#             fmri = self.data_dic["train_fmri"]
#             label = self.data_dic["train_outcome"] 
#         elif self.mode == "test": 
#             fmri = self.data_dic["test_fmri"]
#             label = None
#         else: 
#             raise ValueError("Wrong mode")

#         for ithrow, row in fmri.iterrows(): 
#             participant_id = row["participant_id"]
#             adjacent_dic = {} 

#             for col_name, val in row.items(): 
#                 if col_name == participant_id: 
#                     continue 


                





    






# class GraphDataset(DGLDataset): 

#     def __init__(self, rootfolder, mode = "train"):
#         self.rootfolder = rootfolder
#         self.datafolder = os.path.join(rootfolder, "data") 
#         self.pickle_file = os.path.join(datafolder, "data.pkl") 
#         self.mode = mode.lower() 
        
#         self.save_path = os.path.join(self.datafolder, "cached_graphs")
#         makedirs(self.save_path) 

#         super().__init__(name=f"fmri_graph_{mode}")

#     def save(self): 
#         graph_path = os.path.join(self.save_path, f"{self.mode}_dgl_graph.bin")
#         save_graphs(graph_path, self.graphs, {"labels": self.labels} if self.labels is not None else {})

#     def load(self): 
#         """
#         Load data either from a cached graph or from the original dataset.
#         If a cached graph exists, loads it.
#         Otherwise, loads from the raw data file and processes it.
#         """
#         graph_path = os.path.join(self.save_path, f"{self.mode}_dgl_graph.bin")
#         info_path = os.path.join(self.save_path, f"{self.mode}_info.pkl")

#         # If cached data exists, load it: 
#         if os.path.exists(graph_path) and os.path.exists(info_path):
#             print(f"Loading cached {self.mode} graph from {self.save_path}...")
#             self.graphs, label_dict = load_graphs(graph_path)
#             self.labels = label_dict.get("labels", None)
#             # self.num_classes = load_info(info_path).get("num_classes", 0)
#             return

#         # Otherwise, load raw data from file: 
#         print(f"No cached {self.mode} graph found. Loading from raw data...")
#         train_data_dic, test_data_dic = load_or_cache_data(self.datafolder, self.pickle_file)

#         if self.mode == "train":
#             fmri_data = train_data_dic["train_fmri"]
#             labels = train_data_dic["train_outcome"]
#         elif self.mode == "test":
#             fmri_data = test_data_dic["test_fmri"]
#             labels = None 
#         else:
#             raise ValueError("Invalid mode! Choose 'train' or 'test'.")

#         return fmri_data, labels

#     def process(self): 


    




        










