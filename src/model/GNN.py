# Script for graphic neural network (GNN) 

import sys
import os 
import pandas as pd
import torch 
from dgl.data import DGLDataset 
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info #, load_info

rootfolder = "/u/b/i/bingl/private/fMRI-AHDH" 
sys.path.append(os.path.join(rootfolder))
datafolder = os.path.join(rootfolder, "data")
from src.data.data_loader import load_or_cache_data 

# Load full data:  
pickle_file = os.path.join(datafolder, "data.pkl") 
train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)
# train_quant = train_data_dic["train_quant"]
train_outcome = train_data_dic["train_outcome"]
# train_cate = train_data_dic["train_cate"]
train_fmri = train_data_dic["train_fmri"] 
# test_quant = test_data_dic["test_quant"]
# test_cate = test_data_dic["test_cate"]
test_fmri = test_data_dic["test_fmri"]


class GraphDataset(DGLDataset): 

    def __init__(self, rootfolder, mode = "train"):
        self.rootfolder = rootfolder
        self.datafolder = os.path.join(rootfolder, "data") 
        self.pickle_file = pickle_file = os.path.join(datafolder, "data.pkl") 
        self.mode = mode.lower() 
        
        self.save_path = os.path.join(self.datafolder, "cached_graphs")
        makedirs(self.save_path) 

        super().__init__(name=f"fmri_graph_{mode}")

    def save(self): 
        graph_path = os.path.join(self.save_path, f"{self.mode}_dgl_graph.bin")
        save_graphs(graph_path, self.graphs, {"labels": self.labels} if self.labels is not None else {})
        # info_path = os.path.join(self.save_path, f"{self.mode}_info.pkl")
        # save_info(info_path, {"num_classes": self.num_classes})

    def load(self): 
        """
        Load data either from a cached graph or from the original dataset.
        If a cached graph exists, loads it.
        Otherwise, loads from the raw data file and processes it.
        """
        graph_path = os.path.join(self.save_path, f"{self.mode}_dgl_graph.bin")
        info_path = os.path.join(self.save_path, f"{self.mode}_info.pkl")

        # If cached data exists, load it: 
        if os.path.exists(graph_path) and os.path.exists(info_path):
            print(f"Loading cached {self.mode} graph from {self.save_path}...")
            self.graphs, label_dict = load_graphs(graph_path)
            self.labels = label_dict.get("labels", None)
            # self.num_classes = load_info(info_path).get("num_classes", 0)
            return

        # Otherwise, load raw data from file: 
        print(f"No cached {self.mode} graph found. Loading from raw data...")
        train_data_dic, test_data_dic = load_or_cache_data(self.datafolder, self.pickle_file)

        if self.mode == "train":
            fmri_data = train_data_dic["train_fmri"]
            labels = train_data_dic["train_outcome"]
        elif self.mode == "test":
            fmri_data = test_data_dic["test_fmri"]
            labels = None 
        else:
            raise ValueError("Invalid mode! Choose 'train' or 'test'.")

        return fmri_data, labels

    def process(self): 


    




        










