# Script for graphic neural network (GNN) 

import sys
import os 
import pandas as pd


rootfolder = "/u/b/i/bingl/private/fMRI-AHDH" 
sys.path.append(os.path.join(rootfolder))
datafolder = os.path.join(rootfolder, "data")
from src.data.data_loader import load_or_cache_data 

# Load data:  
pickle_file = os.path.join(datafolder, "data.pkl") 
train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)

train_quant = train_data_dic["train_quant"]
train_outcome = train_data_dic["train_outcome"]
train_cate = train_data_dic["train_cate"]
train_fmri = train_data_dic["train_fmri"] 
test_quant = test_data_dic["test_quant"]
test_cate = test_data_dic["test_cate"]
test_fmri = test_data_dic["test_fmri"]



