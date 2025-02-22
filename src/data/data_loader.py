import os
import pandas as pd
from pathlib import Path
import openpyxl 

def load_data(datapath, filetype = "train"):
    
    data = {} 
    
    if filetype.lower() == "train":
        data_folder = os.path.join(datapath, "TRAIN") 
        expected_files = {
            "train_quant": "TRAIN_QUANTITATIVE_METADATA.xlsx",
            "train_outcome": "TRAINING_SOLUTIONS.xlsx",
            "train_cate": "TRAIN_CATEGORICAL_METADATA.xlsx",
            "train_fmri": "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"
        }           
    elif filetype.lower() == "test":
        data_folder = os.path.join(datapath, "TEST") 
        expected_files = {
            "test_cate": "TEST_CATEGORICAL.xlsx",
            "test_fmri": "TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv",
            "test_quant": "TEST_QUANTITATIVE_METADATA.xlsx"
        }
    else: 
        raise ValueError(f"Hey! your file type {filetype} should be either 'train' or 'test'.") 
    
    for key, filename in expected_files.items(): 
        filepath = os.path.join(data_folder, expected_files[key])
        
        if filename.endswith("xlsx"): 
            data[key] = pd.read_excel(filepath)
        
        if filename.endswith("csv"): 
            data[key] = pd.read_csv(filepath) 
            
    return data