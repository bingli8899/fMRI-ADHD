import os
import pandas as pd
from pathlib import Path
import openpyxl 

def load_train_data(datapath, filetype = "train"):
    
    file_lst = extract_file_path(datapath)
    data = {} 
    
    if filetype.lower() == "train":
        expected_files = {
            "train_quant": "TRAIN_QUANTITATIVE_METADATA.xlsx",
            "train_outcome": "TRAINING_SOLUTIONS.xlsx",
            "train_cate": "TRAIN_CATEGORICAL_METADATA.xlsx",
            "train_fmri": "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv"
        }           
    elif filetype.lower() == "test":
        expected_files = {
            "test_cate": "TEST_CATEGORICAL.xlsx",
            "test_fmri": "TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv",
            "test_quant": "TEST_QUANTITATIVE_METADATA.xlsx"
        }
    else: 
        raise ValueError(f"Hey! your file type {filetype} should be either 'train' or 'test'.") 
    
    for key, filepath in expected_files.items():      
        if filepath.suffix == ".xlsx": 
            data[key] = pd.read_excel(filepath)
        if filepath.suffic == ".csv": 
            data[key] = pd.read_csv(filepath) 
        else: 
            raise ValueError(f"Hey! {datapath} might contain invalid file types. Check it NOW!")

def extract_file_path(datapath, filetype = "train"): 
    
    datapath = Path(datapath)
    
    file_lst = [] 
        
    for dirname, _, filenames in os.walk(datapath):
        for filename in filenames:
            file_path = os.path.join(dirname, filename) 
            file_lst.append(file_path)
            print(f"loading {os.path.join(dirname, filename)}")  
    return file_lst 
