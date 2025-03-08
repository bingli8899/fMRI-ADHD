import os
import pandas as pd
from pathlib import Path
import openpyxl 
import pickle

def load_or_cache_data(datafolder, pickle_file): 
    """
    Loads dataset from cache if available, else, loads it from the data loader,
    Args:
        datafolder (str): Path to the data directory.
        pickle_file (str): Path to the cache file.
    Returns: (train_data_dic, test_data_dic)
    """
    if os.path.exists(pickle_file):  
        with open(pickle_file, "rb") as f:
            train_data_dic, test_data_dic = pickle.load(f)
        print("Loading data from cached files -> Done!")
    else:
        train_data_dic = load_data(datafolder, filetype = "train") 
        test_data_dic = load_data(datafolder, filetype = "test")
        with open(pickle_file, "wb") as f:
            pickle.dump((train_data_dic, test_data_dic), f)
        print("Loading data and save to cached files -> Done!")
    return (train_data_dic, test_data_dic)

def load_data(datapath, filetype = "train"):
    """
    Load data: 
    Args: 
        datapath: The path to data file (both train and test)
        filetype = {train or test}
    returns: 
        Load pd.dataframe (test or train)
    """
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

