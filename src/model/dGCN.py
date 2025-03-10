import sys 
import os
import pandas as pd  
from pathlib import Path  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from pathlib import Path
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

rootfolder = os.path.abspath(os.path.join(Path.cwd())) # rootpath --> top of git repo
# Change the above if you are not in $root/notebook/
sys.path.append(os.path.join(rootfolder))

from src.data.data_loader import load_data


class dGCN():
    
    def __init__(self, rootfolder, mode = "train"):
        self.rootfolder = rootfolder
        self.datafolder = os.path.join(rootfolder, "data")
        self.mode = mode.lower();
        sys.path.append(os.path.join(rootfolder))

    # Used to load data
    def load(self):
        train_data_dic = load_data(datafolder, filetype = "train")
        train_quant = train_data_dic["train_quant"]
        train_outcome = train_data_dic["train_outcome"]
        train_cate = train_data_dic["train_cate"]
        train_fmri = train_data_dic["train_fmri"]

        test_data_dic = load_data(datafolder, filetype = "test")
        test_quant = test_data_dic["test_quant"]
        test_cate = test_data_dic["test_cate"]
        test_fmri = test_data_dic["test_fmri"]