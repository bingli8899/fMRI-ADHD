import sys 
import os
import pandas as pd  
from pathlib import Path  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from pathlib import Path
import glob

rootfolder = os.path.abspath(os.path.join(Path.cwd())) # rootpath --> top of git repo
# Change the above if you are not in $root/notebook/
sys.path.append(os.path.join(rootfolder))

from src.data.data_loader import load_data

datafolder = os.path.join(rootfolder, "data")

train_data_dic = load_data(datafolder, filetype = "train") 
train_quant = train_data_dic["train_quant"]
train_outcome = train_data_dic["train_outcome"]
train_cate = train_data_dic["train_cate"]
train_fmri = train_data_dic["train_fmri"]

test_data_dic = load_data(datafolder, filetype = "test")
test_quant = test_data_dic["test_quant"]
test_cate = test_data_dic["test_cate"]
test_fmri = test_data_dic["test_fmri"]

class dGCN():
    pass