# The basic script to check, clean and impute the data 

import pandas as pd 
import os 
from pathlib import Path  
import utilities as ut
import openpyxl # use read_excel from pd
import missingno as msno
import matplotlib.pyplot as plt

rootfolder = Path.cwd() 
traindata = os.path.join(rootfolder, "data", "TRAIN")
train_cate_path = os.path.join(traindata, "TRAIN_CATEGORICAL_METADATA.xlsx")
train_quant_path = os.path.join(traindata, "TRAIN_QUANTITATIVE_METADATA.xlsx") 
train_outcome_path = os.path.join(traindata, "TRAINING_SOLUTIONS.xlsx") 
train_fmri_path = os.path.join(traindata, "TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv") 

train_cate = pd.read_excel(train_cate_path)
train_quant = pd.read_excel(train_quant_path)
train_outcome = pd.read_excel(train_outcome_path) 
train_fmri = pd.read_csv(train_fmri_path)

# Checking missing values 
for col in train_cate.columns: 
    ut.missing_percentage(train_cate, col)
# Interesting to see only one column "PreInt_Demos_Fam_Child_Ethnicity" has a relatively high (1%) missing percentage

for col in train_quant.columns:
    ut.missing_percentage(train_quant, col)
# Interesting to see only one column "MRI_Track_Age_at_Scan" has a very high (29.7%) missing percentage 

for col in train_fmri.columns: # Nothing is missing 
    ut.missing_percentage(train_fmri, col) 

# For MRI_Track_Age_at_Scan --> Whether data is missing at random? 
plt.figure(figsize=(10,6))
msno.matrix(train_quant)  # Change dataset as needed
plt.show()








