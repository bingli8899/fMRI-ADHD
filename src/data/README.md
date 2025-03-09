# Documentation for src/data 

This is the module for codes to process and analyze data: 

# Initializing data loader: 

```
import sys
import os 

rootfolder = "PATH/TO/OUR/GITHUB/fMRI-AHDH" # change this
sys.path.append(os.path.join(rootfolder))

from src.data.data_loader import load_data, load_or_cache_data
datafolder = os.path.join(rootfolder, "data")
```
If loading from cached files: 
```
pickle_file = os.path.join(datafolder, "data.pkl") 
train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)
train_quant = train_data_dic["train_quant"]
train_outcome = train_data_dic["train_outcome"]
train_cate = train_data_dic["train_cate"]
train_fmri = train_data_dic["train_fmri"] 
test_quant = test_data_dic["test_quant"]
test_cate = test_data_dic["test_cate"]
test_fmri = test_data_dic["test_fmri"]

```
If loading files repeated with specified train or test: 
```
# Load trainning data: 
train_data_dic = load_data(datafolder, filetype = "train") 
train_quant = train_data_dic["train_quant"]
train_outcome = train_data_dic["train_outcome"]
train_cate = train_data_dic["train_cate"]
train_fmri = train_data_dic["train_fmri"] 

# Load test data: 
test_data_dic = load_data(datafolder, filetype = "test")
test_quant = test_data_dic["test_quant"]
test_cate = test_data_dic["test_cate"]
test_fmri = test_data_dic["test_fmri"]
```

# Make a toy dataset to test codes: 
To run and test codes in a much smaller datasets: 

```
# Assuing data loaded from load_data or load_or_cache_data above 
from src.data.make_toy_dataset import make_toy_dataset 

# default setting: 
train_toy = make_toy_dataset(data_dic, 
    dataset_type: str = "train", 
    n_subjects = 30, 
    n_regions = 30, output_path = current_directory)   

# Example usage: 

# To get first 30 participants and first 30 brain regions (ROI) -> default setting 
train_toy =  make_toy_dataset(train_data_dic) 
# This return a dic = {
    "train_fmri": pd.DataFrame,  # fMRI data subset
    "train_outcome": pd.DataFrame  # ADHD + Sex labels 
} 
# This also saves the toy.csv to current_dictory 

test_toy = make_toy_dataset(test_data_dic, data_type = "test", n_subject = 100, output_path = datafolder) 
# This return a dic = {
    "test_fmri": pd.DataFrame,  # fMRI data subset
    "train_outcome": None  # no label 
} 
# This saves toy.csv to datafolder

```

# KNN imputer: 

KNN imputer with categorical data encoded as onehot vector. 

if merge_fmri = False (defult) -> Impute dataset based on the demographical. This one might be better. Output is the imputed demographic data. 

If merge_fmri = true -> merge fmri dataset and conduct KNN imputing. This is not recommended only if there is a strong correlation between demographical missing values and fmri. The output is the imputed data with fmri

Usage: 
```
from src.data.KNN_imputer import KNNImputer_with_OneHotEncoding

df_dic = {
    "train_cate": train_data_dic["train_cate"],
    "train_quant": train_data_dic["train_quant"],
    "test_cate": test_data_dic["test_cate"],
    "test_quant": test_data_dic["test_quant"],
    "train_fmri": train_data_dic["train_fmri"],
    "test_fmri": test_data_dic["test_fmri"]
}

# Initialize processor and apply transformations -> Don't merge fmri data 
processors = KNNImputer_with_OneHotEncoding(k=5)
train_imputed = processors.fit_transform(df_dic)  
test_imputed = processors.transform(test_data_dic)  

# If there you want to merge fmri data 
processors_YESfmri = KNNImputer_with_OneHotEncoding(merge_fmri=True, k=5)
train_imputed_Yesfmri = processors_YESfmri.fit_transform(df_dic)  
test_imputed_Yesfmri = processors_YESfmri.transform(test_data_dic)  

```

