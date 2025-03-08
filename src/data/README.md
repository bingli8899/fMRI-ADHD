# Documentation for src/data 

This is the module for codes to process and analyze data: 

Initializing data loader: 
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

KNN imputer: 

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