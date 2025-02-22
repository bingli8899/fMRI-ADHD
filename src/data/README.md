# Documentation for src/data 

This is the module for codes to process and analyze data: 

Data loader: 

```
from src.data.data_loader import load_data

rootfolder = "PATH/TO/OUR/GITHUB/fMRI-AHDH"
datafolder = os.path.join(rootfolder, "data")

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