# fMRI-AHDH
Repo to document my model for WiDS 2025 

**Root Directory**
## Repository Structure

Please follow the data structure below. 
Data is not pushed and should be downloaded from: https://www.kaggle.com/competitions/widsdatathon2025/data 

- [README.md](README.md)  
- [requirements.txt]
- `data/`  -> Data not pushed 
  - `TEST/`  
  - `TRAIN/`  
- `notebooks/`  
  - [CheckMissingData.ipynb](notebooks/CheckMissingData.ipynb)  
  - [InitialDataCheck.ipynb](notebooks/CheckMissingData.ipynb)
  - [KNNimputer_reasoning.ipynb](notebooks/CheckMissingData.ipynb)
  - [weekly_notes.md](notebooks/weekly_notes.md)  
- `src/` 
  - [__init__.py](src/__init__.py) 
  - `data/`  
    - [__init__.py](src/data/__init__.py)  
    - [data_loader.py](src/data/data_loader.py)  
    - [KNN_imputer.py](src/data/KNN_imputer.py)
    - [make_toy_dataset.py](src/data/make_toy_dataset.py)
    - [scaling.py](src/data/scaling.py)
    - [summarize_csv.py]((src/data/summarize_csv.py))
    - [README.md](src/data/README.md)
  - `model/`
    - [__init__.py](src/model/__init__.py) 
    - [GCN.py](src/model/GCN.py)
    - [NNconv.py](src/model/NNconv.py)
    - [SageGNN.py](src/model/SageGNN.py)
    - [TransConv.py](src/model/TransConv.py)
    - [GATv2.py](src/model/GATv2.py)
    - [DirGNNConv.py](src/model/DirGNNConv.py)
    - [DirGNN_GatConv](src/model/DirGNN_GatConv.py)
  - `utility/`  
    - [__init__.py](src/utility/__init__.py)  
    - [ut_general.py](src/utility/ut_general.py)  
    - [ut_stats.py](src/utility/ut_stats.py) 
    - [ut_model.py](src/utility/ut_model.py) 
    - [ut_visualization.py](src/utility/ut_visualization.py)  
  - `config/`
    -[test_config.yaml](config/test_config.yaml)
    -[train_config_ML.yaml](config/train_config_ML.yaml)
    -[train_config_nnConv.yaml](config/train_config_nnConv.yaml)
    -[train_config_GCN.yaml](config/train_config_GCN.yaml)
    -[train_config_DirGNN.yaml](config/train_config_DirGNN.yaml)
    -[train_config_GATv2.yaml](config/train_config_GATv2.yaml)
    -[train_config_SageGNN.yaml](config/train_config_SageGNN.yaml)
    -[train_config_TransConv.yaml](config/train_config_TransConv.yaml)
    -[train_config_DirGNN_GatConv.yaml](config/train_config_DirGNN_GatConv.yaml)

To run GNN models for trainning; 
```
python main.py --train_config config/train_config_[model_name].yaml 
```
Model specific parameters could be tested within the configuration 

After that, if running inference on the test dataset 
```
python main.py --test_config config/test_config.yaml # Need to change checkpoint pathway inside 
```

To run traditional ML models for trainning and testing: 
```
python MLprediction.py --config config/train_config_ML.yaml 
```
