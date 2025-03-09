# fMRI-AHDH
Repo to document my model for WiDS 2025 

**Root Directory**
## Repository Structure

Please follow the data structure below. 
Data is not pushed and should be downloaded from: https://www.kaggle.com/competitions/widsdatathon2025/data 
Usage of data loader and cacher is listed in [README.md](src/data/README.md) 

- [README.md](README.md)  
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
    - [README.md](src/data/README.md)
  - `model/`
    - [__init__.py](src/model/__init__.py)  
  - `utility/`  
    - [__init__.py](src/utility/__init__.py)  
    - [ut_general.py](src/utility/ut_general.py)  
    - [ut_stats.py](src/utility/ut_stats.py) 
    - [ut_visualization.py](src/utility/ut_visualization.py)  
