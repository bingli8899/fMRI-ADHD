# fMRI-AHDH
Repo to document traditional ML models and Graph Neural Networks (GNN) built for for WiDS 2025, using fMRI connectome data and demographic data to predict sex-specific ADHD diagnosis. 

The repo demostrates ML models (MLprediction.py) and seven GNN model archiectures (GNNprediction.py). 

## 🚀 How about a Quick Start? 
### GNN Model Training
To run GNN models for training:
```bash
python GNNprediction.py --train_config config/train_config_[model_name].yaml 
```
Model specific parameters could be tested within the configuration. This repo contains 9 GNN model architectures, including:
1. **GCN** (Graph Convolutional Network)
2. **GATv2** (Graph Attention Network v2) 
3. **SageGNN** (GraphSAGE)
4. **TransConv** (Transformer Convolution)
5. **NNConv** (Neural Network Convolution)
6. **DirGNN** (Directional GNN)
7. **DirGNN_GatConv** (Ensemble of Directional GNN and GAT) 

### GNN Model Inference
For running inference on the test dataset:
```bash
python GNNprediction.py --test_config config/test_config.yaml # Need to change checkpoint pathway inside 
```

### Traditional ML Models
To run traditional ML models for training and testing:
```bash
python MLprediction.py --config config/train_config_ML.yaml 
```

**ML Model Training & Inference Modes:**
The traditional ML pipeline supports flexible training and inference modes controlled by configuration parameters:

- **Training + Inference Mode:** `training_enabled: True` and `run_inference_on_test: True`
  - Trains new models using cross-validation
  - Automatically runs inference on test data using best trained model
  - Saves both training results and test predictions

- **Inference Only Mode:** `training_enabled: False` and `run_inference_on_test: True`
  - Loads pre-trained model checkpoints
  - Runs inference on test data only
  - Requires existing model files and configurations

- **Training Only Mode:** `training_enabled: True` and `run_inference_on_test: False`
  - Trains and evaluates models using cross-validation
  - Saves trained models but skips test inference
  - Useful for model development and hyperparameter tuning 


## 📋 GNNprediction.py Scripts Documentation
**Purpose:** Main inference script for Graph Neural Network models. tLoads trained GNN models and performs prediction on test data. Below is a figure containing the overview of our major workflow: 
[View PDF Documentation](readme_image/GNN_explained.pdf)
![GNN_explanations](readme_image/GNN_explained.png)

**Key Features:**
- **Data Preprocessing Pipeline:** Comprehensive fMRI connectome data preprocessing and scaling with the following steps:
  1. **Missing Data Imputation:** Uses KNN imputation to handle missing values (implementation in `src/data/KNN_imputer.py`)
  2. **Distribution Normalization:** Applies Fisher-Z transformation to normalize training fMRI data to standard normal distribution, then uses Mean Standard Scaler for test dataset scaling (implementation in `src/data/scaling.py`)
  3. **Class Imbalance Handling:** Implements custom graph augmentation techniques to generate additional graphs for underrepresented labels (implementation in `GNNprediction.py`)
  4. **Demographic Data** Process and normalzie demographic data （`src/utility/ut_general.py`）

- **Graph Building:** Converts fMRI correlation matrices into graph-structured data for GNN processing:
  - **Graph Construction Methods:**
    - **Directional Graphs:** `create_graph_lst()` - Creates directed graphs preserving correlation sign and directionality
    - **Undirectional Graphs:** `create_undirectional_graph_lst()` - Separates positive and negative correlations into distinct graphs
  - **Graph Data Structure:**
    - **Node Features (x):** 200×200 tensor representing brain regions
    - **Edge Indices:** Source and target node pairs for connections
    - **Edge Attributes:** Correlation strengths between brain regions
    - **Labels (y):** ADHD/sex classification labels
    - **Metadata:** Participant IDs for tracking 

- **Model Training:** Trains GNN models based on selected architecture with fully configurable hyperparameters:
  - **Architecture Parameters:**
    - Number of layers and hidden channels
    - Batch normalization and graph normalization options
    - Dropout rates for fully connected (FC) and GNN layers
  - **Pooling Options:**
    - Global pooling methods (Attention, Max, Average pooling)
    - Learnable attention-based pooling mechanisms
  - **Feature Integration:**
    - Metadata incorporation as flattened 81-dimensional vectors
    - Optional MLP preprocessing of demographic features
  - **Advanced Features:**
    - Residual connections in GNN architectures
    - Configurable activation functions (ReLU, LeakyReLU, GELU)
    - Model-specific parameters (alpha values for directional models, attention heads for attention-based models) 
  - **CV-based or model-based hyperparameters:** 
    - K fold CV (Optionally, the script can train on the full training dataset)
    - Number of batch size per epoch 
    - Number of epochs and early stopping based on N epochs 
    - Learning rate scheduler and starting learning rate
    - Label smoothing 
    - Master seed for reproducibility 

- **Traning result monitoring** Train/Validation accuracy and loss are all uploaded to weights and bias `wandb` for better visualization. 
- **Model Loading:** Automatically loads trained GNN models from checkpoint directories
- **Output Generation:** Saves predictions in standardized format for submission

**Input Data:**
- fMRI connectome correlation matrices (19,900 features per participant)
- Demographic metadata (age, sex, handedness, etc.)
- Trained model checkpoint files

**Output:**
- Prediction CSV files with participant IDs and predicted labels
- Confidence scores and model performance metrics
- Inference logs and processing statistics

**Supported GNN Architectures:**
1. **GCN** (Graph Convolutional Network)
   - **Architecture:** Multi-layer spectral graph convolution
   - **Key Features:** Normalized adjacency matrix, configurable normalization layers
   - **Pooling Options:** Global mean/max/add/sort pooling
   - **Metadata Integration:** Concatenation with 81-dimensional demographic features

2. **GATv2** (Graph Attention Network v2)
   - **Architecture:** Multi-head attention mechanism for node interactions
   - **Key Features:** Edge-aware attention, configurable attention heads
   - **Attention Pooling:** GlobalAttention with learnable gate networks
   - **Edge Dimensions:** Supports edge attributes (correlation strengths)

3. **SageGNN** (GraphSAGE)
   - **Architecture:** Sampling and aggregating from node neighborhoods
   - **Key Features:** Configurable aggregation functions (mean, max, LSTM)
   - **Normalization:** Optional L2 normalization of node embeddings
   - **Scalability:** Designed for large-scale graph processing

4. **TransConv** (Transformer Convolution)
   - **Architecture:** Self-attention mechanism adapted for graphs
   - **Key Features:** Multi-head attention, edge-aware transformations
   - **Aggregation:** Max aggregation across attention heads
   - **Batch Normalization:** Optional batch normalization layers

5. **NNConv** (Neural Network Convolution)
   - **Architecture:** Edge-conditioned convolution with neural networks
   - **Key Features:** Edge-specific transformation functions
   - **Neural Networks:** Two-layer MLPs for edge weight processing
   - **Aggregation:** Mean aggregation of neighbor messages

6. **DirGNN** (Directional GNN)
   - **Architecture:** Directional message passing for asymmetric graphs
   - **Key Features:** Alpha parameter for controlling directional influence
   - **Base Convolution:** Built on top of GCN layers
   - **Layers:** Three-layer architecture with configurable hidden dimensions

7. **DirGNN_GatConv** (Ensemble of Directional GNN and GAT)
   - **Architecture:** Hybrid model combining directional and attention mechanisms
   - **Key Features:** Parallel DirGNN and GATv2 branches
   - **Ensemble Method:** Concatenation of both branch outputs
   - **Flexibility:** Independent dropout rates for each branch

##📋 MLprediction.py (Traditional ML Pipeline)
**Location:** `MLprediction.py`

**Purpose:** Traditional machine learning pipeline for fMRI-based ADHD and sex prediction using classical ML algorithms.

**Key Features:**
- **Multiple ML Models:** RandomForest, GradientBoosting, AdaBoost, KNN, SVM, Logistic Regression
- **Feature Selection:** Our configurations allow for choosing between three feature selection methods: 
    - Mutual Information (based on K number of columns selected)
    - Linear Discriminant Analysis (LDA) 
    - No feature selection  
PS: Demographic data could be appeneded or not depending on user's choice. 
- **Class Imbalance Handling:** SMOTE oversampling for balanced training
- **Hyperparameter Optimization:** Grid search with cross-validation. We pre-defined a parameter searching grid and the script will train based on the grid and selected model. 

**Supported ML Models:**
1. **RandomForest** - Ensemble of decision trees with feature bagging
2. **GradientBoosting** - Gradient boosting with configurable learning rates
3. **AdaBoost** - Adaptive boosting with decision tree base estimators
4. **KNN** - K-nearest neighbors with distance weighting
5. **SGDClassifier** - Stochastic gradient descent classifier
6. **LogisticRegression** - Regularized logistic regression
7. **SVM** - Support Vector Machine with RBF kernel
8. **NuSVC** - Nu-Support Vector Classifier

**Feature Selection Options:**
- **Mutual Information:** Selects top-k features based on mutual information scores
- **Linear Discriminant Analysis (LDA):** Dimensionality reduction for classification
- **Combined Features:** fMRI connectome + demographic metadata

**Usage Example:**
```bash
python MLprediction.py --config config/train_config_ML.yaml
```


