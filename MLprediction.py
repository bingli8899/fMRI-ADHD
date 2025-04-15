import os
import sys
import pandas as pd
import numpy as np 
from datetime import datetime 
import argparse
import yaml
import ast 
import re

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from src.data.data_loader import load_or_cache_data
from src.utility.ut_general import relabel_train_outcome, recover_original_label 
from src.data.KNN_imputer import KNNImputer_with_OneHotEncoding
from src.utility.ut_general import normalizing_factors, write_model_grid 
from src.data.scaling import MeanStdScaler
from src.utility.ut_stats import select_top_columns_MutualInfo_4classes
from src.utility.ut_general import dict_to_namespace 

def choose_model_grid(model_name, class_weights_small_diff, class_weights_large_diff, seed):
    """Choose the correct model_grid based on model_name. Function got called in main()"""  
    if model_name == "GradientBoosting": 
        model_grid = { 
        "GradientBoosting": {
            "model": GradientBoostingClassifier,
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.1, 0.05, 0.01],
                "max_depth": [3, 5, 10],
                "criterion": ["friedman_mse", "squared_error"], 
                "subsample": [1.0, 0.8, 0.6], # default = 1.0
                "max_features": ["sqrt", "log2", None], 
                "min_samples_split": [2, 5], 
                "min_samples_leaf": [1, 2] 
            }}} 
    elif model_name == "RandomForest": 
        model_grid = {
        "RandomForest": {
        "model": RandomForestClassifier,
        "param_grid": {
            "n_estimators": [75, 100, 200, 300], # default = 100
            "max_depth": [None, 5, 10, 20], # less depth --> less overfitting, default = None
            "max_features": ["sqrt", "log2", None], # default=”sqrt”
            "min_samples_split": [2, 5, 10], # control regularization 
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy", "log_loss"], # default = gini 
            "class_weight": [class_weights_small_diff, class_weights_large_diff, None,"balanced_subsample"], # The dataset is "balanced" already
            "bootstrap": [True, False],
            "n_jobs": [-1] # use all cores 
        }}} 
    elif model_name == "AdaBoost": 
        model_grid = {"AdaBoost": {
        "model": AdaBoostClassifier,
        "param_grid": {
            "n_estimators": [50, 100, 125],
            "learning_rate": [1.0, 0.5, 0.01],
            "estimator": [
                DecisionTreeClassifier(max_depth=2, criterion="gini", class_weight=class_weights_small_diff),
                DecisionTreeClassifier(max_depth=2, criterion="entropy", class_weight=class_weights_small_diff),
                DecisionTreeClassifier(max_depth=3, criterion="gini", class_weight=class_weights_small_diff),
                DecisionTreeClassifier(max_depth=3, criterion="entropy", class_weight=class_weights_small_diff),
                DecisionTreeClassifier(max_depth=5, criterion="gini", class_weight=class_weights_small_diff),
                DecisionTreeClassifier(max_depth=5, criterion="entropy", class_weight=class_weights_small_diff),
                DecisionTreeClassifier(max_depth=2, criterion="gini", class_weight=class_weights_large_diff),
                DecisionTreeClassifier(max_depth=2, criterion="entropy", class_weight=class_weights_large_diff),
                DecisionTreeClassifier(max_depth=3, criterion="gini", class_weight=class_weights_large_diff),
                DecisionTreeClassifier(max_depth=3, criterion="entropy", class_weight=class_weights_large_diff),
                DecisionTreeClassifier(max_depth=5, criterion="gini", class_weight=class_weights_large_diff),
                DecisionTreeClassifier(max_depth=5, criterion="entropy", class_weight=class_weights_large_diff),
                DecisionTreeClassifier(max_depth=2, criterion="gini"),
                DecisionTreeClassifier(max_depth=2, criterion="entropy"),
                DecisionTreeClassifier(max_depth=3, criterion="gini"),
                DecisionTreeClassifier(max_depth=3, criterion="entropy"),
                DecisionTreeClassifier(max_depth=5, criterion="gini"),
                DecisionTreeClassifier(max_depth=5, criterion="entropy"),
                ]}}}
    elif model_name == "KNN": 
        model_grid = {
        "KNN": {
        "model": KNeighborsClassifier,
        "param_grid": {
                "n_neighbors": [3, 5, 7, 10],
                "weights": ["uniform", "distance"],
                "algorithm": ["ball_tree", "kd_tree", "brute"],
                "leaf_size": [20, 30, 40],
                "p": [1, 2], 
                "metric": ["minkowski", "euclidean", "manhattan"] 
            }}} 
    elif model_name == "SGDClassifier": 
        model_grid = {    
        "SGDClassifier": {
        "model": SGDClassifier,
        "param_grid": {
                "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"], 
                "penalty": ["l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01],  # Regularization strength
                "l1_ratio": [0.15, 0.25, 0.5, 0.75],  # Only used if penalty is elasticnet
                "max_iter": [1000, 2000],
                "tol": [1e-3, 1e-4],
                "learning_rate": ["optimal", "invscaling", "adaptive"],
                "eta0": [0.01, 0.1, 1.0],  # Initial learning rate
                "early_stopping": [True],
                "class_weight": [None, "balanced", class_weights_small_diff, class_weights_large_diff],
                "random_state": [seed],
                "n_jobs": [-1]  # This parameter is ignored by SGDClassifier, but harmless to include for consistency
        }}}
    elif model_name == "LogisticRegression": 
        model_grid = {
        # ---------------Logistic Regression -------------------# 
        # A group of logistic regression to avoid parameter conflicts: 
        "LogisticRegression_lbfgs": {
        "model": LogisticRegression,
        "param_grid": {
            "penalty": ["l2", None],
            "C": [0.01, 0.1, 1, 2.5, 5, 10],
            "solver": ["lbfgs"],
            "class_weight": [None, "balanced", class_weights_small_diff, class_weights_large_diff],
            "max_iter": [200, 300],
            "n_jobs": [-1],
        }},
        "LogisticRegression_liblinear": {
        "model": LogisticRegression,
        "param_grid": {
            "penalty": ["l1", "l2"],
            "C": [0.01, 0.1, 1, 2.5, 5, 10],
            "solver": ["liblinear"],
            "class_weight": [None, "balanced", class_weights_small_diff, class_weights_large_diff],
            "max_iter": [200, 300],
        }},
        "LogisticRegression_saga": {
        "model": LogisticRegression,
        "param_grid": {
            "penalty": ["l1", "l2", "elasticnet"],
            "C": [0.01, 0.1, 1, 2.5, 5, 10],
            "solver": ["saga"],
            "l1_ratio": [0.15, 0.5, 0.9],  # Only for elasticnet
            "class_weight": [None, "balanced", class_weights_small_diff, class_weights_large_diff],
            "max_iter": [200, 300],
            "n_jobs": [-1],
        }},
        "LogisticRegression_newton": {
        "model": LogisticRegression,
        "param_grid": {
            "penalty": ["l2", None],
            "C": [0.01, 0.1, 1, 2.5, 5, 10],
            "solver": ["newton-cg", "newton-cholesky", "sag"],
            "class_weight": [None, "balanced", class_weights_small_diff, class_weights_large_diff],
            "max_iter": [200, 300],
        }}}
    elif model_name == "testing": # This is only used for a quick testing for runnig the code 
        model_grid = {
        "LogisticRegression_lbfgs": {
        "model": LogisticRegression,
        "param_grid": {
            "penalty": [None],
            "C": [0.01],
            "solver": ["lbfgs"],
            "class_weight": [None],
            "max_iter": [200],
            "n_jobs": [-1],
        }}}
    return model_grid
   
def load_and_impute_data(datafolder, task, enable_fmri, k, split = "train"):
    pickle_file = os.path.join(datafolder, "data.pkl") 
    train_data_dic, test_data_dic = load_or_cache_data(datafolder, pickle_file)

    df_dic = {
        "train_cate": train_data_dic["train_cate"],
        "train_quant": train_data_dic["train_quant"],
        "test_cate": test_data_dic["test_cate"],
        "test_quant": test_data_dic["test_quant"],
        "train_fmri": train_data_dic["train_fmri"],
        "test_fmri": test_data_dic["test_fmri"],
        "train_outcome": train_data_dic["train_outcome"]
    }

    print("Imputing metadata...")
    imputer = KNNImputer_with_OneHotEncoding(k=5)
    train_metadata = imputer.fit_transform(df_dic, split="train").sort_values(by="participant_id").reset_index()
    test_metadata = imputer.fit_transform(df_dic, split="test").sort_values(by="participant_id").reset_index()
    print("Imputing done...")

    train_outcome = train_data_dic["train_outcome"].sort_values(by="participant_id")
    train_outcome_relabelled = relabel_train_outcome(train_outcome, task = task)["Label"]

    # De-bugging: 
    # print("Check the initial train_outcome and relabelled dataset")
    # print("train_outcome\n", train_outcome)
    # print("train_outcome_relabelled\n", train_outcome_relabelled)
    # print("Check relabel function\n", relabel_train_outcome(train_outcome))

    if split == "train":
        y_train = train_outcome_relabelled.reset_index(drop=True) 

    # De-buging --> index has been reset and participant_ids have been sorted for metadata 
    # print("check if feature corresponds with participant_ids\n")
    # print("train_metadata\n", train_metadata)
    # print("test_metadata\n", test_metadata)

    # Only the participant_ids from the test set is required: 
    test_participant_ids = test_metadata["participant_id"].values
    
    # pre-process metadata 
    train_meta_processed = preprocess_metadata(train_metadata.drop(columns="participant_id"))
    test_meta_processed = preprocess_metadata(test_metadata.drop(columns="participant_id"))

    # De-bugging
    # print("check the labeling of train_meta and test_meta after processed metadata \n")
    # print("train_meta_processed: \n", train_meta_processed) 
    # print("test_meta_processed: \n", test_meta_processed)

    # Select top k columns based on train and concatenate the matrix: 
    if enable_fmri: 
        print(f"Select top {k} columns to be concated to the dataset...")
        top_k_columns, train_fmri_selected = select_top_columns_MutualInfo_4classes(df_dic["train_fmri"], df_dic["train_outcome"], k = k) 
        train_fmri_selected_sorted = train_fmri_selected.sort_values(by = "participant_id").reset_index(drop=True)

        # de-bugging: 
        # print("train_fmri_selected after selcting top k cols \n", train_fmri_selected) 
        # print("train_fmri_selected after sorting \n", train_fmri_selected_sorted.head(10))
        
        test_fmri_selected = df_dic["test_fmri"][["participant_id"] + top_k_columns]
        test_fmri_selected_sorted = test_fmri_selected.sort_values(by="participant_id").reset_index(drop = True)

        # Double check if the ordering match between meta and fmri dataset: 
        assert (train_metadata["participant_id"].values == train_fmri_selected_sorted["participant_id"].values).all()
        assert (test_metadata["participant_id"].values == test_fmri_selected_sorted["participant_id"].values).all()
        assert (train_outcome["participant_id"].values == train_fmri_selected_sorted["participant_id"].values).all()

        # Drop participant_id and concatenate
        # reset the index and drop it since the index might be out of order 
        # train_fmri_features = train_fmri_selected_sorted.drop(columns="participant_id").reset_index(drop=True)
        # test_fmri_features = test_fmri_selected_sorted.drop(columns="participant_id").reset_index(drop=True)
        
        train_fmri_features = train_fmri_selected_sorted.drop(columns="participant_id")
        test_fmri_features = test_fmri_selected_sorted.drop(columns="participant_id")

        # De-bugging 
        # print("check fmri feature dataset after drop columns: \n") 
        # print("train_fmri_features\n", train_fmri_features) 
        # print("test_fmri_features\n", test_fmri_features)

        # Concatenate metadata + fMRI features
        X_train = pd.concat([train_meta_processed.reset_index(drop=True), train_fmri_features.reset_index(drop=True)], axis=1).values
        X_test = pd.concat([test_meta_processed.reset_index(drop=True), test_fmri_features.reset_index(drop=True)], axis=1).values

    else: # If not concatenate fMRI data
        X_train = train_meta_processed.reset_index(drop=True).values # Need to double check to see if to debug this
        X_test = test_meta_processed.reset_index(drop=True).values
        
    # De-bugging 
    # print("check the final pd datasets for both train and test: \n") 
    # np.set_printoptions(linewidth=np.inf)
    # print("X_train final pandas dataset: \n", X_train[:10])
    # print("labels used\n", y_train[:10])
    # print("X_test final dataset: \n", X_test)

    print("Dataset preparation done")

    return X_train, y_train, X_test, test_participant_ids

def encode_column_into_bins(df, column, bins, labels):
        df['binned'] = pd.cut(df[column], bins=bins, labels=labels, right=True)
        df_encoded = pd.get_dummies(df, columns=['binned'], prefix='', prefix_sep='', dtype=float)
        df_encoded = df_encoded.drop(columns=[column])
        return df_encoded

def preprocess_metadata(metadata):
    
    metadata['EHQ_EHQ_Total'] = metadata['EHQ_EHQ_Total'] / 200 + 0.5
    metadata = encode_column_into_bins(metadata, 'ColorVision_CV_Score' , 
                                       [0, 12, 100], 
                                       ['Color_Blind', 'Normal_Vision'])
    metadata = encode_column_into_bins(metadata, 
                                       'MRI_Track_Age_at_Scan' , 
                                       [0, 4, 11, 17, 30], 
                                       ['Infant', 'Child', "Adolescent", "Adult"])

    # Normalize everything to be between 0 and 1
    # See utility/ut_general
    for col in normalizing_factors:
        metadata[col] /= normalizing_factors[col]

    # Remove features in train which never appear
    columns_which_dont_appear_in_train = ["Basic_Demos_Study_Site_5", "PreInt_Demos_Fam_Child_Race_-1", "Barratt_Barratt_P1_Edu_-1", "Barratt_Barratt_P1_Occ_-1", "Barratt_Barratt_P2_Edu_-1", "Barratt_Barratt_P2_Occ_-1", "Infant"]
    for col in columns_which_dont_appear_in_train:
        metadata = metadata.drop(columns=[col])

    return metadata 

def scaling_X(X, scaler, train = True): 
    if train: 
        return scaler.fit_transform(X)
    else: 
        return scaler.transform(X)

def cross_validation(X, y, scaler, seed, num_folds, model_grid, scaler_enabled = True):
    
    if scaler_enabled: 
        X_scaled = scaling_X(X, scaler)
    
    if np.isnan(X_scaled).any():
        raise ValueError("NaNs found in X after scaling. Check MeanStdScaler.")

    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    smote = SMOTE(random_state=seed)

    results = []

    for model_name, config in model_grid.items():
        print(f"Running model: {model_name}")
        model_class = config["model"]
        grid = ParameterGrid(config["param_grid"])

        for params in grid:
            print(f"For {model_name}, starting a new parameter setting: {params}\n")
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y), 1):
                
                X_train, y_train = X_scaled[train_idx], y[train_idx]
                X_val, y_val = X_scaled[val_idx], y[val_idx]

                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                clf = model_class(**params)
                clf.fit(X_train_resampled, y_train_resampled)

                preds = clf.predict(X_val)
                acc = accuracy_score(y_val, preds)
                fold_scores.append(acc)

                print(f"Fold {fold} Accuracy: {acc}")

            mean_acc = sum(fold_scores) / len(fold_scores)
            print(f"Mean accuracy across all cvs for {model_name} with {params}: {mean_acc}")

            results.append({
            "model": model_name,
            "params": params,
            "mean_accuracy": mean_acc,
            "fold_scores": fold_scores})

    return results


def parse_txt_file(txt_file): 
    result = [] 
    with open(txt_file, "r") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Rank"):
            rank = int(re.findall(r"Rank (\d+):", line)[0])
            model = re.findall(r"Model: (.+)", lines[i + 1].strip())[0]
            params_line = lines[i + 2].strip().replace("Params: ", "")
            params = ast.literal_eval(params_line)
            result.append((rank, model, params))
            i += 4 
        else:
            i += 1
    return result # This is the model configuration to be used 

def run_inference_from_txt(txt_file, X_train, y_train, X_test, test_participant_ids, task, output_name, output_dir, scaler, scaler_enabled = True): 

    if scaler_enabled: 
        X_train_scaled = scaling_X(X_train, scaler)
        X_test_scaled = scaling_X(X_test, scaler, train = False)
    
    top_models = parse_txt_file(txt_file) 

    for rank, model_name, params in top_models:
        print(f"Rank {rank} | Model: {model_name}")
        print(f"Parameters: {params}\n")

        model_class_map = {
            "GradientBoosting": GradientBoostingClassifier,
            "RandomForest": RandomForestClassifier,
            "AdaBoost": AdaBoostClassifier, 
            "KNN": KNeighborsClassifier,
            "SGDClassifier": SGDClassifier,
            "LogisticRegression_lbfgs": LogisticRegression,
            "LogisticRegression_liblinear": LogisticRegression,
            "LogisticRegression_saga": LogisticRegression,
            "LogisticRegression_newton": LogisticRegression
        }

        if model_name not in model_class_map:
            raise ValueError(f"unsupported model: {model_name}")

        model_class = model_class_map[model_name]
        clf = model_class(**params)

        clf.fit(X_train_scaled, y_train) 
        y_pred = clf.predict(X_test_scaled)

        output_file = os.path.join(output_dir, f"final_predictions_{task}_{tag}.csv")
        with open(output_file, "w") as f:
            if task == "four":
                f.write("participant_id,ADHD_Outcome,Sex_F\n")
                for pid, pred in zip(test_participant_ids, y_pred):
                    ADHD_outcome, Sex_F = recover_original_label(pred)
                    f.write(f"{pid},\t{ADHD_outcome},\t{Sex_F}\n")
            else:
                if task == "adhd": 
                    f.write(f"participant_id,ADHD_Outcome\n")
                    for pid, pred in zip(test_participant_ids, y_pred):
                        f.write(f"{pid},{pred}\n")
                else: 
                    f.write(f"participant_id,Sex_Fe\n")
                    for pid, pred in zip(test_participant_ids, y_pred):
                        f.write(f"{pid},{pred}\n")

    print(f"Inference complete on {output_name}")


def main(args):

    now = datetime.now()

    time_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Set up configuration:
    config_path = args.train_config if args.train_config else args.test_config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)


    config = dict_to_namespace(config)

    # Set up folder path: 
    rootfolder = config.root_folder 
    sys.path.append(os.path.join(rootfolder))
    datafolder = rootfolder 

    #Change below to arguments later: 
    model_name = config.model_params.model_name 
    seed = config.model_params.seed
    num_folds = config.model_params.num_folds
    task = config.output_params.task # adhd, sex, four
    scaler_enabled = config.data_preparation.scaler_enabled # bool 
    enable_fmri = config.data_preparation.enable_fmri 
    k = config.data_preparation.k # top k fmri columns to be used

    if config.data_preparation.enable_fmri: 
        output_name = f"{model_name}-{task}Predic-{num_folds}folds-top{k}fmri"
    else: 
        output_name = f"{model_name}-{task}Predic-{num_folds}folds-NOfmri"

    if config.output_params.task == "four": 
        # four class prediction use string "0", "1", "2", "3"
        class_weights_small_diff = {"0": 0.28, "1": 0.24, "2": 0.24, "3": 0.24}
        class_weights_large_diff = {"0": 0.34, "1": 0.22, "2": 0.22, "3": 0.22}
    else: 
        # two class prediction use integar 1 and 0 
        class_weights_small_diff = {0: 0.53, 1: 0.47}
        class_weights_large_diff = {0: 0.6, 1: 0.4}

    # set up the scaler: 
    if scaler_enabled: 
        if config.data_preparation.scaler == "StandardScaler":
            scaler = MeanStdScaler()  

    model_grid = choose_model_grid(model_name, class_weights_small_diff, class_weights_large_diff, seed)

    X_train, y_train, X_test, test_participant_ids = load_and_impute_data(datafolder, task = task, enable_fmri = enable_fmri, k = k) 

    results = cross_validation(X_train, y_train, scaler, seed, num_folds, model_grid, scaler_enabled) 

    output_path = os.path.join(rootfolder, f"{output_name}_results_{time_string}.txt") 

    with open(output_path, "w") as f:
    
        f.write("Configuration Parameters:\n")
        f.write(f"Scaler enabled: {scaler_enabled}\n")
        f.write(f"Scaler type: {scaler.__class__.__name__}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Number of CV folds: {num_folds}\n")
        f.write(f"Enable fmri: {enable_fmri} if to use fmri data to the dataset\n")
        f.write(f"k, number of fmri columns got selected: {k}\n")
        f.write("--------------------------------------------------\n\n")

        write_model_grid(model_grid, f)

        sorted_results = sorted(results, key=lambda x: x["mean_accuracy"], reverse=True)

        # Save top 3 params setting for each model 
        for i, result in enumerate(sorted_results[:3], start=1):
            f.write(f"Rank {i}:\n")
            f.write(f"Model: {result['model']}\n")
            f.write(f"Params: {result['params']}\n")
            f.write(f"Mean cv accuracy: {result['mean_accuracy']:.4f}\n")
            f.write("--------------------------------------------------\n")

            print("#-------Next model--------#\n")
            print(f"Rank {i}:\n")
            print(f"Model: {result['model']}\n")
            print(f"Params: {result['params']}\n")
            print(f"Mean cv accuracy: {result['mean_accuracy']:.4f}\n")
            print("--------------------------------------------------\n")

    if config.run_inference_on_test: 
        txt_file = output_path
        output_dir = config.output_params.output_inference_dir
        os.mkdir(output_dir, exist_ok=True)
        run_inference_from_txt(txt_file, X_train, y_train, X_test, test_participant_ids, task, output_name, output_dir, scaler) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None)
    parser.add_argument("--test_config", type=str, default=None)
    args = parser.parse_args()

    main(args)


