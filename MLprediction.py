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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, NuSVC

from src.data.data_loader import load_or_cache_data
from src.utility.ut_general import relabel_train_outcome, recover_original_label 
from src.data.KNN_imputer import KNNImputer_with_OneHotEncoding
from src.utility.ut_general import normalizing_factors, write_model_grid 
from src.data.scaling import MeanStdScaler
from src.utility.ut_stats import select_top_columns_MutualInfo_4classes, apply_LDA_to_train_and_test
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
            "C": [0.01, 0.02, 0.03],
            "solver": ["lbfgs"],
            "class_weight": [None],
            "max_iter": [200],
            "n_jobs": [-1],
        }}}
    elif model_name == "SVM": 
        model_grid = {
        "SVC": {
            "model": SVC,
            "param_grid": {
                "C": [0.1, 0.5, 0.3, 1, 3, 5, 10],
                "kernel": ['sigmoid', 'linear', 'rbf', 'poly'],
                "gamma": ['scale', 'auto'],
                "degree": [2, 3, 4, 5, 6],  # used only with 'poly' kernel
                "probability": [True],
                "coef0": [0.0, 0.25, 0.5, 0.75, 1, 3, 5],
                "class_weight": [None, 'balanced', class_weights_small_diff, class_weights_large_diff],
                "decision_function_shape": ['ovo']
            }}}
    elif model_name == "NuSVM": 
        model_grid = {
        "NuSVC": {
            "model": NuSVC,
            "param_grid": {
                "nu": [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99],
                "kernel": ['sigmoid', 'linear', 'rbf', 'poly'],
                "gamma": ['scale', 'auto'],
                "degree": [2, 3, 4, 5, 6],
                "probability": [True],
                "coef0": [0.0, 0.25, 0.5, 0.75, 1, 3, 5],
                "class_weight": [None, 'balanced', class_weights_small_diff, class_weights_large_diff]
            }}}
    return model_grid
   
def load_and_impute_data(datafolder, task, ida, mutual_info, enable_fmri, k, scaler, scaler_enabled = True, split = "train"):
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
    
    # de-bugging: 
    # print("test_participant_ids\n", test_participant_ids) 
    
    # pre-process metadata 
    train_meta_processed = preprocess_metadata(train_metadata.drop(columns="participant_id"))
    test_meta_processed = preprocess_metadata(test_metadata.drop(columns="participant_id"))

    # De-bugging
    # print("check the labeling of train_meta and test_meta after processed metadata \n")
    # print("train_meta_processed: \n", train_meta_processed) 
    # print("test_meta_processed: \n", test_meta_processed)

    # Select top k columns based on train and concatenate the matrix: 
    if enable_fmri: 

        if mutual_info: 
            print(f"Selecting top {k} columns to be concated to the train dataset...")
            top_k_columns, train_fmri_selected = select_top_columns_MutualInfo_4classes(df_dic["train_fmri"], 
                                                                                        df_dic["train_outcome"], 
                                                                                        task = task, k = k) 
            train_fmri_selected_sorted = train_fmri_selected.sort_values(by = "participant_id").reset_index(drop=True)
            
            # de-bugging: 
            print("top_k_columns, ", top_k_columns)
            
            print(f"Selecting top {k} columns to be concated to the test dataset...")
            test_fmri_sorted = df_dic["test_fmri"].sort_values(by="participant_id")
            test_fmri_selected_sorted = test_fmri_sorted[["participant_id"] + top_k_columns]
            test_fmri_selected_sorted = test_fmri_selected_sorted.reset_index(drop=True)

            # test_fmri_selected = df_dic["test_fmri"][["participant_id"] + top_k_columns]
            # test_fmri_selected_sorted = test_fmri_selected.sort_values(by="participant_id").reset_index(drop=True)

            # de-bugging: 
            # print("train_fmri_selected after selcting top k cols \n", train_fmri_selected) 
            # print("train_fmri_selected after sorting \n", train_fmri_selected_sorted)
            # print("test_fmri_sorted \n", test_fmri_sorted)
            # print("test_fmri_selected_sorted after sorting \n", test_fmri_selected_sorted)

            # Drop participant_id
            train_fmri_features = train_fmri_selected_sorted.drop(columns="participant_id")
            test_fmri_features = test_fmri_selected_sorted.drop(columns="participant_id")

            # Only scale the fmri dataset: 
            if scaler_enabled: 
                print("Scaling both test and train fmri data")
                train_fmri_features = scaling_X(train_fmri_features, scaler, train = True)
                test_fmri_features = scaling_X(test_fmri_features, scaler, train = False)
            if test_fmri_features.isna().values.any() or train_fmri_features.isna().values.any():
                raise ValueError("NaNs found in X after scaling. Check MeanStdScaler.")
            
            # de-bugging: 
            # print("train_fmri_features_scaled after scaling:\n", train_fmri_features_scaled)
            # print("test_fmri_features_scaled after scaling:\n", test_fmri_features_scaled)

        elif ida: 

            print(f"Doing IDA for both train and test...")

            train_fmri_sorted = df_dic["train_fmri"].sort_values(by="participant_id").reset_index(drop=True)
            test_fmri_sorted = df_dic["test_fmri"].sort_values(by="participant_id").reset_index(drop=True)
            train_outcome = df_dic["train_outcome"].sort_values(by="participant_id").reset_index(drop=True)

            train_participant_ids = train_fmri_sorted["participant_id"]
            test_participant_ids = test_fmri_sorted["participant_id"]

            train_fmri_ready_to_be_scaled = train_fmri_sorted.drop(columns="participant_id")
            test_fmri_ready_to_be_scaled = test_fmri_sorted.drop(columns="participant_id")

            # Double check the ordering since I am scared 
            assert (train_fmri_sorted["participant_id"].values == train_outcome["participant_id"].values).all(), \
            "Mismatch between train_fmri_sorted and train_outcome participant IDs!"

            assert (train_fmri_sorted["participant_id"].values == train_participant_ids.values).all(), \
                "Mismatch between train_fmri_sorted and train_participant_ids!"

            assert (test_fmri_sorted["participant_id"].values == test_participant_ids.values).all(), \
                "Mismatch between test_fmri_sorted and test_participant_ids!"
            
            assert (train_outcome["participant_id"].values == train_participant_ids.values).all(), \
                "Mismatch between train_outcome and train_participant_ids!"

            # Scale the input to ida first: 
            if scaler_enabled: 
                print("Scaling both test and train fmri data")
                train_fmri_scaled = scaling_X(train_fmri_ready_to_be_scaled, scaler, train = True)
                test_fmri_scaled = scaling_X(test_fmri_ready_to_be_scaled, scaler, train = False)
            if test_fmri_scaled.isna().values.any() or train_fmri_scaled.isna().values.any():
                raise ValueError("NaNs found in X after scaling. Check MeanStdScaler.")
            
            # I know this is very redundant but I don't want to think of anything easier at this moment ... 
            train_fmri_scaled.insert(0, "participant_id", train_participant_ids)
            test_fmri_scaled.insert(0, "participant_id", test_participant_ids)

            n_components = 3 if task == "four" else 1 # n_classes - 1 

            train_fmri_selected, test_fmri_selected = apply_LDA_to_train_and_test(train_fmri = train_fmri_scaled, 
                                                                        test_fmri = test_fmri_scaled, 
                                                                        fmri_outcomes = train_outcome, 
                                                                        task = task, 
                                                                        n_components= n_components)

            # de-bugging: 
            print("train_fmri_selected after lda", train_fmri_selected)
            print("test_fmri_selected after lda", test_fmri_selected)

            # Although these below have been sorted. Still sort again since I am scared now ... 
            train_fmri_selected_sorted = train_fmri_selected.sort_values(by = "participant_id").reset_index(drop=True)
            test_fmri_selected_sorted = test_fmri_selected.sort_values(by = "participant_id").reset_index(drop=True)
            
            # drop participant_id again: 
            train_fmri_features = train_fmri_selected_sorted.drop(columns="participant_id")
            test_fmri_features = test_fmri_selected_sorted.drop(columns="participant_id")
        
        else: 
            raise ValueError("When enable_fmri is true, EITHER but NOT BOTH mutual_info or ida below must be set up to true")

        # Double check if the ordering match between meta and fmri dataset: 
        assert (train_metadata["participant_id"].values == train_fmri_selected_sorted["participant_id"].values).all()
        assert (test_metadata["participant_id"].values == test_fmri_selected_sorted["participant_id"].values).all()
        assert (train_outcome["participant_id"].values == train_fmri_selected_sorted["participant_id"].values).all()

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
    return scaler.fit_transform(X) if train else scaler.transform(X)

def cross_validation(X, y, seed, num_folds, model_grid):

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

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

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

def run_inference_from_txt(txt_file, X_train, y_train, X_test, test_participant_ids, task, output_name, output_dir): 
    
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
            "LogisticRegression_newton": LogisticRegression,
            "SVC": SVC, 
            "nuSVC": NuSVC
        }

    # Start with participant ID column
    prediction_df = pd.DataFrame({"participant_id": test_participant_ids})

    for rank, model_name, params in top_models:
        print(f"Rank {rank} - Model: {model_name}")
        print(f"Parameters: {params}\n")

        if model_name not in model_class_map:
            raise ValueError(f"Unsupported model: {model_name}")

        model_class = model_class_map[model_name]
        clf = model_class(**params)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Handle multi-output if task == "four"
        if task == "four":
            ADHD_outcomes = []
            Sex_F = []
            for pred in y_pred:
                adhd_label, sex_label = recover_original_label(pred)
                ADHD_outcomes.append(adhd_label)
                Sex_F.append(sex_label)
            prediction_df[f"ADHD_Outcome_Rank{rank}"] = ADHD_outcomes
            prediction_df[f"Sex_F_Rank{rank}"] = Sex_F

        elif task == "adhd":
            prediction_df[f"ADHD_Outcome_Rank{rank}"] = y_pred
        elif task == "sex":
            prediction_df[f"Sex_F_Rank{rank}"] = y_pred
        else:
            raise ValueError(f"Unsupported task type: {task}")

    # Save to file (overwrite by default)
    output_file = os.path.join(output_dir, f"final_predictions_{output_name}.csv")
    prediction_df.to_csv(output_file, index=False, sep="\t")

    print(f"Inference complete on {output_name} — predictions for all ranks written to {output_file}")


def main(args):

    now = datetime.now()

    time_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Set up configuration:
    config_path = args.config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config = dict_to_namespace(config)

    # Set up folder path: 
    rootfolder = config.root_folder 
    sys.path.append(os.path.join(rootfolder))
    datafolder = os.path.join(rootfolder, "data")
    
    # Set to config --> I am tired of passing config to all functions lol 
    model_name = config.model_params.model_name 
    seed = config.model_params.seed
    num_folds = config.model_params.num_folds
    task = config.output_params.task # adhd, sex, four
    scaler_enabled = config.data_preparation.scaler_enabled # bool 
    enable_fmri = config.data_preparation.enable_fmri 
    k = config.data_preparation.k # top k fmri columns to be used
    training_enabled = config.output_params.training_enabled # bool to enable training 
    testing_enabled = config.output_params.run_inference_on_test # bool to enable testing 
    aggregate_inference = config.output_params.aggregate_inference
    ida = config.data_preparation.ida
    mutual_info = config.data_preparation.mutual_info

    # mutual_info and ida cannot be true at the same time: 
    if mutual_info and ida:
        raise ValueError("OHHH Nooo! You cannot enable both 'mutual_info' and 'ida' at the same time.")

    if config.data_preparation.enable_fmri: 
        if mutual_info: 
            output_name = f"{model_name}-{task}Predic-MutualIndo-top{k}fmri"
        else: #ida 
            output_name = f"{model_name}-{task}Predic-ida"
    else: 
        output_name = f"{model_name}-{task}Predic-NOfmri"

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

    X_train, y_train, X_test, test_participant_ids = load_and_impute_data(datafolder, 
                                                                          task = task, 
                                                                          ida = ida,
                                                                          mutual_info = mutual_info, 
                                                                          enable_fmri = enable_fmri, 
                                                                          k = k, 
                                                                          scaler = scaler, 
                                                                          scaler_enabled = True)

    # de-bugging: 
    # print("X-train after data loading:\n", X_train)
    # print("y_train after data loading:\n", y_train)
    # print("X_test after data loading:\n", X_test)
    # print("test participant ids:\n", test_participant_ids) 

    # output_path is the file that save the final results from the training 
    # the same path is also used to run inference on the test dataset as the inputs for getting model parameters 
    output_path = os.path.join(rootfolder, f"{output_name}_results_{time_string}.txt")  
    # However, if the prediction is run on a different time not listed in the time_string, then the output_path cannot be identified, so the solution is seen below 

    if training_enabled: 

        print("Training on the dataset...")

        results = cross_validation(X_train, y_train, seed, num_folds, model_grid) 

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

    if testing_enabled: 
        print("Running inference on the test dataset...")

        time_pattern = r"_results_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}"
        txt_file = rf"{output_name}{time_pattern}\.txt"

        matched_file = False
        for filename in os.listdir(rootfolder):
            if re.fullmatch(txt_file, filename):
                txt_file = os.path.join(rootfolder, filename)
                matched_file = True
                break
        
        if matched_file: 
            output_dir = config.output_params.output_inference_dir
            os.makedirs(output_dir, exist_ok=True)
            run_inference_from_txt(txt_file, X_train, y_train, X_test, test_participant_ids, task, output_name, output_dir) 
        else: 
            raise ValueError("Oh nooo! No input txt file for run inference!")
        
        if aggregate_inference: 

            print("Now aggregating inference results...") 
            output_file = os.path.join(output_dir, f"final_predictions_{output_name}.csv") 
            df = pd.read_csv(output_file, sep="\t") 

            if task == "four": 
                agg_adhd = []
                agg_sex = []
                consistent_flags = []
            
                for i in range(len(df)):
                    adhd_preds = [df.loc[i, "ADHD_Outcome_Rank1"],
                                df.loc[i, "ADHD_Outcome_Rank2"],
                                df.loc[i, "ADHD_Outcome_Rank3"]]
                    sex_preds = [df.loc[i, "Sex_F_Rank1"],
                                df.loc[i, "Sex_F_Rank2"],
                                df.loc[i, "Sex_F_Rank3"]]
                    
                    # find majority vote: 
                    agg_adhd.append(pd.Series(adhd_preds).mode()[0])
                    agg_sex.append(pd.Series(sex_preds).mode()[0])

                    # Consistency
                    if len(set(adhd_preds)) == 1 and len(set(sex_preds)) == 1:
                        consistent_flags.append("Y")
                    else:
                        consistent_flags.append("N")
                        print(f"Warning: inconsistent prediction(s) found across Rank1–3.") 
                        print("Check the \'consistent_or_not\' column in the aggregated file.")

                df["consistent_or_not"] = consistent_flags
                df["agg_ADHD_outcome"] = agg_adhd
                df["agg_Sex_F"] = agg_sex
                
            elif task == "adhd":
                agg_preds = []
                consistent_flags = []

                for i in range(len(df)):
                    preds = [df.loc[i, "ADHD_Outcome_Rank1"],
                            df.loc[i, "ADHD_Outcome_Rank2"],
                            df.loc[i, "ADHD_Outcome_Rank3"]]
                    agg_preds.append(pd.Series(preds).mode()[0])

                    if len(set(preds)) == 1: 
                        consistent_flags.append("Y")
                    else:
                        consistent_flags.append("N")
                        print(f"Warning: inconsistent prediction(s) found across Rank1–3.") 
                        print("Check the \'consistent_or_not\' column in the aggregated file.")


                df["consistent_or_not"] = consistent_flags
                df["agg_ADHD_Outcome"] = agg_preds

            elif task == "sex":
                agg_preds = []
                consistent_flags = []

                for i in range(len(df)):
                    preds = [df.loc[i, "Sex_F_Rank1"],
                            df.loc[i, "Sex_F_Rank2"],
                            df.loc[i, "Sex_F_Rank3"]]
                    agg_preds.append(pd.Series(preds).mode()[0])
                    
                    if len(set(preds)) == 1: 
                        consistent_flags.append("Y")
                    else:
                        consistent_flags.append("N")
                        print(f"Warning: inconsistent prediction(s) found across Rank1–3.") 
                        print("Check the \'consistent_or_not\' column in the aggregated file.")

                df["consistent_or_not"] = consistent_flags
                df["agg_Sex_F"] = agg_preds

            else:
                raise ValueError(f"Unknown task type {task}. U idiot")

            # Overwrite the output_file 
            df.to_csv(output_file, sep="\t", index=False)
            print(f"Oyay! Aggregation of inference is complete and saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    main(args)


