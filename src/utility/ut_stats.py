
import os 
import numpy as np 
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, shapiro
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from src.utility.ut_general import relabel_train_outcome
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def chi2_with_one_columns(column_lst, target_column, df):
    """
    Calculate chi_2 between the target column and all columns in a column list (in the same df） 
    """
    for col in column_lst:
        contingency_table = pd.crosstab(df[target_column], df[col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"Chi-Square Test between {col} and {target_column}:")
        print(f"Chi-Square Stat: {chi2:.2f}, P-value: {p:.5f}")

def ttest_utest_with_one_column(column_lst, target_column, df): 
    """
    If the target column is normally distributed -> conduct t-test between the target column and all columns in a column list 
    
    If the target column is not normally distributed -> conduct u-test between the target column and all columns in a column list 
    
    Print the p-values for both tests 
    """
    df[target_column] = df[target_column].astype(str)
    for col in column_lst: 
        
        g1 = df[df[target_column] == "1"][col]
        g2 = df[df[target_column] == "0"][col]
        twogroups = df[col]
        p = shapiro(twogroups)[1] 
        
        if p <= 0.05: # Not normally distributed --> Mann-Whitney U test 
            u_stat, p_value_mwu = mannwhitneyu(g1, g2, alternative='two-sided') 
            print(f"p-value for u-test between {col} and {target_column} is {p_value_mwu} with u-stats {u_stat}")
        else: # Normally distributed --> t-test 
            t_stat, p_value_t = ttest_ind(g1, g2, nan_policy='omit')
            print(f"p-value for t-test between {col} and {target_column} is {p_value_t} with t-stats {t_stat}")


def select_top_columns_MutualInfo_4classes(train_fmri, fmri_outcomes, task, k = 100):
    """The function select top k columns in the train dataset with high correlation with the label"""
    
    df_sorted = train_fmri.sort_values(by="participant_id")
    label_sorted = fmri_outcomes.sort_values(by="participant_id") 
    train_label = relabel_train_outcome(label_sorted, task = task)
    
    if (df_sorted["participant_id"].values != train_label["participant_id"].values).all(): 
        raise ValueError("Oh nooooo! Mismatch in participant id")
    
    X = df_sorted.drop(columns = "participant_id")
    y = train_label["Label"].astype(str)

    mi_scores = mutual_info_classif(X, y) 
    top_k_idx = np.argsort(mi_scores)[::-1][:k]
    top_k_columns = X.columns[top_k_idx].tolist()
    
    filtered_df = train_fmri[["participant_id"] + top_k_columns] # still extract from the original dataframe since this will be sorted in create_graph_lst 

    return top_k_columns, filtered_df 


def apply_LDA_to_train_and_test(train_fmri, test_fmri, fmri_outcomes, task, n_components=3):
    """
    Fits LDA on train_fmri and applies it to both train and test fMRI datasets.
    Returns:
        lda_cols: List of LDA column names
        train_lda_df: DataFrame with participant_id + LDA components (train)
        test_lda_df: DataFrame with participant_id + LDA components (test)
    """
    train_sorted = train_fmri.sort_values(by="participant_id").reset_index(drop=True)
    test_sorted = test_fmri.sort_values(by="participant_id").reset_index(drop=True)
    labels_sorted = fmri_outcomes.sort_values(by="participant_id").reset_index(drop=True)
    label_df = relabel_train_outcome(labels_sorted, task = task)

    if not (train_sorted["participant_id"].values == label_df["participant_id"].values).all():
        raise ValueError("OHHH NO! Mismatch in participant IDs!")

    X_train = train_sorted.drop(columns="participant_id").values
    X_test = test_sorted.drop(columns="participant_id").values
    y_train = label_df["Label"].astype(str).values

    # Fit LDA on train set
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    lda_cols = [f"LDA{i+1}" for i in range(X_train_lda.shape[1])]

    # output dataset has participant ids 
    train_lda_df = pd.DataFrame(X_train_lda, columns=lda_cols)
    train_lda_df.insert(0, "participant_id", train_sorted["participant_id"].values)

    test_lda_df = pd.DataFrame(X_test_lda, columns=lda_cols)
    test_lda_df.insert(0, "participant_id", test_sorted["participant_id"].values)

    return train_lda_df, test_lda_df



