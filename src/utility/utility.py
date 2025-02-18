# Scripts to save utility function 
import os 
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, shapiro

def missing_percentage(df, col): 
    missing = df[col].isnull().sum()
    missing_percentage = (missing / len(df[col])) * 100 
    if missing_percentage > 0: # No return is there is no missing value in [col] 
        print(f"col {col} has {missing_percentage} % missing.")
        return col # return col if there is missing value in [col]
    
def return_missing_list(df, df_name): 
    missing_lst = []
    for col in df.columns: 
        missing_col = missing_percentage(df, col)
        missing_lst.append(missing_col)
    
    if missing_lst == []: 
        print(f"{df_name} has no missing value") 
    else: 
        return missing_lst # Only return if there are missing values in df

def chi2_with_one_columns(column_lst, target_column, df):
    for col in column_lst:
        contingency_table = pd.crosstab(df[target_column], df[col])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Test between {col} and {target_column}:")
        print(f"Chi-Square Stat: {chi2:.2f}, P-value: {p:.5f}")

def ttest_utest_with_one_column(column_lst, target_column, df): 
    df[target_column] = df[target_column].astype(str)
    for col in column_lst: 
        
        g1 = df[df[target_column] == "1"][col]
        g2 = df[df[target_column] == "0"][col]
        twogroups = df[col]
        p = shapiro(twogroups)[1] 
        
        if p <= 0.05: # Not normally distributed --> Mann-Whitney U test 
            u_stat, p_value_mwu = mannwhitneyu(g1, g2, alternative='two-sided') 
            print(f"p-value for u-test between {col} and {target_column} is {p_value_mwu}")
        else: # Normally distributed --> t-test 
            t_stat, p_value_t = ttest_ind(g1, g2, nan_policy='omit')
            print(f"p-value for t-test between {col} and {target_column} is {p_value_t}")

    
    


