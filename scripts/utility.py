# Scripts to save utility function 
import os 
import pandas as pd
from scipy.stats import chi2_contingency # Test if columns with missing data is correlated with any columns 

def missing_percentage(df, col): 
    missing = df[col].isnull().sum()
    missing_percentage = (missing / len(df[col])) * 100 
    if missing_percentage > 0: 
        print(f"col {col} has {missing_percentage} % missing.")
        
def extract_file_path(datapath): 
    file_lst = []
    for dirname, _, filenames in os.walk(datapath):
        for filename in filenames:
            file_path = os.path.join(dirname, filename) 
            file_lst.append(file_path)
            print(os.path.join(dirname, filename)) 
    return file_lst 

def chi2_with_one_columns(column_lst, target_column, df):
    for col in column_lst:
        contingency_table = pd.crosstab(df[target_column], df[col])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Test between {col} and {target_column}:")
        print(f"Chi-Square Stat: {chi2:.2f}, P-value: {p:.5f}")
    
    


