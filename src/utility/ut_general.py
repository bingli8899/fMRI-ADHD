# Scripts to save utility function 
import os 
import pandas as pd

def cal_missing_percentage(df, col): 
    
    missing = df[col].isnull().sum()
    missing_percentage = (missing / len(df[col])) * 100 
    
    if missing_percentage > 0: # No return is there is no missing value in [col] 
        print(f"col {col} has {missing_percentage} % missing.")
        return missing_percentage, col # return col if there is missing value in [col]
    return 0, None
    
def return_missing_list(df, df_name): 
    missing_lst = []
    
    for col in df.columns: 
        missing_percentage, missing_col = cal_missing_percentage(df, col)
        if missing_percentage > 0:  
            missing_lst.append(missing_col)
            
    if not missing_lst:  
        print(f"{df_name} has no missing value")  
        return []  # Explicitly return an empty list
    else:  
        return missing_lst
    
    


