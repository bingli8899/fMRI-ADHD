# Scripts to save utility function 
from types import SimpleNamespace
from src.model.GCN import GCN_Model 

# Some helpful mappings/constants
name_to_model = {
    "GCN_Model": GCN_Model
}

# Alabama Parenting Questionnaire: https://www.youthcoalition.net/wp-content/uploads/2022/06/APQ.pdf
# Strengths and Difficulties: https://acamh.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-7610.1997.tb01545.x
# To be consistent with connectome, normalize the values to be between 0 and 1
normalizing_factors = {
    "APQ_P_APQ_P_CP": 3 * 5,
    "APQ_P_APQ_P_PP": 6 * 5,
    "APQ_P_APQ_P_PM": 10 * 5,
    "APQ_P_APQ_P_OPD": 7 * 5,
    "APQ_P_APQ_P_INV": 10 * 5,
    "APQ_P_APQ_P_ID": 6 * 5,
    "SDQ_SDQ_Hyperactivity": 10.0,
    "SDQ_SDQ_Peer_Problems": 10.0,
    "SDQ_SDQ_Conduct_Problems": 10.0,
    "SDQ_SDQ_Emotional_Problems": 10.0,
    "SDQ_SDQ_Difficulties_Total": 40.0, # Sum of hyperactivity, peer problems, conduct problems, emotional_problems
    "SDQ_SDQ_Prosocial": 10.0,
    "SDQ_SDQ_Externalizing": 20.0, # Sum of conduct + hyperactivity (https://www.sdqinfo.org/a0.html)
    "SDQ_SDQ_Generating_Impact": 10.0,
    "SDQ_SDQ_Internalizing": 20.0, # Sum of emotional + peer symptoms (https://www.sdqinfo.org/a0.html)
}

# Convert dictionary to a nested object
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d
    
# Convert SimpleNamespace to a dictionary
def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(i) for i in obj]
    return obj

def cal_missing_percentage(df, col): 
    """
    Calculate missing percentage for each column in a df 
        -> print if missing percentage for a particular column > 0 
        -> return the missing_percentage, col name if missing_percentage > 0 
        -> If missing percentage = 0, return 0, None
    """
    missing = df[col].isnull().sum()
    missing_percentage = (missing / len(df[col])) * 100 
    
    if missing_percentage > 0: # No return is there is no missing value in [col] 
        print(f"col {col} has {missing_percentage} % missing.")
        return missing_percentage, col # return col if there is missing value in [col]
    return 0, None
    
def return_missing_list(df, df_name): 
    """
    Return a list with column name if there is some values missing in the column
        -> If the column has no missing, output None
    """
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

def count_levels_for_columns(col_list, df, printing=True):   
    """
    Count how many unique (categorical) levels for all columns in a df 
    """
    num_lst = [] 
    for col in col_list: 
        s = set(df[col])
        num = len(s)
        num_lst.append(num) 
        
        if printing: 
            print(f"There are {num} levels in {col}, including:")
            print(s) 
    return num_lst  

def check_connect_in_summary_stats(summary_df, stats_type): 
    """
    Find high and low connectivity among all columns in fmri data 
    """
    low = summary_df.loc[stats_type].sort_values().head(10) 
    high = summary_df.loc[stats_type].sort_values(ascending=False).head(10) 
    print(f"Columns with highest connectivity {stats_type} values: ", high)
    print(f"Columns with lowest connectivity {stats_type} values: ", low)
    

def check_columns_set(train_df, test_df): 
    """
    Check if the columns in train_cate and test_cate has the same levels of categories 
        -> Conclusion: They don't :-( See KNN_reasoning.jupyternote
    """
    test_df_cols = test_df.columns
    for col in train_df.columns[1:]: # skip paticipant id
        
        if col in test_df_cols: 
            unique_set_in_test_df = set(test_df[col])
            unique_set_in_train_df = set(train_df[col])
            
            in_test_not_in_train = unique_set_in_test_df - unique_set_in_train_df
            in_train_not_in_test = unique_set_in_train_df - unique_set_in_test_df
            
            if in_test_not_in_train != set(): 
                print(f"For {col}, below are in test but not in train:\n")
                print(in_test_not_in_train)
            if in_train_not_in_test != set():
                print(f"For {col}, below are in train but not in test:\n")
                print(in_train_not_in_test)
            

    
    


