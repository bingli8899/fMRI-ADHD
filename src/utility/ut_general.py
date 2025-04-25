# Scripts to save utility function 
from types import SimpleNamespace 
from src.model.GCN import GCN_Model 
from src.model.DirGNNConv import DirGNN_model
from src.model.NNconv import NNConv_model 
from src.model.GATv2 import GATv2Conv_Model
from src.model.TransConv import TransformerConv_Model
from src.model.DirGNN_GatConv import DirGNN_GatConv_model 
from src.model.SageGNN import SageGNN_model
# from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
# from torch.nn import Linear, Module, ReLU, LayerNorm, BatchNorm1d

# Assign name to model name
name_to_model = {
    "GCN_model": GCN_Model,
    "DirGNN_model": DirGNN_model,
    "NNConv_model": NNConv_model,
    "GATv2_model": GATv2Conv_Model,
    "TransConv_model": TransformerConv_Model,
    "DirGNN_GatConv_model": DirGNN_GatConv_model,
    "SageGNN_model": SageGNN_model,
}


def assign_label(row):
    """Assign new label based on ADHD and sex"""
    if row["ADHD_Outcome"] == 1 and row["Sex_F"] == 1:  # Female ADHD
        return "0"
    elif row["ADHD_Outcome"] == 0 and row["Sex_F"] == 1:  # Female non-ADHD
        return "1"
    elif row["ADHD_Outcome"] == 1 and row["Sex_F"] == 0:  # Male ADHD
        return "2"
    elif row["ADHD_Outcome"] == 0 and row["Sex_F"] == 0:  # Male non-ADHD
        return "3"
    
def recover_original_label(num):
    """Go back to original binary labels from 4 class label"""
    num = int(num)
    if num == 0:
        return (1, 1)
    elif num == 1:
        return (0, 1)
    elif num == 2:
        return (1, 0)
    elif num == 3:
        return (0, 0)
    else:
        raise ValueError("The label should be a digit between 0 and 3 inclusive")


def relabel_train_outcome(train_outcome, remove_previous_label = True, task = "four"): 
    """
    Assigns ADHD labels and optionally removes the previous ADHD_Outcome and Sex_F columns.
    Args: 
    - train_outcome (pd.DataFrame): DataFrame containing "ADHD_Outcome" and "Sex_F" columns.
    - remove_previous_label (bool): If True, removes "ADHD_Outcome" and "Sex_F" columns. 
    Returns:
    - pd.DataFrame: Updated DataFrame with "ADHD_Label" and optionally without original columns.
    """

    if task.lower() == "four":  
        train_outcome["Label"] = train_outcome.apply(assign_label, axis=1)
        
    if task.lower() == "adhd": 
        train_outcome["Label"] = train_outcome["ADHD_Outcome"]

    if task.lower() == "sex": 
        train_outcome["Label"] = train_outcome["Sex_F"]

    if remove_previous_label: 
        train_outcome = train_outcome.drop(columns = ["ADHD_Outcome", "Sex_F"])
    elif not remove_previous_label: 
        print("Will not remove columns \"ADHD_Outcome\" and \"Sex_F\"")
    else: 
        raise ValueError("Hi you got the run emove_previous_label. True or False, plz")
        
    return train_outcome 

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


def write_model_grid(model_grid, file_handle):
    """A function used in ML.py to write """
    file_handle.write("Model Grid Configuration:\n")
    for model_name, config in model_grid.items():
        file_handle.write(f"Model: {model_name}\n")
        file_handle.write("Parameter grid:\n")
        for param, values in config["param_grid"].items():
            file_handle.write(f"  {param}: {values}\n")
        file_handle.write("--------------------------------------------------\n")
    file_handle.write("\n")

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
            

    
    


