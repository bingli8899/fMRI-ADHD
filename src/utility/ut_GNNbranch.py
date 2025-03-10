#  Utility function used in GNN branch, will need to be merged to Main later. 

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


def relabel_train_outcome(train_outcome, remove_previous_label = True): 
    """
    Assigns ADHD labels and optionally removes the previous ADHD_Outcome and Sex_F columns.
    Args: 
    - train_outcome (pd.DataFrame): DataFrame containing "ADHD_Outcome" and "Sex_F" columns.
    - remove_previous_label (bool): If True, removes "ADHD_Outcome" and "Sex_F" columns. 
    Returns:
    - pd.DataFrame: Updated DataFrame with "ADHD_Label" and optionally without original columns.
    """

    train_outcome["Label"] = train_outcome.apply(assign_label, axis=1)

    if remove_previous_label: 
        train_outcome = train_outcome.drop(columns = ["ADHD_Outcome", "Sex_F"])
    elif not remove_previous_label: 
        print("Will not remove columns \"ADHD_Outcome\" and \"Sex_F\"")
    else: 
        raise ValueError("Hi you got the run emove_previous_label. True or False, plz")
        
    return train_outcome