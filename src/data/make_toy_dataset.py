# This script makes a toy dataset which is easier for us to test our codes: 

# Usage 
import pandas as pd
import os 

def make_toy_dataset(data_dic, data_type="train", n_subjects=30, n_regions=30, output_path=None):
    """
    Creates a toy fMRI dataset by selecting a subset of subjects and extracting specified columns.
    Parameters:
        data_dic: data dictionary --> could be train or test 
        type: 'train' or 'test', determines which dataset to use.
        n_subjects (int): Number of subjects to select.
        n_regions (int): Number of brain regions to consider.
        output_file (str): Name of the output CSV file.
    Returns:
        -> If type = "train", return dic{"fmri" = pd.dataframe, "label"}
        -> If type = "test", return dic{"fmri", "label" = None}
    """
    df_type = data_type.lower()

    if output_path is None:
        output_path = os.getcwd()
    
    if df_type == "train":
        df_fmri = data_dic["train_fmri"]
        df_label = data_dic["train_outcome"]
    elif df_type == "test":
        df_fmri = data_dic["test_fmri"]
        df_label = None
    else:
        raise ValueError("Invalid type. Choose 'train' or 'test'.")

    selected_participant_id = df_fmri["participant_id"].iloc[:n_subjects]
    selected_participants = df_fmri[df_fmri["participant_id"].isin(selected_participant_id)]
    region_lst = list(range(n_regions))

    col_lst = [] 
    for roi1 in region_lst: 
        for roi2 in region_lst: 
            if roi1 != roi2: 
                col_name = f"{roi1}throw_{roi2}thcolumn" 
                col_lst.append(col_name)

    selected_columns = [col for col in col_lst if col in selected_participants.columns]
    extracted_df = selected_participants[["participant_id"] + selected_columns]

    if df_label is not None:
        selected_label = df_label[df_label["participant_id"].isin(selected_participant_id)].copy() 
        selected_label["label"] = selected_label.apply(assign_label, axis=1)
        final_label_df = selected_label[["participant_id", "label"]]
        final_label_df.to_csv(os.path.join(output_path, "toy_label_train.csv"), index = False)
        extracted_df.to_csv(os.path.join(output_path, "toy_fmri_train.csv"), index=False)
        dic = {
            "train_fmri": extracted_df,
            "train_outcome": final_label_df
        }
    else: 
        extracted_df.to_csv(os.path.join(output_path, "toy_fmri_test.csv"), index=False) 
        dic = {
            "test_fmri": extracted_df,
            "test_outcome": None
        }

    return dic

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







        







