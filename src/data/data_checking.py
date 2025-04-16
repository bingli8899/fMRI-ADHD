# This script is used to double check labeling, ordering, etc for test and train dataset 
# This script allows us to mannually check the original dataset 
# To make sure our participant_ids correspond with what we are supposed to do

import os
import sys
import argparse

rootfolder = "/u/b/i/bingl/private/fMRI-AHDH"
sys.path.append(os.path.join(rootfolder))

from src.data.data_loader import load_or_cache_data

datafolder = os.path.join(rootfolder, "data")
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

def main(agrs, df_dic): 

    pid = args.pid.strip()
    col = args.col.strip() if agrs.col is not None else None
    train = args.train

    if train:
        df_fmri = df_dic["train_fmri"]
        df_outcome = df_dic["train_outcome"]
        df_quant = df_dic["train_quant"]
        df_cate = df_dic["train_cate"]
        split = "train"
    else:
        df_fmri = df_dic["test_fmri"]
        df_quant = df_dic["test_quant"]
        df_cate = df_dic["test_cate"]
        split = "test"

    print(f"Checking participant {pid} in {split} dataset")

    if col is not None:
        if col in df_fmri.columns:
            print(f"fMRI value for {col}:")
            print(df_fmri.loc[df_fmri["participant_id"] == pid, col])
        if col in df_quant.columns: 
            print(f"quant data for {col}:")
            print(df_quant.loc[df_quant["participant_id"] == pid, col])
        if col in df_cate.columns: 
            print(f"categorical data for {col}:")
            print(df_cate.loc[df_cate["participant_id"] == pid, col])

    else:
        print("\nAll fMRI values for this participant:")
        print(df_fmri.loc[df_fmri["participant_id"] == pid])

        print("\nCategorical data:")
        print(df_cate.loc[df_cate["participant_id"] == pid])

        print("\nQuantitative data:")
        print(df_quant.loc[df_quant["participant_id"] == pid])

    if split == "train": 
        print(df_outcome.loc[df_outcome["participant_id"] == pid])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=str, required=True, help="Participant ID to check")
    parser.add_argument("--col", type=str, default=None, help="fMRI/cate/test column name to check")
    parser.add_argument("--train", action="store_true", help="Use train dataset")
    parser.add_argument("--test", action="store_true", help="Use test dataset")
    args = parser.parse_args()

    main(args, df_dic)

