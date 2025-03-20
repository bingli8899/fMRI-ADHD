from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
import pandas as pd

class KNNImputer_with_OneHotEncoding:
    
    def __init__(self, merge_fmri=False, k=5):
        """
        Goal 1: 
        Convert all categorical variables to one-hot encoding.
        Issues: 
            The dataset does contain ordinal data. 
            Here, we encode all categorical variables with onehot encoding, which could be changed later. 

        Goal 2: Apply KNN imputation to dataset w/o fmri.
        - If merge_fmri == True: Merge fmri with the demographical data first, then apply KNN imputation.
        - If merge_fmri == False: Apply KNN imputation only to demographical data.
        """
        
        self.k = int(k) 
        self.merge_fmri = merge_fmri
        self.onehot_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self.imputer = KNNImputer(n_neighbors=self.k)

    def fit(self, df_dic): 
        """ Fit OneHotEncoder using all possible categories. 
        Issues: 
            -> Test set has categories unseen in train set in some columns 
            -> NaN values unseen in train dataset are filled with "Missing", which is created into a specific column
            -> Get all categories from both train and test datasets 
            -> Create columns for all categories during fitting 
        Issues to be resolved: 
            -> How about ordinal data? 
        """
        
        df_cate = df_dic["train_cate"].copy()
        df_cate_test = df_dic["test_cate"].copy()
        
    
        df_cate = df_cate.drop(columns=['Basic_Demos_Enroll_Year']) # Exclude the year variable because it is very different across train + test
        df_cate_test = df_cate_test.drop(columns=['Basic_Demos_Enroll_Year']) # Exclude the year variable because it is very different across train + test
        
        self.cate_var = df_cate.columns[1:]  # Exclude participant_id in the list with all categorical vairbales. 

        for col in self.cate_var: # Replace NaN values with a single "missing" category in training & test datasets
            df_dic["train_cate"][col] = df_dic["train_cate"][col].fillna(-1).astype(int).astype(str)
            df_dic["test_cate"][col] = df_dic["test_cate"][col].fillna(-1).astype(int).astype(str)

        all_categories = [sorted(set(df_dic["train_cate"][col]).union(set(df_dic["test_cate"][col]))) for col in self.cate_var]
        self.onehot_encoder = OneHotEncoder(
            categories=all_categories, # onehot for all possible categories in train and test 
            handle_unknown="ignore", 
            sparse_output=False
        )

        self.onehot_encoder.fit(df_dic["train_cate"][self.cate_var])
        self.onehot_encoder.fit(df_dic["test_cate"][self.cate_var])  
        return self


    def transform(self, df_dic, split="train"): 
        """ 
        Apply One-Hot Encoding and KNN imputation on test data. 
        """
        merged_df = self.merge_data(df_dic, split=split)
        df_transformed = merged_df.copy()

        for col in self.cate_var: # replaces NaN with "Missing" before encoding
            df_transformed[col] = df_transformed[col].astype(str).fillna("Missing")

        df_transformed = df_transformed.drop(columns=['Basic_Demos_Enroll_Year']) # Exclude the year variable because it is very different across train + test
        
        encoded_data = self.onehot_encoder.transform(df_transformed[self.cate_var]) # encoding 
        encoded_df = pd.DataFrame(
            encoded_data, 
            columns=self.onehot_encoder.get_feature_names_out(self.cate_var),
            index=df_transformed.index
        )

        df_transformed = df_transformed.drop(columns=self.cate_var) # drop the original categorical columns
        df_transformed = df_transformed.join(encoded_df) # add encoded columns 

        data_imputed = self.imputer.fit_transform(df_transformed) # impute
        final_df = pd.DataFrame(data_imputed, 
                                columns=df_transformed.columns, 
                                index=df_transformed.index)

        return final_df

        
    def merge_data(self, df_dic, split = "train", merge_fmri = False): # merge categorical, quantatitive, and fmri dataset 
        """
        Merge quant data and cate data: 
            -> If merge_fmri = True, merge quantatitive data, categorical data, and fmri 
            -> If not, merge quantatitive and categorical data only
        """
        # Need to make deep copy to make sure the original dataset is not over-written 
        df_cate_copy = df_dic[f"{split}_cate"].copy()
        df_quant_copy = df_dic[f"{split}_quant"].copy()
        df_fmri_copy = df_dic[f"{split}_fmri"].copy()
        
        participant_id = df_cate_copy.columns[0]

        df_cate_copy.set_index(participant_id, inplace = True, drop = False)
        df_quant_copy.set_index(participant_id, inplace = True, drop = False)
        
        if self.merge_fmri: 
            df_fmri_copy.set_index(participant_id, inplace=True, drop=False)

        for df in [df_cate_copy, df_quant_copy] if not self.merge_fmri else [df_cate_copy, df_quant_copy, df_fmri_copy]: 
            if participant_id not in df.columns: 
                raise ValueError("Ohh NO! The first column is not participant id")           
            if participant_id in df.columns: 
                df.drop(columns = [participant_id], inplace = True)
        
        if self.merge_fmri: 
            df = df_cate_copy.join([df_quant_copy, df_fmri_copy], how = "inner")
        elif self.merge_fmri == False: 
            df = df_cate_copy.join(df_quant_copy, how = "inner") 
        else: 
            raise ValueError("Ohh no! Seriously? U got the wrong merge_fmri!")

        return df 
    
    def fit_transform(self, df_dic, split="train"): 
        """ Fit and transform the dataset. """
        
        self.fit(df_dic)
        return self.transform(df_dic, split=split)



