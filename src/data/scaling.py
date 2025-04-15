import numpy as np 

class MeanStdScaler(): 
    """
        Step 1: Normalize r by using Fisher-Z 
        Step 2: Scale the test/train dataset: 
            X' = ((X - test_mean) / test_std ) * train_std + train_mean 
    """
    def __init__(self): 
        self.train_mean = None
        self.train_std = None

    def fisherZ_normalization(self, df): 
        df = np.arctanh(df)
        return df 

    def fit(self, df): 
        fisherZ_df = self.fisherZ_normalization(df)
        self.train_mean = fisherZ_df.mean() 
        self.train_std = fisherZ_df.std() 
        return self 

    def transform(self, df): 

        if self.train_mean is None or self.train_std is None: 
            raise ValueError("WHAT? You didn't fit the train_data first") 
        
        fisherZ_df = self.fisherZ_normalization(df)
        test_mean = fisherZ_df.mean()
        test_std = fisherZ_df.std() 

        df_scaled = ((fisherZ_df - test_mean)/test_std) * self.train_std + self.train_mean 
        
        return df_scaled 
    

        
         