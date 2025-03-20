import numpy as np 

class MeanStdScaler(): 
    """
    This scales the test dataset (or train dataset) based on 
    X' = ((X - test_mean) / test_std ) * train_std + train_mean 
    """

    def __init__(self): 
        self.train_mean = None
        self.train_std = None

    def fit(self, df): 
        self.train_mean = df.mean() 
        self.train_std = df.std() 
        return self 

    def transform(self, df): 

        if self.train_mean is None or self.train_std is None: 
            raise ValueErro("WHAT? You didn't fit the train_data first") 
        
        test_mean = df.mean()
        test_std = df.std() 

        df_scaled = ((df - test_mean)/test_std) * self.train_std + self.train_mean 
        
        return df_scaled 



        
         