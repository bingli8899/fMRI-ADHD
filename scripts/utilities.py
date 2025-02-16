# Scripts to save utility function 

def missing_percentage(df, col): 
    missing = df[col].isnull().sum()
    missing_percentage = (missing / len(df[col])) * 100 
    if missing_percentage > 0: 
        print(f"col {col} has {missing_percentage} % missing.")