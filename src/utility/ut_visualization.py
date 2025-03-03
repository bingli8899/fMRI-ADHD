import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram_with_randomly_sampled_N(train_fmri, test_fmri, N=100): # select N items from each column 
    """
    Plot the histogram with randomly sampled N (default: 100) from each column in train_fmri and test_fmri 
    N should be smaller than 304 since this is the max number of rows in test_fmri 
    """
    
    random_train_fmri = []
    random_test_fmri = []

    # Sample from each column --> Hopefully can make the process quicker :(
    for col in train_fmri.columns[1:]: # (excluding the participant ID if present) 
        random_train_fmri.extend(np.random.choice(train_fmri[col].values, size=N, replace=False))

    for col in test_fmri.columns[1:]: 
        random_test_fmri.extend(np.random.choice(test_fmri[col].values, size=N, replace=False))

    random_train_fmri = np.array(random_train_fmri)
    random_test_fmri = np.array(random_test_fmri)

    # Plot distributions between train and test 
    plt.figure(figsize=(10, 6))
    sns.histplot(random_train_fmri, bins=50, color="blue", label="Train Sample", kde=True, alpha=0.6)
    sns.histplot(random_test_fmri, bins=50, color="red", label="Test Sample", kde=True, alpha=0.6)
    plt.legend()
    plt.title(f"Distribution of Sampled fMRI Connectivity Values (N: {N})")
    plt.xlabel("Connectivity Strength")
    plt.ylabel("Frequency")
    plt.show()