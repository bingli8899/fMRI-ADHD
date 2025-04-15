# Documentation of weekly notes: 
# Meeting on March 13 

1. Yihe's model: CNN
    -no metadata 
    -Binary classification (AHDH or non-ADHD) 
    -3 layer CNN 
    -Adam optimizer 
    -Binary crosstropy 
    -MaxPooling after each layer (--) 
    -No cross validation (8:2) 
    -Trainning accuracy 71% and validation accuracy 67% 
    Final model idea: 
    -Improve CNN 

2. HongXi's model: PCA + MLP 
    -with metadata 
    -select top 50% PCA-acis --> 400 PC
    -PCA only on fmri 
    -Concatenate metadata to 400 PC axis and input into MLP 
    -Binary classification 
    -Random forest on binary classification: 
        sex: trainning accuracy 0.81 (no splitting)
        adhd: training accuracy 0.72 (no splitting)
    -XGBoost 
        sex: trainning accuracy 0.79 
        adhd: trainning accuracy 0.74 
    -MPL: 5-fold validation 4 layers 
        Binary classification: 
            -average across all cross-set: 
                sex: 0.69 
                adhd: 0.71 

3. Bing's Model: GNN
    -No metadata
    -Four class classification 
    -trainning accuracy based on 5-fold cross-validation: trainning (~95%) and test (~71%) 
    Improvement: 
    -Improve the graphs 


# Meeting on Feb 27: 
a. Visualize the dataset --> How each column in metadata respond to ADHD diagonosis and sex. Any strong correlation. Need a jupyter notebook to show that. 

b. Each one of us should take one model (timeline: Finished in two weeks). 

c. Talk about how to approach the project: 
    -Scripts follow folder structure
    -Every function/class has a dog string 
    -Time to run complicated scripts to be recorded (optional if too quick)

# Feb 17 to Feb 22: 

To-do list for Bing: 

a. finish KNN inputation and push to github --> Haven't yet down --> sorry 

b. organize github and invite other members 

c. add dog string to functions in utility 