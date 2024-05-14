import pandas as pd
from data_preprocessing import *
from feature_selection import *

from svm import *
from knn import *
from perceptron import *
from decision_forest import *


def main():
    # Preprocess
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()

    # Separate features and labels
    train_df_labels_minmax = train_df_minmax['imdb_score_binned']
    train_df_features_minmax = train_df_minmax.drop('imdb_score_binned', axis=1)

    train_df_labels_std = train_df_std['imdb_score_binned']
    train_df_features_std = train_df_std.drop('imdb_score_binned', axis=1)

    # Static Feature Selection 
    # a) Pearson Correlation
    train_df_features_minmax_corr, test_df_minmax_corr = lin_correlation(train_df_minmax, test_df_minmax)
    
    # b) Principal Component Analysis
    train_df_features_minmax_pca, test_df_minmax_pca = lin_correlation(train_df_minmax, test_df_minmax)

    # c) Decision Tree feature importances (maybe static?)

    # Dynamic Feature Selection: integrated into classifiers

    # CLASSIFIERS: make each of these a single separate function in their own files to declog this file 
       # Print (training) metrics for each (F1, accuracy, precision, recall) for Results section of report. Choose best-performing according to these metrics to run on Kaggle and evaluate (test) accuracy.

    # 1. SVM
    support_vector_machine(train_df_features_minmax, train_df_labels_minmax, test_df_minmax, train_df_features_std, train_df_labels_std, test_df_std, train_df_features_minmax_corr, test_df_minmax_corr)

    # 2. Decision Tree
    # are b and c any different?
    # 2a) MinMax-Scaled, no feature selection
    # 2b) MinMax-Scaled, feature selection = Embedded, Feature Importances
    # 2c) Standardised, feature selection = Embedded, Feature Importances


    # 3. Random Forest
    # 3a) MinMax-Scaled, no feature selection
        # Plot OOB Error Rate for feature quantifier parameter
        # Fit model based on parameter value, predict test labels
        # Produce CSV file

    # 3b) Standardised, no feature selection 
    # 3c) MinMax-Scaled, feature selection = PCA
    # 3d) MinMax-Scaled, feature selection = Pearson Correlation

    # 4. K Nearest Neighbours
    # 4a) MinMax-Scaled, no feature selection
        # Calculate optimum k value
        # Fit model
        # Produce CSV
    # 4b) MinMax-Scaled, feature selection = Filtering 
    # 4c) Standardised, feature selection = Filtering 
    # 4d) Try to find best feature selection. Maybe combination of PCA/PC and KBest?

    # 5. Perceptron
    # 5a) MinMax-Scaled, no feature selection
        # Fit model
        # Produce CSV
    # 5b) Standardised, no feature selection
    # 5c) (whichever performs better), feature selection = Pearson Correlation
    # 5d) (whichever performs better), feature selection = Filtering 

    # 6. Stacking 
        # something is going very wrong here :(

if __name__ == '__main__':
    main()
