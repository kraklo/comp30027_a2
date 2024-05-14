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
    train_df_features_minmax_corr, test_df_features_minmax_corr = lin_correlation(train_df_minmax, test_df_minmax)
    
    # b) Principal Component Analysis
    train_df_features_minmax_pca, test_df_features_minmax_pca = lin_correlation(train_df_minmax, test_df_minmax)

    # c) Decision Tree feature importances (maybe static?)
    test_df_predicted_

    # Dynamic Feature Selection: integrated into classifiers 

    # CLASSIFIERS
    # 1. SVM
    # 

    pass


if __name__ == '__main__':
    main()
