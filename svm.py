# Support Vector Machine
# Assumes linear decision boundary i.e., assumes linear separability 
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score



# Training - Evaluation
# Find decision boundary that maximises distance of datapoints from the line. 
# Hyperparameters to tune: 
    # Kernel
    # C

# Worth doing holdout to determine whether SVM converges to a solution i.e., has good metrics (accuracy, F1, etc.) with different normalisations. 

# 1. SVM with all training data
def run_svm(train_df, test_df):

    clf = svm.SVC()

    # remove imdb column before training
    train_df_labels = train_df['imdb_score_binned']
    train_df_features = train_df.drop('imdb_score_binned', axis=1)

    # clf.fit(train_df_features, train_df_labels)

    # cross validation split
    # scores = cross_val_score(clf, train_df_features, train_df_labels, cv=10)
    # print(scores.mean())
    C = 1.0  # SVM regularization parameter


    models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C, max_iter=10000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))

    # models = (clf.fit(train_df_features, train_df_labels) for clf in models)

    for clf in models:
        print("here")
        scores = cross_val_score(clf, train_df_features, train_df_labels, cv=10)
        print("score for model = ", scores.mean())


    
    # # defining parameter range 
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],  
    #             'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    #             'kernel': ['rbf']}  
    
    # grid = GridSearchCV(clf, param_grid, refit = True, verbose = 3) 
    
    # # fitting the model for grid search 
    # grid.fit(train_df_features, train_df_labels) 

# Test

def main():
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    run_svm(train_df_minmax, test_df_minmax)
    


if __name__ == '__main__':
    main()