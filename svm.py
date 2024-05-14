# Support Vector Machine
# Assumes linear decision boundary i.e., assumes linear separability 
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import *
from feature_selection import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score



# Training - Evaluation
# Find decision boundary that maximises distance of datapoints from the line. 
# Hyperparameters to tune: 
    # Kernel
    # C

# Worth doing holdout to determine whether SVM converges to a solution i.e., has good metrics (accuracy, F1, etc.) with different normalisations. 

# 1. SVM with all training data
def run_svm(train_df_features, train_df_labels, test_df):

    clf = svm.SVC()

    # remove imdb column before training
    # train_df_labels = train_df['imdb_score_binned']
    # train_df_features = train_df.drop('imdb_score_binned', axis=1)

    # clf.fit(train_df_features, train_df_labels)

    # cross validation split
    # scores = cross_val_score(clf, train_df_features, train_df_labels, cv=10)
    # print(scores.mean())
    C = 1.0  # SVM regularization parameter


    # models = (svm.SVC(kernel='linear', C=C),
    #       svm.LinearSVC(C=C, max_iter=10000),
    #       svm.SVC(kernel='rbf', gamma=0.7, C=C),
    #       svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))

    # models = (clf.fit(train_df_features, train_df_labels) for clf in models)
    # scores = cross_val_score(clf, train_df_features, train_df_labels, cv=10)
    # print(scores)

    # for clf in models:
    #     print("here")
    #     scores = cross_val_score(clf, train_df_features, train_df_labels, cv=2)
    #     print("score for model = ", scores.mean())


    
    # defining parameter range 
    param_grid1 = {'C': [0.1, 1, 10, 100, 1000],  
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                'kernel': ['rbf']
                }  
    
    param_grid2 = {'C': [0.1, 1, 10, 100],  
                'kernel': ['poly'],
                'degree': [3, 4, 5],
                }  
    
    #grid1 = GridSearchCV(clf, param_grid1, refit = True, verbose = 3) 
    #grid2 = GridSearchCV(clf, param_grid2, refit = True, verbose = 3) 
    
    # fitting the model for grid search 
    #grid1.fit(train_df_features, train_df_labels) 
    #print("Best parameters for SVC with rbf kernel: ", grid1.best_estimator_)

    # grid2.fit(train_df_features, train_df_labels)
    # print("Best parameters for SVC with polynomial kernel: ", grid2.best_estimator_)

    # best: c = 0.1
    clf = svm.SVC(C=0.1, kernel='poly', degree=3) # same accuracy for all degrees so chose lowest for computational benefit 
    clf2 = svm.SVC(C=0.1, gamma=1, kernel='rbf')

    test_df_copy = test_df.copy()

    clf.fit(train_df_features, train_df_labels)
    clf2.fit(train_df_features, train_df_labels)

    predictions = clf.predict(test_df)
    predictions2 = clf2.predict(test_df_copy)



    print(test_df)
    test_df['imdb_score_binned'] = predictions.tolist()
    test_df_copy['imdb_score_binned'] = predictions2.tolist()

    return test_df, test_df_copy


    # plot results
    results_df = pd.DataFrame(grid2.cv_results_)
    results_df.plot()
    # results_df = pd.DataFrame(grid2.cv_results_).sort_values(by='rank_test_score')
    print(results_df)

# Test

def main():
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    train_df_labels = train_df_minmax['imdb_score_binned']
    train_df_features = train_df_minmax.drop('imdb_score_binned', axis=1)

    # test_df_predicted, test_df_predicted2 = run_svm(train_df_features, train_df_labels, test_df_minmax)
    # test_df_predicted.to_csv('svm1.csv', columns=['id', 'imdb_score_binned'], index=False)
    # test_df_predicted2.to_csv('svm2.csv', columns=['id', 'imdb_score_binned'], index=False)

    # test_df_predicted3, test_df_predicted4 = run_svm(train_df_std, test_df_std)
    # test_df_predicted3.to_csv('svm3.csv', columns=['id', 'imdb_score_binned'], index=False)
    # test_df_predicted4.to_csv('svm4.csv', columns=['id', 'imdb_score_binned'], index=False)

    # train_df_lin, test_df_lin = lin_correlation(train_df_minmax, test_df_minmax)
    # pred_lin1, pred_lin2 = run_svm(train_df_lin, train_df_labels, test_df_lin)
    # print(pred_lin1['imdb_score_binned'].nunique())
    # print(pred_lin2['imdb_score_binned'].nunique())
    # pred_lin1.to_csv("svm_corr1.csv", columns=['id', 'imdb_score_binned'], index=False)
    # pred_lin2.to_csv("svm_corr2.csv", columns=['id', 'imdb_score_binned'], index=False)

    # THEY ARE ALL IDENTICAL????? always predict 2


if __name__ == '__main__':
    main()