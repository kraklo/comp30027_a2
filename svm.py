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

RBF = 1
POLY = 0


# Training - Evaluation
# Find decision boundary that maximises distance of datapoints from the line. 
# Hyperparameters to tune: 
    # Kernel
    # C

def tune_params(features, labels):

    clf = svm.SVC()
    # defining parameter range - get a good range without waiting forever for the tuning 
    param_grid1 = {'C': [0.1, 1, 10],  
                'gamma': [1, 0.1, 0.01], 
                'kernel': ['rbf']
                }  
    
    # param_grid1 = {'C': [0.1, 1],
    #                'gamma': [1, 0.1]
    #             }  
    
    param_grid2 = {'C': [0.1, 1, 10],  
                'kernel': ['poly'],
                'degree': [3, 4, 5],
                }  
    
    grid1 = GridSearchCV(clf, param_grid1, refit = True)
    grid2 = GridSearchCV(clf, param_grid2, refit = True)
    
    #fitting the model for grid search 
    grid1.fit(features, labels)
    # print("Best parameters for SVC with rbf kernel: ", grid1.best_params_)

    grid2.fit(features, labels)
    # print("Best parameters for SVC with polynomial kernel: ", grid2.best_params_)

    grid1_result = grid1.cv_results_
    grid2_result = grid2.cv_results_

    # Plot param tuning results on heatmap 
    mean_scores = grid1_result['mean_test_score'].reshape(len(param_grid1['C']), len(param_grid1['gamma']))
    plt.imshow(mean_scores)
    plt.title("GridSearchCV Accuracy Scores for RBF Kernel")
    plt.xlabel("C")
    plt.ylabel("gamma")
    plt.show()

    mean_scores_2 = grid2_result['mean_test_score'].reshape(len(param_grid2['C']), len(param_grid2['degree']))
    plt.imshow(mean_scores_2)
    plt.title("GridSearchCV Accuracy Scores for Polynomial Kernel")
    plt.xlabel("C")
    plt.ylabel("degree")
    plt.show()
    

    if (grid1.best_score_ > grid2.best_score_):
        return RBF, grid1.best_params_
    
    else:
        return POLY, grid2.best_params_


# 1. SVM with all training data
def run_svm(train_df_features, train_df_labels, test_df):

    # First, tune parameters 
    best_params_kernel, best_params = tune_params(train_df_features, train_df_labels)
    if (best_params_kernel == RBF):
        clf = svm.SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
    else: #== poly
        clf = svm.SVC(C=best_params['C'], degree=best_params['degree'], kernel=best_params['kernel'])

    clf.fit(train_df_features, train_df_labels)

    predictions = pd.DataFrame(clf.predict(test_df))
    predictions.columns = ['imdb_score_binned']

    test_df = pd.concat([test_df.copy(), predictions], axis=1)
    test_df.to_csv('CSVs/svm.csv', columns=['id', 'imdb_score_binned'], index=False)

    return test_df


def support_vector_machine(train_df_features_minmax, train_df_labels_minmax, test_df_minmax, train_df_features_std, train_df_labels_std, test_df_std, train_df_features_minmax_corr, test_df_minmax_corr):
        # 1. SVM
    
    # 1a) MinMax-Scaled with Polynomial/RBF kernel, no feature selection
    # Fit model based on parameters and predict test labels 
    test_df_predicted_minmax = run_svm(train_df_features_minmax, train_df_labels_minmax, test_df_minmax)

    # Produce CSV files of predictions for accuracy testing on Kaggle
    test_df_predicted_minmax.to_csv('svm_minmax_no_feature_selection.csv', columns=['id', 'imdb_score_binned'], index=False)

    # 1b) Standardised with Polynomial/RBF kernel, no feature selection
    # Fit model based on parameters and predict test labels 
    test_df_predicted_std = run_svm(train_df_features_std, train_df_labels_std, test_df_std)

    # Produce CSV files of predictions for accuracy testing on Kaggle
    test_df_predicted_std.to_csv('svm_std_no_feature_selection.csv', columns=['id', 'imdb_score_binned'], index=False)

    # # 1c) MinMax with Polynomial/RBF kernel, feature selection = Pearson Correlation
    # Fit model based on parameters and predict test labels 
    test_df_predicted_minmax_corr = run_svm(train_df_features_minmax_corr, train_df_labels_minmax, test_df_minmax_corr)

    # Produce CSV files of predictions for accuracy testing on Kaggle
    test_df_predicted_minmax_corr.to_csv('svm_minmax_corr.csv', columns=['id', 'imdb_score_binned'], index=False)


# For file testing: 
def main():
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    train_df_labels = train_df_minmax['imdb_score_binned']
    train_df_features = train_df_minmax.drop('imdb_score_binned', axis=1)

    test_df_predicted, test_df_predicted2 = run_svm(train_df_features, train_df_labels, test_df_minmax)
    test_df_predicted.to_csv('svm1.csv', columns=['id', 'imdb_score_binned'], index=False)
    test_df_predicted2.to_csv('svm2.csv', columns=['id', 'imdb_score_binned'], index=False)

    # test_df_predicted3, test_df_predicted4 = run_svm(train_df_std, test_df_std)
    # test_df_predicted3.to_csv('svm3.csv', columns=['id', 'imdb_score_binned'], index=False)
    # test_df_predicted4.to_csv('svm4.csv', columns=['id', 'imdb_score_binned'], index=False)

    # train_df_lin, test_df_lin = lin_correlation(train_df_minmax, test_df_minmax)
    # pred_lin1, pred_lin2 = run_svm(train_df_lin, train_df_labels, test_df_lin)
    # print(pred_lin1['imdb_score_binned'].nunique())
    # print(pred_lin2['imdb_score_binned'].nunique())
    # pred_lin1.to_csv("svm_corr1.csv", columns=['id', 'imdb_score_binned'], index=False)
    # pred_lin2.to_csv("svm_corr2.csv", columns=['id', 'imdb_score_binned'], index=False)

    # THEY ARE ALL IDENTICAL????? always predict label=2


if __name__ == '__main__':
    main()
