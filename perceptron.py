from sklearn.linear_model import Perceptron

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import *
from feature_selection import *


RANDOM_STATE = 123

def run_perceptron(features, labels, test_df):
    print("hello")
    clf = Perceptron(random_state=RANDOM_STATE)

    #filtering(features, labels, 20, 300, 20, clf)
    # uncomment if want kbest 
    selected_features_train, selected_features_test = select_kbest_features(100, f_classif, features, labels, test_df, clf)

    clf.fit(selected_features_train, labels)
    print(clf.score(selected_features_train, labels))

    predictions = pd.DataFrame(clf.predict(selected_features_test))
    predictions.columns = ['imdb_score_binned']
    print(predictions)
    test_df = pd.concat([test_df, predictions], axis=1)
    return test_df


def main():
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    train_df_labels = train_df_minmax['imdb_score_binned']
    train_df_features = train_df_minmax.drop('imdb_score_binned', axis=1)
    
    # test_df_predicted = run_perceptron(train_df_features, train_df_labels, test_df_minmax)
    # print(test_df_predicted)
    # test_df_predicted.to_csv('p_no_feature_selection.csv', columns=['id', 'imdb_score_binned'], index=False)

    # train_df_lin, test_df_lin = lin_correlation(train_df_minmax, test_df_minmax)
    # pred_lin = run_perceptron(train_df_lin, train_df_labels, test_df_lin)
    # print(pred_lin['imdb_score_binned'].nunique())
    # pred_lin.to_csv('p_corr.csv', columns=['id', 'imdb_score_binned'], index=False)

    # train_df_pca, test_df_pca = pca(train_df_minmax, test_df_minmax)
    # pred_pca = run_perceptron(train_df_pca, train_df_labels, test_df_pca)
    # print(train_df_pca)
    # pred_pca.to_csv('p_pca.csv', columns=['id', 'imdb_score_binned'], index=False)

    

    # All of the above just give me all 2s as predictions like SVM??

    # test_df_predicted = run_perceptron(train_df_features, train_df_labels, test_df_minmax)
    # print(test_df_predicted)
    # test_df_predicted.to_csv('p_kbest.csv', columns=['id', 'imdb_score_binned'], index=False)
    # Gives 64% accuracy on kaggle



    




if __name__ == '__main__':
    main()