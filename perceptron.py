from sklearn.linear_model import Perceptron

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import *
from feature_selection import *


RANDOM_STATE = 123

def run_perceptron(features, labels, test_df):
    clf = Perceptron(random_state=RANDOM_STATE)
    clf.fit(features, labels)
    print(clf.score(features, labels))

    predictions = pd.DataFrame(clf.predict(test_df))
    predictions.columns = ['imdb_score_binned']
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

    # All of these just give me all 2s as predictions like SVM




    




if __name__ == '__main__':
    main()