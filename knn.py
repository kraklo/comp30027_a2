# from assignment 1

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import *
from feature_selection import *


RANDOM_STATE = 123

def run_knn(features, labels, test_df):
    k_neighbours = np.arange(2, 200, 2)
    scores = {}
    #https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7
    # for k in k_neighbours:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(features, labels)
    #     scores[k] = cross_val_score(knn, features, labels, cv=5).mean()
    
    # plt.plot(k_neighbours, scores.values(), label="Train Accuracy")
    # plt.xlabel("Number Of Neighbors")
    # plt.ylabel("Accuracy")
    # plt.title("KNN: Varying number of Neighbors")
    # plt.legend()
    # plt.grid()
    # plt.show()

    knn = KNeighborsClassifier(n_neighbors=106) # from graph
    #filtering(features, labels, 10, 300, 20, knn)
    # best = 30 features
    selected_features_train, selected_features_test = select_kbest_features(30, f_classif, features, labels, test_df, knn)

    score = cross_val_score(knn, features, labels, cv=5)
    print(score.mean())
    knn.fit(selected_features_train, labels)
    # print(clf.score(selected_features_train, labels))

    predictions = pd.DataFrame(knn.predict(selected_features_test))
    predictions.columns = ['imdb_score_binned']
    print(predictions)
    test_df = pd.concat([test_df, predictions], axis=1)
    return test_df



    #return test_df


def main():
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    train_df_labels = train_df_minmax['imdb_score_binned']
    train_df_features = train_df_minmax.drop('imdb_score_binned', axis=1)
    
    test_df_predicted = run_knn(train_df_features, train_df_labels, test_df_minmax)
    print(test_df_predicted)
    test_df_predicted.to_csv('knn_kbest.csv', columns=['id', 'imdb_score_binned'], index=False)




    




if __name__ == '__main__':
    main()