import pandas as pd
from math import isnan
import numpy as np
import random


def run_stacking(knn_result, svm_result, decision_tree_result, decision_forest_result, neural_network_result):
    test_df = knn_result.copy()

    results = pd.concat([knn_result['imdb_score_binned'],
                         svm_result['imdb_score_binned'],
                         decision_forest_result['imdb_score_binned'],
                         decision_tree_result['imdb_score_binned'],
                         neural_network_result['imdb_score_binned']], axis=1)

    test_df['imdb_score_binned'] = results.mode(axis=1).apply(lambda x: pick_random(x), axis=1)

    test_df.to_csv('CSVs/stacking.csv', columns=['id', 'imdb_score_binned'], index=False)

    return test_df


def pick_random(nums):
    if isnan(nums[1]):
        return int(nums[0])

    return int(random.choice(nums))


def main():
    knn_result = pd.read_csv('CSVs/knn.csv')
    svm_result = pd.read_csv('CSVs/svm.csv')
    decision_forest_result = pd.read_csv('CSVs/random_forest.csv')
    decision_tree_result = pd.read_csv('CSVs/decision_tree.csv')
    neural_network_result = pd.read_csv('CSVs/neural_network.csv')
    run_stacking(knn_result, svm_result, decision_tree_result, decision_forest_result, neural_network_result)


if __name__ == '__main__':
    main()
