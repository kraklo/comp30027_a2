import pandas as pd


def run_stacking(knn_result, svm_result, decision_forest_result, neural_network_result):
    test_df = knn_result.copy()

    test_df['mean_label'] = (knn_result['imdb_score_binned'] +
                             svm_result['imdb_score_binned'] +
                             decision_forest_result['imdb_score_binned'] +
                             neural_network_result['imdb_score_binned'])

    test_df['imdb_score_binned'] = test_df['mean_label'].apply(lambda x: mean_label(x))

    test_df.to_csv('CSVs/stacking.csv', columns=['id', 'imdb_score_binned'], index_label=False)

    return test_df


def mean_label(label_sum):
    mean = label_sum / 4

    if mean % 1 < 0.5:
        return int(mean)
    else:
        return int(mean) + 1


def main():
    knn_result = pd.read_csv('CSVs/knn.csv')
    svm_result = pd.read_csv('CSVs/svm.csv')
    decision_forest_result = pd.read_csv('CSVs/decision_forest.csv')
    neural_network_result = pd.read_csv('CSVs/neural_network.csv')
    run_stacking(knn_result, svm_result, decision_forest_result, neural_network_result)


if __name__ == '__main__':
    main()
