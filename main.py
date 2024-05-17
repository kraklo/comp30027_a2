from feature_selection import *
from data_preprocessing import *
from knn import run_knn
from svm import run_svm
from decision_forest import run_random_forest, run_decision_tree
from neural_network import run_neural_network, neural_net
from stacking import run_stacking


def main():
    print("Preprocessing data...")
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    train_labels = train_df_minmax['imdb_score_binned']
    train_minmax = train_df_minmax.drop('imdb_score_binned', axis=1)

    print("Running correlation...")
    train_df_lin, test_df_lin = lin_correlation(train_df_minmax, test_df_minmax)

    print("Running knn...")
    knn_result = run_knn(train_minmax, train_labels, test_df_minmax)

    print("Running svm...")
    svm_result = run_svm(train_minmax, train_labels, test_df_minmax)

    print("Running decision tree...")
    decision_tree_result = run_decision_tree(train_df_minmax, test_df_minmax)

    print("Running random forest...")
    decision_forest_result = run_random_forest(train_df_lin, train_labels, test_df_lin)

    print("Training neural network...")
    model, device = neural_net(train_df_std)

    print("Running neural network...")
    neural_network_result = run_neural_network(model, device, test_df_std)

    print("Running stacking...")
    stacking_result = run_stacking(knn_result, svm_result, decision_tree_result, decision_forest_result, neural_network_result)


if __name__ == '__main__':
    main()
