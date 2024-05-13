# Decision Tree extension
# choose features based on importance scores from unselected training data?

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RandomizedSearchCV

RANDOM_STATE = 123

def run_decision_tree(train_df, test_df):
    clf = DecisionTreeClassifier(criterion='entropy')
    # clf1 = DecisionTreeClassifier(criterion='log_loss')
    # clf2 = DecisionTreeClassifier(criterion='gini')
    # clf3 = DecisionTreeClassifier()

    # remove imdb column before training
    train_df_labels = train_df['imdb_score_binned']
    train_df_features = train_df.drop('imdb_score_binned', axis=1)
    score = cross_val_score(clf, train_df_features, train_df_labels, cv=5)
    print(score.mean()) # highest accuracy with training data 
    # score = cross_val_score(clf1, train_df_features, train_df_labels, cv=5)
    # print(score.mean())
    # score = cross_val_score(clf2, train_df_features, train_df_labels, cv=5)
    # print(score.mean())
    # score = cross_val_score(clf3, train_df_features, train_df_labels, cv=5)
    # print(score.mean())
    clf.fit(train_df_features, train_df_labels)
    predictions = clf.predict(test_df)
    test_df['imdb_score_binned'] = predictions.tolist()
    return test_df


def run_random_forest(train_df):
    clf = RandomForestClassifier(criterion='entropy')
    # clf1 = RandomForestClassifier(criterion='log_loss')
    # clf2 = RandomForestClassifier(criterion='gini')
    # clf3 = RandomForestClassifier()

    train_df_labels = train_df['imdb_score_binned']
    train_df_features = train_df.drop('imdb_score_binned', axis=1)
    score = cross_val_score(clf, train_df_features, train_df_labels, cv=5)
    print(score.mean()) # highest accuracy with training data 
    # score = cross_val_score(clf1, train_df_features, train_df_labels, cv=5)
    # print(score.mean())
    # score = cross_val_score(clf2, train_df_features, train_df_labels, cv=5)
    # print(score.mean())
    # score = cross_val_score(clf3, train_df_features, train_df_labels, cv=5)
    # print(score.mean())

    # clf.fit(train_df_features, train_df_labels)
    # predictions = clf.predict(test_df)
    # test_df['imdb_score_binned'] = predictions.tolist()
    # return test_df

# from https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html
def graph_rf(train_df):

    train_df_labels = train_df['imdb_score_binned']
    train_df_features = train_df.drop('imdb_score_binned', axis=1)

    ensemble_clfs = [
        (
            "RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(
                warm_start=True,
                oob_score=True,
                max_features="sqrt",
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(
                warm_start=True,
                max_features="log2",
                oob_score=True,
                random_state=RANDOM_STATE,
            ),
        ),
        # (
        #     "RandomForestClassifier, max_features=None",
        #     RandomForestClassifier(
        #         warm_start=True,
        #         max_features=None,
        #         oob_score=True,
        #         random_state=RANDOM_STATE,
        #     ),
        # ),
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 20
    max_estimators = 1000

    for label, clf in ensemble_clfs:
        print("here")
        for i in range(min_estimators, max_estimators + 1, 10):
            print("here %d", i)
            clf.set_params(n_estimators=i)
            clf.fit(train_df_features, train_df_labels)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()



def main():
    
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    #print(train_df_minmax)
    # test_df_predicted = run_decision_tree(train_df_minmax, test_df_minmax)
    # print(test_df_predicted[['id', 'imdb_score_binned']])
    # test_df_predicted.to_csv('out.csv', columns=['id', 'imdb_score_binned'], index=False)
    #run_random_forest(train_df_minmax)
    graph_rf(train_df_minmax)
    


if __name__ == '__main__':
    main()