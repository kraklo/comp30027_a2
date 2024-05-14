# Feature selection methods: 
    # Wrappers, Embedded, Filtering
    # Necessary for kNN, useful for NB/DT, not necessary for SVM

# Find most correlated features to class label? For linear correlation hence linear separability (somewhat)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import *
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, chi2, SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

THRESHOLD_CORR = 0.045
N_COMPONENTS = 180
RANDOM_STATE = 123
THRESHOLD_DT = 0.01

# dimensionality reduction through selecting features with linear correlation above threshold. simplest and most intuitive but: correlation does not imply causality!
# static, done before input into learner 
def lin_correlation(train_df, test_df):

    features = train_df.drop('imdb_score_binned', axis=1)

    # Calculate linear correlation of each feature to the class label, in descending order
    correlations = pd.DataFrame(train_df.corr()['imdb_score_binned'].sort_values(ascending=False))
    correlations.columns = ['Correlation with imdb_score_binned']
    # print(correlations)

    # Plot all correlations to choose a threshold value: just before the graph flattens out
    # correlations.plot()
    # plt.xlabel('Column Index')
    # plt.ylabel('Correlation')
    # plt.title('Correlation of each feature with imdb_score_binned')
    # plt.show()

    # Features which have a correlation with the class label column greater than the chosen threshold 
    selected_features = correlations[correlations['Correlation with imdb_score_binned'] > THRESHOLD_CORR]

    # print("count=", selected_features.count())
    # print(selected_features.index)
    train_df_selected = features.copy()
    test_df_selected = test_df.copy()

    for feature in features:
        if (feature not in selected_features.index):
            if (feature == 'id'): 
                continue
            
            train_df_selected = train_df_selected.drop(columns=[feature])
            test_df_selected = test_df_selected.drop(columns=[feature])

    # print(train_df_selected)
    # print(test_df_selected)

    return train_df_selected, test_df_selected





    # return df_selected 


# dimensionality reduction according to explained variance 
# static
def pca(train_df, test_df):

    # Remove class label column
    train_df_features = train_df.drop('imdb_score_binned', axis=1)
    train_df_labels = train_df['imdb_score_binned']

    # remove id column and concat to the end again after pca 
    id_column = train_df_features['id']
    train_df_features_no_id = train_df_features.drop(columns=['id'])
    test_df_no_id = test_df.drop(columns=['id'])

    # Fit PCA model to the training data and calculate the cumulative variance ratio i.e., how much variance is explained by n-1 principal components. 
    pca = PCA().fit(train_df_features_no_id)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    # Plot to observe number of components such that maximal variance is explained, but not maximising computational time. 
    plt.plot(cumulative_variance_ratio)

    # Get rid of auto axis scaling 
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False) 

    plt.title('Number of princical components needed for percentage of variance explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()

    # Choose n_components based on graph
    pca = PCA(n_components=N_COMPONENTS) 
    train_df_reduced = pd.DataFrame(pca.fit_transform(train_df_features_no_id))
    train_df_reduced = pd.concat([id_column, train_df_reduced], axis=1)

    test_df_reduced = pd.DataFrame(pca.fit_transform(test_df_no_id))
    test_df_reduced = pd.concat([id_column, test_df_reduced], axis=1)


    # convert all column names to strings
    train_df_reduced.columns = train_df_reduced.columns.astype(str)
    test_df_reduced.columns = test_df_reduced.columns.astype(str)

    train_df_reduced = handle_missing(train_df_reduced)
    test_df_reduced = handle_missing(test_df_reduced)

    return train_df_reduced, test_df_reduced


# forward Sequential Feature Selection (wrapper)
# dynamic: tuning requires input from learner 
# Find n features such that it maximises cross-validated score 
# For the range of n values given, find the n which maximises classifier accuracy 

def forward_sfs(features, labels, n_min, n_max, n_step, classifier):
    max_acc_n = 0
    max_acc = 0

    for i in range(n_min, n_max, n_step):
        # calculate sfs
        sfs = SequentialFeatureSelector(classifier, i)

        # calculate accuracy 
        #sfs.fit_transform(features, labels)
        score = cross_val_score(sfs, features, labels, cv=5)
        if (score.mean() > max_acc):
            max_acc = score.mean()
            max_acc_n = i

    return max_acc_n, max_acc


    
# embedded: decision tree
def selection_dt(features, labels):
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    clf.fit(features, labels)

    feature_names = features.columns
    feature_importances = pd.DataFrame(feature_names)

    # Get importances of all features according to classifier
    feature_importances['feature importances'] = pd.DataFrame(clf.feature_importances_)
    # Sort in descending order 
    feature_importances = feature_importances.sort_values(by='feature importances', ascending=False).reset_index()
    

    feature_importances.columns = ['Feature Index', 'Feature Name', 'Feature Importance']

    plt.plot(feature_importances['Feature Name'], feature_importances['Feature Importance'])
    plt.xticks(np.arange(0, len(feature_importances)+1, 10), rotation=90)
    plt.title("Features and their corresponding Importance according to Decision Tree Classifier")
    plt.xlabel("Features (for visibility, only some names are shown)")
    plt.ylabel("Feature Importance")
    plt.show()

    # Determine threshold through observing plot: return features with importance above this threshold
    features_selected = feature_importances[feature_importances['Feature Importance']>THRESHOLD_DT]
    print(features_selected)




    
# filtering: based on PMI, MI, Chi-Square  
    # dimensionality reduction through univariate feature selection: chi2, fscore, MI 
    # https://scikit-learn.org/stable/modules/feature_selection.html
    # adapted from https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html

# For a given K value, find the K best features according to different metrics
def filtering(features, labels, min_k, max_k, k_step, classifier):

    ensemble_methods = [
        (
            "SelectKBest, metric = f score",
            SelectKBest(f_classif),
        ),
        (
            "SelectKBest, metric = chi squared",
            SelectKBest(chi2),
        ),
        (
            "SelectKBest, metric = mutual information",
            SelectKBest(mutual_info_classif),
        )
    ]

    accuracy = OrderedDict((label, []) for label, _ in ensemble_methods)
    
    for label, kbest in ensemble_methods:
        for i in range(min_k, max_k + 1, k_step):
            print("label kbest with ", i)
            kbest.set_params(k=i)
            res = kbest.fit_transform(features, labels)

            acc = cross_val_score(classifier, res, labels, cv=5).mean()
            #print("accuracy =", acc)
            accuracy[label].append((i, acc))

    for label, acc in accuracy.items():
        print("zip")
        xs, ys = zip(*acc)
        plt.plot(xs, ys, label=label)

    print("hello")

    plt.xlim(min_k, max_k)
    plt.title("Number of k-best features versus accuracy of classifier")
    plt.xlabel("n features")
    plt.ylabel("classifier accuracy")
    plt.legend(loc="upper right")
    plt.show()


# based on filtering graph results
def select_kbest_features(k, metric, features, labels, test_df, classifier):
    kbest = SelectKBest(metric, k=k)
    new_features = kbest.fit_transform(features, labels)
    new_features_test = kbest.transform(test_df)
    #print(new_features)
    return new_features, new_features_test



def main():
    
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()

    # train_df_minmax_corr = lin_correlation(train_df_minmax, test_df_minmax)
    # print(train_df_minmax_corr)
    #train_df_minmax_pca = pca(train_df_minmax)
    knn = KNeighborsClassifier(n_neighbors=5)
    filtering(train_df_minmax.drop('imdb_score_binned', axis=1), train_df_minmax['imdb_score_binned'], 10, 300, 10, knn)
    # selection_dt(train_df_minmax.drop('imdb_score_binned', axis=1), train_df_minmax['imdb_score_binned'])
    


if __name__ == '__main__':
    main()