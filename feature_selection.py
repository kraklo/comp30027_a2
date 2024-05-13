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
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, chi2

# dimensionality reduction through selecting features with linear correlation above threshold. simplest and most intuitive but: correlation does not imply causality!
def lin_correlation(train_df):
    #sns.heatmap(train_df.corr())
    #plt.show()

    correlations = pd.DataFrame(train_df.corr()['imdb_score_binned'].sort_values(ascending=False))
    correlations.columns = ['Correlation with imdb_score_binned']
    print(correlations)

    print(correlations[correlations['Correlation with imdb_score_binned'] > 0.15]) # greater than a threshold 

# dimensionality reduction through univariate feature selection: chi2, fscore, MI 
    # https://scikit-learn.org/stable/modules/feature_selection.html

# for a given K value, calculate 
def filtering(train_df, k):

    train_df_labels = train_df['imdb_score_binned']
    train_df_features = train_df.drop('imdb_score_binned', axis=1)

    train_df_kbest_fscore = SelectKBest(f_classif, k=k).fit_transform(train_df_features, train_df_labels)
    print('k=', k, ', ')
    train_df_kbest_chi2 = SelectKBest(chi2, k=k).fit_transform(train_df_features, train_df_labels)
    train_df_kbest_mi = SelectKBest(mutual_info_classif, k=k).fit_transform(train_df_features, train_df_labels)


# dimensionality reduction according to explained variance 
def pca(train_df):

    train_df_features = train_df.drop('imdb_score_binned', axis=1)

    pca = PCA().fit(train_df_features)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(cumulative_variance_ratio)

    # Get rid of auto axis scaling 
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)   

    plt.title('Number of princical components needed for percentage of variance explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.show()

    pca = PCA(n_components=180) # n_components chosen from above graph 
    train_df_reduced = pd.DataFrame(pca.fit_transform(train_df_features))
    return train_df_reduced


# forward SFS (wrapper)
    
# embedded: decision tree
    
# filtering: based on PMI, MI, Chi-Square  


def main():
    
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    lin_correlation(train_df_minmax)
    #train_df_minmax_pca = pca(train_df_minmax)
    


if __name__ == '__main__':
    main()