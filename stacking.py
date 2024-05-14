from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from feature_selection import *
from data_preprocessing import *

RANDOM_STATE = 123

def run_stacking(features, labels, test):

    # this is pretty much just base models with optimised parameters: is there a way to stack where each takes a different dataset? dont think thats possible. cuz they all perform best with diff feature selection
    # get rid of magic numbers!
    estimators = [
        ('knn', KNeighborsClassifier(n_neighbors=106)),
        ('rf', RandomForestClassifier(n_estimators=200, criterion='entropy', max_features='log2', random_state=RANDOM_STATE)),
        ('perceptron', Perceptron(random_state=RANDOM_STATE)),
        ('svm', svm.SVC(C=0.1, gamma=1, kernel='rbf', random_state=RANDOM_STATE))
    ]

    clf = StackingClassifier(estimators=estimators)
    clf.fit(features, labels) #is it meant to be fit_transform??
    score = cross_val_score(clf, features, labels, cv=5)
    print(score.mean())

    print(clf.predict(test))





def main():
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()
    train_df_labels = train_df_minmax['imdb_score_binned']
    train_df_features = train_df_minmax.drop('imdb_score_binned', axis=1)

    #train_df_lin, test_df_lin = lin_correlation(train_df_minmax, test_df_minmax)

    run_stacking(train_df_features, train_df_labels, test_df_minmax)









if __name__ == '__main__':
    main()