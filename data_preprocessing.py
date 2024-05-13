# Data preprocessing: handling missing values and normalisation

import pandas as pd
import numpy as np
from data_scaling import *
from sklearn.feature_extraction.text import CountVectorizer


# 1. Handling missing values 
def handle_missing(dataset):
    # just drop rows that contain missing values
    # reasoning: refer to lecture. only 1 row (show this)
    return dataset.dropna()


def countvec_to_indexes(countvec):
    vecs = []
    for vec in (countvec == 1):
        vec = np.argwhere(vec)
        if len(vec) == 1:
            vecs.append([vec[0][0], -1])
        elif len(vec) == 2:
            vecs.append([vec[0][0], vec[1][0]])
        elif len(vec) == 0:
            vecs.append([-1, -1])
        else:
            vecs.append([vec[0][0], vec[-1][0]])

    return np.transpose(np.array(vecs))


# 2. replace strings with vectors
def replace_strings(dataset, file_prefix):
    dataset = dataset.drop([
        'director_name',
        'actor_1_name',
        'actor_2_name',
        'plot_keywords',
        'genres',
        'title_embedding',
        'movie_title',
        'actor_3_name'
    ], axis=1)

    director_name = np.load(f'./project_data/features_countvec/{file_prefix}_countvec_features_director_name.npy')
    actor_1_name = np.load(f'./project_data/features_countvec/{file_prefix}_countvec_features_actor_1_name.npy')
    actor_2_name = np.load(f'./project_data/features_countvec/{file_prefix}_countvec_features_actor_2_name.npy')
    plot_keywords = np.load(f'./project_data/features_doc2vec/{file_prefix}_doc2vec_features_plot_keywords.npy')
    genres = np.load(f'./project_data/features_doc2vec/{file_prefix}_doc2vec_features_genre.npy')
    title_embedding = np.load(f'./project_data/features_fasttext/{file_prefix}_fasttext_title_embeddings.npy')

    director_first_name, director_last_name = countvec_to_indexes(director_name)
    actor_1_first_name, actor_1_last_name = countvec_to_indexes(actor_1_name)
    actor_2_first_name, actor_2_last_name = countvec_to_indexes(actor_2_name)

    dataset['director_first_name'] = director_first_name
    dataset['director_last_name'] = director_last_name
    dataset['actor_1_first_name'] = actor_1_first_name
    dataset['actor_1_last_name'] = actor_1_last_name
    dataset['actor_2_first_name'] = actor_2_first_name
    dataset['actor_2_last_name'] = actor_2_last_name

    dataset = pd.concat([
        dataset,
        pd.DataFrame(plot_keywords, columns=[f'plot_keywords{i}' for i in range(plot_keywords.shape[1])]),
        pd.DataFrame(genres, columns=[f'genres{i}' for i in range(genres.shape[1])]),
        pd.DataFrame(title_embedding, columns=[f'title_embedding{i}' for i in range(title_embedding.shape[1])])
    ], axis=1)

    vectorizer = CountVectorizer()

    dataset['country'] = dataset['country'].fillna('')
    dataset['language'] = dataset['language'].fillna('')
    dataset['content_rating'] = dataset['content_rating'].fillna('')

    dataset['country'] = countvec_to_indexes(vectorizer.fit_transform(dataset['country']).toarray())[0]
    dataset['language'] = countvec_to_indexes(vectorizer.fit_transform(dataset['language']).toarray())[0]
    dataset['content_rating'] = countvec_to_indexes(vectorizer.fit_transform(dataset['content_rating']).toarray())[0]

    return dataset


def preprocess():
    train_df = pd.read_csv('project_data/train_dataset.csv')
    test_df = pd.read_csv('project_data/test_dataset.csv')
    
    # replace strings
    train_df = replace_strings(train_df, 'train')
    test_df = replace_strings(test_df, 'test')

    # handle missing values
    train_df = handle_missing(train_df)
    test_df = handle_missing(test_df)

    # 3. Normalisation 
    train_df_minmax, test_df_minmax = scale(train_df, test_df, calc_minmax, minmax_scaler)
    train_df_std, test_df_std = scale(train_df, test_df, calc_meanstdev, standardized_scaler)

    # return train and test dfs scaled 
    return train_df_minmax, test_df_minmax, train_df_std, test_df_std


def main():
    
    train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()


if __name__ == '__main__':
    main()
