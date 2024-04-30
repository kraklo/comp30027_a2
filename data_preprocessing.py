# Data preprocessing: handling missing values and normalisation

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_scaling import *


train_df = pd.read_csv('project_data/train_dataset.csv')
test_df = pd.read_csv('project_data/test_dataset.csv')


# 1. Handling missing values 
def handle_missing (dataset):

    # just drop rows that contain missing values 
    # reasoning: refer to lecture. only 1 row (show this)
    return dataset.dropna()

# 2. replace strings with vectors 

def preprocess():

    train = handle_missing(train_df)
    test = handle_missing(test_df)

    # 3. Normalisation 
    train_df_minmax, test_df_minmax = scale(train, test, calc_minmax, minmax_scaler)
    train_df_std, test_df_std = scale(train, test, calc_meanstdev, standardized_scaler)

    # return train and test dfs scaled 
    return train_df_minmax, test_df_minmax, train_df_std, test_df_std

# should work once strings are replaced with numerical values 
# train_df_minmax, test_df_minmax, train_df_std, test_df_std = preprocess()


