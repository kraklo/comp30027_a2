# From our assignment 1

def calc_minmax(dataframe, attribute):
    return dataframe[attribute].min(), dataframe[attribute].max()


def calc_meanstdev(dataframe, attribute):
    return dataframe[attribute].mean(), dataframe[attribute].std()


def minmax_scaler(number, scaling_values):
    min_value, max_value = scaling_values

    return (number - min_value) / (max_value - min_value)


def standardized_scaler(number, scaling_values):
    mean, stdev = scaling_values

    return (number - mean) / stdev


def scale(train_df, test_df, value_calc, scaler):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for attribute in train_df.columns:
        # skip label column
        if attribute == 'imdb_score_binned':
            continue
        if attribute == 'id':
            continue

        # calculate values from TRAINING dataset
        values = value_calc(train_df, attribute)

        train_df[attribute] = scaler(train_df[attribute], values)
        test_df[attribute] = scaler(test_df[attribute], values)

    return train_df, test_df
