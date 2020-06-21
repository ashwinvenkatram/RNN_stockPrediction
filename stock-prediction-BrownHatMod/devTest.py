from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from yfinance import *
from collections import deque

import numpy as np
import pandas as pd
import random
from parameters import *
from datetime import date, timedelta
import sys


# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
random.seed(314)

def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
              test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the data, default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        # df = si.get_data(ticker)
        tickerData = Ticker(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        tickerData = ticker
    else:
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself

    if PERIOD == "max":
        df = tickerData.history(period=PERIOD)
    else:
        df = tickerData.history(start=STARTDATE, end=ENDDATE, interval=INTERVAL)
    result['df'] = df.copy()

    # copy open[1:] into new column df[feature_columns[-1]] = open[1:]
    # Deleting the last row
    shiftedOpen = df['Open'].values[1:]
    shiftedOpen = np.append(shiftedOpen,0)
    df.insert(column='NextOpen', value=shiftedOpen,loc=len(df.columns))
    df = df[:-1]

    # make sure that the passed feature_columns exist in the dataframe
    df.columns = [c.lower() for c in df.columns]  # converting all to lower case
    for col in feature_columns:
        assert col.lower() in df.columns, f"'{col.lower()}' does not exist in the dataframe/feature columns."

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    # df['future'] = df['adjclose'].shift(-lookup_step)
    df['future'] = df['close'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                test_size=test_size,
                                                                                                shuffle=shuffle)
    print("X_train shape: ", result["X_train"].shape)
    print("X_test shape: ", result["X_test"].shape)
    print("y_train shape: ", result["y_train"].shape)
    print("y_test shape: ", result["y_test"].shape)
    # return the result
    return result

data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)