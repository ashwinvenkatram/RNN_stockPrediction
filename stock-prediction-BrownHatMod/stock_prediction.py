from statsmodels.tsa.stattools import adfuller
from heikenAshi import *

from binancePull import get_all_binance

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from yfinance import *
from collections import deque

import math
import numpy as np
import pandas as pd
import random
from parameters import *
from pickle import dump
import os
from datetime import date, timedelta
import sys

from tensorflow.compat.v1.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.compat.v1.keras.models import Sequential, load_model, save_model
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dense, Dropout, Flatten
# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

# FOR STATIONARY INPUT DATASET
def load_data(ticker, n_steps=50, scale=True, shuffle=False, lookup_step=1,
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
    '''
    FOR YAHOO DATA PULLING
    '''
    # # see if ticker is already a loaded stock from yahoo finance
    # if isinstance(ticker, str):
    #     # load it from yahoo_fin library
    #     # df = si.get_data(ticker)
    #     tickerData = Ticker(ticker)
    # elif isinstance(ticker, pd.DataFrame):
    #     # already loaded, use it directly
    #     tickerData = ticker
    # else:
    #     raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
    #
    # # this will contain all the elements we want to return from this function
    # result = {}
    # # we will also return the original dataframe itself
    #
    # if PERIOD=="max":
    #     df = tickerData.history(period=PERIOD)
    # else:
    #     df = tickerData.history(start=STARTDATE, end=ENDDATE,interval=INTERVAL)


    '''
    FOR BINANCE DATA PULLING
    '''
    df = get_all_binance(ticker, INTERVAL, save=True)
    result = {}
    '''
    REST OF CODE
    '''
    df.insert(column='date', value=df.index.copy(), loc=0)
    # make sure that the passed feature_columns exist in the dataframe
    df.columns = [c.lower() for c in df.columns] # converting all to lower case
    for col in feature_columns:
        assert col.lower() in df.columns, f"'{col.lower()}' does not exist in the dataframe/feature columns."

    # df = compute_heikenAshi(df)

    # Subtract previous data from current data by shifting the column down by 1
    df['Open_logDiff'] = np.log(df['Open'.lower()]) - np.log(df['Open'.lower()]).shift(1)
    df['Close_logDiff'] = np.log(df['Close'.lower()]) - np.log(df['Close'.lower()]).shift(1)
    df['High_logDiff'] = np.log(df['High'.lower()]) - np.log(df['High'.lower()]).shift(1)
    df['Low_logDiff'] = np.log(df['Low'.lower()]) - np.log(df['Low'.lower()]).shift(1)
    df['Volume_logDiff'] = np.log(df['volume'.lower()]) - np.log(df['volume'.lower()]).shift(1)

    # shiftedOpen = df['Open_logDiff'].shift(-lookup_step)
    # df.insert(column='NextOpen_logDiff', value=shiftedOpen, loc=len(df.columns))

    # drop =/- inf by setting to NaN
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df.dropna(inplace=True)

    # contains the full dataframe of OHLCV after dropped datapoints and computing stationary dataset
    # result['df'] = df.copy()

    # scaling to normalize stationary dataset from 0 to 1
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        # feature_columns = ["Open_logDiff", "NextOpen_logDiff","High_logDiff", "Low_logDiff", "Close_logDiff","Volume_logDiff"]
        feature_columns = ["Open_logDiff", "High_logDiff", "Low_logDiff", "Close_logDiff",
                           "Volume_logDiff"]

        scaler = preprocessing.MinMaxScaler()

        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
        dump(scaler, open(os.path.join("results", 'scaler-' + model_name + '.pickle'), 'wb'))

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Close_logDiff'].shift(-LOOKUP_STEP)
    # colsList = df.columns.tolist()
    # colsList = colsList.remove('date')
    # last_sequence = np.array(df[colsList].tail(lookup_step))

    # drop NaNs
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df.dropna(inplace=True)

    # At this point df of stationary OHLCV, groundtruth predClose has been created
    dfTrain, dfTest = mydata_test_train_split(df, TEST_SIZE)

    # contains the full dataframe of OHLCV after dropped datapoints and computing stationary dataset
    result['df'] = df.copy()
    result['dfTrain'] = dfTrain.copy()
    result['dfTest'] = dfTest.copy()

    # scaling
    # if scale:
    #     result, dfTrain, scaler = scalingTrain(dfTrain, result)
    #     result, dfTest = scalingTest(dfTest,scaler, result)
    #
    #     dump(scaler, open(os.path.join("results", 'scaler-'+model_name+'.pickle'),'wb'))

    # creating sequence data from stationary and scaled dataset
    X_Train, Y_Train = create_sequence_data(dfTrain)
    X_Test, Y_Test = create_sequence_data(dfTest)

    # sequence_data = []
    # sequences = deque(maxlen=n_steps)
    # [df['future'].values, df['Close'.lower()].shift(-lookup_step)]
    # entry: input to RNN, stationary and normalized OHLCV
    # target: groundtruth close data (stationary and normalized) -> C_sn
    # target2: groundtruth close data (ha close or close depending on setting) -> C

    # feature_columns = ["Open_logDiff", "High_logDiff", "Low_logDiff", "Close_logDiff","Volume_logDiff"]
    # for entry, target, target2 in zip(df[feature_columns].values, df['future'].values,  df['Close'.lower()]):# -lookup_step.shift()
    #     sequences.append(entry)
    #     if len(sequences) == n_steps:
    #         # At Nth step, N X_train/test values exist, with the N+1th target and Nth close correction factor
    #         sequence_data.append([np.array(sequences), target, target2])
    #         # print([np.array(sequences), target, target2])
    #         # break

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    # last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    # last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    # result['last_sequence'] = last_sequence

    # # construct the X's and y's
    # X, y = [], []
    # # X: input OHLCV_logDiff and normalized
    # # Y: (C_sn, C)
    # for seq, target, target2 in sequence_data:
    #     X.append(seq)
    #     y.append([target, target2])
    #
    # # convert to numpy arrays
    # X = np.array(X)
    # y = np.array(y)
    #
    # # reshape X to fit the neural network
    # X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    # split the dataset
    # result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
    #                                                                             test_size=test_size, shuffle=shuffle)

    # result["y_train_logDiffClose"] = result["y_train"][:,1] # shifted close data for recalculation of actual close
    # result["y_train"] = result["y_train"][:,0] # stationary and normalized data
    #
    # result["y_test_logDiffClose"] = result["y_test"][:, 1] # shifted close data for recalculation of actual close
    # result["y_test"] = result["y_test"][:, 0] # stationary and normalized data

    result["X_train"] = X_Train
    result["y_train_logDiffClose"] = Y_Train[:, 1]  # shifted close data for recalculation of actual close
    result["y_train"] = Y_Train[:, 0]  # stationary and normalized data

    result["X_test"] = X_Test
    result["y_test_logDiffClose"] = Y_Test[:, 1]  # shifted close data for recalculation of actual close
    result["y_test"] = Y_Test[:, 0]  # stationary and normalized data

    # Assuming not shuffled
    # picking up values after first 70 datapoints
    testDataset = pd.DataFrame()
    testDataset['Timestamp'] = result['df']['timestamp'][-len(result["y_test"]):]
    testDataset['Open'] = result['df']['open'][-len(result["y_test"]):]
    testDataset['High'] = result['df']['high'][-len(result["y_test"]):]
    testDataset['Low'] = result['df']['low'][-len(result["y_test"]):]
    testDataset['Close'] = result['df']['close'][-len(result["y_test"]):]
    testDataset['Volume'] = result['df']['volume'][-len(result["y_test"]):]

    # return the result
    return result, testDataset

def scalingTrain(dfTrain, result):
    column_scaler = {}
        # scale the data (prices) from 0 to 1
        # feature_columns = ["Open_logDiff", "NextOpen_logDiff","High_logDiff", "Low_logDiff", "Close_logDiff","Volume_logDiff"]
    feature_columns = ["Open_logDiff", "High_logDiff", "Low_logDiff", "Close_logDiff",
                           "Volume_logDiff", "future"]

    scaler = preprocessing.MinMaxScaler()

    for column in feature_columns:
        scaler = preprocessing.MinMaxScaler()
        dfTrain[column] = scaler.fit_transform(np.expand_dims(dfTrain[column].values, axis=1))
        column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
    result["column_scaler"] = column_scaler
    return result, dfTrain, scaler

def scalingTest(dfTest, scaler, result):
    # scale the data (prices) from 0 to 1
    # feature_columns = ["Open_logDiff", "NextOpen_logDiff","High_logDiff", "Low_logDiff", "Close_logDiff","Volume_logDiff"]
    feature_columns = ["Open_logDiff", "High_logDiff", "Low_logDiff", "Close_logDiff",
                           "Volume_logDiff", "future"]
    for column in feature_columns:
        dfTest[column] = scaler.transform(np.expand_dims(dfTest[column].values, axis=1))

    # add the MinMaxScaler instances to the result returned
    return result, dfTest

def create_sequence_data(df):
    sequence_data = []
    sequences = deque(maxlen=N_STEPS)

    feature_columns = ["Open_logDiff", "High_logDiff", "Low_logDiff", "Close_logDiff","Volume_logDiff"]
    for entry, target, target2 in zip(df[feature_columns].values, df['future'].values,  df['Close'.lower()].values):# df['Close'.lower()]
        sequences.append(entry)
        if len(sequences) == N_STEPS:
            # At Nth step, N X_train/test values exist, with the N+1th target and Nth close correction factor
            sequence_data.append([np.array(sequences), target, target2])
            # print([np.array(sequences), target, target2])
            # break

    # construct the X's and y's
    X, y = [], []
    # X: input OHLCV_logDiff and normalized
    # Y: (C_sn, C)
    for seq, target, target2 in sequence_data:
        X.append(seq)
        y.append([target, target2])

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # reshape X to fit the neural network
    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    return X, y

# dataset will be the full df. to be split into test and train respectively
def mydata_test_train_split(dataset,test_size):
    train_size = 1 - test_size
    rows = dataset.shape[0]

    trainRows = math.floor(rows * train_size)
    trainSet = dataset.copy()[:trainRows]
    testSet = dataset.copy()[trainRows:]
    return trainSet, testSet

# For non stationary input dataset
# def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
#               test_size=0.2, feature_columns=['close', 'volume', 'open', 'high', 'low']):
#     """
#     Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
#     Params:
#         ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
#         n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
#         scale (bool): whether to scale prices from 0 to 1, default is True
#         shuffle (bool): whether to shuffle the data, default is True
#         lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
#         test_size (float): ratio for test data, default is 0.2 (20% testing data)
#         feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
#     """
#     # see if ticker is already a loaded stock from yahoo finance
#     if isinstance(ticker, str):
#         # load it from yahoo_fin library
#         # df = si.get_data(ticker)
#         tickerData = Ticker(ticker)
#     elif isinstance(ticker, pd.DataFrame):
#         # already loaded, use it directly
#         tickerData = ticker
#     else:
#         raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
#
#     # this will contain all the elements we want to return from this function
#     result = {}
#     # we will also return the original dataframe itself
#
#     if PERIOD == "max":
#         df = tickerData.history(period=PERIOD)
#     else:
#         df = tickerData.history(start=STARTDATE, end=ENDDATE, interval=INTERVAL)
#
#     # copy open[1:] into new column df[feature_columns[-1]] = open[1:]
#     # Deleting the last row
#     shiftedOpen = df['Open'].values[1:]
#     shiftedOpen = np.append(shiftedOpen, 0)
#     df.insert(column='NextOpen', value=shiftedOpen, loc=len(df.columns))
#     df = df[:-1]
#
#     result['df'] = df.copy()
#
#     # make sure that the passed feature_columns exist in the dataframe
#     df.columns = [c.lower() for c in df.columns]  # converting all to lower case
#
#     for col in feature_columns:
#         assert col.lower() in df.columns, f"'{col.lower()}' does not exist in the dataframe/feature columns."
#
#     if scale:
#         column_scaler = {}
#         # scale the data (prices) from 0 to 1
#         for column in feature_columns:
#             scaler = preprocessing.MinMaxScaler()
#             df[column.lower()] = scaler.fit_transform(np.expand_dims(df[column.lower()].values, axis=1))
#             column_scaler[column.lower()] = scaler
#
#         # add the MinMaxScaler instances to the result returned
#         result["column_scaler"] = column_scaler
#
#     # add the target column (label) by shifting by `lookup_step`
#     # df['future'] = df['adjclose'].shift(-lookup_step)
#     df['future'] = df['close'].shift(-lookup_step)
#     # last `lookup_step` columns contains NaN in future column
#     # get them before droping NaNs
#     last_sequence = np.array(df[feature_columns].tail(lookup_step))
#
#     # drop NaNs
#     df.dropna(inplace=True)
#
#     sequence_data = []
#     sequences = deque(maxlen=n_steps)
#
#     for entry, target in zip(df[feature_columns].values, df['future'].values):
#         sequences.append(entry)
#         if len(sequences) == n_steps:
#             sequence_data.append([np.array(sequences), target])
#
#     # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
#     # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
#     # this last_sequence will be used to predict in future dates that are not available in the dataset
#     last_sequence = list(sequences) + list(last_sequence)
#     # shift the last sequence by -1
#     last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
#     # add to result
#     result['last_sequence'] = last_sequence
#
#     # construct the X's and y's
#     X, y = [], []
#     for seq, target in sequence_data:
#         X.append(seq)
#         y.append(target)
#
#     # convert to numpy arrays
#     X = np.array(X)
#     y = np.array(y)
#
#     # reshape X to fit the neural network
#     X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
#
#     # split the dataset
#     result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
#                                                                                                 test_size=test_size,
#                                                                                                 shuffle=shuffle)
#     print("X_train shape: ", result["X_train"].shape)
#     print("X_test shape: ", result["X_test"].shape)
#     print("y_train shape: ", result["y_train"].shape)
#     print("y_test shape: ", result["y_test"].shape)
#     # return the result
#     return result

def create_model(sequence_length, units=256, cell=CuDNNLSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    # for i in range(n_layers):
    #     if i == 0:
    #         # first layer
    #         if bidirectional:
    #             model.add(Bidirectional(cell(units, return_sequences=True), input_shape=(None, sequence_length)))
    #         else:
    #             model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
    #     elif i == n_layers - 1:
    #         # last layer
    #         if bidirectional:
    #             model.add(Bidirectional(cell(units, return_sequences=False)))
    #         else:
    #             model.add(cell(units, return_sequences=False))
    #     else:
    #         # hidden layers
    #         if bidirectional:
    #             model.add(Bidirectional(cell(units, return_sequences=True)))
    #         else:
    #             model.add(cell(units, return_sequences=True))
    #     # add dropout after each layer
    #     model.add(Dropout(dropout))
    for i in range(n_layers):
        if i == 0:
            # first layer
            model.add(cell(units, return_sequences=True, input_shape=(None, sequence_length)))
        elif i == n_layers - 1:
            # last layer
            model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[MeanAbsoluteError()], optimizer=optimizer)


    return model