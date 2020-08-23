from stock_prediction import create_model

from statsmodels.tsa.stattools import adfuller
from heikenAshi import *

from binancePull import get_all_binance
import math
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from yfinance import *
from collections import deque
import pickle
import numpy as np
import pandas as pd
import os
import random
from parameters import *
from datetime import date, timedelta
import sys

from tensorflow.compat.v1.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.compat.v1.keras.models import Sequential, load_model, save_model
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dense, Dropout, Flatten

from stock_prediction import mydata_test_train_split, create_sequence_data

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

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        # feature_columns = ["Open_logDiff", "NextOpen_logDiff","High_logDiff", "Low_logDiff", "Close_logDiff","Volume_logDiff"]
        feature_columns = ["Open_logDiff", "High_logDiff", "Low_logDiff", "Close_logDiff",
                           "Volume_logDiff"]

        # scaler = preprocessing.MinMaxScaler()
        scaler = pickle.load(open(os.path.join("results", 'scaler-'+model_name+'.pickle'),'rb'))
        for column in feature_columns:
            df[column] = scaler.transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
        # dump(scaler, open(os.path.join("results", 'scaler-' + model_name + '.pickle'), 'wb'))

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Close_logDiff'].shift(-LOOKUP_STEP)

    # drop NaNs
    df.dropna(inplace=True)

    # At this point df of stationary OHLCV, groundtruth predClose has been created
    dfTrain, dfTest = mydata_test_train_split(df, TEST_SIZE)

    # contains the full dataframe of OHLCV after dropped datapoints and computing stationary dataset
    result['df'] = df.copy()
    result['dfTrain'] = dfTrain.copy()
    result['dfTest'] = dfTest.copy()

    # creating sequence data from stationary and scaled dataset
    X_Train, Y_Train = create_sequence_data(dfTrain)
    X_Test, Y_Test = create_sequence_data(dfTest)


    result["X_train"] = X_Train
    result["y_train_logDiffClose"] = Y_Train[:, 1]  # shifted close data for recalculation of actual close
    result["y_train"] = Y_Train[:, 0]  # stationary and normalized data

    result["X_test"] = X_Test
    result["y_test_logDiffClose"] = Y_Test[:, 1]  # shifted close data for recalculation of actual close
    result["y_test"] = Y_Test[:, 0]  # stationary and normalized data

    # Assuming not shuffled

    testDataset = pd.DataFrame()
    testDataset['open'] = result['dfTest']['open']
    testDataset['high'] = result['dfTest']['high']
    testDataset['low'] = result['dfTest']['low']
    testDataset['close'] = result['dfTest']['close']
    testDataset['volume'] = result['dfTest']['volume']

    # return the result
    return result, testDataset

if __name__=="__main__":
    # load the data
    dataset, testDataset = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                                  feature_columns=FEATURE_COLUMNS, shuffle=False)

    # construct the model
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                         dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)

    # startWindow = len(dataset["X_Test"])-15 # start prediction for last 15

    # log dataframe
    logData = pd.DataFrame()
    closeData = np.array([])
    predCloseData = np.array([])

    # for i in range(10): # math.floor(len(dataset["X_test"])/70)
    # slideWindowX = np.array([dataset["X_test"][startWindow]])
    # actualClose = testDataset["close"][startWindow + 70]

    y_pred = model.predict(dataset["X_test"])


    y_pred = np.squeeze(dataset["column_scaler"]["Close_logDiff"].inverse_transform(y_pred))
    y_pred = np.exp(y_pred + np.log(dataset["y_test_logDiffClose"]))

    y_test = dataset["y_test"]
    y_test = np.squeeze(dataset["column_scaler"]["Close_logDiff"].inverse_transform(np.expand_dims(y_test, axis=1)))
    y_test = np.exp(y_test + np.log(dataset["y_test_logDiffClose"][startWindow + 69: startWindow+75]))

    closeData = np.append(closeData, actualClose)
    predCloseData = np.append(predCloseData, y_pred)

    startWindow = startWindow + 1

    # logData["Date"] = testDataset.index[:10]
    logData.insert(column='close', value=closeData, loc=len(logData.columns))
    logData.insert(column='predClose', value=predCloseData, loc=len(logData.columns))
    # logData["close"] = closeData
    # logData["predClose"] = predCloseData

    filename = './backtest/BTCUSDT/BTCUSDT_TEST_LOGFILE.csv'
    logData.to_csv(filename)
