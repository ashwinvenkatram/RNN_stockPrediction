from backtesting import Backtest, Strategy
from parameters import *

import numpy as np
import pandas as pd
import os
import sys

from yfinance import *

from sklearn import preprocessing

from stock_prediction import create_model

from collections import deque

from heikenAshi import *

global data, result, model, DateList

y_predArr = []
startWindow = 0
prevState = 1
# 1: Sell (currently out of market)
# 0: Buy (currently in market)
waitCount = 0
logData = []

# def load_data(ticker, n_steps=50, scale=True, lookup_step=1, feature_columns=['close', 'volume', 'open', 'high', 'low']):
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
#     if PERIOD=="max":
#         dfOri = tickerData.history(period=PERIOD)
#     else:
#         dfOri = tickerData.history(start=STARTDATE, end=ENDDATE,interval=INTERVAL)
#
#
#     dfCopy = dfOri.copy()
#
#     # copy open[1:] into new column df[feature_columns[-1]] = open[1:]
#     # Deleting the last row
#     shiftedOpen = dfOri['Open'].values[1:]
#     shiftedOpen = np.append(shiftedOpen, 0)
#     dfCopy.insert(column='NextOpen', value=shiftedOpen, loc=len(dfCopy.columns))
#     dfCopy.insert(column='date', value=dfOri.index.copy(), loc=0)
#
#     dfCopy = dfCopy[:-1]
#
#     result['df'] = dfCopy.copy()
#     # make sure that the passed feature_columns exist in the dataframe
#     dfCopy.columns = [c.lower() for c in dfCopy.columns]  # converting all to lower case
#     for col in feature_columns:
#         assert col.lower() in dfCopy.columns, f"'{col.lower()}' does not exist in the dataframe/feature columns."
#
#     if scale:
#         column_scaler = {}
#         # scale the data (prices) from 0 to 1
#         for column in feature_columns:
#             scaler = preprocessing.MinMaxScaler()
#             dfCopy[column] = scaler.fit_transform(np.expand_dims(dfCopy[column].values, axis=1))
#             column_scaler[column] = scaler
#
#         # add the MinMaxScaler instances to the result returned
#         result["column_scaler"] = column_scaler
#
#     # add the target column (label) by shifting by `lookup_step`
#     # df['future'] = df['adjclose'].shift(-lookup_step)
#     dfCopy['future'] = dfCopy['close'].shift(-lookup_step)
#     # last `lookup_step` columns contains NaN in future column
#     # get them before droping NaNs
#     last_sequence = np.array(dfCopy[feature_columns].tail(lookup_step))
#
#     # drop NaNs
#     dfCopy.dropna(inplace=True)
#
#     sequence_data = []
#     sequences = deque(maxlen=n_steps)
#
#     for entry, target in zip(dfCopy[feature_columns].values, dfCopy['future'].values):
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
#     result["y_test"] = np.array(y)
#
#     # reshape X to fit the neural network
#     result["X_test"] = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
#
#     # processed = pd.DataFrame.from_dict(result)
#     # return the result
#     return dfOri, result#, processed


# FOR STATIONARY INPUT DATASET
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

    df.insert(column='date', value=df.index.copy(), loc=0)
    # make sure that the passed feature_columns exist in the dataframe
    df.columns = [c.lower() for c in df.columns]  # converting all to lower case
    for col in feature_columns:
        assert col.lower() in df.columns, f"'{col.lower()}' does not exist in the dataframe/feature columns."

    dfOri = df.copy()
    for cols in df.columns:
        if cols not in feature_columns:
            dfOri = dfOri.drop(columns = [cols], axis=1)


    df = compute_heikenAshi(df)

    # Subtract previous data from current data by shifting the column down by 1\
    # Stationary transformation
    df['Open_logDiff'] = np.log(df['Open'.lower()]) - np.log(df['Open'.lower()]).shift(1)
    df['Close_logDiff'] = np.log(df['Close'.lower()]) - np.log(df['Close'.lower()]).shift(1)

    df['High_logDiff'] = np.log(df['High'.lower()]) - np.log(df['High'.lower()]).shift(1)
    df['Low_logDiff'] = np.log(df['Low'.lower()]) - np.log(df['Low'.lower()]).shift(1)
    df['Volume_logDiff'] = np.log(df['volume'.lower()]) - np.log(df['volume'.lower()]).shift(1)

    shiftedOpen = df['Open_logDiff'].shift(-lookup_step)
    shiftedOpenOri = df['Open'.lower()].shift(-lookup_step)

    df.insert(column='NextOpen_logDiff', value=shiftedOpen, loc=len(df.columns))
    df.insert(column='NextOpen'.lower(), value=shiftedOpenOri, loc=len(df.columns))

    # drop =/- inf by setting to NaN
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    df.dropna(inplace=True)

    result['df'] = df.copy()

    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        feature_columns = ["Open_logDiff", "NextOpen_logDiff", "High_logDiff", "Low_logDiff", "Close_logDiff",
                           "Volume_logDiff"]
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Close_logDiff'].shift(-lookup_step)

    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)
    # [df['future'].values, df['Close'.lower()].shift(-lookup_step)]
    for entry, target, target2 in zip(df[feature_columns].values, df['future'].values,
                                      df['Close'.lower()].shift(-lookup_step)):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target, target2])

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
    for seq, target, target2 in sequence_data:
        X.append(seq)
        y.append([target, target2])

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    result["y_test_logDiffClose"] = y[:, 1]
    result["y_test"] = y[:, 0]

    # reshape X to fit the neural network
    result["X_test"] = X.reshape((X.shape[0], X.shape[2], X.shape[1]))

    # return the result
    return dfOri, result

def prediction():
    global startWindow # slideWindowX,slideWindowY
    global model, y_predArr, processed

    # if startWindow % 70 == 0:
    #     y_predArr = []

    slideWindowX = processed["X_test"][startWindow:startWindow + 70]
    slideWindowY = processed["y_test"][startWindow:startWindow + 70]

    # checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True,
    #                                save_best_only=True, verbose=1)
    # tensorboard = TensorBoard(log_dir=os.path.join("logs/", model_name))

    # if len(slideWindowX) < 70:
    #     print("end of dataset reached\n")
    #     endCheck = 1
    # else:
    #     model.fit(slideWindowX, slideWindowY,
    #               batch_size=BATCH_SIZE,
    #               epochs=1,
    #               verbose=1)

    y_pred = model.predict(slideWindowX)
    # y_pred = processed["column_scaler"]["close"].inverse_transform(y_pred)[0][0]

    # invert close normalization process
    y_pred = np.squeeze(processed["column_scaler"]["Close_logDiff"].inverse_transform(y_pred))

    # df['Close_logDiff'] = np.log(df['Close'.lower()]) - np.log(df['Close'.lower()]).shift(1)

    # non-stationary conversion
    y_pred = np.exp(y_pred + np.log(processed["y_test_logDiffClose"][startWindow:startWindow+70]))

    # Removing NaN
    y_pred = y_pred[~np.isnan(y_pred)]

    y_predArr.append(y_pred)

    startWindow = startWindow + 1

    try:
        slideWindowX = processed["X_test"][startWindow:startWindow + 70]
        slideWindowY = processed["y_test"][startWindow:startWindow + 70]
        endCheck = 0
    except IndexError:
        print("end of dataset reached\n")
        endCheck = 1

    return model, endCheck

# class brownHatStrategy(Strategy):
#     def init(self):
#         global model
#         # construct the model
#         model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
#                              dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
#
#         model_path = os.path.join("results", model_name) + ".h5"
#         model.load_weights(model_path)
#         startWindow = 0
#
#     def next(self):
#         global model, prevState, waitCount
#         if waitCount >= 69 and waitCount < 300: #len(data["Open"]) -
#             try:
#                 currOpen = processed["df"]["Open"][startWindow + 70]
#                 nextOpen = processed["df"]["NextOpen"][startWindow+70] # 0 based, 70th index is 71st data point; outside of the sliding window setup
#                 actualClose = processed["df"]["Close"][startWindow+70] # startWindow iterates in prediction()
#                 date = DateList[startWindow+70]
#                 model, endCheck = prediction()
#                 predClose = y_predArr[-1]
#                 if len(y_predArr) >= 2:
#                     prevPredClose = y_predArr[-2]
#                     prevClose = processed["df"]["Close"][startWindow+69]
#
#                     # print("Open: ", nextOpen, "Pred Close: ", predClose, "Actual Close: ",actualClose)
#                     if DECISIONTYPE == 0:
#                         delta = 100 * (predClose - nextOpen) / nextOpen
#                     elif DECISIONTYPE == 1:
#                         delta = 100 * (predClose - prevClose) / prevClose
#                     elif DECISIONTYPE == 2:
#                         delta = 100 * (predClose - prevPredClose) / prevPredClose
#                     elif DECISIONTYPE == 3:
#                         delta = 100 * (predClose - currOpen) / currOpen
#                     else:
#                         delta = 100 * (predClose - currOpen) / currOpen
#
#                     if delta > 0.1 and prevState == 1:
#                         self.buy()
#                         prevState = 0
#                         print("Date:", date, " Buy! ", currOpen, predClose)
#                         logFormat(prevState, date, waitCount, currOpen, nextOpen, prevPredClose, predClose, prevClose, actualClose, delta)
#                         # logData.append(str(date) + ",Buy," + str(nextOpen) + "," + str(prevPredClose) + "," + str(predClose) + "," + str(delta) + "\n")
#                     elif delta < -0.1 and prevState==0:
#                         self.sell()
#                         prevState = 1
#                         print("Date:", date, " Sell! ", currOpen, predClose)
#                         logFormat(prevState, date, waitCount, currOpen, nextOpen, prevPredClose, predClose, prevClose,actualClose, delta)
#                         # logData.append(str(date) + ",Sell," + str(nextOpen) + "," + str(prevPredClose) + "," + str(predClose) + "," + str(delta) + "\n")
#                     else:
#                         print("Date:", date, " Holding...")
#                         logFormat(2, date, waitCount, currOpen, nextOpen, prevPredClose, predClose, prevClose,actualClose, delta)
#                         # logData.append(str(date) + ",Hold," + str(nextOpen) + "," + str(prevPredClose) + "," + str(predClose) + "," + str(delta) + "\n")
#                         pass
#                     waitCount = waitCount + 1
#
#             except IndexError:
#                 pass
#
#         else:
#             waitCount = waitCount + 1
#             print("Iter:", waitCount, " Waiting...")


# logData.append("Iter: " + str(waitCount) + " Waiting...\n")

class brownHatStrategy(Strategy):
    def init(self):
        global model
        # construct the model
        model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                             dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

        model_path = os.path.join("results", model_name) + ".h5"
        model.load_weights(model_path)
        startWindow = 0

    def next(self):
        global model, prevState, waitCount, processed
        if waitCount >= 69:
            try:
                # print("next")
                # currOpen = processed["df"]['Open'.lower()][startWindow:startWindow + 70]
                # nextOpen = processed["df"]["NextOpen".lower()][startWindow:startWindow+70] # 0 based, 70th index is 71st data point; outside of the sliding window setup
                # actualClose = processed["df"]['Close'.lower()][startWindow:startWindow+70] # startWindow iterates in prediction()
                # date = DateList[startWindow+70]
                model, endCheck = prediction()
                predClose = y_predArr[0][-1]

                # if len(y_predArr) >= 1:
                    # prevPredClose = y_predArr[-2]
                    # prevClose = processed["df"]['Close'.lower()][startWindow+69]
                if 1:
                    delta = 100 * (predClose - self.data.Open[-1]) / self.data.Open[-1]

                    if delta > 0.1 and prevState==1:
                        self.buy()
                        prevState = 0
                        print("Iter:", waitCount, "/3490 Buy")
                    elif delta < -0.1 and prevState==0:
                        self.sell()
                        prevState = 1
                        print("Iter:", waitCount, "/3490 Sell")
                    else:
                        print("Iter:", waitCount, "/3490 Holding...")
                waitCount = waitCount + 1
            except IndexError:
                pass
        else:
            waitCount = waitCount + 1
            print("Iter:", waitCount, "/3490 Waiting...")

def logFormat(prevState, date, waitCount, currOpen, nextOpen, prevPredClose,predClose,prevClose, actualClose, delta):
    if prevState==0:
        buySellTrig = "Buy"
    elif prevState==1:
        buySellTrig = "Sell"
    else:
        buySellTrig = "Hold"

    if DECISIONTYPE == 0:
        logData.append(str(date) + "," + str(waitCount) + "," + str(buySellTrig) + "," + str(nextOpen) + "," + str(prevPredClose) + "," + str(predClose) + "," + str(delta) + "\n")

    elif DECISIONTYPE == 1:
        logData.append(str(date) + "," + str(waitCount) + "," + str(buySellTrig) + "," + str(nextOpen) + "," + str(prevClose) + "," + str(predClose) + "," + str(delta) + "\n")
    elif DECISIONTYPE == 2:
        logData.append(str(date) + "," + str(waitCount) + "," + str(buySellTrig) + "," + str(nextOpen) + "," + str(predClose) + "," + str(prevPredClose) + "," + str(delta) + "\n")
    elif DECISIONTYPE == 3:
        logData.append(str(date) + "," + str(waitCount) + "," + str(buySellTrig) + "," + str(currOpen) + "," + str(predClose) + "," + str(actualClose) + "," + str(delta) + "\n")
    else:
        logData.append(str(date) + "," + str(waitCount) + "," + str(buySellTrig) + "," + str(currOpen) + "," + str(predClose) + "," + str(actualClose) + "," + str(delta) + "\n")

if __name__=="__main__":
    global processed
    if not os.path.isdir("backtest"):
        os.mkdir("backtest")

    if not os.path.isdir(f"backtest/{ticker}"):
        os.mkdir(f"backtest/{ticker}")

    # data: original dataframe, result: dictionary of data normalized for prediction and reinforcement
    # data, processed = load_data(ticker, N_STEPS, True, lookup_step=LOOKUP_STEP, feature_columns=FEATURE_COLUMNS)
    data, processed = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS, shuffle=False)

    # title case for headers
    data.columns = [c.title() for c in data.columns]

    DateList = processed["df"]["date"]
    #
    # processed["df"] = processed["df"].set_index("date")

    bt = Backtest(data, brownHatStrategy, cash=10000, commission=COMMISSION, trade_on_close=False)
    res = bt.run()
    print(res)

    if DECISIONTYPE == 0:
        DECTYPE = "predClose-nextOpen"
        header = "Date,waitCount,State,NextOpen,prevPredClose,predClose,delta\n"
    elif DECISIONTYPE == 1:
        DECTYPE = "predClose-prevClose"
        header = "Date,waitCount,State,NextOpen,prevClose,predClose,delta\n"
    elif DECISIONTYPE == 2:
        DECTYPE = "predClose-prevPredClose"
        header = "Date,waitCount,State,NextOpen,predClose,prevPredClose,delta\n"
    elif DECISIONTYPE == 3:
        DECTYPE = "predClose-currOpen"
        header = "Date,waitCount,State,currOpen,predClose,prevPredClose,delta\n"
    else:
        DECTYPE = "predClose-currOpen"
        header = "Date,waitCount,State,currOpen,predClose,prevPredClose,delta\n"

    bt.plot(filename=f"./backtest/{ticker}/backtest-{DECTYPE}.html")

    fileName = f"./backtest/{ticker}/backtestLogs-{DECTYPE}.csv"

    # with open(fileName,"w") as outfile:
    #     outfile.write(header)
    #     for row in logData:
    #         outfile.write(row)
    # outfile.close()

#  try test.py with marker ^ to mark buy/ sell conditions over time. visualize the system on test plot instead of backtest.plot()
#  Correlation of pred with buy sell condition not clear