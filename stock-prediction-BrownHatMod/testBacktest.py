from stock_prediction import create_model, load_data, np
from parameters import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

import pickle
import numpy as np
from math import sqrt

import backtest
import sys
import os

import pandas as pd

global model, processed

y_predArr = []
startWindow = 0
prevState = 1
# 1: Sell (currently out of market)
# 0: Buy (currently in market)
waitCount = 0
logData = []
buySellTriggers = []

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

def plot_graph(y_pred, y_test, buySellTriggers, r2_score):
    OUTPUT_PATH = f"./backtest/{ticker}/BACKTEST-{DECTYPE}-" + model_name

    figTest = plt.figure()
    plt.style.use('seaborn-whitegrid')
    x_valTest = [i for i in range(len(y_test))]
    x_valPred = [i for i in range(len(y_pred))]
    plt.scatter(x_valTest,y_test[:], c='b',marker="o", s=3)
    plt.scatter(x_valPred,y_pred[:], c='r', marker="o", s=3)
    plt.plot(y_test[:], 'b')
    plt.plot(y_pred[:], 'r')

    x_coorBuy = []
    y_coorBuy = []
    x_coorSell = []
    y_coorSell = []

    for elem in buySellTriggers:
        markDay = elem[0]
        stateType = elem[1]

        # print(len(x_valTest), len(x_valPred), markDay)
        try:
            if stateType==0:
                x_coorBuy.append(x_valTest[markDay])
                y_coorBuy.append(y_test[markDay] - 5)
            elif stateType==1:
                x_coorSell.append(x_valTest[markDay])
                y_coorSell.append(y_test[markDay] + 5)
        except IndexError:
            pass
    # print(len(x_coorBuy), len(y_coorBuy))
    # print(len(x_coorSell), len(y_coorSell))
    plt.scatter(x_coorBuy,y_coorBuy, c='g', marker="^", s=50)
    plt.scatter(x_coorSell, y_coorSell, c='m', marker="v", s=50)

    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.title("Price Prediction Accuracy, R2: " + str(r2_score))

    # with open(f"results/prediction_{model_name}.csv",'w') as outfile:
    #     header = "x_val,y_pred\n"
    #     outfile.write(header)
    #     for i in range(len(x_val)):
    #         outfile.write(f"{x_val[i]},{y_pred[i]}"+"\n")
    # outfile.close()

    pickle.dump(figTest, open(OUTPUT_PATH + '.pickle', 'wb'))
    # plt.savefig(OUTPUT_PATH + '.png')
    plt.show()


# def prediction(model, X_test, y_test,dataset):
#     y_pred = model.predict(X_test)
#     print(len(X_test), len(y_pred))
#     y_test = np.squeeze(dataset["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
#     y_pred = np.squeeze(dataset["column_scaler"]["close"].inverse_transform(y_pred))
#
#     return y_test, y_pred

def predictionBacktest():
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
    y_pred = processed["column_scaler"]["close"].inverse_transform(y_pred)[0][0]
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


class brownHatStrategy:

    def __init__(self, checkflag = 0, cash= float(10000), commission = float(.0)):
        global model, startWindow

        numtrades = {"Buy": 0, "Sell": 0, "Total": 0}

        historyDF = pd.DataFrame(columns=['Iter_Count', 'BuyDate', 'BuyPrice','SellDate','SellPrice','UnitsPurchased','PLpercent', 'GainLoss', 'NetCash'])

        # tradedata = {"Iter Count": 0,"Buy Date": "", "Buy Price": 0, "Sale Date": "", "Sell Price": 0, "Units Purchased": 0., "P/L%": 0}
        # history = []
        percentChange = []
        self.checkFlag = checkflag
        self.numTrades = numtrades
        self.cash = cash
        # self.startSum = cash
        self.commission = commission
        # self.tradeData = tradedata
        self.historydF = historyDF
        self.percentChange = percentChange
        self.currIndex = 0
        # equity = [100 * (self.cash - self.startSum) / self.startSum]
        # self.equity = equity  # in percentage
        # construct the model
        model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                             dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

        model_path = os.path.join("results", model_name) + ".h5"
        model.load_weights(model_path)
        startWindow = 0

    def transaction(self, wCount, date, buySell, pricePoint):
        self.update_numTrades(buySell)
        if buySell==0:
            (units,remainder) = divmod(self.cash,pricePoint)
            # self.history["UnitsPurchased"] = units
            self.cash = remainder
            # self.tradeData["Buy Price"] = pricePoint
            # self.tradeData["Buy Date"] = date
            # self.tradeData["Iter Count"] = wCount
            self.historydF = self.historydF.append({'Iter_Count': wCount, 'BuyDate': date, 'BuyPrice': pricePoint,'SellDate': date, 'SellPrice': pricePoint, 'UnitsPurchased': units, 'PLpercent': 0, 'GainLoss': 0, 'NetCash': 0}, ignore_index=True)
            self.currIndex = len(self.historydF.index) - 1

        elif buySell==1:
            # columns=['Iter_Count', 'BuyDate', 'BuyPrice','SellDate','SellPrice','UnitsPurchased','P/L%', '$Gain/Loss', 'NetCash']

            # sys.exit()
            retAmt = pricePoint * self.historydF.iloc[self.currIndex, 5]
            self.cash = self.cash + retAmt

            # self.tradeData["Sell Price"] = pricePoint
            # self.tradeData["Sale Date"] = date
            # self.tradeData["P/L%"] = 100 * (self.tradeData["Sell Price"] - self.tradeData["Buy Price"])/self.tradeData["Buy Price"]
            # self.tradeData["Iter Count"] = wCount

            # self.history.append(self.tradeData)
            self.historydF.iloc[self.currIndex, 3] = date
            self.historydF.iloc[self.currIndex, 4] = pricePoint
            self.historydF.iloc[self.currIndex, 8] = retAmt

            sellPrice = self.historydF.iloc[self.currIndex, 4]
            buyPrice = self.historydF.iloc[self.currIndex, 2]
            PL = 100 * (sellPrice - buyPrice)/buyPrice

            self.historydF.iloc[self.currIndex, 6] = PL


            prevCash = self.historydF.iloc[self.currIndex-1, 8]
            cashGainLoss = retAmt - prevCash

            self.historydF.iloc[self.currIndex, 7] = cashGainLoss

            # self.update_equity()

            print(waitCount, self.historydF.iloc[self.currIndex])
            # self.reset_tradeData()

    def equityFinal(self):
        return self.historydF.iloc[len(self.historydF.index) - 1, 8]

    # def reset_tradeData(self):
    #     tradedata = {"Buy Date": "", "Buy Price": 0, "Sale Date": "", "Sell Price": 0, "Units Purchased": 0, "P/L%": 0}
    #     self.tradeData = tradedata

    # Should equity be a percentage? updated in an array for every timestep???
    # def update_equity(self):
    #     self.equity.append(100 * (self.historydF.iloc[self.currIndex, 8] - self.startSum)/self.startSum)

    def set_flag(self, checkFlag):
        self.checkFlag = checkFlag

    def set_numTrades(self):
        self.numTrades["Buy"] = 0
        self.numTrades["Sell"] = 0
        self.numTrades["Total"] = 0

    def update_numTrades(self, buySell):
        if buySell==0:
            self.numTrades["Buy"] = self.numTrades["Buy"] + 1
            self.numTrades["Total"] = self.numTrades["Total"] + 1
        elif buySell==1:
            self.numTrades["Sell"] = self.numTrades["Sell"] + 1
            self.numTrades["Total"] = self.numTrades["Total"] + 1

    def get_numTrades(self):
        return self.numTrades

    def get_flag(self):
        return self.checkFlag

    def get_history(self):
        return self.historydF

    def next(self):
        global model, prevState, waitCount, processed
        try:
            if waitCount >= 69:
                currOpen = processed["df"]["Open"][startWindow + 70]  # 0 based, 70th index is 71st data point; outside of the sliding window setup
                nextOpen = processed["df"]["NextOpen"][startWindow+70] # 0 based, 70th index is 71st data point; outside of the sliding window setup
                actualClose = processed["df"]["Close"][startWindow+70] # startWindow iterates in prediction()
                date = processed["df"]["date"][startWindow+70]

                model, endCheck = predictionBacktest()

                self.set_flag(endCheck)

                predClose = y_predArr[-1]
                if len(y_predArr) >= 2:
                    prevPredClose = 0 #y_predArr[-2]
                    prevClose = 0 #processed["df"]["Close"][startWindow+69]

                    # print("Open: ", nextOpen, "Pred Close: ", predClose, "Actual Close: ",actualClose)

                    if DECISIONTYPE == 0:
                        delta = 100 * (predClose - nextOpen)/nextOpen
                        pricePoint = nextOpen
                    elif DECISIONTYPE == 1:
                        delta = 100 * (predClose - prevClose) / prevClose
                        pricePoint = prevClose
                    elif DECISIONTYPE == 2:
                        delta = 100 * (predClose - prevPredClose) / prevPredClose
                        pricePoint = prevPredClose
                    elif DECISIONTYPE == 3:
                        delta = 100 * (predClose - currOpen) / currOpen
                        pricePoint = currOpen
                    else:
                        print("Select 0,1,2,3")
                        sys.exit()

                    if delta > 0.1 and prevState == 1:
                        # self.buy()
                        prevState = 0
                        print("waitCount:", waitCount, "Decision:", prevState, "startWindow:" , startWindow)
                        buySellTriggers.append([waitCount,prevState])

                        self.transaction(waitCount, date, prevState, pricePoint)

                        # logFormat(prevState, date, waitCount, currOpen, nextOpen, prevPredClose,predClose,prevClose, actualClose, delta)
                        # logData.append(str(date) + str(waitCount) + ",Buy," + str(currOpen) + "," + str(prevPredClose) + "," + str(predClose) + "," + str(delta) + "\n")
                    elif delta < -0.1 and prevState==0:
                        # self.sell()
                        prevState = 1
                        print("waitCount:", waitCount, "Decision:", prevState, "startWindow:" , startWindow)
                        buySellTriggers.append([waitCount, prevState])

                        self.transaction(waitCount, date, prevState, pricePoint)

                        # logFormat(prevState, date, waitCount, currOpen, nextOpen, prevPredClose,predClose,prevClose, actualClose, delta)
                        # logData.append(str(date) + str(waitCount)  + ",Sell," + str(currOpen) + "," + str(prevPredClose) + "," + str(predClose) + "," + str(delta) + "\n")
                    else:
                        print("waitCount:", waitCount, " Holding...", "startWindow:" , startWindow)
                        # logFormat(2, date, waitCount, currOpen, nextOpen, prevPredClose,predClose,prevClose, actualClose, delta)
                        # logData.append(str(date) + str(waitCount)  + ",Hold," + str(currOpen) + "," + str(prevPredClose) + "," + str(predClose) + "," + str(delta) + "\n")
                        pass
                    waitCount = waitCount + 1
            else:
                waitCount = waitCount + 1
                print("Iter:", waitCount, " Waiting...")
                # logData.append("Iter: " + str(waitCount) + " Waiting...\n")


        except IndexError:
            print("Time to break out...")
            self.set_flag(1)
            pass

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

def slidingWindowPred(dataset):
    y_test = dataset["y_test"]
    X_test = dataset["X_test"]
    # y_testInv, y_predInv = prediction(model, X_test, y_test,dataset)

    bt = brownHatStrategy()
    bt.set_flag(0)

    while not bt.get_flag():
        bt.next()
    # r2_score = get_accuracy(model, data)
    y_test = np.squeeze(dataset["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))

    # r2_score = calcR2(y_testInv,y_predInv)
    r2_score = calcR2(y_test, y_predArr)

    print("R2 Score:", r2_score)

    tradingStats(bt)

    fileName = f"./backtest/{ticker}/testBacktestLogs-{DECTYPE}.csv"

    with open(fileName, "w") as outfile:
        outfile.write(header)
        for row in logData:
            outfile.write(row)
    outfile.close()

    plot_graph(y_test, y_predArr, buySellTriggers, r2_score)


def tradingStats(bt):
    tradeHistory = bt.get_history()
    lastCount = len(y_predArr)
    max_drawdown = min(tradeHistory.PLpercent)

    tradeHistoryFilt = tradeHistory[tradeHistory.Iter_Count <= lastCount] # filtering out the excess trades that seem to trigger with < 70 sliding window

    equityDollars = tradeHistoryFilt.iloc[-1,8]


    numTradeStats = bt.get_numTrades()

    tradeHistoryFilt.to_csv(f"./backtest/{ticker}/testBacktestTradeLog-{DECTYPE}.csv")
    print("Equity [$]:", equityDollars)
    print("Max Drawdown:", max_drawdown)
    print("# Trades:", numTradeStats["Total"])



def get_accuracy(model, dataset):
    y_test = dataset["y_test"]
    X_test = dataset["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(dataset["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(dataset["column_scaler"]["close"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    return sqrt(mean_squared_error(y_test, y_pred))

def calcR2(y_test,y_pred):
    E1 = np.sum(np.multiply(y_test,y_pred)) # Sum(xy)
    E2 = np.sum(y_test) # Sum of all x
    E3 = np.sum(y_pred) # Sum of all y
    n = len(y_pred)
    E4 = np.sum(np.multiply(y_test,y_test)) # Sum of squares of x
    E5 = np.sum(np.multiply(y_pred, y_pred)) # sum of squares of y

    R2 = ((n*E1) - (E2 * E3))/(np.sqrt((n*E4 - np.power(E2,2)) * (n*E5 - np.power(E3,2))))
    return R2
# def predict(model, data, classification=False):
#     # retrieve the last sequence from data
#     last_sequence = data["last_sequence"][:N_STEPS]
#     print("last_sequence: ", last_sequence)
#     # retrieve the column scalers
#     column_scaler = data["column_scaler"]
#
#     # reshape the last sequence
#     last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
#
#     # expand dimension
#     last_sequence = np.expand_dims(last_sequence, axis=0)
#
#     # get the prediction (scaled from 0 to 1)
#     prediction = model.predict(last_sequence)
#     # get the price (by inverting the scaling)
#     # predicted_price = column_scaler["close"].inverse_transform(prediction)[0][0]
#     predicted_price = column_scaler["close"].inverse_transform(prediction)
#     return predicted_price

if __name__=="__main__":
    if not os.path.isdir("backtest"):
        os.mkdir("backtest")

    if not os.path.isdir(f"backtest/{ticker}"):
        os.mkdir(f"backtest/{ticker}")
# load the data
# data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
#                 feature_columns=FEATURE_COLUMNS, shuffle=False)
    data, processed = backtest.load_data(ticker, N_STEPS, True, lookup_step=LOOKUP_STEP, feature_columns=FEATURE_COLUMNS)

    # construct the model
    # model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
    #                     dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
    #
    # model_path = os.path.join("results", model_name) + ".h5"
    # model.load_weights(model_path)

    # slidingWindowPred(model, data)
    slidingWindowPred(processed)