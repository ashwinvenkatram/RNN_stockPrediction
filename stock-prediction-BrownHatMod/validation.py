from stock_prediction import create_model, load_data, np
from parameters import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

import pickle

import numpy as np
import pandas as pd
from math import sqrt

import sys
import os

def plot_graph(y_pred, y_test, r2_score):
    OUTPUT_PATH = "./results/" + model_name

    figTest = plt.figure()
    plt.style.use('seaborn-whitegrid')
    x_val = [i for i in range(len(y_pred))]
    plt.scatter(x_val,y_test[:], c='b',marker="o", s=3)
    plt.scatter(x_val,y_pred[:], c='r', marker="o", s=3)
    plt.plot(y_test[:], 'b')
    plt.plot(y_pred[:], 'r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.title("Validation Set - Price Prediction Accuracy, R2: " + str(r2_score))

    # with open(f"results/prediction_{model_name}.csv",'w') as outfile:
    #     header = "x_val,y_pred\n"
    #     outfile.write(header)
    #     for i in range(len(x_val)):
    #         outfile.write(f"{x_val[i]},{y_pred[i]}"+"\n")
    # outfile.close()

    pickle.dump(figTest, open(OUTPUT_PATH +'_validation_pred_vs_real' + '.pickle', 'wb'))
    plt.savefig(OUTPUT_PATH +'_validation_pred_vs_real' + '.png')
    plt.show()


def prediction(model, X_test, y_test,dataset):
    y_pred = model.predict(X_test)

    # print(len(X_test), len(y_pred))
    # y_test = np.squeeze(dataset["column_scaler"]["Close_logDiff"].inverse_transform(np.expand_dims(y_test, axis=0)))
    # y_pred = np.squeeze(dataset["column_scaler"]["Close_logDiff"].inverse_transform(y_pred))

    # invert close normalization process
    # load static minmaxscalar from file
    column_scaler = pickle.load(open(os.path.join("results", 'scalerTrain-' + model_name + '.pickle'), 'rb'))
    scaler = column_scaler["Close_logDiff"]

    y_test = np.squeeze(scaler.inverse_transform(np.expand_dims(y_test, axis=1)))
    y_pred = np.squeeze(scaler.inverse_transform(y_pred))

    # Shape of y_test: (5249,)
    # Shape of y_pred: (5249, 1)
    # array([[0.557144 ],
    #        [0.557144 ],
    #        [0.557144 ],
    #        ...,
    #        [0.557144 ],
    #        [0.557144 ],
    #        [0.5571439]], dtype=float32)
    # Hence expand dimensions along column => axis = 1
    # y_test = np.squeeze(dataset["column_scaler"]["Close_logDiff"].inverse_transform(np.expand_dims(y_test, axis=1)))
    # y_pred = np.squeeze(dataset["column_scaler"]["Close_logDiff"].inverse_transform(y_pred))

    # df['Close_logDiff'] = np.log(df['Close'.lower()]) - np.log(df['Close'.lower()]).shift(1)

    # non-stationary conversion
    # y_pred = np.squeeze(y_pred)
    y_test = np.exp(y_test + np.log(dataset["y_val_logDiffClose"]))
    y_pred = np.exp(y_pred + np.log(dataset["y_val_logDiffClose"]))

    # Removing NaN
    # y_test = y_test[~np.isnan(y_test)]
    # y_pred = y_pred[~np.isnan(y_pred)]

    return y_test, y_pred

def testSetPred(model, dataset, valDataset):
    y_test = dataset["y_val"]
    X_test = dataset["X_val"]
    y_testOriClose = valDataset['Close'].values # shift not necessary

    # print(X_test.shape)
    y_testInv, y_predInv = prediction(model, X_test, y_test,dataset)

    # r2_score = get_accuracy(model, data)
    # r2_score = calcR2(y_testInv,y_predInv)

    # Comparing pred with original untouched close data
    r2_score = calcR2(y_testOriClose, y_predInv)
    print("R2 Score:", r2_score)

    # saving OHLCV and predClose to csv
    compDF = pd.DataFrame()
    compDF.insert(column='close', value=y_testOriClose, loc=len(compDF.columns))
    compDF.insert(column='y_predInv', value=y_predInv, loc=len(compDF.columns))
    # OHLCV, nextPredClose for purpose of backtesting

    valDataset['predClose'] = compDF['y_predInv'].values
    valDataset['predClose'] = valDataset['predClose'].shift(-1) # For Pranav to compare each close with corresponding next close

    valDataset.dropna(inplace=True)

    filename = './backtest/BTCUSDT/BTCUSDT_validation_OHLCVpC.csv'
    valDataset.to_csv(filename)

    # plot_graph(y_predInv, y_testInv, r2_score)
    plot_graph(compDF['y_predInv'].values, compDF['close'].values, r2_score)

# def get_accuracy(model, dataset):
#     y_test = dataset["y_val"]
#     X_test = dataset["X_val"]
#     y_pred = model.predict(X_test)
#     # y_test = np.squeeze(dataset["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
#     # y_pred = np.squeeze(dataset["column_scaler"]["close"].inverse_transform(y_pred))
#     y_test = np.squeeze(dataset["column_scaler"]["Close_logDiff"].inverse_transform(np.expand_dims(y_test, axis=0)))
#     y_pred = np.squeeze(dataset["column_scaler"]["Close_logDiff"].inverse_transform(y_pred))
#
#     y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
#     y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
#     return sqrt(mean_squared_error(y_test, y_pred))

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
    # load the data
    data, testDataset, valDataset = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                    feature_columns=FEATURE_COLUMNS, shuffle=False)


    # construct the model
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                        dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)

    testSetPred(model, data, valDataset)