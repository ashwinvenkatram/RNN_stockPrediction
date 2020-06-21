from stock_prediction import create_model, load_data, np
from parameters import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import pickle
import numpy as np
from math import sqrt
import matplotlib.animation as animation

import sys

from time import time


global model
global slideWindowX, slideWindowY
global data

EMAList1 = np.zeros(1)
EMAList2 = np.zeros(1)
EMAList3 = np.zeros(1)

x_TEMA = np.zeros(EMA_SAMPLE_LENGTH-1)
TEMAList = np.zeros(EMA_SAMPLE_LENGTH-1)
x_test = []
y_test = []
x_testPred = []
y_predArr = []

fig = plt.figure()
ax = fig.add_subplot(aspect='equal', autoscale_on=True)
ax.grid()
ax.set_autoscale_on(True)
ax.autoscale_view(True,True,True)

lineTest, = ax.plot(x_test,y_test, c='b',marker='.')#,marker="o", s=3)
linePred, = ax.plot(x_testPred,y_predArr, c='r',marker='.')#, marker="o", s=3)
lineTEMA, = ax.plot(x_TEMA,TEMAList, c='m',marker='.')

# def init():
#     global startWindow, y_predArr
#
#     """initialize animation"""
#     lineTest.set_offsets(np.c_[x_test, y_test])
#     linePred.set_offsets(np.c_[x_testPred, y_predArr])
#
#     startWindow = 0
#     y_predArr = []
#     return lineTest, linePred


def animate(i):
    """perform animation step"""
    global x_TEMA
    model, endCheck = reinforcement()

    y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(data["y_test"], axis=0)))
    x_test = [x for x in range(len(y_test))]
    x_testPred = [tp for tp in range(len(y_predArr))]

    print(np.shape(x_test),np.shape(y_test))
    print(np.shape(x_testPred),np.shape(y_predArr))

    if len(y_predArr) >= EMA_SAMPLE_LENGTH:
        TripleEMA(y_predArr)
        x_TEMA = [tp for tp in range(len(TEMAList))]
        print(np.shape(x_TEMA), np.shape(TEMAList))

    lineTest.set_data(x_test, y_test)
    linePred.set_data(x_testPred, y_predArr)
    lineTEMA.set_data(x_TEMA, TEMAList)



    # linePred.set_offsets(np.c_[x_testPred, y_predArr])
    # lineTest.set_offsets(x_test, y_test)
    # linePred.set_offsets(x_testPred, y_predArr)
    # lineTest.set_data(x_test, y_test)
    # linePred.set_data(x_testPred, y_predArr)

    if endCheck:
        # write configs to yaml
        # write plot to file
        pickle.dump(fig, open(f"./reinforcement/{ticker}/results/" + 'reinforcement' + '.pickle', 'wb'))
        plt.savefig(f"reinforcement/{ticker}/results/" + 'reinforcement' + '.png')

        # write x_test, y_test, y_pred, tripleEMA to file for nicer plotting in excel/matlab
        with open(f"./reinforcement/{ticker}/logs/logfile_test.csv","a") as outfile:
            header = "x_test, y_test" + "\n"
            outfile.write(header)

            for row in range(len(x_test)):
                rowData = f"{x_test[row]}, {y_test[row]}" + "\n"

                outfile.write(rowData)
        outfile.close()

        with open(f"./reinforcement/{ticker}/logs/logfile_pred.csv","a") as outfile:
            header = "x_testPred, y_predArr" + "\n"
            outfile.write(header)

            for row in range(len(x_testPred)):
                rowData = f"{x_testPred[row]}, {y_predArr[row]}" + "\n"

                outfile.write(rowData)
        outfile.close()

        with open(f"./reinforcement/{ticker}/logs/logfile_TEMA.csv","a") as outfile:
            header = "x_TEMA, TEMAList" + "\n"
            outfile.write(header)

            for row in range(len(x_TEMA)):
                rowData = f"{x_TEMA[row]}, {TEMAList[row]}" + "\n"
                outfile.write(rowData)
        outfile.close()

        sys.exit()
    return lineTest, linePred



def reinforcement():
    global slideWindowX,slideWindowY, startWindow
    global model

    # if startWindow % 70 == 0:
    #     y_predArr = []

    slideWindowX = data["X_test"][startWindow:startWindow + 70]
    slideWindowY = data["y_test"][startWindow:startWindow + 70]

    # checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True,
    #                                save_best_only=True, verbose=1)
    # tensorboard = TensorBoard(log_dir=os.path.join("logs/", model_name))

    if len(slideWindowX) < 70:
        print("end of dataset reached\n")
        endCheck = 1
    else:
        model.fit(slideWindowX, slideWindowY,
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  verbose=1)

        y_pred = model.predict(slideWindowX)
        y_pred = data["column_scaler"]["close"].inverse_transform(y_pred)[0][0]

        y_predArr.append(y_pred)

        startWindow = startWindow + 1

        try:
            slideWindowX = data["X_test"][startWindow:startWindow + 70]
            slideWindowY = data["y_test"][startWindow:startWindow + 70]
            endCheck = 0
        except IndexError:
            print("end of dataset reached\n")
            endCheck = 1

    return model, endCheck

def calcEMA(src,lookback):
    multiplier = 2 / (EMA_SAMPLE_LENGTH + 1)
    if lookback == EMA_SAMPLE_LENGTH-1: # 9th value before today when lookback is 8. 0 based counting
        EMA_now = (EMA_SAMPLE_LENGTH - np.sum(src[-1:-10])) / EMA_SAMPLE_LENGTH
    else:
        try:
            lookback = lookback + 1
            EMA_now = src[-1-lookback] * multiplier + calcEMA(src,lookback) * (1 - multiplier)
        except IndexError:
            EMA_now = (EMA_SAMPLE_LENGTH - np.sum(src[-1:-10])) / EMA_SAMPLE_LENGTH

    return EMA_now

def TripleEMA(close):
    global EMAList1, EMAList2, EMAList3, TEMAList
    EMA1 = calcEMA(close,0)
    EMAList1 = np.append(EMAList1,EMA1)

    EMA2 = calcEMA(EMAList1,0)
    EMAList2 = np.append(EMAList2,EMA2)

    EMA3 = calcEMA(EMAList2,0)
    EMAList3 = np.append(EMAList3,EMA3)

    TEMA = 3*EMA1 - 3*EMA2 + EMA3
    TEMAList = np.append(TEMAList,TEMA)


# def plot_graph(y_pred, y_test, r2_score):
#     OUTPUT_PATH = "./results/" + model_name
#
#     figTest = plt.figure()
#     plt.style.use('seaborn-whitegrid')
#     x_val = [i for i in range(len(y_pred))]
#     plt.scatter(x_val,y_test[:], c='b',marker="o", s=10)
#     plt.scatter(x_val,y_pred[:], c='r', marker="o", s=10)
#     # plt.plot(y_test[:], 'b')
#     # plt.plot(y_pred[:], 'r')
#     plt.xlabel("Days")
#     plt.ylabel("Price")
#     plt.legend(["Actual Price", "Predicted Price"])
#     plt.title("Price Prediction Accuracy, R2: " + str(r2_score))
#     pickle.dump(figTest, open(OUTPUT_PATH +'_pred_vs_real' + '.pickle', 'wb'))
#     plt.savefig(OUTPUT_PATH +'_pred_vs_real' + '.png')
#     plt.show()


# def prediction(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     print(len(X_test), len(y_pred))
#     y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
#     y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
#
#     return y_test, y_pred

# def slidingWindowPred(model, data):
#     y_test = data["y_test"]
#     X_test = data["X_test"]
#     y_testInv, y_predInv = prediction(model, X_test, y_test)
#
#     r2_score = get_accuracy(model, data)
#     print("R2 Score:", r2_score)
#     plot_graph(y_predInv, y_testInv, r2_score)

# def get_accuracy(model, data):
    # y_test = data["y_test"]
    # X_test = data["X_test"]
    # y_pred = model.predict(X_test)
    # y_test = np.squeeze(data["column_scaler"]["close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    # y_pred = np.squeeze(data["column_scaler"]["close"].inverse_transform(y_pred))
    # y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_pred[LOOKUP_STEP:]))
    # y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-LOOKUP_STEP], y_test[LOOKUP_STEP:]))
    # return sqrt(mean_squared_error(y_test, y_pred))


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


# load the data
if __name__ =="__main__":
    if not os.path.isdir("reinforcement"):
        os.mkdir("reinforcement")

    if not os.path.isdir(f"reinforcement/{ticker}/"):
        os.mkdir(f"reinforcement/{ticker}/")

    if not os.path.isdir(f"reinforcement/{ticker}/logs"):
        os.mkdir(f"reinforcement/{ticker}/logs")

    if not os.path.isdir(f"reinforcement/{ticker}/results"):
        os.mkdir(f"reinforcement/{ticker}/results")

    data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                    feature_columns=FEATURE_COLUMNS, shuffle=False)

    # construct the model
    model = create_model(N_STEPS, loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                         dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)

    # Initialization: Base model loaded after training
    model_path = os.path.join("results", model_name) + ".h5"
    model.load_weights(model_path)


    dt = 1. / 30  # 30fps
    t0 = time()
    animate(0)
    t1 = time()
    interval = 1000 * dt - (t1 - t0)

    ani = animation.FuncAnimation(fig, animate, blit=True)
    ax.relim()
    # ax.set_xlim(0, 3000)
    # ax.set_ylim(0, 60)
    ax.autoscale(True)
    ax.autoscale_view(True, True, True)
    plt.draw()
    plt.show()

    # # evaluate the model
    # mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
    # # calculate the mean absolute error (inverse scaling)
    #
    # mean_square_error = data["column_scaler"]["adjclose"].inverse_transform([[mse]])[0][0]
    # mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
    # print("Mean Absolute Error:", mean_absolute_error)
    # print("Mean Square Error:", mean_square_error)

    # predict the future price
    # future_price = predict(model, data)
    # print(future_price)
    # print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    # r2_score = get_accuracy(model, data)
    # print("R2 Score:", r2_score)
    # plot_graph(model, data, r2_score)
