import pickle
import os

iter_changes = "dropout_layers_0.4_0.4"
OUT_PATH = "./results/lstm_best_Adam#neuron124/"
OUTPUT_PATH = OUT_PATH +iter_changes+"/"

params = {
    "batch_size": 20,  # 20<16<10, 25 was a bust
    "epochs": 3,
    "lr": 0.00010000,
    "time_steps": 60,
    "num_layers": 1
}

BATCH_SIZE = params["batch_size"]


mainDir = "E:/ashwinWork/RNN_StockPrediction/stock-prediction-BrownHatMod/reinforcement/MU/results/"
# figTrain = 'OHLCO-2020-06-16_HTZ-huber_loss-adam-CuDNNLSTM-seq-70-step-1-layers-3-units-256_LossEpoch.pickle'
figTest = 'reinforcement.pickle'


# figTrain = pickle.load(open(mainDir + figTrain, 'rb'))
figTest = pickle.load(open(mainDir + figTest, 'rb'))
# figTrain.show()
figTest.show()

# Prevent program from closing graphs
username = input("")