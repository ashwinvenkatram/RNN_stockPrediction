import os
from datetime import date, timedelta
# from tensorflow.keras.layers import LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

# Window size or the sequence length
N_STEPS = 70
# Lookup step, 1 is the next day
LOOKUP_STEP = 1

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.3
# features to use
# FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
FEATURE_COLUMNS = ["open", "high", "low", "close","nextopen"]
# date now
date_now = date.today()#time.strftime("%Y-%m-%d")

### yfinance dateframe
PERIOD = "10y" # Anything that is not max will use start and end date as string
               # Valid periods are: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

INTERVAL = "1h" #Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo

if PERIOD != "max":
    ENDDATE = date_now
    if INTERVAL in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h']:
        STARTDATE = ENDDATE - timedelta(729) # Only last 730 days are available on smaller intervals
    else:
        STARTDATE = "2017-01-01"
### model parameters


N_LAYERS = 3 # 3 works for most. Attempting 10 for TVIX
# LSTM cell
CELL = CuDNNLSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 400

# Ticker Information
MOD_SETTING = f"OHLCO-{TEST_SIZE}-"
ticker = "TVIX" # HTZ, IZEA, DAL, NE
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
# model name to save, making it as unique as possible based on parameters
date_old = "2020-06-19"
# model_name = f"{MOD_SETTING}-{date_old}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
model_name = f"{MOD_SETTING}-{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
# model_name = f"{MOD_SETTING}-2020-06-17_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"


# EMA Settings
EMA_SAMPLE_LENGTH = 9
startWindow = 0

if BIDIRECTIONAL:
    model_name += "-b"