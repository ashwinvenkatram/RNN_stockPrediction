import numpy as np
import plotly.graph_objects as go
import pandas as pd

import sys
def plot(df):
    fig = go.Figure(data=[go.Ohlc(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close']
                                 )])

    fig.show()

def rm_nan(dataset):
    OHEn = np.isnan(dataset)
    dataset= dataset[~np.isnan(dataset)]
    return dataset, OHEn

def correction(dataset, indexRm, labelname):
    dataset = dataset.reset_index(name=labelname)
    dataset = dataset.drop(labels=indexRm)
    dataset = dataset[labelname].values
    return dataset

def compute_heikenAshi(df):
    column_names = []

    df_HA = pd.DataFrame(columns=column_names)

    openshift = df['open'].shift(1)
    closeshift = df['close'].shift(1)

    openshift, OHEn_openshift = rm_nan(openshift)
    closeshift, OHEn_closeshift = rm_nan(closeshift)

    indexList = np.where(OHEn_openshift == True)

    indexRm = []
    for i in range(len(indexList)):
        indexRm.append(indexList[i][0])

    open = (openshift + closeshift)/2
    open = open.to_numpy()
    close = (df['open'] + df['high'] + df['low'] + df['close'])/4
    high = df[['open','high','close']].max(axis=1)
    low = df[['open','low','close']].min(axis=1)

    close = correction(close, indexRm, 'close')
    high = correction(high, indexRm, 'high')
    low = correction(low, indexRm, 'low')
    dates = correction(df['date'], indexRm, 'dates')
    volume = correction(df['volume'], indexRm, 'volume')

    # print('open', open.shape)
    # print('high', high.shape)
    # print('low', low.shape)
    # print('close', close.shape)
    # print('date', dates.shape)

    # df_HA['date'] = dates
    # df_HA['open'] = open
    # df_HA['high'] = high
    # df_HA['low'] = low
    # df_HA['close'] = close
    df_HA.insert(column='date', value=dates, loc=len(df_HA.columns))
    df_HA.insert(column='open', value=open, loc=len(df_HA.columns))
    df_HA.insert(column='high', value=high, loc=len(df_HA.columns))
    df_HA.insert(column='low', value=low, loc=len(df_HA.columns))
    df_HA.insert(column='close', value=close, loc=len(df_HA.columns))
    df_HA.insert(column='volume', value=volume, loc=len(df_HA.columns))

    df_HA = df_HA[~df_HA.isin([np.nan, np.inf, -np.inf]).any(1)]
    df_HA.dropna(inplace=True)

    df_HA.insert(column='trend', value=np.where(close > open, 1, 0), loc=len(df_HA.columns))


    pd.set_option('display.max_rows', None)

    return df_HA