import ccxt
from datetime import datetime
import pandas as pd

TICKER = 'BTC/USDT' # market id definition
INTERVAL = '1h'

def importHistorialData(binanceObj, tickerSymbol, timeInterval):
    candles = binanceObj.fetch_ohlcv(tickerSymbol, timeInterval)

    dates = []
    open_data = []
    high_data = []
    low_data = []
    close_data = []
    for candle in candles:
        dates.append(datetime.fromtimestamp(candle[0] / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f'))
        open_data.append(candle[1])
        high_data.append(candle[2])
        low_data.append(candle[3])
        close_data.append(candle[4])

    result = {'dates': dates, 'open': open_data, 'close': close_data, 'high': high_data, 'low': low_data}

    processed = pd.DataFrame.from_dict(result)

    return processed

def cryptoData(exchangeId,tickerSymbol, timeInterval):
    try:
        binanceObj = getattr(ccxt,exchangeId)
        marketsBinance = binanceObj.load_markets()

        if tickerSymbol in marketsBinance.keys():
            # tickerMetaData = marketsBinance[TICKER]
            data = importHistorialData(binanceObj, tickerSymbol, timeInterval)
            return data
        else:
            print(TICKER, ' not in ', binanceObj.id)
            # returns empty dataframe
            return pd.DataFrame()
        
    except AttributeError:
        print(exchangeId, '. Exchange not present in CCXT')
        # returns empty dataframe
        return pd.DataFrame()

if __name__=='__main__':
    binanceObj = ccxt.binance()

    marketsBinance = binanceObj.load_markets()

    if TICKER in marketsBinance.keys():
        # tickerMetaData = marketsBinance[TICKER]
        data = importHistorialData(binanceObj, TICKER, INTERVAL)

    else:
        print(TICKER, ' not in ', binanceObj.id)
        print(pd.DataFrame())




