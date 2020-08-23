from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA
import pandas as pd

# class SmaCross(Strategy):
#     def init(self):
#         price = self.data.Close
#         self.ma1 = self.I(SMA, price, 10)
#         self.ma2 = self.I(SMA, price, 20)
#
#     def next(self):
#         if crossover(self.ma1, self.ma2):
#             self.buy()
#         elif crossover(self.ma2, self.ma1):
#             self.sell()
#

class RNN(Strategy):
    def init(self):
        self.price = self.data.Close # Figure out how to run without init, this line is useless
        # self.pred = self.data.predClose

    def next(self):
        # print(self.price[-1],self.pred[-1])
        # print(self.data.Close[-1],self.data.predClose[-1])

        if (self.data.predClose[-1] > self.data.Close[-1] and self.position.size == 0) : #self.position.is_long == False
            # print('buy')

            self.buy(size = 1)
            # print(self.position.is_long)


        elif (self.data.predClose[-1] < self.data.Close[-1]) and  self.position.size > 0: #-1 index returns most recent value   #self.position.is_long== True
            # print('sell')

            self.position.close(portion = 1)
            # print(self.position.is_long)
            # self.sell()


data = pd.read_csv('./backtest/BTCUSDT/BTCUSDT_OHLCVpC.csv')
# data['Timestamp'] = pd.to_datetime(data['Timestamp'],utc='true',yearfirst = 'true')
data['Timestamp'] = pd.DatetimeIndex(data['Timestamp'], tz = 'utc')

bt = Backtest(data, RNN, commission=.001, exclusive_orders=True)
stats = bt.run()
bt.plot()
print(stats)