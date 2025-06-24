
import backtrader as bt
import datetime

# Sample data (replace with your actual data loading)
data = {
    'date': [datetime.datetime(2023, 1, i+1) for i in range(10)],
    'open': [150 + i for i in range(10)],
    'high': [150 + i + 1 for i in range(10)],
    'low': [150 + i - 1 for i in range(10)],
    'close': [150 + i + 0.5 for i in range(10)],
    'volume': [10000 + i*1000 for i in range(10)],
}

class GoldenCross(bt.Strategy):
    params = (
        ('fast', 50),
        ('slow', 200),
    )

    def __init__(self):
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)

    dataframe = bt.feeds.PandasData(dataname=pd.DataFrame(data))
    cerebro.adddata(dataframe)

    cerebro.addstrategy(GoldenCross)

    results = cerebro.run()
    print("Sharpe Ratio:", results[0].analyzers.sharpe.get_analysis()['sharperatio'])
    print("Max Drawdown:", results[0].analyzers.drawdown.get_analysis()['max']['drawdown'])

    cerebro.plot()

