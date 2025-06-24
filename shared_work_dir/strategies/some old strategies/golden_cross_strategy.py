
import backtrader as bt
import yfinance as yf
import pandas as pd

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

    data = get_data('AAPL', '2022-01-01', '2023-12-31')
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = (data['Date'] - data['Date'].min()).dt.days #Convert to days since start

    data = bt.feeds.PandasData(dataname=data, datetime='Date', open='Open', high='High', low='Low', close='Close', volume='Volume', openinterest=-1)
    cerebro.adddata(data)


    cerebro.addstrategy(GoldenCross)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    results = cerebro.run()
    strat = results[0]

    print(f'Sharpe Ratio: {strat.analyzers.sharpe.get_analysis()["sharperatio"]:.2f}')
    print(f'Max Drawdown: {strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]:.2f}%')

    cerebro.plot()


def get_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data.reset_index()
    return data

