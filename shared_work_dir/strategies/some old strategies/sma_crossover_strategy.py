import backtrader as bt
import yfinance as yf
import json
import os

class SmaCrossover(bt.Strategy):
    params = (
        ('fast', 50),
        ('slow', 200),
    )

    def __init__(self):
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position:  # Not in the market
            if self.crossover > 0:  # Fast SMA crosses above Slow SMA
                self.buy()
        else:  # In the market
            if self.crossover < 0:  # Fast SMA crosses below Slow SMA
                self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    data_df = yf.download('NVDA', start='2022-01-01', end='2023-12-31')
    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(SmaCrossover)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    results = cerebro.run()
    pnl = results[0].broker.getvalue() - results[0].broker.startingcash
    sharpe = results[0].analyzers.sharpe.get_analysis()[\'sharperatio\']
    max_drawdown = results[0].analyzers.drawdown.get_analysis()[\'max\'][\'drawdown\']
    output = {
        "final_value": results[0].broker.getvalue(),
        "pnl": pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown
    }
    print(json.dumps(output, indent=2))

