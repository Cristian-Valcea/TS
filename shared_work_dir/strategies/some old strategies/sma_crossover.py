import backtrader as bt
import pandas as pd

class SmaCrossover():
    params = (
        ('fast', 50),
        ('slow', 200),
    )

    def __init__(self):
        self.fast_sma = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.slow_sma = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position:  # Not in the market
            if self.crossover > 0:  # Fast SMA crosses above Slow SMA
                self.buy()
        else:  # In the market
            if self.crossover < 0:  # Fast SMA crosses below Slow SMA
                self.close()

def main():
    SmaCrossover();
if __name__ == "__main__":
    main()
