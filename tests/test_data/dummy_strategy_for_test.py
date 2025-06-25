import backtrader as bt

class DummySmaCross(bt.Strategy):
    params = (('fast_sma_period', 10), ('slow_sma_period', 30),)

    def __init__(self):
        self.fast_sma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.p.fast_sma_period
        )
        self.slow_sma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.p.slow_sma_period
        )
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

class AnotherStrategy(bt.Strategy): # For testing loading different classes
    def __init__(self):
        pass
    def next(self):
        pass
