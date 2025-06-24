"""A strategy for trading based on a Golden Crossover."""
import backtrader as bt

class GoldenCross(bt.Strategy):
    """Implements the Golden Cross strategy."""
    params = (
        ('fast_ma', 50),
        ('slow_ma', 200),
    )

    def __init__(self):
        super().__init__()
        # pylint: disable=no-member
        self.fast_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.fast_ma
        )
        # pylint: disable=no-member
        self.slow_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.slow_ma
        )
        # pylint: disable=no-member
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        """Defines the logic for each bar."""
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()