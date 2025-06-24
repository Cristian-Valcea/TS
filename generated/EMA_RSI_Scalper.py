import backtrader as bt

class EMA_RSI_Scalper(bt.Strategy):
    params = (
        ('ema_period', 9),
        ('rsi_period', 14),
        ('stop_loss_pips', 5),
        ('take_profit_pips', 10),
    )

    def __init__(self):
        self.ema = bt.ind.EMA(self.data.close, period=self.params.ema_period)
        self.rsi = bt.ind.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None

    def next(self):
        if self.order:
            return

        if self.data.close[0] > self.ema[0] and self.rsi[0] > 30 and not self.position:
            self.order = self.buy()
        elif self.data.close[0] < self.ema[0] and self.rsi[0] < 70 and self.position:
            self.order = self.sell()

        if self.position:
            if self.position.size > 0:
                stop_loss = self.position.price - self.params.stop_loss_pips * self.data.close.pip_value
                take_profit = self.position.price + self.params.take_profit_pips * self.data.close.pip_value
                if self.data.close[0] < stop_loss:
                    self.close()
                elif self.data.close[0] > take_profit:
                    self.close()


