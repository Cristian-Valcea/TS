import backtrader as bt

class EMA_RSI_Scalping(bt.Strategy):
    params = (
        ('ema_fast', 9),
        ('ema_slow', 21),
        ('rsi_period', 14),
        ('stop_loss_percent', 0.01),
        ('take_profit_percent', 0.02),
    )

    def __init__(self):
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.params.ema_slow)
        self.rsi = bt.ind.RSI(self.data.close, period=self.params.rsi_period)
        self.order = None
        self.buyprice = 0

    def next(self):
        if self.order:
            return

        if self.position.size == 0:
            if self.data.close[0] > self.ema_fast[0] and self.rsi[0] > 30:
                self.buyprice = self.data.close[0]
                self.order = self.buy()
            elif self.data.close[0] < self.ema_slow[0] and self.rsi[0] < 70:
                self.buyprice = self.data.close[0]
                self.order = self.sell()

        else:
            if self.position.size > 0:
                stop_loss = self.buyprice * (1 - self.params.stop_loss_percent)
                take_profit = self.buyprice * (1 + self.params.take_profit_percent)
                if self.data.close[0] <= stop_loss:
                    self.close()
                elif self.data.close[0] >= take_profit:
                    self.close()
            elif self.position.size < 0:
                stop_loss = self.buyprice * (1 + self.params.stop_loss_percent)
                take_profit = self.buyprice * (1 - self.params.take_profit_percent)
                if self.data.close[0] >= stop_loss:
                    self.close()
                elif self.data.close[0] <= take_profit:
                    self.close()
