
import backtrader as bt
class DummyStrategy(bt.Strategy): # Must match strategy_class_name
    params = (('sma_period', 20), ('rsi_period', 14), ('printlog', False))
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0].close, period=self.params.sma_period)
        self.rsi = bt.indicators.RSI(self.datas[0].close, period=self.params.rsi_period)
        self.order = None
    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]: return
        if order.status in [order.Completed]:
            if order.isbuy(): self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell(): self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
        self.order = None
    def next(self):
        if self.order: return
        if not self.position:
            if self.datas[0].close[0] > self.sma[0] and self.rsi[0] < 30: # Example different condition
                self.order = self.buy()
        else:
            if self.datas[0].close[0] < self.sma[0] or self.rsi[0] > 70:
                self.order = self.sell()
    def stop(self):
        if self.params.printlog: self.log(f'(SMA {self.params.sma_period} RSI {self.params.rsi_period}) Ending Value {self.broker.getvalue():.2f}')
