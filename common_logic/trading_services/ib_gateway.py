# TraderAgent/ib_gateway.py

from ib_insync import IB, Stock, util

class IBGateway:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    def connect(self):
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        print(f"✅ Connected to Interactive Brokers at {self.host}:{self.port}")

    def disconnect(self):
        self.ib.disconnect()
        print("🔌 Disconnected from IB")

    def buy_stock(self, symbol: str, quantity: int):
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        order = self.ib.marketOrder('BUY', quantity)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)  # Wait for order processing
        return trade

    def sell_stock(self, symbol: str, quantity: int):
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        order = self.ib.marketOrder('SELL', quantity)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        return trade

