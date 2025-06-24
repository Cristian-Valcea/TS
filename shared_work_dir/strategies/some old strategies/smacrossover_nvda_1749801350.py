# ===================================================================================
# This is a template for a self-contained Backtrader strategy script.
# The CodeAgent's job is to replace the placeholder class SmaCrossover:
    def __init__(self, ticker, fast_window, slow_window, start, end):
        self.ticker = ticker
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.start = start
        self.end = end

    def generate_signals(self, data):
        # Validate data
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")
        if len(data) < self.slow_window:
            raise ValueError("Data length must be greater than the slow window.")

        # Calculate moving averages
        data['FastMA'] = data['Close'].rolling(window=self.fast_window).mean()
        data['SlowMA'] = data['Close'].rolling(window=self.slow_window).mean()

        # Generate signals
        data['Signal'] = 0.0
        data['Signal'][data['FastMA'] > data['SlowMA']] = 1.0  # Buy signal
        data['Signal'][data['FastMA'] < data['SlowMA']] = -1.0 # Sell signal

        # Handle the first few rows with NaN values due to moving average calculation
        data['Signal'].fillna(0, inplace=True)

        return data
# with the actual strategy logic requested by the user.
# ===================================================================================

import backtrader as bt
import yfinance as yf
import json
import argparse
import datetime

# ===================================================================================
class SmaCrossover:
    def __init__(self, ticker, fast_window, slow_window, start, end):
        self.ticker = ticker
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.start = start
        self.end = end

    def generate_signals(self, data):
        # Validate data
        if 'Close' not in data.columns:
            raise ValueError("Data must contain a 'Close' column.")
        if len(data) < self.slow_window:
            raise ValueError("Data length must be greater than the slow window.")

        # Calculate moving averages
        data['FastMA'] = data['Close'].rolling(window=self.fast_window).mean()
        data['SlowMA'] = data['Close'].rolling(window=self.slow_window).mean()

        # Generate signals
        data['Signal'] = 0.0
        data['Signal'][data['FastMA'] > data['SlowMA']] = 1.0  # Buy signal
        data['Signal'][data['FastMA'] < data['SlowMA']] = -1.0 # Sell signal

        # Handle the first few rows with NaN values due to moving average calculation
        data['Signal'].fillna(0, inplace=True)

        return data
# ===================================================================================

# Example of what the agent should generate to replace the placeholder above:
#
# class SmaCrossover(bt.Strategy):
#     params = (('fast', 50), ('slow', 200),)
#
#     def __init__(self):
#         self.fast_sma = bt.indicators.SMA(self.data.close, period=self.p.fast)
#         self.slow_sma = bt.indicators.SMA(self.data.close, period=self.p.slow)
#         self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)
#
#     def next(self):
#         if not self.position:
#             if self.crossover > 0:
#                 self.buy()
#         elif self.crossover < 0:
#             self.close()
#
# ===================================================================================


def run_backtest(strategy_name, ticker, start_date, end_date):
    """
    Generic backtest runner function.
    """
    cerebro = bt.Cerebro(stdstats=False) # We'll add our own analyzers
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    try:
        data_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data_df.empty:
            print(json.dumps({"status": "error", "message": f"No data found for ticker {ticker}"}))
            return
        data_feed = bt.feeds.PandasData(dataname=data_df)
        cerebro.adddata(data_feed)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Failed to download or process data: {str(e)}"}))
        return

    # Dynamically get the strategy class from this script's global scope
    strategy_class = globals().get(strategy_name)
    if not strategy_class:
        print(json.dumps({"status": "error", "message": f"Strategy class '{strategy_name}' not found in the script."}))
        return
        
    cerebro.addstrategy(strategy_class)
    
    # Add standard analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    try:
        results = cerebro.run()
        strat = results[0]

        # Extract and print metrics as JSON
        sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio')
        max_drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)
        total_return = strat.analyzers.returns.get_analysis().get('rtot', 0)

        output = {
            "status": "success",
            "final_value": strat.broker.getvalue(),
            "pnl": strat.broker.getvalue() - strat.broker.startingcash,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "total_return_pct": total_return * 100
        }
        print(json.dumps(output, indent=2))

    except Exception as e:
        print(json.dumps({"status": "error", "message": f"An error occurred during backtest execution: {str(e)}"}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a Backtrader strategy.')
    parser.add_argument('--strategy_name', required=True, help='The name of the strategy class to run.')
    parser.add_argument('--ticker', required=True, help='Stock ticker symbol.')
    parser.add_argument('--start_date', required=True, help='Start date in YYYY-MM-DD format.')
    parser.add_argument('--end_date', required=True, help='End date in YYYY-MM-DD format.')
    
    args = parser.parse_args()
    run_backtest(args.strategy_name, args.ticker, args.start_date, args.end_date)
