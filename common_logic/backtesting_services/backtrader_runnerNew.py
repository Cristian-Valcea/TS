# common_logic/backtesting_services/backtrader_runner.py

# --- Robust Pathing Setup ---
import sys
from pathlib import Path

# Get the directory of the current script (backtesting_services)
SCRIPT_DIR = Path(__file__).resolve().parent
# CORRECTED: Go up TWO levels to reach the project root (common_logic -> TradingSystem)
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Add the project root to the Python path so it can find the 'agents' and other modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT)) # Insert at the front to ensure it's checked first
# --- End Pathing Setup ---

import backtrader as bt
import pandas as pd
import numpy as np
import importlib.util
import os
import logging
import json
from typing import Dict, Any
import argparse

# These imports will now work correctly
from agents.utils.timeframe_utils import parse_timeframe_string
from agents.utils.chart_util import plot_strategy_behavior

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class PandasDataFeed(bt.feeds.PandasData):
    """Custom PandasData feed to map standard yfinance/CSV column names."""
    lines = ('open', 'high', 'low', 'close', 'volume')
    params = (('datetime', None), ('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close'), ('volume', 'Volume'), ('openinterest', -1))


class DetailedTradeAnalyzer(bt.Analyzer):
    """Custom analyzer to store detailed information about each closed trade."""
    def __init__(self):
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'entry_price': trade.price,
                'exit_price': trade.close,
                'dt_opened': bt.num2date(trade.dtopen).isoformat(),
                'dt_closed': bt.num2date(trade.dtclose).isoformat(),
            })

    def get_analysis(self) -> Dict[str, Any]:
        return {'closed_trades': self.trades}


def load_strategy_class_from_file(strategy_file_path: str, strategy_class_name: str) -> type:
    """Dynamically loads a strategy class from a Python file."""
    if not os.path.exists(strategy_file_path):
        raise FileNotFoundError(f"Strategy file not found: {strategy_file_path}")
    module_name = f"strategy_module_{Path(strategy_file_path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file_path)
    if spec is None:
        raise ImportError(f"Could not load spec from {strategy_file_path}")
    strategy_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = strategy_module
    spec.loader.exec_module(strategy_module)
    if not hasattr(strategy_module, strategy_class_name):
        raise AttributeError(f"Strategy class '{strategy_class_name}' not found in {strategy_file_path}")
    return getattr(strategy_module, strategy_class_name)


def run_backtest(strategy_file_path: str, data_file_path: str, strategy_class_name: str, **strategy_kwargs: Any) -> Dict[str, Any]:
    """Runs a Backtrader backtest for a given strategy and data."""
    logger.info(f"Starting backtest for '{strategy_class_name}' with data '{data_file_path}'")
    if strategy_kwargs:
        logger.info(f"Strategy Parameters: {strategy_kwargs}")

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # Simplified and robust data loading
    df = pd.read_csv(data_file_path, index_col='Date', parse_dates=True)
    data_feed = PandasDataFeed(dataname=df)
    cerebro.adddata(data_feed)

    StrategyClass = load_strategy_class_from_file(strategy_file_path, strategy_class_name)
    cerebro.addstrategy(StrategyClass, **strategy_kwargs)

    # Add standard analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(DetailedTradeAnalyzer, _name='detailed_trades')

    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    sharpe = strat.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0.0)
    cagr = strat.analyzers.returns.get_analysis().get('rnorm100', 0.0)
    drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0.0)

    # This can be expanded greatly
    metrics = {
        'status': 'completed',
        'final_portfolio_value': final_value,
        'sharpe_ratio': sharpe if sharpe is not None else 0.0,
        'cagr_pct': cagr if cagr is not None else 0.0,
        'max_drawdown_pct': drawdown if drawdown is not None else 0.0,
        'strategy_params_used': strategy_kwargs,
        'trades': strat.analyzers.detailed_trades.get_analysis()['closed_trades']
    }
    return metrics


if __name__ == '__main__':
    # This is the ONLY main block. It's designed to be called by other scripts.
    parser = argparse.ArgumentParser(description="Run a single Backtrader backtest.")
    parser.add_argument('--strategy-file', required=True, help="Absolute path to the strategy Python file.")
    parser.add_argument('--data-file', required=True, help="Absolute path to the CSV data file.")
    # ADDED: Argument for the strategy class name
    parser.add_argument('--strategy-class-name', required=True, help="The name of the strategy class inside the file.")
    parser.add_argument('--strategy-params', default="{}", help="JSON string of strategy parameters.")
    
    args = parser.parse_args()

    try:
        # --- THIS IS THE CORRECTED SECTION ---
        # Convert relative paths from arguments to absolute paths
        # by joining them with the project root directory.
        absolute_strategy_path = PROJECT_ROOT / args.strategy_file
        absolute_data_path = PROJECT_ROOT / args.data_file
        # --- END CORRECTION ---

        # --- VERIFICATION STEP ---
        # This print statement is NEW. If you see this, the correct code is running.
        print(f"\n--- DEBUG: VERIFYING PATHS ---")
        print(f"Project Root: {PROJECT_ROOT}")
        print(f"Absolute Strategy Path: {absolute_strategy_path}")
        print(f"Absolute Data Path: {absolute_data_path}")
        print(f"File Exists? {absolute_data_path.exists()}")
        print(f"--- END DEBUG ---\n")
        # --- END VERIFICATION ---
   
        params = json.loads(args.strategy_params)
        
        # Call the backtest function with all required arguments
        metrics = run_backtest(
            strategy_file_path=args.strategy_file,
            data_file_path=args.data_file,
            strategy_class_name=args.strategy_class_name, # Pass the new argument
            **params
        )
        
        # Print the results as a single JSON string for the calling script
        print(json.dumps(metrics, indent=2))
        
    except Exception as e:
        logger.error(f"Backtest run failed with error: {e}", exc_info=True)
        # Print an error JSON
        print(json.dumps({
            "status": "error",
            "error_message": str(e),
            "error_type": type(e).__name__
        }))
        # CORRECTED: Exit with a non-zero code to signal failure
        sys.exit(1)
