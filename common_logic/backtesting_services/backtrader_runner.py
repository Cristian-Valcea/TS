# common_logic/backtesting_services/backtrader_runner.py

import sys
from pathlib import Path

# --- Robust Pathing Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Pathing Setup ---

import backtrader as bt
import pandas as pd
import importlib.util
import os
import logging
import json
import argparse
from typing import Dict, Any

from agents.utils.timeframe_utils import parse_timeframe_string
from agents.utils.chart_util import plot_strategy_behavior

logger = logging.getLogger(__name__)

class PandasDataFeed(bt.feeds.PandasData):
    lines = ('open', 'high', 'low', 'close', 'volume')
    params = (('datetime', None), ('open', 'Open'), ('high', 'High'), ('low', 'Low'), ('close', 'Close'), ('volume', 'Volume'), ('openinterest', -1))

def load_strategy_class_from_file(strategy_file_path: str, strategy_class_name: str) -> type:
    if not os.path.exists(strategy_file_path): raise FileNotFoundError(f"Strategy file not found: {strategy_file_path}")
    module_name = f"strategy_module_{Path(strategy_file_path).stem}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
    spec = importlib.util.spec_from_file_location(module_name, strategy_file_path)
    if spec is None or spec.loader is None: raise ImportError(f"Could not load spec from {strategy_file_path}")
    strategy_module = importlib.util.module_from_spec(spec); sys.modules[module_name] = strategy_module; spec.loader.exec_module(strategy_module)
    if not hasattr(strategy_module, strategy_class_name): raise AttributeError(f"Strategy class '{strategy_class_name}' not found in {strategy_file_path}")
    return getattr(strategy_module, strategy_class_name)

def run_backtest(
    strategy_file_path: str,
    data_file_path: str,
    strategy_class_name: str,
    initial_cash: float,      # It takes these arguments...
    commission_pct: float,    **strategy_kwargs: Any
) -> Dict[str, Any]:
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission_pct)

    # --- THIS IS THE FINAL, CORRECTED DATA LOADING LOGIC ---
    try:
        # Step 1: Load the CSV, skipping the first 3 junk rows.
        # Step 2: Manually assign the correct column names.
        # Step 3: Tell pandas the first column (index_col=0) is the index and contains dates.
        df = pd.read_csv(
            data_file_path,
            skiprows=3,
            header=None, # The file has no valid header row now
            names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
            index_col=0,
            parse_dates=True
        )
        
        # Step 4: Sanitize the data to ensure it's all numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
    except Exception as e:
        logger.error(f"Error processing data file {data_file_path}: {e}", exc_info=True)
        raise
    # --- END OF CORRECTED DATA LOADING ---

    data_feed = PandasDataFeed(dataname=df)
    cerebro.adddata(data_feed)
    
    StrategyClass = load_strategy_class_from_file(strategy_file_path, strategy_class_name)
    cerebro.addstrategy(StrategyClass, **strategy_kwargs)
    
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    
    try:
        results = cerebro.run()
        strat = results[0]
    except Exception as e:
        logger.error(f"Cerebro run failed: {e}", exc_info=True)
        return {"status": "error", "error_message": f"Cerebro run failed: {e}"}

    sharpe = strat.analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0.0)
    
    metrics = {
        'status': 'completed',
        'final_portfolio_value': cerebro.broker.getvalue(),
        'sharpe_ratio': sharpe if sharpe is not None else 0.0,
        'strategy_params_used': strategy_kwargs
    }
    return metrics

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    parser = argparse.ArgumentParser(description="Run a single Backtrader backtest.")
    parser.add_argument('--strategy-file', required=True)
    parser.add_argument('--data-file', required=True)
    parser.add_argument('--strategy-class-name', required=True)
    parser.add_argument('--strategy-params', default="{}")
    # We can add arguments for backtester settings here if needed
    parser.add_argument('--initial-cash', type=float, default=100000.0)
    parser.add_argument('--commission', type=float, default=0.001)
    
    args = parser.parse_args()

    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        absolute_strategy_path = project_root / args.strategy_file
        absolute_data_path = project_root / args.data_file
        params = json.loads(args.strategy_params)
        
        metrics = run_backtest(
            strategy_file_path=str(absolute_strategy_path),
            data_file_path=str(absolute_data_path),
            strategy_class_name=args.strategy_class_name,
            initial_cash=args.initial_cash,     # Pass backtester settings
            commission_pct=args.commission,   # Pass backtester settings
            **params
        )
        
        print(json.dumps(metrics, indent=2))
        
    except Exception as e:
        logger.error(f"Backtest run failed with error: {e}", exc_info=True)
        print(json.dumps({"status": "error", "error_message": str(e), "error_type": type(e).__name__}))
        sys.exit(1)
