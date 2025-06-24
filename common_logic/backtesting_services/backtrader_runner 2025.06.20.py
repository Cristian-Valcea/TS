# common_logic/backtesting_services/backtrader_runner.py

import backtrader as bt
import pandas as pd
import numpy as np  # For numerical operations, if needed
import importlib.util
import os
import sys
import logging
from typing import Dict, Any, List, Optional
import json
import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Pathing Setup ---


print(SCRIPT_DIR)
print(PROJECT_ROOT)
from agents.utils.timeframe_utils import parse_timeframe_string
from agents.utils.chart_util import plot_strategy_behavior

# Assuming performance_metrics.py will be in the same directory or correctly pathed
# from .performance_metrics import calculate_comprehensive_metrics # Example

# Configure logging
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PandasDataFeed(bt.feeds.PandasData):
    """
    Custom PandasData feed to map standard CSV column names.
    'datetime' column is expected to be the index of the DataFrame.
    """
    lines = ('open', 'high', 'low', 'close', 'volume')
    params = (
        ('datetime', None),  # Tells backtrader to use the DataFrame index for datetime
        ('open', 'Open'),    # Maps 'Open' column in DataFrame to 'open' line
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1), # -1 indicates no 'openinterest' column
    )


class DetailedTradeAnalyzer(bt.Analyzer):
    """
    Custom analyzer to store detailed information about each closed trade
    and the equity curve.
    """
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.daily_equity = {} 

    def start(self):
            super(DetailedTradeAnalyzer, self).start()
            if hasattr(self.strategy, 'broker') and self.strategy.broker is not None and \
               hasattr(self.strategy.broker, 'getvalue') and self.strategy.broker.getvalue() is not None:
                self.equity_curve.append(self.strategy.broker.getvalue()) # <--- This can be None if no data yet
            else:
                logger.warning("DetailedTradeAnalyzer.start: Broker or initial equity value not available.")
                self.equity_curve.append(self.strategy.broker.startingcash if hasattr(self.strategy, 'broker') and self.strategy.broker else 0) # Fallback

    def next(self):
        super(DetailedTradeAnalyzer, self).next()
        if hasattr(self.strategy, 'broker') and self.strategy.broker is not None and \
           hasattr(self.strategy.broker, 'getvalue') and self.strategy.broker.getvalue() is not None:
            self.equity_curve.append(self.strategy.broker.getvalue())
            # Ensure datas[0] and datetime are available
            if hasattr(self.strategy, 'datas') and self.strategy.datas and \
               hasattr(self.strategy.datas[0], 'datetime'):
                try:
                    current_dt_obj = self.strategy.datas[0].datetime.datetime(0) 
                    self.daily_equity[current_dt_obj.date()] = self.strategy.broker.getvalue()
                except IndexError: # Can happen if data isn't long enough for datetime(0)
                    logger.warning("DetailedTradeAnalyzer.next: Could not get current datetime for daily equity.")
            else:
                logger.warning("DetailedTradeAnalyzer.next: Strategy datas or datetime not available for daily equity.")
        else:
            logger.warning("DetailedTradeAnalyzer.next: Broker or equity value not available for equity curve.")



    def notify_trade(self, trade):
        if trade.isclosed:
            entry_type = 'UNKNOWN_ENTRY'
            # Using trade.long which you confirmed exists from your debugger
            if trade.long: # If trade.long is True, it was a long trade (buy to open)
                entry_type = 'LONG_TRADE_ENTRY_BUY'
            else: # If not trade.long, it was a short trade (sell to open)
                entry_type = 'SHORT_TRADE_ENTRY_SELL'
            
            # The trade.size becomes 0 when closed. trade.value is also 0.
            # To get entry size/value, one might need to look at orders or track it differently.
            # For now, let's log what's available.
            # The pnl, pnlcomm are the most important outcomes of the closed trade.

            self.trades.append({
                'symbol': getattr(trade.data, '_name', 'UNKNOWN'), # Get data feed name if available
                'ref': trade.ref,
                'status_text': 'Closed', # Since trade.isclosed is True
                'status_code': trade.status, # This will be bt.Trade.Closed (integer 2)
                'entry_type': entry_type,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'size_at_close': trade.size,  # This is 0 for closed trades
                'value_at_close': trade.value, # This is 0 for closed trades
                # 'entry_price': trade.price, # This is average entry price of the trade
                # 'entry_value': trade.value_at_entry, # Need to confirm this attribute, or calculate
                'commission_total': trade.commission, # Total commission for the trade
                'dt_opened': bt.num2date(trade.dtopen).isoformat() if trade.dtopen else None,
                'dt_closed': bt.num2date(trade.dtclose).isoformat() if trade.dtclose else None,
                'bar_opened': trade.baropen,
                'bar_closed': trade.barclose,
                'trade_len_bars': trade.barlen,
            })

    def get_analysis(self) -> Dict[str, Any]:
        return {
            'equity_curve': self.equity_curve,
            'daily_equity': self.daily_equity,
            'closed_trades': self.trades
        }


def load_strategy_class_from_file(strategy_file_path: str, strategy_class_name: str) -> type:
    """
    Dynamically loads a strategy class from a Python file.
    Ensures unique module name to prevent caching issues if reloading different versions.
    """
    if not os.path.exists(strategy_file_path):
        raise FileNotFoundError(f"Strategy file not found: {strategy_file_path}")

    # Create a unique module name based on file path and a timestamp to avoid import caching issues
    # especially if the file content changes between runs in the same Python session.
    module_name = f"strategy_module_{os.path.splitext(os.path.basename(strategy_file_path))[0]}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
    
    spec = importlib.util.spec_from_file_location(module_name, strategy_file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for module {module_name} from {strategy_file_path}")
    
    strategy_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = strategy_module # Add to sys.modules before execution
    
    try:
        spec.loader.exec_module(strategy_module)
    except Exception as e:
        # Clean up if exec_module fails
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise ImportError(f"Failed to execute module {strategy_file_path}: {e}")

    if not hasattr(strategy_module, strategy_class_name):
        # Clean up the loaded module if the class is not found
        if module_name in sys.modules:
            del sys.modules[module_name]
        raise AttributeError(f"Strategy class '{strategy_class_name}' not found in {strategy_file_path}")
    
    return getattr(strategy_module, strategy_class_name)

def run_backtest(
    strategy_file_path: str,
    data_file_path: str,
    strategy_class_name: str = "CustomStrategy",
    initial_cash: float = 100000.0,
    commission_pct: float = 0.000, # Example: 0.1% = 0.001
    data_date_column: str = "Date",
    data_dt_format: str = None, # E.g. '%Y-%m-%d %H:%M:%S' if not auto-inferred by pandas
    min_data_points_required: Optional[int] = None, # New parameter
    **strategy_kwargs: Any # Parameters for the strategy
) -> Dict[str, Any]:
    """
    Runs a Backtrader backtest for a given strategy and data.

    Args:
        strategy_file_path: Path to the Python file containing the strategy class.
        data_file_path: Path to the CSV data file.
        strategy_class_name: Name of the strategy class within the strategy file.
        initial_cash: Starting cash for the backtest.
        commission_pct: Commission percentage per trade (e.g., 0.001 for 0.1%).
        data_date_column: Name of the datetime column in the CSV.
        data_dt_format: Optional datetime format string for parsing the date column.
        **strategy_kwargs: Keyword arguments to be passed as parameters to the strategy.

    Returns:
        A dictionary containing various performance metrics.
    """
    logger.info(f"Starting backtest for strategy '{strategy_class_name}' from '{strategy_file_path}'")
    logger.info(f"Data: '{data_file_path}', Initial Cash: {initial_cash}, Commission: {commission_pct*100:.3f}%")
    if strategy_kwargs:
        logger.info(f"Strategy Parameters: {strategy_kwargs}")

    try:
        #this is for normal backtester 
        df = pd.read_csv(
            data_file_path,
            skiprows=3,  # Skip the first 3 non-data rows
            index_col=0,  # Use first column as index
            parse_dates=True,  # Parse index as dates
            names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume']  # Provide correct column names
        )
        df.index.name = 'datetime'  # Rename index for backtrader
        '''
        # this is intraday backtester
        df = pd.read_csv(
            data_file_path,
            header=0,
            parse_dates=['Date'],
            index_col='Date'
        )

        # Drop any rows with invalid dates
        df = df[~df.index.isna()]
        # Backtrader expects a plain datetime index
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        '''

        df.index.name = "datetime"
        print(df)

        logger.info(f"Successfully read CSV. Columns: {df.columns.tolist()}")
        '''
        # Check if the date column exists
        if data_date_column not in df.columns:
            logger.error(f"Date column '{data_date_column}' not found in CSV file {data_file_path}. Available columns: {df.columns.tolist()}")
            raise ValueError(f"Date column '{data_date_column}' not found in data.")
        '''

        '''
        # Convert the date column to datetime objects
        # If data_dt_format is provided, use it. Otherwise, let pandas infer.
        try:
            if data_dt_format:
                df[data_date_column] = pd.to_datetime(df[data_date_column], format=data_dt_format)
            else:
                df[data_date_column] = pd.to_datetime(df[data_date_column]) # Pandas will try to infer
        except Exception as e_date:
            logger.error(f"Failed to parse date column '{data_date_column}' with format '{data_dt_format if data_dt_format else 'inferred'}'. Error: {e_date}")
            logger.error(f"Sample data from date column: {df[data_date_column].head().tolist()}")
            raise ValueError(f"Date parsing failed for column '{data_date_column}'.") from e_date

        # Set the datetime column as index
        df = df.set_index(data_date_column)
        df.index.name = 'datetime' # Backtrader often expects the index to be named 'datetime'
        '''

        logger.info(f"Data loaded successfully. Index type: {type(df.index)}, Sample index: {df.index[:3]}")
        # NEW: Data length check
        if min_data_points_required is not None and len(df) < min_data_points_required:
            err_msg = (f"Insufficient data: Found {len(df)} data points in '{data_file_path}', "
                       f"but strategy requires at least {min_data_points_required} points.")
            logger.error(err_msg)
            return {"error": err_msg, "status": "insufficient_data", "data_points_found": len(df), "data_points_required": min_data_points_required}

        logger.info(f"Data loaded and validated. Points: {len(df)}")

    except FileNotFoundError:
        logger.error(f"Data file not found: {data_file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load or parse data from {data_file_path}: {e}")
        raise
    cerebro = bt.Cerebro() # stdstats=False because we add our own analyzers
    cerebro.broker.setcash(initial_cash)
    if commission_pct > 0:
        cerebro.broker.setcommission(commission=commission_pct)

    #timeframe, compression = parse_timeframe_string(strategy_json["timeframe"])
    data_feed = PandasDataFeed(
        dataname=df, 
        name=os.path.basename(data_file_path).split('.')[0],
        #timeframe=timeframe,
        #compression=compression,
        #name="intraday"
    ) # Give data feed a name
    cerebro.adddata(data_feed)

    try:
        strategy_class = load_strategy_class_from_file(strategy_file_path, strategy_class_name)
    except Exception as e:
        logger.error(f"Failed to load strategy class: {e}")
        raise
        
    cerebro.addstrategy(strategy_class, **strategy_kwargs)

    # Add standard analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days) # Specify timeframe
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(DetailedTradeAnalyzer, _name='detailedtradeanalyzer') # Our custom one

    try:
        logger.info("Running Cerebro backtest...")
        results_run = cerebro.run(stdstats=True) 
        if not results_run:
            logger.error("Cerebro run did not return any strategy instances.")
            return {"error": "Cerebro run returned no results.", "status": "failed"}
        strat_instance = results_run[0] 
        logger.info("Cerebro run completed.")
    except Exception as e:
        logger.error(f"Exception during Cerebro run: {e}")
        import traceback; logger.error(traceback.format_exc()) # Log full traceback
        return {"error": str(e), "status": "cerebro_run_failed"}
    '''
    # --- Strategy Behavior Plotting --- also inserted for intraday backtester
    strat = results_run[0]

    # collect only the Trade objects
    trade_objs = []
    for data_feed, trades_list in strat._trades.items():
        # trades_list is a *list* of Trade objects
        for trade in trades_list:
            if trade.isclosed:  # now trade is guaranteed to be a Trade, so this works
                trade_objs.append(trade)

    # now transform trade_objs into whatever you plot
    trades = [{
        "entry_dt":   bt.num2date(trade.dtopen),
        "exit_dt":    bt.num2date(trade.dtclose),
        "entry_price": trade.price,
        "exit_price":  trade.close,
    } for trade in trade_objs]
    plot_strategy_behavior(df, trades)
    # --- Strategy Behavior Plotting --- also inserted for intraday backtester
    '''

    # --- Metrics Extraction ---
    logger.info("Starting metrics extraction...")
    final_equity = cerebro.broker.getvalue() # Access broker directly from cerebro
    initial_cash_val = cerebro.broker.startingcash # Get starting cash from broker for consistency

    pnl = final_equity - initial_cash_val # Use actual starting cash from broker
    pnl_pct = (pnl / initial_cash_val) * 100 if initial_cash_val != 0 else 0
    logger.info(f"Initial cash (from broker): {initial_cash_val}, Final equity: {final_equity}")



    # --- Metrics Extraction ---
    #final_equity = cerebro.broker.getvalue()
    #pnl = final_equity - initial_cash
    #pnl_pct = (pnl / initial_cash) * 100 if initial_cash > 0 else 0


    sharpe_analysis = {}
    if hasattr(strat_instance.analyzers, 'sharpe') and strat_instance.analyzers.sharpe is not None:
        sharpe_analysis = strat_instance.analyzers.sharpe.get_analysis()
    else:
        logger.warning("SharpeRatio analyzer not found or not run.")
    sharpe_ratio = sharpe_analysis.get('sharperatio', float('nan'))


    annual_return_analysis = {}
    if hasattr(strat_instance.analyzers, 'annual_return') and strat_instance.analyzers.annual_return is not None:
        annual_return_analysis = strat_instance.analyzers.annual_return.get_analysis()
    else:
        logger.warning("AnnualReturn analyzer not found or not run.")
    avg_annual_return_pct = (sum(annual_return_analysis.values()) / len(annual_return_analysis) * 100) if annual_return_analysis else float('nan')


    drawdown_analysis = {}
    if hasattr(strat_instance.analyzers, 'drawdown') and strat_instance.analyzers.drawdown is not None:
        drawdown_analysis = strat_instance.analyzers.drawdown.get_analysis()
    else:
        logger.warning("DrawDown analyzer not found or not run.")
    max_drawdown_pct = drawdown_analysis.get('max', {}).get('drawdown', float('nan'))
    max_drawdown_money = drawdown_analysis.get('max', {}).get('moneydown', float('nan'))



    trade_analysis = {}
    if hasattr(strat_instance.analyzers, 'tradeanalyzer') and strat_instance.analyzers.tradeanalyzer is not None:
        trade_analysis = strat_instance.analyzers.tradeanalyzer.get_analysis()
    else:
        logger.warning("TradeAnalyzer not found or not run.")
    # ... (rest of trade_analysis parsing with .get() and defaults) ...
    total_trades = trade_analysis.get('total', {}).get('total', 0)


    open_trades = trade_analysis.get('total', {}).get('open', 0)
    closed_trades_count = trade_analysis.get('total', {}).get('closed', 0)
    
    winning_trades = trade_analysis.get('won', {}).get('total', 0)
    losing_trades = trade_analysis.get('lost', {}).get('total', 0)
    win_rate_pct = (winning_trades / closed_trades_count) * 100 if closed_trades_count > 0 else 0.0
    
    avg_win_pnl = trade_analysis.get('won', {}).get('pnl', {}).get('average', float('nan'))
    avg_loss_pnl = trade_analysis.get('lost', {}).get('pnl', {}).get('average', float('nan'))
    profit_factor = abs(trade_analysis.get('won', {}).get('pnl', {}).get('total', 0) / trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0)) if trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0) != 0 else float('inf')


    custom_trade_analysis = {}
    if hasattr(strat_instance.analyzers, 'detailedtradeanalyzer') and strat_instance.analyzers.detailedtradeanalyzer is not None:
        custom_trade_analysis = strat_instance.analyzers.detailedtradeanalyzer.get_analysis()
    else:
        logger.warning("DetailedTradeAnalyzer not found or not run.")

    metrics = {
        'status': 'completed',
        'initial_cash': initial_cash,
        'final_equity': final_equity,
        'net_pnl': pnl,
        'net_pnl_pct': pnl_pct,
        'sharpe_ratio': sharpe_ratio,
        'avg_annual_return_pct': avg_annual_return_pct,
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_money': max_drawdown_money,
        'total_trades': total_trades, # Includes open and closed
        'closed_trades_count': closed_trades_count,
        'open_trades_count': open_trades,
        'winning_trades_count': winning_trades,
        'losing_trades_count': losing_trades,
        'win_rate_pct': win_rate_pct,
        'average_win_pnl': avg_win_pnl,
        'average_loss_pnl': avg_loss_pnl,
        'profit_factor': profit_factor,
        'equity_curve': custom_trade_analysis['equity_curve'], # Full equity curve (per bar)
        'daily_equity_curve': custom_trade_analysis['daily_equity'], # End-of-day equity
        'detailed_closed_trades': custom_trade_analysis['closed_trades'],
        'strategy_params_used': strategy_kwargs,
        # 'raw_bt_analyzers': { # Optionally include raw analyzer outputs for deeper inspection
        #     'sharpe': sharpe_analysis,
        #     'annual_return': annual_return_analysis,
        #     'drawdown': drawdown_analysis,
        #     'tradeanalyzer': trade_analysis
        # }
    }
    
    # If you have a separate comprehensive metrics calculation function:
    # from .performance_metrics import calculate_comprehensive_metrics
    # metrics = calculate_comprehensive_metrics(metrics_input_dict) 
    # This would take the dictionary above (or parts of it) and potentially calculate more (Sortino, Calmar, etc.)

    # --- THIS IS THE FIX ---
    # Convert datetime.date keys in daily_equity_curve to strings for JSON compatibility
    if 'daily_equity_curve' in metrics:
        metrics['daily_equity_curve'] = {
            key.isoformat(): value for key, value in metrics['daily_equity_curve'].items()
        }
    # --- END FIX ---



    logger.info(f"Backtest completed. Final Equity: {final_equity:.2f}, PnL: {pnl:.2f} ({pnl_pct:.2f}%)")

    # --- Strategy Behavior Plotting --- also inserted for intraday backtester
    #plot_strategy_behavior(df, custom_trade_analysis['closed_trades'])

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a single Backtrader backtest.")
    parser.add_argument('--strategy-file', required=True, help="Path to the strategy Python file, relative to project root.")
    parser.add_argument('--data-file', required=True, help="Path to the CSV data file, relative to project root.")
    parser.add_argument('--strategy-class-name', required=True, help="The name of the strategy class inside the file.")
    parser.add_argument('--strategy-params', default="{}", help="JSON string of strategy parameters.")
    parser.add_argument('--no-plot', action='store_true', help="Disable plot generation at the end of the backtest.")

    args = parser.parse_args()

    try:
        absolute_strategy_path = PROJECT_ROOT / args.strategy_file
        absolute_data_path = PROJECT_ROOT / args.data_file

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
        
        metrics = run_backtest(
            strategy_file_path=absolute_strategy_path,
            data_file_path=absolute_data_path,
            strategy_class_name=args.strategy_class_name,
            **params
        )
        
        print(json.dumps(metrics, indent=2))
        
    except Exception as e:
        logger.error(f"Backtest run failed with error: {e}", exc_info=True)
        print(json.dumps({"status": "error", "error_message": str(e), "error_type": type(e).__name__}))
        sys.exit(1)