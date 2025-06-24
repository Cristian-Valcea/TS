import sys
import os
import json # For pretty printing dict results
from typing import Dict, Any, Optional

#from autogen import tool
#from autogen_core.tools import tool # NEW for autogen-core based versions
from autogen_core.tools import FunctionTool, BaseTool # It imports these# Ensure common_logic is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the function directly, not the class, if run_backtest is a standalone function
# If BacktraderRunner is a class and run_backtest is its method, we import the class.
# Based on your file, it's a standalone function in backtrader_runner.py
from common_logic.backtesting_services.backtrader_runner import run_backtest as execute_single_backtest
# If it was a class method:
# from common_logic.backtesting_services.backtrader_runner import BacktraderRunner

def run_strategy_backtest(
    strategy_file_path: str,
    data_file_path: str,
    strategy_class_name: str = "CustomStrategy", # Default if not specified by agent
    initial_cash: float = 100000.0,
    commission_pct: float = 0.000,
    strategy_params_json: Optional[str] = None # Strategy params as a JSON string
) -> str:
    """
    Runs a single backtest for a given trading strategy using Backtrader.

    Args:
        strategy_file_path (str): RELATIVE path to the Python file containing the Backtrader strategy class.
                                  Example: "common_logic/strategies/my_strategy.py"
        data_file_path (str): RELATIVE path to the CSV data file for backtesting.
                              Example: "data/my_data.csv"
        strategy_class_name (str, optional): The name of the strategy class within the strategy_file.
                                             Defaults to "CustomStrategy".
        initial_cash (float, optional): Initial cash for the backtest. Defaults to 100000.0.
        commission_pct (float, optional): Commission percentage (e.g., 0.001 for 0.1%). Defaults to 0.0.
        strategy_params_json (str, optional): A JSON string representing a dictionary of parameters
                                              to pass to the strategy. Example: '{"sma_period": 20, "rsi_level": 30}'.
                                              Defaults to None (no parameters).

    Returns:
        str: A JSON string summarizing the backtest results, including performance metrics.
             Returns a JSON string with an error message if the backtest fails.
    """
    strategy_kwargs = {}
    if strategy_params_json:
        try:
            strategy_kwargs = json.loads(strategy_params_json)
            if not isinstance(strategy_kwargs, dict):
                return json.dumps({"error": "strategy_params_json must be a JSON dictionary string."})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON in strategy_params_json: {e}"})

    print(f"Tool 'run_strategy_backtest' called with:")
    print(f"  strategy_file_path: {strategy_file_path}")
    print(f"  data_file_path: {data_file_path}")
    print(f"  strategy_class_name: {strategy_class_name}")
    print(f"  strategy_kwargs: {strategy_kwargs}")

    # --- This part depends on how backtrader_runner.py is structured ---
    # Option 1: If run_backtest is a standalone function (as suggested by your file)
    try:
        # The execute_single_backtest (aliased from run_backtest) should take strategy_kwargs directly
        results = execute_single_backtest(
            strategy_file_path=strategy_file_path,
            data_file_path=data_file_path,
            strategy_class_name=strategy_class_name,
            initial_cash=initial_cash,
            commission_pct=commission_pct,
            # data_date_column and data_dt_format can be added if needed, or use defaults in run_backtest
            **strategy_kwargs # Unpack the dictionary here
        )
        # Remove bulky items from results for cleaner agent summary, agent can request full if needed
        if 'equity_curve' in results: del results['equity_curve']
        if 'daily_equity_curve' in results: del results['daily_equity_curve']
        # detailed_closed_trades can also be very long
        if 'detailed_closed_trades' in results and len(results['detailed_closed_trades']) > 10:
             results['detailed_closed_trades_summary'] = f"{len(results['detailed_closed_trades'])} trades (details omitted for brevity)"
             del results['detailed_closed_trades']

        return json.dumps(results, default=str) # Use default=str for datetime objects etc.
    except FileNotFoundError as e:
        return json.dumps({"error": str(e), "status": "file_not_found"})
    except AttributeError as e: # E.g. strategy class not found
        return json.dumps({"error": str(e), "status": "attribute_error"})
    except ImportError as e:
        return json.dumps({"error": str(e), "status": "import_error"})
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        return json.dumps({"error": f"An unexpected error occurred during backtest: {e}", "traceback": tb_str, "status": "failed"})

    # Option 2: If BacktraderRunner is a class and run_backtest is its method
    # runner = BacktraderRunner(
    #     strategy_file_path=strategy_file_path,
    #     data_feed_path=data_file_path, # Adjust param name if different
    #     strategy_params=strategy_kwargs, # Adjust param name if different
    #     cash=initial_cash,
    #     commission=commission_pct
    # )
    # try:
    #     results = runner.run_backtest() # This method should return the dict
    #     # ... (result processing and JSON dumping as above) ...
    # except Exception as e:
    #     # ... (error handling as above) ...
