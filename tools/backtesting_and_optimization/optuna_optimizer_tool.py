# tools/backtesting_and_optimization/optuna_optimizer_tool.py
import optuna
import json
import os
import sys
import logging # For logging within the tool
from typing import Dict, Any, Optional

from autogen_core.tools import FunctionTool, BaseTool # It imports these# Ensure common_logic is in the Python path

# Configure a logger for this tool module
logger = logging.getLogger(__name__)

# Ensure common_logic is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the standalone run_backtest function from your backtrader_runner
try:
    from common_logic.backtesting_services.backtrader_runner import run_backtest as execute_backtest_for_optuna
    BACKTRADER_RUNNER_AVAILABLE = True
    logger.info("Successfully imported run_backtest from common_logic.backtesting_services.backtrader_runner")
except ImportError as e:
    logger.error(f"Failed to import run_backtest from common_logic.backtesting_services.backtrader_runner: {e}")
    BACKTRADER_RUNNER_AVAILABLE = False
    def execute_backtest_for_optuna(*args, **kwargs): # Dummy function
        raise ImportError("backtrader_runner.run_backtest could not be imported for OptunaOptimizerTool.")


class OptunaOptimizerTool:
    """
    A tool for optimizing trading strategy parameters using Optuna and the backtrader_runner.
    """

    def __init__(self):
        # These will be set when optimize_strategy is called, to be used by the objective function
        self._current_strategy_file_path: Optional[str] = None
        self._current_data_feed_path: Optional[str] = None
        self._current_strategy_class_name: Optional[str] = None
        self._current_initial_cash: Optional[float] = None
        self._current_commission_pct: Optional[float] = None
        self._current_target_metric: Optional[str] = None
        self._current_direction: Optional[str] = None
        self._current_min_data_points: Optional[int] = None # For data check in runner

        # Optuna verbosity (optional)
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        logger.info("OptunaOptimizerTool initialized.")

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna study.
        This is an internal method called by Optuna during study.optimize().
        """
        if not BACKTRADER_RUNNER_AVAILABLE:
            logger.error("Backtrader runner is not available for Optuna objective function.")
            # Return a very bad value to penalize this trial heavily
            return -float('inf') if self._current_direction == "maximize" else float('inf')

        if not all([
            self._current_strategy_file_path, self._current_data_feed_path,
            self._current_strategy_class_name, self._current_initial_cash is not None,
            self._current_commission_pct is not None, self._current_target_metric, self._current_direction
        ]):
            logger.error("Optuna objective function called without necessary parameters being set on the instance.")
            return -float('inf') if self._current_direction == "maximize" else float('inf')

        # Suggest parameters for this trial
        # This part needs the `optuna_params_config` which was passed to `optimize_strategy`
        # We should have stored it or passed it appropriately. Let's assume it's on `self`
        if not hasattr(self, '_current_optuna_params_config') or not self._current_optuna_params_config:
            logger.error("Optuna params config not available in objective function.")
            return -float('inf') if self._current_direction == "maximize" else float('inf')

        strategy_params_for_trial = {}
        for param_name, config in self._current_optuna_params_config.items():
            param_type = config.get("type", "float")
            if param_type == "int":
                strategy_params_for_trial[param_name] = trial.suggest_int(
                    param_name, config["low"], config["high"], step=config.get("step", 1)
                )
            elif param_type == "float":
                # Optuna's suggest_float has 'log' param, not 'log_uniform' for the call itself
                strategy_params_for_trial[param_name] = trial.suggest_float(
                    param_name, config["low"], config["high"], log=config.get("log", False)
                )
            elif param_type == "categorical":
                strategy_params_for_trial[param_name] = trial.suggest_categorical(
                    param_name, config["choices"]
                )
            else:
                logger.warning(f"Unknown Optuna param_type '{param_type}' for '{param_name}'. Skipping.")
                return -float('inf') if self._current_direction == "maximize" else float('inf')
        
        logger.debug(f"Optuna Trial: Suggested strategy_params: {strategy_params_for_trial}")

        try:
            # Call the standalone run_backtest function
            # Ensure printlog=False for strategies that support it, to keep Optuna logs clean
            # Your run_backtest function might need to accept 'printlog' in **strategy_kwargs
            run_params = strategy_params_for_trial.copy()
            # run_params['printlog'] = False # If your strategy takes this

            results = execute_backtest_for_optuna(
                strategy_file_path=self._current_strategy_file_path,
                data_file_path=self._current_data_feed_path,
                strategy_class_name=self._current_strategy_class_name,
                initial_cash=self._current_initial_cash,
                commission_pct=self._current_commission_pct,
                min_data_points_required=self._current_min_data_points, # Pass this through
                **run_params # Unpack Optuna-suggested params
            )
        except Exception as e:
            logger.error(f"Error during backtest trial for params {strategy_params_for_trial}: {e}")
            return -float('inf') if self._current_direction == "maximize" else float('inf')

        if not results or results.get("status") != "completed":
            logger.warning(f"Backtest failed or returned non-completed status for params {strategy_params_for_trial}. Results: {results}")
            return -float('inf') if self._current_direction == "maximize" else float('inf')

        metric_value = results.get(self._current_target_metric)

        if metric_value is None:
            logger.warning(f"Target metric '{self._current_target_metric}' not found in backtest results: {results.keys()}. Penalizing trial.")
            return -float('inf') if self._current_direction == "maximize" else float('inf')
        
        # Check for NaN or infinity
        if not isinstance(metric_value, (int, float)) or metric_value != metric_value or \
           metric_value == float('inf') or metric_value == -float('inf'):
            logger.warning(f"Metric value '{self._current_target_metric}' is invalid ({metric_value}) for params {strategy_params_for_trial}. Penalizing trial.")
            return -float('inf') if self._current_direction == "maximize" else float('inf')

        logger.info(f"Optuna Trial with params {strategy_params_for_trial} -> {self._current_target_metric}: {float(metric_value):.4f}")
        return float(metric_value)

    def optimize_strategy(
        self,
        strategy_file_path: str,
        data_file_path: str,
        optuna_params_config_json: str, # Agent provides JSON string
        strategy_class_name: str = "CustomStrategy", # Default to this if not provided
        n_trials: int = 10, # Default to a small number for quick tests
        target_metric: str = "sharpe_ratio", # Common target
        direction: str = "maximize", # Common direction
        output_dir: str = "optimization_results",
        initial_cash: float = 100000.0,
        commission_pct: float = 0.000, # Consistent with run_backtest
        min_data_points_for_backtest: Optional[int] = 50 # Min data for each trial run
    ) -> str: # Return JSON string
        """
        Optimizes trading strategy parameters using Optuna.

        Args:
            strategy_file_path (str): Path to the Python file of the Backtrader strategy.
            data_file_path (str): Path to the CSV data file for backtesting.
            optuna_params_config_json (str): JSON STRING for Optuna parameters.
                Example: '{"sma_period": {"type": "int", "low": 10, "high": 50}, "rsi_period": {"type": "int", "low": 7, "high": 21}}'
            strategy_class_name (str, optional): Name of the strategy class. Defaults to "CustomStrategy".
            n_trials (int, optional): Number of Optuna trials. Defaults to 10.
            target_metric (str, optional): Metric from backtest results to optimize (e.g., "sharpe_ratio", "net_pnl_pct"). Defaults to "sharpe_ratio".
            direction (str, optional): "maximize" or "minimize". Defaults to "maximize".
            output_dir (str, optional): Directory to save study results. Defaults to "optimization_results".
            initial_cash (float, optional): Initial cash for backtesting. Defaults to 100000.0.
            commission_pct (float, optional): Commission percentage (e.g., 0.001 for 0.1%). Defaults to 0.000.
            min_data_points_for_backtest (int, optional): Minimum data points required for each backtest trial to be valid. Defaults to 50.

        Returns:
            str: JSON string with optimization results ("best_params", "best_value", "message").
        """
        if not BACKTRADER_RUNNER_AVAILABLE:
            return json.dumps({"error": "Backtrader runner tool is not available.", "status": "misconfigured"})

        logger.info(f"Optuna Tool optimize_strategy called:")
        logger.info(f"  Strategy: {strategy_file_path} (Class: {strategy_class_name})")
        logger.info(f"  Data: {data_file_path}")
        logger.info(f"  Trials: {n_trials}, Target: {target_metric}, Direction: {direction}")
        logger.info(f"  Output Dir: {output_dir}, Initial Cash: {initial_cash}, Commission: {commission_pct}")
        logger.info(f"  Min Data Points: {min_data_points_for_backtest}")
        logger.info(f"  Optuna Param Config JSON: {optuna_params_config_json}")

        try:
            optuna_params_config = json.loads(optuna_params_config_json)
            if not isinstance(optuna_params_config, dict):
                return json.dumps({"error": "optuna_params_config_json must be a valid JSON dictionary string."})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON in optuna_params_config_json: {e}"})
        except Exception as e_parse: # Catch other potential errors during parsing
            return json.dumps({"error": f"Error processing optuna_params_config_json: {e_parse}"})


        # Store parameters for the objective function to access via self
        self._current_strategy_file_path = strategy_file_path
        self._current_data_feed_path = data_file_path
        self._current_strategy_class_name = strategy_class_name
        self._current_initial_cash = initial_cash
        self._current_commission_pct = commission_pct
        self._current_target_metric = target_metric
        self._current_direction = direction
        self._current_optuna_params_config = optuna_params_config # Store the parsed dict
        self._current_min_data_points = min_data_points_for_backtest


        os.makedirs(output_dir, exist_ok=True)
        study_name_prefix = os.path.basename(strategy_file_path).replace('.py', '')
        study_name = f"opt_{study_name_prefix}_{target_metric}"
        
        # Using in-memory storage for simplicity in this example.
        # For persistence, use a database URL:
        # storage_url = f"sqlite:///{os.path.join(output_dir, study_name + '.db')}"
        # study = optuna.create_study(study_name=study_name, storage=storage_url, direction=direction, load_if_exists=True)
        try:
            study = optuna.create_study(direction=self._current_direction, study_name=study_name)
            study.optimize(self._objective, n_trials=n_trials, timeout=None) # Pass the method
        except Exception as e_study:
            logger.error(f"Error during Optuna study optimization: {e_study}")
            import traceback
            logger.error(traceback.format_exc())
            return json.dumps({
                "error": f"Optuna optimization failed: {str(e_study)}",
                "best_params": None, "best_value": None, "status": "optuna_failed"
            })

        best_params_file = os.path.join(output_dir, f"{study_name}_best_params.json")
        try:
            with open(best_params_file, 'w') as f:
                json.dump(study.best_params, f, indent=4)
            save_msg = f"Best parameters saved to: {best_params_file}"
        except Exception as e_save:
            save_msg = f"Failed to save best parameters to JSON: {e_save}"
            logger.error(save_msg)


        logger.info(f"Optimization complete. Best parameters: {study.best_params}, Best value ({target_metric}): {study.best_value}")
        logger.info(save_msg)

        return json.dumps({
            "message": f"Optimization complete. {save_msg}",
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials_completed": len(study.trials),
            "study_name": study.study_name,
            "status": "completed"
        }, indent=4)


if __name__ == '__main__':
    # This basicConfig is for standalone testing of this file only
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:L%(lineno)d - %(message)s')
    logger.info("OptunaOptimizerTool standalone test initiated.")

    tool_instance = OptunaOptimizerTool()
    
    # Define dummy strategy and data paths (relative to project root)
    # Assumes this script is in tools/backtesting_and_optimization/
    # So project_root is two levels up.
    # The paths passed to optimize_strategy should be relative to where the MAIN autogen script runs from.
    # For this standalone test, we assume they are relative to the project root.
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_path = os.path.abspath(os.path.join(current_file_dir, "../../"))

    # For the test, use paths relative to the project root for clarity in the call
    dummy_strategy_rel_path = "common_logic/strategies/dummy_strategy.py"
    dummy_data_rel_path = "data/dummy_data.csv" # Using the CSV from backtrader_runner

    # Ensure dummy strategy and data files exist (copied from backtrader_runner's main for self-containment)
    # Ensure dummy_strategy.py exists (create if not)
    full_dummy_strategy_path = os.path.join(project_root_path, dummy_strategy_rel_path)
    os.makedirs(os.path.dirname(full_dummy_strategy_path), exist_ok=True)
    if not os.path.exists(full_dummy_strategy_path):
        with open(full_dummy_strategy_path, 'w') as f:
            f.write("""
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
""")
        logger.info(f"Created dummy strategy file: {full_dummy_strategy_path}")

    # Ensure dummy_data.csv exists (create if not)
    full_dummy_data_path = os.path.join(project_root_path, dummy_data_rel_path)
    if not os.path.exists(full_dummy_data_path):
        logger.info(f"Creating dummy data file: {full_dummy_data_path}")
        import pandas as pd # Import pandas here for standalone test
        import numpy as np  # Import numpy here
        num_data_points = 100
        dummy_df_data = {
            'Date': pd.date_range(start='2022-01-03', periods=num_data_points, freq='B'),
            'Open': [150 + i*0.1 - (i%10)*0.3 + abs(np.random.randn()*0.5) for i in range(num_data_points)],
            'High': [152 + i*0.15 - (i%10)*0.2 + abs(np.random.randn()*0.6) for i in range(num_data_points)],
            'Low': [148 + i*0.05 + (i%10)*0.2 - abs(np.random.randn()*0.5) for i in range(num_data_points)],
            'Close': [151 + i*0.12 - (i%10)*0.25 + np.random.randn()*0.5 for i in range(num_data_points)],
            'Volume': [1000000 + i*10000 + np.random.randint(-50000, 50000) for i in range(num_data_points)]
        }
        dummy_df = pd.DataFrame(dummy_df_data)
        dummy_df['Low'] = np.minimum(dummy_df['Low'], dummy_df[['Open', 'Close', 'High']].min(axis=1))
        dummy_df['High'] = np.maximum(dummy_df['High'], dummy_df[['Open', 'Close', 'Low']].max(axis=1))
        dummy_df.to_csv(full_dummy_data_path, index=False)
        logger.info("Dummy data CSV created.")

    optuna_params_definition = {
        "sma_period": {"type": "int", "low": 5, "high": 30, "step": 5},
        "rsi_period": {"type": "int", "low": 7, "high": 21, "step": 1}
        # Ensure these params ('sma_period', 'rsi_period') are accepted by DummyStrategy
    }
    optuna_params_definition_json = json.dumps(optuna_params_definition)

    logger.info(f"Test: strategy_file_path='{dummy_strategy_rel_path}'")
    logger.info(f"Test: data_file_path='{dummy_data_rel_path}'")
    logger.info(f"Test: optuna_params_config_json='{optuna_params_definition_json}'")

    results_json_str = tool_instance.optimize_strategy(
        strategy_file_path=dummy_strategy_rel_path, # Agent will provide path relative to project root
        data_file_path=dummy_data_rel_path,       # Agent will provide path relative to project root
        optuna_params_config_json=optuna_params_definition_json,
        strategy_class_name="DummyStrategy", # Matches class in dummy_strategy.py
        n_trials=5, # Very small for quick test
        target_metric="sharpe_ratio",
        direction="maximize",
        output_dir=os.path.join(project_root_path, "test_optuna_tool_results"), # Ensure output dir is clear
        initial_cash=10000.0,
        commission_pct=0.001,
        min_data_points_for_backtest=60 # Needs to be enough for longest lookback (e.g. sma_period=30)
    )
    print("\n--- Optuna Optimizer Tool Standalone Test Results ---")
    try:
        parsed_results = json.loads(results_json_str)
        print(json.dumps(parsed_results, indent=4))
    except json.JSONDecodeError:
        print("Tool returned non-JSON string:")
        print(results_json_str)