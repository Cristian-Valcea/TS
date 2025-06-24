# agents/backtesting_and_optimization_agent.py
import sys
import os
from autogen import AssistantAgent

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from tools.backtesting_and_optimization.optuna_optimizer_tool import OptunaOptimizerTool

class BacktestingAndOptimizationAgent(AssistantAgent):
    def __init__(self, name="BO_Agent", llm_config=None, **kwargs):
        system_message = """You are the Backtesting and Optimization Agent (B&O Agent).
Your primary role is to optimize trading strategies using the 'optimize_strategy' tool.

To use the 'optimize_strategy' tool, you need to provide the following arguments:
1.  `strategy_file_path`: (string) The RELATIVE path to the Python file of the Backtrader strategy.
    Example: "common_logic/strategies/dummy_strategy.py"
2.  `data_feed_path`: (string) The RELATIVE path to the CSV data file for historical data.
    Example: "data/dummy_ohlcv.csv"
3.  `optuna_params_config`: (JSON string representing a dictionary) This specifies which strategy parameters to optimize,
    their types ('int', 'float', 'categorical'), and their ranges or choices.
    Example for JSON string:
    '{
        "sma_period": {"type": "int", "low": 10, "high": 50, "step": 2},
        "rsi_period": {"type": "int", "low": 7, "high": 21},
        "some_float_param": {"type": "float", "low": 0.1, "high": 0.9, "log": false},
        "choice_param": {"type": "categorical", "choices": ["optionA", "optionB"]}
    }'
    Ensure this is a valid JSON string when calling the tool.
4.  `n_trials`: (integer, optional, default is 25 in the tool) Number of optimization trials.
    A good starting range for real optimization might be 50-200.
5.  `target_metric`: (string, optional, default "sharpe_ratio") The metric from backtest results to optimize.
    Common options: "sharpe_ratio", "sqn", "annual_return".
6.  `direction`: (string, optional, default "maximize") Whether to "maximize" or "minimize" the target_metric.
7.  `output_dir`: (string, optional, default "optimization_results") Directory to save optimization artifacts.
    It's good practice to specify this, perhaps related to the strategy name.
8.  `initial_cash`: (float, optional, default 100000.0) Initial cash for backtesting.
9.  `commission`: (float, optional, default 0.001) Commission per trade.


Your Process:
- When asked to optimize a strategy, first identify all the required parameters for the `optimize_strategy` tool.
- If the user provides the `optuna_params_config` as a Python dictionary, you MUST convert it into a JSON string before calling the tool.
- If any information is missing (especially `strategy_file_path`, `data_feed_path`, or `optuna_params_config`), ask the user for clarification.
- Confirm the parameters, then call the `optimize_strategy` tool.
- Present the results clearly, including the best parameters found and the best value of the target metric.
- State where the detailed results (like the best_params.json file) are saved.
- If the tool returns an error, report the error to the user.

All file paths should be relative to the project root (e.g., `autogen_trading_system/`).
Do not make up file paths or parameter configurations; ask if they are not clear.
"""
        super().__init__(name=name, system_message=system_message, llm_config=llm_config, **kwargs)

        # Instantiate and register the tool
        self.optimizer_tool_instance = OptunaOptimizerTool()
        self.register_function(
            function_map={
                "optimize_strategy": self.optimizer_tool_instance.optimize_strategy
            }
        )
        print(f"{self.name} initialized with OptunaOptimizerTool.")
