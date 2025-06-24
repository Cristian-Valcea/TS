# agents/backtester_agent.py

# --- Python Standard Library Imports ---
import sys
import os
import logging
import json
from pathlib import Path

# --- Autogen and Third-party Imports ---
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent

# --- Project-Specific Imports ---
# Add project root to path to allow importing from other directories
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the core backtesting function from your existing runner
# Make sure the path 'common_logic.backtesting_services.backtrader_runner' is correct for your project structure
try:
    from common_logic.backtesting_services.backtrader_runner import run_backtest
except ImportError:
    logging.error("CRITICAL: Could not import 'run_backtest' from 'common_logic.backtesting_services.backtrader_runner'. Please ensure the file exists and the path is correct.")
    raise

from config import get_llm_client # Your LLM configuration function

# --- Logger Setup ---
logger = logging.getLogger(__name__)
def convert_keys_to_str(obj):
    """Recursively converts dictionary keys to strings."""
    if isinstance(obj, dict):
        return {
            str(key) if not isinstance(key, dict) else convert_keys_to_str(key): 
                convert_keys_to_str(value) if isinstance(value, (dict, list)) else value
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) if isinstance(item, (dict, list)) else item for item in obj]
    else:
        return obj

# =================================================================================
# 1. DEFINE THE TOOL FUNCTION
# This function is a "wrapper" around your powerful run_backtest.
# It handles calling it and formatting the results as a string for the LLM.
# =================================================================================

def run_strategy_backtest_from_files(
    strategy_file_path: str,
    data_file_path: str,
    strategy_class_name: str = "CustomStrategy",
    strategy_params_json: str = "{}"
) -> str:
    """
    Invokes the core backtesting engine with file paths and parameters.

    Args:
        strategy_file_path (str): The path to the Python file with the strategy class.
        data_file_path (str): The path to the CSV data file for the backtest.
        strategy_class_name (str): The name of the strategy class in the file.
        strategy_params_json (str): A JSON string of parameters for the strategy (e.g., '{"period": 20}').

    Returns:
        str: A JSON string containing the detailed backtest results or an error message.
    """
    logger.info(f"Backtester Tool: Received request to run '{strategy_file_path}' with data '{data_file_path}'")
    
    try:
        # Convert the JSON parameter string into a dictionary
        strategy_params = json.loads(strategy_params_json)
        logger.info(f"Parsed strategy parameters: {strategy_params}")

        # Call your robust backtesting function
        results_dict = run_backtest(
            strategy_file_path=strategy_file_path,
            data_file_path=data_file_path,
            strategy_class_name=strategy_class_name,
            **strategy_params # Unpack the dictionary into keyword arguments
        )
        logger.info(f"Parsed strategy parameters: {results_dict}")
        # --- THIS IS THE FIX ---
        # Before serializing, we must convert the datetime.date keys in the
        # 'daily_equity' dictionary to strings.
        if 'daily_equity' in results_dict and isinstance(results_dict['daily_equity'], dict):
            results_dict['daily_equity'] = {
                key.isoformat(): value for key, value in results_dict['daily_equity'].items()
            }
        # --- END OF FIX ---
        #         
        # The LLM needs a string, so we convert the results dictionary to a JSON string.
        # This is a clean and structured way to pass complex data.
        # We need a custom JSON encoder for non-serializable objects like datetime.date
        def custom_json_serializer(obj):
            if isinstance(obj, (Path, os.PathLike)):
                return str(obj)
            # Add other non-serializable types here if they appear in your results
            if hasattr(obj, 'isoformat'): # Handles datetime, date objects
                return obj.isoformat()
            if isinstance(obj, float) and (obj == float('inf') or obj == float('-inf') or obj != obj): # handle inf/-inf/nan
                return str(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        # Apply recursive conversion to all dictionary keys
        results_dict = convert_keys_to_str(results_dict)

        # Then proceed with JSON serialization
        results_json = json.dumps(results_dict, indent=2, default=custom_json_serializer)


        #results_json = json.dumps(results_dict, indent=2, default=custom_json_serializer)
        logger.info("Backtester Tool: Backtest completed successfully. Returning JSON results.")
        return results_json

    except json.JSONDecodeError as e:
        error_msg = f"Error: Invalid JSON format for strategy_params_json. Details: {e}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        # Create a more informative error message that explicitly names the agent
        # who should handle the problem.
        error_msg = (
            f"The backtest failed with a critical error: {e}. "
            f"This is likely a problem with the strategy code or how the data is handled. "
            f"**CodeAgent**, please analyze the error and the strategy file at '{strategy_file_path}' to fix it."
        )
        # --- END OF FIX ---
        logger.error(error_msg, exc_info=True)
        return error_msg

# =================================================================================
# 2. WRAP THE FUNCTION IN A FunctionTool
# =================================================================================

backtesting_tool = FunctionTool(
    run_strategy_backtest_from_files,
    name="run_strategy_backtest",
    description="Runs a full backtest using a strategy file, a data file, and parameters. Returns a JSON string of performance metrics."
)


# =================================================================================
# 3. DEFINE THE AGENT'S SYSTEM MESSAGE
# This is adapted from your previous file - it's very good.
# =================================================================================

backtester_system_message = """You are a specialized Backtester Agent. Your primary function is to run backtests on trading strategies.

**CRITICAL INSTRUCTIONS:**
1. Wait for the `DataProvisioningAgent` to provide a data file path.
2. Wait for the `CodeAgent` to provide a strategy file path. The CodeAgent's message will also contain the name of the strategy class. It will look like: "Successfully saved strategy 'TheClassName' to '.../path/to/file.py'".
3. You MUST parse the **exact class name** from the CodeAgent's message.
4. Once you have the data file path, the strategy file path, AND the exact strategy class name, call your `run_strategy_backtest` tool with these three arguments.
5. If the tool returns an error, report the full error message and explicitly state that the **CodeAgent** needs to fix the script.
6. If the tool is successful, present the key metrics from the JSON results clearly to the group.
"""

# =================================================================================
# 4. INITIALIZE AND EXPORT THE AGENT
# =================================================================================

logger.info("Attempting to configure and initialize BacktesterAgent...")

llm_config_or_client = get_llm_client("BacktesterAgent")

init_kwargs = {
    "name": "BacktesterAgent",
    "description": "Runs backtests on Python strategy files using provided data files and reports detailed performance metrics.",
    "system_message": backtester_system_message,
    "tools": [backtesting_tool], # Provide the tool we just created
}

if llm_config_or_client is None:
    logger.critical("CRITICAL (BacktesterAgent): get_llm_client returned None.")
    raise ValueError("BacktesterAgent LLM config is None.")
else:
    if isinstance(llm_config_or_client, dict):
        init_kwargs["llm_config"] = llm_config_or_client
    else:
        init_kwargs["model_client"] = llm_config_or_client

try:
    backtester_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ BacktesterAgent '{backtester_agent.name}' initialized successfully with the backtrader_runner tool.")
except Exception as e:
    logger.error(f"❌ FAILED to initialize BacktesterAgent: {e}", exc_info=True)
    raise