# agents/backtester_agent.py (Cleaned and Corrected)

import sys
import os
import logging
from pathlib import Path

# --- Autogen and Project-Specific Imports ---
# Add project root to the Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from config import get_llm_client

# --- Import the REAL tool function ---
# This function should call your backtrader_runner.py script
from tools.backtesting_and_optimization.backtesting_tools import run_strategy_backtest

logger = logging.getLogger(__name__)

# =================================================================================
# 1. WRAP THE REAL TOOL
# We take the imported Python function and wrap it in a FunctionTool
# so the AssistantAgent can use it.
# =================================================================================

backtesting_tool = FunctionTool(
    run_strategy_backtest,
    name="run_strategy_backtest",
    description="Runs a backtest using a strategy file, a data file, and optional parameters. Returns a JSON string of the results."
)

# This is the list of tools the agent will have.
tools_for_backtester = [backtesting_tool]

# =================================================================================
# 2. DEFINE THE AGENT'S SYSTEM MESSAGE
# This instructs the agent on how and when to use its tool.
# =================================================================================

backtester_system_message = """You are a specialized Backtester Agent.
Your primary function is to run backtests on trading strategies using the `run_strategy_backtest` tool.

**Workflow:**
1.  Wait until both the `DataProvisioningAgent` and the `CodeAgent` have announced that the data and strategy script files are ready.
2.  Once you have the file paths for the strategy script and the data CSV, you MUST call the `run_strategy_backtest` tool.
3.  Provide the necessary arguments to the tool: `strategy_file_path`, `data_file_path`, and `strategy_class_name`. The class name is often provided in the user's initial request.
4.  After the tool runs, present the key metrics from the resulting JSON (e.g., PnL, Sharpe Ratio, Max Drawdown) clearly to the group.
5.  If the tool returns an error, report the full error so the `CodeAgent` can perform a fix.
"""

# =================================================================================
# 3. INITIALIZE AND EXPORT THE AGENT
# Simplified and robust initialization.
# =================================================================================

logger.info("Attempting to configure BacktesterAgent...")

# Prepare initialization arguments
init_kwargs = {
    "name": "BacktesterAgent",
    "description": "Backtests Python strategies using file paths and reports performance metrics.",
    "system_message": backtester_system_message,
    "tools": tools_for_backtester
}

# Get the LLM client from the central configuration
backtester_llm_client = get_llm_client("BacktesterAgent")

if backtester_llm_client is None:
    raise ValueError("BacktesterAgent LLM client configuration failed and is None.")
else:
    # Handle both client objects and llm_config dicts
    if isinstance(backtester_llm_client, dict):
        init_kwargs["llm_config"] = backtester_llm_client
    else:
        init_kwargs["model_client"] = backtester_llm_client

try:
    backtester_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ BacktesterAgent '{backtester_agent.name}' initialized successfully with the correct tool.")
except Exception as e:
    logger.error(f"❌ FAILED to initialize BacktesterAgent: {e}", exc_info=True)
    raise
