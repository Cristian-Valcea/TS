# agents/backtester_agent.py (Simplified Final Version)

import logging
import os
import json
from pathlib import Path
import sys

from autogen.coding import LocalCommandLineCodeExecutor
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import get_llm_client, SHARED_WORK_DIR

logger = logging.getLogger(__name__)

# The executor simply runs code. It doesn't need to know about strategies.
executor = LocalCommandLineCodeExecutor(work_dir=SHARED_WORK_DIR)

def execute_python_file(file_path: str) -> str:
    """
    Executes a Python script from a given file path and returns its output.

    Args:
        file_path (str): The absolute or relative path to the Python script.

    Returns:
        str: The combined stdout and stderr from the execution.
    """
    logger.info(f"Backtester Tool: Attempting to execute Python file: {file_path}")
    try:
        # The executor can directly run a file.
        result = executor.execute_code_blocks(
            code_blocks=[("python", file_path)]
            #code_blocks=[(f"python {file_path}", "")]
            #code_blocks=[CodeBlock(language=f"python {file_path}", code="")]
        )

        if result.exit_code == 0:
            logger.info(f"Backtester Tool: Script executed successfully. Output:\n{result.output}")
            return f"Execution successful. Results:\n{result.output}"
        else:
            error_message = f"Execution failed with exit code {result.exit_code}.\nERROR:\n{result.output}"
            logger.error(f"Backtester Tool: {error_message}")
            return f"{error_message}\n\n**CodeAgent**, please fix the script at '{file_path}'."
            
    except Exception as e:
        logger.exception("Backtester Tool: An exception occurred.")
        return f"An exception occurred: {str(e)}\n\n**CodeAgent**, please fix the script at '{file_path}'."

# --- Create the new, simpler tool ---
backtest_execution_tool = FunctionTool(
    execute_python_file,
    name="execute_backtest_script",
    description="Executes a self-contained Python backtesting script from a file and returns the output."
)

# --- Define the new, simpler system message ---
backtester_system_message = """You are a simple but essential Backtester Agent.

**Your ONLY job is to execute scripts.**
1. Wait for the `CodeAgent` to provide the path to a Python script.
2. Once you have the file path, use your `execute_backtest_script` tool to run it.
3. Report the full, verbatim results from the tool back to the group, whether it's a success or an error.
4. Do not analyze, do not comment, do not write code. Just execute and report.
"""

# --- Agent Initialization ---
logger.info("Attempting to configure and initialize BacktesterAgent...")
llm_config_or_client = get_llm_client("BacktesterAgent")
init_kwargs = {
    "name": "BacktesterAgent",
    "description": "Executes self-contained Python backtesting scripts using a secure code execution tool.",
    "system_message": backtester_system_message,
    "tools": [backtest_execution_tool],
}
if llm_config_or_client is None:
    raise ValueError("BacktesterAgent LLM config is None.")
else:
    if isinstance(llm_config_or_client, dict):
        init_kwargs["llm_config"] = llm_config_or_client
    else:
        init_kwargs["model_client"] = llm_config_or_client
try:
    backtester_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ BacktesterAgent '{backtester_agent.name}' initialized successfully.")
except Exception as e:
    logger.error(f"❌ FAILED to initialize BacktesterAgent: {e}", exc_info=True)
    raise