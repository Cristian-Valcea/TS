# agents/backtester_agent.py (Final, Robust Version using subprocess)

import logging
import sys
from pathlib import Path
import subprocess  # <-- Use Python's built-in subprocess module

# --- Autogen Imports ---
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent

# --- Project-Specific Imports ---
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import get_llm_client

logger = logging.getLogger(__name__)

# --- NO MORE LocalCommandLineCodeExecutor ---

# --- NEW, ROBUST TOOL FUNCTION using subprocess ---
def execute_shell_command(command: str) -> str:
    """
    Executes a shell command directly using Python's subprocess module.
    This is more robust than the Autogen executor for this task on Windows.

    Args:
        command (str): The full shell command to execute.

    Returns:
        str: The combined stdout and stderr from the command execution.
    """
    logger.info(f"Backtester Tool: Attempting to execute command via subprocess: '{command}'")
    try:
        # Using shell=True runs the command through the default system shell (cmd.exe on Windows),
        # which bypasses the PowerShell execution policy issue entirely.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=180  # 3-minute timeout for data downloads
        )

        # Combine stdout and stderr for a complete log
        full_output = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

        if result.returncode == 0:
            logger.info(f"Backtester Tool: Command executed successfully. Output:\n{result.stdout}")
            # We only need to return stdout on success as it contains the JSON
            return f"Execution successful. Results:\n{result.stdout}"
        else:
            error_message = f"Execution failed with exit code {result.returncode}.\n\n{full_output}"
            logger.error(f"Backtester Tool: {error_message}")
            return error_message

    except subprocess.TimeoutExpired:
        error_message = f"Execution timed out after 180 seconds. The process was terminated."
        logger.error(f"Backtester Tool: {error_message}")
        return error_message
    except Exception as e:
        logger.exception("Backtester Tool: An unhandled exception occurred during subprocess execution.")
        return f"A critical error occurred during command execution: {str(e)}"

# --- Create the tool (This part remains the same) ---
backtest_execution_tool = FunctionTool(
    execute_shell_command,
    name="execute_backtest_command",
    description="Executes a shell command to run a backtesting script with its required arguments. The script must be self-contained and print results to stdout."
)

# --- System message (This part remains the same) ---
backtester_system_message = """You are a simple but essential Backtester Agent.

**Your ONLY job is to execute commands.**
1. You will be given a full shell command to run a Python script with its arguments.
2. You MUST use your `execute_backtest_command` tool to run it.
3. Pass the ENTIRE, UNMODIFIED command string to the `command` argument of your tool.
4. Report the full, verbatim results from the tool back, whether it's a success or an error.
5. Do not analyze, comment, or write code. Just execute and report.
"""

# --- Agent Initialization (This part remains the same) ---
logger.info("Attempting to configure and initialize BacktesterAgent...")
llm_config_or_client = get_llm_client("BacktesterAgent")
init_kwargs = {
    "name": "BacktesterAgent",
    "description": "Executes shell commands to run Python backtesting scripts.",
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