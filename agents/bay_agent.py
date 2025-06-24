# agents/bay_agent.py
import json
import logging
import subprocess
import sys
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from config import get_llm_client

logger = logging.getLogger(__name__)

def run_bayesian_optimization(
    strategy_file_path: str,
    data_file_path: str,
    strategy_class_name: str,
    parameter_space_json: str,
    num_iterations: int = 15,
    num_init_points: int = 5,
    objective_metric: str = "sharpe_ratio"
) -> str:
    """Runs Bayesian Optimization on a strategy to maximize an objective metric."""
    logger.info(f"BAY_Agent Tool: Starting Bayesian Optimization for '{strategy_file_path}'.")
    command = [
        sys.executable,
        "common_logic/optimizing_services/bayesian_runner.py",
        "--strategy-file", strategy_file_path,
        "--data-file", data_file_path,
        "--strategy-class-name", strategy_class_name,
        "--param-space", parameter_space_json,
        "--n-iter", str(num_iterations),
        "--init-points", str(num_init_points),
        "--objective", objective_metric,
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=900) # 15-min timeout
        return process.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Optimization script failed.\nStderr: {e.stderr}"
        logger.error(f"BAY_Agent Tool: {error_message}")
        return json.dumps({"status": "error", "message": error_message})
    except Exception as e:
        logger.error(f"BAY_Agent Tool: An unexpected error occurred: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)})

bayesian_optimization_tool = FunctionTool(run_bayesian_optimization,description="bayesian optimization", name="run_bayesian_optimization")

bay_agent_system_message = """You are BAY_Agent, a specialist in financial strategy optimization.
Your ONLY task is to use the `run_bayesian_optimization` tool when requested by the Orchestrator.
You will be given the file paths for the strategy and data, the strategy class name, and a JSON string for the parameter search space.
You must call the tool with these exact inputs. After the tool runs, report the complete JSON output it provides back to the Orchestrator without any changes.
"""

bay_agent = AssistantAgent(
    name="BAY_Agent",
    system_message=bay_agent_system_message,
    model_client= get_llm_client("BAY_Agent"), # Assumes a mapping exists in your config
    tools=[bayesian_optimization_tool]
)
