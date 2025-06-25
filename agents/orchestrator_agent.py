# agents/orchestrator_agent.py
"""
This module defines the OrchestratorAgent, a central component that manages
the workflow of creating, reviewing, backtesting, and optimizing trading strategies.
It operates as a deterministic state machine, coordinating various specialized agents.
"""

import logging
import re
import os
import json
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional

# --- Project-Specific Imports ---
# Import other agents that this orchestrator will control.
try:
    from agents.data_provisioning_agent import data_provisioning_agent
    from agents.code_agent import code_agent
    from agents.reviewer_agent import reviewer_agent
    from agents.backtester_agent import backtester_agent
    from agents.bay_agent import bay_agent # Bayesian Optimization Agent
except ImportError as e:
    logging.critical(f"OrchestratorAgent: Failed to import one or more sub-agents: {e}. This is a critical error.")
    # Depending on the application's needs, you might re-raise or handle this gracefully.
    # For now, we'll let it proceed, but agents might be None.
    data_provisioning_agent = code_agent = reviewer_agent = backtester_agent = bay_agent = None


# Import configuration for directory paths.
try:
    from config import STRATEGIES_DIR
except ImportError:
    logging.error("OrchestratorAgent: Could not import STRATEGIES_DIR from config. Defaulting to 'shared_work_dir/strategies'.")
    # Provide a fallback if config is not available, though this might indicate a setup issue.
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    STRATEGIES_DIR = PROJECT_ROOT / "shared_work_dir" / "strategies"
    os.makedirs(STRATEGIES_DIR, exist_ok=True)


logger = logging.getLogger(__name__) # Logger specific to this module.

class OrchestratorAgent:
    """
    Controls the end-to-end workflow for trading strategy development.

    This agent is designed to be LLM-less (it does not make calls to an LLM itself).
    Instead, it follows a predefined sequence of steps, tasking other specialized
    LLM-powered or rule-based agents to perform specific actions like data fetching,
    code generation, code review, backtesting, and parameter optimization.

    The workflow relies on a "strategy schema" (a JSON definition of a strategy's
    parameters) that is expected to be provided by the `CodeAgent`. This schema
    is used for configuring backtests and optimization runs.
    """
    def __init__(self):
        """
        Initializes the OrchestratorAgent and its subordinate agents.
        """
        self.name = "OrchestratorAgent"
        # Assign imported agent instances. These should ideally be singletons or configured instances.
        self.data_provisioning_agent = data_provisioning_agent
        self.code_agent = code_agent
        self.reviewer_agent = reviewer_agent
        self.backtester_agent = backtester_agent
        self.bay_agent = bay_agent # Bayesian Optimization agent

        # Log initialization status, checking if all required agents are available.
        required_agents = {
            "DataProvisioningAgent": self.data_provisioning_agent,
            "CodeAgent": self.code_agent,
            "ReviewerAgent": self.reviewer_agent,
            "BacktesterAgent": self.backtester_agent,
            "BayesianOptimizationAgent": self.bay_agent
        }
        missing_agents = [name for name, agent_instance in required_agents.items() if agent_instance is None]
        if missing_agents:
            logger.error(f"OrchestratorAgent initialized with missing agents: {', '.join(missing_agents)}. Workflow may fail.")
        else:
            logger.info("✅ OrchestratorAgent initialized successfully with all subordinate agents.")

    async def _parse_agent_response(self, agent_response_obj: Any, field_name: str = 'content') -> Optional[str]:
        """
        Safely extracts content from an agent's response object.
        Handles cases where the response or its messages might be None or malformed.
        """
        if not agent_response_obj or not hasattr(agent_response_obj, 'messages') or not agent_response_obj.messages:
            logger.warning(f"Agent response object or its messages list is empty/invalid.")
            return None

        last_message = agent_response_obj.messages[-1]
        if not hasattr(last_message, field_name):
            logger.warning(f"Last message in agent response does not have field '{field_name}'.")
            return None

        return getattr(last_message, field_name)

    async def run(self, initial_task: str) -> Dict[str, Any]:
        """
        Executes the main deterministic workflow.

        The workflow involves:
        1.  **Data Provisioning**: Fetches historical market data.
        2.  **Code Generation & Schema Definition**: Generates Python strategy code and its parameter schema.
        3.  **Code Review**: Reviews the generated code for quality and security.
        4.  **Initial Backtest**: Runs a baseline backtest with default parameters.
        5.  **Parameter Optimization**: (If applicable) Optimizes strategy parameters using Bayesian Optimization.
        6.  **Final Result Parsing**: Parses and returns the results of the workflow.

        Args:
            initial_task (str): A description of the strategy to be developed.
                                This is used to prompt the CodeAgent.
                                Example: "Create a Golden Cross strategy for MSFT from 2021 to 2023."

        Returns:
            Dict[str, Any]: A dictionary containing the status of the workflow and results.
                            Possible statuses include "error", "success", "success_with_optimization".
                            The dictionary includes detailed messages and data depending on the outcome.
        """
        logger.info(f"--- 🚀 OrchestratorAgent: Starting Full Workflow ---")
        logger.info(f"Initial Task: {initial_task}")
        
        # --- Configuration (Hardcoded for now, ideally from `initial_task` or a config file) ---
        # TODO: Parse these details from `initial_task` using an LLM or regex.
        strategy_name = "GoldenCross" # Example
        ticker = "MSFT"               # Example
        start_date = "2021-01-01"     # Example
        end_date = "2023-12-31"       # Example
        # --- End Configuration ---

        # === STEP 1: DATA PROVISIONING ===
        logger.info("\n--- Orchestrator: Tasking DataProvisioningAgent ---")
        data_request = f"Fetch and save historical daily data for {ticker} from {start_date} to {end_date}."

        data_agent_response_obj = await self.data_provisioning_agent.run(task=data_request)
        data_agent_response_content = await self._parse_agent_response(data_agent_response_obj)

        if not data_agent_response_content:
            return {"status": "error", "message": "DataProvisioningAgent failed to provide a valid response."}

        path_match = re.search(r"saved it to '(.*?)'", data_agent_response_content)
        if not path_match:
            err_msg = f"Could not parse data file path from DataProvisioningAgent's response: {data_agent_response_content}"
            logger.error(err_msg)
            return {"status": "error", "message": err_msg}

        data_file_path = path_match.group(1).strip()
        logger.info(f"Orchestrator: Data provisioning successful. Data is ready at: {data_file_path}")

        # === STEP 2: PROMPT FOR CODE GENERATION (WITH SCHEMA) ===
        # This prompt instructs the CodeAgent to return both the Python code and a JSON schema.
        script_generation_prompt = textwrap.dedent(f"""
            You are an expert Python code generator specializing in the Backtrader library.

            **User Request:**
            {initial_task}

            **CRITICAL INSTRUCTIONS:**
            Your entire response MUST be in two distinct parts, separated by the exact string "---SCHEMA---" on its own line.

            **PART 1: PYTHON STRATEGY CODE**
            - Provide the complete Python code for the Backtrader strategy class.
            - The class MUST be named '{strategy_name}'.
            - Enclose the Python code within a standard markdown code block: ```python ... ```.
            - The code must include clear docstrings for the class and its methods.
            - Include necessary `# pylint: disable=no-member` comments for Backtrader attributes accessed via `self.p` or `self.datas[0]`.

            **PART 2: STRATEGY PARAMETER SCHEMA**
            - After the Python code block, on a new line, write the separator: ---SCHEMA---
            - Following the separator, provide a valid JSON object that describes the strategy's optimizable parameters.
            - Each key in the JSON object must be the exact parameter name as used in the `params` tuple of the strategy.
            - The value for each key must be an object containing:
                - "type": (e.g., "int", "float")
                - "default": The default value for the parameter.
                - "min": The minimum value for optimization.
                - "max": The maximum value for optimization.
                - "description": A brief explanation of the parameter.

            **PERFECT EXAMPLE of the entire response format:**
            ```python
            \"\"\"A strategy for trading based on a Golden Crossover.\"\"\"
            import backtrader as bt

            class GoldenCross(bt.Strategy):
                \"\"\"Implements the Golden Cross strategy using two SMAs.\"\"\"
                params = (('fast_ma', 50), ('slow_ma', 200),) # pylint: disable=no-member

                def __init__(self):
                    super().__init__()
                    # pylint: disable=no-member
                    self.fast_sma = bt.indicators.SimpleMovingAverage(
                        self.datas[0].close, period=self.p.fast_ma
                    )
                    # pylint: disable=no-member
                    self.slow_sma = bt.indicators.SimpleMovingAverage(
                        self.datas[0].close, period=self.p.slow_ma
                    )
                    # pylint: disable=no-member
                    self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

                def next(self):
                    \"\"\"Defines the trading logic for each bar.\"\"\"
                    if not self.position: # Not in the market
                        if self.crossover > 0: # Golden cross
                            self.buy()
                    elif self.crossover < 0: # Death cross
                        self.close() # Exit position
            ```
            ---SCHEMA---
            {{
                "fast_ma": {{
                    "type": "int", "default": 50, "min": 10, "max": 80,
                    "description": "Period for the fast moving average."
                }},
                "slow_ma": {{
                    "type": "int", "default": 200, "min": 100, "max": 250,
                    "description": "Period for the slow moving average."
                }}
            }}
            """)
        
        # === STEP 3: CODE GENERATION AND REVIEW CYCLE ===
        max_fix_attempts = 2  # Allow a couple of attempts for the CodeAgent to fix issues.
        strategy_schema: Optional[Dict[str, Any]] = None
        final_script_content: Optional[str] = None
        strategy_file_path: Optional[Path] = None

        for attempt in range(max_fix_attempts):
            logger.info(f"\n--- Code Generation & Review Cycle: Attempt {attempt + 1}/{max_fix_attempts} ---")

            code_agent_response_obj = await self.code_agent.run(task=script_generation_prompt)
            raw_code_agent_response = await self._parse_agent_response(code_agent_response_obj)

            if not raw_code_agent_response:
                return {"status": "error", "message": "CodeAgent failed to provide a response."}
            
            # --- Robust Parsing of Code and Schema ---
            try:
                if "---SCHEMA---" not in raw_code_agent_response:
                    raise ValueError("Schema separator '---SCHEMA---' not found in CodeAgent response.")
                
                code_part, schema_part = raw_code_agent_response.split("---SCHEMA---", 1)

                code_match = re.search(r'```python(.*?)```', code_part, re.DOTALL | re.IGNORECASE)
                if not code_match:
                    raise ValueError("Python code block (```python ... ```) not found in the first part of CodeAgent's response.")
                
                final_script_content = textwrap.dedent(code_match.group(1)).strip() # Dedent and strip whitespace

                # Ensure schema_part is valid JSON
                strategy_schema = json.loads(schema_part.strip())
                logger.info(f"Orchestrator: Successfully parsed code and strategy schema: {json.dumps(strategy_schema, indent=2)}")

            except (ValueError, json.JSONDecodeError) as e:
                err_msg = f"Failed to parse response from CodeAgent on attempt {attempt + 1}. Error: {e}. Response was: {raw_code_agent_response[:500]}..." # Log snippet
                logger.error(err_msg)
                script_generation_prompt += (
                    f"\n\n**PREVIOUS ATTEMPT FAILED DUE TO PARSING ERROR:** {e}. "
                    "Please ensure your response strictly follows the two-part format: Python code block, then '---SCHEMA---', then the JSON schema."
                )
                if attempt == max_fix_attempts - 1:
                    return {"status": "error", "message": "CodeAgent failed to produce valid parsable output after multiple attempts.", "details": err_msg}
                continue # Try prompting CodeAgent again with feedback.

            # --- Save Generated Script ---
            # Ensure STRATEGIES_DIR is a Path object and exists
            if not isinstance(STRATEGIES_DIR, Path): STRATEGIES_DIR = Path(STRATEGIES_DIR)
            os.makedirs(STRATEGIES_DIR, exist_ok=True)

            file_name = f"{strategy_name.lower().replace(' ', '_')}_{ticker.lower()}.py"
            strategy_file_path = STRATEGIES_DIR / file_name
            try:
                with open(strategy_file_path, 'w', encoding='utf-8') as f:
                    f.write(final_script_content)
                logger.info(f"Orchestrator: Saved generated strategy script to: {strategy_file_path}")
            except IOError as e:
                logger.error(f"Orchestrator: Failed to save strategy script to {strategy_file_path}: {e}")
                return {"status": "error", "message": f"IOError saving script: {e}"}


            # --- Code Review ---
            logger.info("\n--- Orchestrator: Tasking ReviewerAgent to review the generated script ---")
            review_task = f"Please review the Python code quality, correctness (for Backtrader), and security of the script located at: {strategy_file_path}"

            reviewer_response_obj = await self.reviewer_agent.run(task=review_task)
            review_feedback = await self._parse_agent_response(reviewer_response_obj)

            if not review_feedback:
                 return {"status": "error", "message": "ReviewerAgent failed to provide feedback."}

            logger.info(f"Orchestrator: Review feedback received:\n---\n{review_feedback}\n---")
            
            # Simple check for review success (can be made more sophisticated)
            # TODO: Use more robust review parsing (e.g., structured output from ReviewerAgent)
            pylint_passed = "pylint analysis passed" in review_feedback.lower() or "no major issues found" in review_feedback.lower()
            bandit_passed = "no security issues identified" in review_feedback.lower() # Example for security

            if pylint_passed and bandit_passed: # Add other checks if needed
                logger.info("Orchestrator: Code review PASSED. Proceeding to backtest.")
                break # Exit loop, code is good
            else:
                logger.warning("Orchestrator: Code review FAILED. Preparing for fix attempt...")
                script_generation_prompt += (
                    f"\n\n**PREVIOUS CODE REVIEW FAILED.** Please address the following issues in the script:\n{review_feedback}\n"
                    "Ensure you regenerate the full Python code block and the ---SCHEMA--- section correctly."
                )
                if attempt == max_fix_attempts - 1:
                    return {"status": "error", "message": "Code review failed after maximum attempts.", "details": review_feedback}

        # After loop, check if we successfully got script and schema
        if not final_script_content or not strategy_schema or not strategy_file_path:
            return {"status": "error", "message": "Failed to generate and review strategy code successfully."}

        # === STEP 4: INITIAL BACKTEST (BASELINE RUN) ===
        # Use default parameters from the schema for the first backtest.
        default_backtest_params = {p_name: details["default"] for p_name, details in strategy_schema.items()}
        logger.info(f"Orchestrator: Using default parameters for initial backtest: {default_backtest_params}")

        backtester_task_desc = (
            f"Please run a backtest using the following configuration:\n"
            f"- Strategy File Path: \"{strategy_file_path}\"\n"
            f"- Data File Path: \"{data_file_path}\"\n"
            f"- Strategy Class Name: \"{strategy_name}\"\n"
            f"- Strategy Parameters (JSON): '{json.dumps(default_backtest_params)}'"
        )
        logger.info(f"\n--- Orchestrator: Tasking BacktesterAgent for initial backtest ---")

        backtester_response_obj = await self.backtester_agent.run(task=backtester_task_desc)
        initial_backtest_content = await self._parse_agent_response(backtester_response_obj)

        if not initial_backtest_content:
            return {"status": "error", "message": "BacktesterAgent failed to provide results for the initial run."}

        try:
            # Attempt to parse the JSON part of the backtester's response
            json_match = re.search(r'\{.*\}', initial_backtest_content, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in BacktesterAgent's response.")
            initial_results_json = json.loads(json_match.group(0))

            if initial_results_json.get("status") != "completed":
                logger.error(f"Initial backtest did not complete successfully. Details: {initial_results_json}")
                return {"status": "error", "message": "Initial backtest failed.", "details": initial_results_json}

            logger.info(f"Orchestrator: Initial backtest successful. Baseline Sharpe Ratio: {initial_results_json.get('sharpe_ratio', 'N/A')}")
        except (json.JSONDecodeError, ValueError) as e:
            err_msg = f"Could not parse JSON results from initial backtest. Error: {e}. Response: {initial_backtest_content[:500]}"
            logger.error(err_msg)
            return {"status": "error", "message": err_msg}

        # === STEP 5: PARAMETER OPTIMIZATION STAGE (using Bayesian Optimization) ===
        logger.info("\n--- Orchestrator: Preparing for Bayesian Optimization Stage ---")
        # Construct the parameter space for the optimizer from the schema.
        param_space_for_optimizer = {
            p_name: [details["min"], details["max"]]
            for p_name, details in strategy_schema.items()
            if "min" in details and "max" in details # Only include params with min/max for optimization
        }

        if not param_space_for_optimizer:
            logger.warning("Orchestrator: No optimizable parameters (with min/max) found in the strategy schema. Skipping optimization.")
            return {"status": "success", "message": "Initial backtest complete. No optimization performed as no optimizable parameters were defined in the schema.", "results": initial_results_json}

        logger.info(f"Orchestrator: Parameter search space for optimization constructed: {param_space_for_optimizer}")
        
        optimization_task_desc = (
            f"Please run Bayesian Optimization with the following configuration:\n"
            f"- Strategy File Path: \"{strategy_file_path}\"\n"
            f"- Data File Path: \"{data_file_path}\"\n"
            f"- Strategy Class Name: \"{strategy_name}\"\n"
            f"- Parameter Search Space (JSON): '{json.dumps(param_space_for_optimizer)}'"
        )
        
        logger.info(f"\n--- Orchestrator: Tasking Bayesian Optimization Agent (BAY_Agent) ---")
        optimization_response_obj = await self.bay_agent.run(task=optimization_task_desc)
        final_optimization_content = await self._parse_agent_response(optimization_response_obj)

        if not final_optimization_content:
            return {"status": "error", "message": "BayesianOptimizationAgent (BAY_Agent) failed to provide output."}

        # === STEP 6: PARSE FINAL OPTIMIZATION RESULTS AND CONCLUDE ===
        logger.info("\n--- Orchestrator: Final Workflow Complete. Parsing optimization results. ---")
        try:
            json_match_opt = re.search(r'\{.*\}', final_optimization_content, re.DOTALL)
            if not json_match_opt:
                 raise ValueError("No JSON object found in BayesianOptimizationAgent's response.")
            parsed_optimization_json = json.loads(json_match_opt.group(0))

            if parsed_optimization_json.get("status") == "success":
                logger.info(f"✅ Optimization Successful! Best parameters and results found.")
                return {
                    "status": "success_with_optimization",
                    "message": "Workflow completed with successful optimization.",
                    "optimization_results": parsed_optimization_json,
                    "initial_backtest_results": initial_results_json # Include baseline for comparison
                }
            else:
                logger.error(f"Optimization process reported an error. Details: {parsed_optimization_json}")
                return {"status": "error", "message": "Optimization process reported an error.", "details": parsed_optimization_json}
        except (json.JSONDecodeError, ValueError) as e:
            err_msg = f"Could not parse JSON from optimization results. Error: {e}. Response: {final_optimization_content[:500]}"
            logger.error(err_msg)
            return {"status": "error", "message": err_msg}
