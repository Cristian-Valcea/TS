# agents/orchestrator_agent.py

import logging
import re
import os
import json
import textwrap
from pathlib import Path

# --- Project-Specific Imports ---
# It needs to know about all the agents it controls.
from agents.data_provisioning_agent import data_provisioning_agent
from agents.code_agent import code_agent
from agents.reviewer_agent import reviewer_agent
from agents.backtester_agent import backtester_agent
from agents.bay_agent import bay_agent

from config import STRATEGIES_DIR

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    An LLM-less agent that controls the entire workflow using a deterministic state machine.
    It tasks other agents, parses their output, and makes decisions based on a robust
    "strategy schema" provided by the CodeAgent.
    """
    def __init__(self):
        self.name = "OrchestratorAgent"
        self.data_provisioning_agent = data_provisioning_agent
        self.code_agent = code_agent
        self.reviewer_agent = reviewer_agent
        self.backtester_agent = backtester_agent
        self.bay_agent = bay_agent
        logger.info(f"✅ OrchestratorAgent initialized with schema-aware workflow including Optimization.")

    async def run(self, initial_task: str) -> dict:
        """The main, deterministic workflow logic."""
        logger.info(f"--- 🚀 OrchestratorAgent: Starting Full Workflow ---")
        logger.info(f"Initial Task: {initial_task}")
        
        strategy_name = "GoldenCross"
        ticker = "MSFT"
        start_date = "2021-01-01"
        end_date = "2023-12-31"

        # === STEP 1: DATA PROVISIONING ===
        logger.info("\n--- Orchestrator: Tasking DataProvisioningAgent to get data ---")
        data_request = f"Fetch and save historical daily data for {ticker} from {start_date} to {end_date}."
        data_result = await self.data_provisioning_agent.run(task=data_request)
        if not (data_result and hasattr(data_result, 'messages') and data_result.messages):
            return {"status": "error", "message": "DataProvisioningAgent failed."}
        data_agent_response = data_result.messages[-1].content
        match = re.search(r"saved it to '(.*?)'", data_agent_response)
        if not match: return {"status": "error", "message": f"Could not parse data file path from DataProvisioningAgent's response: {data_agent_response}"}
        data_file_path = match.group(1).strip()
        logging.info(f"Orchestrator: Data is ready at {data_file_path}")

        # === STEP 2: PROMPT FOR CODE & SCHEMA (NEW "TWO-PART" FORMAT) ===
        script_generation_prompt = textwrap.dedent(f"""
            You are an expert Python code generator for the Backtrader library.

            **User Request:**
            {initial_task}

            **CRITICAL INSTRUCTIONS:**
            Your response MUST be in two parts, separated by the string "---SCHEMA---".

            **PART 1: PYTHON CODE**
            - Provide the complete Python code for the strategy class named '{strategy_name}'.
            - Enclose the code in a markdown block: ```python ... ```.
            - The code MUST include docstrings and `# pylint: disable=no-member` comments.

            **PART 2: SCHEMA**
            - After the code block, on a new line, write the separator: ---SCHEMA---
            - After the separator, provide a valid JSON object describing the strategy's parameters.
            - Each key in the JSON must be the exact parameter name, and the value must be an object with "type", "default", "min", "max", and "description".

            **PERFECT EXAMPLE of the entire response you must generate:**
            ```python
            \"\"\"A strategy for trading based on a Golden Crossover.\"\"\"
            import backtrader as bt

            class GoldenCross(bt.Strategy):
                \"\"\"Implements the Golden Cross strategy.\"\"\"
                params = (('fast_ma', 50), ('slow_ma', 200),)

                def __init__(self):
                    super().__init__()
                    # pylint: disable=no-member
                    self.fast_sma = bt.indicators.SimpleMovingAverage(
                        self.data.close, period=self.p.fast_ma
                    )
                    # pylint: disable=no-member
                    self.slow_sma = bt.indicators.SimpleMovingAverage(
                        self.data.close, period=self.p.slow_ma
                    )
                    # pylint: disable=no-member
                    self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

                def next(self):
                    \"\"\"Defines the logic for each bar.\"\"\"
                    if not self.position:
                        if self.crossover > 0:
                            self.buy()
                    elif self.crossover < 0:
                        self.close()
            ```
            ---SCHEMA---
            {{
                "fast_ma": {{
                    "type": "int",
                    "default": 50,
                    "min": 10,
                    "max": 80,
                    "description": "The period for the fast moving average."
                }},
                "slow_ma": {{
                    "type": "int",
                    "default": 200,
                    "min": 100,
                    "max": 250,
                    "description": "The period for the slow moving average."
                }}
            }}
            """)
        
        # === STEP 3: CODE-REVIEW CYCLE ===
        max_fix_attempts = 2
        strategy_schema = None 
        for i in range(max_fix_attempts):
            logging.info(f"\n--- Code-Review Cycle: Attempt {i+1}/{max_fix_attempts} ---")
            code_agent_result = await self.code_agent.run(task=script_generation_prompt)
            if not (code_agent_result and hasattr(code_agent_result, 'messages') and code_agent_result.messages):
                return {"status": "error", "message": "CodeAgent failed."}
            raw_response = code_agent_result.messages[-1].content
            
            # --- NEW, ROBUST PARSING LOGIC ---
            try:
                if "---SCHEMA---" not in raw_response:
                    raise ValueError("Schema separator '---SCHEMA---' not found in CodeAgent response.")
                
                code_part, schema_part = raw_response.split("---SCHEMA---", 1)
                
                # Extract code from the markdown block
                code_match = re.search(r'```python(.*)```', code_part, re.DOTALL)
                if not code_match: raise ValueError("Python code block not found.")
                final_script_content = textwrap.dedent(code_match.group(1)).strip()

                # Parse the schema part as JSON
                strategy_schema = json.loads(schema_part)
                logging.info(f"Orchestrator: Successfully parsed code and schema: {strategy_schema}")
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                logging.error(f"Orchestrator: Failed to parse response from CodeAgent on attempt {i+1}. Error: {e}")
                script_generation_prompt += "\n\n**PREVIOUS ATTEMPT FAILED:** Your response was not formatted correctly. Please provide the python code block, followed by the '---SCHEMA---' separator, followed by the JSON schema."
                if i == max_fix_attempts - 1: return {"status": "error", "message": "CodeAgent failed to produce valid output.", "details": raw_response}
                continue 

            file_name = f"{strategy_name.lower()}_{ticker.lower()}.py"
            strategy_file_path = STRATEGIES_DIR / file_name
            with open(strategy_file_path, 'w', encoding='utf-8') as f: f.write(final_script_content)
            logging.info(f"Orchestrator: Saved script to: {strategy_file_path}")

            logging.info("\n--- Orchestrator: Tasking ReviewerAgent to review script ---")
            review_task = f"Please review the code quality and security of the script at: {strategy_file_path}"
            reviewer_result = await self.reviewer_agent.run(task=review_task)
            if not (reviewer_result and hasattr(reviewer_result, 'messages') and reviewer_result.messages):
                 return {"status": "error", "message": "ReviewerAgent failed."}
            
            review_feedback = reviewer_result.messages[-1].content
            logging.info(f"Orchestrator: Review feedback:\n---\n{review_feedback}\n---")
            pylint_passed = "pylint analysis passed" in review_feedback.lower() or "no major issues found" in review_feedback.lower()
            bandit_passed = "no issues identified" in review_feedback.lower()
            if pylint_passed and bandit_passed:
                logging.info("Orchestrator: Review PASSED. Proceeding.")
                break 
            else:
                logging.warning("Orchestrator: Review FAILED. Preparing for fix attempt...")
                script_generation_prompt += f"\n\n**PREVIOUS REVIEW FAILED.** Fix these issues:\n{review_feedback}"
                if i == max_fix_attempts - 1: return {"status": "error", "message": "Review failed after max attempts.", "details": review_feedback}
    
        # === STEP 4: INITIAL BACKTEST (BASELINE) ===
        if not strategy_schema: return {"status": "error", "message": "No valid strategy schema was stored."}
        backtest_params = {p_name: details["default"] for p_name, details in strategy_schema.items()}
        logging.info(f"Orchestrator: Using default parameters for initial backtest: {backtest_params}")
        backtester_task = f"""Please run a backtest with:
- strategy_file_path: "{strategy_file_path}"
- data_file_path: "{data_file_path}"
- strategy_class_name: "{strategy_name}"
- strategy_params_json: '{json.dumps(backtest_params)}'"""
        logging.info(f"\n--- Orchestrator: Tasking BacktesterAgent for initial run ---")
        backtester_result = await self.backtester_agent.run(task=backtester_task)
        if not (backtester_result and hasattr(backtester_result, 'messages') and backtester_result.messages):
            return {"status": "error", "message": "BacktesterAgent failed."}
        initial_backtest_content = backtester_result.messages[-1].content
        try:
            initial_results_json = json.loads(re.search(r'\{.*\}', initial_backtest_content, re.DOTALL).group(0))
            if initial_results_json.get("status") != "completed": return {"status": "error", "message": "Initial backtest failed.", "details": initial_results_json}
            logging.info(f"Orchestrator: Initial backtest successful. Baseline Sharpe Ratio: {initial_results_json.get('sharpe_ratio')}")
        except Exception as e: return {"status": "error", "message": "Could not parse initial backtest results.", "details": str(e)}

        # === STEP 5: OPTIMIZATION STAGE ===
        logging.info("\n--- Orchestrator: Preparing for Optimization Stage ---")
        param_space_for_optimizer = {p_name: [details["min"], details["max"]] for p_name, details in strategy_schema.items() if "min" in details and "max" in details}
        if not param_space_for_optimizer:
            logging.warning("Orchestrator: No optimizable parameters found in schema. Skipping optimization.")
            return {"status": "success", "message": "Backtest complete, no optimization performed.", "results": initial_results_json}

        logging.info(f"Orchestrator: Optimization search space constructed: {param_space_for_optimizer}")
        optimization_task = f"""Please run Bayesian Optimization with:
- strategy_file_path: "{strategy_file_path}"
- data_file_path: "{data_file_path}"
- strategy_class_name: "{strategy_name}"
- parameter_space_json: '{json.dumps(param_space_for_optimizer)}'"""
        
        logging.info(f"\n--- Orchestrator: Tasking BAY_Agent with optimization ---")
        optimization_result_chat = await self.bay_agent.run(task=optimization_task)
        if not (optimization_result_chat and hasattr(optimization_result_chat, 'messages') and optimization_result_chat.messages):
            return {"status": "error", "message": "BAY_Agent failed to produce output."}
        final_optimization_content = optimization_result_chat.messages[-1].content
        
        # === STEP 6: PARSE FINAL OPTIMIZATION RESULTS ===
        logging.info("\n--- Orchestrator: Final Workflow Complete. ---")
        try:
            json_str_match = re.search(r'\{.*\}', final_optimization_content, re.DOTALL)
            parsed_json = json.loads(json_str_match.group(0))
            if parsed_json.get("status") == "success":
                logging.info(f"✅ Optimization Successful! Best parameters found.")
                return {"status": "success_with_optimization", "optimization_results": parsed_json, "initial_results": initial_results_json}
            else: return {"status": "error", "message": "Optimization reported an error.", "details": parsed_json}
        except Exception as e: return {"status": "error", "message": "Could not parse JSON from optimization.", "details": final_optimization_content}

