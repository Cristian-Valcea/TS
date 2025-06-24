# agents/orchestrator_agent.py

import logging
import re
import os
import json
import textwrap
from pathlib import Path

# --- Project-Specific Imports ---
from agents.data_provisioning_agent import data_provisioning_agent
from agents.code_agent import code_agent
from agents.reviewer_agent import reviewer_agent
from agents.backtester_agent import backtester_agent
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
        logger.info(f"✅ OrchestratorAgent initialized with schema-aware workflow.")

    async def run(self, initial_task: str) -> dict:
        """The main, deterministic workflow logic."""
        logger.info(f"--- 🚀 OrchestratorAgent: Starting Full Workflow ---")
        logger.info(f"Initial Task: {initial_task}")
        
        # --- Parameter Extraction (can be made more dynamic later) ---
        strategy_name = "SmaCrossover"
        ticker = "NVDA"
        start_date = "2022-01-01"
        end_date = "2023-12-31"

        # =========================================================================
        # === STEP 1: DATA PROVISIONING
        # =========================================================================
        logger.info("\n--- Orchestrator: Tasking DataProvisioningAgent to get data ---")
        data_request = f"Fetch and save historical daily data for {ticker} from {start_date} to {end_date}."
        data_result = await self.data_provisioning_agent.run(task=data_request)
        
        if not (data_result and hasattr(data_result, 'messages') and data_result.messages):
            return {"status": "error", "message": "DataProvisioningAgent failed or produced no output."}
        
        data_agent_response = data_result.messages[-1].content
        match = re.search(r"saved it to '(.*?)'", data_agent_response)
        if not match:
             return {"status": "error", "message": f"Could not parse data file path from DataProvisioningAgent's response: {data_agent_response}"}
        
        data_file_path = match.group(1).strip()
        logging.info(f"Orchestrator: Data is ready at {data_file_path}")

        # =========================================================================
        # === STEP 2: CODE GENERATION & SCHEMA DEFINITION PROMPT
        # =========================================================================
        script_generation_prompt = textwrap.dedent(f"""
            You are an expert Python code generator for the Backtrader library who writes PEP 8 compliant, production-quality code.

            **User Request:**
            {initial_task}

            **CRITICAL INSTRUCTIONS:**
            Your response MUST be a single, valid JSON object. This JSON object must have two top-level keys: "code" and "schema".

            1.  **The "code" key:** The value must be a string containing the full Python code for the strategy class named '{strategy_name}'.
                - The class MUST inherit from `bt.Strategy`.
                - It MUST contain a `params` tuple for strategy parameters.
                - It MUST NOT contain an `if __name__ == '__main__':` block or any code to run a backtest.

            2.  **The "schema" key:** The value must be a JSON object describing the parameters defined in the code's `params` tuple.
                - Each key in the schema must be the exact parameter name used in the code (e.g., "fast_ma").
                - The value for each parameter must be an object with "type", "default", and "description".

            **PERFECT EXAMPLE of the entire JSON response you must generate:**
            ```json
            {{
              "code": "import backtrader as bt\\n\\nclass SmaCrossover(bt.Strategy):\\n    params = (('fast_ma', 50), ('slow_ma', 200),)\\n\\n    def __init__(self):\\n        super().__init__()\\n        self.fast_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.fast_ma)\\n        self.slow_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.slow_ma)\\n        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)\\n\\n    def next(self):\\n        if not self.position:\\n            if self.crossover > 0:\\n                self.buy()\\n        elif self.crossover < 0:\\n            self.close()",
              "schema": {{
                "fast_ma": {{
                  "type": "int",
                  "default": 50,
                  "description": "The period for the fast moving average."
                }},
                "slow_ma": {{
                  "type": "int",
                  "default": 200,
                  "description": "The period for the slow moving average."
                }}
              }}
            }}
            ```
        """)
        # =========================================================================
        # === STEP 3: CODE-REVIEW CYCLE
        # =========================================================================
        max_fix_attempts = 2
        strategy_schema = None # Initialize schema to None
        for i in range(max_fix_attempts):
            logging.info(f"\n--- Code-Review Cycle: Attempt {i+1}/{max_fix_attempts} ---")
            
            # --- A. CODE GENERATION ---
            code_agent_result = await self.code_agent.run(task=script_generation_prompt)
            if not (code_agent_result and hasattr(code_agent_result, 'messages') and code_agent_result.messages):
                return {"status": "error", "message": "CodeAgent failed."}
            
            raw_response = code_agent_result.messages[-1].content
            
            # --- B. PARSE SCHEMA AND CODE (Robust version) ---
            try:
                # Find the JSON object within the raw response, in case the LLM adds extra text.
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if not json_match:
                    raise json.JSONDecodeError("No JSON object found in the response.", raw_response, 0)

                parsed_response = json.loads(json_match.group(0))
                final_script_content = parsed_response["code"]
                strategy_schema = parsed_response["schema"]
                logging.info(f"Orchestrator: Successfully parsed strategy code and schema: {strategy_schema}")
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Orchestrator: Failed to parse JSON from CodeAgent on attempt {i+1}. Error: {e}")
                script_generation_prompt += f"\n\n**PREVIOUS ATTEMPT FAILED:** Your last response was not valid JSON or was missing keys. Please try again, adhering strictly to the required format."
                if i == max_fix_attempts - 1:
                    return {"status": "error", "message": "CodeAgent failed to produce valid JSON.",.backtester_agent = backtester_agent
        logger.info(f"✅ OrchestratorAgent initialized with schema-aware workflow.")

    async def run(self, initial_task: str) -> dict:
        """The main, deterministic workflow logic."""
        logger.info(f"--- 🚀 OrchestratorAgent: Starting Full Workflow ---")
        logging.info(f"Initial Task: {initial_task}")
        
        strategy_name = "SmaCrossover"
        ticker = "NVDA"
        start_date = "2022-01-01"
        end_date = "2023-12-31"

        # =========================================================================
        # === STEP 1: DATA PROVISIONING
        # =========================================================================
        logger.info("\n--- Orchestrator: Tasking DataProvisioningAgent to get data ---")
        data_request = f"Fetch and save historical daily data for {ticker} from {start_date} to {end_date}."
        data_result = await self.data_provisioning_agent.run(task=data_request)
        
        if not (data_result and hasattr(data_result, 'messages') and data_result.messages):
            return {"status": "error", "message": "DataProvisioningAgent failed or produced no output."}
        
        data_agent_response = data_result.messages[-1].content
        match = re.search(r"saved it to '(.*?)'", data_agent_response)
        if not match:
             return {"status": "error", "message": f"Could not parse data file path from DataProvisioningAgent's response: {data_agent_response}"}
        
        data_file_path = match.group(1).strip()
        logging.info(f"Orchestrator: Data is ready at {data_file_path}")

        # =========================================================================
        # === STEP 2: CODE GENERATION & SCHEMA DEFINITION PROMPT
        # =========================================================================
        # CORRECTED: The prompt is now dedented to remove leading whitespace
        script_generation_prompt = textwrap.dedent(f"""
            You are an expert Python code generator for the Backtrader library who writes PEP 8 compliant, production-quality code.

            **User Request:**
            {initial_task}

            **CRITICAL INSTRUCTIONS:**
            Your response MUST be a single, valid JSON object. This JSON object must have two top-level keys: "code" and "schema".

            1.  **The "code" key:** The value must be a string containing the full Python code for the strategy class named '{strategy_name}'.
                - The class MUST inherit from `bt.Strategy`.
                - It MUST contain a `params` tuple for strategy parameters.
                - It MUST NOT contain an `if __name__ == '__main__':` block or any code to run a backtest.

            2.  **The "schema" key:** The value must be a JSON object describing the parameters defined in the code's `params` tuple.
                - Each key in the schema must be the exact parameter name used in the code (e.g., "fast_ma").
                - The value for each parameter must be an object with "type", "default", and "description".

            **PERFECT EXAMPLE of the entire JSON response you must generate:**
            ```json
            {{
              "code": "import backtrader as bt\\n\\nclass SmaCrossover(bt.Strategy):\\n    params = (('fast_ma', 50), ('slow_ma', 200),)\\n\\n    def __init__(self):\\n        super().__init__()\\n        self.fast_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.fast_ma)\\n        self.slow_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.slow_ma)\\n        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)\\n\\n    def next(self):\\n        if not self.position:\\n            if self.crossover > 0:\\n                self.buy()\\n        elif self.crossover < 0:\\n            self.close()",
              "schema": {{
                "fast_ma": {{
                  "type": "int",
                  "default": 50,
                  "description": "The period for the fast moving average."
                }},
                "slow_ma": {{
                  "type": "int",
                  "default": 200,
                  "description": "The period for the slow moving average."
                }}
              }}
            }}
            ```
            """)
        # =========================================================================
        # === STEP 3: CODE-REVIEW CYCLE
        # =========================================================================
        max_fix_attempts = 2
        strategy_schema = None 
        for i in range(max_fix_attempts):
            logging.info(f"\n--- Code-Review Cycle: Attempt {i+1}/{max_fix_attempts} ---")
            
            code_agent_result = await self.code_agent.run(task=script_generation_prompt)
            if not (code_agent_result and hasattr(code_agent_result, 'messages') and code_agent_result.messages):
                return {"status": "error", "message": "CodeAgent failed."}
            
            raw_response = code_agent_result.messages[-1].content
            
            # CORRECTED: More robust JSON parsing
            try:
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if not json_match:
                    raise json.JSONDecodeError("No JSON object found in the response.", raw_response, 0)
                parsed_response = json.loads(json_match.group(0))
                final_script_content = parsed_response["code"]
                strategy_schema = parsed_response["schema"]
                logging.info(f"Orchestrator: Successfully parsed strategy code and schema: {strategy_schema}")
            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Orchestrator: Failed to parse JSON from CodeAgent on attempt {i+1}. Error: {e}")
                script_generation_prompt += f"\n\n**PREVIOUS ATTEMPT FAILED:** Your last response was not valid JSON or was missing keys. Please try again."
                if i == max_fix_attempts - 1:
                    return {"status": "error", "message": "CodeAgent failed to produce valid JSON.", "details": raw_response}
                continue 

            # CORRECTED: Added encoding='utf-8' for robust file writing
            file_name = f"{strategy_name.lower()}_{ticker.lower()}.py"
            strategy_file_path = STRATEGIES_DIR / file_name
            with open(strategy_file_path, 'w', encoding='utf-8') as f:
                f.write(final_script_content)
            logging.info(f"Orchestrator: Saved script to: {strategy_file_path}")

            logging.info("\n--- Orchestrator: Tasking ReviewerAgent to review script ---")
            review_task = f"Please review the code quality and security of the script at: {strategy_file_path}"
            reviewer_result = await self.reviewer_agent.run(task=review_task)
        
            if not (reviewer_result and hasattr(reviewer_result, 'messages') and reviewer_result.messages):
                 return {"status": "error", "message": "ReviewerAgent failed."}
        
            review_feedback = reviewer_result.messages[-1].content
            logging.info(f"Orchestrator: Review feedback:\n---\n{review_feedback}\n---")

            if "code review passed" in review_feedback.lower():
                logging.info("Orchestrator: Review PASSED. Proceeding to backtest.")
                break 
            else:
                logging.warning("Orchestrator: Review FAILED. Preparing for fix attempt...")
                script_generation_prompt += f"\n\n**PREVIOUS REVIEW FAILED.** Fix these issues:\n{review_feedback}"
                if i == max_fix_attempts - 1:
                    logging.error("Orchestrator: Maximum fix attempts reached. Aborting.")
                    return {"status": "error", "message": "Review failed after max attempts.", "details": review_feedback}
    
        # =========================================================================
        # === STEP 4: BACKTEST THE SCRIPT USING THE SCHEMA
        # =========================================================================
        if not strategy_schema:
            return {"status": "error", "message": "Code generation and review finished, but no valid strategy schema was stored."}

        backtest_params = {
            param_name: details["default"]
            for param_name, details in strategy_schema.items()
        }
        logging.info(f"Orchestrator: Using parameters from schema for backtest: {backtest_params}")

        backtester_task = f"""
        The code review has passed for the strategy script located at '{strategy_file_path}'.
        The data is available at '{data_file_path}'.

        Please run a backtest using your `run_strategy_backtest` tool with the following arguments:
        - strategy_file_path: "{strategy_file_path}"
        - data_file_path: "{data_file_path}"
        - strategy_class_name: "{strategy_name}"
        - strategy_params_json: '{json.dumps(backtest_params)}'
        """

        logging.info(f"\n--- Orchestrator: Tasking BacktesterAgent with a DYNAMIC and CORRECT tool request ---")
        backtester_result = await self.backtester_agent.run(task=backtester_task)

        if not (backtester_result and hasattr(backtester_result, 'messages') and backtester_result.messages):
            return {"status": "error", "message": "BacktesterAgent failed."}
        
        final_backtester_content = backtester_result.messages[-1].content
    
        # =========================================================================
        # === STEP 5: PARSE AND RETURN FINAL RESULTS
        # =========================================================================
        logging.info("\n--- Orchestrator: Workflow Complete. ---")
        try:
            json_str_match = re.search(r'\{.*\}', final_backtester_content, re.DOTALL)
            if not json_str_match:
                raise ValueError("No JSON object found in the backtester's response.")
    
            parsed_json = json.loads(json_str_match.group(0))

            if parsed_json.get("status") == "completed":
                logging.info("✅ Backtest Successful!")
                return {"status": "success", "results": parsed_json}
            else:
                error_detail = parsed_json.get('error_message') or parsed_json.get('error')
                logging.error(f"Backtest ran but reported an internal error: {error_detail}")
                return {"status": "error", "message": "Backtest reported an internal error.", "details": parsed_json}

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to parse JSON from backtester output: {e}")
            return {"status": "error", "message": "Could not parse JSON from backtest.", "details": final_backtester_content}

# Instantiate a single instance for easy import
orchestrator_agent = OrchestratorAgent()
