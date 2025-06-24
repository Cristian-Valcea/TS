# agents/orchestrator_agent.py (Final Version with all steps)

import logging
import re
import os
import json
import textwrap
import datetime
from pathlib import Path
import asyncio

# --- Project-Specific Imports ---
# It now needs to know about all the agents it controls.
from agents.data_provisioning_agent import data_provisioning_agent
from agents.code_agent import code_agent
from agents.reviewer_agent import reviewer_agent
from agents.backtester_agent import backtester_agent
from config import STRATEGIES_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    """
    A special, LLM-less agent that controls the entire workflow, including data provisioning,
    code generation, code review, and backtesting.
    """
    def __init__(self):
        self.name = "OrchestratorAgent"
        self.data_provisioning_agent = data_provisioning_agent
        self.code_agent = code_agent
        self.reviewer_agent = reviewer_agent
        self.backtester_agent = backtester_agent
        logger.info(f"✅ OrchestratorAgent initialized with full workflow.")

    async def run(self, initial_task: str) -> dict:
        """The main, deterministic workflow logic."""
        logging.info(f"--- 🚀 OrchestratorAgent: Starting Full Workflow ---")
        logging.info(f"Initial Task: {initial_task}")
        
        # --- Parameter Extraction ---
        strategy_name = "SmaCrossover"
        ticker = "NVDA"
        start_date = "2022-01-01"
        end_date = "2023-12-31"

        # === STEP 1: GET THE DATA ===
        logging.info("\n--- Orchestrator: Tasking DataProvisioningAgent to get data ---")
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

        # === STEP 2: GENERATE FULL SCRIPT (NEW LOGIC) ===
        # The prompt now asks for the COMPLETE script.
        # We pass the data_file_path so it can be hardcoded into the script.
        script_generation_prompt = f"""
You are an expert Python code generator for the Backtrader library who writes PEP 8 compliant, production-quality code.

**User Request:**
{initial_task}

**CRITICAL SCRIPT REQUIREMENTS:**
1.  The script MUST contain all necessary imports for the strategy (e.g., `import backtrader as bt`).
2.  The script MUST define a single Python class named '{strategy_name}' that inherits from `bt.Strategy`.
3.  The class MUST contain a `params` tuple for strategy parameters.
4.  The `__init__` method MUST correctly define the indicators (e.g., using `self.data.close`).
5.  The script MUST NOT contain an `if __name__ == '__main__':` block.
6.  The script MUST NOT contain any code to create a Cerebro engine, load data, or run a backtest. It should ONLY define the strategy class.

Your response MUST be ONLY the raw Python code for the script. Do not include markdown.

**PERFECT EXAMPLE of the entire file content you should generate:**
```python
\"\"\"A simple Backtrader strategy class for SMA Crossover.\"\"\"
import backtrader as bt

class SmaCrossover(bt.Strategy):
    \"\"\"Implements the SMA Crossover strategy.\"\"\"
    params = (('fast', 50), ('slow', 200))

    def __init__(self):
        super().__init__()
        self.fast_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.fast
        )
        self.slow_sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.slow
        )
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()        
```
"""
        logging.info("\n--- Orchestrator: Tasking CodeAgent to generate FULL SCRIPT ---")
        
        
        # === STEP 3: CODE GENERATION & REVIEW LOOP ===
        max_fix_attempts = 2
        for i in range(max_fix_attempts):
            logging.info(f"\n--- Code-Review Cycle: Attempt {i+1}/{max_fix_attempts} ---")

            code_agent_result = await self.code_agent.run(task=script_generation_prompt)
            if not (code_agent_result and hasattr(code_agent_result, 'messages') and code_agent_result.messages):
                return {"status": "error", "message": "CodeAgent failed."}
            
            raw_script = code_agent_result.messages[-1].content
            # We still clean it, just in case
            final_script_content = textwrap.dedent(re.sub(r'^```python\n|```$', '', raw_script, flags=re.MULTILINE)).strip()

            # B. SAVE SCRIPT TO FILE
            file_name = f"{strategy_name.lower()}_{ticker.lower()}.py"
            strategy_file_path = STRATEGIES_DIR / file_name
            
            with open(strategy_file_path, 'w') as f:
                f.write(final_script_content)
            logging.info(f"Orchestrator: Saved script to: {strategy_file_path}")

            # C. REVIEW THE SCRIPT
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
                initial_task = f"{initial_task}\n\n**PREVIOUS ATTEMPT FAILED.** Fix these issues:\n{review_feedback}"
                if i == max_fix_attempts - 1:
                    logging.error("Orchestrator: Maximum fix attempts reached. Aborting.")
                    return {"status": "error", "message": "Review failed after max attempts.", "details": review_feedback}
        
        # === STEP 4: BACKTEST THE REVIEWED SCRIPT ===
        '''
        backtester_command = (
            f"python \"{strategy_file_path}\" "
            f"--strategy_name {strategy_name} "
            f"--ticker {ticker} " # Note: The script fetches its own data, this is for consistency/logging
            f"--start_date {start_date} "
            f"--end_date {end_date}"
        )
        logging.info(f"\n--- Orchestrator: Tasking BacktesterAgent to execute command ---\n{backtester_command}")
        backtester_result = await self.backtester_agent.run(task=backtester_command)
        '''

        backtester_task = f"""
        The code review has passed for the strategy script located at '{strategy_file_path}'.
        The data is available at '{data_file_path}'.

        Please run a backtest using your `run_strategy_backtest` tool with the following arguments:
        - strategy_file_path: "{strategy_file_path}"
        - data_file_path: "{data_file_path}"
        - strategy_class_name: "{strategy_name}"
        """

        logging.info(f"\n--- Orchestrator: Tasking BacktesterAgent with a direct tool request ---")
        backtester_result = await self.backtester_agent.run(task=backtester_task)

        if not (backtester_result and hasattr(backtester_result, 'messages') and backtester_result.messages):
            return {"status": "error", "message": "BacktesterAgent failed."}
            
        final_backtester_content = backtester_result.messages[-1].content
        
        # === STEP 5: PARSE AND RETURN FINAL RESULTS ===
        logging.info("\n--- Orchestrator: Workflow Complete. ---")
        try:
            # First, try to parse the entire response as JSON, as the tool might return it directly.
            # A more robust check looks for the JSON within the string.
            json_str_match = re.search(r'\{.*\}', final_backtester_content, re.DOTALL)
            if not json_str_match:
                raise ValueError("No JSON object found in the backtester's response.")
        
            parsed_json = json.loads(json_str_match.group(0))
    
            # Check the status field within the JSON
            if parsed_json.get("status") == "completed":
                logging.info("✅ Backtest Successful!")
                return {"status": "success", "results": parsed_json}
            else:
                # The backtest ran but reported an error in its own JSON
                logging.error(f"Backtest ran but reported an internal error: {parsed_json.get('error')}")
                return {"status": "error", "message": "Backtest reported an internal error.", "details": parsed_json}

        except (json.JSONDecodeError, ValueError) as e:
            # This catches cases where the response is not valid JSON at all
            logging.error(f"Failed to parse JSON from backtester output: {e}")
            return {"status": "error", "message": "Could not parse JSON from backtest.", "details": final_backtester_content}

# Instantiate a single instance
orchestrator_agent = OrchestratorAgent()