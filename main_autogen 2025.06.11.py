# main_autogen.py (Orchestrator Final Version)

import logging
# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)

import sys
import asyncio
from pathlib import Path
import re
import json
import os
import textwrap # <-- ADD THIS IMPORT
from config import STRATEGIES_DIR, PROJECT_ROOT


# --- Autogen and Project-Specific Imports ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from agents.code_agent import code_agent
from agents.backtester_agent import backtester_agent
from agents.data_provisioning_agent import data_provisioning_agent
from autogen_agentchat.messages import TextMessage

async def run_trading_workflow():
    """
    A deterministic, orchestrated workflow for creating and backtesting a strategy.
    This version programmatically fixes indentation and removes markdown code blocks.
    """
    logging.info("--- 🚀 Starting Orchestrated Trading System Workflow (v4) ---")

    # === STEP 1: DEFINE THE TASK & READ THE TEMPLATE ===
    strategy_name = "SmaCrossover"
    ticker = "NVDA"
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    user_request = (
        f"Create a simple moving average (SMA) crossover strategy named '{strategy_name}' for the '{ticker}' ticker. "
        f"Use a 50-day fast moving average and a 200-day slow moving average."
    )
    
    try:
        template_path = PROJECT_ROOT / "common_logic" / "templates" / "strategy_template.py"
        with open(template_path, 'r') as f:
            strategy_template = f.read()
        logging.info("Orchestrator: Successfully read the strategy template.")
    except Exception as e:
        logging.error(f"Orchestrator: Failed to read strategy template: {e}")
        return

    # === STEP 2: TASK THE SIMPLIFIED CodeAgent TO GENERATE A SNIPPET ===
    snippet_generation_prompt = f"""
    Based on the user's request below, please generate the Python code for the strategy class.

    **User Request:**
    {user_request}

    **Code Template (for context):**
    ```python
    {strategy_template}
    Use code with caution.
    Your response must be ONLY the Python code for the class, 
    ready to be inserted into the template's ##STRATEGY_CLASS_DEFINITION## placeholder.
    """
    logging.info("\n--- Orchestrator: Tasking CodeAgent to generate strategy SNIPPET ---")
    code_agent_result = await code_agent.run(task=snippet_generation_prompt)

    if code_agent_result is None or not hasattr(code_agent_result, 'messages') or not code_agent_result.messages:
        logging.error("CodeAgent task failed or produced no messages.")
        return

    # ** THE ULTIMATE FIX: Strip markdown and fix indentation **
    raw_snippet = code_agent_result.messages[-1].content

    # 1. Remove markdown code block fences (```python ... ```)
    cleaned_snippet = re.sub(r'^```python\n|```$', '', raw_snippet, flags=re.MULTILINE).strip()

    # 2. Fix any common leading indentation
    strategy_snippet = textwrap.dedent(cleaned_snippet).strip()

    logging.info(f"Orchestrator: Received and CLEANED strategy snippet from CodeAgent:\n---\n{strategy_snippet}\n---")

    # === STEP 3: ORCHESTRATOR FILLS TEMPLATE AND SAVES FILE ===
    final_script_content = strategy_template.replace("##STRATEGY_CLASS_DEFINITION##", strategy_snippet)

    file_name = f"{strategy_name.lower()}_{ticker.lower()}.py"
    strategy_file_path = STRATEGIES_DIR / file_name

    try:
        with open(strategy_file_path, 'w') as f:
            f.write(final_script_content)
        logging.info(f"Orchestrator: Successfully saved final script to: {strategy_file_path}")
    except Exception as e:
        logging.error(f"Orchestrator: Failed to save final script: {e}")
        return
    
    # === STEP 4: CONSTRUCT & EXECUTE THE BACKTEST COMMAND ===
    # ... (rest of the function is identical and correct) ...
    backtester_command = (
        f"python \"{strategy_file_path}\" "
        f"--strategy_name {strategy_name} "
        f"--ticker {ticker} "
        f"--start_date {start_date} "
        f"--end_date {end_date}"
    )

    logging.info(f"\n--- Orchestrator: Tasking BacktesterAgent to execute command ---\n{backtester_command}")
    backtester_result = await backtester_agent.run(task=backtester_command)

    # === STEP 5: PARSE AND DISPLAY FINAL RESULTS ===
    if backtester_result is None or not hasattr(backtester_result, 'messages') or not backtester_result.messages:
        logging.error("Backtester task failed or produced no messages.")
        return

    final_backtester_content = backtester_result.messages[-1].content

    logging.info("\n--- Orchestrator: Workflow Complete. Final Results ---")
    if "Execution successful" not in final_backtester_content:
        logging.error("Backtest execution failed.")
        print(f"\nBacktester Error Report:\n{final_backtester_content}")
    else:
        try:
            json_str_match = re.search(r'\{.*\}', final_backtester_content, re.DOTALL)
            if json_str_match:
                parsed_json = json.loads(json_str_match.group(0))
                print("\n✅ Backtest Successful!")
                print(json.dumps(parsed_json, indent=4))
            else:
                logging.warning("Could not find JSON in the backtester output. Displaying raw output.")
                print(final_backtester_content)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON from backtester output.")
            print(f"Raw Backtester Output:\n{final_backtester_content}")


async def main():
    await run_trading_workflow()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
