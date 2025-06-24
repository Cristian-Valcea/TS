# agents/code_agent.py
import logging
import sys
import re
import os
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from pathlib import Path

# --- Project-Specific Imports ---
# This ensures we can import from the project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import shared directory paths and the LLM client factory
from config import get_llm_client, STRATEGIES_DIR

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.info("Attempting to configure CodeAgent...")

# =================================================================================
# 1. DEFINE THE TOOL FUNCTION
# This function allows the agent to save the code it writes into a file.
# =================================================================================

def save_strategy_code_to_file(file_name: str, code_content: str) -> str:
    """
    Saves the provided Python code into a file in the shared strategies directory.

    Args:
        file_name (str): The name for the file (e.g., "sma_crossover_strategy.py").
        code_content (str): The complete Python code to be saved.

    Returns:
        str: A message indicating success and the full file path, or an error message.
    """
    try:
        # Enhanced file name validation
        if not file_name or not isinstance(file_name, str):
            return "Error: File name must be a non-empty string."
            
        # Ensure filename has .py extension
        if not file_name.endswith('.py'):
            file_name = f"{file_name}.py"
            
        # Sanitize filename - allow only alphanumeric, underscore, dash and period
        safe_name = re.sub(r'[^\w\-\.]', '_', file_name)
        if safe_name != file_name:
            logger.warning(f"Sanitized file name from '{file_name}' to '{safe_name}'")
            file_name = safe_name
            
        # Ensure we're not allowing directory traversal
        if ".." in file_name or "/" in file_name or "\\" in file_name:
            return "Error: Invalid file name. It cannot contain path traversal elements."

        # Validate code content
        if not code_content or not isinstance(code_content, str):
            return "Error: Code content must be a non-empty string."

        # Basic Python syntax validation
        try:
            compile(code_content, '<string>', 'exec')
        except SyntaxError as se:
            return f"Error: The provided code contains syntax errors: {se}"

        # Extract strategy class name for the success message
        class_match = re.search(r'class\s+(\w+)\s*\(', code_content)
        strategy_name = class_match.group(1) if class_match else "Unknown"

        full_path = STRATEGIES_DIR / file_name
        logger.info(f"CodeAgent Tool: Attempting to save code to {full_path}")
        
        # Create backup if file exists
        if full_path.exists():
            backup_path = full_path.with_suffix(f".bak.{int(Path.now().timestamp())}.py")
            full_path.rename(backup_path)
            logger.info(f"Created backup of existing file at {backup_path}")
        
        with open(full_path, "w") as f:
            f.write(code_content)
        
        success_message = f"Successfully saved strategy '{strategy_name}' to '{full_path}'"
        logger.info(success_message)
        return success_message

    except Exception as e:
        error_message = f"Failed to save code to file '{file_name}'. Error: {e}"
        logger.error(error_message, exc_info=True)
        return error_message

# =================================================================================
# 2. WRAP THE FUNCTION IN A FunctionTool
# =================================================================================

code_saving_tool = FunctionTool(
    save_strategy_code_to_file,
    name="save_strategy_code_to_file",
    description="Saves a string of Python code into a specified file within the strategies directory."
)

# =================================================================================
# 3. DEFINE THE AGENT'S SYSTEM MESSAGE
# =================================================================================

code_agent_system_message = f"""You are an expert-level Code Agent that writes and saves Python scripts for the Backtrader framework.

**CRITICAL INSTRUCTIONS:**
1.  Your primary task is to write a **single, complete, self-contained Python script** based on the user's request.
2.  The script **MUST** be self-contained. It must include all necessary imports (e.g., `backtrader`, `yfinance`, `pandas`, `json`).
3.  The script **MUST** define a strategy class with the exact name requested (e.g., 'SmaCrossover') and nothing else no backtracker no result formatting
    NO DQATA LOADING from CSV (replace with your actual file path)
    NO Cerebro
    NO results processing or printing
4.  Once you have composed the complete script, you **MUST** use your `save_strategy_code_to_file` tool.
5.  **This is your final step. After calling the tool, you MUST respond with ONLY the exact output from that tool and nothing else. Do not add any extra words, conversation, or summaries.**

**EXAMPLE of your final response:**
"Successfully saved strategy 'SmaCrossover' to 'C:\\...\\sma_crossover.py'"

**EXAMPLE SCRIPT STRUCTURE:**

```python
import backtrader as bt
import yfinance as yf
import json

class SmaCrossover(bt.Strategy):
    # ... strategy logic ...
```
"""

# =================================================================================
# 4. INITIALIZE AND EXPORT THE AGENT WITH FALLBACK CONFIGURATION
# =================================================================================

logger.info("Attempting to configure and initialize CodeAgent...")

# Add fallback configuration in case primary config fails
DEFAULT_LLM_CONFIG = {
    "config_list": [{"model": "gpt-3.5-turbo"}],
    "temperature": 0.2,
    "cache_seed": 42
}

llm_config_or_client = get_llm_client("CodeAgent")

init_kwargs = {
    "name": "CodeAgent",
    "description": "Writes and saves Backtrader strategy scripts to files.",
    "system_message": code_agent_system_message,
    "tools": [code_saving_tool],
}

if llm_config_or_client is None:
    logger.warning("Primary LLM config for CodeAgent is None. Attempting fallback configuration.")
    
    # Check environment variables for API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    
    if openai_key:
        logger.info("Using OpenAI fallback configuration")
        init_kwargs["llm_config"] = DEFAULT_LLM_CONFIG
    elif gemini_key:
        logger.info("Using Gemini fallback configuration")
        try:
            from autogen.oai.client import ModelClient
            from autogen.oai.openai import OpenAIWrapper
            
            gemini_config = {
                "config_list": [{"model": "gemini-1.5-flash", "api_key": gemini_key}]
            }
            init_kwargs["llm_config"] = gemini_config
        except ImportError:
            logger.error("Failed to import necessary modules for Gemini fallback")
            raise ValueError("CodeAgent LLM config is None and fallback configuration failed.")
    else:
        logger.critical("CRITICAL: No API keys found for fallback configuration.")
        raise ValueError("CodeAgent LLM config is None and no API keys available for fallback.")
else:
    if isinstance(llm_config_or_client, dict):
        init_kwargs["llm_config"] = llm_config_or_client
    else:
        init_kwargs["model_client"] = llm_config_or_client

try:
    code_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ CodeAgent '{code_agent.name}' initialized successfully.")

    # Verify agent configuration
    logger.debug(f"CodeAgent configuration: Name={code_agent.name}")
    if hasattr(code_agent, '_model_client') and code_agent._model_client is not None:
        logger.debug(f"CodeAgent model client info: {getattr(code_agent._model_client, 'model_info', 'NO_MODEL_INFO')}")
    else:
        logger.debug("CodeAgent using llm_config (no model client)")
        
except Exception as e:
    logger.error(f"Error initializing CodeAgent: {e}", exc_info=True)
    logger.error(f"LLM Client passed to CodeAgent was: {llm_config_or_client}")
    raise






















if llm_config_or_client is None:
    logger.critical("CRITICAL (CodeAgent): get_llm_client returned None.")
    raise ValueError("CodeAgent LLM config is None.")
else:
    if isinstance(llm_config_or_client, dict):
        init_kwargs["llm_config"] = llm_config_or_client
    else:
        init_kwargs["model_client"] = llm_config_or_client

try:
    code_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ CodeAgent '{code_agent.name}' initialized successfully.")

    logger.info("CodeAgent initialized successfully.")
    logger.critical(f"CODE_AGENT (POST-INIT): Name: {code_agent.name}")
    logger.critical(f"CODE_AGENT (POST-INIT): _model_client IS: {getattr(code_agent, '_model_client', 'NOT_FOUND')}")
    if hasattr(code_agent, '_model_client') and code_agent._model_client is not None:
        logger.critical(f"CODE_AGENT (POST-INIT): _model_client.model_info IS: {getattr(code_agent._model_client, 'model_info', 'NOT_FOUND_ON_CLIENT')}")
    else:
        logger.error("CODE_AGENT (POST-INIT): _model_client is None or not found after init!")

except Exception as e:
    logger.error(f"Error initializing CodeAgent: {e}")
    logger.error(f"LLM Client passed to CodeAgent was: {llm_config_or_client}")
    raise





old_system_message="""You are a Python code generation expert specializing in financial trading algorithms.
        Your primary role is to write robust, self-contained Python scripts for backtesting trading strategies.
        Once you have composed the complete script, you **MUST** use your `save_strategy_code_to_file` tool.

        **INSTRUCTIONS:**
        1.  Based on the user's request or the ongoing conversation, write a complete Python script.
        2.  The script MUST be self-contained. It needs all necessary imports (e.g., `backtrader`, `pandas`, `yfinance`, `json`).
        3.  The script MUST print its final performance metrics as a JSON string to standard output. This is critical for the other agents to parse the results.
        4.  Enclose the final Python code in a single markdown block (```python ... ```).
        5.  Do not provide explanations before or after the code block, just the code itself.
        6.  Once you have composed the complete script, you **MUST** use your `save_strategy_code_to_file` tool.
        7.  **This is your final step. After calling the tool, you MUST respond with ONLY the exact output from that tool and nothing else. Do not add any extra words, conversation, or summaries.**

        **EXAMPLE of final print statement in your code:**
        ```python
        # ... at the end of the backtesting script ...
        import json
        final_portfolio_value = cerebro.broker.getvalue()
        # Add other metrics as needed
        results = {'final_portfolio_value': final_portfolio_value}
        print(json.dumps(results, indent=2)) 
        The strategy class should be named MyGeneratedStrategy. 
        Save the strategy to 'common_logic/strategies/generated_strategy.py' 
        and parameters to 'common_logic/strategies/generated_strategy_params.json'. 
        State these file paths in your response.
        Remember *** Once you have composed the complete script, you **MUST** use your `save_strategy_code_to_file` tool.
        """
