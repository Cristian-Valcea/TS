# agents/data_provisioning_agent.py

# --- Python Standard Library Imports ---
import logging
import os
import sys
from pathlib import Path

# --- Third-party Imports ---
import yfinance as yf
import pandas as pd
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent

# --- Project-Specific Imports ---
# This ensures we can import the config module from the project root
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import get_llm_client

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# =================================================================================
# 1. DEFINE THE SHARED WORKSPACE
# This must be consistent across all agents that use the file system.
# =================================================================================
SHARED_WORK_DIR = project_root / "shared_work_dir"
DATA_DIR = SHARED_WORK_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)
logger.info(f"DataProvisioningAgent will save data to: {DATA_DIR}")


# =================================================================================
# 2. DEFINE THE TOOL FUNCTION
# This function downloads data using yfinance and saves it as a CSV.
# =================================================================================
def fetch_and_save_financial_data(
    ticker: str, 
    start_date: str, 
    end_date: str,
    interval: str = "1d"
) -> str:
    """
    Fetches historical financial data for a given ticker and saves it to a CSV file.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "GOOGL").
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        interval (str): The data interval (e.g., "1d" for daily, "1h" for hourly).

    Returns:
        str: A message indicating success and the file path, or an error message.
    """
    file_name = f"{ticker}_{start_date}_to_{end_date}_{interval}.csv"
    file_path = DATA_DIR / file_name
    
    logger.info(f"Data Tool: Attempting to fetch data for {ticker} and save to {file_path}")
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            error_message = f"No data found for ticker {ticker} in the specified date range."
            logger.warning(f"Data Tool: {error_message}")
            return error_message
            
        data.to_csv(file_path)
        success_message = f"Successfully downloaded data for {ticker} and saved it to '{file_path}'"
        logger.info(f"Data Tool: {success_message}")
        return success_message
        
    except Exception as e:
        error_message = f"Failed to fetch or save data for {ticker}. Error: {e}"
        logger.error(f"Data Tool: {error_message}", exc_info=True)
        return error_message

# =================================================================================
# 3. WRAP THE FUNCTION IN A FunctionTool
# =================================================================================

data_fetching_tool = FunctionTool(
    fetch_and_save_financial_data,
    name="fetch_and_save_financial_data",
    description="Fetches historical stock data and saves it to a local CSV file for other agents to use."
)

# =================================================================================
# 4. DEFINE THE AGENT'S SYSTEM MESSAGE
# =================================================================================

data_provisioner_system_message = f"""You are a Data Provisioning Agent. Your sole responsibility is to fetch financial data and save it to the shared workspace.

**INSTRUCTIONS:**
1.  When you receive a request for data (e.g., "get Apple stock from 2022 to 2023"), you MUST use your `fetch_and_save_financial_data` tool.
2.  Extract the necessary arguments (ticker, start_date, end_date) from the request to call the tool.
3.  After the tool runs, you MUST report the exact result back to the group. This will be a success message with the file path or an error message.
4.  Do not perform any other task. Do not analyze data, write code, or answer general questions. Your only function is to call your tool and report the result.
"""

# =================================================================================
# 5. INITIALIZE AND EXPORT THE AGENT
# =================================================================================

logger.info("Attempting to configure and initialize DataProvisioningAgent...")

llm_config_or_client = get_llm_client("DataProvisioningAgent") # You will need to map this in your config

init_kwargs = {
    "name": "DataProvisioningAgent",
    "description": "Fetches and saves financial data to files.",
    "system_message": data_provisioner_system_message,
    "tools": [data_fetching_tool],
}

if llm_config_or_client is None:
    logger.critical("CRITICAL (DataProvisioningAgent): get_llm_client returned None.")
    raise ValueError("DataProvisioningAgent LLM config is None.")
else:
    if isinstance(llm_config_or_client, dict):
        init_kwargs["llm_config"] = llm_config_or_client
    else:
        init_kwargs["model_client"] = llm_config_or_client

try:
    data_provisioning_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ DataProvisioningAgent '{data_provisioning_agent.name}' initialized successfully.")
except Exception as e:
    logger.error(f"❌ FAILED to initialize DataProvisioningAgent: {e}", exc_info=True)
    raise
