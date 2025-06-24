# config.py (Final, Cleaned Version)

import os
import logging
from pathlib import Path
from typing import Any, Dict
import time
import asyncio

# --- Autogen Imports ---
try:
    from autogen_core.models import ChatCompletionClient
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    logging.info("Successfully imported core Autogen clients.")
except ImportError as e:
    logging.critical(f"Failed to import essential Autogen components: {e}")
    raise

# --- Centralized Path Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent
SHARED_WORK_DIR = PROJECT_ROOT / "shared_work_dir"
DATA_DIR = SHARED_WORK_DIR / "data"
STRATEGIES_DIR = SHARED_WORK_DIR / "strategies"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STRATEGIES_DIR, exist_ok=True)
logging.info(f"Shared workspace initialized at: {SHARED_WORK_DIR}")

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- GLOBAL RATE LIMITING SOLUTION ---
class ThrottledClientWrapper:
    _last_request_time: float = 0
    
    def __init__(self, client: ChatCompletionClient, model_name: str, seconds_per_request: float = 2.0):
        self._real_client = client
        self._model_name = model_name
        self._seconds_per_request = seconds_per_request
        logging.info(f"Initialized ThrottledClientWrapper for '{self._model_name}' with a {self._seconds_per_request}s delay.")

    async def create(self, *args, **kwargs) -> Any:
        current_time = time.monotonic()
        time_since_last_request = current_time - ThrottledClientWrapper._last_request_time
        if time_since_last_request < self._seconds_per_request:
            wait_time = self._seconds_per_request - time_since_last_request
            logging.info(f"Rate limit throttle: waiting for {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
        ThrottledClientWrapper._last_request_time = time.monotonic()
        return await self._real_client.create(*args, **kwargs)

    @property
    def model_info(self) -> Dict[str, Any]:
        return self._real_client.model_info

# --- Simplified Agent LLM Mappings ---
# Only contains agents that actually use an LLM in the orchestrator workflow.
AGENT_LLM_MAPPINGS = {
    "DataProvisioningAgent": { # <<< NEW AGENT CONFIGURATION
        "model": "gemini-1.5-flash-latest",
        "model_info": {
            "family": "gemini",
            "vision": False,
            "function_calling": True,
            "structured_output": True, 
            "json_output": True, 
            "max_tokens": 8192
        }
    },
    "InstantCodeAgent": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "CodeAgent": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "BacktesterAgent": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "ReviewerAgent": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "StrategyDesignerAgent": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "StrategyRaterAgent": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "BAY_Agent": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },


    "UniverseSelector": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "TimeWindowPlanner": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "EventTagger": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "SamplerSplitter": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },
    "IBKRFetcherValidator": {
        "model": "gemini-1.5-flash-latest",
        "model_info": {"family": "gemini", "vision": False, "function_calling": True, "json_output": True, "structured_output": True, "max_tokens": 8192}
    },

}


def get_llm_client(agent_name: str):
    """Creates and returns a throttled Gemini client."""
    agent_config = AGENT_LLM_MAPPINGS.get(agent_name)
    if not agent_config:
        logging.error(f"No LLM configuration found for agent: '{agent_name}'")
        return None

    if not GEMINI_API_KEY:
        logging.error(f"GEMINI_API_KEY not found in environment for agent '{agent_name}'.")
        return None
        
    model_name = agent_config["model"]
    model_info = agent_config.get("model_info", {})

    logging.info(f"Configuring LLM for '{agent_name}': Provider='gemini', Model='{model_name}'")
    
    try:
        real_client = OpenAIChatCompletionClient(model=model_name, api_key=GEMINI_API_KEY, model_info=model_info)
        throttled_client = ThrottledClientWrapper(client=real_client, model_name=model_name)
        logging.info(f"  Successfully configured Throttled Gemini client for '{agent_name}'.")
        return throttled_client
    except Exception as e:
        logging.exception(f"  Failed to initialize Gemini client for '{agent_name}'.")
        return None
