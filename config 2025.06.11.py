# config.py (Final, Comprehensive, Corrected Version)

import os
import logging
from pathlib import Path
import time
import asyncio
from typing import Any, Dict

# --- Autogen Imports ---
try:
    from autogen_core.models import ChatCompletionClient
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    logging.info("Successfully imported core Autogen clients.")
except ImportError as e:
    logging.critical(f"Failed to import essential Autogen components: {e}")
    logging.critical("Please ensure 'autogen-ext[ollama,openai]' is installed correctly.")
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

# =================================================================================
# GLOBAL RATE LIMITING SOLUTION
# =================================================================================
class ThrottledClientWrapper:
    _last_request_time: float = 0

    def __init__(self, client: ChatCompletionClient, model_name: str, seconds_per_request: float = 2.0):
        self._real_client = client
        self._model_name = model_name
        self._seconds_per_request = seconds_per_request
        logging.info(f"Initialized ThrottledClientWrapper for model '{self._model_name}' with a {self._seconds_per_request}s delay.")

    async def create(self, *args, **kwargs) -> Any:
        current_time = time.monotonic()
        time_since_last_request = current_time - ThrottledClientWrapper._last_request_time
        if time_since_last_request < self._seconds_per_request:
            wait_time = self._seconds_per_request - time_since_last_request
            logging.info(f"Rate limit throttle: waiting for {wait_time:.2f} seconds before calling {self._model_name}.")
            await asyncio.sleep(wait_time)
        ThrottledClientWrapper._last_request_time = time.monotonic()
        return await self._real_client.create(*args, **kwargs)

    @property
    def model_info(self) -> Dict[str, Any]:
        return self._real_client.model_info
# =================================================================================

# --- Agent LLM Configuration Mapping (FINAL, COMPREHENSIVE VERSION) ---
# Providing the full set of required keys for model_info to satisfy validation.
AGENT_LLM_MAPPINGS = {
    "CodeAgent": {
        "type": "gemini",
        "model": "gemini-1.5-flash-latest",
        "model_info": {
            "family": "gemini",
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "max_tokens": 8192
        }
    },
    "BacktesterAgent": {
        "type": "gemini",
        "model": "gemini-1.5-flash-latest",
        "model_info": {
            "family": "gemini",
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "max_tokens": 8192
        }
    },
    "DataProvisioningAgent": {
        "type": "gemini",
        "model": "gemini-1.5-flash-latest",
        "model_info": {
            "family": "gemini",
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "max_tokens": 8192
        }
    },
}

def get_llm_client(agent_name: str):
    """
    Creates and returns an LLM client based on the agent's configuration.
    """
    agent_config = AGENT_LLM_MAPPINGS.get(agent_name)
    if not agent_config:
        logging.error(f"No LLM configuration found for agent: '{agent_name}'")
        return None

    provider_type = agent_config.get("type")
    model_name = agent_config.get("model")
    model_info = agent_config.get("model_info", {})

    logging.info(f"Configuring LLM for '{agent_name}': Provider='{provider_type}', Model='{model_name}'")

    if provider_type == "gemini":
        if not GEMINI_API_KEY:
            logging.error(f"GEMINI_API_KEY environment variable not found or is empty for agent '{agent_name}'.")
            return None
        try:
            real_client = OpenAIChatCompletionClient(
                model=model_name,
                api_key=GEMINI_API_KEY,
                model_info=model_info
            )
            throttled_client = ThrottledClientWrapper(client=real_client, model_name=model_name)
            logging.info(f"  Successfully configured Throttled Gemini client for '{agent_name}'.")
            return throttled_client
        except Exception as e:
            logging.error(f"  An exception occurred while initializing the Gemini client for '{agent_name}'.")
            logging.exception(e)
            return None
    else:
        logging.error(f"Unsupported LLM provider type '{provider_type}' for agent '{agent_name}'.")
        return None