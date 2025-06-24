# config.py (Simplified Version)
import os
import logging
from pathlib import Path
from typing import Any, Dict
import time
import asyncio

#from autogen_core.models import ModelInfo
from autogen_core.models import  ChatCompletionClient

# --- Centralized Path Configuration ---
# Define the project root as the parent directory of this config file
PROJECT_ROOT = Path(__file__).resolve().parent
# Define the shared workspace where agents will read/write files
SHARED_WORK_DIR = PROJECT_ROOT / "shared_work_dir"
DATA_DIR = SHARED_WORK_DIR / "data"
STRATEGIES_DIR = SHARED_WORK_DIR / "strategies"
# Create the directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STRATEGIES_DIR, exist_ok=True)
logging.info(f"Project Root set to: {PROJECT_ROOT}")
logging.info(f"Shared workspace directories initialized at: {SHARED_WORK_DIR}")


# Add error handling to the ModelInfo import
try:
    from autogen_core.models import ModelInfo
    MODEL_INFO_AVAILABLE = True
    logging.info("Successfully imported ModelInfo from autogen_core.models")
except ImportError as e:
    logging.error(f"Failed to import ModelInfo from autogen_core.models: {e}. "
                  "Ensure 'autogen_core' is installed correctly and dependencies are aligned.")
    MODEL_INFO_AVAILABLE = False
    ModelInfo = None  # Provide a placeholder

try:
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    AUTOGEN_EXT_OLLAMA_AVAILABLE = True
    logging.info("Successfully imported OllamaChatCompletionClient from autogen-ext.")
except ImportError as e:
    logging.error(f"Failed to import OllamaChatCompletionClient from autogen-ext: {e}. "
                  "Ensure 'autogen-ext[ollama]' is installed correctly and dependencies are aligned.")
    AUTOGEN_EXT_OLLAMA_AVAILABLE = False
    OllamaChatCompletionClient = None # Placeholder
try:
    from autogen_core.models import ModelFamily # Example path
    MODEL_FAMILY_AVAILABLE = True
    logging.info("Successfully imported ModelFamily from autogen_core.models")
except ImportError as e:
    logging.error(f"Failed to import ModelFamily from autogen-ext: {e}. "
                  "Ensure 'autogen-ext[ollama]' is installed correctly and dependencies are aligned.")
    MODEL_FAMILY_AVAILABLE = False
# After the ModelFamily import attempt
if not MODEL_FAMILY_AVAILABLE:
    # Create a placeholder
    class ModelFamily:
        R1 = "R1"
        # Add other model families as needed

try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_EXT_OPENAI_AVAILABLE = True
    logging.info("Successfully imported OpenAIChatCompletionClient from autogen-ext.")
except ImportError as e:
    logging.warning(f"OpenAIChatCompletionClient from autogen-ext not found: {e}.")
    AUTOGEN_EXT_OPENAI_AVAILABLE = False
    OpenAIChatCompletionClient = None # Placeholder


# =================================================================================
# GLOBAL RATE LIMITING SOLUTION (Final, Robust Version)
# =================================================================================
class ThrottledClientWrapper:
    """A robust wrapper that adds a delay to a real ChatCompletionClient."""
    _last_request_time: float = 0
    
    def __init__(self, client: ChatCompletionClient, model_name: str, seconds_per_request: float = 5.0):
        self._real_client = client
        self._model_name = model_name  # Store the model name explicitly
        self._seconds_per_request = seconds_per_request
        logging.info(f"Initialized ThrottledClientWrapper for model '{self._model_name}' with a {self._seconds_per_request}s delay.")

    async def create(self, *args, **kwargs) -> Any:
        current_time = time.monotonic()
        time_since_last_request = current_time - ThrottledClientWrapper._last_request_time
        
        if time_since_last_request < self._seconds_per_request:
            wait_time = self._seconds_per_request - time_since_last_request
            logging.info(f"Rate limit throttle: waiting for {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
        
        ThrottledClientWrapper._last_request_time = time.monotonic()
        logging.info("ThrottledClientWrapper: Delay complete, passing call to real client.")
        return await self._real_client.create(*args, **kwargs)

    # --- Delegate other necessary methods/properties to the real client ---
    def cost(self, *args, **kwargs) -> float:
        return self._real_client.cost(*args, **kwargs)
        
    def dump_component(self, *args, **kwargs) -> Dict[str, Any]:
        return self._real_client.dump_component(*args, **kwargs)

    @property
    def model_info(self) -> Dict[str, Any]:
        return self._real_client.model_info
# =================================================================================


# --- Global LLM Settings ---
USE_OVERALL_LOCAL_LLM = True # True to default to local Ollama, False to default to OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # For Gemini

#print(f"DEBUG: GEMINI_API_KEY from env: {GEMINI_API_KEY}") # <<< ADD THIS LINE
#print(f"DEBUG: OPENAI_API_KEY from env: {OPENAI_API_KEY}") # Good to check this


AGENT_LLM_MAPPINGS = {
    "CodeAgent": {
        "type": "gemini",
        "gemini_model": "gemini-1.5-flash-latest",
        "gemini_model_info": {
            "family": "gemini",
            "vision": False,
            "function_calling": True,
            "structured_output": True, 
            "json_output": True, 
            "max_tokens": 8192
        }
    },
    "DataProvisioningAgent": { # <<< NEW AGENT CONFIGURATION
        "type": "gemini",
        "gemini_model": "gemini-1.5-flash-latest",
        "gemini_model_info": {
            "family": "gemini",
            "vision": False,
            "function_calling": True,
            "structured_output": True, 
            "json_output": True, 
            "max_tokens": 8192
        }
    },
    "BacktesterAgent": { # <<< SWITCH TO GEMINI
        "type": "gemini",
        "gemini_model": "gemini-1.5-flash-latest", # Or "gemini-1.5-pro-latest"
        # As per docs, provide model_info for Gemini models, especially newer ones
        "gemini_model_info": {
            "vision": False, # gemini-1.5-flash can do vision, but set based on need
            "function_calling": True, # ESSENTIAL FOR TOOLS
            "json_output": True,    # Good if you expect JSON
            "family": "gemini",     # Or "unknown" if not sure what autogen-ext expects
            "structured_output": True,
            "max_tokens": 8192 # Check actual limits for the model
        }
    },
    "ReviewerAgent": { # <<< SWITCH TO GEMINI
        "type": "gemini",
        "gemini_model": "gemini-1.5-flash-latest", # Or "gemini-1.5-pro-latest"
        # As per docs, provide model_info for Gemini models, especially newer ones
        "gemini_model_info": {
            "vision": False, # gemini-1.5-flash can do vision, but set based on need
            "function_calling": True, # ESSENTIAL FOR TOOLS
            "json_output": True,    # Good if you expect JSON
            "family": "gemini",     # Or "unknown" if not sure what autogen-ext expects
            "structured_output": True,
            "max_tokens": 8192 # Check actual limits for the model
        }
    },
    "BO_Agent": {
        "type": "gemini",
        "gemini_model": "gemini-1.5-flash-latest", # Or "gemini-1.5-pro-latest"
        # As per docs, provide model_info for Gemini models, especially newer ones
        "gemini_model_info": {
            "vision": False, # gemini-1.5-flash can do vision, but set based on need
            "function_calling": True, # ESSENTIAL FOR TOOLS
            "json_output": True,    # Good if you expect JSON
            "family": "gemini",     # Or "unknown" if not sure what autogen-ext expects
            "structured_output": True,
            "max_tokens": 8192 # Check actual limits for the model
        }
    },
    "TradingAgent": { # Example: Switch to Gemini if it needs tools
        "type": "ollama",
        "ollama_model": "llama3:8b",
        "ollama_model_info": { # <<< ADD THIS
            "family": ModelFamily.R1, 
            "vision": False, 
            "function_calling": False, # Selector doesn't need tool use usually
            "structured_output": True, 
            "json_output": False,      # <<< ADD THIS (or True if needed and supported)
            "max_tokens": 4096,
        }
    },
    "UserProxy": { # UserProxyAgent usually takes llm_config for OpenAI
        "type": "gemini",
        "gemini_model": "gemini-1.5-flash-latest",
        "gemini_model_info": {
            "vision": False, "function_calling": True, "json_output": True,
            "family": "gemini", "structured_output": True, "max_tokens": 8192
        }
    },
    "Selector": {
        "type": "gemini",
        "gemini_model": "gemini-1.5-flash-latest",
        "gemini_model_info": {
            "vision": False, "function_calling": True, "json_output": True,
            "family": "gemini", "structured_output": True, "max_tokens": 8192
        }
    },
    "DEFAULT_CONFIG": { # Define fallbacks for each type
        "type": "ollama", # Default provider if "type" is missing or "default"
        "ollama_model": "llama3:8b",
        "ollama_model_info": {"family": ModelFamily.R1, "vision": False, "function_calling": False, "structured_output": True, "json_output": False, "max_tokens": 4096,},
        "openai_model": "gpt-3.5-turbo", # Fallback OpenAI model
        "gemini_model": "gemini-1.5-flash-latest", # Fallback Gemini model
        "gemini_model_info": {"vision": False, "function_calling": True, "json_output": True, "family": "gemini", "structured_output": True, "max_tokens": 8192}    
    }
}



def get_llm_client(agent_name_or_key: str = None):
    config_key = agent_name_or_key if agent_name_or_key in AGENT_LLM_MAPPINGS else "DEFAULT_CONFIG"
    agent_config = AGENT_LLM_MAPPINGS[config_key]
    default_config = AGENT_LLM_MAPPINGS["DEFAULT_CONFIG"]

    # Determine client type: "ollama", "openai", or "gemini"
    # If agent_config has "type", use it. Else use default_config's "type".
    client_provider_type = agent_config.get("type", default_config.get("type", "ollama")) # Default to ollama if all else fails

    logging.info(f"Configuring LLM for '{agent_name_or_key or 'DEFAULT'}': Provider='{client_provider_type}'")

    if client_provider_type == "ollama":
        if not AUTOGEN_EXT_OLLAMA_AVAILABLE or OllamaChatCompletionClient is None:
            logging.error("OllamaChatCompletionClient (autogen-ext) not available for local LLM.")
            return None
        
        model_name = agent_config.get("ollama_model", default_config["ollama_model"])
        base_url_to_use = agent_config.get("base_url", OLLAMA_BASE_URL)
        explicit_model_info = agent_config.get("ollama_model_info", default_config.get("ollama_model_info"))
        
        client_constructor_args = {"model": model_name, "base_url": base_url_to_use}
        if explicit_model_info: client_constructor_args["model_info"] = explicit_model_info
        
        logging.info(f"  Instantiating OllamaChatCompletionClient: {client_constructor_args}")
        try:
            client = OllamaChatCompletionClient(**client_constructor_args)
            logging.info(f"  Successfully configured autogen-ext Ollama client for '{model_name}'. Client's model_info: {getattr(client, 'model_info', 'N/A')}")
            return client
        except Exception as e:
            logging.exception(f"  Error initializing autogen-ext OllamaChatCompletionClient for {model_name}")
            # THIS LOG IS VERY IMPORTANT - IT WILL SHOW THE ACTUAL OLLAMA CLIENT INIT ERROR
            logging.error(f"--- START: Detailed Error initializing OllamaChatCompletionClient for model '{model_name}' ---")
            logging.exception(e) # This prints the full traceback of the caught exception
            logging.error(f"--- END: Detailed Error initializing OllamaChatCompletionClient for model '{model_name}' ---")
            logging.error(f"  Failed with constructor args: {client_constructor_args}")
            return None

    elif client_provider_type == "gemini":
        if not AUTOGEN_EXT_OPENAI_AVAILABLE or OpenAIChatCompletionClient is None or not MODEL_INFO_AVAILABLE:
            logging.error("OpenAIChatCompletionClient or ModelInfo (for Gemini) not available.")
            return None
        if not GEMINI_API_KEY:
            logging.error(f"GEMINI_API_KEY not configured (Agent: {agent_name_or_key}).")
            return None

        model_name = agent_config.get("gemini_model", default_config["gemini_model"])
        # The Gemini API endpoint is different from OpenAI's.
        # OpenAIChatCompletionClient needs to be told to use Gemini's endpoint.
        # This is often done by setting `base_url` or `api_base` to the Gemini endpoint.
        # Check autogen-ext docs for using OpenAIChatCompletionClient with non-OpenAI endpoints.
        # Example: base_url="https://generativelanguage.googleapis.com/v1beta" (or similar, check Gemini docs)
        # For some libraries, you might just provide the model name prefixed like "gemini/gemini-1.5-flash-latest"
        # and the client handles the endpoint.
        # The doc snippet you found for autogen-ext implies it handles it if `api_key` is set for Gemini.
        # It seems it might internally recognize "gemini-*" models.

        gemini_model_info_dict = agent_config.get("gemini_model_info", default_config.get("gemini_model_info"))
        gemini_model_info_obj = None
        if gemini_model_info_dict and ModelInfo is not None:
             try:
                gemini_model_info_obj = ModelInfo(**gemini_model_info_dict)
             except Exception as e:
                logging.error(f"Failed to create ModelInfo object for Gemini: {e}. Model_info dict was: {gemini_model_info_dict}")


        client_params = {
            "model": model_name, # e.g., "gemini-1.5-flash-latest"
            "api_key": GEMINI_API_KEY,
            # "base_url": "YOUR_GEMINI_API_ENDPOINT_IF_NEEDED_BY_CLIENT", # See note above
        }
        if gemini_model_info_obj:
            client_params["model_info"] = gemini_model_info_obj
        
        logging.debug(f"  Instantiating OpenAIChatCompletionClient for Gemini: {client_params}")
        try:
            # 1. Create the real client instance
            real_gemini_client = OpenAIChatCompletionClient(**client_params)
            logging.info(f"  Successfully configured base Gemini client for model '{model_name}'.")
            
            # 2. Wrap the real client in our ThrottledClientWrapper
            client = ThrottledClientWrapper(client=real_gemini_client,model_name=model_name)
            logging.info(f"  Successfully configured OpenAIChatCompletionClient for Gemini model '{model_name}'. Client's model_info: {getattr(client, 'model_info', 'N/A')}")
            return client
        except Exception as e:
            logging.exception(f"  Error initializing OpenAIChatCompletionClient for Gemini model {model_name}")
            return None

    elif client_provider_type == "openai":
        # For UserProxyAgent or explicit OpenAI choice
        if agent_name_or_key == "UserProxy" or not AUTOGEN_EXT_OPENAI_AVAILABLE or OpenAIChatCompletionClient is None:
            if not OPENAI_API_KEY:
                logging.error(f"OPENAI_API_KEY not configured for standard llm_config (Agent: {agent_name_or_key}).")
                return None
            model_name = agent_config.get("openai_model", default_config["openai_model"])
            timeout = agent_config.get("openai_timeout", default_config.get("openai_timeout", 60))
            logging.info(f"  Using standard AutoGen llm_config for OpenAI: model='{model_name}' (Agent: {agent_name_or_key})")
            return {
                "config_list": [{"model": model_name, "api_key": OPENAI_API_KEY, "timeout": int(timeout)}],
                "cache_seed": None
            }
        else: # Use autogen-ext OpenAIChatCompletionClient for other agents using OpenAI
            if not OPENAI_API_KEY:
                logging.error(f"OPENAI_API_KEY not configured for autogen-ext OpenAIChatCompletionClient (Agent: {agent_name_or_key}).")
                return None
            model_name = agent_config.get("openai_model", default_config["openai_model"])
            timeout = agent_config.get("openai_timeout", default_config.get("openai_timeout", 60))
            
            # model_info for OpenAI models is usually auto-detected by OpenAIChatCompletionClient
            # but you could provide it if needed for a new/unknown OpenAI model.
            # openai_model_info_dict = agent_config.get("openai_model_info") 
            # openai_model_info_obj = ModelInfo(**openai_model_info_dict) if openai_model_info_dict and ModelInfo else None

            client_params = {"model": model_name, "api_key": OPENAI_API_KEY, "timeout": timeout}
            # if openai_model_info_obj: client_params["model_info"] = openai_model_info_obj

            logging.info(f"  Instantiating OpenAIChatCompletionClient (autogen-ext) for OpenAI: {client_params}")
            try:
                client = OpenAIChatCompletionClient(**client_params)
                logging.info(f"  Successfully configured OpenAIChatCompletionClient for OpenAI model '{model_name}'. Client's model_info: {getattr(client, 'model_info', 'N/A')}")
                return client
            except Exception as e:
                logging.exception(f"  Error initializing autogen-ext OpenAIChatCompletionClient for OpenAI model {model_name}")
                return None
    
    logging.error(f"Could not determine a valid LLM client for agent '{agent_name_or_key or 'DEFAULT'}' with provider type '{client_provider_type}'.")
    return None





'''
def get_llm_client(agent_name_or_key: str = None):
    config_key = agent_name_or_key if agent_name_or_key in AGENT_LLM_MAPPINGS else "DEFAULT_CONFIG"
    agent_config = AGENT_LLM_MAPPINGS[config_key]
    default_config = AGENT_LLM_MAPPINGS["DEFAULT_CONFIG"]

    client_type_pref = agent_config.get("type", default_config["type"])
    use_local_llm = USE_OVERALL_LOCAL_LLM
    if client_type_pref == "ollama": use_local_llm = True
    elif client_type_pref == "openai": use_local_llm = False
    
    provider = "Ollama (autogen-ext)" if use_local_llm else "OpenAI"
    logging.info(f"Configuring LLM for '{agent_name_or_key or 'DEFAULT'}': Provider='{provider}'")

    if use_local_llm:
        if not AUTOGEN_EXT_OLLAMA_AVAILABLE or OllamaChatCompletionClient is None:
            logging.error("OllamaChatCompletionClient (autogen-ext) not available for local LLM.")
            return None
        
        model_name = agent_config.get("ollama_model", default_config["ollama_model"])
        # Get base_url from agent_config first, then global, then default for client
        base_url_to_use = agent_config.get("base_url", OLLAMA_BASE_URL)
        
        # Parameters for OllamaChatCompletionClient constructor
        # Check its definition for available parameters (e.g., timeout, specific ollama options)
        client_params = {"model": model_name, "base_url": base_url_to_use}
        # Example: client_params["request_timeout"] = agent_config.get("ollama_timeout", 120)
        ollama_explicit_model_info = agent_config.get("ollama_model_info") # Get from mapping

        logging.info(f"  Attempting Ollama (autogen-ext): model='{model_name}', base_url='{base_url_to_use}', explicit_model_info='{ollama_explicit_model_info}'")
        try:
                client_constructor_args = {
                    "model": model_name,
                    "base_url": base_url_to_use,
                }
                if ollama_explicit_model_info: # Only pass model_info if it's actually defined
                    client_constructor_args["model_info"] = ollama_explicit_model_info
                
                # You might also need to pass other parameters from agent_config directly
                # if OllamaChatCompletionClient supports them, e.g., timeout
                # request_timeout = agent_config.get("ollama_timeout", 120) # Example
                # if request_timeout: client_constructor_args["request_timeout"] = request_timeout


                client = OllamaChatCompletionClient(**client_constructor_args)
                logging.info(f"  Successfully configured autogen-ext Ollama client for '{model_name}'. Client's actual model_info: {client.model_info}")
                return client

#            return OllamaChatCompletionClient(**client_params)
        except Exception as e:
            logging.exception(f"  Error initializing autogen-ext OllamaChatCompletionClient for {model_name}")
            return None
    else: # OpenAI
        # For UserProxyAgent, we still prefer returning an llm_config dict
        if agent_name_or_key == "UserProxy" or not AUTOGEN_EXT_OPENAI_AVAILABLE or OpenAIChatCompletionClient is None:
            if not OPENAI_API_KEY:
                logging.error(f"OpenAI API key not configured for standard llm_config (Agent: {agent_name_or_key}).")
                return None
            model_name = agent_config.get("openai_model", default_config["openai_model"])
            timeout = agent_config.get("openai_timeout", default_config.get("openai_timeout", 60))
            logging.info(f"  Using standard AutoGen llm_config for OpenAI: model='{model_name}' (Agent: {agent_name_or_key})")
            return {
                "config_list": [{"model": model_name, "api_key": OPENAI_API_KEY, "timeout": int(timeout)}],
                "cache_seed": None
            }
        else: # Use autogen-ext OpenAIChatCompletionClient for other agents
            if not OPENAI_API_KEY:
                logging.error(f"OpenAI API key not configured for autogen-ext OpenAIChatCompletionClient (Agent: {agent_name_or_key}).")
                return None
            model_name = agent_config.get("openai_model", default_config["openai_model"])
            timeout = agent_config.get("openai_timeout", default_config.get("openai_timeout", 60))
            # Parameters for OpenAIChatCompletionClient constructor
            client_params = {"model": model_name, "api_key": OPENAI_API_KEY, "timeout": timeout}
            # Example: client_params["temperature"] = agent_config.get("openai_temperature", 0.7)

            logging.info(f"  Instantiating OpenAIChatCompletionClient (autogen-ext): {client_params}")
            try:
                return OpenAIChatCompletionClient(**client_params)
            except Exception as e:
                logging.exception(f"  Error initializing autogen-ext OpenAIChatCompletionClient for {model_name}")
                return None
    return None # Should not be reached
'''