# config.py (Simplified Version)
import os
import logging

from autogen_core.models import ModelInfo,ModelFamily

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

try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_EXT_OPENAI_AVAILABLE = True
    logging.info("Successfully imported OpenAIChatCompletionClient from autogen-ext.")
except ImportError as e:
    logging.warning(f"OpenAIChatCompletionClient from autogen-ext not found: {e}.")
    AUTOGEN_EXT_OPENAI_AVAILABLE = False
    OpenAIChatCompletionClient = None # Placeholder



# --- Global LLM Settings ---
USE_OVERALL_LOCAL_LLM = True # True to default to local Ollama, False to default to OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # For Gemini

#print(f"DEBUG: GEMINI_API_KEY from env: {GEMINI_API_KEY}") # <<< ADD THIS LINE
#print(f"DEBUG: OPENAI_API_KEY from env: {OPENAI_API_KEY}") # Good to check this

# You might need ModelFamily from autogen_ext if you use it for ollama_model_info
try:
    from autogen_ext.models.ollama import ModelFamily # If you want to use this enum
    MODEL_FAMILY_AVAILABLE = True
except ImportError:
    MODEL_FAMILY_AVAILABLE = False
    class ModelFamily: R1 = "R1" # Dummy fallback
    logging.warning("autogen_ext.models.ollama.ModelFamily not found. Using dummy.")


AGENT_LLM_MAPPINGS = {
    "CodeAgent": {
        "type": "ollama",
        "ollama_model": "gemma3:1b",
        "ollama_model_info": { # <<< ADD THIS
            "family": ModelFamily.R1, 
            "vision": False, 
            "function_calling": False, # Selector doesn't need tool use usually
            "structured_output": True, 
            "json_output": False,      # <<< ADD THIS (or True if needed and supported)
            "max_tokens": 4096,
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
    "BO_Agent": {
        "type": "ollama",
        "ollama_model": "gemma3:1b",
        "ollama_model_info": { # <<< ADD THIS
            "family": ModelFamily.R1, 
            "vision": False, 
            "function_calling": False,
            "structured_output": True, 
            "json_output": False,      # <<< ADD THIS (or True if needed and supported)
            "max_tokens": 4096,
        }
    },
    "TradingAgent": { # Example: Switch to Gemini if it needs tools
        "type": "ollama",
        "ollama_model": "gemma3:1b",
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
        "type": "ollama",
        "ollama_model": "gemma3:1b",
        "ollama_model_info": { # <<< ADD THIS
            "family": ModelFamily.R1, 
            "vision": False, 
            "function_calling": False, # Selector doesn't need tool use usually
            "structured_output": True, 
            "json_output": False,      # <<< ADD THIS (or True if needed and supported)
            "max_tokens": 4096,
        }
    },
    "DEFAULT_CONFIG": { # Define fallbacks for each type
        "type": "ollama", # Default provider if "type" is missing or "default"
        "ollama_model": "gemma3:1b",
        "ollama_model_info": {"family": ModelFamily.R1, "vision": False, "function_calling": False, "structured_output": True, "json_output": False, "max_tokens": 4096,},
        "openai_model": "gpt-3.5-turbo", # Fallback OpenAI model
        "gemini_model": "gemini-1.5-flash-latest", # Fallback Gemini model
        "gemini_model_info": {"vision": False, "function_calling": True, "json_output": True, "family": "gemini", "structured_output": True, "max_tokens": 8192}    
    }
}
'''
logging.basicConfig(
    level=logging.INFO, # Or logging.DEBUG for more verbosity
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - Line:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler() # Ensures output to console
    ],
    force=True
)
'''
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
        if not AUTOGEN_EXT_OPENAI_AVAILABLE or OpenAIChatCompletionClient is None or ModelInfo is None:
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
            client = OpenAIChatCompletionClient(**client_params)
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