# config.py
import os
import logging
import time
import requests # For Ollama health check
import json # For your custom OllamaClient
import aiohttp 

from autogen_core.models import ModelResult, MessageContent, AssistantMessage, ModelUsage # IMPORT THESE

# --- Attempt to import from autogen_ext ---
AUTOGEN_EXT_OLLAMA_CLIENT_AVAILABLE = False
AUTOGEN_EXT_OPENAI_CLIENT_AVAILABLE = False
OllamaChatCompletionClient = None
OpenAIChatCompletionClient = None
ModelFamily = None # This was also from autogen_ext

try:
    from autogen_ext.models.ollama import OllamaChatCompletionClient, ModelFamily
    AUTOGEN_EXT_OLLAMA_CLIENT_AVAILABLE = True
    logging.info("Successfully imported OllamaChatCompletionClient and ModelFamily from autogen_ext.")
except ImportError:
    logging.warning("autogen_ext.models.ollama.OllamaChatCompletionClient not found. Will use custom Ollama client if needed.")

try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_EXT_OPENAI_CLIENT_AVAILABLE = True
    logging.info("Successfully imported OpenAIChatCompletionClient from autogen_ext.")
except ImportError:
    logging.warning("autogen_ext.models.openai.OpenAIChatCompletionClient not found. Will use standard AutoGen llm_config for OpenAI if needed.")


# config.py

# ... (other imports at the top of config.py: os, logging, time, json)
import aiohttp # Ensure this is imported

# ... (AUTOGEN_EXT_... flag definitions, ModelFamily placeholder if needed) ...

class CustomOllamaClient:
    def __init__(self, model_name="tinyllama", base_url="http://localhost:11434", model_info=None, **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.name = model_name # For AutoGen compatibility
        self.model_info = model_info if model_info is not None else {"name": model_name}
        # Determine function_calling based on model_info if provided, else default
        self.function_calling = self.model_info.get("function_calling", True) if isinstance(self.model_info, dict) else True
        logging.info(f"CustomOllamaClient initialized: model={self.model_name}, base_url={self.base_url}, function_calling={self.function_calling}")

    def __getitem__(self, key):
        # For compatibility with how AutoGen might access client properties
        if key == "name":
            return self.model_name
        elif key == "model_info":
            return self.model_info
        elif key == "function_calling":
            return self.function_calling
        logging.warning(f"CustomOllamaClient.__getitem__ called with unhandled key: {key}")
        raise KeyError(f"Unsupported key requested in CustomOllamaClient: {key}")
    
    async def create(self, messages, **kwargs):
        """
        Generates a response from the Ollama model using the /api/chat endpoint asynchronously.
        Converts message objects to dictionaries before sending and includes detailed logging.
        """
        # --- Attempt to import SystemMessage for specific handling ---
        # This import is scoped within the method if you prefer, or can be at module level.
        try:
            from autogen_core.models._types import SystemMessage
            AUTOGEN_CORE_SYSTEM_MESSAGE_AVAILABLE = True
        except ImportError:
            SystemMessage = None # Define it as None if not found
            AUTOGEN_CORE_SYSTEM_MESSAGE_AVAILABLE = False
            logging.debug("autogen_core.models._types.SystemMessage not found for specific handling in CustomOllamaClient.create.")

        # --- Convert message objects to plain dictionaries ---
        processed_messages = []
        for msg in messages:
            role = "user" # Default role
            content = ""  # Default content
            msg_dict_to_add = {}

            if isinstance(msg, dict):
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", ""))
                msg_dict_to_add = {"role": role, "content": content}
                # Carry over other relevant fields if they exist
                if "name" in msg: msg_dict_to_add["name"] = str(msg["name"])
                if "tool_calls" in msg: msg_dict_to_add["tool_calls"] = msg["tool_calls"] # Assume serializable

            elif AUTOGEN_CORE_SYSTEM_MESSAGE_AVAILABLE and isinstance(msg, SystemMessage):
                role = "system"
                content = str(msg.content) if msg.content is not None else ""
                msg_dict_to_add = {"role": role, "content": content}
                logging.debug(f"Processed autogen_core SystemMessage (content snippet): {content[:100]}...")

            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                role = str(msg.role) if msg.role is not None else "user"
                content = str(msg.content) if msg.content is not None else ""
                
                # Heuristic: if AutoGen uses a class named 'SystemMessage' and role isn't 'system'
                if type(msg).__name__ == 'SystemMessage' and role != 'system':
                    role = 'system'
                
                msg_dict_to_add = {"role": role, "content": content}
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    msg_dict_to_add["tool_calls"] = msg.tool_calls # Assume serializable
                if hasattr(msg, 'name') and msg.name:
                    msg_dict_to_add["name"] = str(msg.name)
            else:
                # Fallback for completely unknown message types
                content = str(msg)
                logging.warning(f"CustomOllamaClient: Encountered an unknown message type: {type(msg)}. "
                                f"Using fallback: role='user', content='{content[:100]}...'")
                msg_dict_to_add = {"role": "user", "content": content}
            
            # Basic validation: ensure role is not empty
            if not msg_dict_to_add.get("role"):
                msg_dict_to_add["role"] = "user" # Final fallback for role

            processed_messages.append(msg_dict_to_add)

        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "stream": False # Not using streaming for this client
        }
        
        # Allow overriding Ollama options via kwargs if needed for advanced use
        # Example: temperature, top_p, seed etc.
        # These are often passed under an 'options' key in the payload for Ollama
        ollama_options = kwargs.get("options", {}) # Get existing options from kwargs
        if 'temperature' in kwargs: ollama_options['temperature'] = kwargs['temperature']
        if 'seed' in kwargs: ollama_options['seed'] = kwargs['seed']
        # Add more specific kwargs handling if needed

        if ollama_options: # Only add 'options' to payload if there are any
            payload['options'] = ollama_options

        api_url = f"{self.base_url}/api/chat"
        
        # --- CRITICAL LOGGING before sending the request ---
        logging.info(f"CustomOllamaClient: Sending request to {api_url}. Model: {self.model_name}.")
        try:
            # Use json.dumps for logging to see the exact string being formed for the HTTP body
            logging.info(f"CustomOllamaClient: Full payload being sent: {json.dumps(payload, indent=2)}")
        except TypeError as te:
            logging.error(f"CustomOllamaClient: Could not JSON serialize payload for logging: {te}")
            logging.info(f"CustomOllamaClient: Raw payload (problematic part might be in processed_messages): {payload}")
        # --- END CRITICAL LOGGING ---

        error_message = "Ollama call failed with an unspecified error." # Default error message
        response_json = None # Initialize to handle cases where response_json is not set

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    response_status = response.status
                    response_text_content = await response.text() # Read the full text content for logging/parsing

                    logging.info(f"CustomOllamaClient: Ollama response status: {response_status}")
                    logging.info(f"CustomOllamaClient: Ollama response text snippet: {response_text_content[:500]}...")

                    response.raise_for_status() # Will raise ClientResponseError for 4xx/5xx
                    
                    # If raise_for_status didn't throw, we have a 2xx response
                    try:
                        response_json = json.loads(response_text_content) # Parse the text we already read
                    except json.JSONDecodeError as jde:
                        logging.error(f"CustomOllamaClient (async): Error decoding JSON from successful (2xx) Ollama response: {jde}. Response text: {response_text_content}")
                        error_message = f"Error decoding JSON from successful Ollama response: {str(jde)}"
                        # Proceed to return error structure, as we can't parse the good response
                        return {
                            "choices": [{"index": 0, "message": {"role": "assistant", "content": error_message, "tool_calls": None},"finish_reason": "error",}],
                            "usage": None, "model": self.model_name,
                        }


            # --- Process successful response_json ---
            generated_message_obj = response_json.get("message", {})
            generated_text = generated_message_obj.get("content", "")
            tool_calls_from_ollama = generated_message_obj.get("tool_calls") # For future tool use
            # CONSTRUCT THE DICT TO RETURN
            client_response_dict = {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": generated_text,
                            "tool_calls": tool_calls_from_ollama,
                        },
                        "finish_reason": response_json.get("done_reason", "stop"),
                    }
                ],
                "usage": {
                    "prompt_tokens": response_json.get("prompt_eval_count", 0),
                    "completion_tokens": response_json.get("eval_count", 0),
                    "total_tokens": response_json.get("prompt_eval_count", 0) + response_json.get("eval_count", 0)
                },
                "model": response_json.get("model", self.model_name),
                # Optional: Add id, created, system_fingerprint if AssistantAgent expects them, though usually not critical
                # "id": "chatcmpl-" + str(uuid.uuid4()), # Example
                # "created": int(time.time()),          # Example
            }
            logging.info(f"CustomOllamaClient: Successfully processed response. Returning dict: {json.dumps(client_response_dict, indent=2)}")
            return client_response_dict
            
            '''
            # old return >>>>>>>>>>>>>>>
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant", # Ollama's /api/chat response.message.role is 'assistant'
                            "content": generated_text,
                            "tool_calls": tool_calls_from_ollama, # Pass along if present
                        },
                        "finish_reason": response_json.get("done_reason", "stop"), # Use done_reason from Ollama if available
                    }
                ],
                "usage": {
                    "prompt_tokens": response_json.get("prompt_eval_count", 0),
                    "completion_tokens": response_json.get("eval_count", 0),
                    "total_tokens": response_json.get("prompt_eval_count", 0) + response_json.get("eval_count", 0)
                },
                "model": response_json.get("model", self.model_name), # Use model from response if available
            }
            '''
        
        except aiohttp.ClientResponseError as e:
            # This error means the server responded with 4xx or 5xx
            logging.error(f"CustomOllamaClient (async): HTTP {e.status} Error from {api_url}: {e.message}. "
                          f"Response body: {response_text_content[:500] if 'response_text_content' in locals() else 'Could not read body'}")
            error_message = f"Ollama API Error ({e.status}): {e.message}. Body: {response_text_content[:200] if 'response_text_content' in locals() else 'N/A'}"
        except aiohttp.ClientError as e:
            # Other client errors (connection, timeout before response, SSL issues etc.)
            logging.error(f"CustomOllamaClient (async): Client Connection/Timeout Error for {api_url}: {e}")
            error_message = f"Async HTTP Connection/Timeout Error for Ollama: {str(e)}"
        except json.JSONDecodeError as e:
            # This would only happen if json.loads(payload_for_logging) failed, which is unlikely here now
            # but kept for completeness if other json ops were added.
            logging.error(f"CustomOllamaClient (async): Internal JSON processing error: {e}.")
            error_message = f"Async Internal JSON error: {str(e)}"
        except Exception as e:
            # Catch-all for any other unexpected error during the process
            logging.exception(f"CustomOllamaClient (async): Unexpected critical error during 'create'. Payload logged above.") # Use logging.exception to include stack trace
            error_message = f"Async Unexpected critical error with Ollama: {type(e).__name__} - {str(e)}"
        
        # Fallback error response structure if any exception occurred above
        return {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": error_message, "tool_calls": None},"finish_reason": "error",}],
            "usage": None, "model": self.model_name,
        }
        
# ... (rest of config.py: AGENT_LLM_MAPPINGS, get_llm_client, etc.) ...# --- Global LLM Settings ---
USE_OVERALL_LOCAL_LLM = True
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE_ENV_NOT_SET") # Set your key or use env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Define a placeholder ModelFamily if autogen_ext isn't available
if ModelFamily is None:
    class _DummyModelFamily:
        R1 = "R1_placeholder_family" # Placeholder value
    ModelFamily = _DummyModelFamily
    logging.info("Using dummy ModelFamily as autogen_ext.ModelFamily was not found.")


# --- Agent-Specific LLM Configurations ---
AGENT_LLM_MAPPINGS = {
    "CodeAgent": {
        "type": "openai", "openai_model": "gpt-4-turbo-preview", "openai_timeout": 180,
    },
    "BacktesterAgent": {
        "type": "default", "ollama_model": "mistral:latest",
        "ollama_model_info": { "vision": False, "function_calling": True, "json_output": True, "family": ModelFamily.R1, "structured_output": True, },
        "openai_model": "gpt-3.5-turbo", "openai_timeout": 90,
    },
    "BO_Agent": {
        "type": "ollama", "ollama_model": "gemma3:1b",
        "ollama_model_info": { "vision": False, "function_calling": True, "json_output": False, "family": ModelFamily.R1, "structured_output": True, },
    },
    "TradingAgent": {
        "type": "default", "ollama_model": "tinyllama",
        "ollama_model_info": { "vision": False, "function_calling": True, "json_output": False, "family": ModelFamily.R1, "structured_output": True, },
        "openai_model": "gpt-4o-mini", "openai_timeout": 60,
    },
    "UserProxy": {
        "type": "default", "ollama_model": "mistral:latest",
        "ollama_model_info": {"function_calling": True, "json_output": True},
        "openai_model": "gpt-4o-mini", "openai_timeout": 120,
    },
    "UserProxy": {
        "type": "openai", # This has a higher chance of yielding an llm_config dict
        "openai_model": "gpt-4o-mini", # Or your preferred OpenAI model for UserProxy
        "openai_timeout": 120,
    },
    "Selector": {
        "type": "default", "ollama_model": "gemma3:1b",
        "ollama_model_info": {"function_calling": False}, # Selector usually doesn't need tools
        "openai_model": "gpt-3.5-turbo", "openai_timeout": 30,
    },
    "DEFAULT_CONFIG": {
        "type": "default", "ollama_model": "gemma3:1b",
        "ollama_model_info": { "vision": False, "function_calling": True, "json_output": False, "family": ModelFamily.R1, "structured_output": True, },
        "openai_model": "gpt-3.5-turbo", "openai_timeout": 60,
    }
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_llm_client(agent_name_or_key: str = None):
    config_key = agent_name_or_key if agent_name_or_key in AGENT_LLM_MAPPINGS else "DEFAULT_CONFIG"
    agent_config = AGENT_LLM_MAPPINGS[config_key]
    default_config = AGENT_LLM_MAPPINGS["DEFAULT_CONFIG"]

    client_type = agent_config.get("type", default_config["type"])
    use_local_llm = USE_OVERALL_LOCAL_LLM
    if client_type == "ollama": use_local_llm = True
    elif client_type == "openai": use_local_llm = False

    resolved_type_msg = 'Local Ollama' if use_local_llm else 'OpenAI'
    logging.info(f"Configuring LLM for '{agent_name_or_key or 'DEFAULT'}': Specified type='{client_type}', Resolved to '{resolved_type_msg}'")

    if use_local_llm:
        model_name = agent_config.get("ollama_model", default_config["ollama_model"])
        base_url = agent_config.get("base_url", OLLAMA_BASE_URL)
        
        model_info = default_config.get("ollama_model_info", {}).copy()
        model_info.update(agent_config.get("ollama_model_info", {}))
        if "function_calling" not in model_info and config_key != "Selector":
             model_info["function_calling"] = True
        # The `family` attribute might not be used by CustomOllamaClient unless you add logic for it.
        # For now, `model_info` is primarily for `function_calling` in CustomOllamaClient.

        logging.info(f"  Attempting Ollama: model='{model_name}', base_url='{base_url}'")
        try:
            requests.get(f"{base_url}/api/tags", timeout=3).raise_for_status()
            logging.info(f"  Ollama server at {base_url} is responsive.")
        except requests.exceptions.RequestException as e:
            logging.error(f"  Ollama server at {base_url} is NOT responding: {e}. Client creation may fail or lead to errors.")

        if AUTOGEN_EXT_OLLAMA_CLIENT_AVAILABLE:
            logging.info("  Using autogen_ext.models.ollama.OllamaChatCompletionClient.")
            try:
                return OllamaChatCompletionClient(model=model_name, base_url=base_url, model_info=model_info)
            except Exception as e:
                logging.error(f"  Error initializing autogen_ext Ollama client: {e}. Falling back to custom client.")
        
        logging.info("  Using CustomOllamaClient.")
        try:
            # Pass model_info to CustomOllamaClient for its own interpretation (e.g. function_calling)
            return CustomOllamaClient(model_name=model_name, base_url=base_url, model_info=model_info)
        except Exception as e:
            logging.error(f"  Error initializing CustomOllamaClient: {e}. Fallback to OpenAI if configured.")
            # Fallback to OpenAI only if primary was Ollama and it failed catastrophically
            if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY_HERE_ENV_NOT_SET":
                logging.warning("  Attempting fallback to default OpenAI client due to CustomOllamaClient error.")
                fb_openai_model = default_config.get("openai_model", "gpt-3.5-turbo")
                fb_openai_timeout = default_config.get("openai_timeout", 60)
                if AUTOGEN_EXT_OPENAI_CLIENT_AVAILABLE:
                    try:
                        return OpenAIChatCompletionClient(model=fb_openai_model, api_key=OPENAI_API_KEY, timeout=fb_openai_timeout)
                    except Exception as e_ext_openai:
                         logging.error(f"  Error initializing autogen_ext OpenAI fallback client: {str(e_ext_openai)}. No client returned.")
                         return None # Or a dummy client
                else: # Use standard AutoGen llm_config for OpenAI
                    logging.info(f"  Fallback OpenAI (standard AutoGen): model='{fb_openai_model}'")
                    return { # Return an llm_config dict for standard AutoGen agent
                        "config_list": [{"model": fb_openai_model, "api_key": OPENAI_API_KEY, "timeout": fb_openai_timeout}],
                        "cache_seed": None # Or a seed value
                    }
            else:
                logging.error("  OpenAI API key not configured for fallback. Cannot create OpenAI client.")
                return None # Or a dummy client

    else: # Configure OpenAI
        if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_HERE_ENV_NOT_SET":
            logging.error("OpenAI API key is not configured.")
            return None # Or a dummy client

        model_name = agent_config.get("openai_model", default_config["openai_model"])
        timeout = agent_config.get("openai_timeout", default_config.get("openai_timeout", 60))
        logging.info(f"  Attempting OpenAI: model='{model_name}', timeout='{timeout}'")

        if AUTOGEN_EXT_OPENAI_CLIENT_AVAILABLE:
            logging.info("  Using autogen_ext.models.openai.OpenAIChatCompletionClient.")
            max_retries = 3; base_retry_delay = 5
            for attempt in range(max_retries):
                try:
                    return OpenAIChatCompletionClient(model=model_name, api_key=OPENAI_API_KEY, timeout=timeout)
                except Exception as e:
                    logging.error(f"  Attempt {attempt+1} for autogen_ext OpenAI client failed: {e}")
                    if attempt < max_retries - 1: time.sleep(base_retry_delay * (attempt + 1))
                    else: logging.error("  Max retries reached for autogen_ext OpenAI client."); break
        
        # Fallback to standard AutoGen llm_config if autogen_ext OpenAI client failed or not available
        logging.info("  Using standard AutoGen llm_config for OpenAI.")
        return { # Return an llm_config dict
            "config_list": [{"model": model_name, "api_key": OPENAI_API_KEY, "timeout": int(timeout)}],
            "cache_seed": None # Or a seed value like 42
            # You can add temperature, etc. here if needed by standard AutoGen
        }
    return None # Should not be reached if logic is correct, but as a failsafe

# --- How to use this updated config.py ---
# from config import get_llm_client
# from autogen import AssistantAgent
#
# # Example for an agent that will get a model_client object
# code_agent_config_obj = get_llm_client("CodeAgent")
# if isinstance(code_agent_config_obj, dict): # It's an llm_config for standard AutoGen
#     code_agent = AssistantAgent(name="CodeAgent", llm_config=code_agent_config_obj, system_message="...")
# elif code_agent_config_obj: # It's a model_client object
#     code_agent = AssistantAgent(name="CodeAgent", model_client=code_agent_config_obj, system_message="...")
# else:
#     print("Failed to configure CodeAgent LLM.")
#
# # Example for BO_Agent (likely gets CustomOllamaClient or autogen_ext Ollama client)
# bo_agent_model_client = get_llm_client("BO_Agent")
# if bo_agent_model_client and not isinstance(bo_agent_model_client, dict):
#    bo_agent = BacktestingAndOptimizationAgent(name="BO_Agent", model_client=bo_agent_model_client)
# elif isinstance(bo_agent_model_client, dict):
#    print("BO_Agent received an llm_config dict, but expected a model_client object. Check config.")
# else:
#    print("Failed to configure BO_Agent LLM.")
