# config.py
import os
import logging
import time
import json # Ensure json is imported
import uuid # For generating IDs if needed by ModelResult

import aiohttp # Ensure this is imported

# --- Import necessary types from autogen_core ---
from autogen_core.models import (
    ModelResult,
    MessageContent,      # Union[str, List[Dict[str, Any]]]
    AssistantMessage,    # Has content: MessageContent, role: str = "assistant", tool_calls: List | None = None
    ModelUsage,          # Has prompt_tokens: int, completion_tokens: int
    LLMMessage,          # Base for UserMessage, SystemMessage etc.
    FunctionCall         # For tool_calls if supported
)
from autogen_core.tools import BaseTool # For type hinting 'tools' argument
from autogen_core import CancellationToken # For type hinting 'cancellation_token'

# Attempt to import SystemMessage for specific handling
try:
    from autogen_core.models._types import SystemMessage as CoreSystemMessage # Alias to avoid clash
    AUTOGEN_CORE_SYSTEM_MESSAGE_AVAILABLE = True
except ImportError:
    CoreSystemMessage = None # Define it as None if not found
    AUTOGEN_CORE_SYSTEM_MESSAGE_AVAILABLE = False
    logging.debug("autogen_core.models._types.SystemMessage not found for specific handling in CustomOllamaClient.")


class CustomOllamaClient: # Implements autogen_core.models.ChatCompletionClient interface (implicitly)
    def __init__(self, model_name="tinyllama", base_url="http://localhost:11434", model_info=None, **kwargs):
        self.model_name = model_name
        self.base_url = base_url
        self.name = model_name # For AutoGen compatibility
        # model_info for ChatCompletionClient should indicate 'function_calling' and 'vision' capabilities
        default_client_model_info = {
            "name": model_name,
            "function_calling": True, # Assume by default, override in AGENT_LLM_MAPPINGS if needed
            "vision": False # Assume no vision by default
        }
        if model_info is not None and isinstance(model_info, dict):
            default_client_model_info.update(model_info)
        self.model_info = default_client_model_info # This is what AssistantAgent checks

        logging.info(f"CustomOllamaClient initialized: model={self.model_name}, base_url={self.base_url}, "
                     f"function_calling={self.model_info['function_calling']}, vision={self.model_info['vision']}")

    def __getitem__(self, key):
        if key == "name": return self.model_name
        if key == "model_info": return self.model_info # Crucial for AssistantAgent
        if key == "function_calling": return self.model_info.get("function_calling", False) # From model_info
        logging.warning(f"CustomOllamaClient.__getitem__ called with unhandled key: {key}")
        raise KeyError(f"Unsupported key requested in CustomOllamaClient: {key}")

    # This method is required by the Component interface if you want to save/load agent configs
    def dump_component(self) -> dict:
        # Return a serializable representation of how to reconstruct this client
        return {
            "type": f"{self.__class__.__module__}.{self.__class__.__name__}", # Gets "config.CustomOllamaClient"
            "params": {
                "model_name": self.model_name,
                "base_url": self.base_url,
                "model_info": self.model_info # Include model_info for reconstruction
            }
        }

    @classmethod
    def load_component(cls, config: dict) -> 'CustomOllamaClient':
        if config.get("type") != f"{cls.__module__}.{cls.__name__}":
            raise ValueError(f"Cannot load component of type {config.get('type')} into {cls.__name__}")
        params = config.get("params", {})
        return cls(**params)


    async def create(
        self,
        messages: List[LLMMessage], # Expects list of LLMMessage objects (UserMessage, SystemMessage etc.)
        tools: List[BaseTool] | None = None, # tools argument might be passed by AssistantAgent
        cancellation_token: CancellationToken | None = None, # cancellation_token might be passed
        **kwargs  # For additional like temperature, seed, options
    ) -> ModelResult: # CRITICAL: Return type is ModelResult
        """
        Generates a response from the Ollama model and returns an autogen_core.models.ModelResult.
        """
        processed_messages_for_ollama_api = []
        for msg_input in messages:
            role = "user"
            content_str = ""

            if hasattr(msg_input, 'role') and isinstance(msg_input.role, str):
                role = msg_input.role
            else:
                logging.warning(f"CustomOllamaClient: msg_input of type {type(msg_input)} missing string 'role' attribute. Defaulting to 'user'. Msg: {msg_input}")

            if hasattr(msg_input, 'content'):
                if isinstance(msg_input.content, str):
                    content_str = msg_input.content
                elif isinstance(msg_input.content, list): # For potential multimodal content
                    # Ollama's /api/chat expects a specific format for images within content.
                    # Example: {"type": "text", "text": "describe"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                    # This basic client will just join text parts for now.
                    # A full multimodal client would need to handle image data correctly.
                    text_parts = [str(p.get("text")) for p in msg_input.content if isinstance(p, dict) and p.get("type") == "text"]
                    content_str = "\n".join(text_parts)
                    if not text_parts and msg_input.content: # No text parts, stringify first element
                        content_str = str(msg_input.content[0]) if msg_input.content else ""
                else:
                    content_str = str(msg_input.content) # Fallback for other content types
            else:
                 logging.warning(f"CustomOllamaClient: msg_input of type {type(msg_input)} missing 'content' attribute. Using empty string. Msg: {msg_input}")


            # Ensure system role is correctly mapped if it's a CoreSystemMessage instance
            if AUTOGEN_CORE_SYSTEM_MESSAGE_AVAILABLE and isinstance(msg_input, CoreSystemMessage) and role != "system":
                role = "system"
            
            processed_messages_for_ollama_api.append({"role": role, "content": content_str})

        payload = {
            "model": self.model_name,
            "messages": processed_messages_for_ollama_api,
            "stream": False
        }
        
        # Handle Ollama-specific options from kwargs (e.g. temperature, seed)
        # These are usually passed under an 'options' key in the payload for Ollama's API
        ollama_options = kwargs.get("options", {}) # Get 'options' dict if passed in kwargs
        if 'temperature' in kwargs: ollama_options['temperature'] = kwargs['temperature']
        if 'seed' in kwargs: ollama_options['seed'] = kwargs['seed']
        # Add more specific kwargs to ollama_options mapping if needed

        if ollama_options: # Only add 'options' to payload if there are any
            payload['options'] = ollama_options

        # Tools for Ollama (if model supports it and format is known)
        # Ollama's tool support is evolving. For now, this client doesn't format tools for Ollama API.
        # If 'tools' arg is passed by AssistantAgent, we acknowledge it but don't send to basic Ollama /api/chat.
        # A more advanced Ollama client would handle tool formatting.
        if tools:
            logging.debug(f"CustomOllamaClient: Received tools argument, but basic client doesn't pass them to Ollama /api/chat. Tools: {[t.name for t in tools]}")


        api_url = f"{self.base_url}/api/chat"
        logging.info(f"CustomOllamaClient: Sending request to {api_url}. Model: {self.model_name}.")
        try:
            logging.info(f"CustomOllamaClient: Full payload for Ollama API: {json.dumps(payload, indent=2)}")
        except TypeError as te:
            logging.error(f"CustomOllamaClient: Could not JSON serialize payload for logging: {te}")
            logging.info(f"CustomOllamaClient: Raw payload: {payload}")

        final_content: MessageContent = "Ollama call failed: unspecified error." # Default error content
        final_tool_calls: List[FunctionCall] | None = None
        model_usage: ModelUsage | None = None
        response_text_content = "" # To store raw response text for debugging

        try:
            # Check for cancellation before making the call
            if cancellation_token and cancellation_token.is_cancelled:
                logging.info("CustomOllamaClient: Operation cancelled before making API call.")
                # ModelResult expects content, so provide a cancellation message
                return ModelResult(content="Operation cancelled.", usage=None, cost=None)


            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    response_status = response.status
                    response_text_content = await response.text()

                    logging.info(f"CustomOllamaClient: Ollama response status: {response_status}")
                    logging.info(f"CustomOllamaClient: Ollama response text snippet: {response_text_content[:500]}...")
                    
                    # Link cancellation token to the response handling
                    if cancellation_token:
                        cancellation_token.link_aiohttp_response(response)

                    response.raise_for_status() # Will raise ClientResponseError for 4xx/5xx
                    
                    response_json = json.loads(response_text_content)

            # --- Process successful response_json from Ollama /api/chat ---
            ollama_message_obj = response_json.get("message", {})
            final_content = ollama_message_obj.get("content", "") # This should be a string
            
            # Ollama's /api/chat might return tool_calls if model and version support it.
            # The format would need to be compatible with autogen_core.FunctionCall.
            # Example: [{"name": "func_name", "arguments": "{...}"}]
            raw_tool_calls = ollama_message_obj.get("tool_calls")
            if raw_tool_calls and isinstance(raw_tool_calls, list):
                final_tool_calls = []
                for tc_data in raw_tool_calls:
                    if isinstance(tc_data, dict) and "name" in tc_data and "arguments" in tc_data:
                        final_tool_calls.append(
                            FunctionCall(
                                id="toolcall_" + str(uuid.uuid4())[:8], # Ollama may not provide ID, generate one
                                name=str(tc_data["name"]),
                                arguments=json.dumps(tc_data["arguments"]) if isinstance(tc_data["arguments"], dict) else str(tc_data["arguments"])
                            )
                        )
                    else:
                        logging.warning(f"CustomOllamaClient: Received malformed tool_call data from Ollama: {tc_data}")
                if not final_tool_calls: final_tool_calls = None # Ensure it's None if list is empty after processing

            model_usage = ModelUsage(
                prompt_tokens=response_json.get("prompt_eval_count", 0),
                completion_tokens=response_json.get("eval_count", 0)
                # total_tokens can be derived or is sometimes provided
            )
            # Cost is not typically provided by Ollama, so leave as None for ModelResult

        except aiohttp.ClientResponseError as e:
            logging.error(f"CustomOllamaClient (async): HTTP {e.status} Error from {api_url}: {e.message}. "
                          f"Response body: {response_text_content[:500]}")
            final_content = f"Ollama API Error ({e.status}): {e.message}. Body: {response_text_content[:200]}"
        except aiohttp.ClientError as e: # Other client errors (connection, timeout before response, SSL issues etc.)
            logging.error(f"CustomOllamaClient (async): Client Connection/Timeout Error for {api_url}: {e}")
            final_content = f"Async HTTP Connection/Timeout Error for Ollama: {str(e)}"
        except json.JSONDecodeError as e:
            logging.error(f"CustomOllamaClient (async): Error decoding JSON response from Ollama: {e}. Response text: {response_text_content}")
            final_content = f"Error decoding JSON from Ollama response: {str(e)}. Response text: {response_text_content[:200]}"
        except asyncio.CancelledError:
            logging.info("CustomOllamaClient: create task was cancelled.")
            final_content = "Ollama call cancelled."
            # ModelResult must be returned
        except Exception as e:
            logging.exception("CustomOllamaClient (async): Unexpected critical error during 'create'.")
            final_content = f"Async Unexpected critical error with Ollama: {type(e).__name__} - {str(e)}"
        
        # Construct and return ModelResult
        # ModelResult expects 'content' to be MessageContent (str or List of dicts for multimodal)
        # If final_tool_calls is not None, AssistantAgent expects content to be None or empty string usually,
        # and tool_calls to be populated.
        # If no tool calls, content should be the string response.
        
        result_content_for_model_result: MessageContent
        if final_tool_calls:
            result_content_for_model_result = final_tool_calls # ModelResult.content can be List[FunctionCall]
        elif isinstance(final_content, str):
            result_content_for_model_result = final_content
        else: # Should not happen if final_content is always set to string
            result_content_for_model_result = str(final_content)


        model_result_obj = ModelResult(
            content=result_content_for_model_result, # This is now correctly MessageContent
            usage=model_usage,
            cost=None # Ollama doesn't provide cost
            # id, created, finish_reason, etc. can be added if available/needed
        )
        logging.info(f"CustomOllamaClient: Returning ModelResult. Content type: {type(model_result_obj.content)}, "
                     f"Content snippet: {str(model_result_obj.content)[:100] if not isinstance(model_result_obj.content, list) else 'ToolCalls'}")
        return model_result_obj

# ... (rest of config.py: AGENT_LLM_MAPPINGS, get_llm_client)
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
