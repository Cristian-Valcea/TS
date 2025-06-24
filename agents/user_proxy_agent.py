# agents/user_proxy_agent.py
import sys
import os
import logging
from typing import Sequence # For type hinting if needed later

from autogen_agentchat.agents import UserProxyAgent as OfficialUserProxyAgent # Using the version from autogen-agentchat
from autogen_agentchat.messages import TextMessage, ChatMessage # For produced_message_types if we needed to patch

# Ensure project root is in path to import config
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config import get_llm_client, OPENAI_API_KEY, AGENT_LLM_MAPPINGS # Assuming config provides these

class CompatibleUserProxyAgent(OfficialUserProxyAgent):
    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)





# Configure a logger for this module if you want specific naming
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger
# If root logger is already configured well (e.g. in main_autogen.py or config.py), this line might not be needed
# or can be used to set a more specific level for this module's logs if desired.

logger.info("Attempting to configure UserProxyAgent (from autogen_agentchat.agents)...")

# --- Determine LLM configuration for UserProxy ---
# UserProxyAgent typically takes an llm_config dictionary, not a model_client object.
# get_llm_client for "UserProxy" is already designed to return an llm_config dict if type is "openai".
actual_llm_config_for_user_proxy = get_llm_client("UserProxy")

if actual_llm_config_for_user_proxy is None:
    logger.error("UserProxy LLM configuration failed (get_llm_client returned None). "
                 "UserProxy will be initialized without an LLM and can only relay human input.")
elif not isinstance(actual_llm_config_for_user_proxy, dict):
    logger.error(f"UserProxy received an unexpected LLM configuration type: {type(actual_llm_config_for_user_proxy)}. "
                 "Expected a dict (llm_config) or None. Initializing without LLM.")
    actual_llm_config_for_user_proxy = None # Ensure it's None if not a dict
else:
    logger.info(f"UserProxy will be configured with llm_config: {actual_llm_config_for_user_proxy}")
# --- End LLM configuration ---

try:
    # Instantiate autogen_agentchat.agents.UserProxyAgent
    # Provide only the arguments that this version's __init__ likely accepts.
    user_proxy = CompatibleUserProxyAgent(
    #user_proxy = UserProxyAgent(
        name="UserProxy",
        description="Acts as the primary interface for user interaction, initiates tasks, and provides human input when required by the system.",
        #llm_config=actual_llm_config_for_user_proxy, # Pass the llm_config dict or None
        # system_message: The UserProxyAgent from autogen-agentchat might take this if it's based on AssistantAgent or similar.
        # If it's a simpler UserProxy, it might not. Let's assume it might for now, but be ready to remove if TypeError.
        #system_message="""You are the User Proxy. You will provide tasks to other agents.
        #When the system needs human input, you will be prompted.
        #Reply TERMINATE when the overall task is done or if you want to end the conversation.
        #(This TERMINATE instruction is for the *human user* to type, the group chat termination condition will catch it)."""
        
        # Parameters like code_execution_config, human_input_mode, is_termination_msg, max_consecutive_auto_reply
        # are generally NOT part of the constructor for UserProxyAgent in autogen-agentchat 0.6.x.
        # They are handled by the group chat / team or other mechanisms.
    )
    logger.info(f"UserProxyAgent '{user_proxy.name}' (from autogen_agentchat.agents) initialized successfully.")

except TypeError as te:
    logger.error(f"FATAL TypeError during UserProxyAgent initialization: {te}")
    logger.error("Please check the constructor arguments for 'autogen_agentchat.agents.UserProxyAgent'. "
                 "Comment out or adjust arguments like 'system_message' or 'description' if they are causing the error.")
    raise # Re-raise the TypeError to halt execution and force a fix.
except Exception as e:
    logger.error(f"FATAL: Other error during UserProxyAgent initialization: {e}")
    raise

# The UserProxyAgent from autogen_agentchat.agents (version 0.6.1) SHOULD have the
# `produced_message_types` property because it inherits from BaseChatAgent, which inherits
# from Agent, which in autogen-core should have this defined or expect subclasses to define it.
# If it's missing, it indicates a deeper issue with the autogen-agentchat UserProxyAgent class itself
# or its base classes in your specific installation/version.
if not hasattr(user_proxy, 'produced_message_types'):
    logger.error(f"CRITICAL WARNING: UserProxyAgent instance '{user_proxy.name}' from 'autogen_agentchat.agents' "
                 "is missing the 'produced_message_types' property. This will likely cause errors with SelectorGroupChat.")
    # You could try monkey-patching it here as a last resort if the class itself is broken,
    # but it's better to understand why it's missing from the library's class.
    # user_proxy.produced_message_types = (TextMessage,)
    # logger.info("Attempted to monkey-patch 'produced_message_types' onto UserProxyAgent.")