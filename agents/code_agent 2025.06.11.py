# agents/code_agent.py (Final, Simplified "Snippet Generator" Version)

import logging
import sys
from pathlib import Path

# NOTE: We no longer need FunctionTool here
from autogen_agentchat.agents import AssistantAgent

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import get_llm_client

logger = logging.getLogger(__name__)

# --- AGENT SYSTEM MESSAGE (RADICALLY SIMPLIFIED) ---
code_agent_system_message = """You are an expert Python code generation agent.
You will be given a user's request and a code template.
Your ONLY task is to generate the Python code for the strategy class to fill the placeholder in the template.
Your response MUST be ONLY the Python code for the class definition.
Do NOT include the template, any explanations, or any markdown formatting like ```python. Just the raw class code.
"""

# --- AGENT INITIALIZATION (NO TOOLS) ---
logger.info("Attempting to configure CodeAgent (Snippet Generator)...")
llm_config_or_client = get_llm_client("CodeAgent")

# Note: The 'tools' key is removed.
init_kwargs = {
    "name": "CodeAgent",
    "description": "Generates a Python strategy class snippet based on a template and request.",
    "system_message": code_agent_system_message,
}

if llm_config_or_client is None:
    raise ValueError("CodeAgent LLM config is None.")
else:
    if isinstance(llm_config_or_client, dict):
        init_kwargs["llm_config"] = llm_config_or_client
    else:
        init_kwargs["model_client"] = llm_config_or_client

try:
    # We can still use AssistantAgent; it just won't use tools.
    code_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ CodeAgent '{code_agent.name}' (Snippet Generator) initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing CodeAgent: {e}", exc_info=True)
    raise