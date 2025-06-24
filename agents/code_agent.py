# agents/code_agent.py (Snippet Generator Version)

import logging
import sys
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config import get_llm_client

logger = logging.getLogger(__name__)

# --- AGENT SYSTEM MESSAGE (RADICALLY SIMPLIFIED) ---
code_agent_system_message = """You are an expert Python code generator for the Backtrader library.
Your ONLY task is to generate a Python script file containing just the strategy class and its necessary imports.
Your response MUST be ONLY the raw Python code. Do not include any `if __name__ == '__main__':` block, Cerebro setup, or data loading code.
"""

# --- AGENT INITIALIZATION (NO TOOLS) ---
logger.info("Attempting to configure CodeAgent (Snippet Generator)...")
init_kwargs = {
    "name": "CodeAgent",
    "description": "Generates a Python strategy class snippet based on a template and request.",
    "system_message": code_agent_system_message,
}

llm_client = get_llm_client("CodeAgent")
if llm_client is None:
    raise ValueError("CodeAgent LLM config is None.")
else:
    init_kwargs["model_client"] = llm_client

try:
    code_agent = AssistantAgent(**init_kwargs)
    logger.info(f"✅ CodeAgent '{code_agent.name}' (Snippet Generator) initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing CodeAgent: {e}", exc_info=True)
    raise