import sys
import os
import logging
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from config import get_llm_client 

# Setup basic logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

trading_agent_client = get_llm_client("TradingAgent")
trading_agent = AssistantAgent(
    name="TradingAgent",
    description="Deploys validated strategy to live or paper trading using IBKR API.",
    model_client=trading_agent_client,
    system_message="""
You are a live trading agent.
You execute Python-based strategies using IBKR API (via ib_insync).
Your tasks:
1. Load the strategy object or logic from file.
2. Subscribe to live data.
3. Execute simulated or real trades using IBKR.
4. Report every order, position, and PnL update in structured format.
5. Handle restart/resume requests or reconfigurations.

End output with "TRADING DEPLOYED" or "SIMULATION COMPLETE"."""
)

