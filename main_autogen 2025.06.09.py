import logging
import sys
logging.basicConfig(
    level=logging.INFO, # Use DEBUG for max verbosity
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:L%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)], # Explicitly to stdout
    force=True # (Python 3.8+) This will remove any existing handlers on the root logger
)
logging.info("MAIN_AUTOGEN.PY: Root logging configured forcefully.")

import os
from pathlib import Path
from typing import Sequence

# --- Autogen and Project-Specific Imports ---
# Add project root to the Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import asyncio

from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import ChatMessage
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage

#current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(current_dir)

from config import get_llm_client 

# Attempt to import SelectorGroupChat and other autogen components
try:
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.ui import Console
    logging.info("Using 'autogen_agentchat' custom library.")
except ImportError:
    logging.info("'autogen_agentchat' not found. Falling back to standard 'autogen' GroupChat if possible, "
          "but SelectorGroupChat specific features might not work as intended by original main_autogen.py. "
          "Consider installing/providing 'autogen_agentchat'.")
    # Fallback or error for standard autogen if SelectorGroupChat is essential and custom
    # For now, we'll assume it's available as per user's main_autogen.py
    # If it's not, the script will fail at the import.
    # If it's a wrapper around standard autogen, we might try to use standard autogen:
    from autogen import GroupChat, GroupChatManager, UserProxyAgent, AssistantAgent
    # This fallback path would require re-implementing the selector logic.
    # For now, let's stick to the user's specified import.
    # If SelectorGroupChat is not available, this script will need modification.

# Import your agents
from agents.code_agent import code_agent
from agents.reviewer_agent import reviewer_agent
from agents.backtester_agent import backtester_agent
from agents.trading_agent import trading_agent
from agents.user_proxy_agent import user_proxy
from agents.bo_agent import bo_agent
from agents.data_provisioning_agent import data_provisioning_agent
from agents.selector_agent import Selector



# Optionally set a lower level for specific noisy libraries if needed
# logging.getLogger("autogen_agentchat._selector_group_chat").setLevel(logging.INFO)


# =================================================================================
# OUR SOLUTION TO THE RATE LIMIT PROBLEM
# A custom class that adds a delay before each agent selection.
# =================================================================================
class DelayedSelectorGroupChat(SelectorGroupChat):
    def __init__(self, *args, delay_seconds: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_seconds = delay_seconds
        logging.info(f"Initialized DelayedSelectorGroupChat with a {self.delay_seconds}-second delay between turns.")

    async def select_speaker(self, thread: Sequence[BaseAgentEvent | BaseChatMessage]) -> list[str] | str:
        # This is the method called before each selection. We add our delay here.
        logging.info(f"--- Waiting for {self.delay_seconds} seconds to respect API rate limits...")
        await asyncio.sleep(self.delay_seconds)
        logging.info("--- Delay finished. Selecting next speaker.")
        return await super().select_speaker(thread)
# =================================================================================




async def main():
    """
    Main function to initialize and run the multi-agent system.
    """
    logging.info("--- 🚀 Starting Multi-Agent Trading System ---")

    # --- Agent Initialization ---
    # Agents are now imported directly, as they are initialized in their own modules.
    # We just need to initialize the agents that aren't singletons, like Selector and UserProxy.
    '''
    # UserProxyAgent remains LLM-less for now
    user_proxy_agent = UserProxyAgent(
        name="UserProxyAgent",
        # The user proxy doesn't need a system message as it's the entry point.
        human_input_mode="NEVER", # Set to "ALWAYS" to manually provide input at each step
        max_consecutive_auto_reply=1, # Allow it to reply once (e.g., with the initial prompt)
    )
    '''

    # The complete list of agents for the group chat
    agent_list = [
        user_proxy, 
        data_provisioning_agent, 
        code_agent, 
        reviewer_agent,
        backtester_agent, 
        trading_agent]
    agent_names = [agent.name for agent in agent_list]
    logging.info(f"Agents in the group chat: {agent_names}")

    # The Selector chooses the next speaker from the agent list.
    selector = Selector(
        name="Selector",
        llm_config=get_llm_client("Selector"),
        agent_names=agent_names
    )
    logging.info("Selector agent initialized.")

    # --- Group Chat and Manager Setup ---
    # --- Group Chat and Manager Setup ---
    # The SelectorGroupChat IS the selector. It needs the LLM config directly.
    # The 'selector_selection_message' is its system prompt for making selections.
    selector_prompt = f"""You are a master AI orchestrator. Your only task is to select the next agent to act.
The available agents are: {', '.join(agent_names)}.

Analyze the full conversation history and the last message to decide which agent is most appropriate to speak next to achieve the overall goal.

- If data is needed, choose **DataProvisioningAgent**.
- If a strategy script needs to be written, choose **CodeAgent**.
- If a script has been written and is ready for review, choose ReviewerAgent.
- If a script has failed a review and needs fixing, choose CodeAgent.
- If a script has passed review and is ready for testing, choose **BacktesterAgent**.
- If a backtest is complete and successful, choose **TradingAgent**.
- If the team is stuck or needs clarification, choose **UserProxy**.

Your entire response MUST be just the agent's name and nothing else.
For example, if CodeAgent should speak next, your response is:
CodeAgent
"""

    # Get the LLM client for the selector mechanism
    selector_model_client = get_llm_client("Selector")
    if selector_model_client is None:
        logging.critical("Could not get a model client for the Selector. Halting.")
        return

    # Initialize the team using the new API's signature
    team = SelectorGroupChat(
        participants=chat_participants,
        model_client=selector_model_client,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,
        # max_turns=20 # Optional: You can set a turn limit
    )
    logging.info("SelectorGroupChat (Team) initialized.")

    # --- Start the Conversation ---
    # This initial prompt is designed to trigger the full workflow.
    initial_prompt = (
        "Hello team. Here is the plan:\n"
        "1. First, I need the historical daily data for NVIDIA Inc. (NVDA) from 2022-01-01 to 2023-12-31.\n"
        "2. After the data is ready, create a simple moving average (SMA) crossover strategy. It should buy when the 50-day SMA crosses above the 200-day SMA, and sell when it crosses below.\n"
        "3. Once the strategy script is saved, backtest it using the NVDA data we just downloaded.\n"
        "4. Finally, report the key performance metrics from the backtest.\n\n"
        "Let's begin"
    )

    logging.info(f"\n--- Initiating chat with the following prompt ---\n{initial_prompt}\n-------------------------------------------------")

    # The Console UI component will handle displaying the streaming output.
    # The stream includes all agent messages, tool calls, etc.
    console_ui = Console()
    await console_ui.run(team.run_stream(task=initial_task))

    logging.info("--- 🏁 Chat finished. ---")


if __name__ == "__main__":
    main()

















# Optional termination conditions
termination = MaxMessageTermination(max_messages=30) | TextMentionTermination("TERMINATE")

# Selector prompt template
selector_prompt = """You are a highly intelligent AI orchestrator for a multi-agent trading system.
Your sole responsibility in this turn is to select the next agent to act from the provided list of agents.
Based on the conversation history, the last message, and the overall goal, decide which agent is most appropriate to speak next.

Available agents and their roles:
{roles}

Conversation History:
{history}


Full Conversation History (newest messages are typically at the end):
{history}

Based on the FULL conversation history, especially the LATEST messages, and the overall goal, select the MOST appropriate next agent.
Consider the following:
1. Is new code required? (CodeAgent)
2. Does existing code need backtesting or performance analysis? (BacktesterAgent)
3. Is a strategy ready for simulated or live deployment? (TradingAgent)
4. Is user input, clarification, or approval needed? (UserProxy)

Your response MUST be ONLY the exact name of ONE agent from the list: {participants}.
Do not include any other words, punctuation, explanations, or code.
For example, if CodeAgent should speak next, your entire response must be:
CodeAgent
Select the next agent to act:
"""


# Make sure your SelectorGroupChat initialization correctly 
# populates {last_speaker_name} and {last_speaker_message}
# if your version of SelectorGroupChat supports these specific placeholders.
# If not, the general {history} will have to suffice, 
# and the prompt instructions become even more critical.

chat_participants = [
     user_proxy,
     code_agent,
     data_provisioning_agent, 
     backtester_agent,
     bo_agent, # If you have this agent
     trading_agent
]

selector_model_client = get_llm_client("Selector")
# Create the team
team = SelectorGroupChat(
    chat_participants,
    #admin =user_proxy,  # Admin agent to oversee the chat
    model_client=selector_model_client,  # All agents share this model client
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=False
)

# Initial task
task = """Hello team. Here is the plan:\n"
        "1. First, get the historical daily data for NVIDIA Inc. (NVDA) from 2022-01-01 to 2023-12-31.\n"
        "2. After the data is ready, create a Python script for a simple moving average (SMA) crossover strategy. **The strategy class inside the script MUST be named 'SmaCrossover'**. It should buy when the 50-day SMA crosses above the 200-day SMA and sell when it crosses below.\n"
        "3. Once the strategy script is saved, the backtester will use the file path and the class name 'SmaCrossover' to run the test.\n"
        "4. Finally, report the key performance metrics from the backtest.\n\n"
        "Let's begin with step 1."""


async def wrapped_stream_generator(original_stream):
    async for item in original_stream:
        if isinstance(item, dict):
            # This is a simplified wrapping. You might need more sophisticated
            # logic to correctly populate all fields of ChatMessage or AgentEvent.
            # Especially 'source' needs to be correctly determined.
            # The 'role' from an LLM response dict is usually 'assistant'.
            # The 'source' for ChatMessage is the agent's name.
            # This assumes the dict might have a 'name' field for the speaker.
            print(f"DEBUG WRAPPER: Wrapping dict: {item}") # Debug
            source_name = item.get("name") or item.get("source") or "UnknownAgent" # Guess source

            # Reconstruct models_usage if it was flattened into the dict
            # usage_info = item.get("models_usage")
            # models_usage_obj = None
            # if isinstance(usage_info, dict):
            #     models_usage_obj = ModelUsage(
            #         prompt_tokens=usage_info.get("prompt_tokens", 0),
            #         completion_tokens=usage_info.get("completion_tokens", 0)
            #     )

            # Create a ChatMessage object.
            # You need to ensure all required fields for ChatMessage are present.
            # Refer to the ChatMessage class definition.
            yield ChatMessage(
                source=source_name,
                content=item.get("content", str(item)), # Fallback for content
                role=item.get("role", "assistant"), # Default to assistant if from LLM
                # models_usage=models_usage_obj, # If you reconstruct it
                # tool_calls=item.get("tool_calls") # If present
            )
        else:
            yield item





# Run the chat
async def main():
    # New approach with wrapper:
    logging.info("--- 🚀 Starting Multi-Agent Trading System ---")
    # Logging agent client states BEFORE starting the stream
    logging.critical("--- PRE-STREAM AGENT CLIENT CHECK ---")
    for p_agent_instance in chat_participants: # Use the list you used to init the team
        agent_name = getattr(p_agent_instance, 'name', 'UnknownAgentName')
        model_client_instance = getattr(p_agent_instance, '_model_client', 'NO_MODEL_CLIENT_ATTR') # _model_client is typical for AssistantAgent
        
        # For UserProxyAgent, it uses llm_config
        if hasattr(p_agent_instance, 'llm_config') and not hasattr(p_agent_instance, '_model_client'):
            llm_cfg = getattr(p_agent_instance, 'llm_config', 'NO_LLM_CONFIG_ATTR')
            logging.critical(f"AGENT_CHECK ({agent_name}): Has llm_config: {llm_cfg is not None}. Config details: {llm_cfg}")
            if llm_cfg is None:
                 logging.error(f"AGENT_CHECK ({agent_name}): LLM_CONFIG IS NONE!")

        elif model_client_instance == 'NO_MODEL_CLIENT_ATTR':
            logging.warning(f"AGENT_CHECK ({agent_name}): Does not have a _model_client attribute.")
        elif model_client_instance is None:
            logging.error(f"AGENT_CHECK ({agent_name}): _model_client IS NONE!")
        else:
            client_name = getattr(model_client_instance, 'name', type(model_client_instance).__name__)
            client_model_info = getattr(model_client_instance, 'model_info', 'NO_MODEL_INFO_ATTR')
            logging.critical(f"AGENT_CHECK ({agent_name}): _model_client type: {type(model_client_instance).__name__}, "
                             f"client name/model: {client_name}, client_model_info: {client_model_info}")
    logging.critical("--- END PRE-STREAM AGENT CLIENT CHECK ---")



    objRun = team.run_stream(task=task)
    transformed_stream = wrapped_stream_generator(objRun)
    '''
    
    # Find BacktesterAgent instance (example, might need adjustment)
    bt_agent_instance = None
    for p_agent in team.participants: # Assuming team.participants holds the agent instances
        if p_agent.name == "BacktesterAgent":
            bt_agent_instance = p_agent
            break
    
    if bt_agent_instance:
        logging.critical(f"MAIN: BacktesterAgent instance found. Its _model_client is: {getattr(bt_agent_instance, '_model_client', 'NOT_FOUND')}")
    else:
        logging.error("MAIN: Could not find BacktesterAgent instance in team.participants")
    '''
    await Console(transformed_stream) # This line will error out
    #    await Console(team.run_stream(task=task))

# Standard entry point for an async script
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")