# main_autogen.py (Simplified Orchestrator-based version)

import logging
import sys
import asyncio
from pathlib import Path
import json

# --- Forceful Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:L%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logging.info("MAIN_AUTOGEN.PY: Root logging configured.")

# --- Project-Specific Imports ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the single instance of our new orchestrator
from agents.orchestrator_agent import OrchestratorAgent
#from agents.orchestrator_agent import orchestrator_agent
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from config import get_llm_client 

from agents.StrategyDesignerAgent import strategy_designer
#from agents.UIUserProxyAgent import ui_user


def run_strategy_loop():
    print("👋 Welcome to the Intraday Strategy Designer!")
    print("💡 Type your strategy request (e.g., 'Create a breakout strategy using Bollinger Bands')")

    user_input = input("📝 Your request: ")

    # Start conversation
    strategy_designer.reset()
    '''
    ui_user.reset()
    
    # Link agents
    ui_user.initiate_chat(
        recipient=strategy_designer,
        message=user_input
    )
    '''
async def main():
    """
    Main entry point. Defines the task and tells the orchestrator to run.
    """
    '''
    initial_task = (
        "Create a simple moving average (SMA) crossover strategy named 'SmaCrossover' for the 'NVDA' ticker. "
        "Use a 50-day fast moving average and a 200-day slow moving average. "
        "The backtest period should be from 2022-01-01 to 2023-12-31."
    )
    '''
    initial_task = "Create a Golden Cross strategy named 'GoldenCross' for the 'MSFT' ticker. This typically involves a 50-day moving average crossing above a 200-day moving average. The backtest period is from 2021-01-01 to 2023-12-31."
    # Initialize the OrchestratorAgent
    orchestrator_agent = OrchestratorAgent()
    # Kick off the workflow and get the final result
    final_result = await orchestrator_agent.run(initial_task)
    # Display the final, structured result
    logging.info(f"MAIN_AUTOGEN.PY: Workflow finished successfully.\n{json.dumps(final_result, indent=4)}")
    return final_result



    # --- Start the strategy loop (if needed) ---
    logging.info("--- 🚀 Starting Multi-Agent Trading System ---")
    # Logging agent client states BEFORE starting the stream
    logging.critical("--- PRE-STREAM AGENT CLIENT CHECK ---")
    chat_participants = []
    if not chat_participants:
        logging.error("No participants found in the chat.")
        return

    for p_agent_instance in chat_participants:
        agent_name = getattr(p_agent_instance, 'name', 'UnknownAgentName')
        model_client_instance = getattr(p_agent_instance, '_model_client', 'NO_MODEL_CLIENT_ATTR')
        
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

    # Get the LLM client for the selector mechanism
    selector_model_client = get_llm_client("Selector")
    if selector_model_client is None:
        logging.critical("Could not get a model client for the Selector. Halting.")
        return

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



    # Initialize the team using the new API's signature
    team = SelectorGroupChat(
        participants=chat_participants,
        model_client=selector_model_client,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,
        # max_turns=20 # Optional: You can set a turn limit
    )
    logging.info("SelectorGroupChat (Team) initialized.")

    
    task={"Train and backtest a simple moving average crossover strategy for NVDA."}
    try:
        objRun = team.run_stream(task=task)
        transformed_stream = wrapped_stream_generator(objRun)
        await Console(transformed_stream)  # Ensure Console is awaitable
    except Exception as e:
        logging.error(f"Error in main execution: {e}")    




async def wrapped_stream_generator(original_stream):
    async for item in original_stream:
        if isinstance(item, dict):
            # This is a simplified wrapping. You might need more sophisticated
            # logic to correctly populate all fields of ChatMessage or AgentEvent.
            # Especially 'source' needs to be correctly determined.
            # The 'role' from an LLM response dict is usually 'assistant'.
            # The 'source' for ChatMessage is the agent's name.
            # This assumes the dict might have a 'name' field for the speaker.
            logging.info(f"DEBUG WRAPPER: Wrapping dict: {item}") # Debug
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



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")

