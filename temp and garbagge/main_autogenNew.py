# main_autogen.py

# ... other imports ...
from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.utils import ChatResult

from agents.code_agent import CodeAgent
from agents.backtester_agent import BacktesterAgent
from agents.trading_agent import TradingAgent
from agents.selector_agent import Selector

from autogen_agentchat.messages import ChatMessage


# Import the tool
from autogen_agentchat.tools import PythonCodeExecutionTool

# ... logging and client setup ...

async def main():
    logging.info("--- Starting Multi-Agent Trading System ---")

    # ... other client initializations ...

    # --- Tool Initialization ---
    # The PythonCodeExecutionTool will manage a docker container or local process
    # to safely execute code.
    code_execution_tool = PythonCodeExecutionTool()
    logging.info(f"Initialized tool: {code_execution_tool.name}")
    
    # --- Agent Initialization ---
    logging.info("Initializing agents...")

    code_agent = CodeAgent(
        name="CodeAgent",
        llm_config=LLMConfig(llm_client=ollama_client)
    )

    # Pass the tool to the BacktesterAgent
    backtester_agent = BacktesterAgent(
        name="BacktesterAgent",
        llm_config=LLMConfig(llm_client=gemini_client, function_calling=True),
        tools=[code_execution_tool] # <-- PASS THE TOOL HERE
    )
    
    trading_agent = TradingAgent(
        name="TradingAgent",
        llm_config=LLMConfig(llm_client=ollama_client)
    )

    # UserProxyAgent remains LLM-less for now
    user_proxy_agent = UserProxyAgent(
        name="UserProxyAgent",
    )

    agent_list = [user_proxy_agent, code_agent, backtester_agent, trading_agent]
    agent_names = [agent.name for agent in agent_list]

    selector = Selector(
        name="Selector",
        llm_config=LLMConfig(llm_client=ollama_client),
        agent_names=agent_names
    )
    
    # Initial task
    task = """Create a Golden Cross strategy using SMA(50) and SMA(200) and test it on AAPL 2022–2023.
    Report metrics like Sharpe ratio and max drawdown."""

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



if __name__ == "__main__":
    asyncio.run(main())

