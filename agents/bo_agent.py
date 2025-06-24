# agents/bo_agent.py
import logging
from autogen_agentchat.agents import AssistantAgent
from config import get_llm_client
from tools.backtesting_and_optimization.optuna_optimizer_tool import OptunaOptimizerTool

logger = logging.getLogger(__name__)
logger.info("Attempting to configure BO_Agent...")

# --- Tool and Memory Placeholders (Replace with your actual implementations) ---
# You'll need to define/import your PythonCodeExecutionTool and its executor,
# and your user_memory instance.
try:
    # Attempt to import your actual tools and memory
    from autogen_ext.tools._code_execution import PythonCodeExecutionTool # Example import path
    # from my_project_tools import get_code_executor # Example for an executor
    # from my_project_memory import user_memory_instance as user_memory # Example for memory
    
    # executor = get_code_executor() # Instantiate your executor
    # For now, using placeholders if actual ones aren't ready:
    class PythonCodeExecutionTool:
        def __init__(self, executor=None, name="python_code_executor"): # Add name for tool
            self.name = name
            self.description = "Executes Python code in a secure environment and returns the output."
            self.executor = executor
            logging.info(f"Placeholder PythonCodeExecutionTool initialized with executor: {executor}")

        # Dummy schema for placeholder tool
        @property
        def schema(self) -> dict:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The Python code to execute."}
                    },
                    "required": ["code"],
                },
            }
        
        async def run(self, code: str, **kwargs): # Placeholder run method
            logging.info(f"Placeholder PythonCodeExecutionTool: Would execute code: {code[:100]}...")
            return "Placeholder: Code execution result."

    executor = None # Define your actual executor instance here
    user_memory = None  # Define your actual user_memory instance here (if used by BacktesterAgent)
    TOOL_AVAILABLE = True if PythonCodeExecutionTool and executor is not None else False
    logging.info(f"PythonCodeExecutionTool available: {TOOL_AVAILABLE}")

except ImportError as e:
    logging.error(f"Could not import actual tools/memory for BacktesterAgent: {e}. Using placeholders.")
    class PythonCodeExecutionTool:
        def __init__(self, executor=None, name="python_code_executor_dummy"):
             self.name = name
             self.description = "Dummy Python Code Execution Tool."
             logging.info("Dummy PythonCodeExecutionTool initialized.")
        @property
        def schema(self) -> dict: return {"name": self.name, "description": self.description, "parameters": {}} # Minimal schema
        async def run(self, **kwargs): return "Dummy tool execution."

    executor = None
    user_memory = None
    TOOL_AVAILABLE = False
# --- End Tool and Memory Placeholders ---






bo_agent_llm_config_or_client = get_llm_client("BO_Agent") # BO_Agent needs an LLM config
# Instantiate the tool
optimizer_tool_instance = OptunaOptimizerTool()

init_kwargs_bo = {
    "name": "BO_Agent",
    "description": "Optimizes trading strategy parameters using Optuna.",
    "system_message": """You are a Backtesting and Optimization (B&O) Agent.
Your primary role is to optimize trading strategies using the 'optimize_strategy' tool.
To use the tool, you need:
1. `strategy_file_path`: Path to the Python strategy file.
2. `data_feed_path`: Path to the CSV data file.
3. `optuna_params_config`: A Python dictionary (which you will convert to JSON string for the tool if needed, but the tool takes a dict) specifying parameters to optimize, their types, and ranges. Example: {"sma_period": {"type": "int", "low": 10, "high": 50}, ...}
4. `n_trials` (optional, int): Number of Optuna trials.
5. `target_metric` (optional, str): Metric to optimize (e.g., "sharpe_ratio").
6. `direction` (optional, str): "maximize" or "minimize".

When asked to optimize:
- Gather all required parameters.
- Call the `optimize_strategy` tool.
- Report the best parameters and best metric value found.
- Indicate where results are saved.""",
"tools": [optimizer_tool_instance.optimize_strategy], # Register the method as a tool
}

bo_agent_llm_config_or_client = get_llm_client("BO_Agent") # This gets an OllamaChatCompletionClient
if bo_agent_llm_config_or_client is None:
    logger.critical("CRITICAL (BO_Agent): get_llm_client returned None. Agent cannot be initialized with LLM capabilities.")
    # Option: Raise an error to stop execution if this agent is critical
    raise ValueError("BO_Agent LLM client configuration failed and is None.")
    # Or create a truly dummy, non-AssistantAgent placeholder if the script must continue
    # class NonFunctionalBacktester: name = "BacktesterAgent_ErrorState"
    # backtester_agent = NonFunctionalBacktester()
else:
    if isinstance(bo_agent_llm_config_or_client, dict): # It's an llm_config
        init_kwargs_bo["llm_config"] = bo_agent_llm_config_or_client
        logger.info(f"BO_Agent configured with llm_config. Tool support depends on actual model and tools list.")
        # For llm_config (typically OpenAI), function calling is generally assumed if tools are provided.
        if TOOL_AVAILABLE:
            init_kwargs_bo["tools"] = [PythonCodeExecutionTool(executor=executor)]
            logger.info("BO_Agent (llm_config): Tools enabled (PythonCodeExecutionTool).")
        else:
            logger.warning("BO_Agent (llm_config): PythonCodeExecutionTool or executor not available. Tools disabled.")

    else: # It's a model_client object
        init_kwargs_bo["model_client"] = bo_agent_llm_config_or_client
        client_model_info = getattr(bo_agent_llm_config_or_client, 'model_info', {})
        client_name = getattr(bo_agent_llm_config_or_client, 'name', 'UnknownClient')

        if client_model_info.get("function_calling", False):
            if TOOL_AVAILABLE:
                init_kwargs_bo["tools"] = [PythonCodeExecutionTool(executor=executor)]
                logger.info(f"BO_Agent (model_client: {client_name}): Client reports function_calling=True. Tools enabled (PythonCodeExecutionTool). Model_info: {client_model_info}")
            else:
                init_kwargs_bo["tools"] = [] # Explicitly empty if tool itself isn't ready
                logger.warning(f"BO_Agent (model_client: {client_name}): Client reports function_calling=True, but PythonCodeExecutionTool or executor not available. Tools disabled. Model_info: {client_model_info}")
        else:
            init_kwargs_bo["tools"] = [] # Explicitly empty
            logger.warning(f"BO_Agent (model_client: {client_name}): Client reports function_calling=False. Tools disabled. Model_info: {client_model_info}")

bo_agent = AssistantAgent(**init_kwargs_bo)
logger.info(f"BO_Agent '{bo_agent.name}' initialized successfully.")
