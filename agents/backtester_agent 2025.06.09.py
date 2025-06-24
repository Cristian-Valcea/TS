# agents/backtester_agent.py
import sys
import os
import logging # Make sure logging is imported

from autogen.coding import LocalCommandLineCodeExecutor
from autogen_core.tools import FunctionTool
import tempfile
import shutil
import atexit

# Ensure project root is in path to import config
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from autogen_agentchat.agents import AssistantAgent
from config import get_llm_client # Your function to get the LLM client or config
from tools.backtesting_and_optimization.backtesting_tools import run_strategy_backtest # Import your backtesting tool function
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


logger = logging.getLogger(__name__) # Use a named logger for this module



# Create an executor instance
# You might want to configure a work_dir that gets cleaned up
temp_dir = tempfile.mkdtemp()
executor = LocalCommandLineCodeExecutor(work_dir=temp_dir) # timeout=60
logging.info(f"BacktesterAgent: Initialized LocalCommandLineCodeExecutor with work_dir: {temp_dir}")

# After creating temp_dir
def cleanup_temp_dir():
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up temp directory: {e}")

atexit.register(cleanup_temp_dir)


def execute_python_code_for_backtest(code_to_execute: str, file_name: str = "strategy_to_run.py") -> str:
    """
    Saves the given Python code to a file and executes it.
    Returns the combined stdout and stderr of the execution.
    The code should print its results (e.g., backtest metrics as JSON) to stdout.
    Args:
        code_to_execute (str): The Python code string to execute.
        file_name (str, optional): The name of the file to save the code to.
    """
    logger.info(f"Backtester Tool: Attempting to execute code saved to {file_name}...")
    try:
        # The executor saves the code to a file in its work_dir
        execution_result = executor.execute_code_blocks(
            code_blocks=[("python", code_to_execute)] # Expects list of (language, code)
        )
        # execution_result will be a CodeExecutionResult object
        # result_str = f"Exit Code: {execution_result.exit_code}\nOutput:\n{execution_result.output}"
        if execution_result.exit_code == 0:
            logger.info(f"Backtester Tool: Code executed successfully. Output: {execution_result.output[:500]}...")
            return execution_result.output # This should be the JSON string of metrics
        else:
            logger.error(f"Backtester Tool: Code execution failed. Exit code: {execution_result.exit_code}. Output: {execution_result.output}")
            return f"Code execution failed with exit code {execution_result.exit_code}. Error: {execution_result.output}"
    except Exception as e:
        logger.exception("Backtester Tool: Exception during code execution.")
        return f"Exception during code execution: {str(e)}"

# Create the FunctionTool
backtest_execution_tool = FunctionTool(
    func=execute_python_code_for_backtest,
    name="execute_python_backtest",
    description="Saves Python backtesting code (e.g., using Backtrader) to a file and runs it. The code MUST print its results as a JSON string to stdout."
)
tools_for_backtester = [backtest_execution_tool]
TOOL_AVAILABLE = True # Assuming executor setup is successful


'''
# Assuming TOOL_AVAILABLE logic for PythonCodeExecutionTool is separate for now
# For this specific backtesting tool, it's always available if imported.
tools_for_backtester = [run_strategy_backtest]
# If you also want it to execute arbitrary code:
# if TOOL_AVAILABLE (for PythonCodeExecutionTool):
#    tools_for_backtester.append(PythonCodeExecutionTool(executor=executor))
'''


logger.info("Attempting to configure BacktesterAgent...")
# Get the LLM configuration (either a client object or an llm_config dict)
# The "BacktesterAgent" key should be in your AGENT_LLM_MAPPINGS
backtester_agent_llm_config_or_client = get_llm_client("BacktesterAgent")

# Prepare common initialization arguments for AssistantAgent
init_kwargs_backtester = {
    "name": "BacktesterAgent",
    "description": "Backtests Python strategies using historical data and reports performance metrics like Sharpe Ratio and Max Drawdown. Can analyze provided strategy code and suggest improvements for backtesting.",
    "system_message": """You are a specialized Backtester Agent.
Your primary function is to run backtests on provided trading strategies using the 'run_strategy_backtest' tool.
To use the tool, you need:
1.  `strategy_file_path`: Path to the Python strategy file (e.g., "common_logic/strategies/my_strategy.py").
2.  `data_file_path`: Path to the CSV data file (e.g., "data/historical_data.csv").
3.  `strategy_class_name`: The name of the strategy class in the file (defaults to "CustomStrategy" if not provided).
4.  `strategy_params_json` (optional): A JSON string of strategy parameters (e.g., '{"period": 20}').

When asked to backtest a strategy:
- If the user or another agent provides the strategy file path, data path, and optionally parameters, call the 'run_strategy_backtest' tool.
- Present the key metrics from the JSON results clearly (e.g., PnL, Sharpe Ratio, Max Drawdown).
- If the tool returns an error, report the error.
- Do NOT write code for the strategy itself; assume it's provided in a file by the CodeAgent.
- Do NOT perform optimization; that's for the BO_Agent.""",
    "tools": tools_for_backtester
}

# Add memory if it's defined and you intend to use it
# if user_memory is not None:
#     init_kwargs_backtester["memory"] = [user_memory] # Memory is a list of Memory instances
# else:
#     logging.warning("BacktesterAgent: User memory not available or not configured.")


if backtester_agent_llm_config_or_client is None:
    logger.critical("CRITICAL (BacktesterAgent): get_llm_client returned None. Agent cannot be initialized with LLM capabilities.")
    # Option: Raise an error to stop execution if this agent is critical
    raise ValueError("BacktesterAgent LLM client configuration failed and is None.")
    # Or create a truly dummy, non-AssistantAgent placeholder if the script must continue
    # class NonFunctionalBacktester: name = "BacktesterAgent_ErrorState"
    # backtester_agent = NonFunctionalBacktester()
else:
    if isinstance(backtester_agent_llm_config_or_client, dict): # It's an llm_config
        init_kwargs_backtester["llm_config"] = backtester_agent_llm_config_or_client
        logger.info(f"BacktesterAgent configured with llm_config. Tool support depends on actual model and tools list.")
        # For llm_config (typically OpenAI), function calling is generally assumed if tools are provided.
        if TOOL_AVAILABLE:
            init_kwargs_backtester["tools"] = [PythonCodeExecutionTool(executor=executor)]
            logger.info("BacktesterAgent (llm_config): Tools enabled (PythonCodeExecutionTool).")
        else:
            logger.warning("BacktesterAgent (llm_config): PythonCodeExecutionTool or executor not available. Tools disabled.")

    else: # It's a model_client object
        init_kwargs_backtester["model_client"] = backtester_agent_llm_config_or_client
        client_model_info = getattr(backtester_agent_llm_config_or_client, 'model_info', {})
        client_name = getattr(backtester_agent_llm_config_or_client, 'name', 'UnknownClient')

        if client_model_info.get("function_calling", False):
            if TOOL_AVAILABLE:
                init_kwargs_backtester["tools"] = [PythonCodeExecutionTool(executor=executor)]
                logger.info(f"BacktesterAgent (model_client: {client_name}): Client reports function_calling=True. Tools enabled (PythonCodeExecutionTool). Model_info: {client_model_info}")
            else:
                init_kwargs_backtester["tools"] = [] # Explicitly empty if tool itself isn't ready
                logger.warning(f"BacktesterAgent (model_client: {client_name}): Client reports function_calling=True, but PythonCodeExecutionTool or executor not available. Tools disabled. Model_info: {client_model_info}")
        else:
            init_kwargs_backtester["tools"] = [] # Explicitly empty
            logger.warning(f"BacktesterAgent (model_client: {client_name}): Client reports function_calling=False. Tools disabled. Model_info: {client_model_info}")
    
    try:
        backtester_agent = AssistantAgent(**init_kwargs_backtester)
        logger.info(f"BacktesterAgent '{backtester_agent.name}' initialized successfully.")
        # Detailed post-init check (optional here if get_llm_client handled errors well)
        internal_model_client = getattr(backtester_agent, '_model_client', None)
        if internal_model_client:
            logger.info(f"BACKTESTER_AGENT (POST-INIT): _model_client type: {type(internal_model_client)}, model_info: {getattr(internal_model_client, 'model_info', 'N/A')}")
        elif init_kwargs_backtester.get("llm_config"):
             logger.info(f"BACKTESTER_AGENT (POST-INIT): Initialized with llm_config.")
        else:
            logger.error("BACKTESTER_AGENT (POST-INIT): _model_client is None AND no llm_config was used.")

    except ValueError as ve: # Catches "The model does not support function calling" from AssistantAgent.__init__
        logger.error(f"ValueError during BacktesterAgent initialization: {ve}")
        logger.error(f"Initialization kwargs used for BacktesterAgent: {init_kwargs_backtester}")
        raise
    except Exception as e_other:
        logger.error(f"Unexpected error during BacktesterAgent initialization: {e_other}")
        logger.error(f"Initialization kwargs used for BacktesterAgent: {init_kwargs_backtester}")
        raise
