import asyncio
from autogen_agentchat.agents import AssistantAgent
from config import get_llm_client


llm_client = get_llm_client("StrategyDesignerAgent") # You will need to map this in your config
# System message: instructs the model how to behave
system_prompt = (
    "You are a senior quantitative analyst tasked with designing intraday trading strategies.\n"
    "You must reply with:\n\n"
    "1. A human-readable strategy description in **Markdown**\n"
    "2. A structured `JSON` object with:\n"
    "- strategy_name\n"
    "- style (e.g., 'momentum', 'reversal')\n"
    "- indicators (list)\n"
    "- entry (text)\n"
    "- exit (text)\n"
    "- stop_loss (text)\n"
    "- take_profit (text)\n"
    "- timeframe (e.g., '5m', '15m')\n\n"
    "Always wrap the JSON in triple backticks like:\n"
    "```json\n{...}\n```\n"
)

# Define the AssistantAgent
strategy_designer = AssistantAgent(
    name="StrategyGeneratorAgent",
    model_client=llm_client,
    system_message=system_prompt,
    description="Generates structured intraday trading strategies with Markdown and JSON output.",
)

def design_strategy(user_prompt: str) -> str:
    """
    Generate a trading strategy response using the AssistantAgent
    Args:
        user_prompt: The user's textual prompt describing desired strategy.
    Returns:
        The assistant's raw response content (Markdown + JSON).
    """
    # Run the agent synchronously
    result = asyncio.run(strategy_designer.run(task=user_prompt))
    # Extract the final assistant message as content
    return result.messages[-1].content

def refine_strategy(feedback: str) -> str:
    """
    Refine an existing strategy based on user feedback.
    Args:
        feedback: The user's refinement instructions.
    Returns:
        The assistant's refined strategy response.
    """
    result = asyncio.run(strategy_designer.run(task=feedback))
    return result.messages[-1].content


