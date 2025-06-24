from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import json
import re   


from config import get_llm_client


# Load LLM model
llm_client = get_llm_client("InstantCodeAgent")

# System prompt to constrain output
system_prompt = (
    "You are a Python expert in quantitative finance and the Backtrader framework.\n"
    "You will receive a strategy as a structured JSON input.\n\n"
    "Your task is to generate **ONLY** the valid Backtrader strategy class based on that strategy.\n\n"
    "🧾 Constraints:\n"
    "- Output must be **pure Python code only** — no markdown, no triple backticks.\n"
    "- Output must include: necessary imports + a single `bt.Strategy` class definition.\n"
    "- DO NOT include: `if __name__ == '__main__'`, `Cerebro`, data loading, or any test code.\n\n"
    "📐 Requirements:\n"
    "- Name the class using the `strategy_name` from JSON (e.g. `class VWAPMomentum(bt.Strategy):`)\n"
    "- Define all indicators listed in JSON in `__init__()`\n"
    "- Implement `entry`, `exit`, `stop_loss`, `take_profit` logic inside `next()`\n"
    "- If stop_loss or take_profit refer to % thresholds, calculate them using order.executed.price or a stored entry price\n"
    "- Ensure `self.order` is used to track open orders and prevent overlap\n"
    "- Use `self.position` and `self.buy()` / `self.sell()` for trades\n"
    "- If the strategy is 'momentum', avoid reversal logic like 'RSI < 30'; ensure RSI > 50 or similar\n"
    "- Keep logic clear and only use Backtrader conventions\n"
    "- Do not generate or mention markdown, backticks, or `streamlit`, `pandas`, or `cerebro` objects\n\n"
    "🎯 Example indicator references: `bt.ind.EMA(...)`, `bt.ind.RSI(...)`\n"
    "🎯 Example SL/TP references: `self.executed_price * (1 - stop_loss_percent)`\n"
)
# Create the CodeAgent
instant_code_agent = AssistantAgent(
    name="InstantCodeAgent",
    model_client=llm_client,
    system_message=system_prompt,
    description="Generates Backtrader strategy classes from strategy JSON input."
)

# Wrapper to feed JSON as prompt and return raw code
def generate_code_from_json(strategy_json: dict) -> str:
    strategy_input = f"```json\n{json.dumps(strategy_json, indent=2)}\n```"
    result = asyncio.run(instant_code_agent.run(task=strategy_input))
    return extract_raw_python(result.messages[-1].content)

def extract_raw_python(text: str) -> str:
    """Remove Markdown-style code fences like ```python and ```"""
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
