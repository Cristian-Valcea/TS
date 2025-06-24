from autogen_agentchat.agents import AssistantAgent
from config import get_llm_client
import asyncio
import json, re

llm = get_llm_client("StrategyRaterAgent")
system = """
You are an expert quantitative strategist. 
Given a trading strategy defined as JSON, 
assign integer scores 1–5 for these dimensions:
- risk_reward: goodness of the stop-loss/take-profit balance
- clarity: how clearly the rules translate into code
- novelty: originality vs. standard patterns
- overall: your overall recommendation

Return ONLY a JSON object, for example:
```json
{"risk_reward":4,"clarity":5,"novelty":3,"overall":4}
"""

strategy_rater = AssistantAgent(
name="StrategyRaterAgent",
model_client=llm,
system_message=system,
description="Rates trading strategies across multiple dimensions."
)

def auto_rate_strategy(strategy_json: dict) -> dict:
    prompt = (
        "Here is a trading strategy to rate:\n\n"
        f"```json\n{json.dumps(strategy_json, indent=2)}\n```"
        "\n\nPlease return **only** a JSON object with scores for "
        "`risk_reward`, `clarity`, `novelty` (0–10 each) and an overall `score`."
    )
    # drive the coroutine to completion
    result = asyncio.run(strategy_rater.run(task=prompt))
    # 3) Extract the last assistant message
    assistant_msg = None
    for msg in result.messages[::-1]:
        if msg.source == "StrategyRaterAgent":
            assistant_msg = msg.content
            break
    if assistant_msg is None:
        return {}  # no reply at all
    # 4) Pull out the JSON block
    #    e.g. ```json { ... } ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", assistant_msg, re.DOTALL)
    if not m:
        # didn't find a fenced JSON, try raw braces
        m = re.search(r"(\{.*\})", assistant_msg, re.DOTALL)

    if not m:
        # give up
        return {}

    raw = m.group(1).strip()

    # 5) Parse, with fallback on error
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}

