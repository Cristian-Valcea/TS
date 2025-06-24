import json
import asyncio
import re


from autogen import UserProxyAgent
from agents.StrategyDesignerAgent import strategy_designer
from agents.InstantCodeAgent import instant_code_agent
#from agents.BacktesterAgent import backtester_agent

# ui_user acts as the proxy for Streamlit user input
ui_user = UserProxyAgent(
    name="UIUser",
    human_input_mode="NEVER",
    code_execution_config=False,
)

def generate_strategy_from_prompt(prompt: str) -> str:
    """Send a prompt to the StrategyGeneratorAgent (AssistantAgent)"""
    return asyncio.run(strategy_designer.run(task=prompt)).messages[-1].content

def parse_strategy_output(agent_reply: str) -> tuple[str, dict | None]:
    """Split Markdown + JSON output into (markdown_text, json_dict)"""
    match = re.search(r"```json(.*?)```", agent_reply, re.DOTALL)
    json_part = None
    if match:
        try:
            json_part = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    markdown_part = agent_reply.split("```json")[0].strip()
    return markdown_part, json_part




def send_to_ui_user(prompt: str):
    """Send one-off message to StrategyDesignerAgent via ui_user"""
    response = ui_user.send_message(prompt, recipient=strategy_designer)
    return response.chat_history[-1]["content"]

def orchestrate_strategy_flow(prompt: str):
    """Starts a multi-agent chain: Strategy → (future Code → Backtest)"""
    ui_user.reset()
    strategy_designer.reset()
    response = ui_user.initiate_chat(recipient=strategy_designer, message=prompt)
    return response.chat_history[-1]["content"]

def generate_code_from_strategy_json(strategy_json: dict):
    """Pass structured JSON to CodeAgent for code generation"""
    json_block = f"```json\n{json.dumps(strategy_json, indent=2)}\n```"
    response = ui_user.send_message(json_block, recipient=instant_code_agent)
    return response.chat_history[-1]["content"]

def run_backtest_from_code(code: str):
    """Backtest using BacktesterAgent"""
#    code_block = f"```python\n{code}\n```"
#    response = ui_user.send_message(code_block, recipient=backtester_agent)
#    return response.chat_history[-1]["content"]
    return None


'''
strategy_raw = generate_strategy_from_prompt("Create a scalping strategy using EMA and RSI.")
markdown, strategy_json = parse_strategy_output(strategy_raw)
'''