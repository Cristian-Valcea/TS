import json
import os
import asyncio
from agents.StrategyDesignerAgent import strategy_designer
from agents.InstantCodeAgent import generate_code_from_json
from agents.InstantCodeAgent import instant_code_agent
from agents.backtester_agent import backtester_agent
import re

def generate_strategy_from_prompt(prompt: str):
    result = asyncio.run(strategy_designer.run(task=prompt))
    return result.messages[-1].content if result and result.messages else "❌ No response"

def generate_code_from_strategy_json(strategy_json: dict):
    json_block = f"```json\n{json.dumps(strategy_json, indent=2)}\n```"
    result = asyncio.run(instant_code_agent.run(task=json_block))
    return result.messages[-1].content if result and result.messages else "❌ No response"

def parse_strategy_output(output: str):
    """
    Extracts the Markdown description and JSON structure from the agent's reply.
    Returns a tuple of (markdown_text, parsed_json) where parsed_json may be None on failure.
    """
    # Find JSON block
    match = re.search(r"```json(.*?)```", output, re.DOTALL)
    # Strip out the JSON block to leave only markdown
    markdown = re.sub(r"```json.*?```", "", output, flags=re.DOTALL).strip()
    json_part = None
    if match:
        try:
            json_part = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            json_part = None
    return markdown, json_part


def extract_class_name_from_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        code = f.read()
    match = re.search(r"class\s+(\w+)\s*\(.*?\):", code)
    if match:
        return match.group(1)
    raise ValueError("No class definition found in strategy file.")

def run_backtest_from_file(path: str) -> str:
    strategy_class = extract_class_name_from_file(path)
    task = f'''
    The strategy code is saved at: "{path}"
    Please run a backtest using your `run_strategy_backtest` tool with the following:
    - strategy_file_path: "{path}"
    - data_file_path: "data.csv"
    - strategy_class_name: "{strategy_class}"
    '''
    result = asyncio.run(backtester_agent.run(task=task))
    return result.messages[-1].content if result and result.messages else "❌ No result returned from backtester."

def save_code_to_class_named_file(code: str, directory="generated") -> str:
    os.makedirs(directory, exist_ok=True)
    match = re.search(r"class\s+(\w+)\s*\(.*?\):", code)
    if not match:
        raise ValueError("No class name found in strategy code.")
    class_name = match.group(1)
    file_path = os.path.join(directory, f"{class_name}.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return file_path

def run_backtest_from_code(code: str) -> str:
    task = (
        "Please run a backtest using your `run_strategy_backtest` tool using the given Python code. "
        "Only output performance metrics. Use matplotlib to save a plot if the strategy triggers any trades.\n\n"
        f"```python\n{code}\n```"
    )
    result = asyncio.run(backtester_agent.run(task=task))
    return result.messages[-1].content if result and result.messages else "❌ No result returned from backtester."

