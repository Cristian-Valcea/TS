# agents/selector_agent.py

from autogen_core import Agent
from autogen.llm_config import LLMConfig

class Selector(Agent):
    def __init__(self, name: str, llm_config: LLMConfig, agent_names: list[str]):
        
        # --- NEW, REFINED SYSTEM MESSAGE ---
        system_message = f"""You are an AI orchestrator. Your ONLY task is to select the next agent.
The available agents are: {', '.join(agent_names)}.

Based ONLY on the last message in the conversation, output the EXACT NAME of the single most appropriate agent from the list to speak next.
Your entire response MUST be just the agent's name and nothing else.

For example, if the next agent should be CodeAgent, your response is:
CodeAgent

DO NOT add any explanations, punctuation, or other text.
Select the next agent to act:
**VALID RESPONSES:**
{', '.join(agent_names)}
"""
        # --- END NEW SYSTEM MESSAGE ---

        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            # Description is optional but can be helpful for other agents
            description=f"A selector agent that chooses the next agent from [{', '.join(agent_names)}]."
        )
