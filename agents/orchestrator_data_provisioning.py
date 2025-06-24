# orchestrator_data_provisioning.py

import json
import asyncio
import logging

from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import TaskResult
# (Optionally) import a termination condition if you want to enforce a max‐turns:
# from autogen_agentchat.conditions import MaxMessageTermination

# your five agent instances (must be imported, each is e.g. an AssistantAgent)
from agents.data_provisioning_intraday import (
    universe_selector,
    time_window_planner,
    event_tagger,
    sampler_splitter,
    ibkr_fetcher_validator,
)


# 1) Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s:%(message)s",
    force=True

)
logger = logging.getLogger("data-provisioning")


# 2) Build the DiGraph via the fluent builder
builder = DiGraphBuilder()
builder \
    .add_node(universe_selector) \
    .add_node(time_window_planner) \
    .add_node(event_tagger) \
    .add_node(sampler_splitter) \
    .add_node(ibkr_fetcher_validator) \
    .add_edge(universe_selector, time_window_planner) \
    .add_edge(time_window_planner, event_tagger) \
    .add_edge(event_tagger, sampler_splitter) \
    .add_edge(sampler_splitter, ibkr_fetcher_validator)

graph = builder.build()                   # <-- this is a DiGraph
participants = builder.get_participants() # <-- this is a List[ChatAgent]

# 3) Create your GraphFlow team
#    By default it will stop when it reaches the leaf node (IBKRFetcherValidator)
team = GraphFlow(
    participants=participants,
    graph=graph,
    # Optional: enforce a hard stop after N turns
    # termination_condition=MaxMessageTermination(len(participants))
)

async def run_data_provisioning(initial_request: dict, parameters: dict) -> str:
    """Run through each agent, printing in-flight messages as we go."""
    payload = {"request": initial_request, "parameters": parameters}
    start = TextMessage(content=json.dumps(payload), source="user")

    # instead of .run(), consume run_stream()
    async for event in team.run_stream(task=[start]):
        # If it's a TaskResult, we're done
        if isinstance(event, TaskResult):
            # the last chat message is at messages[-1]
            final = event.messages[-1]
            logger.info("🏁 Final agent output:")
            logger.info(f"   [{final.source}] {final.content}")
            return final.content

        # Otherwise it's either a ChatMessage or an AgentEvent
        if isinstance(event, BaseChatMessage):
            logger.debug(f"💬 [{event.source}] {event.content}")
        else:
            # e.g. ModelClientStreamingChunkEvent, MemoryQueryEvent, etc.
            logger.debug(f"🔧 Event: {event}")

if __name__ == "__main__":
    initial_request = {
        "objectives": ["momentum", "mean_reversion", "reversal"],
        "candidates": ["AAPL", "MSFT", "NVDA"]
    }
    parameters = {
        "max_missing_pct": 0.05,
        "min_volume_per_bar": 100
    }

    final = asyncio.run(run_data_provisioning(initial_request, parameters))
    print("Final provisioning output:\n", final)
