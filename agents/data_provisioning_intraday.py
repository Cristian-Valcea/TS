from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools.ibkr_tools import fetch_5min_bars
from config import get_llm_client

# Wrap IBKR stub as a FunctionTool
ibkr_tool = FunctionTool(
    fetch_5min_bars,
    name="fetch_5min_bars",
    description="Fetch 5-min OHLCV bars from IBKR (stub)."
)

# Shared LLM client
llm_client = get_llm_client("DataProvisioningAgent")

# 1. UniverseSelector
universe_selector = AssistantAgent(
    name="UniverseSelector",
    model_client=llm_client,
    system_message="""
You are UniverseSelector.
Input: JSON { "objectives": ["momentum", "mean_reversion", ...], 
         "candidates": [optional list of symbols] }.
Output: JSON { "momentum": [...], "mean_reversion": [...], "reversal": [...] }.
Filter for avg volume>1e6, minimal halts, good extended-hours volume.
""",
    description="Selects high-liquidity tickers per objective."
)

# 2. TimeWindowPlanner
time_window_planner = AssistantAgent(
    name="TimeWindowPlanner",
    model_client=llm_client,
    system_message="""
You are TimeWindowPlanner.
Input: JSON { "universe": { ... }, "objectives": [...] }.
Output: JSON { "date_ranges": { "AAPL": {"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}, ... },
               "time_range": {"start":"04:00","end":"20:00","tz":"US/Eastern"} }.
Select earnings seasons, FOMC days, high-VIX windows accordingly.
""",
    description="Determines date & time ranges per symbol."
)

# 3. EventTagger
event_tagger = AssistantAgent(
    name="EventTagger",
    model_client=llm_client,
    system_message="""
You are EventTagger.
Input: JSON { "date_ranges": {...} }.
Output: JSON list of { "symbol":..., "date":..., "events":[...] }.
Annotate each date with earnings, macro events.
""",
    description="Annotates special events in each date range."
)

# 4. SamplerSplitter
sampler_splitter = AssistantAgent(
    name="SamplerSplitter",
    model_client=llm_client,
    system_message="""
You are SamplerSplitter.
Input: JSON { "tagged_ranges": [...], "parameters": {"max_missing_pct":..., "min_volume_per_bar":...} }.
Output: JSON { 
  "training_splits": [{"symbol":...,"start":...,"end":...},...],
  "test_splits":    [{"symbol":...,"start":...,"end":...},...]
}.
Include 3–5 warm-up days before each test window.
""",
    description="Creates non-overlapping train/test splits with warm-ups."
)

# 5. IBKRFetcherValidator
ibkr_fetcher_validator = AssistantAgent(
    name="IBKRFetcherValidator",
    model_client=llm_client,
    tools=[ibkr_tool],
    system_message="""
You are IBKRFetcherValidator.
Input: JSON { 
  "splits": { "training": [...], "test": [...] },
  "parameters": {"max_missing_pct":..., "min_volume_per_bar":...}
}.
For each split, loop <=30-day windows and call fetch_5min_bars().
Validate:
  - missing_pct ≤ max_missing_pct
  - volume ≥ min_volume_per_bar
Output: JSON { "validated": [
    {"symbol":...,"split":"training","path":"training_{symbol}.csv"},
    ...
  ]}.
Save each DataFrame.to_csv(path, index=False).
""",
    description="Fetches & validates bars, writes out final CSVs."
)

