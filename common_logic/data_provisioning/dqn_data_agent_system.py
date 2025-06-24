# common_logic/data_provisioning/dqn_data_agent_system.py

import yfinance as yf
import os
import sys
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pytz
import asyncio

# --- Autogen imports ---
import autogen
from autogen_agentchat.agents import AssistantAgent

#from autogen import Agent, UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager

# --- Robust Pathing Setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --- End Pathing Setup ---

from config import get_llm_client


logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "data_provisioning.log"))
    ],
    force=True

)

# --- Tools ---
def fetch_5min_bars(symbol: str, start: str, end: str, use_rth: bool = False) -> pd.DataFrame:
    """
    Call IBKR API: reqHistoricalData with barSizeSetting='5 mins', useRTH=use_rth, whatToShow='TRADES'.
    Return a timezone-corrected pandas.DataFrame with columns [Date, Open, High, Low, Close, Volume].
    """
    # TODO: implement via IBKR Python API
    # This is a stub that generates random data for testing
    logger.info(f"Fetching 5-min bars for {symbol} from {start} to {end} (useRTH={use_rth})")
    
    # Create date range for 5-minute bars
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    # Generate dates at 5-minute intervals
    dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # Filter for market hours if use_rth is True
    if use_rth:
        # Keep only timestamps between 9:30 AM and 4:00 PM Eastern time
        market_hours = [(d.hour > 9 or (d.hour == 9 and d.minute >= 30)) and d.hour < 16 for d in dates]
        dates = dates[market_hours]
    
    # Generate random data
    import numpy as np
    n = len(dates)
    
    # Base price around 100 with some randomness per symbol
    base_price = 100 + hash(symbol) % 400
    
    # Create OHLCV data with some randomness
    data = {
        'Date': dates,
        'Open': np.random.normal(base_price, base_price * 0.01, n),
        'High': np.zeros(n),
        'Low': np.zeros(n),
        'Close': np.zeros(n),
        'Volume': np.random.randint(100, 10000, n)
    }
    
    # Adjust High, Low, Close based on Open
    for i in range(n):
        daily_volatility = data['Open'][i] * 0.005  # 0.5% volatility
        data['High'][i] = data['Open'][i] + abs(np.random.normal(0, daily_volatility))
        data['Low'][i] = data['Open'][i] - abs(np.random.normal(0, daily_volatility))
        data['Close'][i] = np.random.normal(data['Open'][i], daily_volatility)
        
        # Ensure High >= Open >= Low and High >= Close >= Low
        data['High'][i] = max(data['High'][i], data['Open'][i], data['Close'][i])
        data['Low'][i] = min(data['Low'][i], data['Open'][i], data['Close'][i])
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    return df


def select_with_real_data(self, candidates: List[str], objectives: List[str]) -> List[str]:
    """Select stocks using real market data and objective criteria."""
    
    # Fetch real market data
    market_data = self.fetch_market_data(candidates)
    
    # Filter for liquidity
    high_liquidity = [ticker for ticker in candidates 
                     if ticker in market_data and 
                     market_data[ticker].get("avg_volume", 0) > 1_000_000]
    
    # Filter for minimal halts
    low_halts = [ticker for ticker in high_liquidity
                if market_data[ticker].get("halt_frequency", float('inf')) < 0.01]
    
    # Ensure sector diversification
    sectors = {}
    for ticker in low_halts:
        sector = market_data[ticker].get("sector", "Unknown")
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(ticker)
    
    # Select top stocks from each sector
    selected_stocks = []
    for sector_stocks in sectors.values():
        # Take top 1-2 stocks from each sector
        selected_stocks.extend(sector_stocks[:min(2, len(sector_stocks))])
    
    return selected_stocks


# --- Agent Configuration ---
class UniverseSelector(AssistantAgent):
    def __init__(self, name="UniverseSelector"):
        super().__init__(
            name=name,
            system_message="""You are the UniverseSelector agent. Your task is to choose a set of high-liquidity tickers 
            for each objective (momentum, mean-reversion, reversal). Filter the user-provided universe or default list 
            by >1M avg volume, minimal halts, and split by objective category.""",
            model_client=get_llm_client(name),  # Assumes a mapping exists in your config
            tools=[]  # Add appropriate tools if needed
        )
    
    async def select_universe(self, objectives: List[str], user_universe: Optional[List[str]] = None) -> List[str]:
        """
        Select universe of tickers based on objectives using LLM analysis.
        
        Args:
            objectives: List of objectives (momentum, mean-reversion, reversal)
            user_universe: Optional user-provided list of tickers
            
        Returns:
            List of selected tickers
        """
        logger.info(f"Selecting universe for objectives: {objectives}")
        
        # Default universe if none provided
        default_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ", 
                            "IBM", "GE", "BA", "DIS", "PFE", "KO", "WMT", "XOM", "HD", "PG"]
        
        # Start with user universe if provided, otherwise use default
        candidates = user_universe if user_universe else default_universe
        
        # Create a prompt for the LLM to analyze stocks based on objectives
        prompt = f"""
        As a financial expert, analyze these stocks for trading suitability: {', '.join(candidates)}
        
        For each of these trading objectives: {', '.join(objectives)}
        
        Select the most suitable stocks considering:
        1. High liquidity (>1M average daily volume)
        2. Characteristic behavior matching each objective:
           - momentum: stocks with strong trends and follow-through
           - mean-reversion: stocks that oscillate around stable price levels
           - reversal: stocks that frequently change direction at extremes
        3. Minimal trading halts or liquidity issues
        
        For each stock that meets these criteria, produce a result in JSON format exactly as specified below. Do not include any additional text or explanation outside of the JSON object.

        Example JSON format:   
        ```json
        {{
            "selected_stocks": ["TICKER1", "TICKER2", ...],
            "rationale": {{
                "TICKER1": "Reason for selection",
                "TICKER2": "Reason for selection",
                ...
            }}
        }}
        ```
        Output only the JSON object.
        """
        
        #print(f"DEBUG - Querying LLM for stock selection based on objectives: {objectives}")
        
        try:
            # Use the agent's LLM capabilities to get a response
            #response = self.model_client.complete(prompt)
            response = await self.run(task=prompt)

            #print(f"DEBUG - LLM Response: {response}")
            
            # Parse the JSON response
            # First find JSON content between triple backticks if present
            import re
            import json
            

            # Try to extract the UniverseSelector message content robustly
            response_text = None
            if isinstance(response, list):
                for msg in response:
                    if getattr(msg, "source", None) == "UniverseSelector":
                        response_text = getattr(msg, "content", "")
                        break
            elif hasattr(response, "messages"):
                for msg in response.messages:
                    if getattr(msg, "source", None) == "UniverseSelector":
                        response_text = getattr(msg, "content", "")
                        break
            elif hasattr(response, "source") and getattr(response, "source") == "UniverseSelector":
                response_text = getattr(response, "content", "")
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)

            if not response_text:
                logger.warning("No UniverseSelector message content found.")
                return candidates[:min(len(candidates), 10)]

            #print("Raw UniverseSelector content:")
            #print(response_text)

            # Extract JSON block
            json_match = re.search(
                r"```json\s*\n([\s\S]*?)\n```", response_text, re.MULTILINE
            )
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'(\{[\s\S]*"selected_stocks"[\s\S]*\})', response_text)
                if json_match:
                    json_str = json_match.group(1).strip()
                else:
                    logger.warning("Could not parse LLM response as JSON, using default stocks")
                    return candidates[:min(len(candidates), 10)]

            # Clean up common LLM artifacts
            json_str = json_str.replace("...", "")  # Remove ellipsis if present
            json_str = re.sub(r",\s*}", "}", json_str)  # Remove trailing commas in objects
            json_str = re.sub(r",\s*]", "]", json_str)  # Remove trailing commas in arrays

            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError: {e}")
                logger.error(f"Raw response_text:\n{response_text}")
                logger.error(f"Extracted JSON string:\n{json_str}")
                # Optionally, try ast.literal_eval as a last resort
                import ast
                try:
                    result = ast.literal_eval(json_str)
                    logger.warning("Used ast.literal_eval as fallback for JSON parsing.")
                except Exception as e2:
                    logger.error(f"Fallback parsing also failed: {e2}")
                    return candidates[:min(len(candidates), 10)]

            # Validate result structure
            if not isinstance(result, dict) or "selected_stocks" not in result:
                logger.error("Parsed JSON does not contain 'selected_stocks'.")
                return candidates[:min(len(candidates), 10)]

            selected_stocks = result.get("selected_stocks", [])
            '''
            if "rationale" in result:
                print("\nDEBUG - Selection Rationale:")
                for ticker, reason in result["rationale"].items():
                    print(f"  {ticker}: {reason}")
            '''
            if not selected_stocks:
                logger.warning("LLM returned empty stock list, using default stocks")
                return candidates[:min(len(candidates), 10)]
            return selected_stocks
        except Exception as e:
            logger.error(f"Error querying LLM for stock selection: {e}")
            # Fallback to a simple selection algorithm if LLM fails
            return candidates[:min(len(candidates), 10)]



class TimeWindowPlanner(AssistantAgent):
    def __init__(self, name="TimeWindowPlanner"):
        super().__init__(
            name=name,
            system_message = """
            You are the TimeWindowPlanner agent. Your task is to determine date and time ranges (including outside RTH) for each ticker, based on trading objectives and volatility zones. For each objective, select recent earnings seasons, FOMC/CPI dates, and high-VIX windows.

            Input: JSON object mapping objectives to lists of tickers, e.g.:
            {
              "momentum": ["AAPL", "MSFT", "NVDA", "TSLA"],
              "mean_reversion": ["AAPL", "MSFT"],
              "reversal": ["NVDA", "TSLA"]
            }

            Output: JSON object with this format (no extra text):
            {
              "date_ranges": {
                "AAPL": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
                "MSFT": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
                ...
              },
              "time_range": {"start_time": "04:00", "end_time": "20:00", "timezone": "US/Eastern"}
            }
            Output only the JSON object.
            """,
            model_client=get_llm_client(name),  # Assumes a mapping exists in your config
        )
    
    async def plan_time_windows(self, universe: Dict[str, list], objectives: List[str]) -> Dict:        
        """
        Plan time windows for data collection.
        
        Args:
            universe: List of tickers
            objectives: List of objectives
            
        Returns:
            Dict with date_ranges and time_range
        """
        logger.info(f"Planning time windows for universe of {len(universe)} tickers")

        

        # Use the agent's LLM capabilities to get a response
        #response = self.model_client.complete(prompt)
            
        prompt = (
                f"Objectives: {', '.join(objectives)}\n"
                f"Universe:\n{json.dumps(universe, indent=2)}\n"
                "For each ticker, select appropriate date ranges as described."
        )
        print(f"DEBUG - Querying LLM for time window planning with prompt:\n{prompt}\n")
        response = await self.run(task=prompt)

        # Extract content if response is a message object/list
        response_text = None
        if isinstance(response, list):
            for msg in response:
                if getattr(msg, "source", None) == "TimeWindowPlanner":
                    response_text = getattr(msg, "content", "")
                    break
        elif hasattr(response, "messages"):
            for msg in response.messages:
                if getattr(msg, "source", None) == "TimeWindowPlanner":
                    response_text = getattr(msg, "content", "")
                    break
        elif hasattr(response, "source") and getattr(response, "source") == "TimeWindowPlanner":
            response_text = getattr(response, "content", "")
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)

        if not response_text:
            logger.error("No TimeWindowPlanner message content found.")
            return {}

        #print(f"DEBUG - Raw TimeWindowPlanner content:\n{response_text}\n") 
        import re


        # Extract JSON block
        json_match = re.search(r"```json\s*\n([\s\S]*?)\n```", response_text, re.MULTILINE)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_match = re.search(r'(\{[\s\S]*"date_ranges"[\s\S]*\})', response_text)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                logger.error("Could not parse LLM response as JSON.")
                return {}

        # Clean up common LLM artifacts
        json_str = json_str.replace("...", "")
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        try:
            result = json.loads(json_str)
        except Exception as e:
            logger.error(f"Error parsing JSON from LLM response: {e}")
            return {}

        # Validate result structure
        if not isinstance(result, dict) or "date_ranges" not in result:
            logger.error("Parsed JSON does not contain 'date_ranges'.")
            return {}

        return result

class EventTagger(AssistantAgent):
    def __init__(self, name="EventTagger"):
        super().__init__(
            name=name,
            system_message="""You are the EventTagger agent. Your task is to annotate each date range with event flags 
            for special days (earnings, macro announcements). Cross-reference earnings calendar and macro schedule, 
            and emit warnings if an event day has low extended-hours volume.""",
            model_client=get_llm_client(name),  # Assumes a mapping exists in your config
        )
    

    def get_events_yfinance(self,symbol: str, start: str, end: str) -> list:

        """
        Fetch earnings, splits, and dividends for a symbol between start and end dates using yfinance.
        Returns a list of dicts: [{'date': 'YYYY-MM-DD', 'event': 'earnings'|'split'|'dividend'}]
        """
        ticker = yf.Ticker(symbol)
        events = []

        # Earnings
        try:
            earnings = ticker.earnings_dates
            if earnings is not None:
                for date in earnings.index:
                    date_str = str(date.date())
                    if start <= date_str <= end:
                        events.append({'date': date_str, 'event': 'earnings'})
        except Exception as e:
            logger.warning(f"Could not fetch earnings for {symbol}: {e}")

        # Splits
        try:
            splits = ticker.splits
            if splits is not None and not splits.empty:
                for date in splits.index:
                    date_str = str(date.date())
                    if start <= date_str <= end:
                        events.append({'date': date_str, 'event': 'split'})
        except Exception as e:
            logger.warning(f"Could not fetch splits for {symbol}: {e}")

        # Dividends
        try:
            dividends = ticker.dividends
            if dividends is not None and not dividends.empty:
                for date in dividends.index:
                    date_str = str(date.date())
                    if start <= date_str <= end:
                        events.append({'date': date_str, 'event': 'dividend'})
        except Exception as e:
            logger.warning(f"Could not fetch dividends for {symbol}: {e}")

        return events


    def tag_events(self, date_ranges: Dict) -> List[Dict]:
        """
        Tag events for each date range using yfinance.
        Args:
            date_ranges: Dict with date ranges for each symbol
        Returns:
            List of tagged ranges
        """
        logger.info("Tagging events for date ranges using yfinance")
        tagged_ranges = []

        for symbol, dr in date_ranges.items():
            start = dr["start"]
            end = dr["end"]
            events = self.get_events_yfinance(symbol, start, end)
            # For each event, add to tagged_ranges
            for event in events:
                tagged_ranges.append({
                    "symbol": symbol,
                    "date": event["date"],
                    "events": [event["event"]]
                })
        return tagged_ranges


    def tag_eventsOld(self, date_ranges: Dict) -> List[Dict]:
        """
        Tag events for each date range.
        
        Args:
            date_ranges: Dict with date ranges for each symbol
            
        Returns:
            List of tagged ranges
        """
        logger.info("Tagging events for date ranges")
        
        # This would normally pull from an earnings/events database
        # For now, we'll use a static sample of events
        sample_events = {
            "AAPL": [
                {"date": "2024-01-25", "event": "earnings"},
                {"date": "2024-04-30", "event": "earnings"}
            ],
            "MSFT": [
                {"date": "2024-01-30", "event": "earnings"},
                {"date": "2024-04-25", "event": "earnings"}
            ],
            "NVDA": [
                {"date": "2024-02-21", "event": "earnings"},
                {"date": "2024-05-22", "event": "earnings"}
            ]
        }
        
        # Global macro events
        macro_events = [
            {"date": "2024-03-20", "event": "FOMC"},
            {"date": "2024-05-10", "event": "CPI"},
            {"date": "2024-02-02", "event": "NFP"}
        ]
        
        # DEBUG: Print date ranges and event dates
        print("\nDEBUG - Date Ranges for symbols:")
        for symbol, date_range in date_ranges.items():
            print(f"  {symbol}: {date_range['start']} to {date_range['end']}")
    
        print("\nDEBUG - Sample Events:")
        for symbol, events in sample_events.items():
            print(f"  {symbol}: {[e['date'] for e in events]}")
    
        print(f"\nDEBUG - Macro Events: {[e['date'] for e in macro_events]}")
    
        tagged_ranges = []
        
        for symbol, date_range in date_ranges.items():
            start = pd.to_datetime(date_range["start"])
            end = pd.to_datetime(date_range["end"])
            
            # Collect all dates in range
            all_dates = pd.date_range(start=start, end=end, freq='D')
            
            for date in all_dates:
                date_str = date.strftime("%Y-%m-%d")
                events = []
                
                # Add symbol-specific events
                if symbol in sample_events:
                    for event in sample_events[symbol]:
                        if event["date"] == date_str:
                            events.append(event["event"])
                
                # Add macro events
                for event in macro_events:
                    if event["date"] == date_str:
                        events.append(event["event"])
                
                if events:
                    tagged_ranges.append({
                        "symbol": symbol,
                        "date": date_str,
                        "events": events
                    })
        
        return tagged_ranges

class SamplerSplitter(AssistantAgent):
    def __init__(self, name="SamplerSplitter"):
        super().__init__(
            name=name,
            system_message="""You are the SamplerSplitter agent. Your task is to create non-overlapping training/test 
            splits with warm-up days, avoiding leakage. For each symbol, allocate 80% of days to training, 20% to test 
            with 3–5 warm-up days. Write out CSV files: training_{symbol}.csv, test_{symbol}.csv.""",
            model_client=get_llm_client(name),  # Assumes a mapping exists in your config
        )
    
    def create_splits(self, tagged_ranges: List[Dict], parameters: Dict) -> Dict[str, List[str]]:
        """
        Create training and test splits.
        
        Args:
            tagged_ranges: List of tagged date ranges
            parameters: Dict of parameters
            
        Returns:
            Dict with training_files and test_files paths
        """
        logger.info("Creating training/test splits")
        
        # Group by symbol
        symbol_dates = {}
        for item in tagged_ranges:
            symbol = item["symbol"]
            date = item["date"]
            
            if symbol not in symbol_dates:
                symbol_dates[symbol] = []
            
            symbol_dates[symbol].append(date)
        
        # Sort dates for each symbol
        for symbol in symbol_dates:
            symbol_dates[symbol] = sorted(symbol_dates[symbol])
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(PROJECT_ROOT, "data", "dqn")
        os.makedirs(output_dir, exist_ok=True)
        
        training_files = []
        test_files = []
        
        for symbol, dates in symbol_dates.items():
            # Calculate split point (80% training, 20% test)
            split_idx = int(len(dates) * 0.8)
            
            # Warm-up days (we'll use 5)
            warmup_days = 5
            if split_idx > warmup_days:
                split_idx -= warmup_days
            
            # Split dates
            train_dates = dates[:split_idx]
            test_dates = dates[split_idx:]
            
            # Add warmup days to test
            test_dates_with_warmup = dates[split_idx-warmup_days:] if split_idx > warmup_days else dates
            
            # Create file paths
            train_file = os.path.join(output_dir, f"training_{symbol}.csv")
            test_file = os.path.join(output_dir, f"test_{symbol}.csv")
            
            # Write stub files to indicate the split (actual data will be fetched by IBKRFetcherValidator)
            with open(train_file, 'w') as f:
                f.write(f"# Training data for {symbol}\n")
                f.write(f"# Dates: {','.join(train_dates)}\n")
            
            with open(test_file, 'w') as f:
                f.write(f"# Test data for {symbol}\n")
                f.write(f"# Dates: {','.join(test_dates_with_warmup)}\n")
                f.write(f"# Warmup days: {warmup_days}\n")
            
            training_files.append(train_file)
            test_files.append(test_file)
        
        return {
            "training_files": training_files,
            "test_files": test_files
        }

class IBKRFetcherValidator(AssistantAgent):
    def __init__(self, name="IBKRFetcherValidator"):
        super().__init__(
            name=name,
            system_message="""You are the IBKRFetcherValidator agent. Your task is to fetch OHLCV data from IBKR in 
            30-day chunks, validate completeness and volume. Check missing_pct ≤ parameters.max_missing_pct, 
            volume ≥ parameters.min_volume_per_bar. Log warnings or exclude segments; write final validated CSVs to disk.""",
            model_client=get_llm_client(name),  # Assumes a mapping exists in your config
        )
    
    def fetch_and_validate(self, training_files: List[str], test_files: List[str], parameters: Dict) -> List[str]:
        """
        Fetch and validate OHLCV data.
        
        Args:
            training_files: List of training file paths
            test_files: List of test file paths
            parameters: Dict of parameters
            
        Returns:
            List of validated file paths
        """
        logger.info("Fetching and validating OHLCV data")
        
        all_files = training_files + test_files
        validated_files = []
        
        for file_path in all_files:
            symbol = os.path.basename(file_path).split('_')[1].split('.')[0]
            
            # Read date range from stub file
            with open(file_path, 'r') as f:
                lines = f.readlines()
                dates_line = [line for line in lines if line.startswith("# Dates:")][0]
                dates = dates_line.replace("# Dates:", "").strip().split(',')
            
            if not dates:
                logger.warning(f"No dates found in {file_path}")
                continue
            
            # Sort dates
            dates = sorted(dates)
            start_date = dates[0]
            end_date = dates[-1]
            
            # Process in 30-day chunks
            current_date = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            all_data = []
            
            while current_date <= end_dt:
                chunk_end = min(current_date + timedelta(days=30), end_dt)
                
                # Fetch data
                df = fetch_5min_bars(
                    symbol=symbol,
                    start=current_date.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    use_rth=False  # Include extended hours
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol} from {current_date} to {chunk_end}")
                    current_date = chunk_end + timedelta(days=1)
                    continue
                
                # Validate data
                trading_days = df.index.map(lambda x: x.date()).unique()
                expected_days = pd.date_range(start=current_date, end=chunk_end, freq='B')
                expected_days = [d.date() for d in expected_days]
                
                # Check for missing days
                missing_days = set(expected_days) - set(trading_days)
                missing_pct = len(missing_days) / len(expected_days) if expected_days else 0
                
                if missing_pct > parameters.get("max_missing_pct", 0.05):
                    logger.warning(f"High missing data ({missing_pct:.2%}) for {symbol} from {current_date} to {chunk_end}")
                
                # Check for low volume
                low_volume_bars = df[df['Volume'] < parameters.get("min_volume_per_bar", 100)]
                if not low_volume_bars.empty:
                    logger.warning(f"{len(low_volume_bars)} bars with low volume for {symbol} from {current_date} to {chunk_end}")
                
                # Keep the validated data
                all_data.append(df)
                
                # Move to next chunk
                current_date = chunk_end + timedelta(days=1)
            
            if all_data:
                # Combine all chunks
                combined_df = pd.concat(all_data)
                
                # Write to CSV
                output_file = file_path.replace(".csv", "_validated.csv")
                combined_df.to_csv(output_file)
                logger.info(f"Wrote validated data to {output_file}")
                validated_files.append(output_file)
            else:
                logger.warning(f"No valid data for {symbol}, skipping")
        
        return validated_files

# --- Orchestration ---
class DataProvisioningOrchestrator:
    def __init__(self, parameters: Dict = None):
        """
        Initialize the orchestrator with parameters.
        
        Args:
            parameters: Dict of parameters
        """
        self.parameters = parameters or {
            "max_missing_pct": 0.05,
            "min_volume_per_bar": 100
        }
        
        # Initialize agents
        self.universe_selector = UniverseSelector()
        self.time_window_planner = TimeWindowPlanner()
        self.event_tagger = EventTagger()
        self.sampler_splitter = SamplerSplitter()
        self.ibkr_fetcher_validator = IBKRFetcherValidator()
        
        logger.info("DataProvisioningOrchestrator initialized with parameters: %s", self.parameters)
    
    async def run(self, objectives: List[str], user_universe: Optional[List[str]] = None) -> Dict:
        """
        Run the full data provisioning pipeline.
        
        Args:
            objectives: List of objectives (momentum, mean-reversion, reversal)
            user_universe: Optional user-provided list of tickers
            
        Returns:
            Dict with results
        """
        logger.info("Starting data provisioning pipeline")
        
        # Step 1: Select universe
        universe = await self.universe_selector.select_universe(objectives, user_universe)
        logger.info(f"Selected universe: {universe}")
        
        # Step 2: Plan time windows
        window_plan = await self.time_window_planner.plan_time_windows(universe, objectives)
        logger.info(f"Time window plan created")
        print(f"WINDOW PLAN :::: {window_plan}")
        # Step 3: Tag events
        tagged_ranges = self.event_tagger.tag_events(window_plan["date_ranges"])
        logger.info(f"Tagged {len(tagged_ranges)} event days")
        
        '''
        # TODO (for next session)
        - Continue from EventTagger.tag_events (yfinance version working)
        - Next: Add Polygon.io support for event tagging
        - Last tested: <date>

        Let's continue from where we left off. Here is my current EventTagger class:"
        and paste the code.

        # TODO / SESSION RESUME TEMPLATE

        ## Last Working State
        - [ ] EventTagger.tag_events using yfinance is working and integrated.
        - [ ] DataProvisioningOrchestrator runs end-to-end with Yahoo Finance event tagging.

        ## Next Steps
        - [ ] Add Polygon.io support for event tagging (swap out yfinance logic).
        - [ ] (Optional) Add macro event tagging (FOMC, CPI, NFP) from external sources.
        - [ ] Test pipeline with a larger universe and longer date ranges.

        ## Notes
        - Last tested: <insert date here>
        - Current file: common_logic/data_provisioning/dqn_data_agent_system.py
        - Key class to resume: EventTagger

        ## How to Resume
        1. Open this file and review the last working code.
        2. Instruct GitHub Copilot:  
           > "Let's continue from where we left off. Here is my current EventTagger class:"
           (Paste the class code if needed.)
        3. Proceed with the next steps above.

        ---


        '''


        # Step 4: Create splits
        split_files = self.sampler_splitter.create_splits(tagged_ranges, self.parameters)
        logger.info(f"Created {len(split_files['training_files'])} training files and {len(split_files['test_files'])} test files")
        
        # Step 5: Fetch and validate data
        validated_files = self.ibkr_fetcher_validator.fetch_and_validate(
            split_files["training_files"],
            split_files["test_files"],
            self.parameters
        )
        logger.info(f"Validated {len(validated_files)} files")
        
        return {
            "universe": universe,
            "window_plan": window_plan,
            "tagged_ranges": tagged_ranges,
            "split_files": split_files,
            "validated_files": validated_files
        }

# --- Main execution ---
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run DQN data provisioning pipeline")
    parser.add_argument("--objectives", nargs="+", default=["momentum", "mean-reversion", "reversal"],
                       help="Trading objectives (momentum, mean-reversion, reversal)")
    parser.add_argument("--universe", nargs="+", default=None,
                       help="Optional list of tickers to use")
    parser.add_argument("--max-missing-pct", type=float, default=0.05,
                       help="Maximum fraction of missing 5-min bars allowed")
    parser.add_argument("--min-volume-per-bar", type=int, default=100,
                       help="Minimum shares per 5-min bar")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for data files")
    
    args = parser.parse_args()
    
    # Set up parameters
    parameters = {
        "max_missing_pct": args.max_missing_pct,
        "min_volume_per_bar": args.min_volume_per_bar
    }
    
    # Set output directory if provided
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run orchestrator
    orchestrator = DataProvisioningOrchestrator(parameters)
    results = orchestrator.run(args.objectives, args.universe)
    
    # Print results summary
    print("\nData Provisioning Pipeline Results:")
    print(f"- Universe: {results['universe']}")
    print(f"- Date ranges: {len(results['window_plan']['date_ranges'])} symbols")
    print(f"- Tagged events: {len(results['tagged_ranges'])} days")
    print(f"- Training files: {len(results['split_files']['training_files'])}")
    print(f"- Test files: {len(results['split_files']['test_files'])}")
    print(f"- Validated files: {len(results['validated_files'])}")
    
    # Output file paths
    print("\nValidated data files:")
    for file_path in results['validated_files']:
        print(f"- {file_path}")

if __name__ == "__main__":
    # Set up parameters
    parameters = {
        "max_missing_pct": 0.05,
        "min_volume_per_bar": 100
    }

    # Create orchestrator
    orchestrator = DataProvisioningOrchestrator(parameters)


    # Run the pipeline
    results = asyncio.run(orchestrator.run(
        objectives=["momentum", "mean-reversion", "reversal"],
        user_universe=["AAPL", "MSFT", "NVDA", "TSLA"]
    ))
    # Access the validated data files
    validated_files = results["validated_files"]
    print("\nValidated data files:")
    for file_path in results['validated_files']:
        print(f"- {file_path}")
