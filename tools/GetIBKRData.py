# -*- coding: utf-8 -*-
# GetIBKRData.py

import os
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import joblib
from ib_insync import IB, Stock, Forex, Contract, util # Add other contract types if needed
import backtrader as bt

CACHE_DIR = "cache_ibkr"  # Separate cache directory for IBKR data
os.makedirs(CACHE_DIR, exist_ok=True)

# IBKR Connection Parameters (Consider moving to a config file)
IBKR_HOST = '127.0.0.1'
IBKR_PORT = 7497  # 7497 for TWS, 4001 for Gateway (default paper), 4002 for Gateway (default live)
IBKR_CLIENT_ID = 10 # Choose an unused client ID
CONNECTION_RETRIES = 3
RETRY_DELAY_SECONDS = 5 # Delay between retry attempts
CONNECTION_TIMEOUT_SECONDS = 10 # Timeout for each connection attempt

# Global IB instance to manage connection
ib = IB()

async def connect_ibkr(retries=CONNECTION_RETRIES, 
                       delay_seconds=RETRY_DELAY_SECONDS,
                       timeout_seconds=CONNECTION_TIMEOUT_SECONDS):
    """
    Connects to IBKR if not already connected, with retry logic.
    """
    if not ib.isConnected():
        for attempt in range(1, retries + 1):
            try:
                print(f"Attempt {attempt}/{retries}: Connecting to IBKR at {IBKR_HOST}:{IBKR_PORT} (ClientID: {IBKR_CLIENT_ID})...")
                await ib.connectAsync(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=timeout_seconds)
                print("Successfully connected to IBKR.")
                return  # Exit if connected successfully
            except (ConnectionRefusedError, asyncio.TimeoutError, OSError) as e: # Common connection issues
                print(f"Connection attempt {attempt} failed: {type(e).__name__} - {e}")
                if attempt < retries:
                    print(f"Ensure IB TWS/Gateway is running and API connections are enabled.")
                    print(f"Retrying in {delay_seconds} seconds...")
                    await asyncio.sleep(delay_seconds)
                else:
                    print("All connection attempts to IBKR failed.")
                    raise ConnectionError(
                        f"Failed to connect to IBKR at {IBKR_HOST}:{IBKR_PORT} after {retries} attempts. "
                        f"Last error: {type(e).__name__} - {e}. "
                        "Please ensure IB TWS/Gateway is running, API connections are enabled, "
                        "and network settings are correct."
                    ) from e
            except Exception as e: # Catch any other unexpected error during connection
                print(f"An unexpected error occurred during connection attempt {attempt}: {type(e).__name__} - {e}")
                if attempt < retries:
                    print(f"Retrying in {delay_seconds} seconds...")
                    await asyncio.sleep(delay_seconds)
                else:
                    print("All connection attempts failed due to an unexpected error.")
                    raise ConnectionError(
                        f"Failed to connect to IBKR due to an unexpected error after {retries} attempts. "
                        f"Last error: {type(e).__name__} - {e}."
                    ) from e
    # else:
    #     print("Already connected to IBKR.") # Or just silently proceed

# ... (rest of your _get_cache_filename, _load_from_cache, etc. functions remain the same) ...

def _get_cache_filename(contract_symbol, start_date_str, end_date_str, bar_size_str, what_to_show_str):
    """Generates a unique cache filename."""
    # Sanitize bar_size_str for filename (e.g., '1 min' -> '1_min')
    safe_bar_size = bar_size_str.replace(' ', '_')
    return os.path.join(CACHE_DIR, f"{contract_symbol}_{start_date_str}_{end_date_str}_{safe_bar_size}_{what_to_show_str}.pkl")

def _load_from_cache(cache_file):
    """Loads data from a joblib cache file."""
    try:
        print(f"Loading from cache: {cache_file}")
        df = joblib.load(cache_file)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        print(f"Loaded {len(df)} rows from cache.")
        return df
    except Exception as e:
        print(f"Error loading cache file {cache_file}: {e}")
        return None

def _save_to_cache(df, cache_file):
    """Saves data to a joblib cache file."""
    try:
        joblib.dump(df, cache_file)
        print(f"Cached data to {cache_file}")
    except Exception as e:
        print(f"Error saving cache file {cache_file}: {e}")

def _get_contract(symbol, sec_type='STK', exchange='SMART', currency='USD', primary_exchange=None):
    """
    Creates an IB Contract object.
    For non-US stocks, you might need to specify primary_exchange (e.g., 'LSE' for London Stock Exchange).
    For forex, sec_type='CASH'. For futures, sec_type='FUT', etc.
    """
    if sec_type.upper() == 'STK':
        contract = Stock(symbol, exchange, currency)
        if primary_exchange: # For ambiguous symbols like some Canadian stocks
            contract.primaryExchange = primary_exchange
    elif sec_type.upper() == 'CASH' or sec_type.upper() == 'FOREX':
        if len(symbol) != 6:
            raise ValueError(f"Forex symbol '{symbol}' should be 6 characters (e.g., EURUSD).")
        # For Forex, the symbol itself defines the pair (e.g., EURUSD).
        # Exchange is typically 'IDEALPRO'. Currency is the quote currency.
        contract = Forex(symbol) # symbol is like 'EURUSD'
        contract.exchange = 'IDEALPRO' # IDEALPRO is the IBKR ECN for Forex
        # The currency parameter for Forex usually refers to the quote currency,
        # e.g., for EURUSD, the quote currency is USD.
        # ib_insync's Forex contract usually infers this, but explicit can be good.
        # contract.currency = symbol[3:] # e.g., USD from EURUSD
    # ... other sec_types


        # Forex pairs typically don't need an exchange specified at the contract level for historical data,
        # but SMART routing might not work as expected for all pairs; sometimes specific exchanges like IDEALPRO are used.
    # Add other contract types (Futures, Options, CFDs, etc.) as needed
    # elif sec_type.upper() == 'FUT':
    #     contract = Future(symbol, lastTradeDateOrContractMonth, exchange, currency=currency) # Requires more details
    else:
        raise ValueError(f"Unsupported security type: {sec_type}")
    return contract

def _get_duration_and_chunk_params(bar_size):
    """
    Returns max duration string and chunk size in days for IBKR historical data requests.
    Ref: https://interactivebrokers.github.io/tws-api/historical_limitations.html
    These are general guidelines; some flexibility exists. ib_insync might handle some larger requests.
    """
    # (max_duration_for_single_request, typical_chunk_duration_for_looping_days)
    # Durations: S (Seconds), D (Day), W (Week), M (Month), Y (Year)
    if bar_size in ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs']:
        return '1800 S', timedelta(seconds=1800) # 30 minutes
    elif bar_size == '1 min':
        return '1 D', timedelta(days=1) # Can often get up to 7 days
    elif bar_size == '5 mins':
        return '2 D', timedelta(days=2)
    elif bar_size == '15 mins':
        return '1 W', timedelta(days=7)
    elif bar_size == '30 mins':
        return '1 W', timedelta(days=7)
    elif bar_size == '1 hour':
        return '1 M', timedelta(days=30)
    elif bar_size == '4 hours':
        return '1 M', timedelta(days=30) # Or '2 M'
    elif bar_size == '1 day':
        return '1 Y', timedelta(days=365) # Can request many years
    else:
        raise ValueError(f"Unsupported bar size for IBKR: {bar_size}")


async def get_ibkr_data_async(
    ib_instance: IB,  # Pass the IB instance to avoid global state issues
    ticker_symbol: str,
    start_date: str,  # YYYY-MM-DD
    end_date: str,    # YYYY-MM-DD
    bar_size: str = '1 day',  # e.g., '1 min', '1 hour', '1 day'
    what_to_show: str = 'TRADES',  # e.g., 'TRADES', 'MIDPOINT', 'BID', 'ASK'
    use_rth: bool = True, # Use Regular Trading Hours
    sec_type: str = 'STK', # STK, CASH (Forex), FUT, OPT
    exchange: str = 'SMART',
    currency: str = 'USD',
    primary_exchange: str = None # e.g. 'NASDAQ', 'NYSE', 'LSE' etc. for ambiguous stocks
):
    """
    Asynchronously fetches historical data from IBKR, with caching.
    Returns a pandas DataFrame and a Backtrader PandasData feed.
    """
    await connect_ibkr()
    # No need to check ib.isConnected() here again, as connect_ibkr would have raised if failed.
    #if not ib.isConnected():
    #    raise ConnectionError("IBKR not connected. Cannot fetch data.")

    contract = _get_contract(ticker_symbol, sec_type, exchange, currency, primary_exchange)
    # It's good practice to qualify contracts *after* a successful connection.
    try:
        print(f"Qualifying contract: {contract}")
        await ib.qualifyContractsAsync(contract) 
        print(f"Contract qualified: {contract}")
    except Exception as e:
        raise RuntimeError(f"Failed to qualify contract {contract}: {e}")

    cache_file = _get_cache_filename(ticker_symbol, start_date, end_date, bar_size, what_to_show)

    if os.path.exists(cache_file):
        df = _load_from_cache(cache_file)
        if df is not None and not df.empty:
            # Ensure standard Backtrader column names if loaded from cache
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 
                               'close': 'Close', 'volume': 'Volume', 'date': 'datetime'},
                      inplace=True, errors='ignore') # errors='ignore' if they are already correct
            if 'datetime' not in df.columns and df.index.name == 'date':
                 df.index.name = 'datetime' # Ensure index is named datetime for PandasData
            elif 'datetime' in df.columns: # if 'datetime' is a column, set it as index
                 df.set_index('datetime', inplace=True)

            if not isinstance(df.index, pd.DatetimeIndex): # Double check index
                df.index = pd.to_datetime(df.index)

            bt_data = bt.feeds.PandasData(dataname=df)
            return df, bt_data
        else:
            print(f"Cache for {ticker_symbol} was empty or failed to load. Re-fetching.")


    print(f"Fetching data for {contract.symbol} from {start_date} to {end_date}, Bar: {bar_size}")

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59) # Include full end day

    #df = await _fetch_historical_data_chunked(contract, end_date_dt, start_date_dt, bar_size, what_to_show, use_rth)
    df = await _fetch_historical_data_chunked(ib_instance, contract, end_date_dt, start_date_dt, bar_size, what_to_show, use_rth)

    if df.empty:
        raise RuntimeError(f"No data fetched for {ticker_symbol} from {start_date} to {end_date}")

    # Standardize column names for Backtrader (IBKR usually returns lowercase)
    # util.df already returns 'date', 'open', 'high', 'low', 'close', 'volume', 'barCount', 'average'
    # We need 'Open', 'High', 'Low', 'Close', 'Volume' and datetime index
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 
                       'close': 'Close', 'volume': 'Volume'}, 
              inplace=True)
    
    # Ensure 'datetime' index
    if 'date' in df.columns: # util.df creates 'date' column as index
        df.index.name = 'datetime'
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index.name = 'datetime'
    else: # Should not happen if util.df was used
        raise ValueError("DataFrame index is not a datetime object after fetching from IBKR.")

    # Select only standard OHLCV columns for Backtrader, plus OpenInterest if available and needed
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[required_cols]
    df.dropna(inplace=True) # Drop rows with any NaNs in essential columns

    if df.empty:
        raise RuntimeError(f"Data for {ticker_symbol} became empty after processing/dropna.")

    _save_to_cache(df, cache_file)
    bt_data = bt.feeds.PandasData(dataname=df)
    return df, bt_data


# --- Synchronous Wrapper ---
# In GetIBKRData.py

def get_ibkr_data_sync(ib_instance: IB, ticker_symbol: str, start_date: str, end_date: str, **kwargs):
    """
    Synchronous wrapper for get_ibkr_data_async.
    Manages the asyncio event loop by calling asyncio.run().
    """
    try:
        # The most straightforward way for a sync wrapper is to just use asyncio.run()
        # It will create a new event loop, run the coroutine, and close the loop.
        return asyncio.run(get_ibkr_data_async(ib_instance, ticker_symbol, start_date, end_date, **kwargs))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e) or \
           "Nesting asyncio event loops is not supported" in str(e):
            # This means asyncio.run() was called from within an existing loop (e.g. Jupyter)
            print("ERROR: Async loop issue. `asyncio.run()` cannot be nested directly. "
                  "If in Jupyter or an async environment, try `import nest_asyncio; nest_asyncio.apply()` "
                  "at the start of your script/notebook, OR ensure you `await get_ibkr_data_async` "
                  "from your async code instead of calling this sync wrapper.")
            # Re-raise a more specific error or the original one
            raise RuntimeError("Event loop conflict: asyncio.run() cannot be nested. See console for advice.") from e
        raise # Re-raise other RuntimeErrors

def get_ibkr_data_syncOld1(ticker_symbol: str, start_date: str, end_date: str, **kwargs):
    """
    Synchronous wrapper for get_ibkr_data_async.
    Manages the asyncio event loop.
    """
    try:
        # Try to get the running loop if one exists (e.g., in Jupyter/async context)
        loop = asyncio.get_running_loop()
        if loop.is_running():
            # If called from an already running asyncio loop.
            # This situation is tricky. Directly calling another asyncio.run() here will fail.
            # The best way is for the caller to `await` the async version.
            # If you must call from a sync function within an async context,
            # you'd typically use loop.create_task and then somehow wait for it,
            # or use a library like `nest_asyncio`.
            # For now, let's assume this sync wrapper is primarily for non-async top-level calls.
            print("WARNING: get_ibkr_data_sync called from an existing running event loop. "
                  "This can lead to issues. Prefer `await get_ibkr_data_async` in async code.")
            # This is a simplification; proper handling here is complex without `nest_asyncio`
            # For simplicity, we'll try asyncio.run() and let it error if nested.
            return asyncio.run(get_ibkr_data_async(ticker_symbol, start_date, end_date, **kwargs))

    except RuntimeError as e: # Catches "no running event loop"
        if "no running event loop" in str(e).lower():
            # No loop is running, so we can safely use asyncio.run()
            return asyncio.run(get_ibkr_data_async(ticker_symbol, start_date, end_date, **kwargs))
        elif "cannot be called from a running event loop" in str(e) or \
             "Nesting asyncio event loops is not supported" in str(e):
            # This means asyncio.run() was called from within an existing loop
            print("ERROR: Async loop issue. `asyncio.run()` cannot be nested. "
                  "If in Jupyter, try `import nest_asyncio; nest_asyncio.apply()` at the start. "
                  "If in async code, use 'await get_ibkr_data_async'.")
            raise RuntimeError("Event loop conflict. See console for details.") from e
        else:
            raise # Re-raise other RuntimeErrors

    # Fallback if get_running_loop() didn't raise but also didn't mean we should use asyncio.run()
    # This path should ideally not be hit if the above logic is correct.
    # Default to trying asyncio.run() as it's the most common case for a sync wrapper.
    return asyncio.run(get_ibkr_data_async(ticker_symbol, start_date, end_date, **kwargs))


def get_ibkr_data_syncOld(ticker_symbol: str, start_date: str, end_date: str, **kwargs):
    """
    Synchronous wrapper for get_ibkr_data_async.
    Manages the asyncio event loop.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If called from an already running asyncio loop (e.g. Jupyter notebook with %autoawait)
            # Create a task and run it.
            # This part might need adjustment based on where it's called from.
            # For simple script execution, a new loop is often fine if no other async code is running.
            print("Async event loop is already running. Creating task.")
            task = loop.create_task(get_ibkr_data_async(ticker_symbol, start_date, end_date, **kwargs))
            # This doesn't block and wait in a simple script, so might not be suitable alone.
            # Consider using `nest_asyncio` if you frequently run into "loop already running" issues.
            # Or, structure your main application to be async.
            # For now, let's assume it's called from a non-async context or a context that handles this.
            # A simpler approach for non-async callers:
            return asyncio.run(get_ibkr_data_async(ticker_symbol, start_date, end_date, **kwargs))
        else:
            return loop.run_until_complete(get_ibkr_data_async(ticker_symbol, start_date, end_date, **kwargs))
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e) or "Nesting asyncio event loops is not supported" in str(e):
            # This happens if asyncio.run() is called from within an existing loop (e.g. Jupyter)
            # One solution is to use nest_asyncio.apply() at the start of your script/notebook
            # import nest_asyncio
            # nest_asyncio.apply()
            # Then you can just call: return asyncio.run(...)
            # Or, if you are sure you are in a jupyter notebook or similar:
            # loop = asyncio.get_event_loop()
            # return loop.create_task(get_ibkr_data_async(ticker_symbol, start_date, end_date, **kwargs))
            # The above create_task won't block and wait for result in a simple script.
            print("ERROR: Async loop issue. If in Jupyter, try `import nest_asyncio; nest_asyncio.apply()` at the start.")
            print("For now, trying to get/create a new loop if possible, or re-raising.")
            # Fallback to new loop if not running, otherwise re-raise
            try:
                return asyncio.run(get_ibkr_data_async(ticker_symbol, start_date, end_date, **kwargs))
            except RuntimeError: # Catch the nested error again
                 # If still failing, it means we are truly in a running loop that doesn't allow asyncio.run()
                 # The user will need to manage how they call this async function.
                 # e.g., `await get_ibkr_data_async(...)` if their code is already async.
                 raise Exception("Failed to run async IBKR data fetch. Event loop conflict. "
                                 "If in async code, use 'await get_ibkr_data_async'. "
                                 "If in sync code in Jupyter, try 'nest_asyncio'.") from e
        raise


# --- Example Usage (for testing this module directly) ---
async def main_test():
    try:
        # Example: Fetch data for AAPL stock
        #df_aapl, bt_aapl = await get_ibkr_data_async('AAPL', '2022-01-01', '2023-01-01', bar_size='1 day')
        #print("\nAAPL Data:")
        #print(df_aapl.head())
        #print(df_aapl.info())

        # Example: Fetch data for EURUSD Forex
        df_nvda, bt_nvda = await get_ibkr_data_async(
            ib,
            'NVDA', '2023-11-01', '2023-12-01', 
            bar_size='15 mins', sec_type='STK', exchange='SMART',
            currency='USD',
            primary_exchange='NASDAQ'
        )
        print("\nNVIDIA Stock Data:")
        if df_nvda is not None and not df_nvda.empty:
            print(df_nvda.head())
            print(df_nvda.info())



        print("Fetching IBKR data for MSFT...")
        df_msft, bt_feed_msft = await get_ibkr_data_async(
            ib,
            ticker_symbol='MSFT',
            start_date='2022-01-01',
            end_date='2023-01-01',
            bar_size='1 day',
            sec_type='STK',
            exchange='SMART',
            currency='USD'
        )
        print("MSFT Data from IBKR:")
        if df_msft is not None and not df_msft.empty:
            print(df_msft.head())
        else:
            print("No data for MSFT returned.")




    except Exception as e:
        print(f"An error occurred in main_test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ib.isConnected():
            print("Disconnecting IBKR.")
            ib.disconnect()



# This is your example function
def run_my_backtest():
    try:
        print("Fetching IBKR data for MSFT...")
        df_msft, bt_feed_msft = get_ibkr_data_sync(
            ib,
            ticker_symbol='MSFT',
            start_date='2022-01-01',
            end_date='2023-01-01',
            bar_size='1 day',
            sec_type='STK',
            exchange='SMART',
            currency='USD'
        )
        print("MSFT Data from IBKR:")
        if df_msft is not None and not df_msft.empty:
            print(df_msft.head())
        else:
            print("No data for MSFT returned.")

        # ... (rest of your Backtrader setup using bt_feed_msft) ...

    except ConnectionError as ce:
        print(f"Connection Error during backtest: {ce}")
    except Exception as e:
        print(f"Error in backtest execution: {e}")
        import traceback
        traceback.print_exc()



'''

if __name__ == '__main__':
    # This part should be in the SCRIPT that CALLS GetIBKRData, not inside GetIBKRData.py itself
    # if you have a separate main_test in GetIBKRData.py
    
    # --- Option 1: Run main_test from GetIBKRData.py ---
    # from GetIBKRData import main_test as run_ibkr_module_test
    # print("--- Running GetIBKRData.py main_test ---")
    # asyncio.run(run_ibkr_module_test()) # Assuming main_test is async
    # print("--- Finished GetIBKRData.py main_test ---")

    # --- Option 2: Run your application's backtest ---
    print("\n--- Running application backtest (run_my_backtest) ---")
    run_my_backtest() # This calls the sync wrapper

    # Ensure disconnection at the very end of your application
    if ib.isConnected():
        print("Disconnecting IBKR at script end.")
        ib.disconnect()
        # ib_insync may need a moment for graceful disconnect in some cases
        # If you use asyncio.run() for the disconnect, it needs to be in an async func
        # asyncio.run(ib.disconnectAsync()) # if you want to be very proper with async disconnect
    print("Script finished.")
    
'''








if __name__ == '__main__':
    # If running this script directly:
    # util.patchAsyncio() # May be needed on Windows for ProactorEventLoop with ib_insync
    # For Jupyter or other environments where a loop is already running,
    # you might need nest_asyncio:
    # import nest_asyncio
    # nest_asyncio.apply()
    try:
        asyncio.run(main_test())
    except RuntimeError as e:
         if "cannot be called from a running event loop" in str(e):
            print("Main test failed due to running event loop. This can happen in some IDEs/Jupyter.")
            print("Try running specific test functions or ensure nest_asyncio is used if applicable.")
         else:
            raise


#Explanation and Key Points for GetIBKRData.py:
#CACHE_DIR: Separate cache for IBKR data (cache_ibkr).

#IBKR Connection:
#IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID: Configuration for connecting. Make sure your TWS/Gateway is running and API connections are enabled.

#ib = IB(): Global ib_insync instance.

#connect_ibkr(): Async function to establish the connection. Called before data fetching.

#Caching:

#_get_cache_filename(): Creates a unique filename including bar size and data type.

#_load_from_cache(), _save_to_cache(): Use joblib (same as your Yahoo script).

#Contract Definition (_get_contract):

#IBKR requires a Contract object. This helper creates one.
#Starts with STK (stocks) and CASH (Forex). Can be extended for Futures, Options, etc.
#íb.qualifyContractsAsync(contract): Very important step to resolve contract details with IBKR's servers.

#Duration & Chunking Logic (_get_duration_and_chunk_params, _fetch_historical_data_chunked):
#IBKR has limits on how much data can be fetched in a single reqHistoricalData call, depending on barSizeSetting.

#_get_duration_and_chunk_params provides rough guidelines for max request duration and a typical chunk size for iterating.

#_fetch_historical_data_chunked is the core fetching loop. It:

#Starts from end_date_dt and works backward.
#Uses max_req_duration_str for each request.
#Concatenates results.
#Includes a politeness delay (asyncio.sleep(2)) to respect IBKR rate limits (approx. 60 requests/min).
#Filters the final concatenated DataFrame to the precise start_date_dt and end_date_dt.

#Main Fetching Function (get_ibkr_data_async):

#The main asynchronous function.

#Handles connection, contract qualification, cache checking, fetching, processing, and saving to cache.

#Converts IBKR's output (often lowercase column names, 'date' column) to Backtrader's expected format (camel case like 'Open', 'High', 'datetime' index).

#Synchronous Wrapper (get_ibkr_data_sync):

#Provides a way to call the async fetching logic from synchronous code by managing the asyncio event loop.

#Includes basic error handling for common asyncio loop issues (e.g., when running in Jupyter notebooks where a loop might already be active). For robust Jupyter use, nest_asyncio is often recommended.

#Column Naming: IBKR data (via util.df) typically comes with columns like 'date', 'open', 'high', 'low', 'close', 'volume'. The code renames these to 'datetime' (for the index) and 'Open', 'High', 'Low', 'Close', 'Volume' as expected by bt.feeds.PandasData.

#Rate Limiting: The asyncio.sleep(2) is a simple measure. For very heavy use, more sophisticated rate limit handling might be needed.

#Error Handling: Basic error handling is included. Robust production systems would need more.

#primary_exchange: For stocks that might be listed on multiple exchanges under the same symbol (e.g., some Canadian stocks like 'RY' which is on TSE and NYSE), specifying primary_exchange (e.g., primary_exchange='TSE' or primary_exchange='NYSE') helps IBKR find the correct contract. For most major US stocks on SMART, it's not strictly necessary but can be good practice.

#How to Use GetIBKRData.py (Example):

# In your main backtesting script or notebook

# If running in Jupyter and facing event loop issues:
# import nest_asyncio
# nest_asyncio.apply()

#Next Steps :

#Integrate into Backtesting: Adapt your main backtesting script to use get_ibkr_data_sync (or _async if your script is async) to fetch data.
#Expand Contract Types: If you trade futures, options, etc., you'll need to enhance the _get_contract function and possibly the parameters to get_ibkr_data_async.

#This IBKR module provides a solid starting point. 
#Remember that financial data APIs, especially for brokerage platforms, 
#can have many nuances (contract specification, rate limits, data types). 
