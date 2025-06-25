import yfinance as yf
import pandas as pd
from datetime import time, datetime
import pandas_market_calendars as mcal
import os
from pathlib import Path

# Assuming config.py is in the parent directory of common_logic
try:
    from config import DATA_DIR # Centralized data directory
except ImportError:
    # Fallback if config.py is not found or DATA_DIR is not defined
    print("Warning: Could not import DATA_DIR from config. Defaulting to 'project_root/data'.")
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Adjust if structure is different
    DATA_DIR = PROJECT_ROOT / "data"
    os.makedirs(DATA_DIR, exist_ok=True)


LIQUID_TICKERS = {'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'SPY', 'QQQ'}

# Custom Exceptions for better error handling
class DataFetchingError(Exception):
    """Base class for errors in the data fetching process."""
    pass

class TickerNotAllowedError(DataFetchingError):
    """Raised when a ticker is not in the allowed list."""
    pass

class NoDataFoundError(DataFetchingError):
    """Raised when yfinance returns no data."""
    pass

class DataColumnMissingError(DataFetchingError):
    """Raised when essential OHLCV columns are missing after download."""
    pass

class DataQualityError(DataFetchingError):
    """Raised when data quality checks fail (e.g., too many missing days)."""
    pass


def fetch_intraday_period(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",
    max_missing_days: int = 2,
    include_extended_hours: bool = False
) -> str:
    """
    Fetches, cleans, validates, and saves intraday financial data for a given ticker.

    Args:
        ticker (str): The stock ticker symbol.
        period (str): The period for which to fetch data (e.g., "5d", "1mo").
        interval (str): The data interval (e.g., "1m", "5m", "1h").
        max_missing_days (int): Maximum number of days with missing or incomplete data allowed.
        include_extended_hours (bool): Whether to include pre/post-market data.

    Returns:
        str: A message indicating success (including file path) or failure with error details.

    Raises:
        TickerNotAllowedError: If the ticker is not in the LIQUID_TICKERS list.
        NoDataFoundError: If yfinance returns an empty DataFrame.
        DataColumnMissingError: If essential OHLCV columns cannot be identified.
        DataQualityError: If data fails quality checks (e.g., too many missing days).
        DataFetchingError: For other underlying errors during the process.
    """
    ticker = ticker.upper()
    if ticker not in LIQUID_TICKERS:
        raise TickerNotAllowedError(f"Ticker {ticker} not in allowed liquid list: {LIQUID_TICKERS}")

    # Generate a dynamic filename
    # Using a timestamp for uniqueness if period is not fixed like "5d"
    # For fixed periods, yfinance usually gives consistent start/end, but let's be safe.
    current_dt_str = datetime.now().strftime("%Y%m%d")
    file_name = f"{ticker}_{period}_{interval}_{current_dt_str}.csv"
    output_path = DATA_DIR / file_name

    try:
        # 1) Download data
        print(f"Attempting to download: Ticker={ticker}, Period={period}, Interval={interval}")
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False # Keep OHLC separate
        )
        
        if df.empty:
            raise NoDataFoundError(f"No data returned from yfinance for {ticker} with period={period}, interval={interval}.")
        
        print(f"Downloaded data shape: {df.shape}. Columns: {df.columns}")
        
        # 2) Reset index to ensure 'Datetime' or 'Date' is a column for processing
        df = df.reset_index()
        
        # 3) Standardize DataFrame columns (OHLCV and Date)
        # Identify the datetime column (could be 'Datetime', 'Date', 'index', etc.)
        date_col_name = None
        for col_candidate in df.columns:
            col_str = str(col_candidate).lower()
            if 'date' in col_str or 'time' in col_str or 'index' == col_str : # Common names for datetime column
                date_col_name = col_candidate
                break
        if date_col_name is None:
            raise DataColumnMissingError("Could not identify the primary datetime column in downloaded data.")

        # Standardize to 'Date' and convert to US/Eastern
        # yfinance intraday data is typically already timezone-aware (often UTC or local market time)
        # If it's naive, localize to UTC then convert. If aware, convert directly.
        if df[date_col_name].dt.tz is None:
             df['Date'] = pd.to_datetime(df[date_col_name]).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        else:
             df['Date'] = pd.to_datetime(df[date_col_name]).dt.tz_convert('US/Eastern')

        # Map OHLCV columns robustly, handling potential MultiIndex from yfinance
        # yfinance sometimes returns columns like ('Open', ''), ('High', '')
        def find_ohlcv_col(df_cols, keyword):
            for col in df_cols:
                col_name_parts = [str(part).lower() for part in (col if isinstance(col, tuple) else (col,))]
                if any(keyword in part for part in col_name_parts):
                    # Avoid 'Adj Close' if 'Close' is present
                    if keyword == 'close' and any('adj' in part for part in col_name_parts):
                        continue
                    return col
            return None

        ohlcv_map = {
            'Open': find_ohlcv_col(df.columns, 'open'),
            'High': find_ohlcv_col(df.columns, 'high'),
            'Low': find_ohlcv_col(df.columns, 'low'),
            'Close': find_ohlcv_col(df.columns, 'close'),
            'Volume': find_ohlcv_col(df.columns, 'volume')
        }

        final_df_data = {'Date': df['Date']}
        for target_col, source_col in ohlcv_map.items():
            if source_col is None:
                raise DataColumnMissingError(f"Could not find column for '{target_col}' in downloaded data. Columns: {df.columns}")
            final_df_data[target_col] = df[source_col]
        
        df = pd.DataFrame(final_df_data) # Create a clean DataFrame with standardized column names
        print(f"DataFrame after column standardization. Columns: {df.columns}, Shape: {df.shape}")
        
        # 4) Filter by Regular Trading Hours (if requested)
        if not include_extended_hours:
            df = df[(df["Date"].dt.time >= time(9,30)) & (df["Date"].dt.time <= time(16,0))]
            if df.empty:
                 raise DataQualityError("No data remaining after filtering for regular trading hours.")
            print(f"DataFrame after RTH filter. Shape: {df.shape}")

        # 5) Strip out flat candles & zero volume bars
        is_flat_candle = (df["Open"] == df["High"]) & (df["High"] == df["Low"]) & (df["Low"] == df["Close"])
        has_zero_volume = (df["Volume"] == 0)
        df = df[~is_flat_candle & ~has_zero_volume]
        if df.empty:
            raise DataQualityError("No data remaining after filtering out flat/zero-volume candles.")
        print(f"DataFrame after flat/zero-volume filter. Shape: {df.shape}")

        # 6) Validate data against expected NYSE trading days
        # Ensure data is sorted by date before this step
        df = df.sort_values(by='Date').reset_index(drop=True)

        # Get the date range from the actual fetched data
        actual_start_date = df["Date"].min().date()
        actual_end_date = df["Date"].max().date()

        nyse_calendar = mcal.get_calendar("XNYS") # New York Stock Exchange calendar
        # Get expected trading sessions (datetimes) within the actual data's date range
        expected_sessions_dt = nyse_calendar.valid_days(start_date=actual_start_date.strftime("%Y-%m-%d"),
                                                        end_date=actual_end_date.strftime("%Y-%m-%d"))
        expected_trading_dates = [d.date() for d in expected_sessions_dt] # Convert to just date objects

        # Group actual data by date
        actual_data_by_date = df.groupby(df["Date"].dt.date)
        
        missing_or_incomplete_days = []
        data_quality_log = ["🕵️ Candle count per trading day:"]
        
        for expected_day in expected_trading_dates:
            if expected_day not in actual_data_by_date.groups:
                data_quality_log.append(f"❌ {expected_day}: Entire day MISSING")
                missing_or_incomplete_days.append(expected_day)
            else:
                candles_on_day = len(actual_data_by_date.get_group(expected_day))
                # Basic check for completeness (e.g., for 5m interval, expect 78 bars in a 6.5h session)
                # This threshold might need adjustment based on interval and typical market behavior
                expected_bars_per_day = (60 / int(interval[:-1])) * 6.5 if interval.endswith('m') else 1
                if candles_on_day < expected_bars_per_day * 0.9: # Allow for some minor discrepancies
                    data_quality_log.append(f"⚠️ {expected_day}: {candles_on_day} candles (potentially incomplete)")
                    missing_or_incomplete_days.append(expected_day)
                else:
                    data_quality_log.append(f"✅ {expected_day}: {candles_on_day} candles")

        if len(missing_or_incomplete_days) > max_missing_days:
            error_detail = f"Too many missing or incomplete days: {len(missing_or_incomplete_days)} found (max allowed: {max_missing_days}). "
            error_detail += f"Days: {', '.join(map(str,missing_or_incomplete_days[:5]))}" # Show first 5
            error_detail += "\n" + "\n".join(data_quality_log)
            raise DataQualityError(error_detail)
        
        # 7) Save the cleaned and validated DataFrame to CSV
        # Ensure only standard columns are saved
        df_to_save = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df_to_save.to_csv(output_path, index=False)
        
        success_message = (
            f"✅ Successfully fetched, cleaned, and saved {period} of {interval} data for {ticker} to '{output_path}'.\n"
            + "\n".join(data_quality_log)
        )
        return success_message

    except (TickerNotAllowedError, NoDataFoundError, DataColumnMissingError, DataQualityError) as e:
        # These are custom, already specific errors. Re-raise them.
        # Or, if the function is only expected to return a string:
        # return f"❌ Error: {str(e)}"
        raise # Re-raise for the caller (e.g., Streamlit app) to handle
    except Exception as e:
        # Catch any other unexpected errors (yfinance issues, pandas errors, etc.)
        import traceback
        error_details = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
        # For string return type:
        # return f"❌ Unexpected Error: {str(e)}"
        raise DataFetchingError(error_details) from e