import yfinance as yf
import pandas as pd
from datetime import time
import pandas_market_calendars as mcal

LIQUID_TICKERS = {'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'SPY', 'QQQ'}

def fetch_intraday_period(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",  # Added interval parameter
    max_missing_days: int = 2,
    include_extended_hours: bool = False
) -> str:
    ticker = ticker.upper()
    if ticker not in LIQUID_TICKERS:
        return f"❌ Ticker {ticker} not in allowed liquid list."

    try:
        # 1) Download data
        df = yf.download(
            ticker, 
            period=period,
            interval=interval, 
            progress=False, 
            auto_adjust=False
        )
        
        print(f"Downloaded data shape: {df.shape}")
        print(f"Column structure: {df.columns}")
        
        if df.empty:
            return f"❌ No data from yfinance for {ticker}."
        
        # 2) Reset index to get datetime column first
        df = df.reset_index()
        
        # 3) Create a completely new DataFrame with flattened structure
        new_df = pd.DataFrame()
        
        # Find and add the date column
        date_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'date' in col_str or 'time' in col_str or col_str == 'index':
                date_col = col
                break
                
        if date_col is None:
            return "❌ Could not find date/time column"
            
        # Add date column
        new_df['Date'] = pd.to_datetime(df[date_col], utc=True).dt.tz_convert("US/Eastern")
        
        # Map OHLCV columns
        ohlcv_map = {'Open': None, 'High': None, 'Low': None, 'Close': None, 'Volume': None}
        
        # Find columns by examining names
        for col in df.columns:
            if col == date_col:
                continue
                
            col_str = str(col).lower()
            if isinstance(col, tuple):
                # For MultiIndex, check both levels
                for part in col:
                    part_str = str(part).lower()
                    if 'open' in part_str:
                        ohlcv_map['Open'] = col
                        break
                    elif 'high' in part_str:
                        ohlcv_map['High'] = col
                        break
                    elif 'low' in part_str:
                        ohlcv_map['Low'] = col
                        break
                    elif 'close' in part_str and 'adj' not in part_str:
                        ohlcv_map['Close'] = col
                        break
                    elif 'volume' in part_str:
                        ohlcv_map['Volume'] = col
                        break
            else:
                # For single-level index
                if 'open' in col_str:
                    ohlcv_map['Open'] = col
                elif 'high' in col_str:
                    ohlcv_map['High'] = col
                elif 'low' in col_str:
                    ohlcv_map['Low'] = col
                elif 'close' in col_str and 'adj' not in col_str:
                    ohlcv_map['Close'] = col
                elif 'volume' in col_str:
                    ohlcv_map['Volume'] = col
        
        # Copy data to new DataFrame
        for target, source in ohlcv_map.items():
            if source is not None:
                new_df[target] = df[source]
            else:
                return f"❌ Could not find column for {target}"
        
        # Replace with clean DataFrame
        df = new_df
        print(f"Final clean columns: {df.columns}")
        
        # 4) Filter by Regular Trading Hours
        if not include_extended_hours:
            df = df[(df["Date"].dt.time >= time(9,30)) & (df["Date"].dt.time <= time(16,0))]
        
        # 5) Strip out flat candles & zero volume
        flat = (df["Open"]==df["High"]) & (df["High"]==df["Low"]) & (df["Low"]==df["Close"])
        df = df[~flat & (df["Volume"]>0)]
        if df.empty:
            return "❌ All bars filtered out (flat/zero-volume)."
        
        # 6) Build expected NYSE dates
        start = df["Date"].min().date().strftime("%Y-%m-%d")
        end = df["Date"].max().date().strftime("%Y-%m-%d")
        nyse = mcal.get_calendar("XNYS")
        sessions = nyse.valid_days(start_date=start, end_date=end)
        expected = [d.date() for d in sessions]
        
        # 7) Group & report
        grouped = df.groupby(df["Date"].dt.date)
        missing = []
        log = ["🕵️ Candle count per trading day:"]
        for day in expected:
            if day not in grouped.groups:
                log.append(f"❌ {day}: MISSING")
                missing.append(day)
            else:
                cnt = len(grouped.get_group(day))
                log.append(f"✅ {day}: {cnt}")
                if cnt < (int(60/5)*6.5):  # e.g. 5m bars in 6.5h = 78
                    missing.append(day)
        
        if len(missing) > max_missing_days:
            return "\n".join(log) + f"\n❌ Too many incomplete days: {', '.join(map(str,missing))}"
        
        # 8) Save CSV with unique filename
        df[["Date","Open","High","Low","Close","Volume"]].to_csv("data.csv", index=False)
        return "\n".join(log) + f"\n✅ Saved {period} data for {ticker} to data.csv"
    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}"