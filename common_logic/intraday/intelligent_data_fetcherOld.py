import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import pandas_market_calendars as mcal

# Liquid tickers list (example: can be expanded)
LIQUID_TICKERS = {'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'SPY', 'QQQ'}

def fetch_intraday_period(ticker: str, period: str = "5d", max_missing_days: int = 2) -> str:
    if ticker.upper() not in LIQUID_TICKERS:
        return f"❌ Ticker {ticker} is not in the approved liquid tickers list."

    df = yf.download(ticker, period=period, interval="5m", progress=False)

    print(df.head())  # Debugging line to check the initial data

    if df.empty:
        return f"❌ No data returned from yfinance for {ticker}."

    # Ensure index is datetime with timezone
    if not isinstance(df.index, pd.DatetimeIndex):
        return "❌ Data error: Index is not datetime."

    # Convert to US/Eastern
    df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
    df.index.name = "Date"
    df.reset_index(inplace=True)

    if df['Date'].empty or pd.isna(df['Date'].min()):
        return "❌ Data error: Missing timestamps in Date column."

    # Confirm 'Date' is present and valid
    if "Date" not in df.columns or df["Date"].isna().all():
        return "❌ Data error: Missing or invalid timestamps in Date column."

    # Validate .min() and .max()
    try:
        start_valid = df["Date"].min().strftime("%Y-%m-%d")
        end_valid = df["Date"].max().strftime("%Y-%m-%d")
    except Exception as e:
        return f"❌ Data error during datetime conversion: {str(e)}"

    df = df[(df['Date'].dt.time >= datetime.strptime("09:30", "%H:%M").time()) &
            (df['Date'].dt.time <= datetime.strptime("16:00", "%H:%M").time())]

    df = df[~((df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close']))]
    df = df[df['Volume'] > 0]

    grouped = df.groupby(df['Date'].dt.date)
    nyse = mcal.get_calendar("XNYS")

    sessions = nyse.valid_days(
        start_date=df['Date'].min().date().strftime("%Y-%m-%d"),
        end_date=df['Date'].max().date().strftime("%Y-%m-%d")
    )

    expected_days = [d.date() for d in sessions]

    incomplete_days = []
    debug_log = ["🕵️ Candle count per trading day:"]

    for day in expected_days:
        if day not in grouped.groups:
            incomplete_days.append((day, 0))
            debug_log.append(f"❌ {day}: MISSING")
        else:
            count = len(grouped.get_group(day))
            debug_log.append(f"✅ {day}: {count} candles")
            if count < 75:
                incomplete_days.append((day, count))

    if len(incomplete_days) > max_missing_days:
        debug_lines = "\n".join(debug_log)
        bad_days = ", ".join(str(d[0]) for d in incomplete_days)
        return f"{debug_lines}\n❌ Too many incomplete trading days: {bad_days}"

    df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
    df.to_csv("data.csv", index=False)

    debug_lines = "\n".join(debug_log)
    return f"{debug_lines}\n✅ Saved {period} 5-minute intraday data for {ticker} to data.csv"
