
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import pandas_market_calendars as mcal

LIQUID_TICKERS = {'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'SPY', 'QQQ'}



def fetch_intraday_period(
    ticker: str,
    period: str = "5d",
    interval: str = "5m",
    max_missing_days: int = 2,
    include_extended_hours: bool = False
) -> str:
    """
    Fetches intraday OHLCV data for a given ticker and period, filters for trading hours,
    validates data completeness per trading day, and saves to 'data.csv'.

    Returns a multi-line report including per-day candle counts and a summary.
    """
    # Validate ticker
    if ticker.upper() not in LIQUID_TICKERS:
        return f"❌ Ticker {ticker} is not supported."

    # Download data
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return f"❌ No data returned for {ticker}."

        # 1) If yfinance returned a MultiIndex (e.g. for multiple tickers),
    #    collapse to the second level so you get plain “Open, High, Low, Close, Volume”.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)

    # 2) Make sure the index is a datetime and then pull it into a “Date” column
    try:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
    except Exception as e:
        return f"❌ Data error: Failed to convert datetime index: {e}"

    # Now create a flat “Date” column
    df = df.reset_index().rename(columns={'index': 'Date'})

    # 3) Guard against totally missing timestamps
    if df['Date'].isna().all():
        return "❌ Data error: Unable to determine valid date range"

    # Convert index to US/Eastern timezone
    try:
        df.index = pd.to_datetime(df.index, utc=True).tz_convert('US/Eastern')
    except Exception as e:
        return f"❌ Failed to parse/index timezone: {e}"

    # Reset index
    df.index.name = 'Date'
    df.reset_index(inplace=True)

    # Ensure valid Date
    if df['Date'].isna().all():
        return "❌ Invalid timestamps in data."

    # Filter trading hours
    if not include_extended_hours:
        df = df[(df['Date'].dt.time >= time(9,30)) & (df['Date'].dt.time <= time(16,0))]

    # Remove flat candles and zero volume
    flat = (df['Open']==df['High'])&(df['High']==df['Low'])&(df['Low']==df['Close'])
    df = df.loc[~flat & (df['Volume']>0)]
    if df.empty:
        return "❌ All rows filtered out (flat/volume/hours)."

    # Determine valid trading days via calendar
    try:
        nyse = mcal.get_calendar('XNYS')
        start_date = df['Date'].min().strftime('%Y-%m-%d')
        end_date   = df['Date'].max().strftime('%Y-%m-%d')
        sessions = nyse.valid_days(start_date=start_date, end_date=end_date)
        expected_days = [d.date() for d in sessions]
    except Exception as e:
        return f"❌ Calendar error: {e}"

    # Group by date and report counts
    grouped = df.groupby(df['Date'].dt.date)
    report = ['🕵️ Candle count per trading day:']
    incomplete = []
    total_candles = len(df)

    for day in expected_days:
        cnt = int(grouped.get_group(day).shape[0]) if day in grouped.groups else 0
        marker = '✅' if cnt>0 else '❌'
        report.append(f"{marker} {day}: {cnt} candles")
        if cnt < int( (pd.Timedelta(interval).seconds/60) * ((16-9.5)*60) ):
            incomplete.append(day)

    summary = f"Total days: {len(grouped)} | Total candles: {total_candles}"
    report.append(summary)

    # Check missing tolerance
    if len(incomplete) > max_missing_days:
        days = ', '.join(map(str, incomplete))
        return '\n'.join(report) + f"\n❌ Too many incomplete days: {days}"

    # Save CSV
    df.to_csv('data.csv', index=False, columns=['Date','Open','High','Low','Close','Volume'])
    report.append(f"✅ Saved {period} {interval} data for {ticker} to data.csv")

    return '\n'.join(report)

