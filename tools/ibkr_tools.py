import pandas as pd
import sys
from pathlib import Path

# Ensure the project root is in sys.path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from tools.GetIBKRData import get_ibkr_data_sync, ib

def fetch_5min_bars(
    symbol: str,
    start: str,
    end: str,
    use_rth: bool = False
) -> pd.DataFrame:
    """
    IBKR-backed implementation of fetch_5min_bars.
    Returns a timezone-aware DataFrame with ['Date','Open','High','Low','Close','Volume'].
    """
    # bar_size must match your chunking logic: '5 mins'
    df, _bt_feed = get_ibkr_data_sync(
        ib_instance=ib,
        ticker_symbol=symbol,
        start_date=start,
        end_date=end,
        bar_size='5 mins',      # exactly this string
        what_to_show='TRADES',  # matches your IBKR requests
        use_rth=use_rth
    )
    # If your system expects the column 'Date' instead of index name:
    df = df.reset_index().rename(columns={'datetime': 'Date'}).set_index('Date')
    return df

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Test IBKR fetch_5min_bars"
    )
    parser.add_argument(
        "--symbol", "-s", default="AAPL",
        help="Ticker symbol to fetch"
    )
    parser.add_argument(
        "--start", "-b", default="2024-01-15",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", "-e", default="2024-01-18",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--use-rth", action="store_true",
        help="Only regular trading hours"
    )
    args = parser.parse_args()

    try:
        df = fetch_5min_bars(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            use_rth=True #args.use_rth,
        )
    except Exception as err:
        print(f"Error fetching bars: {err}", file=sys.stderr)
        sys.exit(1)

    # show a quick summary
    print(f"\nFetched {len(df)} rows for {args.symbol} from {args.start} to {args.end}\n")
    print(df.head())
    print("\nData types / info:")
    print(df.info())