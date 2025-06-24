import backtrader as bt

def parse_timeframe_string(tf_string: str):
    """Convert string like '5m' or '1h' to (bt.TimeFrame, compression) tuple."""
    tf_string = tf_string.strip().lower()
    if tf_string.endswith('m'):
        return bt.TimeFrame.Minutes, int(tf_string[:-1])
    elif tf_string.endswith('h'):
        return bt.TimeFrame.Minutes, int(tf_string[:-1]) * 60
    elif tf_string.endswith('d'):
        return bt.TimeFrame.Days, int(tf_string[:-1])
    else:
        raise ValueError(f"Unsupported timeframe format: {tf_string}")
