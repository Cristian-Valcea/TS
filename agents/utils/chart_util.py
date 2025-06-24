import os
import matplotlib.pyplot as plt

def plot_strategy_behavior(df, trades, output_path="sandbox/strategy_plot.png"):
    '''
    df: original dataframe used in backtest (must include 'Close')
    trades: list of dicts with keys ['entry_index', 'exit_index', 'entry_price', 'exit_price', 'size']
    output_path: file path to save the plot
    '''
    if df.empty or 'Close' not in df.columns:
        print("Invalid or missing Close column in data.")
        return

    # Ensure the output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df['Close'], label='Close', alpha=0.7)

    for trade in trades:
        entry_i = trade.get('entry_index')
        exit_i = trade.get('exit_index')
        entry_p = trade.get('entry_price')
        exit_p = trade.get('exit_price')
        size = trade.get('size', 1)

        if entry_i is not None and exit_i is not None:
            ax.plot(df.index[entry_i], entry_p, 'g^', label='Entry' if size > 0 else 'Short Entry')
            ax.plot(df.index[exit_i], exit_p, 'rv', label='Exit')

    ax.set_title("Strategy Entry/Exit Points")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
