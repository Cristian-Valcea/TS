if __name__ == '__main__':
    # 1. Create cerebro instance
    cerebro = bt.Cerebro()
    # 2. Download Data
    data_df = yf.download('NVDA', start='2022-01-01', end='2023-12-31')
    # 3. Add data to cerebro
    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)
    # 4. Add strategy
    cerebro.addstrategy(SmaCrossover)
    # 5. Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    # 6. Run backtest
    results = cerebro.run()
    # 7. Print final results as JSON
    pnl = results[0].broker.getvalue() - results[0].broker.startingcash
    sharpe = results[0].analyzers.sharpe.get_analysis()['sharperatio']
    max_drawdown = results[0].analyzers.drawdown.get_analysis()['max']['drawdown']
    
    output = {{
        "final_value": results[0].broker.getvalue(),
        "pnl": pnl,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown
    }}
    print(json.dumps(output, indent=2))```

