# common_logic/backtesting_services/backtrader_runner.py
# ... (imports and other parts of the class) ...

class BacktraderRunner:
    def __init__(self, strategy_file_path: str, data_feed_path: str,
                 strategy_params: Optional[Dict[str, Any]] = None,
                 cash: float = 100000.0, commission: float = 0.001,
                 from_date: Optional[str] = None, to_date: Optional[str] = None,
                 analyzers: Optional[List[Tuple[Any, Dict]]] = None): # Allow custom analyzers
        # ... (other init code) ...
        self.custom_analyzers_to_add = analyzers # Store custom analyzers

    def run_backtest(self) -> Dict[str, Any]:
        # ... (initial setup) ...
        try:
            # --- Strategy Loading ---
            # ... (strategy loading logic) ...

            # --- Cerebro Setup ---
            self.cerebro = bt.Cerebro(stdstats=False) # Disable standard anaylzers initially
            self.cerebro.addstrategy(StrategyClass, **self.strategy_params)

            # --- Data Feed ---
            # ... (data feed loading logic) ...

            self.cerebro.broker.setcash(self.cash)
            self.cerebro.broker.setcommission(commission=self.commission)

            # --- Analyzers ---
            # Add default named analyzers
            self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Years)
            self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
            self.cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
            self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
            self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns', timeframe=bt.TimeFrame.Years)
            self.cerebro.addanalyzer(bt.analyzers.LogReturnsRolling, _name='log_returns_rolling', timeframe=bt.TimeFrame.Months)

            # Add any custom analyzers passed during instantiation
            if self.custom_analyzers_to_add:
                for analyzer_cls, analyzer_kwargs in self.custom_analyzers_to_add:
                    self.cerebro.addanalyzer(analyzer_cls, **analyzer_kwargs)
            
            # --- Run Backtest ---
            # ... (run and results extraction logic) ...
            
            # Ensure results extraction uses the names
            # Example:
            # for name, analyzer in self.results[0].analyzersbyname.items():
            #    self.analyzer_results[name] = analyzer.get_analysis()

        # ... (rest of the run_backtest and class) ...
