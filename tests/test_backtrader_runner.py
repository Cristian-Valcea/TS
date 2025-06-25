import unittest
from pathlib import Path
import sys
import os
import backtrader as bt

# Add project root to sys.path to allow importing from common_logic
# Assuming this test file is in 'tests/' and 'common_logic' is at the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common_logic.backtesting_services.backtrader_runner import load_strategy_class_from_file, run_backtest

# Define the path to the directory containing test data (dummy strategy and data files)
TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"

class TestBacktraderRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create dummy files if they don't exist (though create_file_with_block should handle it)"""
        os.makedirs(TEST_DATA_DIR, exist_ok=True)

        dummy_strategy_content = """
import backtrader as bt

class DummySmaCross(bt.Strategy):
    params = (('fast_sma_period', 10), ('slow_sma_period', 30),)

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.fast_sma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.p.fast_sma_period
        )
        self.slow_sma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.p.slow_sma_period
        )
        self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position

class AnotherStrategy(bt.Strategy):
    def __init__(self):
        pass
    def next(self):
        pass
"""
        with open(TEST_DATA_DIR / "dummy_strategy_for_test_loader.py", "w") as f:
            f.write(dummy_strategy_content)

    def test_load_strategy_class_from_file_success(self):
        """Test successfully loading a strategy class."""
        strategy_file = TEST_DATA_DIR / "dummy_strategy_for_test_loader.py"
        StrategyClass = load_strategy_class_from_file(str(strategy_file), "DummySmaCross")
        self.assertTrue(issubclass(StrategyClass, bt.Strategy))
        self.assertEqual(StrategyClass.__name__, "DummySmaCross")

    def test_load_another_strategy_class_from_file(self):
        """Test successfully loading a different strategy class from the same file."""
        strategy_file = TEST_DATA_DIR / "dummy_strategy_for_test_loader.py"
        StrategyClass = load_strategy_class_from_file(str(strategy_file), "AnotherStrategy")
        self.assertTrue(issubclass(StrategyClass, bt.Strategy))
        self.assertEqual(StrategyClass.__name__, "AnotherStrategy")

    def test_load_strategy_class_file_not_found(self):
        """Test loading from a non-existent strategy file."""
        with self.assertRaises(FileNotFoundError):
            load_strategy_class_from_file("non_existent_strategy.py", "AnyStrategy")

    def test_load_strategy_class_name_not_found(self):
        """Test loading a non-existent strategy class name from an existing file."""
        strategy_file = TEST_DATA_DIR / "dummy_strategy_for_test_loader.py"
        with self.assertRaises(AttributeError):
            load_strategy_class_from_file(str(strategy_file), "NonExistentStrategy")

    def test_load_strategy_syntax_error(self):
        """Test loading a strategy file with syntax errors."""
        syntax_error_file = TEST_DATA_DIR / "syntax_error_strategy.py"
        with open(syntax_error_file, "w") as f:
            f.write("import backtrader as bt\nclass SyntaxErrorStrategy(bt.Strategy):\n  def __init__(self):\n    self.var = ") # Intentional syntax error

        with self.assertRaises(ImportError): # Or potentially SyntaxError depending on Python version and how importlib handles it
            load_strategy_class_from_file(str(syntax_error_file), "SyntaxErrorStrategy")
        os.remove(syntax_error_file) # Clean up


    def test_run_backtest_success(self):
        """Test a successful backtest run with dummy data and strategy."""
        strategy_file = TEST_DATA_DIR / "dummy_strategy_for_test_loader.py" # Use the one created in setUpClass
        data_file = TEST_DATA_DIR / "dummy_data_for_test.csv"

        # Ensure dummy data file exists (it was created by a previous tool call)
        if not data_file.exists():
            # Create a minimal version if it's missing, matching backtrader_runner's expected format
            dummy_data_content = """,,,
,,,
,,,
Date,Open,High,Low,Close,Volume
2023-01-02,150.0,152.0,149.0,151.0,100000
2023-01-03,151.2,153.5,150.5,152.5,120000
2023-01-04,152.0,152.5,150.0,150.5,110000
2023-01-05,150.7,151.0,148.0,149.0,130000
2023-01-06,149.5,150.0,147.5,148.0,105000
2023-01-09,148.0,150.0,147.0,149.5,115000
2023-01-10,150.0,155.0,149.0,154.0,140000
2023-01-11,154.0,156.0,153.0,155.0,135000
2023-01-12,155.2,157.0,154.5,156.5,125000
2023-01-13,156.0,158.0,155.0,157.0,145000
"""
            with open(data_file, "w") as f:
                f.write(dummy_data_content)
            print(f"Warning: Dummy data file was missing, created at {data_file}")


        results = run_backtest(
            strategy_file_path=str(strategy_file),
            data_file_path=str(data_file),
            strategy_class_name="DummySmaCross",
            initial_cash=10000.0,
            commission_pct=0.001,
            fast_sma_period=5, # Override default param
            slow_sma_period=10 # Override default param
        )
        self.assertIsNotNone(results)
        self.assertEqual(results.get("status"), "completed")
        self.assertIn("final_portfolio_value", results)
        self.assertIn("sharpe_ratio", results)
        self.assertIn("strategy_params_used", results)
        self.assertEqual(results["strategy_params_used"]["fast_sma_period"], 5)
        self.assertTrue(results["final_portfolio_value"] > 0) # Basic check

    def test_run_backtest_data_file_not_found(self):
        """Test run_backtest with a non-existent data file."""
        strategy_file = TEST_DATA_DIR / "dummy_strategy_for_test_loader.py"
        with self.assertRaises(FileNotFoundError): # As per current run_backtest implementation
            run_backtest(
                strategy_file_path=str(strategy_file),
                data_file_path="non_existent_data.csv",
                strategy_class_name="DummySmaCross",
                initial_cash=10000.0,
                commission_pct=0.001
            )

    def test_run_backtest_strategy_file_not_found(self):
        """Test run_backtest with a non-existent strategy file."""
        data_file = TEST_DATA_DIR / "dummy_data_for_test.csv"
        results = run_backtest( # load_strategy_class_from_file is called inside run_backtest
            strategy_file_path="non_existent_strategy.py",
            data_file_path=str(data_file),
            strategy_class_name="AnyStrategy",
            initial_cash=10000.0,
            commission_pct=0.001
        )
        self.assertEqual(results.get("status"), "error")
        self.assertIn("Failed to load strategy", results.get("error_message", ""))
        self.assertIn("FileNotFoundError", results.get("error_message", ""))


    def test_run_backtest_cerebro_error_handling(self):
        """
        Test if run_backtest handles errors from cerebro.run() gracefully.
        This is a bit tricky to simulate perfectly without a known-to-fail strategy/data.
        We can simulate by providing a strategy that might fail during init with bad params
        or by patching cerebro.run() to raise an exception.
        For now, let's test with a strategy that might have issues if data is too short.
        """
        strategy_file = TEST_DATA_DIR / "dummy_strategy_for_test_loader.py"
        # Create tiny data file that might cause indicator errors
        tiny_data_file = TEST_DATA_DIR / "tiny_data.csv"
        tiny_data_content = """,,,
,,,
,,,
Date,Open,High,Low,Close,Volume
2023-01-02,150.0,152.0,149.0,151.0,100000
"""
        with open(tiny_data_file, "w") as f:
            f.write(tiny_data_content)

        results = run_backtest(
            strategy_file_path=str(strategy_file),
            data_file_path=str(tiny_data_file),
            strategy_class_name="DummySmaCross", # SMA periods might be too long for 1 data point
            initial_cash=10000.0,
            commission_pct=0.001,
            fast_sma_period=10, # Default, likely too long
            slow_sma_period=30  # Default, likely too long
        )
        # Depending on how Backtrader/strategy handles insufficient data for indicators,
        # this might complete with no trades or error out.
        # The current run_backtest returns an error dict if cerebro.run() fails.
        if results.get("status") == "error":
            self.assertIn("Cerebro run failed", results.get("error_message", ""))
        else:
            # If it completes (e.g., strategy handles short data gracefully), check status
            self.assertEqual(results.get("status"), "completed")
            # And likely no trades, or very basic metrics
            self.assertIn("final_portfolio_value", results)

        os.remove(tiny_data_file) # Clean up


if __name__ == '__main__':
    unittest.main()
