import os
import sys
from ib_insync import IB, Stock, util, Order
import asyncio
import time
import pandas as pd
import numpy as np
from typing import Tuple
import traceback
import logging

'''
Example of valid evaluate code:
  if latest['macd'] > latest['macd_signal'] and latest['rsi'] < 30:
      return 'BUY'
  
'''
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from TraderAgent.trading.ib_order_tracker import log_trade_event, register_trade_event_handlers
from TraderAgent.utils.db import upsert_position

class LiveStrategyRunner:
    def __init__(self, symbol: str, evaluate_code: str, quantity: int = 1):
        self.ib = IB()
        self.symbol = symbol
        self.quantity = quantity
        self.contract = Stock(symbol, 'SMART', 'USD')
        self.evaluate_func = self._compile_evaluate_code(evaluate_code)
        self._running = True  # For graceful shutdown if needed
        self.logger = logging.getLogger(f"LiveStrategyRunner.{self.symbol}")
        self.last_decision = None

        self.position = 0
        self.entry_price = None
        self.highest_price = None
        self.stop_order = None
        self.trailing_amount = 1.0  # Example: $1 trailing
        
        # subscribe to the Position Event
        self.ib.positionEvent += self._on_position_update
        # Initialize command queue
        self.command_queue = asyncio.Queue()


    # event handler for position messages
    def _on_position_update(self, account, contract, position, avgCost, **kwargs):
        symbol = getattr(contract, "symbol", None)
        if symbol:
            upsert_position(symbol, position, avgCost)
            self.logger.info(f"[POSITION] Updated DB: {symbol} qty={position} avg_price={avgCost}")

    def _compile_evaluate_code(self, code: str):
        """
        Dynamically compile the `evaluate()` strategy function from a code string.
        The code must define a function named 'evaluate' that takes a dict of indicators.
        """
        local_vars = {}
        try:
            exec(code, {}, local_vars)
            if 'evaluate' not in local_vars:
                raise ValueError("evaluate function not defined")
            return local_vars['evaluate']
        except Exception as e:
            print(f"[ERROR] Failed to compile evaluate function: {e}")
            return lambda _: "HOLD"



    async def run(self, interval_seconds: int = 60, max_retries: int = 3):
        await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)
        self.logger.info(f"[LIVE] Connected to IBKR TWS for symbol: {self.symbol}")

        while self._running:
            start_time = time.time()
            df = await self._fetch_historical_data_with_retries(max_retries, interval_seconds)
            if df is None or df.empty:
                await self._handle_passive_wait(interval_seconds)
                continue

            try:
                await self._process_strategy(df, interval_seconds)
            except Exception as e:
                self.logger.error(f"Exception in live strategy runner: {e}\n{traceback.format_exc()}")

            elapsed = time.time() - start_time
            self.logger.debug(f"Loop iteration took {elapsed:.2f} seconds")
            await asyncio.sleep(interval_seconds)

    def update_strategy(self, new_evaluate_code: str) -> bool:
        """
        Hot-swap the evaluate function with new code.
        Returns True if successful, False otherwise.
        """
        try:
            new_func = self._compile_evaluate_code(new_evaluate_code)
            self.evaluate_func = new_func
            self.logger.info("[STRATEGY] Strategy code updated successfully.")
            return True
        except Exception as e:
            self.logger.error(f"[STRATEGY] Failed to update strategy: {e}")
            return False


    async def command_listener(self):
        while self._running:
            try:
                cmd = await self.command_queue.get()
                if cmd["type"] == "update_strategy":
                    new_code = cmd["evaluate_code"]
                    success = self.update_strategy(new_code)
                    if success:
                        self.logger.info(f"[COMMAND] Strategy updated for {self.symbol}")
                    else:
                        self.logger.error(f"[COMMAND] Failed to update strategy for {self.symbol}")
                elif cmd["type"] == "stop":
                    self.logger.info(f"[COMMAND] Stop received for {self.symbol}")
                    self.stop()
            except Exception as e:
                self.logger.error(f"[COMMAND] Exception in command_listener: {e}")


    async def _fetch_historical_data_with_retries(self, max_retries: int, interval_seconds: int) -> pd.DataFrame:
        retries = 0
        df = None
        while retries < max_retries:
            try:
                self.logger.info(f"Fetching historical data for {self.symbol}")
                bars = await self.ib.reqHistoricalDataAsync(
                    self.contract,
                    endDateTime='',
                    durationStr='1 D',
                    barSizeSetting='5 mins',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
                df = util.df(bars) if bars is not None else None
                if df is not None and not df.empty:
                    return df
                else:
                    self.logger.warning("[LIVE] No historical data retrieved or DataFrame is empty.")
            except Exception as e:
                self.logger.error(f"Exception fetching historical data: {e}")
            retries += 1
            await asyncio.sleep(5)
        self.logger.error(f"[LIVE] Failed to fetch valid data after {max_retries} attempts.")
        return df

    async def _process_strategy(self, df: pd.DataFrame, interval_seconds: int):
        latest = self._compute_indicators(df)
        if latest is None:
            self.logger.warning("[LIVE] No valid indicators computed.")
            await asyncio.sleep(interval_seconds)
            return
        self.logger.debug(
            f"Indicators: close={latest['close']}, sma={latest['sma']}, ema20={latest['ema20']}, "
            f"macd={latest['macd']}, rsi={latest['rsi']}, upper_band={latest['upper_band']}, "
            f"lower_band={latest['lower_band']}, atr={latest['atr']}, stochastic_k={latest['stochastic_k']}"
        )

        decision = self.evaluate_func(latest)
        self.logger.info(f"Strategy decision: {decision} | Latest: {latest}")

        # delegate all trade placement & stops to new helper
        await self._maybe_execute(decision, latest)

        await self._update_trailing_stop(latest)


    async def _maybe_execute(self, decision: str, latest: dict):
        """
        Handles transitions between HOLD/BUY/SELL:
         - on BUY: place a MKT buy + initial stop
         - on SELL: place a MKT sell + cancel any stop
         - on trailing stop: update if price moved up
        """
        # only act on change
        if decision == self.last_decision:
            # but still update trailing stop if we’re long
            await self._update_trailing_stop(latest['close'])
            return

        # BUY logic
        if decision == 'BUY':
            buy_order = Order('BUY', totalQuantity=self.quantity, orderType='MKT')
            trade = self.ib.placeOrder(self.contract, buy_order)
            register_trade_event_handlers(trade)
            log_trade_event("📤 ORDER PLACED", trade)

            self.position    = self.quantity
            self.entry_price = latest['close']
            self.highest_price = self.entry_price

            # initial stop‐loss
            stop_price = self.entry_price - self.trailing_amount
            stop_order = Order('SELL', orderType='STP',
                               totalQuantity=self.quantity,
                               auxPrice=stop_price)
            stop_trade = self.ib.placeOrder(self.contract, stop_order)
            register_trade_event_handlers(stop_trade)
            log_trade_event("📤 STOP ORDER PLACED", stop_trade)

            self.stop_order = stop_order

        # SELL logic
        elif decision == 'SELL' and self.position > 0:
            sell_order = Order('SELL', totalQuantity=self.position, orderType='MKT')
            trade = self.ib.placeOrder(self.contract, sell_order)
            register_trade_event_handlers(trade)
            log_trade_event("📤 ORDER PLACED", trade)

            # reset all
            self.position = 0
            self.entry_price = None
            if self.stop_order:
                self.ib.cancelOrder(self.stop_order)
                self.stop_order = None

        # HOLD or any other → do nothing extra

        # update last_decision
        self.last_decision = decision

    async def _update_trailing_stop(self, current_price: float):
        """
        If we are long and price has moved up, bump up the stop‐loss.
        """
        if self.position > 0 and current_price > self.highest_price:
            self.highest_price = current_price
            new_stop = self.highest_price - self.trailing_amount

            # only if we already have a stop and new_stop is higher
            if self.stop_order and new_stop > self.stop_order.auxPrice:
                self.ib.cancelOrder(self.stop_order)

                stop_order = Order('SELL', orderType='STP',
                                   totalQuantity=self.quantity,
                                   auxPrice=new_stop)
                trade = self.ib.placeOrder(self.contract, stop_order)
                register_trade_event_handlers(trade)
                log_trade_event("📤 TRAILING STOP UPDATED", trade)

                self.stop_order = stop_order

    async def _handle_trade_decision(self, decision: str, latest: dict):
        if decision == 'BUY':
            order = Order(action='BUY', totalQuantity=self.quantity, orderType='MKT')
            trade = self.ib.placeOrder(self.contract, order)
            register_trade_event_handlers(trade)
            log_trade_event("📤 ORDER PLACED", trade)
            self.logger.info(f"Trade submitted: {trade}")
            self.position = self.quantity
            self.entry_price = latest['close']
            self.highest_price = latest['close']
            self.logger.info(f"Order placed: BUY {self.quantity} {self.symbol} at {self.entry_price}")

            # Place initial stop-loss
            stop_price = self.entry_price - self.trailing_amount
            self.stop_order = Order(
                action='SELL',
                orderType='STP',
                totalQuantity=self.quantity,
                auxPrice=stop_price
            )
            stop_trade = self.ib.placeOrder(self.contract, self.stop_order)
            register_trade_event_handlers(stop_trade)
            log_trade_event("📤 STOP ORDER PLACED", stop_trade)
            self.logger.info(f"Initial stop-loss placed at {stop_price}")

        elif decision == 'SELL':
            order = Order(action='SELL', totalQuantity=self.quantity, orderType='MKT')
            trade = self.ib.placeOrder(self.contract, order)
            register_trade_event_handlers(trade)
            log_trade_event("📤 ORDER PLACED", trade)
            self.logger.info(f"Order placed: SELL {self.position} {self.symbol} at {latest['close']}")
            self.position = 0
            self.entry_price = None
            self.highest_price = None
            if self.stop_order:
                self.ib.cancelOrder(self.stop_order)
                self.logger.info("Stop order cancelled after SELL")
                self.stop_order = None
        else:
            self.logger.info("No trade signal or already in position.")
    '''
    async def _update_trailing_stop(self, latest: dict):
        if self.position > 0 and self.highest_price is not None:
            if latest['close'] > self.highest_price:
                self.highest_price = latest['close']
                new_stop = self.highest_price - self.trailing_amount
                if self.stop_order and new_stop > self.stop_order.auxPrice:
                    self.ib.cancelOrder(self.stop_order)
                    self.stop_order = Order(
                        action='SELL',
                        orderType='STP',
                        totalQuantity=self.quantity,
                        auxPrice=new_stop
                    )
                    stop_trade = self.ib.placeOrder(self.contract, self.stop_order)
                    register_trade_event_handlers(stop_trade)
                    log_trade_event("📤 TRAILING STOP UPDATED", stop_trade)
                    self.logger.info(f"Trailing stop updated to {new_stop}")
    '''
    async def _handle_passive_wait(self, interval_seconds: int):
        self.logger.error("[LIVE] Entering passive wait state. Awaiting further instructions or next retry.")
        await asyncio.sleep(interval_seconds)


    '''    
    async def run(self, interval_seconds: int = 60, max_retries: int = 3):
        """
        Main loop for fetching market data and executing strategy.
        """
        await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)
        self.logger.info(f"[LIVE] Connected to IBKR TWS for symbol: {self.symbol}")

        while self._running:
            start_time = time.time()
            retries = 0
            df = None
            while retries < max_retries:
                try:
                    self.logger.info(f"Fetching historical data for {self.symbol}")
                    bars = await self.ib.reqHistoricalDataAsync(
                        self.contract,
                        endDateTime='',
                        durationStr='1 D',
                        barSizeSetting='5 mins',
                        whatToShow='TRADES',
                        useRTH=True,
                        formatDate=1
                    )

                    df = util.df(bars) if bars is not None else None
                    if df is not None and not df.empty:
                        break  # Success
                    else:
                        self.logger.warning("[LIVE] No historical data retrieved or DataFrame is empty.")
                except Exception as e:
                    self.logger.error(f"Exception fetching historical data: {e}")
                retries += 1
                await asyncio.sleep(5)  # Wait before retry

            if df is None or df.empty:
                self.logger.error(f"[LIVE] Failed to fetch valid data after {max_retries} attempts. Entering passive wait.")
                await asyncio.sleep(interval_seconds)
                continue


            try:
                    latest = self._compute_indicators(df)
                    if latest is None:
                        self.logger.warning("[LIVE] No valid indicators computed.")
                        await asyncio.sleep(interval_seconds)
                        continue
                    self.logger.debug(f"Indicators: close={latest['close']}, sma={latest['sma']}, ema20={latest['ema20']}, macd={latest['macd']}, rsi={latest['rsi']}, upper_band={latest['upper_band']}, lower_band={latest['lower_band']}, atr={latest['atr']}, stochastic_k={latest['stochastic_k']}")

                    decision = self.evaluate_func(latest)
                    self.logger.info(f"Strategy decision: {decision} | Latest: {latest}")
                    
                    # Entry logic
                    if decision != self.last_decision:
                        if decision == 'BUY':
                            order = Order(action='BUY', totalQuantity=self.quantity, orderType='MKT')
                            trade = self.ib.placeOrder(self.contract, order)
                            register_trade_event_handlers(trade)
                            log_trade_event("📤 ORDER PLACED", trade)
                            self.logger.info(f"Trade submitted: {trade}")
                            self.position = self.quantity
                            self.entry_price = latest['close']
                            self.highest_price = latest['close']
                            self.logger.info(f"Order placed: BUY {self.quantity} {self.symbol} at {self.entry_price}")
            
                            # Place initial stop-loss
                            stop_price = self.entry_price - self.trailing_amount
                            self.stop_order = Order(
                                action='SELL',
                                orderType='STP',
                                totalQuantity=self.quantity,
                                auxPrice=stop_price
                            )
                            stop_trade = self.ib.placeOrder(self.contract, self.stop_order)
                            register_trade_event_handlers(stop_trade)
                            log_trade_event("📤 STOP ORDER PLACED", stop_trade)
                            self.logger.info(f"Initial stop-loss placed at {stop_price}")

                        elif decision == 'SELL':
                            order = Order(action='SELL', totalQuantity=self.quantity, orderType='MKT')
                            trade = self.ib.placeOrder(self.contract, order)
                            register_trade_event_handlers(trade)
                            log_trade_event("📤 ORDER PLACED", trade)
                            self.logger.info(f"Order placed: SELL {self.position} {self.symbol} at {latest['close']}")
                            self.position = 0
                            self.entry_price = None
                            self.highest_price = None
                            if self.stop_order:
                                self.ib.cancelOrder(self.stop_order)
                                self.logger.info("Stop order cancelled after SELL")
                                self.stop_order = None
                        else:
                            self.logger.info("No trade signal or already in position.")
                        self.last_decision = decision

                    # Trailing stop-loss update logic
                    if self.position > 0 and self.highest_price is not None:
                        if latest['close'] > self.highest_price:
                            self.highest_price = latest['close']
                            new_stop = self.highest_price - self.trailing_amount
                            if self.stop_order and new_stop > self.stop_order.auxPrice:
                                # Cancel old stop and place new one
                                self.ib.cancelOrder(self.stop_order)
                                self.stop_order = Order(
                                    action='SELL',
                                    orderType='STP',
                                    totalQuantity=self.quantity,
                                    auxPrice=new_stop
                                )
                                stop_trade = self.ib.placeOrder(self.contract, self.stop_order)
                                register_trade_event_handlers(stop_trade)
                                log_trade_event("📤 TRAILING STOP UPDATED", stop_trade)
                                self.logger.info(f"Trailing stop updated to {new_stop}")

            except Exception as e:
                self.logger.error(f"Exception in live strategy runner: {e}\n{traceback.format_exc()}")

            elapsed = time.time() - start_time
            self.logger.debug(f"Loop iteration took {elapsed:.2f} seconds")
            await asyncio.sleep(interval_seconds)

    '''
    async def fetch_historical_dataframe(
        self,
        contract,
        duration: str = '2 D',
        bar_size: str = '5 mins',
        what_to_show: str = 'TRADES',
        use_rth: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data and return it as a pandas DataFrame.

        Args:
            contract: IB contract object (e.g., Stock).
            duration: Duration string (e.g., '1 D', '1 W').
            bar_size: Bar size (e.g., '5 mins', '1 hour').
            what_to_show: Data type ('TRADES', 'MIDPOINT', etc.).
            use_rth: Use regular trading hours only.

        Returns:
            pd.DataFrame with datetime index and OHLCV columns.
        """
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False
        )

        if not bars:
            return pd.DataFrame()

        df = pd.DataFrame([bar.__dict__ for bar in bars])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Keep only relevant columns
        return df[['open', 'high', 'low', 'close', 'volume']]


    async def fetch_historical_dataframe_(
        self,
        contract,
        duration: str = '2 D',
        bar_size: str = '5 mins',
        what_to_show: str = 'TRADES',
        use_rth: bool = True,
        return_array: bool = False,
        normalize: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fetch historical data and return as DataFrame and optionally feature array.

        Args:
            contract: IB contract (e.g., Stock).
            duration: Duration string for historical data.
            bar_size: Interval per bar (e.g., '5 mins').
            what_to_show: Type of data (TRADES, MIDPOINT, etc).
            use_rth: Use Regular Trading Hours only.
            return_array: Whether to return a NumPy array of features.
            normalize: Whether to normalize the features.

        Returns:
            (DataFrame, FeatureArray) if return_array is True
            else (DataFrame, None)
        """
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False
        )

        if not bars:
            return pd.DataFrame(), np.array([])

        df = pd.DataFrame([bar.__dict__ for bar in bars])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Compute optional features (e.g., returns, RSI, etc.) here if needed

        if return_array:
            feature_data = df.values.astype(np.float32)

            if normalize:
                # Basic min-max normalization
                min_vals = feature_data.min(axis=0)
                max_vals = feature_data.max(axis=0)
                feature_data = (feature_data - min_vals) / (max_vals - min_vals + 1e-8)

            return df, feature_data

        return df, None

    '''
    example usage
    df = await self.fetch_historical_dataframe(
    contract=self.contract,
    duration='5 D',
    bar_size='15 mins'
    )

    print(df.tail())

    '''



    def stop(self):
        """
        Gracefully stop the runner loop.
        """
        self._running = False
        try:
            self.ib.positionEvent -= self._on_position_update
        except Exception:
            pass

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        macd, macd_signal = self._calculate_macd(df['close'])
        upper_band, lower_band = self._calculate_bollinger_bands(df['close'])
        rsi = self._calculate_rsi(df['close']).iloc[-1]
        sma = df['close'].rolling(window=50).mean().iloc[-1]
        ema20 = df['close'].ewm(span=20).mean().iloc[-1]
        atr = self._calculate_average_true_range(df)
        stochastic_k = self._calculate_stochastic_oscillator(df).iloc[-1]
    
        return {
            'close': df['close'].iloc[-1],
            'sma': sma,
            'ema20': ema20,
            'macd': macd.iloc[-1],
            'macd_signal': macd_signal.iloc[-1],
            'rsi': rsi,
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1],
            'atr': atr,
            'stochastic_k': stochastic_k
        }

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate RSI from a pandas Series of prices.
        """
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    #'macd': (df.close.ewm(span=12, adjust=False).mean() - df.close.ewm(span=26, adjust=False).mean()).iloc[-1],
    def _calculate_macd(self, series: pd.Series) -> tuple:
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20) -> tuple:
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        return upper_band, lower_band

    def _calculate_average_true_range(self, df: pd.DataFrame, period: int = 14) -> float:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        return atr

    def _calculate_stochastic_oscillator(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        low14 = df['low'].rolling(window=period).min()
        high14 = df['high'].rolling(window=period).max()
        k = 100 * (df['close'] - low14) / (high14 - low14)
        return k




'''
import pandas as pd

def backtest_strategy(
    df: pd.DataFrame,
    evaluate_code: str,
    quantity: int = 1,
    initial_cash: float = 100000,
    fee_per_trade: float = 0.0
):
    """
    Backtest a strategy using the same evaluate() code as LiveStrategyRunner.
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    evaluate_code: string defining an evaluate(latest) function
    """
    # Compile the evaluate function
    local_vars = {}
    exec(evaluate_code, {}, local_vars)
    evaluate = local_vars['evaluate']

    cash = initial_cash
    position = 0
    entry_price = 0
    trades = []
    equity_curve = []

    # Pre-calculate indicators
    df['sma'] = df['close'].rolling(window=50).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['rsi'] = calculate_rsi(df['close'])
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['upper_band'] = sma20 + 2 * std20
    df['lower_band'] = sma20 - 2 * std20

    for i in range(50, len(df)):
        latest = {
            'close': df['close'].iloc[i],
            'sma': df['sma'].iloc[i],
            'ema20': df['ema20'].iloc[i],
            'macd': df['macd'].iloc[i],
            'macd_signal': df['macd_signal'].iloc[i],
            'rsi': df['rsi'].iloc[i],
            'upper_band': df['upper_band'].iloc[i],
            'lower_band': df['lower_band'].iloc[i],
            # Add more indicators as needed
        }
        decision = evaluate(latest)

        # Simulate order execution at close price
        if decision == 'BUY' and position == 0:
            position = quantity
            entry_price = df['close'].iloc[i]
            cash -= entry_price * quantity + fee_per_trade
            trades.append({'type': 'BUY', 'price': entry_price, 'index': i})
        elif decision == 'SELL' and position > 0:
            exit_price = df['close'].iloc[i]
            cash += exit_price * position - fee_per_trade
            trades.append({'type': 'SELL', 'price': exit_price, 'index': i})
            position = 0
            entry_price = 0

        # Track equity
        market_value = position * df['close'].iloc[i]
        equity = cash + market_value
        equity_curve.append(equity)

    # Final results
    results = {
        'final_equity': equity_curve[-1] if equity_curve else initial_cash,
        'trades': trades,
        'equity_curve': equity_curve,
        'return_pct': ((equity_curve[-1] / initial_cash) - 1) * 100 if equity_curve else 0
    }
    return results

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Example usage:
if __name__ == "__main__":
    # Load your historical data into a DataFrame
    df = pd.read_csv('your_data.csv')  # Must have columns: open, high, low, close, volume

    # Example evaluate code as a string
    evaluate_code = """
def evaluate(latest):
    if latest['macd'] > latest['macd_signal'] and latest['rsi'] < 30:
        return 'BUY'
    elif latest['macd'] < latest['macd_signal'] and latest['rsi'] > 70:
        return 'SELL'
    else:
        return 'HOLD'
"""

    results = backtest_strategy(df, evaluate_code)
    print(f"Final equity: {results['final_equity']:.2f}")
    print(f"Total trades: {len(results['trades'])}")
    print(f"Return: {results['return_pct']:.2f}%")





def evaluate(data):
    if data['close'] > data['sma'] and data['rsi'] < 70:
        return "BUY"
    elif data['rsi'] > 70:
        return "SELL"
    else:
        return "HOLD"

'''