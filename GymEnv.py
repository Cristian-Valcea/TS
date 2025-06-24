# GymEnv.py


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

class IntradayTradingEnv(gym.Env):
    """
    Gym‐compatible env wrapping:
      - a historical intraday bar feed (as a pd.DataFrame of numeric features)
      - per-step P&L bookkeeping and reward calc
    """
    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int,
        initial_cash: float = 100_000.0,
        commission: float = 0.0005,
        use_real_ibkr: bool = False,
    ):
        super().__init__()

        # Validate DataFrame
        required_cols = {"Close"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        # -- store the DataFrame and its column names --
        #    data must be a DataFrame of shape (n_steps, n_features)
        #    including a column named "Close"
        self.df = data.reset_index(drop=True).copy()
        self.feature_cols = list(self.df.columns)
        self.data = self.df.values.astype(np.float32)
        self.n_features = self.data.shape[1]

        self.window_size = window_size
        self.initial_cash = initial_cash
        self.commission = commission
        self.use_real_ibkr = use_real_ibkr

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, self.data.shape[1]),
            dtype=np.float32,
        )
        # 0 = flat, 1 = long, 2 = short
        self.action_space = spaces.Discrete(3)

        # now initialize/reset all dynamic state
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0  # +1 for long, –1 for short, 0 flat
        self.equity = self.initial_cash
        self.prev_equity = self.initial_cash
        obs = self._get_obs()
        return obs, {}  # Gymnasium API

    def step(self, action: int):
        try:
            # pull the current bar’s Close price
            close_idx = self.feature_cols.index("Close")
            price = float(self.data[self.current_step, close_idx])

            # execute trade & update cash/position/equity
            self._execute_action(action, price)

            # compute reward as P&L change since last step
            reward = self.equity - self.prev_equity
            self.prev_equity = self.equity

            # advance to next bar
            self.current_step += 1

            obs = self._get_obs()
            terminated = self.current_step >= len(self.data)
            truncated = False
            info = {"equity": self.equity}

            return obs, reward, terminated, truncated, info
        except Exception as e:
            logging.error(f"Error in step: {e}")
            raise

    def _execute_action(self, action: int, price: float):
        # 0=flat: close any position
        if action == 0 and self.position != 0:
            self.cash += self.position * price * (1 - self.commission)
            self.position = 0
        # 1=long
        elif action == 1 and self.position <= 0:
            if self.position < 0:
                # close short first
                self.cash += -1 * price * (1 - self.commission)
            self.position = 1
            self.cash -= price * (1 + self.commission)
        # 2=short
        elif action == 2 and self.position >= 0:
            if self.position > 0:
                # close long first
                self.cash += price * (1 - self.commission)
            self.position = -1
            self.cash += price * (1 - self.commission)

        # update total equity
        self.equity = self.cash + self.position * price

    def _get_obs(self) -> np.ndarray:
        start = self.current_step - self.window_size
        obs = self.data[start : self.current_step]
        if np.isnan(obs).any():
            raise ValueError(f"NaN in obs at step {self.current_step}")
        return obs




    def render(self, mode="human"):
        # optional: plot equity curve or the last few bars / actions
        pass

    def _get_observation(self) -> np.ndarray:
        """
        Gathers the last `window_size` bars,
        computes any indicators, and stacks them into an array.
        """
        window = self.df.iloc[self.step_idx - self.window_size : self.step_idx]
        closes = window["Close"].to_numpy()

        # example indicators (you’d compute real EMA/RSI here)
        ema = pd.Series(closes).ewm(span=10).mean().to_numpy()
        delta = np.diff(closes, prepend=closes[0])
        gain = np.maximum(delta, 0)
        loss = -np.minimum(delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = (100 - (100 / (1 + rs))).to_numpy()

        # stack into shape = (window_size, num_features)
        feat = np.vstack([closes, ema, rsi]).T.astype(np.float32)
        return feat


    def render(self, mode="human"):
        """
        (Optional) visualize P&L curve or the last few bars/actions.
        """
        pass

    def _get_obs(self):
        start = self.current_step - self.window_size
        obs = self.data[start:self.current_step]
        if np.isnan(obs).any():
            raise ValueError(f"NaN in obs at step {self.current_step}")
        return obs