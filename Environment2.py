import gym
from gym import spaces
import numpy as np
import pandas as pd


class SuperiorTradingEnv2(gym.Env):
    """
    SB3-compatible superior trading environment with internal normalization.
    Observations are normalized, but info contains raw unnormalized financial values.
    Compatible with Gymnasium API (reset -> (obs, info), step -> (obs, reward, terminated, truncated, info)).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=10, initial_balance=10000,
                 transaction_cost=0.001, slippage=0.001, sharpe_window=30):
        super().__init__()

        # Raw df (for info / metrics)
        self.raw_df = df.reset_index(drop=True)

        # Normalized df (for agent observations)
        self.df = self._normalize_df(self.raw_df)

        self.assets = [col.split('_')[0] for col in df.columns if col.endswith('Close')]
        self.n_assets = len(self.assets)

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.sharpe_window = sharpe_window

        # States
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_assets, dtype=np.float32)
        self.total_asset = self.initial_balance
        self.max_steps = len(df) - window_size
        self.returns = []

        # Gym spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.df.shape[1]),
            dtype=np.float32
        )

    # --------------------- Normalization ---------------------
    def _normalize_df(self, df):
        df_norm = df.copy()
        ohlc_cols = [col for col in df.columns if col.split('_')[-1] in ["Open", "High", "Low", "Close"]]
        vol_cols = [col for col in df.columns if col.split('_')[-1] == "Volume"]

        # OHLC -> percent change
        df_norm[ohlc_cols] = df[ohlc_cols].pct_change().fillna(0)

        # Volume -> log + z-score
        for col in vol_cols:
            df_norm[col] = np.log1p(df[col])
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / (df_norm[col].std() + 1e-6)

        return df_norm

    # --------------------- Observation ---------------------
    def _get_observation(self):
        obs = self.df.iloc[self.current_step:self.current_step + self.window_size].values
        return obs.astype(np.float32)

    # --------------------- Portfolio Value ---------------------
    def _calculate_total_asset(self):
        prices = np.array([
            self.raw_df[f"{asset}_Close"].iloc[self.current_step + self.window_size - 1]
            for asset in self.assets
        ])
        return self.balance + np.sum(self.shares_held * prices)

    # --------------------- Step ---------------------
    def step(self, action):
        prev_total_asset = self.total_asset

        # Normalize action to sum to 1 (portfolio allocation)
        action = np.clip(action, 0, 1)
        action = action / (np.sum(action) + 1e-8)

        prices = np.array([
            self.raw_df[f"{asset}_Close"].iloc[self.current_step + self.window_size - 1]
            for asset in self.assets
        ])

        desired_total_value = self.total_asset * action
        current_total_value = self.shares_held * prices
        delta_value = desired_total_value - current_total_value

        cost = 0
        for i in range(self.n_assets):
            if delta_value[i] > 0:  # Buy
                buy_amount = delta_value[i] * (1 + self.transaction_cost + self.slippage)
                if self.balance >= buy_amount:
                    shares_to_buy = buy_amount / prices[i]
                    self.shares_held[i] += shares_to_buy
                    self.balance -= buy_amount
                    cost += delta_value[i] * (self.transaction_cost + self.slippage)
            else:  # Sell
                shares_to_sell = -delta_value[i] / prices[i]
                if self.shares_held[i] >= shares_to_sell:
                    sell_amount = -delta_value[i] * (1 - self.transaction_cost - self.slippage)
                    self.shares_held[i] -= shares_to_sell
                    self.balance += sell_amount
                    cost += -delta_value[i] * (self.transaction_cost + self.slippage)

        self.total_asset = self._calculate_total_asset()

        # Reward: step return with rolling Sharpe
        step_return = (self.total_asset - prev_total_asset - cost) / (prev_total_asset + 1e-6)
        self.returns.append(step_return)

        if len(self.returns) > self.sharpe_window:
            mean_ret = np.mean(self.returns[-self.sharpe_window:])
            std_ret = np.std(self.returns[-self.sharpe_window:]) + 1e-6
            sharpe_ratio = mean_ret / std_ret
            reward = step_return * sharpe_ratio
        else:
            reward = step_return

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # could add max time or early stopping condition

        info = {
            "total_asset": self.total_asset,
            "balance": self.balance,
            "shares_held": self.shares_held.copy(),
            "step_return": step_return,
            "total_profit": self.total_asset - self.initial_balance,
        }

        return self._get_observation(), reward, terminated, truncated, info

    # --------------------- Reset ---------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.n_assets, dtype=np.float32)
        self.total_asset = self.initial_balance
        self.returns = []

        obs = self._get_observation()
        info = {}
        return obs, info

    # --------------------- Render ---------------------
    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Total Asset: {self.total_asset:.2f}, Balance: {self.balance:.2f}, Shares Held: {self.shares_held}")

