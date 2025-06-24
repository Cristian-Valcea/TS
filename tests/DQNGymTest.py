# install first (in your venv or notebook):
# pip install stable-baselines3[extra] gym pandas numpy

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import torch


from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from tools.GetIBKRData import _load_from_cache

# --- your env implementation ---
from GymEnv import IntradayTradingEnv  


def load_data() -> pd.DataFrame:
    # load a historical 1-min bar CSV into a DataFrame
    
    #df = pd.read_csv("AAPL_1min_bars.csv", parse_dates=["Date"], index_col="Date")

    #df = pd.read_pickle("cache_ibkr/AAPL_2024-01-15_2024-01-18_5_mins_TRADES.pkl")
    df = _load_from_cache("cache_ibkr/AAPL_2024-01-15_2024-01-18_5_mins_TRADES.pkl")
    # ensure it has a "Close" column, etc.
    return df


def main():
    # 1. Prepare data & create env
    df = load_data()
    print(f"Loaded {len(df)} rows of 5‐min bars with columns: {df.columns.tolist()}")
    print(df.head())

    # 2. Compute indicators & drop any NaNs
    import ta
    df["rsi14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["ema10"] = ta.trend.EMAIndicator(df["Close"], window=10).ema_indicator()
    df.dropna(inplace=True)  # now every row has valid rsi14 & ema10

    # 3. Bring your datetime index into a column
    #    (so you can extract hour/minute)
    df = df.reset_index()     # moves the index into df["datetime"]
    
    # 4. Extract time-of-day features
    df["hour"]   = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute

    # 5. Drop the datetime column if you no longer need it
    df = df.drop(columns="datetime")

    # 6. Keep only the numeric features your env will consume
    feature_cols = [
        "hour", "minute",
        "Open", "High", "Low", "Close", "Volume",
        "rsi14", "ema10",
    ]
    
    df = df[feature_cols]
    print("Features ready:", df.columns.tolist())
    print(df.head())

    # 7. Create the gym env
    env = IntradayTradingEnv(
        data=df,        # or data=df if your env accepts a DataFrame
        window_size=100,
        initial_cash=100_000.0,
        commission=0.0005,
        use_real_ibkr=False,
    )
    # quick smoke-test
    obs, _ = env.reset()
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, done={done}, equity={info['equity']:.2f}")
        if done:
            obs, _ = env.reset()



    check_env(env, warn=True)


    # 8 Wrap for vectorization & monitoring
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    vec_env = VecMonitor(DummyVecEnv([lambda: env]))


    # 9. Create the RL model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=10_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[256,256], activation_fn=torch.nn.ReLU),
        target_update_interval=500,
        train_freq=4,
        tensorboard_log="./dqn_tensorboard/",
        verbose=1,
    )

    # 10) Optionally set up evaluation
    eval_env = VecMonitor(DummyVecEnv([lambda: env]))
    eval_cb = EvalCallback(eval_env, best_model_save_path="./best_model/",
                           log_path="./logs/", eval_freq=5_000)


    # 11). Train!
    TIMESTEPS = 200_000
    model.learn(total_timesteps=TIMESTEPS, log_interval=10)

    # 12. Save the trained agent
    model.save("dqn_intraday_agent")

    # n. (Optional) Reload it later
    # model = DQN.load("dqn_intraday_agent", env=env)

    # n+1 Later, inference
    #obs = eval_env.reset()
    #for _ in range(1000):
    #    action, _ = model.predict(obs, deterministic=True)
    #    obs, reward, done, info = eval_env.step(action)
    #    if done:
    #        obs = eval_env.reset()


    # 13. Evaluate performance over, say, 10 episodes
    mean_reward, std_reward = evaluate_policy(
        model, model.get_env(), n_eval_episodes=10, render=False
    )
    print(f"Mean reward over 10 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

    # 7. Run one rollout and log P&L curve
    obs = env.reset()
    done = False
    pnl_curve = []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        pnl_curve.append(info["equity"] - env.initial_cash)

    # e.g. plot with matplotlib
    import matplotlib.pyplot as plt
    plt.plot(pnl_curve)
    plt.title("DQN Intraday P&L Curve")
    plt.xlabel("Bars")
    plt.ylabel("P&L (USD)")
    plt.show()


if __name__ == "__main__":
    main()

