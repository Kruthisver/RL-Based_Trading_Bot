# 📈 Reinforcement Learning Trading Bot

This project implements a **Reinforcement Learning (RL) trading agent** that learns to optimize portfolio allocation across multiple assets.  
It leverages **Stable Baselines3 (SB3)** with a **custom Gym environment** (`SuperiorTradingEnv2`) designed to simulate realistic trading conditions, including transaction costs, slippage, and Sharpe ratio–based reward shaping.  

---

## ✨ Key Features
- 🏦 **Custom trading environment** (`Environment2.py`) compatible with SB3 & Gymnasium API  
- 📊 **Portfolio allocation**: actions represent percentage allocation across assets  
- ⚖️ **Sharpe ratio–based rewards** to balance returns and risk  
- 🔄 **Normalized observations** for stable learning (OHLC percent changes & log-scaled volume)  
- 🧩 **Configurable parameters**:
  - Window size (observation length)  
  - Initial trading capital  
  - Transaction costs & slippage  
  - Rolling Sharpe ratio window  
- 📓 **End-to-end notebook (`RL_Model.ipynb`)** showing:
  - Data loading & preprocessing  
  - Training with PPO (or other SB3 agents)
  - Optuna based  Hyperparameter Tuning 
  - Evaluation & backtesting  
  - Portfolio performance visualization  
- Added the **(`RL_Model.py`)** file for easy implementation
---

## ⚙️ Installation

Clone this repository and install the dependencies:

```bash
git clone https://github.com/yourusername/rl-trading-bot.git
cd rl-trading-bot

pip install -r requirements.txt
