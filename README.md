Based on the code and markdown cells contained in your notebook, here is a comprehensive and professional `README.md` file.

# 🤖 PPO-Powered Bitcoin Trading Agent

A Deep Reinforcement Learning (DRL) project that implements a **Proximal Policy Optimization (PPO)** agent to trade Bitcoin (BTC/USDT) on hourly timeframes. The agent utilizes an **Actor-Critic** architecture with **1D Convolutional layers** for feature extraction, optimized via **Optuna** to outperform the market buy-and-hold strategy.

## 📊 Key Results (Unseen Test Set)

The model was evaluated on a strictly held-out test set (Oct 2025).

| Metric | Market Baseline (Buy & Hold) | PPO Agent | Performance Delta (Alpha) |
| :--- | :---: | :---: | :---: |
| **Total Return** | +4.78% | **+8.63%** | **+3.85%** |

## 🚀 Project Overview

This solution addresses the challenge of autonomous trading using a full DRL pipeline:

1.  **Data Engineering:** Calculation of technical indicators and removal of look-ahead bias.
2.  **Baseline Creation:** Comparison against a Random Agent to ensure learning capability.
3.  **Custom Architecture:** Implementation of a PyTorch-based PPO agent from scratch.
4.  **Hyperparameter Tuning:** Automated search using Tree-structured Parzen Estimator (TPE) via Optuna.
5.  **Robust Evaluation:** Walk-forward validation and final testing on unseen data.

## 🧠 Model Architecture

The agent uses an **Actor-Critic** network designed for time-series data:

  * **Input:** A rolling window of **48 hours** of market features.
  * **Feature Extractor:** 2-layer **1D Convolutional Neural Network (CNN)** to capture temporal patterns and local trends.
  * **Shared Layers:** Dynamic linear layers to process flattened features.
  * **Heads:**
      * **Actor:** Outputs logits for 9 discrete actions (Short $\leftrightarrow$ Neutral $\leftrightarrow$ Long).
      * **Critic:** Estimates the value function $V(s)$.

## 🛠️ Features & Data

The model uses **Binance BTC/USDT 1h** data. The following features are engineered and normalized:

  * **Trend:** MACD, EMA (12, 26).
  * **Volatility:** Bollinger Bands, ATR (Average True Range).
  * **Momentum:** RSI (Relative Strength Index).
  * **Volume:** On-Balance Volume (OBV), Volume Z-Score.
  * **Risk:** Rolling Sharpe Ratio.
  * **Time:** Cyclical encoding (Sine/Cosine) for Hour and Day of Week.

*Note: All features are rigorously lagged (shifted) to prevent look-ahead bias.*

## ⚙️ Hyperparameters (Optuna Optimized)

The final model was trained using the following hyperparameters identified as optimal during the Optuna study:

  * **Learning Rate:** $4.25 \times 10^{-5}$
  * **Gamma ($\gamma$):** 0.99 (Long-term horizon)
  * **GAE Lambda ($\lambda$):** 0.95
  * **Entropy Beta:** $1.69 \times 10^{-3}$
  * **Clip Epsilon:** 0.115
  * **Network:** 4 Layers, Hidden Size 256, Batch Size 128.

## 📂 Project Structure

The notebook follows this execution flow:

1.  **Environment Setup:** Install `gym-trading-env`, `torch`, `optuna`.
2.  **Preprocessing:** Feature engineering and chronological train/validation/test splitting.
3.  **Baseline:** Evaluation of a `RandomAgent`.
4.  **Agent Implementation:** Coding the `ActorCriticNetwork` and `PPOAgent` classes.
5.  **Optimization:** Running the `objective` function with Optuna to maximize validation Excess Return.
6.  **Final Training:** Training the best model configuration on the full training set (1M steps) with early stopping based on validation performance.
7.  **Evaluation:** Loading the best weights (`models/ppo_trading_agent_v2.pth`) and testing on the unseen evaluation dataset.

## 💻 Usage

### Prerequisites

  * Python 3.8+
  * Jupyter Notebook / Google Colab

### Installation

```bash
pip install -r requirements.txt
# Key dependencies: gym-trading-env, torch, pandas, numpy, optuna, matplotlib
```

### Training & Evaluation

Run the cells in `Project_Reinforcement_learning_v2.ipynb` sequentially.

1.  **Pre-training:** The notebook downloads data and prepares features.
2.  **Optimization:** The Optuna cell will run for \~50 trials (configurable).
3.  **Production Train:** The final loop trains the agent and saves the best model to `models/`.
4.  **Inference:** The final cell loads the saved model and produces the `evaluation_results.csv`.

## 📈 Reward Function

To prevent the agent from learning a trivial "stay neutral" strategy, a custom reward function is used:
$$Reward = PnL + \text{Neutrality Penalty}$$

  * **PnL:** Realized profit/loss from the position.
  * **Neutrality Penalty:** A small negative reward for holding no position ($position = 0$), incentivizing active market participation when profitable.

-----

**Author:** Yann MEUROU & Téo Desquatrevaux
**Date:** November 2025

