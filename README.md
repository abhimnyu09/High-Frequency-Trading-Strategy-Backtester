# High-Frequency Trading Strategy Backtester

This project is a complete Python framework for sourcing high-frequency cryptocurrency data, engineering predictive features, training a machine learning model, and backtesting a trading strategy. The goal is to simulate the workflow of a quantitative researcher in identifying and testing a potential "alpha" signal.

## Overview

The script automatically performs the following steps:
1.  **Data Acquisition:** Downloads historical high-frequency trade data for BTC/USDT directly from the Binance public API.
2.  **Feature Engineering:** Resamples the tick data into time-based bars (e.g., 5 seconds) and calculates advanced features like:
    * OHLCV (Open, High, Low, Close, Volume)
    * **Trade Flow Imbalance (TFI):** A measure of aggressive buying vs. selling pressure.
    * **Rolling Volatility:** The standard deviation of recent returns.
    * **Feature Interactions:** Combining features to create new signals (e.g., TFI x Volatility).
3.  **Modeling:** Trains a **LightGBM** classifier to predict the direction of the next price movement based on the engineered features. The model includes regularization and other tuned hyperparameters to fight overfitting.
4.  **Backtesting:** Simulates the trading strategy based on the model's predictions. The backtest includes:
    * **Confidence Thresholding:** Only placing trades when the model's predicted probability is above a certain level.
    * **Transaction Costs:** Applying a realistic per-trade fee to simulate real-world conditions.
5.  **Analysis:** Generates a plot of the cumulative profit/loss (PnL) to evaluate the strategy's performance.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abhimnyu09
    cd High-Frequency-Trading-Strategy-Backtester
    ```

2.  **Install dependencies:**
    Make sure you have Python 3 installed. Then, run the following command to install the required libraries:
    ```bash
    pip3 install requests pandas scikit-learn lightgbm matplotlib
    ```

3.  **Run the script:**
    ```bash
    python3 main.py
    ```
    The script will download the latest data, train the model, and display the PnL plot.
    For Windows,use pip and python in place of pip3 and python3.

## Results and Analysis

A typical run of this script will produce a **negative or flat PnL curve**. This is a realistic and expected outcome. The primary goal of this project is not to provide a "get rich quick" algorithm, but to build a robust research framework.

The descending PnL demonstrates the fundamental challenge of quantitative trading: finding a true predictive signal ("alpha") that is strong enough to consistently overcome transaction costs and market noise. The value of this project lies in the complete pipeline for testing new ideas, features, and models.
