import requests
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def get_historical_trades(symbol='BTCUSDT', hours=12):
    """Downloads historical aggregated trades for a specified number of hours."""
    end_time = int(time.time() * 1000)
    start_time = end_time - (hours * 60 * 60 * 1000)
    all_trades = []
    current_start_time = start_time
    
    print(f"Downloading trades for the last {hours} hours...")

    while current_start_time < end_time:
        url = 'https://api.binance.com/api/v3/aggTrades'
        params = {
            'symbol': symbol,
            'startTime': int(current_start_time),
            'endTime': int(end_time),
            'limit': 1000
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            break
        all_trades.extend(data)
        current_start_time = data[-1]['T'] + 1
    
    full_df = pd.DataFrame(all_trades).drop_duplicates('a')
    full_df['T'] = pd.to_datetime(full_df['T'], unit='ms')
    full_df.rename(columns={'p': 'price', 'q': 'quantity', 'T': 'timestamp', 'm': 'isBuyerMaker'}, inplace=True)
    full_df[['price', 'quantity']] = full_df[['price', 'quantity']].apply(pd.to_numeric)
    
    print(f"Successfully downloaded {len(full_df)} unique trades.")
    return full_df[['timestamp', 'price', 'quantity', 'isBuyerMaker']]

# Downloading data
real_trades_df = get_historical_trades(hours=12)
real_trades_df.set_index('timestamp', inplace=True)

# Defining interval and resample
resample_interval = '5s'
resampled_df = real_trades_df['price'].resample(resample_interval).ohlc()
resampled_df['volume'] = real_trades_df['quantity'].resample(resample_interval).sum()

# Engineer features
def calculate_tfi(df_slice):
    aggressive_buys = df_slice[df_slice['isBuyerMaker'] == False]['quantity'].sum()
    aggressive_sells = df_slice[df_slice['isBuyerMaker'] == True]['quantity'].sum()
    total_volume = aggressive_buys + aggressive_sells
    return (aggressive_buys - aggressive_sells) / total_volume if total_volume > 0 else 0

resampled_df['tfi'] = real_trades_df.resample(resample_interval).apply(calculate_tfi)
resampled_df['returns'] = np.log(resampled_df['close'] / resampled_df['close'].shift(1))
resampled_df['volatility'] = resampled_df['returns'].rolling(window=5).std()

# Create target variable
look_ahead_period = 5
resampled_df['future_close'] = resampled_df['close'].shift(-look_ahead_period)
resampled_df['target'] = np.sign(resampled_df['future_close'] - resampled_df['close'])
final_df = resampled_df.dropna()
final_df['tfi_x_vol'] = final_df['tfi'] * final_df['volatility']

# Define features and split data
features = ['open', 'high', 'low', 'close', 'volume', 'tfi', 'volatility', 'tfi_x_vol']
X = final_df[features]
y = final_df['target']
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train model
model = lgb.LGBMClassifier(
    objective='multiclass', num_class=3, random_state=42,
    n_estimators=200, learning_rate=0.05, num_leaves=40,
    reg_alpha=0.1, reg_lambda=0.1
)
model.fit(X_train[y_train != 0], y_train[y_train != 0])

# Evaluate model
probabilities = model.predict_proba(X_test)
predictions = model.classes_[np.argmax(probabilities, axis=1)]
confidence = np.max(probabilities, axis=1)
accuracy = accuracy_score(y_test[y_test != 0], predictions[y_test != 0])
print(f"\nLightGBM Model Accuracy: {accuracy:.4f}")

# Attach results to the test DataFrame
test_df = final_df[split_index:].copy()
test_df['prediction'] = predictions
test_df['confidence'] = confidence

# Run the Backtest
confidence_threshold = 0.45
transaction_cost_pct = 0.0004

pnl_vector = (test_df['future_close'] / test_df['close'] - 1) * test_df['prediction'] - transaction_cost_pct
pnl_vector = pnl_vector.where(test_df['confidence'] > confidence_threshold, 0)
test_df['pnl'] = pnl_vector.fillna(0)
test_df['pnl'] = test_df['pnl'].where(test_df['prediction'] != 0, 0)
test_df['cumulative_pnl'] = test_df['pnl'].cumsum()

# Plot the Results
plt.figure(figsize=(12, 6))
plt.plot(test_df['cumulative_pnl'])
plt.title('Cumulative PnL from Advanced Strategy (After Costs)')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()
