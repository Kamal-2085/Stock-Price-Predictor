import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Download stock data
# -----------------------------
data = yf.download("COALINDIA.NS", period="2y", interval="1d")

data = data[['Close','Volume']]

# -----------------------------
# Feature Engineering
# -----------------------------

# Moving averages
data['MA5'] = data['Close'].rolling(5).mean()
data['MA20'] = data['Close'].rolling(20).mean()

# Daily return
data['Return'] = data['Close'].pct_change()

# Momentum
data['Momentum'] = data['Close'] - data['Close'].shift(5)

# Volatility
data['Volatility'] = data['Return'].rolling(10).std()

# -----------------------------
# RSI
# -----------------------------
delta = data['Close'].diff()

gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()

rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# -----------------------------
# MACD
# -----------------------------
ema12 = data['Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Close'].ewm(span=26, adjust=False).mean()

data['MACD'] = ema12 - ema26
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# -----------------------------
# Target variable
# -----------------------------
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

data = data.dropna()

# -----------------------------
# Features and labels
# -----------------------------
X = data[['MA5','MA20','Return','Momentum','Volatility','RSI','MACD','MACD_signal']]
y = data['Target']

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# Predict latest trend
# -----------------------------
latest_data = X.iloc[-1:]

prediction = model.predict(latest_data)
probability = model.predict_proba(latest_data)[0]

# probability of price going UP
up_prob = probability[1]

# convert probability → 1–10 scale
trend_score = round(up_prob * 9 + 1)

# -----------------------------
# Output
# -----------------------------
print("Probability of Upward Trend:", round(up_prob,2))
print("Graph Analyzer Score (1-10):", trend_score)