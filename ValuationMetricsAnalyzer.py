import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------
# Load ticker
# -----------------------------------

ticker_symbol = "COALINDIA.NS"
ticker = yf.Ticker(ticker_symbol)

info = ticker.info

# -----------------------------------
# Extract valuation metrics
# -----------------------------------

market_cap = info.get("marketCap",0)

pe_ratio = info.get("trailingPE",0)

eps = info.get("trailingEps",0)

book_value = info.get("bookValue",0)

dividend_yield = info.get("dividendYield",0)

beta = info.get("beta",1)

# -----------------------------------
# Feature Vector
# -----------------------------------

features = np.array([
    market_cap,
    pe_ratio,
    eps,
    book_value,
    dividend_yield,
    beta
]).reshape(1,-1)

# -----------------------------------
# Dummy training data
# (later replace with real dataset)
# -----------------------------------

X_train = np.random.rand(200,6)

y_train = np.random.uniform(1,10,200)

# -----------------------------------
# Train Model
# -----------------------------------

model = RandomForestRegressor(n_estimators=300)

model.fit(X_train,y_train)

# -----------------------------------
# Predict Valuation Score
# -----------------------------------

score = model.predict(features)[0]

score = max(1,min(10,score))

print("Valuation Metrics Score (1-10):",round(score,2))