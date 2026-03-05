import numpy as np
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# Pull scores from analyzer modules
# -------------------------------

def _get_score(module_name, attr_candidates, default):
    try:
        mod = __import__(module_name)
    except Exception as e:
        print(f"Warning: couldn't import {module_name}: {e}")
        return float(default)
    for attr in attr_candidates:
        if hasattr(mod, attr):
            try:
                return float(getattr(mod, attr))
            except Exception:
                continue
    print(f"Warning: {module_name} has no attributes {attr_candidates}; using default {default}")
    return float(default)

# try to read scores from existing analyzer modules; fall back to sensible defaults
graph_score = _get_score('GraphAnalyzer', ['trend_score', 'graph_score', 'final_score', 'score'], 8.0)
profile_score = _get_score('CompanyProfileAnalyzer', ['final_score', 'profile_score', 'company_score', 'score'], 7.9)
market_score = _get_score('MarketAnalyzer', ['market_score', 'final_score', 'score'], 6.5)
valuation_score = _get_score('ValuationAnalyzer', ['valuation_score', 'final_score', 'score'], 7.3)
news_score = _get_score('NewsAnalyzer', ['news_score', 'sentiment_score', 'final_score', 'score'], 5.9)

# -------------------------------
# Feature Vector
# -------------------------------

X_input = np.array([
    graph_score,
    profile_score,
    market_score,
    valuation_score,
    news_score
]).reshape(1,-1)

# -------------------------------
# Training Data (synthetic)
# -------------------------------

np.random.seed(42)

X_train = np.random.uniform(1,10,(500,5))

# weighted target generation
y_train = (
    0.30 * X_train[:,0] +
    0.20 * X_train[:,1] +
    0.15 * X_train[:,2] +
    0.20 * X_train[:,3] +
    0.15 * X_train[:,4]
)

# -------------------------------
# Train ML Model
# -------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train,y_train)

# -------------------------------
# Predict Final Score
# -------------------------------

final_score = model.predict(X_input)[0]

final_score = max(1,min(10,final_score))

final_score = round(final_score,2)

print("Final EquityIQ Score:", final_score)

# -------------------------------
# Recommendation Logic
# -------------------------------

if final_score < 2.5:
    recommendation = "STRONG SELL"

elif final_score < 5:
    recommendation = "SELL / WAIT"

elif final_score < 7.5:
    recommendation = "HOLD / MAY BUY"

elif final_score < 9:
    recommendation = "BUY"

else:
    recommendation = "STRONG BUY"

print("Recommendation:", recommendation)