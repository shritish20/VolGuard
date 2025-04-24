# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# Page setup
st.set_page_config(page_title="VolGuard", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚙️ VolGuard Controls")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 14, 7)
    capital = st.number_input("Capital (₹)", 100000, 1_00_00_000, 10_00_000, step=100000)
    risk_mode = st.radio("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
    st.info("India VIX is loaded from your GitHub repo.\nNIFTY from Yahoo Finance.")
    st.markdown("Made by Shritish Shukla")

# Fetch NIFTY Data
st.title("🛡️ VolGuard: AI Copilot for Options")
st.markdown("Your volatility-driven strategy assistant")

with st.spinner("Fetching NIFTY..."):
    nifty = yf.download("^NSEI", period="1y", interval="1d", auto_adjust=True)
    if nifty.empty:
        st.error("Could not fetch NIFTY data.")
        st.stop()

# VIX from GitHub CSV
with st.spinner("Loading India VIX from GitHub..."):
    vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/india_vix.csv"
    try:
        vix_df = pd.read_csv(vix_url)
        vix_df["Date"] = pd.to_datetime(vix_df["Date"], format="%d-%b-%Y", errors="coerce")
        vix_df = vix_df.dropna(subset=["Date"])
        vix_df = vix_df.set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})
    except Exception as e:
        st.error(f"VIX file failed to load: {e}")
        st.stop()

# Align and clean
nifty = nifty[~nifty.index.duplicated()]
vix_df = vix_df[~vix_df.index.duplicated()]
df = pd.DataFrame(index=nifty.index)
df["NIFTY"] = nifty["Close"]
df["VIX"] = vix_df["VIX"].reindex(df.index).ffill().bfill()
df.dropna(inplace=True)

# Basic features
df["IV"] = df["VIX"] * (1 + np.random.normal(0, 0.05, len(df)))
df["RV"] = df["NIFTY"].pct_change().rolling(5).std() * np.sqrt(252) * 100
df["IVRV_Gap"] = df["IV"] - df["RV"]
df["PCR"] = np.clip(1.1 + np.random.normal(0, 0.1, len(df)), 0.8, 1.8)
df["IV_Skew"] = np.random.normal(0, 0.5, len(df))
df["DTE"] = np.random.choice([3, 7, 14, 21, 28], len(df))
df["Event"] = np.where((df.index.day < 3) | (df["DTE"] <= 3), 1, 0)

# GARCH Forecast
returns = np.log(df["NIFTY"] / df["NIFTY"].shift(1)).dropna()
model = arch_model(returns, vol="Garch", p=1, q=1)
fit = model.fit(disp="off")
garch_vol = np.sqrt(fit.forecast(horizon=forecast_horizon).variance.iloc[-1]) * np.sqrt(252) * 100

# XGBoost Forecast
df["Target"] = df["RV"].shift(-1)
features = ["VIX", "IV", "PCR", "IVRV_Gap", "IV_Skew", "DTE"]
df.dropna(inplace=True)
X = df[features]
y = df["Target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05)
xgb.fit(X_scaled[:-forecast_horizon], y[:-forecast_horizon])
last_input = X_scaled[-1]
xgb_preds = [xgb.predict([last_input])[0] for _ in range(forecast_horizon)]

# Blend
w_garch = 0.6 if risk_mode == "Conservative" else 0.5 if risk_mode == "Moderate" else 0.4
w_xgb = 1 - w_garch
blended = [w_garch * g + w_xgb * x for g, x in zip(garch_vol, xgb_preds)]

# Output Forecast
future_dates = [df.index[-1] + timedelta(days=i+1) for i in range(forecast_horizon)]
vol_df = pd.DataFrame({
    "Date": future_dates,
    "GARCH": garch_vol,
    "XGBoost": xgb_preds,
    "Blended": blended
})

st.subheader("📈 Volatility Forecast (Annualized %)")
st.dataframe(vol_df.set_index("Date").round(2))

# Plot
fig, ax = plt.subplots()
ax.plot(future_dates, blended, label="Blended", color="blue", marker='o')
ax.plot(future_dates, garch_vol, label="GARCH", linestyle='--', alpha=0.6)
ax.plot(future_dates, xgb_preds, label="XGBoost", linestyle='--', alpha=0.6)
ax.set_title("Next 7-Day Volatility Forecast")
ax.set_ylabel("Volatility (%)")
ax.legend()
st.pyplot(fig)

# Regime Detection & Strategy
avg_vol = np.mean(blended)
iv_hv = df["IVRV_Gap"].iloc[-1]
event = df["Event"].iloc[-1]

if event:
    strategy = "Straddle Buy" if df["IV"].iloc[-1] > 25 else "Calendar Spread"
    reason = "Event-driven uncertainty"
elif avg_vol < 15:
    strategy = "Iron Fly" if iv_hv > 3 else "Butterfly"
    reason = "Low volatility environment"
elif avg_vol < 20:
    strategy = "Short Strangle"
    reason = "Moderate vol, favorable for neutral selling"
else:
    strategy = "Jade Lizard" if iv_hv > 5 else "Debit Spread"
    reason = "High vol — protect risk"

st.subheader("📊 Strategy Recommendation")
st.markdown(f"**Regime**: {'EVENT' if event else 'NORMAL'} | **Avg Forecast Vol**: `{avg_vol:.2f}%`")
st.markdown(f"**Suggested Strategy**: `{strategy}` — _{reason}_")
