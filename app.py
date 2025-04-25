import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# UI Styling
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #2b3e50; color: white; border-radius: 5px; padding: 8px 16px;}
    .stMetric {background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    h1, h2, h3 {color: #2b3e50; font-family: 'Arial', sans-serif;}
    .stSidebar {background-color: #e9ecef;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🛡️ VolGuard: AI-Powered Trading Copilot")
st.markdown("Your disciplined partner for volatility-driven strategy insights.")

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Parameters")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7)
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
    risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
    st.markdown("---")
    st.info("India VIX & NIFTY from GitHub | Built by Shritish")

# Load NIFTY from CSV
try:
    nifty = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/Nifty50.csv")
    nifty["Date"] = pd.to_datetime(nifty["Date"])
    nifty = nifty.set_index("Date")
    st.success("✅ Successfully loaded NIFTY data")
except Exception as e:
    st.error(f"Error loading NIFTY data: {e}")
    st.stop()

# Load India VIX
try:
    vix = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/india_vix.csv")
    vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
    vix = vix.dropna(subset=["Date"])
    vix = vix.set_index("Date")
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})
    vix = vix.sort_index()
    st.success("✅ Successfully loaded India VIX")
except Exception as e:
    st.error(f"Error loading India VIX: {e}")
    st.stop()

# Align data
nifty = nifty[~nifty.index.duplicated()]
vix = vix[~vix.index.duplicated()]
df = pd.DataFrame(index=nifty.index)
df["NIFTY_Close"] = nifty["Close"]
df["VIX"] = vix.reindex(nifty.index).ffill().bfill()["VIX"]
df["Log_Returns"] = np.log(df["NIFTY_Close"] / df["NIFTY_Close"].shift(1))
df = df.dropna()

# GARCH Model
with st.spinner("Running GARCH model..."):
    garch_model = arch_model(df["Log_Returns"] * 100, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    forecast = garch_fit.forecast(horizon=forecast_horizon)
    garch_vol = np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252)

# XGBoost Forecast
with st.spinner("Running XGBoost forecast..."):
    df['Target_Vol'] = df['Log_Returns'].rolling(5).std().shift(-1) * np.sqrt(252) * 100
    df = df.dropna()
    features = ["VIX", "NIFTY_Close"]
    X = df[features]
    y = df['Target_Vol']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = XGBRegressor(n_estimators=100)
    model.fit(X_scaled, y)
    last_row = scaler.transform([X.iloc[-1]])
    xgb_vol = []
    for _ in range(forecast_horizon):
        pred = model.predict(last_row)[0]
        xgb_vol.append(pred)

# Blend Forecasts
garch_wt = 0.6 if risk_tolerance == "Conservative" else 0.5 if risk_tolerance == "Moderate" else 0.4
xgb_wt = 1 - garch_wt
blended_vol = [(g * garch_wt + x * xgb_wt) for g, x in zip(garch_vol, xgb_vol)]
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

# Display Forecasts
st.subheader("Forecasted Volatility")
forecast_df = pd.DataFrame({
    "Date": future_dates.strftime("%d-%b-%Y"),
    "GARCH (%)": [f"{v:.2f}" for v in garch_vol],
    "XGBoost (%)": [f"{v:.2f}" for v in xgb_vol],
    "Blended (%)": [f"{v:.2f}" for v in blended_vol]
})
st.dataframe(forecast_df, use_container_width=True)

# Plot Forecasts
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(future_dates, blended_vol, label="Blended Volatility", marker='o')
ax.set_title("Next 7-Day Volatility Forecast")
ax.set_ylabel("Volatility (%)")
ax.grid(True)
ax.legend()
st.pyplot(fig)
