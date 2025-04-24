import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed
np.random.seed(42)

# Page configuration
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# Custom CSS for a polished, professional UI
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #2b3e50; color: white; border-radius: 5px; padding: 8px 16px;}
    .stMetric {background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .stPlotlyChart, .stPlot {background-color: #ffffff; border-radius: 8px; padding: 10px;}
    h1, h2, h3 {color: #2b3e50; font-family: 'Arial', sans-serif;}
    .stSidebar {background-color: #e9ecef;}
    .stTextInput>label, .stSlider>label {color: #2b3e50; font-weight: bold;}
    .tooltip {position: relative; display: inline-block; cursor: help; border-bottom: 1px dotted #2b3e50;}
    .tooltip .tooltiptext {visibility: hidden; width: 200px; background-color: #2b3e50; color: #fff; text-align: center; border-radius: 6px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s;}
    .tooltip:hover .tooltiptext {visibility: visible; opacity: 1;}
    </style>
    """, unsafe_allow_html=True)

# Title and subtitle
st.title("🛡️ VolGuard: AI-Powered Trading Copilot")
st.markdown("Your disciplined partner for options trading, focusing on volatility, risk, and edge.")

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Trading Parameters")
    forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 14, 7, help="Number of days to forecast volatility.")
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000, help="Your trading capital.")
    risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1, help="Adjusts forecast weights and capital allocation.")
    journal_entry = st.text_area("Trade Journal", placeholder="Log your thoughts or trade rationale...", help="Record your decisions for behavioral tracking.")
    st.markdown("---")
    st.info("**Data Sources**: NIFTY (CSV), India VIX (CSV)")
    st.markdown("Built by Shritish Shukla & AI Co-Founder")

# Quick Action Bar
col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    if st.button("Refresh Data", help="Reload market data and forecasts"):
        st.rerun()
with col2:
    st.download_button("Download Report", data="VolGuard Report Placeholder", file_name="volguard_report.csv", help="Download full analysis (coming soon)")

# Step 1: Load NIFTY Data from CSV
with st.spinner("Loading NIFTY data from CSV..."):
    try:
        nifty = pd.read_csv("Nifty50.csv")
        nifty["Date"] = pd.to_datetime(nifty["Date"], errors="coerce")
        nifty = nifty.dropna(subset=["Date"])
        nifty = nifty.set_index("Date")
        nifty = nifty.sort_index()
        # Ensure the data has enough entries
        if len(nifty) < 10:
            raise ValueError("NIFTY data from CSV is too short.")
        st.info("✅ Successfully loaded NIFTY 50 data from CSV")
    except Exception as e:
        st.error(f"Error loading NIFTY data from CSV: {e}")
        st.warning("Falling back to mock NIFTY data for demonstration.")
        import pandas as pd
        import numpy as np
        mock_dates = pd.date_range(start="2024-04-25", end="2025-04-24", freq="B")
        mock_close = np.random.normal(loc=22000, scale=500, size=len(mock_dates))
        nifty = pd.DataFrame({
            "Open": mock_close,
            "High": mock_close + np.random.uniform(0, 200, len(mock_dates)),
            "Low": mock_close - np.random.uniform(0, 200, len(mock_dates)),
            "Close": mock_close,
            "Adj Close": mock_close,
            "Volume": np.random.randint(1000000, 5000000, len(mock_dates))
        }, index=mock_dates)
        nifty = nifty[~nifty.index.duplicated(keep='first')]

# Process NIFTY data
nifty_close = nifty["Close"].to_numpy().flatten()
dates = nifty.index
n_days = len(nifty)
log_returns = np.log(nifty["Close"] / nifty["Close"].shift(1)).dropna()

# Step 2: Load India VIX
with st.spinner("Loading India VIX data..."):
    try:
        vix_real = pd.read_csv("india_vix.csv")
        vix_real["Date"] = pd.to_datetime(vix_real["Date"], format="%d-%b-%Y", errors="coerce")
        vix_real = vix_real.dropna(subset=["Date"])
        vix_real = vix_real.set_index("Date")
        vix_real = vix_real[["Close"]].rename(columns={"Close": "VIX"})
        vix_real = vix_real.sort_index()
        vix_real = vix_real.loc[dates.min():dates.max()]
    except Exception as e:
        st.error(f"Error loading VIX data: {e}")
        st.stop()

# Align VIX with NIFTY
vix_real = vix_real[~vix_real.index.duplicated(keep='first')]
vix_aligned = vix_real.reindex(dates).ffill().bfill()
vix_simulated = vix_aligned["VIX"].to_numpy()

# Create DataFrame
df = pd.DataFrame({
    "NIFTY_Close": nifty_close,
    "VIX": vix_simulated,
    "Total_Capital": capital
}, index=dates)

# Step 3: Feature Engineering
with st.spinner("Engineering trading signals..."):
    risk_free_rate = 0.06
    strike_step = 100

    def black_scholes(S, K, T, r, sigma, option_type="call"):
        try:
            T = max(T, 1e-6)
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)
            else:
                return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)
        except:
            return 0

    def dynamic_ivp(x):
        if len(x) >= 5:
            return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
        return 50.0

    event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.15, 1.0)
    df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.05, n_days)) * event_spike
    df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 40)
    df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp).interpolate().fillna(50.0)
    market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
    df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.05, n_days) + market_trend * -5, 0.8, 1.8)
    df["IV_Skew"] = np.clip(np.random.normal(0, 0.5, n_days) + (df["VIX"] / 20 - 1) * 2, -2, 2)
    df["Days_to_Expiry"] = np.random.choice([1, 3, 7, 14, 21, 28], n_days)
    df["Event_Flag"] = np.where((df.index.month % 3 == 0) & (df.index.day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
    df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
    df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"]).clip(0, 50)
    df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
    df["Spot_MaxPain_Diff_Pct"] = np.clip(np.abs(np.random.lognormal(-2, 0.3, n_days)), 0.1, 0.8)
    df["FII_Index_Fut_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
    df["Capital_Pressure_Index"] = np.clip(df["FII_Index_Fut_Pos"] / 3e4 + df["PCR"], -1.5, 1.5)
    df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * 5, 0.5, 2.0)

    straddle_prices = []
    for i in range(n_days):
        S = df["NIFTY_Close"].iloc[i]
        K = round(S / strike_step) * strike_step
        T = df["Days_to_Expiry"].iloc[i] / 365
        sigma = df["ATM_IV"].iloc[i] / 100
        call_price = black_scholes(S, K, T, risk_free_rate, sigma, "call")
        put_price = black_scholes(S, K, T, risk_free_rate, sigma, "put")
        straddle_price = (call_price + put_price) * (S / 1000)
        straddle_price = np.clip(straddle_price, 50, 350)
        straddle_prices.append(straddle_price)
    df["Straddle_Price"] = straddle_prices

# Display market snapshot
st.subheader("Market Snapshot")
col1, col2, col3, col4 = st.columns(4)
col1.metric("NIFTY Close", f"{df['NIFTY_Close'].iloc[-1]:,.2f}")
col2.metric("India VIX", f"{df['VIX'].iloc[-1]:.2f}%", help="Market volatility index")
col3.metric("ATM IV", f"{df['ATM_IV'].iloc[-1]:.2f}%", help="At-the-money implied volatility")
col4.metric("Realized Vol", f"{df['Realized_Vol'].iloc[-1]:.2f}%", help="Historical volatility")

# Risk Dashboard
st.subheader("Risk Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("IV-RV Spread", f"{df['ATM_IV'].iloc[-1] - df['Realized_Vol'].iloc[-1]:.2f}%", help="Implied vs. realized volatility gap")
col2.metric("Capital Pressure", f"{df['Capital_Pressure_Index'].iloc[-1]:.2f}", help="FII and PCR-based market pressure")
col3.metric("VIX Change", f"{df['VIX_Change_Pct'].iloc[-1]:.2f}%", help="Daily VIX percentage change")

# Step 4: Volatility Forecasting
st.subheader("Volatility Forecast")
with st.spinner("Generating forecasts..."):
    # GARCH Forecast with Confidence Intervals
    df['Log_Returns'] = np.log(df['NIFTY_Close'] / df['NIFTY_Close'].shift(1)).dropna()
    returns = df['Log_Returns'].dropna()
    garch_model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
    garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252) * 100
    garch_std = np.sqrt(garch_forecast.variance.iloc[-1].values) * 0.1  # Simplified CI
    if df["Event_Flag"].iloc[-1] == 1:
        garch_vols *= 1.1

    # XGBoost Forecast
    df['Target_Vol'] = df['Realized_Vol'].shift(-1)
    df_temp = df.dropna()
    feature_cols = ['VIX', 'ATM_IV', 'IVP', 'PCR', 'IV_Skew', 'Straddle_Price', 'Days_to_Expiry', 'Event_Flag', 'Capital_Pressure_Index', 'Advance_Decline_Ratio']
    X = df_temp[feature_cols]
    y = df_temp['Target_Vol']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    xgb_vols = []
    current_row = X.iloc[-1].copy()
    current_row_df = pd.DataFrame([current_row], columns=feature_cols)
    current_row_scaled = scaler.transform(current_row_df)
    for _ in range(forecast_horizon):
        next_vol = model.predict(current_row_scaled)[0]
        xgb_vols.append(next_vol)
        current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
        current_row["VIX"] *= np.random.uniform(0.99, 1.01)
        current_row_df = pd.DataFrame([current_row], columns=feature_cols)
        current_row_scaled = scaler.transform(current_row_df)
    if df["Event_Flag"].iloc[-1] == 1:
        xgb_vols = [v * 1.1 for v in xgb_vols]

    # Blended Forecast
    garch_weight = 0.6 if risk_tolerance == "Conservative" else 0.5 if risk_tolerance == "Moderate" else 0.4
    xgb_weight = 1 - garch_weight
    blended_vols = [(garch_weight * g + xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
    future_dates_str = [d.strftime('%d-%m-%Y') for d in future_dates]

# Display forecasts
st.markdown("<span class='tooltip'>**Blended Volatility Forecast**<span class='tooltiptext'>Combines GARCH and XGBoost models for robust predictions.</span></span>", unsafe_allow_html=True)
forecast_df = pd.DataFrame({
    "Date": future_dates_str,
    "GARCH (%)": [f"{v:.2f}" for v in garch_vols],
    "XGBoost (%)": [f"{v:.2f}" for v in xgb_vols],
    "Blended (%)": [f"{v:.2f}" for v in blended_vols]
})
st.table(forecast_df)

# Plot forecasts with confidence intervals
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(future_dates, blended_vols, marker='o', linestyle='-', color='#dc3545', label="Blended Forecast")
ax.fill_between(future_dates, blended_vols - garch_std, blended_vols + garch_std, color='#dc3545', alpha=0.2, label="Confidence Interval")
ax.set_xlabel("Date")
ax.set_ylabel("Volatility (%)")
ax.set_title("Volatility Forecast with Confidence Interval")
plt.xticks(rotation=45)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
st.pyplot(fig)

# Step 5: Regime Classification & Strategy Recommendation
st.subheader("Strategy & Risk Management")
latest = df.iloc[-1]
avg_vol = np.mean(blended_vols)
iv_hv_gap = latest["ATM_IV"] - latest["Realized_Vol"]
event_flag = latest["Event_Flag"]
pcr = latest["PCR"]
iv_skew = latest["IV_Skew"]

# Regime Classification
if event_flag == 1:
    regime = "EVENT-DRIVEN"
elif avg_vol < 15:
    regime = "LOW"
elif avg_vol < 20:
    regime = "MEDIUM"
else:
    regime = "HIGH"

# Strategy Recommendation
strategy, reason, tags, confidence_score, risk_reward = "Hold Cash", "No clear edge.", ["Neutral"], 0.5, 1.0
if regime == "LOW":
    if iv_hv_gap > 3 and latest["Days_to_Expiry"] < 10:
        strategy = "Butterfly Spread"
        reason = "Low volatility and short expiry favor pinning strategies."
        tags = ["Neutral", "Theta", "Expiry Play"]
        risk_reward = 2.0
        confidence_score = 0.75
    else:
        strategy = "Iron Fly"
        reason = "Stable conditions for delta-neutral premium selling."
        tags = ["Neutral", "Theta"]
        confidence_score = 0.7
elif regime == "MEDIUM":
    if iv_skew > 1:
        strategy = "Calendar Spread"
        reason = "Moderate volatility with skew suggests term structure plays."
        tags = ["Volatility", "Skew"]
        risk_reward = 1.8
        confidence_score = 0.8
    else:
        strategy = "Short Strangle"
        reason = "Balanced conditions for premium collection."
        tags = ["Neutral", "Premium"]
        confidence_score = 0.75
elif regime == "HIGH":
    if iv_hv_gap > 5:
        strategy = "Jade Lizard"
        reason = "High IV with skew favors defined-risk plays."
        tags = ["Skew", "Volatility"]
        risk_reward = 1.5
        confidence_score = 0.65
    else:
        strategy = "Debit Spread"
        reason = "High volatility favors directional plays with limited risk."
        tags = ["Directional", "Volatility"]
        risk_reward = 1.4
        confidence_score = 0.6
elif regime == "EVENT-DRIVEN":
    if latest["ATM_IV"] > 25:
        strategy = "Straddle Buy"
        reason = "Event-driven uncertainty favors gamma exposure."
        tags = ["High Gamma", "Event"]
        risk_reward = 1.3
        confidence_score = 0.85
    else:
        strategy = "Calendar Spread"
        reason = "Event with moderate IV suggests term structure opportunities."
        tags = ["Volatility", "Event"]
        risk_reward = 1.5
        confidence_score = 0.8

# Risk Filters
capital_alloc = {"LOW": 0.2, "MEDIUM": 0.15, "HIGH": 0.1, "EVENT-DRIVEN": 0.12}
if risk_tolerance == "Conservative":
    capital_alloc = {k: v * 0.8 for k, v in capital_alloc.items()}
elif risk_tolerance == "Aggressive":
    capital_alloc = {k: v * 1.2 for k, v in capital_alloc.items()}
deploy = capital * capital_alloc.get(regime, 0.1)
max_loss = deploy * 0.2
total_exposure = deploy / capital
risk_flags = []
if total_exposure > 0.7:
    risk_flags.append("Exposure exceeds 70% cap")
if regime in ["HIGH", "EVENT-DRIVEN"] and strategy in ["Short Strangle", "Iron Fly"]:
    risk_flags.append("No naked legs in HIGH/EVENT regimes")
if latest["VIX_Change_Pct"] > 8:
    risk_flags.append("VIX spike detected")
if journal_entry and len(journal_entry.split()) > 50:
    risk_flags.append("Long journal entry may indicate emotional trading")

# Behavioral Monitoring
behavior_score = 8
if deploy > 0.5 * capital:
    behavior_score -= 2
if len(risk_flags) > 1:
    behavior_score -= 1
if journal_entry and "urgent" in journal_entry.lower():
    behavior_score -= 1
behavior_score = max(1, behavior_score)
behavior_warnings = []
if behavior_score < 7:
    behavior_warnings.append("Consider pausing and reviewing strategy")
if journal_entry and len(journal_entry.split()) > 50:
    behavior_warnings.append("Simplify journal to focus on rationale")

# Display strategy
st.markdown(f"""
### Trading Recommendation
- **Volatility Regime**: {regime} (Avg Vol: {avg_vol:.2f}%)
- **Strategy**: {strategy}
- **Reason**: {reason}
- **Tags**: {', '.join(tags)}
- **Confidence Score**: {confidence_score:.2f}
- **Risk-Reward**: {risk_reward:.2f}:1
- **Capital to Deploy**: ₹{deploy:,.0f}
- **Max Risk**: ₹{max_loss:,.0f}
""")
if risk_flags:
    st.warning("🚨 **Risk Flags**: " + ", ".join(risk_flags))
if behavior_warnings:
    st.info("🧠 **Behavioral Note**: " + ", ".join(behavior_warnings))
st.metric("Behavior Score", f"{behavior_score}/10", help="Reflects trading discipline")

# Step 6: Export Data
df.to_csv("volguard_data.csv")
forecast_log = pd.DataFrame({
    "Date": future_dates,
    "GARCH_Vol": garch_vols,
    "XGBoost_Vol": xgb_vols,
    "Blended_Vol": blended_vols
})
forecast_log.to_csv("volguard_forecast_log.csv")

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download Market Data", data=open("volguard_data.csv", "rb"), file_name="volguard_data.csv", help="Save processed market data")
with col2:
    st.download_button("Download Forecast Log", data=open("volguard_forecast_log.csv", "rb"), file_name="volguard_forecast_log.csv", help="Save volatility forecasts")

# Journal Confirmation
if journal_entry:
    st.success("Journal entry logged. Review it to maintain discipline.")
