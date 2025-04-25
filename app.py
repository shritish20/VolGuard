import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Page config
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #2b3e50; color: white; border-radius: 5px; padding: 8px 16px;}
    .stMetric {background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    h1, h2, h3 {color: #2b3e50; font-family: 'Arial', sans-serif;}
    .stSidebar {background-color: #e9ecef;}
    .warning {color: #d32f2f; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🛡️ VolGuard: AI-Powered Trading Copilot")
st.markdown("Your disciplined partner for volatility-driven options trading.")
st.write(f"Date: {datetime.now().strftime('%d-%b-%Y')}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Parameters")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7)
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
    risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"], index=1)
    
    st.header("Actions")
    run_button = st.button("Run VolGuard")
    st.markdown("---")
    st.info("Built by Shritish Shukla & AI Co-Founder | Protection First, Edge Always")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'forecast_log' not in st.session_state:
    st.session_state.forecast_log = None
if 'feature_stats' not in st.session_state:
    st.session_state.feature_stats = None

# Function to fetch and process data
def load_data():
    # Fetch NIFTY data
    try:
        nifty = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/Nifty50.csv")
        nifty["Date"] = pd.to_datetime(nifty["Date"], format="%Y-%m-%d", errors="coerce")
        nifty = nifty.dropna(subset=["Date"]).set_index("Date")
        if nifty.empty or len(nifty) < 10:
            raise ValueError("NIFTY data is empty or too short.")
    except Exception as e:
        st.error(f"Error fetching NIFTY data: {e}")
        return None
    
    # Fetch VIX data
    try:
        vix_data = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv")
        vix_data["Date"] = pd.to_datetime(vix_data["Date"], format="%d-%b-%Y", errors="coerce")
        vix_data = vix_data.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})
        vix_data = vix_data.sort_index()
    except Exception as e:
        st.error(f"Error fetching VIX data: {e}")
        return None
    
    nifty = nifty[~nifty.index.duplicated(keep='first')]
    vix_data = vix_data[~vix_data.index.duplicated(keep='first')]
    dates = nifty.index
    n_days = len(nifty)
    
    vix_aligned = vix_data.reindex(dates).ffill().bfill()
    
    df = pd.DataFrame({
        "NIFTY_Close": nifty["Close"],
        "VIX": vix_aligned["VIX"]
    }, index=dates)
    
    return df, n_days, dates

# Black-Scholes for Straddle Pricing
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

# Generate synthetic features
def generate_features(df, n_days, dates):
    risk_free_rate = 0.06
    strike_step = 100
    
    event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
    df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
    df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)
    
    def dynamic_ivp(x):
        if len(x) >= 5:
            return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
        return 50.0
    df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp).interpolate().fillna(50.0)
    
    market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
    df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
    df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
    df["Spot_MaxPain_Diff_Pct"] = np.clip(np.abs(np.random.lognormal(-2, 0.5, n_days)), 0.1, 1.0)
    df["Days_to_Expiry"] = np.random.choice([1, 3, 7, 14, 21, 28], n_days)
    df["Event_Flag"] = np.where((df.index.month % 3 == 0) & (df.index.day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
    
    fii_trend = np.random.normal(0, 10000, n_days)
    fii_trend[::30] *= -1
    df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
    df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
    
    df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)
    df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
    df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"]).clip(0, 50)
    df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)
    df["Capital_Pressure_Index"] = np.clip((df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3, -2, 2)
    df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30, -2, 2)
    df["Total_Capital"] = capital
    df["PnL_Day"] = np.random.normal(0, 5000, n_days) * (1 - df["Event_Flag"] * 0.5)
    
    straddle_prices = []
    for i in range(n_days):
        S = df["NIFTY_Close"].iloc[i]
        K = round(S / strike_step) * strike_step
        T = df["Days_to_Expiry"].iloc[i] / 365
        sigma = df["ATM_IV"].iloc[i] / 100
        call_price = black_scholes(S, K, T, risk_free_rate, sigma, "call")
        put_price = black_scholes(S, K, T, risk_free_rate, sigma, "put")
        straddle_price = np.clip((call_price + put_price) * (S / 1000), 50, 400)
        straddle_prices.append(straddle_price)
    df["Straddle_Price"] = straddle_prices
    
    df = df.interpolate().fillna(method='bfill')
    return df

# Volatility forecasting
def forecast_volatility(df, forecast_horizon, risk_tolerance, capital):
    df['Log_Returns'] = np.log(df['NIFTY_Close'] / df['NIFTY_Close'].shift(1)).dropna()
    returns = df['Log_Returns'].dropna()
    
    # GARCH
    garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
    garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
    if df["Event_Flag"].iloc[-1] == 1:
        garch_vols *= 1.1
    
    # XGBoost
    df['Target_Vol'] = df['Realized_Vol'].shift(-1)
    df = df.dropna()
    feature_cols = [
        'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
        'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
        'FII_Option_Pos', 'Advance_Decline_Ratio', 'Capital_Pressure_Index', 'Gamma_Bias'
    ]
    X = df[feature_cols]
    y = df['Target_Vol']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    
    split_index = int(len(X) * 0.8)
    X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    xgb_vols = []
    current_row = X.iloc[-1].copy()
    current_row_df = pd.DataFrame([current_row], columns=feature_cols)
    current_row_scaled = scaler.transform(current_row_df)
    for _ in range(forecast_horizon):
        next_vol = model.predict(current_row_scaled)[0]
        xgb_vols.append(next_vol)
        current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
        current_row["VIX"] *= np.random.uniform(0.98, 1.02)
        current_row["Straddle_Price"] *= np.random.uniform(0.98, 1.02)
        current_row_df = pd.DataFrame([current_row], columns=feature_cols)
        current_row_scaled = scaler.transform(current_row_df)
    
    if df["Event_Flag"].iloc[-1] == 1:
        xgb_vols = [v * 1.1 for v in xgb_vols]
    
    # Blend forecasts
    realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()
    garch_diff = np.abs(garch_vols[0] - realized_vol)
    xgb_diff = np.abs(xgb_vols[0] - realized_vol)
    garch_weight = xgb_diff / (garch_diff + xgb_diff) if (garch_diff + xgb_diff) > 0 else 0.5
    if risk_tolerance == "Conservative":
        garch_weight = 0.6
    elif risk_tolerance == "Aggressive":
        garch_weight = 0.4
    xgb_weight = 1 - garch_weight
    blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
    
    forecast_log = pd.DataFrame({
        "Date": future_dates,
        "GARCH_Vol": garch_vols,
        "XGBoost_Vol": xgb_vols,
        "Blended_Vol": blended_vols
    })
    
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return forecast_log, blended_vols, realized_vol, feature_importance, rmse

# Strategy engine
def strategy_engine(df, blended_vols, capital):
    latest = df.iloc[-1]
    avg_vol = np.mean(blended_vols)
    iv = latest["ATM_IV"]
    hv = latest["Realized_Vol"]
    iv_hv_gap = iv - hv
    iv_skew = latest["IV_Skew"]
    pcr = latest["PCR"]
    dte = latest["Days_to_Expiry"]
    event_flag = latest["Event_Flag"]
    
    # Regime Classification
    if event_flag == 1:
        regime = "EVENT-DRIVEN"
    elif avg_vol < 15:
        regime = "LOW"
    elif avg_vol < 20:
        regime = "MEDIUM"
    else:
        regime = "HIGH"
    
    # Strategy Selector
    strategy = "Undefined"
    reason = "N/A"
    tags = []
    confidence_score = 0.5  # Simplified
    risk_reward = 1.5 if iv_hv_gap > 5 else 1.0
    
    if regime == "LOW":
        if iv_hv_gap > 5 and dte < 10:
            strategy = "Butterfly Spread"
            reason = "Low vol & short expiry favors pinning strategies."
            tags = ["Neutral", "Theta", "Expiry Play"]
            risk_reward = 2.0
        else:
            strategy = "Iron Fly"
            reason = "Low volatility and time decay favors delta-neutral Iron Fly."
            tags = ["Neutral", "Theta", "Range Bound"]
    
    elif regime == "MEDIUM":
        if iv_hv_gap > 3 and iv_skew > 2:
            strategy = "Calendar Spread"
            reason = "IV skew and medium vol suggest potential vol expansion."
            tags = ["Volatility Play", "Directional", "Skew"]
            risk_reward = 1.8
        else:
            strategy = "Short Strangle"
            reason = "Balanced vol, no events, and premium-rich environment."
            tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
    
    elif regime == "HIGH":
        if iv_hv_gap > 10:
            strategy = "Jade Lizard"
            reason = "High IV + call skew = Jade Lizard for defined upside risk."
            tags = ["Skewed", "Volatility", "Defined Risk"]
            risk_reward = 1.2
        else:
            strategy = "Debit Spread"
            reason = "High vol implies limited premium edge. Go directional."
            tags = ["Directional", "Volatility Hedge", "Defined Risk"]
    
    elif regime == "EVENT-DRIVEN":
        if iv > 30 and dte < 5:
            strategy = "Calendar Spread"
            reason = "Event + near expiry + IV spike → term structure opportunity."
            tags = ["Volatility", "Event", "Calendar"]
            risk_reward = 1.5
        else:
            strategy = "Straddle Buy"
            reason = "Event-based uncertainty. Straddle captures large moves."
            tags = ["High Gamma", "Event", "Directional Bias"]
            risk_reward = 1.3
    
    # Capital Allocation
    capital_alloc = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
    deploy = capital * capital_alloc.get(regime, 0.2)
    max_loss = deploy * 0.2
    total_exposure = deploy / capital
    
    # Risk Filters
    risk_flags = []
    if regime in ["HIGH", "EVENT-DRIVEN"] and strategy in ["Short Strangle", "Iron Fly"]:
        risk_flags.append("⚠️ No naked legs allowed in HIGH/EVENT-DRIVEN regimes")
    if latest["PnL_Day"] < -0.03 * capital:
        risk_flags.append("⚠️ Daily drawdown exceeds 3%")
    if latest["VIX_Change_Pct"] > 10:
        risk_flags.append("⚠️ High VIX spike detected")
    
    # Behavioral Monitoring
    behavior_score = 8 if deploy < 0.5 * capital else 6
    behavior_warnings = ["Consider reducing position size"] if behavior_score < 7 else []
    
    return regime, strategy, reason, tags, confidence_score, risk_reward, deploy, max_loss, risk_flags, behavior_score, behavior_warnings

# Main dashboard
if run_button:
    with st.spinner("Running VolGuard calculations..."):
        df, n_days, dates = load_data()
        if df is None:
            st.stop()
        
        st.success("✅ Data loaded successfully")
        df = generate_features(df, n_days, dates)
        forecast_log, blended_vols, realized_vol, feature_importance, rmse = forecast_volatility(df, forecast_horizon, risk_tolerance, capital)
        regime, strategy, reason, tags, confidence_score, risk_reward, deploy, max_loss, risk_flags, behavior_score, behavior_warnings = strategy_engine(df, blended_vols, capital)
        
        st.session_state.df = df
        st.session_state.forecast_log = forecast_log
        st.session_state.feature_stats = df[[
            "NIFTY_Close", "VIX", "ATM_IV", "IVP", "PCR", "VIX_Change_Pct",
            "Spot_MaxPain_Diff_Pct", "Straddle_Price", "IV_Skew", "Realized_Vol",
            "Days_to_Expiry", "Event_Flag", "FII_Index_Fut_Pos", "FII_Option_Pos",
            "Advance_Decline_Ratio", "Capital_Pressure_Index", "Gamma_Bias"
        ]].describe()
        
        # Volatility Forecast Panel
        st.subheader("📈 Volatility Forecast")
        forecast_df = pd.DataFrame({
            "Date": forecast_log["Date"].dt.strftime("%d-%b-%Y"),
            "GARCH (%)": [f"{v:.2f}" for v in forecast_log["GARCH_Vol"]],
            "XGBoost (%)": [f"{v:.2f}" for v in forecast_log["XGBoost_Vol"]],
            "Blended (%)": [f"{v:.2f}" for v in forecast_log["Blended_Vol"]]
        })
        st.dataframe(forecast_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Forecasted Volatility", f"{np.mean(blended_vols):.2f}%")
        with col2:
            st.metric("Recent Realized Volatility (5-day)", f"{realized_vol:.2f}%")
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(forecast_log["Date"], forecast_log["GARCH_Vol"], marker='o', linestyle='--', label="GARCH")
        ax.plot(forecast_log["Date"], forecast_log["XGBoost_Vol"], marker='o', linestyle='--', label="XGBoost")
        ax.plot(forecast_log["Date"], forecast_log["Blended_Vol"], marker='o', linestyle='-', label="Blended")
        ax.set_title("Volatility Forecast")
        ax.set_ylabel("Volatility (%)")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Regime & Strategy Panel
        st.subheader("🎯 Regime & Strategy")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Volatility Regime", regime, f"Avg Vol: {np.mean(blended_vols):.2f}%")
        with col2:
            st.metric("Suggested Strategy", strategy)
        
        st.write(f"**Reason**: {reason}")
        st.write(f"**Tags**: {', '.join(tags)}")
        st.write(f"**Confidence Score**: {confidence_score:.2f}")
        st.write(f"**Risk-Reward Expectation**: {risk_reward:.2f}:1")
        st.write(f"**Capital to Deploy**: ₹{deploy:,.0f}")
        st.write(f"**Max Risk Allowed**: ₹{max_loss:,.0f}")
        
        # Risk & Behavioral Panel
        st.subheader("⚠️ Risk & Behavioral Monitoring")
        if risk_flags:
            st.markdown("**Risk Flags**:")
            for flag in risk_flags:
                st.markdown(f"<p class='warning'>{flag}</p>", unsafe_allow_html=True)
        else:
            st.write("No risk flags detected.")
        
        st.metric("Behavior Score", f"{behavior_score}/10")
        if behavior_warnings:
            st.markdown("**Behavioral Warnings**:")
            for warning in behavior_warnings:
                st.markdown(f"<p class='warning'>{warning}</p>", unsafe_allow_html=True)
        
        st.text_area("Journaling Prompt", "Log your trade rationale here...", height=100)
        
        # Feature Statistics Panel
        with st.expander("📊 Feature Statistics"):
            st.dataframe(st.session_state.feature_stats, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df.index, df["VIX"], label="VIX", color="blue")
            ax.plot(df.index, df["ATM_IV"], label="ATM IV", color="green")
            ax.plot(df.index, df["IVP"], label="IVP", color="red")
            ax.plot(df.index, df["Straddle_Price"] / 10, label="Straddle Price (scaled)", color="orange")
            ax.set_title("Key Feature Trends")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        
        # XGBoost Feature Importance
        st.subheader("🔍 XGBoost Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(feature_importance["Feature"], feature_importance["Importance"])
        ax.set_title("Feature Importance")
        ax.set_ylabel("Importance")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(f"XGBoost Test RMSE: {rmse:.2f}%")
        
        # Download buttons
        csv = st.session_state.forecast_log.to_csv(index=False)
        st.download_button("Download Forecast Log", csv, "volguard_forecast_log.csv", "text/csv")
        
        options_csv = st.session_state.df.to_csv()
        st.download_button("Download Options Data", options_csv, "volguard_options_data.csv", "text/csv")
