import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import requests
import io
import warnings
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Page config
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# Custom CSS for modern UI
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #1a1a2e, #0f1c2e);
            color: #e5e5e5;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background: linear-gradient(90deg, #0f3460, #16213e);
            color: white;
            border-radius: 20px;
            padding: 12px 25px;
            font-size: 18px;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        }
        .stMetric {
            background: linear-gradient(135deg, rgba(22, 33, 62, 0.9), rgba(15, 52, 96, 0.7));
            border-radius: 20px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s;
        }
        .stMetric:hover {
            transform: translateY(-5px);
        }
        h1 {
            color: #e94560;
            font-size: 42px;
            text-align: center;
            text-shadow: 0 0 10px rgba(233, 69, 96, 0.5);
            animation: fadeIn 1s;
        }
        h2 {
            color: #00d4ff;
            font-size: 26px;
            margin-bottom: 15px;
            text-shadow: 0 0 5px rgba(0, 212, 255, 0.5);
            animation: slideIn 0.8s;
        }
        .card {
            background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9));
            border-radius: 20px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            animation: popIn 0.6s;
            backdrop-filter: blur(5px);
            transition: transform 0.3s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .gauge {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%);
            display: inline-block;
            text-align: center;
            line-height: 120px;
            color: white;
            font-weight: bold;
            font-size: 20px;
            box-shadow: 0 0 15px rgba(233, 69, 96, 0.5);
            animation: rotateIn 1s;
        }
        .progress-bar {
            background: #16213e;
            border-radius: 10px;
            height: 20px;
            width: 100%;
            overflow: hidden;
            box-shadow: 0 0 5px rgba(0, 212, 255, 0.3);
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #e94560, #ffcc00);
            transition: width 1s ease-in-out;
        }
        .regime-badge {
            display: inline-block;
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
        }
        .regime-low { background: #28a745; color: white; }
        .regime-medium { background: #ffc107; color: black; }
        .regime-high { background: #dc3545; color: white; }
        .regime-event { background: #ff6f61; color: white; }
        .signal-box {
            background: rgba(15, 52, 96, 0.9);
            border-radius: 15px;
            padding: 15px;
            margin: 10px 0;
            transition: transform 0.3s;
            animation: fadeInUp 0.8s;
        }
        .signal-box:hover {
            transform: translateY(-5px);
        }
        .risk-flag {
            color: #e94560;
            font-size: 20px;
            animation: pulse 1.5s infinite;
        }
        .sidebar .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #a0a0a0;
            font-size: 14px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 30px;
        }
        @keyframes fadeIn {
            from {opacity: 0;} to {opacity: 1;}
        }
        @keyframes slideIn {
            from {transform: translateX(-100%);} to {transform: translateX(0);}
        }
        @keyframes popIn {
            from {transform: scale(0); opacity: 0;} to {transform: scale(1); opacity: 1;}
        }
        @keyframes rotateIn {
            from {transform: rotate(-180deg); opacity: 0;} to {transform: rotate(0); opacity: 1;}
        }
        @keyframes fadeInUp {
            from {opacity: 0; transform: translateY(20px);} to {opacity: 1; transform: translateY(0);}
        }
        @keyframes pulse {
            0% {transform: scale(1);} 50% {transform: scale(1.1);} 100% {transform: scale(1);}
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🛡️ VolGuard: Your AI Trading Copilot")
st.markdown("**Protection First, Edge Always** 
| Built by Shritish Shukla & Salman Azim")

# Initialize session state
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Trading Controls")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7, key="horizon_slider")
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000, key="capital_input")
    risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1, key="risk_select")
    st.markdown("**Backtest Parameters**")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-10-21"), key="start_date")
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-29"), key="end_date")
    strategy_choice = st.selectbox("Strategy", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Debit Spread", "Straddle Buy"], key="strategy_select")
    run_button = st.button("Activate VolGuard", key="run_button")
    st.markdown("---")
    st.markdown("**Motto:** Deploy with edge, survive, outlast.")

# Function to load data
@st.cache_data
def load_data():
    try:
        logger.debug("Fetching NIFTY 50 data...")
        nifty = yf.download("^NSEI", period="1y", interval="1d")
        if nifty.empty or len(nifty) < 200:
            st.error("Failed to fetch sufficient NIFTY 50 data from Yahoo Finance.")
            return None
        nifty = nifty[["Close"]].rename(columns={"Close": "NIFTY_Close"})
        nifty.index = pd.to_datetime(nifty.index)
        nifty = nifty[~nifty.index.duplicated(keep='first')]
        nifty_series = nifty["NIFTY_Close"].squeeze()

        logger.debug("Fetching India VIX data...")
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
        try:
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch India VIX data: {str(e)}")
            return pd.DataFrame({"NIFTY_Close": nifty_series}, index=nifty.index)

        vix.columns = vix.columns.str.strip()
        if "Date" not in vix.columns or "Close" not in vix.columns:
            st.error("india_vix.csv is missing required columns.")
            return pd.DataFrame({"NIFTY_Close": nifty_series}, index=nifty.index)
        vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
        vix = vix.dropna(subset=["Date"])
        if vix.empty:
            st.error("VIX data is empty.")
            return pd.DataFrame({"NIFTY_Close": nifty_series}, index=nifty.index)
        vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
        vix.index = pd.to_datetime(vix.index)
        vix = vix[~vix.index.duplicated(keep='first')]
        vix_series = vix["VIX"].squeeze()

        common_dates = nifty_series.index.intersection(vix_series.index)
        if len(common_dates) < 200:
            st.error(f"Insufficient overlapping dates: {len(common_dates)} found.")
            return pd.DataFrame({"NIFTY_Close": nifty_series}, index=nifty.index)
        
        df = pd.DataFrame({
            "NIFTY_Close": nifty_series.loc[common_dates],
            "VIX": vix_series.loc[common_dates]
        }, index=common_dates)
        
        if df["NIFTY_Close"].isna().sum() > 0 or df["VIX"].isna().sum() > 0:
            df = df.ffill().bfill()
        if df.empty:
            st.error("DataFrame is empty after processing.")
            return None

        logger.debug("Data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None

# Function to generate synthetic options features
@st.cache_data
def generate_synthetic_features(df, capital):
    try:
        n_days = len(df)
        np.random.seed(42)
        risk_free_rate = 0.06
        strike_step = 100

        def calculate_days_to_expiry(dates):
            days_to_expiry = []
            for date in dates:
                days_ahead = (3 - date.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                next_expiry = date + pd.Timedelta(days=days_ahead)
                dte = (next_expiry - date).days
                days_to_expiry.append(dte)
            return np.array(days_to_expiry)

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

        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
        df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
        df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)

        def dynamic_ivp(x):
            if len(x) >= 5:
                return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
            return 50.0
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp)
        df["IVP"] = df["IVP"].interpolate().fillna(50.0)

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2, 0.5, n_days))
        df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 1.0)
        df["Event_Flag"] = np.where((df.index.month % 3 == 0) & (df.index.day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
        fii_trend = np.random.normal(0, 10000, n_days)
        fii_trend[::30] *= -1
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
        df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)
        df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"])
        df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)
        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)
        df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3
        df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -2, 2)
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
            straddle_price = (call_price + put_price) * (S / 1000)
            straddle_price = np.clip(straddle_price, 50, 400)
            straddle_prices.append(straddle_price)
        df["Straddle_Price"] = straddle_prices

        if df.isna().sum().sum() > 0:
            df = df.interpolate().fillna(method='bfill')
        
        df.to_csv("volguard_options_data.csv")
        logger.debug("Synthetic features generated successfully.")
        return df
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
        logger.error(f"Error generating features: {str(e)}")
        return None

# Function to forecast volatility
@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    try:
        df.index = pd.to_datetime(df.index)
        df_garch = df.tail(len(df))
        if len(df_garch) < 200:
            st.error(f"Insufficient data for GARCH: {len(df_garch)} days.")
            return None, None, None, None, None, None, None, None

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).dropna() * 100
        garch_model = arch_model(df_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
        garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)

        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()

        df_xgb = df.tail(len(df))
        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna()

        feature_cols = [
            'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
            'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
            'FII_Option_Pos'
        ]
        X = df_xgb[feature_cols]
        y = df_xgb['Target_Vol']

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
        current_row = df_xgb[feature_cols].iloc[-1].copy()
        for i in range(forecast_horizon):
            current_row_df = pd.DataFrame([current_row], columns=feature_cols)
            current_row_scaled = scaler.transform(current_row_df)
            next_vol = model.predict(current_row_scaled)[0]
            xgb_vols.append(next_vol)

            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] *= np.random.uniform(0.98, 1.02)
            current_row["Straddle_Price"] *= np.random.uniform(0.98, 1.02)
            current_row["VIX_Change_Pct"] = (current_row["VIX"] / df_xgb["VIX"].iloc[-1] - 1) * 100
            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.1))
            current_row["Realized_Vol"] = np.clip(next_vol * np.random.uniform(0.95, 1.05), 5, 50)
            current_row["IVP"] = current_row["IVP"] * np.random.uniform(0.99, 1.01)
            current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.05), 0.7, 2.0)
            current_row["Spot_MaxPain_Diff_Pct"] = np.clip(current_row["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0.95, 1.05), 0.1, 1.0)
            current_row["Event_Flag"] = df_xgb["Event_Flag"].iloc[-1]
            current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 1000)
            current_row["FII_Option_Pos"] += np.random.normal(0, 500)
            current_row["IV_Skew"] = np.clip(current_row["IV_Skew"] + np.random.normal(0, 0.1), -3, 3)

        xgb_vols = np.clip(xgb_vols, 5, 50)
        if df["Event_Flag"].iloc[-1] == 1:
            xgb_vols = [v * 1.1 for v in xgb_vols]

        garch_diff = np.abs(garch_vols[0] - realized_vol)
        xgb_diff = np.abs(xgb_vols[0] - realized_vol)
        garch_weight = xgb_diff / (garch_diff + xgb_diff) if (garch_diff + xgb_diff) > 0 else 0.5
        xgb_weight = 1 - garch_weight
        blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]
        confidence_score = min(100, max(50, 80 - abs(garch_diff - xgb_diff)))

        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * forecast_horizon
        })
        logger.debug("Volatility forecast completed.")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        st.error(f"Error in volatility forecasting: {str(e)}")
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None

# Function to generate trading strategy
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score):
    try:
        latest = df.iloc[-1]
        avg_vol = np.mean(forecast_log["Blended_Vol"])
        iv = latest["ATM_IV"]
        hv = latest["Realized_Vol"]
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"]
        pcr = latest["PCR"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]
        capital = latest["Total_Capital"]

        if event_flag == 1:
            regime = "EVENT-DRIVEN"
        elif avg_vol < 15:
            regime = "LOW"
        elif avg_vol < 20:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0
        confidence = 0.5 + 0.5 * (1 - np.abs(forecast_log["GARCH_Vol"].iloc[0] - forecast_log["XGBoost_Vol"].iloc[0]) / max(forecast_log["GARCH_Vol"].iloc[0], forecast_log["XGBoost_Vol"].iloc[0]))

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

        capital_alloc = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc.get(regime, 0.2) * position_size
        max_loss = deploy * 0.2
        total_exposure = deploy / capital

        risk_flags = []
        if regime in ["HIGH", "EVENT-DRIVEN"] and strategy in ["Short Strangle", "Iron Fly"]:
            risk_flags.append("No naked legs allowed in HIGH/EVENT-DRIVEN regimes")
        if latest["PnL_Day"] < -0.03 * capital:
            risk_flags.append("Daily drawdown exceeds 3%")
        if latest["VIX_Change_Pct"] > 10:
            risk_flags.append("High VIX spike detected")
        if total_exposure > 0.7:
            risk_flags.append("Total exposure exceeds 70%")

        behavior_score = 8 if deploy < 0.5 * capital else 6
        behavior_warnings = ["Consider reducing position size"] if behavior_score < 7 else []

        logger.debug("Trading strategy generated successfully.")
        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence,
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure,
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings
        }
    except Exception as e:
        st.error(f"Error generating strategy: {str(e)}")
        logger.error(f"Error generating strategy: {str(e)}")
        return None

# Function for backtesting
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.debug("Starting backtest...")
        backtest_results = []
        lot_size = 25
        base_transaction_cost = 0.002
        stt = 0.0005
        portfolio_pnl = 0
        risk_free_rate = 0.06 / 126
        nifty_returns = df["NIFTY_Close"].pct_change()

        def run_strategy_engine(day_data, avg_vol, portfolio_pnl):
            try:
                iv = day_data["ATM_IV"]
                hv = day_data["Realized_Vol"]
                iv_hv_gap = iv - hv
                iv_skew = day_data["IV_Skew"]
                dte = day_data["Days_to_Expiry"]
                event_flag = day_data["Event_Flag"]

                if portfolio_pnl < -0.1 * day_data["Total_Capital"]:
                    return None, None, "Portfolio drawdown limit reached", [], 0, 0, 0

                if event_flag == 1:
                    regime = "EVENT-DRIVEN"
                elif avg_vol < 15:
                    regime = "LOW"
                elif avg_vol < 20:
                    regime = "MEDIUM"
                else:
                    regime = "HIGH"

                strategy = "Undefined"
                reason = "N/A"
                tags = []
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
                        strategy = "Iron Condor"
                        reason = "Medium vol and skew favor wide-range Iron Condor."
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward = 1.8
                    else:
                        strategy = "Short Strangle"
                        reason = "Balanced vol, premium-rich environment for Short Strangle."
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

                capital = day_data["Total_Capital"]
                capital_alloc = {"LOW": 0.08, "MEDIUM": 0.06, "HIGH": 0.04, "EVENT-DRIVEN": 0.04}
                deploy = capital * capital_alloc.get(regime, 0.04)
                max_loss = deploy * 0.025
                return regime, strategy, reason, tags, deploy, max_loss, risk_reward
            except Exception as e:
                logger.error(f"Error in strategy engine: {str(e)}")
                return None, None, "Strategy engine failed", [], 0, 0, 0

        def get_dynamic_slippage(strategy, iv, dte):
            base_slippage = 0.005
            iv_multiplier = min(iv / 20, 2.5)
            dte_factor = 1.5 if dte < 5 else 1.0
            if strategy == "Iron Condor":
                return base_slippage * 1.8 * iv_multiplier * dte_factor
            elif strategy == "Butterfly Spread":
                return base_slippage * 2.2 * iv_multiplier * dte_factor
            elif strategy == "Iron Fly":
                return base_slippage * 1.5 * iv_multiplier * dte_factor
            elif strategy in ["Calendar Spread", "Jade Lizard", "Debit Spread", "Straddle Buy"]:
                return base_slippage * 1.2 * iv_multiplier * dte_factor
            return base_slippage * iv_multiplier * dte_factor

        def apply_volatility_shock(pnl, nifty_move, iv, event_flag):
            shock_prob = 0.35 if event_flag == 1 else 0.20
            if np.random.rand() < shock_prob:
                shock_factor = nifty_move / (iv * 100) if iv != 0 else 1.0
                shock = -abs(pnl) * min(shock_factor * 1.5, 2.0)
                return shock
            return pnl

        def apply_liquidity_discount(premium):
            if np.random.rand() < 0.05:
                return premium * 0.8
            return premium

        def apply_execution_delay(premium):
            if np.random.rand() < 0.10:
                return premium * 0.9
            return premium

        for i in range(1, len(df)):
            try:
                day_data = df.iloc[i]
                prev_day = df.iloc[i-1]
                date = day_data.name
                avg_vol = df["Realized_Vol"].iloc[max(0, i-5):i].mean()

                regime, strategy, reason, tags, deploy, max_loss, risk_reward = run_strategy_engine(day_data, avg_vol, portfolio_pnl)
                
                if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy_choice):
                    continue

                extra_cost = 0.001 if "Iron" in strategy else 0
                total_cost = base_transaction_cost + extra_cost + stt
                slippage = get_dynamic_slippage(strategy, day_data["ATM_IV"], day_data["Days_to_Expiry"])
                entry_price = day_data["Straddle_Price"]
                lots = int(deploy / (entry_price * lot_size))
                lots = max(1, min(lots, 2))

                decay_factor = max(0.75, 1 - day_data["Days_to_Expiry"] / 10)
                premium = entry_price * lot_size * lots * (1 - slippage - total_cost) * decay_factor
                premium = apply_liquidity_discount(premium)
                premium = apply_execution_delay(premium)

                iv_factor = min(day_data["ATM_IV"] / avg_vol, 1.5) if avg_vol != 0 else 1.0
                breakeven_factor = 0.04 if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) else 0.06
                breakeven = entry_price * (1 + iv_factor * breakeven_factor)
                nifty_move = abs(day_data["NIFTY_Close"] - prev_day["NIFTY_Close"])
                loss = max(0, nifty_move - breakeven) * lot_size * lots

                if strategy in ["Short Strangle", "Iron Fly", "Iron Condor"]:
                    max_strategy_loss = premium * 0.6 if strategy in ["Iron Fly", "Iron Condor"] else premium * 0.8
                    loss = min(loss, max_strategy_loss)
                    pnl = premium - loss
                elif strategy in ["Calendar Spread", "Straddle Buy"]:
                    max_strategy_loss = premium * 0.7
                    loss = min(loss, max_strategy_loss)
                    pnl = premium - loss if nifty_move > breakeven else -loss
                else:
                    payoff = premium if nifty_move <= breakeven else 0
                    loss = min(loss, premium * 0.35)
                    pnl = payoff - loss

                pnl = apply_volatility_shock(pnl, nifty_move, day_data["ATM_IV"], day_data["Event_Flag"])
                if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) and np.random.rand() < 0.08:
                    gap_loss = premium * np.random.uniform(0.5, 1.0)
                    pnl -= gap_loss
                if np.random.rand() < 0.02:
                    crash_loss = premium * np.random.uniform(1.0, 1.5)
                    pnl -= crash_loss

                pnl = max(-max_loss, min(pnl, max_loss * 1.5))
                portfolio_pnl += pnl

                backtest_results.append({
                    "Date": date,
                    "Regime": regime,
                    "Strategy": strategy,
                    "PnL": pnl,
                    "Capital_Deployed": deploy,
                    "Max_Loss": max_loss,
                    "Risk_Reward": risk_reward
                })
            except Exception as e:
                logger.error(f"Error in backtest loop at index {i}: {str(e)}")
                continue

        backtest_df = pd.DataFrame(backtest_results)
        if len(backtest_df) == 0:
            logger.warning("No trades generated in backtest.")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max() if len(backtest_df) > 0 else 0

        backtest_df.set_index("Date", inplace=True)
        returns = backtest_df["PnL"] / df["Total_Capital"].reindex(backtest_df.index, method="ffill")
        nifty_returns = df["NIFTY_Close"].pct_change().reindex(backtest_df.index, method="ffill").fillna(0)
        excess_returns = returns - nifty_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(126) if excess_returns.std() != 0 else 0
        sortino_ratio = excess_returns.mean() / excess_returns[excess_returns < 0].std() * np.sqrt(126) if len(excess_returns[excess_returns < 0]) > 0 and excess_returns[excess_returns < 0].std() != 0 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 else 0

        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_perf["Win_Rate"] = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_perf["Win_Rate"] = backtest_df.groupby("Regime")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)

        logger.debug("Backtest completed successfully.")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

# Main execution
if run_button:
    with st.spinner("Initializing AI Copilot..."):
        df = load_data()
        if df is not None:
            df = generate_synthetic_features(df, capital)
            if df is not None:
                # Display Latest Market Data
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📊 Latest Market Snapshot")
                last_date = df.index[-1].strftime("%d-%b-%Y") if not df.empty and pd.notna(df.index[-1]) else datetime.now().strftime("%d-%b-%Y")
                last_nifty = df["NIFTY_Close"].iloc[-1] if "NIFTY_Close" in df.columns and not df["NIFTY_Close"].isna().iloc[-1] else "N/A"
                prev_nifty = df["NIFTY_Close"].iloc[-2] if "NIFTY_Close" in df.columns and len(df) >= 2 and not df["NIFTY_Close"].isna().iloc[-2] else "N/A"
                last_vix = df["VIX"].iloc[-1] if "VIX" in df.columns and not df["VIX"].isna().iloc[-1] else "N/A"
                prev_vix = df["VIX"].iloc[-2] if "VIX" in df.columns and len(df) >= 2 and not df["VIX"].isna().iloc[-2] else "N/A"
                col1, col2 = st.columns(2)
                with col1:
                    nifty_change = last_nifty - prev_nifty if last_nifty != "N/A" and prev_nifty != "N/A" else "N/A"
                    nifty_change_display = f"{nifty_change:+,.2f}" if nifty_change != "N/A" else "N/A"
                    st.metric(
                        label="NIFTY 50 Last Close",
                        value=f"{last_nifty:,.2f}" if last_nifty != "N/A" else "N/A",
                        delta=nifty_change_display,
                        delta_color="normal",
                        help="Latest NIFTY 50 closing price"
                    )
                with col2:
                    vix_change = last_vix - prev_vix if last_vix != "N/A" and prev_vix != "N/A" else "N/A"
                    vix_change_display = f"{vix_change:+.2f}" if vix_change != "N/A" else "N/A"
                    st.metric(
                        label="India VIX",
                        value=f"{last_vix:.2f}%" if last_vix != "N/A" else "N/A",
                        delta=vix_change_display,
                        delta_color="normal",
                        help="Latest India VIX value"
                    )
                st.markdown(f"**Last Updated**: {last_date}")
                st.markdown('</div>', unsafe_allow_html=True)

                with st.spinner("Predicting market volatility..."):
                    forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df, forecast_horizon)

                if forecast_log is not None:
                    # Volatility Forecast Card
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📈 Volatility Forecast")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Blended Volatility", f"{np.mean(blended_vols):.2f}%", help="Weighted average of GARCH and XGBoost forecasts")
                    with col2:
                        st.metric("Realized Volatility (5-day)", f"{realized_vol:.2f}%", help="Historical volatility over the last 5 days")
                    with col3:
                        st.metric("Model Accuracy (RMSE)", f"{rmse:.2f}%", help="XGBoost model's root mean squared error")
                        st.markdown(f'<div class="gauge">{int(confidence_score)}%</div>', unsafe_allow_html=True)
                        st.markdown("**Confidence Score**", unsafe_allow_html=True)
                        st.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {confidence_score}%"></div></div>', unsafe_allow_html=True)

                    # Forecast Chart
                    st.markdown("### Volatility Forecast Trend")
                    chart_data = pd.DataFrame({
                        "Date": forecast_log["Date"],
                        "GARCH": garch_vols,
                        "XGBoost": xgb_vols,
                        "Blended": blended_vols
                    }).set_index("Date")
                    st.line_chart(chart_data, color=["#e94560", "#00d4ff", "#ffcc00"], use_container_width=True)

                    # Daily Breakdown
                    st.markdown("### Daily Volatility Breakdown")
                    for i in range(forecast_horizon):
                        date = forecast_log["Date"].iloc[i].strftime("%d-%b-%Y")
                        st.markdown(f'<div class="signal-box">📡 {date} | GARCH: {garch_vols[i]:.2f}% | XGBoost: {xgb_vols[i]:.2f}% | Blended: {blended_vols[i]:.2f}%</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Feature Importance
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("🔍 Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': [
                            'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
                            'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
                            'FII_Option_Pos'
                        ],
                        'Importance': feature_importances
                    }).sort_values(by='Importance', ascending=False)
                    st.dataframe(feature_importance, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Trading Strategy Card
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("🎯 Trading Strategy")
                    strategy = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score)
                    if strategy is not None:
                        regime_class = {
                            "LOW": "regime-low",
                            "MEDIUM": "regime-medium",
                            "HIGH": "regime-high",
                            "EVENT-DRIVEN": "regime-event"
                        }.get(strategy["Regime"], "regime-low")
                        st.markdown(f"""
                            **Volatility Regime**: <span class="regime-badge {regime_class}">{strategy["Regime"]}</span> (Avg Vol: {np.mean(blended_vols):.2f}%)  
                            **Suggested Strategy**: {strategy["Strategy"]}  
                            **Reason**: {strategy["Reason"]}  
                            **Tags**: {', '.join(strategy["Tags"])}  
                            **Confidence Score**: {strategy["Confidence"]:.2f}  
                            **Risk-Reward Expectation**: {strategy["Risk_Reward"]:.2f}:1  
                            **Capital to Deploy**: ₹{strategy["Deploy"]:,.0f}  
                            **Max Risk Allowed**: ₹{strategy["Max_Loss"]:,.0f}  
                            **Exposure**: {strategy["Exposure"]*100:.2f}%  
                            **Risk Flags**: {', '.join(strategy["Risk_Flags"]) if strategy["Risk_Flags"] else "None"}  
                            **Behavior Score**: {strategy["Behavior_Score"]}/10  
                            **Behavioral Warnings**: {', '.join(strategy["Behavior_Warnings"]) if strategy["Behavior_Warnings"] else "None"}
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Journaling Prompt Card
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📝 Journaling Prompt")
                    journal = st.text_area("Reflect on your discipline today:", height=120, key="journal_input")
                    if st.button("Save Reflection", key="save_button"):
                        st.success("Reflection saved!")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Export Functionality
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📤 Export Insights")
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = forecast_log.to_csv(index=False)
                        st.download_button(
                            label="Download Forecast (CSV)",
                            data=csv,
                            file_name="volguard_forecast.csv",
                            mime="text/csv"
                        )
                    with col2:
                        if strategy is not None:
                            strategy_df = pd.DataFrame([strategy])
                            strategy_csv = strategy_df.to_csv(index=False)
                            st.download_button(
                                label="Download Strategy (CSV)",
                                data=strategy_csv,
                                file_name="volguard_strategy.csv",
                                mime="text/csv"
                            )
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Backtest Section (Last)
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📊 Backtest Performance")
                    st.markdown("Evaluate historical performance of your selected strategy.")
                    if st.button("Run Backtest", key="backtest_button"):
                        st.session_state.backtest_run = True
                        df_backtest = df.loc[start_date:end_date]
                        if len(df_backtest) == 0:
                            st.error("No data available for the selected date range.")
                            st.session_state.backtest_run = False
                        else:
                            with st.spinner("Running backtest..."):
                                backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(df_backtest, capital, strategy_choice, start_date, end_date)
                                st.session_state.backtest_results = (backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf)

                    # Display backtest results if available
                    if st.session_state.backtest_run and st.session_state.backtest_results is not None:
                        backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = st.session_state.backtest_results
                        if len(backtest_df) == 0:
                            st.error("No trades generated. Try a different strategy or date range.")
                        else:
                            st.markdown(f"**Period**: {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}")
                            st.markdown(f"**Strategy Tested**: {strategy_choice}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total PnL", f"₹{total_pnl:,.2f}", help="Total profit and loss")
                                st.metric("Total Return", f"{total_pnl / capital * 100:.2f}%", help="Percentage return on capital")
                            with col2:
                                st.metric("Win Rate", f"{win_rate:.2%}", help="Percentage of winning trades")
                                st.metric("Max Drawdown", f"₹{max_drawdown:,.2f} ({max_drawdown / capital * 100:.2f}%)", help="Maximum loss from peak")
                            with col3:
                                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", help="Risk-adjusted return")
                                st.metric("Sortino Ratio", f"{sortino_ratio:.2f}", help="Downside risk-adjusted return")
                                st.metric("Calmar Ratio", f"{calmar_ratio:.2f}", help="Return per unit of drawdown")

                            st.markdown("### Strategy-wise Performance")
                            st.dataframe(strategy_perf, use_container_width=True)

                            st.markdown("### Regime-wise Performance")
                            st.dataframe(regime_perf, use_container_width=True)

                            st.markdown("### Cumulative PnL")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            backtest_df["PnL"].cumsum().plot(ax=ax, color="#e94560", linewidth=2)
                            ax.set_title("Cumulative PnL Over Time", color="#e5e5e5", fontsize=14)
                            ax.set_xlabel("Date", color="#e5e5e5", fontsize=12)
                            ax.set_ylabel("PnL (₹)", color="#e5e5e5", fontsize=12)
                            ax.grid(True, color="#a0a0a0", linestyle="--", alpha=0.5)
                            ax.set_facecolor("#0f1c2e")
                            fig.set_facecolor("#1a1a2e")
                            ax.tick_params(colors="#e5e5e5")
                            st.pyplot(fig)

                            st.markdown("**Note**: Expected real-world performance is ~50-70% of synthetic results due to execution, liquidity, and unforeseen market events.")

                            st.download_button(
                                label="Download Backtest Results (CSV)",
                                data=backtest_df.reset_index().to_csv(index=False),
                                file_name="backtest_results.csv",
                                mime="text/csv"
                            )
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Footer
                    st.markdown("""
                        <div class="footer">
                            VolGuard: Protection First, Edge Always | Built by Shritish Shukla & Salman Azim<br>
                            "This application is a decision support tool, not a recommender system. Please use the provided information to make informed decisions."
                        </div>
                    """, unsafe_allow_html=True)

else:
    st.info("Set parameters and activate VolGuard to begin your journey.")
    st.warning("⚠️ This application is a decision support tool, not a recommender system. Please use the provided information to make informed decisions.")
