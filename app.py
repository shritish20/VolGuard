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
import warnings
from py5paisa import FivePaisaClient
import os
from dotenv import load_dotenv
import time

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# Custom CSS for an enhanced, modern UI
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

# Initialize session state
if "client" not in st.session_state:
    st.session_state.client = None
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "data_source" not in st.session_state:
    st.session_state.data_source = "Public Data"

# Header
st.title("🛡️ VolGuard: Your AI Trading Copilot")
st.markdown("**Protection First, Edge Always** | Made by Shritish & Salman")

# Sidebar
with st.sidebar:
    st.header("⚙️ Trading Controls")
    page = st.selectbox("Navigate", ["Login", "Dashboard"])
    if page == "Dashboard":
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7, key="horizon_slider")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000, key="capital_input")
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1, key="risk_select")
        run_button = st.button("Activate VolGuard", key="run_button")
    st.markdown("---")
    st.markdown("**Motto:** Deploy with edge, survive, outlast.")

# Login Page
if page == "Login":
    st.subheader("🛡️ VolGuard: Login")
    st.markdown("Connect to 5paisa for live data or use public data.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("5paisa API Login")
        totp_code = st.text_input("TOTP Code", placeholder="6-digit TOTP from Authenticator", type="password")
        if st.button("Login with 5paisa"):
            try:
                # Validate environment variables
                required_keys = ["APP_NAME", "APP_SOURCE", "USER_ID", "PASSWORD", "USER_KEY", "ENCRYPTION_KEY", "CLIENT_CODE", "PIN"]
                missing_keys = [key for key in required_keys if not os.getenv(key)]
                if missing_keys:
                    st.error(f"Missing environment variables: {', '.join(missing_keys)}. Check your .env file.")
                    st.stop()

                cred = {
                    "APP_NAME": os.getenv("APP_NAME"),
                    "APP_SOURCE": os.getenv("APP_SOURCE"),
                    "USER_ID": os.getenv("USER_ID"),
                    "PASSWORD": os.getenv("PASSWORD"),
                    "USER_KEY": os.getenv("USER_KEY"),
                    "ENCRYPTION_KEY": os.getenv("ENCRYPTION_KEY")
                }
                client = FivePaisaClient(cred=cred)
                client_code = os.getenv("CLIENT_CODE")
                pin = os.getenv("PIN")
                
                # Attempt to get TOTP session
                response = client.get_totp_session(client_code, totp_code, pin)
                if response and response.get("Message") == "SUCCESS" and "UserId" in response:
                    st.session_state.client = client
                    st.session_state.logged_in = True
                    st.session_state.data_source = "Live 5paisa Data"
                    st.success("✅ Successfully Logged In!")
                    # Test market feed to confirm connection
                    test_feed = client.fetch_market_feed([{"Exch": "N", "ExchType": "C", "Symbol": "NIFTY 50", "Expiry": "", "StrikePrice": "0", "OptionType": ""}])
                    if test_feed and "Success" in test_feed and test_feed["Success"] and len(test_feed["Success"]) > 0:
                        st.write("✅ Connection confirmed with 5paisa.")
                    else:
                        st.warning("⚠️ Connection established, but test feed failed. Check API limits or market hours.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Login failed: {response.get('Message', 'Invalid response')}. Check TOTP code, credentials, or API status.")
            except Exception as e:
                st.error(f"Login failed: {str(e)}. Check TOTP code, credentials, or API status.")
    with col2:
        st.subheader("No API?")
        if st.button("Use Public Data"):
            st.session_state.logged_in = False
            st.session_state.client = None
            st.session_state.data_source = "Public Data"
            st.success("Using public data.")
            st.rerun()

# Function to load data
@st.cache_data
def load_data():
    try:
        # Load NIFTY data
        if st.session_state.client:
            try:
                market_feed = st.session_state.client.fetch_market_feed([{"Exch": "N", "ExchType": "C", "Symbol": "NIFTY 50", "Expiry": "", "StrikePrice": "0", "OptionType": ""}])
                if market_feed and "Success" in market_feed and market_feed["Success"] and len(market_feed["Success"]) > 0:
                    nifty_price = float(market_feed["Success"][0]["LastRate"])
                    nifty_data = pd.DataFrame({"NIFTY_Close": [nifty_price]}, index=[datetime.now().date()])
                else:
                    st.error("Failed to fetch 5paisa NIFTY price. Falling back to public data.")
                    raise Exception("No valid 5paisa data")
            except Exception as e:
                st.error(f"5paisa data fetch failed: {str(e)}. Falling back to public data.")
                nifty = yf.download("^NSEI", period="1y", interval="1d")
                if nifty.empty or len(nifty) < 200:
                    st.error("Failed to fetch sufficient NIFTY 50 data from Yahoo Finance.")
                    return None
                nifty = nifty[["Close"]].rename(columns={"Close": "NIFTY_Close"})
                nifty.index = pd.to_datetime(nifty.index).date
                nifty = nifty[~nifty.index.duplicated(keep='first')]
                nifty_series = nifty["NIFTY_Close"]
        else:
            nifty = yf.download("^NSEI", period="1y", interval="1d")
            if nifty.empty or len(nifty) < 200:
                st.error("Failed to fetch sufficient NIFTY 50 data from Yahoo Finance.")
                return None
            nifty = nifty[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            nifty.index = pd.to_datetime(nifty.index).date
            nifty = nifty[~nifty.index.duplicated(keep='first')]
            nifty_series = nifty["NIFTY_Close"]

        # Load India VIX data from GitHub
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
        vix = pd.read_csv(vix_url)
        vix.columns = vix.columns.str.strip().str.lower()
        if "date" not in vix.columns or "close" not in vix.columns:
            st.error(f"india_vix.csv is missing required columns. Found columns: {vix.columns.tolist()}")
            return None
        vix["date"] = pd.to_datetime(vix["date"], format="%d-%b-%Y", errors="coerce")
        vix = vix.dropna(subset=["date"])
        if vix.empty:
            st.error("VIX data is empty.")
            return None
        vix = vix[["date", "close"]].set_index("date").rename(columns={"close": "VIX"})
        vix.index = pd.to_datetime(vix.index).date
        vix = vix[~vix.index.duplicated(keep='first')]
        vix_series = vix["VIX"]

        # Align data
        if not st.session_state.client:
            common_dates = nifty_series.index.intersection(vix_series.index)
            if len(common_dates) < 200:
                st.error(f"Insufficient overlapping dates: {len(common_dates)} found.")
                return None
            
            nifty_data = nifty_series.loc[common_dates].to_numpy().flatten()
            vix_data = vix_series.loc[common_dates].to_numpy().flatten()
            df = pd.DataFrame({
                "NIFTY_Close": nifty_data,
                "VIX": vix_data
            }, index=common_dates)
        else:
            df = pd.DataFrame({
                "NIFTY_Close": nifty_data["NIFTY_Close"].iloc[0],
                "VIX": vix_series.iloc[-1]
            }, index=[datetime.now().date()])

        # Handle missing data
        if df["NIFTY_Close"].isna().sum() > 0 or df["VIX"].isna().sum() > 0:
            df = df.ffill().bfill()
        if df.empty:
            st.error("DataFrame is empty after processing.")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to generate synthetic options features
def generate_synthetic_features(df):
    n_days = len(df)
    np.random.seed(42)
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

    # ATM Implied Volatility with event spikes
    event_spike = np.where((pd.to_datetime(df.index).month % 3 == 0) & (pd.to_datetime(df.index).day < 5), 1.2, 1.0)
    df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
    df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)

    # Implied Volatility Percentile
    def dynamic_ivp(x):
        if len(x) >= 5:
            return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
        return 50.0
    df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp)
    df["IVP"] = df["IVP"].interpolate().fillna(50.0)

    # Put-Call Ratio
    market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
    df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
    
    # VIX Change
    df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
    
    # Max Pain Difference
    df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2, 0.5, n_days))
    df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 1.0)
    
    # Days to Expiry
    df["Days_to_Expiry"] = np.random.choice([1, 3, 7, 14, 21, 28], n_days)
    
    # Event Flag
    df["Event_Flag"] = np.where((pd.to_datetime(df.index).month % 3 == 0) & (pd.to_datetime(df.index).day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
    
    # FII Positions
    fii_trend = np.random.normal(0, 10000, n_days)
    fii_trend[::30] *= -1
    df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
    df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
    
    # IV Skew
    df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)
    
    # Realized Volatility
    df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
    df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"])
    df["Realized_Vol"] = np.clip(df["Realized_Vol"], 0, 50)
    
    # Advance/Decline Ratio
    df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.2, n_days) + market_trend * 10, 0.5, 2.0)
    
    # Capital Pressure Index
    df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 3e4 + df["FII_Option_Pos"] / 1e4 + df["PCR"]) / 3
    df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -2, 2)
    
    # Gamma Bias
    df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - df["Days_to_Expiry"]) / 30, -2, 2)
    
    # Capital and PnL
    df["Total_Capital"] = capital
    df["PnL_Day"] = np.random.normal(0, 5000, n_days) * (1 - df["Event_Flag"] * 0.5)
    
    # Options Prices
    straddle_prices = []
    call_prices = []
    put_prices = []
    for i in range(n_days):
        S = df["NIFTY_Close"].iloc[i]
        K = round(S / strike_step) * strike_step
        T = df["Days_to_Expiry"].iloc[i] / 365
        sigma = df["ATM_IV"].iloc[i] / 100
        call_price = black_scholes(S, K, T, risk_free_rate, sigma, "call")
        put_price = black_scholes(S, K, T, risk_free_rate, sigma, "put")
        straddle_price = (call_price + put_price)
        straddle_price = np.clip(straddle_price, 50, 400)
        straddle_prices.append(straddle_price)
        call_prices.append(call_price)
        put_prices.append(put_price)
    df["Straddle_Price"] = straddle_prices
    df["Call_Price"] = call_prices
    df["Put_Price"] = put_prices

    # Handle any remaining NaNs
    if df.isna().sum().sum() > 0:
        df = df.interpolate().fillna(method='bfill')
    
    return df

# Function to forecast volatility
def forecast_volatility_future(df, forecast_horizon):
    df.index = pd.to_datetime(df.index)
    df_garch = df.tail(len(df))
    if len(df_garch) < 200:
        st.error(f"Insufficient data for GARCH: {len(df_garch)} days.")
        return None, None, None, None, None, None, None, None

    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

    # GARCH Model
    df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).dropna() * 100
    garch_model = arch_model(df_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
    garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
    garch_vols = np.clip(garch_vols, 5, 50)

    # Realized Volatility
    realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()

    # XGBoost Model
    df_xgb = df.tail(len(df))
    df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
    df_xgb = df_xgb.dropna()

    feature_cols = [
        'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
        'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
        'FII_Option_Pos', 'Advance_Decline_Ratio', 'Capital_Pressure_Index', 'Gamma_Bias'
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
    try:
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
            current_row["Advance_Decline_Ratio"] = np.clip(current_row["Advance_Decline_Ratio"] + np.random.normal(0, 0.05), 0.5, 2.0)
            current_row["Capital_Pressure_Index"] = (current_row["FII_Index_Fut_Pos"] / 3e4 + current_row["FII_Option_Pos"] / 1e4 + current_row["PCR"]) / 3
            current_row["Capital_Pressure_Index"] = np.clip(current_row["Capital_Pressure_Index"], -2, 2)
            current_row["Gamma_Bias"] = np.clip(current_row["IV_Skew"] * (30 - current_row["Days_to_Expiry"]) / 30, -2, 2)
    except Exception as e:
        st.error(f"Error in XGBoost forecasting loop: {str(e)}")
        return None, None, None, None, None, None, None, None

    xgb_vols = np.clip(xgb_vols, 5, 50)
    if df["Event_Flag"].iloc[-1] == 1:
        xgb_vols = [v * 1.1 for v in xgb_vols]

    # Blended Forecast
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
    return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_

# Function to generate trading strategy
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score):
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

    # Capital Allocation
    capital_alloc = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
    position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
    deploy = capital * capital_alloc.get(regime, 0.2) * position_size
    max_loss = deploy * 0.2
    total_exposure = deploy / capital

    # Risk Filters
    risk_flags = []
    if regime in ["HIGH", "EVENT-DRIVEN"] and strategy in ["Short Strangle", "Iron Fly"]:
        risk_flags.append("No naked legs allowed in HIGH/EVENT-DRIVEN regimes")
    if latest["PnL_Day"] < -0.03 * capital:
        risk_flags.append("Daily drawdown exceeds 3%")
    if latest["VIX_Change_Pct"] > 10:
        risk_flags.append("High VIX spike detected")

    # Behavioral Monitoring
    behavior_score = 8 if deploy < 0.5 * capital else 6
    behavior_warnings = ["Consider reducing position size"] if behavior_score < 7 else []

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

# Main execution
if page == "Dashboard" and run_button:
    with st.spinner("Initializing AI Copilot..."):
        df = load_data()
        if df is not None:
            st.markdown(f"<span style='color: #00d4ff; font-size: 14px;'>Data Source: {st.session_state.data_source}</span>", unsafe_allow_html=True)
            df = generate_synthetic_features(df)

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
                        'FII_Option_Pos', 'Advance_Decline_Ratio', 'Capital_Pressure_Index', 'Gamma_Bias'
                    ],
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)
                st.dataframe(feature_importance, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Trading Strategy Card
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("🎯 Trading Strategy")
                strategy = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score)
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
                    strategy_df = pd.DataFrame([strategy])
                    strategy_csv = strategy_df.to_csv(index=False)
                    st.download_button(
                        label="Download Strategy (CSV)",
                        data=strategy_csv,
                        file_name="volguard_strategy.csv",
                        mime="text/csv"
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                # Footer
                st.markdown("""
                    <div class="footer">
                        VolGuard: Protection First, Edge Always | Built by Shritish Shukla & AI Co-Founder<br>
                        "We don't predict direction - we predict conditions. We deploy edge, survive, and outlast."
                    </div>
                """, unsafe_allow_html=True)

else:
    st.info("Set parameters and activate VolGuard to begin your journey.")
