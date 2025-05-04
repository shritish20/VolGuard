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
from logging.handlers import RotatingFileHandler
from py5paisa import FivePaisaClient
import time
import os
from functools import wraps
import threading
import re

# Setup logging with rotation
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler("volguard_pro.log", maxBytes=10*1024*1024, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Page config
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #1a1a2e, #0f1c2e);
            color: #e5e5e5;
            font-family: 'Inter', 'Roboto', sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: #16213e;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .stTabs [data-baseweb="tab"] {
            color: #a0a0a0;
            font-weight: 500;
            padding: 10px 20px;
            border-radius: 8px;
            transition: background 0.3s;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #e94560;
            color: white;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #2a2a4a;
            color: white;
        }
        .sidebar .stButton>button {
            width: 100%;
            background: #0f3460;
            color: white;
            border-radius: 10px;
            padding: 12px;
            margin: 5px 0;
            transition: transform 0.3s;
        }
        .sidebar .stButton>button:hover {
            transform: scale(1.05);
            background: #e94560;
        }
        .card {
            background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9));
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            transition: transform 0.3s;
            animation: fadeIn 0.6s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .strategy-carousel {
            display: flex;
            overflow-x: auto;
            gap: 20px;
            padding: 10px;
            scrollbar-width: thin;
        }
        .strategy-card {
            flex: 0 0 auto;
            width: 300px;
            background: #16213e;
            border-radius: 15px;
            padding: 20px;
            transition: transform 0.3s;
        }
        .strategy-card:hover {
            transform: scale(1.05);
        }
        .stMetric {
            background: rgba(15, 52, 96, 0.7);
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        .gauge {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0 0 15px rgba(233, 69, 96, 0.5);
            animation: rotateIn 1.2s ease-in-out;
        }
        .progress-bar {
            background: #16213e;
            border-radius: 10px;
            height: 20px;
            width: 100%;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #e94560, #ffcc00);
            transition: width 1s ease-in-out;
        }
        .regime-badge {
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
        }
        .regime-low { background: #28a745; color: white; }
        .regime-medium { background: #ffc107; color: black; }
        .regime-high { background: #dc3545; color: white; }
        .regime-event { background: #ff6f61; color: white; }
        .alert-banner {
            background: #dc3545;
            color: white;
            padding: 15px;
            border-radius: 10px;
            position: sticky;
            top: 0;
            z-index: 100;
            animation: pulse 1.5s infinite;
        }
        .stButton>button {
            background: #e94560;
            color: white;
            border-radius: 10px;
            padding: 12px 25px;
            font-size: 16px;
            transition: transform 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: #ffcc00;
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
            from { opacity: 0; } to { opacity: 1; }
        }
        @keyframes rotateIn {
            from { transform: rotate(-180deg); opacity: 0; } to { transform: rotate(0); opacity: 1; }
        }
        @keyframes pulse {
            0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "violations" not in st.session_state:
    st.session_state.violations = 0
if "journal_complete" not in st.session_state:
    st.session_state.journal_complete = False
if "trades" not in st.session_state:
    st.session_state.trades = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "client" not in st.session_state:
    st.session_state.client = None

# Rate Limiter with thread safety
class RateLimiter:
    def __init__(self, calls_per_minute=60):
        self.calls_per_minute = calls_per_minute
        self.call_timestamps = []
        self.lock = threading.Lock()

    def can_make_call(self):
        with self.lock:
            now = time.time()
            self.call_timestamps = [t for t in self.call_timestamps if now - t < 60]
            if len(self.call_timestamps) < self.calls_per_minute:
                self.call_timestamps.append(now)
                logger.debug(f"API call allowed. Remaining calls: {self.calls_per_minute - len(self.call_timestamps)}")
                return True
            logger.warning("Rate limit exceeded, waiting...")
            return False

rate_limiter = RateLimiter(calls_per_minute=60)

# API Retry Decorator
def retry_api(max_retries=3, delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                if rate_limiter.can_make_call():
                    try:
                        result = func(*args, **kwargs)
                        logger.debug(f"API call succeeded for {func.__name__}")
                        return result
                    except Exception as e:
                        logger.error(f"API call failed for {func.__name__}: {str(e)}, attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            time.sleep(delay)
                else:
                    time.sleep(60 / rate_limiter.calls_per_minute)
            st.error(f"API call failed after {max_retries} attempts for {func.__name__}")
            logger.error(f"API call failed after {max_retries} attempts for {func.__name__}")
            return None
        return wrapper
    return decorator

# 5paisa Client Setup
try:
    cred = {
        "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
        "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
        "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
        "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
        "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
        "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
    }
    client = FivePaisaClient(cred=cred)
except KeyError as e:
    st.error(f"Missing 5paisa credential: {str(e)}")
    logger.error(f"Missing 5paisa credential: {str(e)}")
    client = None

# Sidebar Login and Controls
with st.sidebar:
    st.header("🔐 5paisa Login")
    totp_code = st.text_input("TOTP (from Authenticator App)", type="password")
    if st.button("Login"):
        if not client:
            st.error("5paisa client initialization failed. Check credentials.")
        elif not re.match(r"^\d{6}$", totp_code):
            st.error("Invalid TOTP: Must be 6 digits")
        else:
            try:
                response = client.get_totp_session(
                    st.secrets["fivepaisa"]["CLIENT_CODE"],
                    totp_code,
                    st.secrets["fivepaisa"]["PIN"]
                )
                if client.get_access_token():
                    st.session_state.client = client
                    st.session_state.logged_in = True
                    st.success("✅ Logged in successfully")
                    logger.info("User logged in successfully")
                else:
                    st.error("❌ Login failed: Invalid TOTP or credentials")
                    logger.error("Login failed: Invalid TOTP or credentials")
            except Exception as e:
                st.error(f"Login error: {str(e)}")
                logger.error(f"Login error: {str(e)}")

    if st.session_state.logged_in:
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.logged_in = False
            st.session_state.client = None
            st.success("✅ Logged out successfully")
            logger.info("User logged out successfully")
        st.header("⚙️ Trading Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
        dte_preference = st.slider("DTE Preference (days)", 7, 30, 15)
        st.markdown("**Backtest Parameters**")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-29"))
        strategy_choice = st.selectbox("Strategy", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard"])
        run_button = st.button("Run Analysis")
        st.markdown("---")
        st.markdown("**Motto:** Deploy with edge, survive, outlast.")
        # Black-Scholes IV calculation
def black_scholes_call(S, K, T, r, sigma):
    try:
        if T <= 0 or S <= 0 or K <= 0:
            logger.warning(f"Invalid inputs for Black-Scholes: S={S}, K={K}, T={T}")
            return np.nan
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    except Exception as e:
        logger.error(f"Black-Scholes error: {str(e)}")
        return np.nan

def implied_volatility(S, K, T, r, market_price, option_type='call', tol=1e-5, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        try:
            if option_type == 'call':
                model_price = black_scholes_call(S, K, T, r, sigma)
            else:
                model_price = black_scholes_call(S, K, T, r, sigma) + K * np.exp(-r * T) - S
            if np.isnan(model_price):
                logger.warning(f"NaN model price for IV: S={S}, K={K}, T={T}, sigma={sigma}")
                return 20.0
            diff = model_price - market_price
            if abs(diff) < tol:
                return sigma * 100
            vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
            if vega <= 0:
                logger.warning(f"Zero/negative vega for IV: S={S}, K={K}, T={T}")
                return 20.0
            sigma -= diff / vega
        except Exception as e:
            logger.error(f"IV calculation error: {str(e)}")
            return 20.0
    logger.warning(f"IV convergence failed: S={S}, K={K}, T={T}")
    return 20.0

# Max Pain calculation (optimized)
def max_pain(df, nifty_spot):
    try:
        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["LastRate"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["LastRate"]
        strikes = df["StrikeRate"].unique()
        if len(calls) < 5 or len(puts) < 5:
            logger.warning(f"Insufficient CE/PE data for max pain: CE={len(calls)}, PE={len(puts)}")
            return nifty_spot, 0.0  # Fallback to spot price
        # Vectorized pain calculation
        pain = np.zeros(len(strikes))
        for i, K in enumerate(strikes):
            call_loss = np.maximum(strikes - K, 0) * calls.reindex(strikes, fill_value=0)
            put_loss = np.maximum(K - strikes, 0) * puts.reindex(strikes, fill_value=0)
            pain[i] = call_loss.sum() + put_loss.sum()
        max_pain_strike = strikes[np.argmin(pain)]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Max pain error: {str(e)}")
        return nifty_spot, 0.0

# Fetch real data from 5paisa
@retry_api(max_retries=3, delay=5)
def fetch_nifty_data():
    try:
        nifty_req = [{"Exch": "N", "ExchType": "C", "ScripCode": 999920000, "Symbol": "NIFTY"}]
        nifty_data = client.fetch_market_feed(nifty_req)
        if not nifty_data or "Data" not in nifty_data or not nifty_data["Data"]:
            raise Exception("Failed to fetch Nifty 50 index price")
        nifty_spot = nifty_data["Data"][0].get("LastRate", nifty_data["Data"][0].get("LastTradedPrice", 0))
        if nifty_spot <= 0:
            raise Exception("Invalid Nifty price: <= 0")

        # Fetch dynamic expiry
        expiries = client.get_expiry("N", "NIFTY")
        if not expiries or "Expiry" not in expiries:
            raise Exception("Failed to fetch expiries")
        expiry_timestamp = min([e["Timestamp"] for e in expiries["Expiry"]])  # Nearest expiry
        option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
        if not option_chain or "Options" not in option_chain:
            raise Exception("Failed to fetch option chain")

        df = pd.DataFrame(option_chain["Options"])
        required_cols = ["StrikeRate", "CPType", "LastRate", "OpenInterest"]
        if not all(col in df.columns for col in required_cols):
            raise Exception(f"Missing columns: {set(required_cols) - set(df.columns)}")
        if len(df[df["CPType"] == "CE"]) < 10 or len(df[df["CPType"] == "PE"]) < 10:
            raise Exception("Insufficient CE/PE rows in option chain")
        if (df["LastRate"] < 0).any():
            raise Exception("Invalid option prices: Negative values detected")
        if (df["StrikeRate"] <= 0).any():
            raise Exception("Invalid strike prices: <= 0")

        df["StrikeRate"] = df["StrikeRate"].astype(float)
        atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
        atm_data = df[df["StrikeRate"] == atm_strike]
        atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
        atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
        straddle_price = atm_call + atm_put

        iv_df = df[(df["StrikeRate"] >= atm_strike - 100) & (df["StrikeRate"] <= atm_strike + 100)].copy()
        T = (datetime.fromtimestamp(expiry_timestamp/1000) - datetime.now()).days / 365.0
        if T <= 0:
            logger.warning("Invalid time to expiry, using default 7 days")
            T = 7 / 365.0
        r = 0.06
        iv_df["IV (%)"] = iv_df.apply(
            lambda row: implied_volatility(
                S=nifty_spot, K=row["StrikeRate"], T=T, r=r, market_price=row["LastRate"],
                option_type='call' if row["CPType"] == "CE" else 'put'
            ), axis=1
        )
        if iv_df["IV (%)"].isna().all():
            logger.warning("All IV calculations failed, setting default 20%")
            iv_df["IV (%)"] = 20.0

        calls = df[df["CPType"] == "CE"]
        puts = df[df["CPType"] == "PE"]
        call_oi = calls["OpenInterest"].sum()
        put_oi = puts["OpenInterest"].sum()
        pcr = put_oi / call_oi if call_oi != 0 else 2.0  # Clip to max 2.0
        pcr = np.clip(pcr, 0.7, 2.0)

        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)

        atm_iv = iv_df[iv_df["StrikeRate"] == atm_strike]["IV (%)"].mean()
        if np.isnan(atm_iv):
            atm_iv = 20.0
        vix_change_pct = 0
        iv_file = "atm_iv_history.csv"
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            prev_atm_iv = iv_history["ATM_IV"].iloc[-1] if not iv_history.empty else atm_iv
            vix_change_pct = ((atm_iv - prev_atm_iv) / prev_atm_iv * 100) if prev_atm_iv != 0 else 0
        pd.DataFrame({"Date": [datetime.now()], "ATM_IV": [atm_iv]}).to_csv(iv_file, mode='a', header=not os.path.exists(iv_file), index=False)

        return {
            "nifty_spot": nifty_spot,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": vix_change_pct,
            "atm_iv": atm_iv,
            "option_chain": df
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching 5paisa data: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching 5paisa data: {str(e)}")
        return None

# Load data
@st.cache_data(ttl=600)
def load_data():
    try:
        real_data = fetch_nifty_data()
        required_cols = ["NIFTY_Close", "VIX"]
        if real_data is None:
            logger.warning("Falling back to GitHub CSV")
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            try:
                response = requests.get(nifty_url)
                response.raise_for_status()
                nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig", errors="replace")
            except UnicodeDecodeError:
                nifty = pd.read_csv(io.StringIO(response.text), encoding="latin1")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            try:
                response = requests.get(vix_url)
                response.raise_for_status()
                vix = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig", errors="replace")
            except UnicodeDecodeError:
                vix = pd.read_csv(io.StringIO(response.text), encoding="latin1")
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty.loc[common_dates, "NIFTY_Close"],
                "VIX": vix.loc[common_dates, "VIX"]
            }, index=common_dates)
            if not all(col in df.columns for col in required_cols):
                raise Exception(f"Missing columns in fallback data: {set(required_cols) - set(df.columns)}")
            df = df.ffill().bfill()
        else:
            latest_date = datetime.now().date()
            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["atm_iv"]]
            }, index=[pd.to_datetime(latest_date)])
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            try:
                response = requests.get(nifty_url)
                response.raise_for_status()
                nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig", errors="replace")
            except UnicodeDecodeError:
                nifty = pd.read_csv(io.StringIO(response.text), encoding="latin1")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            try:
                response = requests.get(vix_url)
                response.raise_for_status()
                vix = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig", errors="replace")
            except UnicodeDecodeError:
                vix = pd.read_csv(io.StringIO(response.text), encoding="latin1")
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)

            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"],
                "VIX": vix["VIX"]
            }).dropna()
            historical_df = historical_df[historical_df.index < pd.to_datetime(latest_date)]
            df = pd.concat([historical_df, df], axis=0)
            df = df[~df.index.duplicated(keep='last')]
            if not all(col in df.columns for col in required_cols):
                raise Exception(f"Missing columns in combined data: {set(required_cols) - set(df.columns)}")
            df = df.sort_index()

        logger.debug("Data loaded successfully.")
        return df, real_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None, None

# Generate synthetic features
@st.cache_data
def generate_synthetic_features(df, real_data, capital):
    try:
        n_days = len(df)
        np.random.seed(42)
        risk_free_rate = 0.06
        strike_step = 100

        if real_data:
            base_pcr = real_data["pcr"]
            base_iv = real_data["atm_iv"]
            base_straddle_price = real_data["straddle_price"]
            base_max_pain_diff_pct = real_data["max_pain_diff_pct"]
            base_vix_change_pct = real_data["vix_change_pct"]
        else:
            base_pcr = 1.0
            base_iv = 20.0
            base_straddle_price = 200.0
            base_max_pain_diff_pct = 0.5
            base_vix_change_pct = 0.0

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
                    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                else:
                    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
                return max(price, 0)
            except:
                logger.warning(f"Black-Scholes failed: S={S}, K={K}, T={T}")
                return 0

        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        # Dynamic event flag (quarterly + VIX spike + NIFTY move)
        event_spike = np.where(
            ((df.index.month % 3 == 0) & (df.index.day < 5)) |
            (df["VIX"] > 20) |
            (df["NIFTY_Close"].pct_change().abs() > 0.02),
            1.2, 1.0
        )
        df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
        df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)
        if real_data:
            df["ATM_IV"].iloc[-1] = base_iv

        def dynamic_ivp(x):
            if len(x) >= 1:
                return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / max(1, len(x) - 1)) * 100
            return 50.0
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=1).apply(dynamic_ivp)
        df["IVP"] = df["IVP"].interpolate().fillna(50.0)

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
        if real_data:
            df["PCR"].iloc[-1] = base_pcr

        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        if real_data:
            df["VIX_Change_Pct"].iloc[-1] = base_vix_change_pct

        df["Spot_MaxPain_Diff_Pct"] = np.abs(np.random.lognormal(-2, 0.5, n_days))
        df["Spot_MaxPain_Diff_Pct"] = np.clip(df["Spot_MaxPain_Diff_Pct"], 0.1, 1.0)
        if real_data:
            df["Spot_MaxPain_Diff_Pct"].iloc[-1] = base_max_pain_diff_pct

        df["Event_Flag"] = np.where(
            ((df.index.month % 3 == 0) & (df.index.day < 5)) |
            (df["VIX"] > 20) |
            (df["NIFTY_Close"].pct_change().abs() > 0.02),
            1, 0
        )
        fii_trend = np.random.normal(0, 10000, n_days)
        fii_trend[::30] *= -1
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 5000, n_days)).astype(int)
        df["IV_Skew"] = np.clip(np.random.normal(0, 0.8, n_days) + (df["VIX"] / 15 - 1) * 3, -3, 3)
        df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=1).std() * np.sqrt(252) * 100
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
        if real_data:
            df["Straddle_Price"].iloc[-1] = base_straddle_price

        if df.isna().sum().sum() > 0:
            df = df.interpolate().fillna(method='bfill')
            # Check last row for NaNs
            if df.iloc[-1].isna().any():
                logger.warning("NaNs in last row, filling with mean values")
                df.iloc[-1] = df.iloc[-1].fillna(df.mean())

        df.to_csv("volguard_hybrid_data.csv")
        logger.debug("Synthetic features generated successfully.")
        return df
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
        logger.error(f"Error generating features: {str(e)}")
        return None
# Define feature_cols globally
feature_cols = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos'
]

# Forecast volatility
@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    try:
        if forecast_horizon < 1:
            raise ValueError("Forecast horizon must be at least 1 day")
        df.index = pd.to_datetime(df.index)
        df_garch = df.tail(len(df))
        if len(df_garch) < 200:
            st.error(f"Insufficient data for GARCH: {len(df_garch)} days.")
            logger.error(f"Insufficient data for GARCH: {len(df_garch)} days")
            return None, None, None, None, None, None, None, None

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        # GARCH Model
        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).ffill() * 100
        if df_garch['Log_Returns'].isna().all():
            logger.error("All Log_Returns are NaN")
            return None, None, None, None, None, None, None, None
        garch_model = arch_model(df_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
        try:
            garch_fit = garch_model.fit(disp="off")
        except Exception as e:
            logger.error(f"GARCH fit failed: {str(e)}")
            return None, None, None, None, None, None, None, None
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
        garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)

        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()

        # XGBoost Model
        df_xgb = df.tail(len(df))
        missing_cols = [col for col in feature_cols if col not in df_xgb.columns]
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}, filling with zeros")
            for col in missing_cols:
                df_xgb[col] = 0
        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna()

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
            event_intensity = min(0.2, df["VIX"].iloc[-1] / 100)  # Dynamic based on VIX
            xgb_vols = [v * (1 + event_intensity) for v in xgb_vols]

        garch_diff = np.abs(garch_vols[0] - realized_vol)
        xgb_diff = np.abs(xgb_vols[0] - realized_vol)
        total_diff = garch_diff + xgb_diff
        garch_weight = xgb_diff / total_diff if total_diff > 0 else 0.5
        xgb_weight = 1 - garch_weight
        blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]
        confidence_score = min(100, max(50, 80 - abs(garch_diff - xgb_diff))) if total_diff > 0 else 50

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

# Generate trading strategy with stop trading recovery
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
    try:
        if capital <= 0:
            raise ValueError("Capital must be positive")
        latest = df.iloc[-1]
        avg_vol = np.mean(forecast_log["Blended_Vol"])
        iv = latest["ATM_IV"]
        hv = latest["Realized_Vol"]
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"]
        pcr = latest["PCR"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]

        # Stop trading recovery logic
        iv_file = "atm_iv_history.csv"
        iv_recovery = False
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            recent_ivs = iv_history.tail(2)["ATM_IV"] if not iv_history.empty else pd.Series([iv, iv])
            if len(recent_ivs) >= 2 and all(recent_ivs < 25):
                iv_recovery = True
                st.session_state.violations = 0
                st.success("✅ Trading resumed: IV <25% for 2 days")
                logger.info("Trading resumed due to IV recovery")

        # Enhanced risk flags
        risk_flags = []
        if latest["VIX"] > 25:
            risk_flags.append("VIX > 25% - High Volatility Risk")
        if latest.get("Exposure", 0) > 70:  # Fixed typo
            risk_flags.append("Exposure > 70% - High Exposure Risk")
        if latest["PnL_Day"] < -0.05 * capital:
            risk_flags.append("Weekly Loss > 5% - High Loss Risk")
        if latest["VIX_Change_Pct"] > 10:
            risk_flags.append("High VIX Spike Detected")
        if latest["ATM_IV"] > 30:
            risk_flags.append("IV Spike > 30% - Extreme Volatility")
        if abs(latest["NIFTY_Close"] / df["NIFTY_Close"].iloc[-2] - 1) * 100 > 3:
            risk_flags.append("NIFTY Move > 3% - Large Market Move")

        if risk_flags and not iv_recovery:
            st.session_state.violations += 1
            if st.session_state.violations >= 2 and not st.session_state.journal_complete:
                st.error("🚨 Discipline Lock: Complete Journaling to Unlock Trading")
                logger.warning("Discipline lock triggered")
                return None

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
                tags = ["Volatility Play", "Neutral", "Skew"]
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
                strategy = "Iron Condor"
                reason = "High vol favors wide-range Iron Condor for premium collection."
                tags = ["Neutral", "Theta", "Range Bound"]

        elif regime == "EVENT-DRIVEN":
            if iv > 30 and dte < 5:
                strategy = "Calendar Spread"
                reason = "Event + near expiry + IV spike → term structure opportunity."
                tags = ["Volatility", "Event", "Calendar"]
                risk_reward = 1.5
            else:
                strategy = "Iron Fly"
                reason = "Event-based uncertainty favors defined-risk Iron Fly."
                tags = ["Neutral", "Theta", "Event"]

        # Dynamic capital allocation
        capital_alloc_base = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
        risk_multipliers = {"Conservative": 0.7, "Moderate": 1.0, "Aggressive": 1.3}
        capital_alloc = {k: v * risk_multipliers[risk_tolerance] for k, v in capital_alloc_base.items()}
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc.get(regime, 0.2) * position_size
        max_loss = deploy * 0.2
        total_exposure = deploy / capital

        # Improved behavior score
        behavior_score = 8
        behavior_warnings = []
        if deploy > 0.5 * capital:
            behavior_score -= 2
            behavior_warnings.append("High capital deployment")
        if len(risk_flags) > 1:
            behavior_score -= len(risk_flags)
            behavior_warnings.append("Multiple risk flags detected")
        if risk_tolerance == "Conservative" and regime in ["HIGH", "EVENT-DRIVEN"]:
            behavior_score -= 1
            behavior_warnings.append("Conservative profile in high-risk regime")
        behavior_score = max(1, min(10, behavior_score))

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

# Fetch portfolio data with per-position stop-loss
@retry_api(max_retries=3, delay=5)
def fetch_portfolio_data(capital):
    try:
        positions = client.positions()
        if not positions:
            st.info("No open positions found.")
            logger.warning("No positions found")
            return None

        total_pnl = 0
        total_margin = 0
        total_exposure = 0
        stop_loss_positions = []
        for position in positions:
            pnl = position.get("ProfitLoss", 0)
            qty = position.get("Qty", 25)
            last_rate = position.get("LastRate", 0)
            premium = position.get("Premium", last_rate * qty)
            total_pnl += pnl
            total_margin += position.get("MarginUsed", 0)
            total_exposure += position.get("Exposure", 0)
            if premium > 0 and pnl < -0.5 * premium:
                stop_loss_positions.append(position.get("ScripCode", "Unknown"))
                logger.warning(f"Stop-loss triggered for ScripCode: {position.get('ScripCode', 'Unknown')}")

        exposure_pct = total_exposure / capital * 100 if capital > 0 else 0
        return {
            "weekly_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure": exposure_pct,
            "stop_loss_positions": stop_loss_positions
        }
    except Exception as e:
        st.error(f"Error fetching portfolio: {str(e)}")
        logger.error(f"Error fetching portfolio: {str(e)}")
        return None   
# Backtest function
def run_backtest(df, capital, start_date, end_date, strategy_choice):
    try:
        # Validate inputs
        if start_date >= end_date:
            st.error("Start date must be before end date")
            logger.error("Invalid date range: start_date >= end_date")
            return None
        required_cols = ["Strategy", "Deploy", "Max_Loss", "Risk_Reward", "NIFTY_Close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in data: {missing_cols}")
            logger.error(f"Missing columns in backtest data: {missing_cols}")
            return None

        # Filter data by date and strategy
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        if strategy_choice != "All Strategies":
            df = df[df["Strategy"] == strategy_choice]
        if df.empty:
            st.error("No data available for the selected date range and strategy")
            logger.error("Empty dataframe after filtering")
            return None

        # Dynamic slippage based on strategy
        slippage_rates = {
            "Iron Condor": 0.01,
            "Butterfly Spread": 0.005,
            "Iron Fly": 0.007,
            "Short Strangle": 0.008,
            "Calendar Spread": 0.006,
            "Jade Lizard": 0.009
        }

        initial_capital = capital
        trade_log = []
        current_capital = capital
        trades = []
        max_drawdown = 0
        peak_capital = capital

        for date, row in df.iterrows():
            strategy = row["Strategy"]
            deploy = row["Deploy"]
            max_loss = row["Max_Loss"]
            risk_reward = row["Risk_Reward"]
            slippage = slippage_rates.get(strategy, 0.005)

            if deploy > current_capital:
                deploy = current_capital
            if deploy <= 0:
                continue

            outcome = np.random.choice(
                ["win", "loss", "breakeven"],
                p=[0.4 * risk_reward, 0.3 / risk_reward, 0.3]
            )
            if outcome == "win":
                pnl = deploy * risk_reward * (1 - slippage)
            elif outcome == "loss":
                pnl = -max_loss * (1 + slippage)
            else:
                pnl = 0

            current_capital += pnl
            current_capital = max(0, current_capital)  # Floor at 0
            drawdown = (peak_capital - current_capital) / peak_capital * 100
            max_drawdown = max(max_drawdown, drawdown)
            peak_capital = max(peak_capital, current_capital)

            trade_log.append({
                "Date": date,
                "Strategy": strategy,
                "PnL": round(pnl, 2),
                "Capital": round(current_capital, 2),
                "Deployed": round(deploy, 2)
            })
            trades.append({
                "Date": date,
                "Strategy": strategy,
                "PnL": round(pnl, 2),
                "Capital": round(current_capital, 2),
                "Outcome": outcome
            })

        trade_df = pd.DataFrame(trade_log)
        if trade_df.empty:
            st.error("No trades executed in backtest")
            logger.error("No trades executed in backtest")
            return None

        # Save trade log with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trade_df.to_csv(f"trade_log_{timestamp}.csv", index=False)
        logger.info(f"Trade log saved: trade_log_{timestamp}.csv")

        final_capital = trade_df["Capital"].iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital * 100
        sharpe_ratio = (trade_df["PnL"].mean() / trade_df["PnL"].std()) * np.sqrt(252) if trade_df["PnL"].std() != 0 else 0
        win_rate = len([t for t in trades if t["PnL"] > 0]) / len(trades) * 100 if trades else 0

        st.session_state.backtest_results = {
            "Initial_Capital": initial_capital,
            "Final_Capital": final_capital,
            "Total_Return": total_return,
            "Max_Drawdown": max_drawdown,
            "Sharpe_Ratio": sharpe_ratio,
            "Win_Rate": win_rate,
            "Trade_Log": trade_df,
            "Trades": trades
        }
        st.session_state.backtest_run = True
        logger.info("Backtest completed successfully")
        return st.session_state.backtest_results
    except Exception as e:
        st.error(f"Error in backtesting: {str(e)}")
        logger.error(f"Error in backtesting: {str(e)}")
        return None

# Main execution and UI
def main():
    try:
        # Reset trades on new run
        st.session_state.trades = []

        # Check login
        if not st.session_state.logged_in:
            st.error("Please log in to proceed.")
            logger.warning("User not logged in")
            return

        # Load data
        with st.spinner("Loading market data..."):
            df, real_data = load_data()
        if df is None or real_data is None:
            st.error("Failed to load data. Check internet or credentials.")
            logger.error("Data loading failed")
            return

        # Generate synthetic features
        with st.spinner("Generating features..."):
            df = generate_synthetic_features(df, real_data, capital)
        if df is None:
            st.error("Failed to generate features.")
            logger.error("Feature generation failed")
            return

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🛠️ Strategy", "📝 Journal", "💼 Portfolio"])

        with tab1:
            st.header("Market Dashboard")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NIFTY Spot", f"₹{real_data['nifty_spot']:.2f}")
            with col2:
                st.metric("ATM IV", f"{real_data['atm_iv']:.2f}%")
            with col3:
                st.metric("PCR", f"{real_data['pcr']:.2f}")
            with st.spinner("Forecasting volatility..."):
                forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df, forecast_horizon)
            if forecast_log is not None:
                st.subheader("Volatility Forecast")
                st.dataframe(forecast_log.style.format({"GARCH_Vol": "{:.2f}", "XGBoost_Vol": "{:.2f}", "Blended_Vol": "{:.2f}", "Confidence": "{:.2f}"}))
                fig, ax = plt.subplots()
                ax.plot(forecast_log["Date"], forecast_log["Blended_Vol"], label="Blended Forecast", color="#e94560")
                ax.axhline(realized_vol, linestyle="--", color="grey", label="Realized Vol")
                ax.legend()
                ax.set_title("Volatility Forecast")
                plt.tight_layout()
                st.pyplot(fig)

        with tab2:
            st.header("Trading Strategy")
            if run_button:
                with st.spinner("Generating strategy..."):
                    strategy_data = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital)
                if strategy_data:
                    st.markdown(f"<div class='regime-badge regime-{strategy_data['Regime'].lower()}'>{strategy_data['Regime']} Regime</div>", unsafe_allow_html=True)
                    st.subheader(strategy_data["Strategy"])
                    st.write(f"**Reason**: {strategy_data['Reason']}")
                    st.write(f"**Tags**: {', '.join(strategy_data['Tags'])}")
                    st.write(f"**Confidence**: {strategy_data['Confidence']:.2%}")
                    st.write(f"**Risk/Reward**: {strategy_data['Risk_Reward']:.2f}")
                    st.write(f"**Deploy**: ₹{strategy_data['Deploy']:,.2f}")
                    st.write(f"**Max Loss**: ₹{strategy_data['Max_Loss']:,.2f}")
                    st.write(f"**Exposure**: {strategy_data['Exposure']:.2%}")
                    if strategy_data["Risk_Flags"]:
                        st.markdown("**⚠️ Risk Flags**")
                        for flag in strategy_data["Risk_Flags"]:
                            st.markdown(f"- {flag}")
                    st.write(f"**Behavior Score**: {strategy_data['Behavior_Score']}/10")
                    if strategy_data["Behavior_Warnings"]:
                        st.markdown("**⚠️ Behavior Warnings**")
                        for warning in strategy_data["Behavior_Warnings"]:
                            st.markdown(f"- {warning}")

                    if st.button("Execute Trade"):
                        try:
                            order_response = client.place_order(
                                OrderType="B",
                                Exchange="N",
                                ExchangeType="D",
                                ScripCode=999920000,
                                Qty=25,
                                Price=strategy_data["Deploy"] / 25
                            )
                            if order_response:
                                st.session_state.trades.append({
                                    "Timestamp": datetime.now(),
                                    "Strategy": strategy_data["Strategy"],
                                    "Deploy": strategy_data["Deploy"],
                                    "Status": "Executed"
                                })
                                st.success("✅ Trade executed successfully")
                                logger.info("Trade executed successfully")
                            else:
                                st.error("❌ Trade execution failed")
                                logger.error("Trade execution failed")
                        except Exception as e:
                            st.error(f"Trade execution error: {str(e)}")
                            logger.error(f"Trade execution error: {str(e)}")

        with tab3:
            st.header("Trade Journal")
            if st.session_state.trades:
                trade_df = pd.DataFrame(st.session_state.trades)
                st.dataframe(trade_df)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv = trade_df.to_csv(index=False)
                st.download_button(
                    label="Download Journal",
                    data=csv,
                    file_name=f"trade_journal_{timestamp}.csv",
                    mime="text/csv"
                )
                if st.button("Mark Journal Complete"):
                    st.session_state.journal_complete = True
                    st.session_state.violations = 0
                    st.success("✅ Journal marked complete. Trading unlocked.")
                    logger.info("Journal marked complete")
            else:
                st.info("No trades recorded yet.")

        with tab4:
            st.header("Portfolio")
            with st.spinner("Fetching portfolio..."):
                portfolio_data = fetch_portfolio_data(capital)
            if portfolio_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Weekly PnL", f"₹{portfolio_data['weekly_pnl']:,.2f}")
                with col2:
                    st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                with col3:
                    st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
                if portfolio_data["stop_loss_positions"]:
                    st.markdown("**⚠️ Stop-Loss Triggered**")
                    for scrip in portfolio_data["stop_loss_positions"]:
                        st.markdown(f"- ScripCode: {scrip}")
            else:
                st.info("No portfolio data available.")

        # Backtest
        if run_button:
            with st.spinner("Running backtest..."):
                backtest_results = run_backtest(df, capital, start_date, end_date, strategy_choice)
            if backtest_results:
                st.header("Backtest Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{backtest_results['Total_Return']:.2f}%")
                with col2:
                    st.metric("Max Drawdown", f"{backtest_results['Max_Drawdown']:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{backtest_results['Sharpe_Ratio']:.2f}")
                st.write(f"**Win Rate**: {backtest_results['Win_Rate']:.2f}%")
                st.subheader("Equity Curve")
                fig, ax = plt.subplots()
                ax.plot(backtest_results["Trade_Log"]["Date"], backtest_results["Trade_Log"]["Capital"], color="#e94560")
                ax.set_title("Equity Curve")
                plt.tight_layout()
                st.pyplot(fig)
                st.subheader("Trade Log")
                st.dataframe(backtest_results["Trade_Log"])

    except Exception as e:
        st.error(f"Application error: {str(e)}. Please check logs for details.")
        logger.error(f"Application error: {str(e)}")

# Footer
st.markdown(f"""
    <div class='footer'>
        VolGuard Pro | Version {datetime.now().strftime("%Y.%m")} | Powered by xAI
        <style>
            @media (max-width: 600px) {{
                .footer {{
                    padding: 15px;
                    font-size: 12px;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    text-align: center;
                }}
            }}
        </style>
    </div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
