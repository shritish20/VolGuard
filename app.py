import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
import logging
import re
from py5paisa import FivePaisaClient
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
from functools import lru_cache
from time import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #1a1a2e, #0f1c2e); color: #e5e5e5; font-family: 'Inter', sans-serif; }
        .stTabs [data-baseweb="tab-list"] { background: #16213e; border-radius: 10px; padding: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
        .stTabs [data-baseweb="tab"] { color: #a0a0a0; font-weight: 500; padding: 10px 20px; border-radius: 8px; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #e94560; color: white; font-weight: 700; }
        .stTabs [data-baseweb="tab"]:hover { background: #2a2a4a; color: white; }
        .sidebar .stButton>button { width: 100%; background: #0f3460; color: white; border-radius: 10px; padding: 12px; margin: 5px 0; }
        .sidebar .stButton>button:hover { transform: scale(1.05); background: #e94560; }
        .card { background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9)); border-radius: 15px; padding: 20px; margin: 15px 0; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); }
        .card:hover { transform: translateY(-5px); }
        .strategy-carousel { display: flex; overflow-x: auto; gap: 20px; padding: 10px; }
        .strategy-card { flex: 0 0 auto; width: 300px; background: #16213e; border-radius: 15px; padding: 20px; }
        .strategy-card:hover { transform: scale(1.05); }
        .stMetric { background: rgba(15, 52, 96, 0.7); border-radius: 15px; padding: 15px; text-align: center; }
        .gauge { width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px; }
        .regime-badge { padding: 8px 15px; border-radius: 20px; font-weight: bold; font-size: 14px; text-transform: uppercase; }
        .regime-low { background: #28a745; color: white; }
        .regime-medium { background: #ffc107; color: black; }
        .regime-high { background: #dc3545; color: white; }
        .regime-event { background: #ff6f61; color: white; }
        .alert-banner { background: #dc3545; color: white; padding: 15px; border-radius: 10px; position: sticky; top: 0; z-index: 100; }
        .stButton>button { background: #e94560; color: white; border-radius: 10px; padding: 12px 25px; font-size: 16px; }
        .stButton>button:hover { transform: scale(1.05); background: #ffcc00; }
        .footer { text-align: center; padding: 20px; color: #a0a0a0; font-size: 14px; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
default_state = {
    "backtest_run": False,
    "backtest_results": None,
    "violations": 0,
    "journal_complete": False,
    "trades": [],
    "logged_in": False,
    "client": None,
    "real_time_market_data": None,
    "prepared_orders": None
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Helper Functions
def parse_5paisa_date_string(date_string):
    """Parse 5paisa API's /Date(1234567890000+0000)/ format to timestamp."""
    try:
        match = re.search(r'/Date\((\d+)(?:[+-]\d+)?\)/', date_string)
        if match:
            timestamp = int(match.group(1))
            if timestamp < 0 or timestamp > 9999999999999:
                return None
            return timestamp
        return None
    except Exception as e:
        logger.error(f"Error parsing date string {date_string}: {e}")
        return None

def _calculate_max_pain(df: pd.DataFrame, nifty_spot: float):
    """Calculate max pain strike and difference percentage from spot."""
    try:
        valid_cptypes = {"CE", "PE"}
        if not df["CPType"].isin(valid_cptypes).all():
            logger.error("Invalid CPType values in option chain.")
            return None, None
        strikes = np.unique(df["StrikeRate"])
        call_oi = df[df["CPType"] == "CE"][["StrikeRate", "OpenInterest"]].set_index("StrikeRate")
        put_oi = df[df["CPType"] == "PE"][["StrikeRate", "OpenInterest"]].set_index("StrikeRate")
        
        def calc_loss(k):
            call_loss = np.maximum(k - strikes, 0) @ call_oi.reindex(strikes, fill_value=0)["OpenInterest"]
            put_loss = np.maximum(strikes - k, 0) @ put_oi.reindex(strikes, fill_value=0)["OpenInterest"]
            return call_loss + put_loss
        
        losses = np.array([calc_loss(k) for k in strikes])
        if losses.size == 0:
            return None, None
        max_pain_strike = strikes[np.argmin(losses)]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {e}")
        return None, None

# 5paisa Client Initialization
def initialize_5paisa_client(totp_code):
    """Initialize 5paisa client with TOTP authentication."""
    try:
        if not totp_code.isdigit() or len(totp_code) != 6:
            st.error("Invalid TOTP code. Must be a 6-digit number.")
            return None
        required_keys = ["APP_NAME", "APP_SOURCE", "USER_ID", "PASSWORD", "USER_KEY", "ENCRYPTION_KEY", "CLIENT_CODE", "PIN"]
        if not all(key in st.secrets["fivepaisa"] for key in required_keys):
            st.error("Missing 5paisa credentials in secrets.toml.")
            return None
        cred = {
            "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
            "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
            "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
            "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
            "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
            "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
        }
        client = FivePaisaClient(cred=cred)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.get_totp_session(
                    st.secrets["fivepaisa"]["CLIENT_CODE"],
                    totp_code,
                    st.secrets["fivepaisa"]["PIN"]
                )
                if client.get_access_token():
                    logger.info("5paisa client initialized successfully")
                    st.session_state.logged_in = True
                    return client
                else:
                    logger.warning(f"TOTP session attempt {attempt+1} failed.")
                    if attempt < max_retries - 1:
                        time.sleep(1)
            except Exception as e:
                logger.error(f"TOTP session attempt {attempt+1} error: {e}")
                if attempt == max_retries - 1:
                    st.error(f"Login failed after {max_retries} attempts: {e}")
                    return None
        return None
    except Exception as e:
        logger.error(f"Error initializing 5paisa client: {e}")
        return None

# Data Fetching
@lru_cache(maxsize=1)
def fetch_real_time_market_data_cached(client, cache_time: float):
    """Fetch and cache real-time market data from 5paisa API."""
    return fetch_real_time_market_data(client)

def fetch_real_time_market_data(client):
    """Fetch NIFTY 50, India VIX, and option chain data from 5paisa API."""
    try:
        current_time = time()
        if "market_data_cache_time" not in st.session_state or \
           current_time - st.session_state.get("market_data_cache_time", 0) > 60:
            result = fetch_real_time_market_data_cached(client, current_time)
            st.session_state.market_data_cache_time = current_time
            return result
        return fetch_real_time_market_data_cached(client, st.session_state.market_data_cache_time)
    except Exception as e:
        logger.error(f"Error fetching cached market data: {e}")
        return None

def fetch_real_time_market_data(client):
    """Core function to fetch real-time market data."""
    try:
        logger.info("Fetching real-time data from 5paisa API")
        def safe_get(data, *keys, default=None):
            current = data
            for key in keys:
                try:
                    current = current[key]
                except (KeyError, TypeError, IndexError):
                    return default
            return current

        nifty_req = [
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920000},  # NIFTY 50
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920005}   # India VIX
        ]
        market_feed = client.fetch_market_feed(nifty_req)
        nifty_spot = safe_get(market_feed, "Data", 0, "LastRate", default=0) or \
                     safe_get(market_feed, "Data", 0, "LastTradedPrice", default=0)
        vix = safe_get(market_feed, "Data", 1, "LastRate", default=0) or \
              safe_get(market_feed, "Data", 1, "LastTradedPrice", default=0)
        if not nifty_spot or not vix:
            raise Exception("Missing NIFTY or VIX price")

        expiries = client.get_expiry("N", "NIFTY")
        expiry_list = sorted(
            [(parse_5paisa_date_string(e["ExpiryDate"]), e) for e in safe_get(expiries, "Expiry", default=[])],
            key=lambda x: x[0] or float("inf")
        )
        if not expiry_list:
            raise Exception("Failed to fetch expiries")
        near_expiry, far_expiry = expiry_list[:2]
        near_oc = client.get_option_chain("N", "NIFTY", near_expiry[1]["ExpiryDate"])
        far_oc = client.get_option_chain("N", "NIFTY", far_expiry[1]["ExpiryDate"])
        df_near = pd.DataFrame(safe_get(near_oc, "Options", default=[]))
        df_far = pd.DataFrame(safe_get(far_oc, "Options", default=[]))
        if df_near.empty or df_far.empty:
            raise Exception("Failed to fetch option chain")

        required_cols = ["StrikeRate", "CPType", "LastRate", "OpenInterest", "ScripCode"]
        if not all(col in df_near.columns for col in required_cols):
            raise Exception("Required columns missing in near option chain")
        df_near["StrikeRate"] = df_near["StrikeRate"].astype(float)
        atm_strike = df_near["StrikeRate"].iloc[(df_near["StrikeRate"] - nifty_spot).abs().argmin()]
        atm_data = df_near[df_near["StrikeRate"] == atm_strike]
        atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
        atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
        straddle_price = atm_call + atm_put

        calls = df_near[df_near["CPType"] == "CE"]
        puts = df_near[df_near["CPType"] == "PE"]
        call_oi = calls["OpenInterest"].sum()
        put_oi = puts["OpenInterest"].sum()
        pcr = put_oi / call_oi if call_oi != 0 else float("inf")

        max_pain_strike, max_pain_diff_pct = _calculate_max_pain(df_near, nifty_spot)
        if max_pain_strike is None:
            raise Exception("Max pain calculation failed")

        vix_change_pct = 0
        iv_file = "vix_history.csv"
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            prev_vix = iv_history["VIX"].iloc[-1] if not iv_history.empty else vix
            vix_change_pct = ((vix - prev_vix) / prev_vix * 100) if prev_vix != 0 else 0
        pd.DataFrame({"Date": [datetime.now().date()], "VIX": [vix]}).to_csv(iv_file, mode='a', header=not os.path.exists(iv_file), index=False)

        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": vix_change_pct,
            "option_chain_near": df_near,
            "option_chain_far": df_far,
            "near_expiry": near_expiry[1]["ExpiryDate"],
            "far_expiry": far_expiry[1]["ExpiryDate"],
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa API data: {e}")
        return None

def load_data(client):
    """Load historical and real-time data from GitHub and 5paisa API."""
    try:
        logger.info("Loading data")
        real_data = fetch_real_time_market_data(client) if client else None
        data_source = "5paisa API (LIVE)" if real_data else "GitHub CSV (FALLBACK)"
        logger.info(f"Data source: {data_source}")

        nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
        vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
        
        if real_data is None:
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            if nifty.index.duplicated().sum() > 0:
                logger.warning(f"Found {nifty.index.duplicated().sum()} duplicate dates in nifty_50.csv.")
                nifty = nifty.groupby(nifty.index).last()

            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})
            if vix.index.duplicated().sum() > 0:
                logger.warning(f"Found {vix.index.duplicated().sum()} duplicate dates in india_vix.csv.")
                vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates)
            df = df.sort_index().ffill().bfill()
        else:
            latest_date = pd.to_datetime(real_data.get("timestamp", datetime.now())).date()
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            if nifty.index.duplicated().sum() > 0:
                nifty = nifty.groupby(nifty.index).last()

            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})
            if vix.index.duplicated().sum() > 0:
                vix = vix.groupby(vix.index).last()

            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"],
                "VIX": vix["VIX"]
            }).dropna()
            historical_df = historical_df[historical_df.index < pd.to_datetime(latest_date)]

            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["vix"]]
            }, index=[pd.to_datetime(latest_date)])
            df = pd.concat([historical_df, df]).groupby(level=0).last().sort_index()

        logger.debug(f"Data loaded successfully from {data_source}. Shape: {df.shape}")
        return df, real_data, data_source
    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Error loading data: {e}")
        return None, None, None

def fetch_all_api_portfolio_data(client, capital):
    """Fetch comprehensive portfolio data from 5paisa API."""
    try:
        logger.info("Fetching portfolio data")
        portfolio_data = {}
        for key, func in [
            ("holdings", client.holdings),
            ("margin", client.margin),
            ("positions", client.positions),
            ("orders", client.order_book)
        ]:
            try:
                portfolio_data[key] = func()
            except Exception as e:
                logger.error(f"Failed to fetch {key}: {e}")
                portfolio_data[key] = None
        total_pnl = sum(pos.get("ProfitLoss", 0) for pos in portfolio_data.get("positions", []) if isinstance(pos, dict))
        total_margin = sum(pos.get("MarginUsed", 0) for pos in portfolio_data.get("margin", []) if isinstance(pos, dict))
        total_exposure = sum(pos.get("Exposure", 0) for pos in portfolio_data.get("positions", []) if isinstance(pos, dict))
        return {
            "weekly_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure": total_exposure / capital * 100 if capital > 0 else 0,
            "holdings": portfolio_data.get("holdings", []),
            "orders": portfolio_data.get("orders", [])
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {e}")
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "holdings": [], "orders": []}

# Feature Generation
@st.cache_data
def generate_historical_features(df, capital):
    """Generate features for historical data."""
    df = df.copy()
    n_days = len(df)
    np.random.seed(42)
    strike_step = 100

    def calculate_days_to_expiry(dates):
        days_to_expiry = []
        for date in dates:
            days_ahead = (3 - date.weekday()) % 7 or 7
            next_expiry = date + pd.Timedelta(days=days_ahead)
            dte = (next_expiry - date).days
            days_to_expiry.append(dte)
        return np.array(days_to_expiry)

    df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
    event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
    df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
    df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)

    df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(
        lambda x: (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100 if len(x) >= 5 else 50.0
    ).interpolate().fillna(50.0)

    market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
    df["PCR"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * -10, 0.7, 2.0)
    df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
    df["Spot_MaxPain_Diff_Pct"] = np.clip(np.abs(np.random.lognormal(-2, 0.5, n_days)), 0.1, 1.0)
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
    df["Straddle_Price"] = np.clip(np.random.normal(200, 50, n_days), 50, 400)
    return df.interpolate().fillna(method='bfill')

def generate_features(df, real_data, capital):
    """Generate features with live data integration."""
    try:
        logger.info("Generating features")
        df = generate_historical_features(df.iloc[:-1], capital)
        live_row = df.iloc[-1:].copy()
        if real_data:
            live_row["ATM_IV"] = real_data["vix"]
            live_row["PCR"] = real_data["pcr"]
            live_row["VIX_Change_Pct"] = real_data["vix_change_pct"]
            live_row["Spot_MaxPain_Diff_Pct"] = real_data["max_pain_diff_pct"]
            live_row["Straddle_Price"] = real_data["straddle_price"]
        df = pd.concat([df, live_row])
        try:
            df.to_csv("volguard_hybrid_data.csv")
        except PermissionError:
            logger.error("Permission denied when writing to volguard_hybrid_data.csv")
            st.error("Cannot save volguard_hybrid_data.csv: Permission denied")
        return df
    except Exception as e:
        st.error(f"Error generating features: {e}")
        logger.error(f"Error generating features: {e}")
        return None

# Volatility Forecasting
feature_cols = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos'
]

@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    """Forecast future volatility using GARCH and XGBoost."""
    try:
        logger.info("Forecasting volatility")
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df_garch = df.tail(len(df))
        min_garch_data = 100
        if len(df_garch) < min_garch_data:
            logger.warning(f"Only {len(df_garch)} returns for GARCH; using VIX as fallback.")
            garch_vols = np.full(forecast_horizon, df["VIX"].iloc[-1] or 15.0)
        else:
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
            "Date": pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_horizon, freq='B'),
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * forecast_horizon
        })
        logger.debug("Volatility forecast completed")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        st.error(f"Error in volatility forecasting: {e}")
        logger.error(f"Error in volatility forecasting: {e}")
        return None, None, None, None, None, None, None, None

# Backtesting
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    """Run backtest for selected strategy and date range."""
    try:
        logger.info(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")
        df = df.groupby(df.index).last().loc[start_date:end_date].copy()
        if len(df) < 50:
            st.error(f"Backtest failed: Insufficient data ({len(df)} days)")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        required_cols = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Backtest failed: Missing columns {missing_cols}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        backtest_results = []
        lot_size = 25
        base_transaction_cost = 0.002
        stt = 0.0005
        portfolio_pnl = 0
        risk_free_rate = 0.06 / 126

        def run_strategy_engine(day_data, avg_vol):
            iv = day_data["ATM_IV"]
            hv = day_data["Realized_Vol"]
            iv_hv_gap = iv - hv
            iv_skew = day_data["IV_Skew"]
            dte = day_data["Days_to_Expiry"]
            event_flag = day_data["Event_Flag"]

            if portfolio_pnl < -0.1 * day_data["Total_Capital"]:
                return None, None, "Portfolio drawdown limit reached", [], 0, 0, 0

            regime = "EVENT-DRIVEN" if event_flag == 1 else \
                     "LOW" if avg_vol < 15 else \
                     "MEDIUM" if avg_vol < 20 else "HIGH"

            strategy = "Undefined"
            reason = "N/A"
            tags = []
            risk_reward = 1.5 if iv_hv_gap > 5 else 1.0

            if regime == "LOW":
                if iv_hv_gap > 5 and dte < 10:
                    strategy = "Butterfly Spread"
                    reason = "Low vol & short expiry favors pinning strategies"
                    tags = ["Neutral", "Theta", "Expiry Play"]
                    risk_reward = 2.0
                else:
                    strategy = "Iron Fly"
                    reason = "Low volatility and time decay favors delta-neutral Iron Fly"
                    tags = ["Neutral", "Theta", "Range Bound"]

            elif regime == "MEDIUM":
                if iv_hv_gap > 3 and iv_skew > 2:
                    strategy = "Iron Condor"
                    reason = "Medium vol and skew favor wide-range Iron Condor"
                    tags = ["Neutral", "Theta", "Range Bound"]
                    risk_reward = 1.8
                else:
                    strategy = "Short Strangle"
                    reason = "Balanced vol, premium-rich environment for Short Strangle"
                    tags = ["Neutral", "Premium Selling", "Volatility Harvest"]

            elif regime == "HIGH":
                if iv_hv_gap > 10:
                    strategy = "Jade Lizard"
                    reason = "High IV + call skew = Jade Lizard for defined upside risk"
                    tags = ["Skewed", "Volatility", "Defined Risk"]
                    risk_reward = 1.2
                else:
                    strategy = "Iron Condor"
                    reason = "High vol favors wide-range Iron Condor for premium collection"
                    tags = ["Neutral", "Theta", "Range Bound"]

            elif regime == "EVENT-DRIVEN":
                if iv > 30 and dte < 5:
                    strategy = "Short Straddle"
                    reason = "Event + near expiry + IV spike → high premium capture"
                    tags = ["Volatility", "Event", "Neutral"]
                    risk_reward = 1.5
                else:
                    strategy = "Calendar Spread"
                    reason = "Event-based uncertainty favors term structure opportunity"
                    tags = ["Volatility", "Event", "Calendar"]

            capital = day_data["Total_Capital"]
            capital_alloc = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.06}
            deploy = capital * capital_alloc.get(regime, 0.06)
            max_loss = deploy * 0.025
            return regime, strategy, reason, tags, deploy, max_loss, risk_reward

        def get_dynamic_slippage(strategy, iv, dte):
            base_slippage = 0.005
            iv_multiplier = min(iv / 20, 2.5)
            dte_factor = 1.5 if dte < 5 else 1.0
            strategy_multipliers = {
                "Iron Condor": 1.8, "Butterfly Spread": 2.2, "Iron Fly": 1.5,
                "Short Strangle": 1.6, "Calendar Spread": 1.3, "Jade Lizard": 1.4,
                "Short Straddle": 1.5
            }
            return base_slippage * strategy_multipliers.get(strategy, 1.0) * iv_multiplier * dte_factor

        for i in range(1, len(df)):
            day_data = df.iloc[i]
            prev_day = df.iloc[i-1]
            date = day_data.name
            historical_df = df.iloc[:i]
            forecast_log, *_ = forecast_volatility_future(historical_df, forecast_horizon=1)
            avg_vol = forecast_log["Blended_Vol"].iloc[0] if forecast_log is not None else df["Realized_Vol"].iloc[max(0, i-5):i].mean()

            regime, strategy, reason, tags, deploy, max_loss, risk_reward = run_strategy_engine(day_data, avg_vol)
            if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy_choice):
                continue

            extra_cost = 0.001 if "Iron" in strategy else 0
            total_cost = base_transaction_cost + extra_cost + stt
            slippage = get_dynamic_slippage(strategy, day_data["ATM_IV"], day_data["Days_to_Expiry"])
            entry_price = day_data["Straddle_Price"]
            lots = max(1, min(int(deploy / (entry_price * lot_size)), 2))
            decay_factor = max(0.75, 1 - day_data["Days_to_Expiry"] / 10)
            premium = entry_price * lot_size * lots * (1 - slippage - total_cost) * decay_factor
            iv_factor = min(day_data["ATM_IV"] / avg_vol, 1.5) if avg_vol != 0 else 1.0
            breakeven = entry_price * (1 + iv_factor * (0.04 if day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25 else 0.06))
            nifty_move = abs(day_data["NIFTY_Close"] - prev_day["NIFTY_Close"])
            loss = min(max(0, nifty_move - breakeven) * lot_size * lots, premium * (0.6 if strategy in ["Iron Fly", "Iron Condor"] else 0.8))
            pnl = max(-max_loss, min(premium - loss, max_loss * 1.5))
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

        backtest_df = pd.DataFrame(backtest_results)
        if backtest_df.empty:
            logger.warning(f"No trades generated for {strategy_choice}")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df)
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max()
        backtest_df.set_index("Date", inplace=True)
        returns = backtest_df["PnL"] / df["Total_Capital"].reindex(backtest_df.index, method="ffill").fillna(capital)
        nifty_returns = df["NIFTY_Close"].pct_change().reindex(backtest_df.index, method="ffill").fillna(0)
        excess_returns = returns - nifty_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(126) if excess_returns.std() != 0 else 0
        sortino_ratio = excess_returns.mean() / excess_returns[excess_returns < 0].std() * np.sqrt(126) if len(excess_returns[excess_returns < 0]) > 0 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 else float('inf')

        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_perf["Win_Rate"] = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: len(x[x > 0]) / len(x)).reset_index(drop=True)
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_perf["Win_Rate"] = backtest_df.groupby("Regime")["PnL"].apply(lambda x: len(x[x > 0]) / len(x)).reset_index(drop=True)

        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"Error running backtest: {e}")
        logger.error(f"Error running backtest: {e}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

# Strategy Generation
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
    """Generate trading strategy based on market conditions."""
    try:
        logger.info("Generating trading strategy")
        latest = df.iloc[-1]
        avg_vol = np.mean(forecast_log["Blended_Vol"])
        iv = latest["ATM_IV"]
        hv = latest["Realized_Vol"]
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"]
        pcr = latest["PCR"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]

        risk_flags = []
        if latest["VIX"] > 25:
            risk_flags.append("VIX > 25% - High Volatility Risk")
        if latest["Spot_MaxPain_Diff_Pct"] > 70:
            risk_flags.append("Exposure > 70% - High Exposure Risk")
        if latest["PnL_Day"] < -0.05 * capital:
            risk_flags.append("Weekly Loss > 5% - High Loss Risk")
        if latest["VIX_Change_Pct"] > 10:
            risk_flags.append("High VIX Spike Detected")

        if risk_flags:
            st.session_state.violations += 1
            if st.session_state.violations >= 2 and not st.session_state.journal_complete:
                st.error("🚨 Discipline Lock: Complete Journaling to Unlock Trading")
                return None

        regime = "EVENT-DRIVEN" if event_flag == 1 else \
                 "LOW" if avg_vol < 15 else \
                 "MEDIUM" if avg_vol < 20 else "HIGH"

        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0
        confidence = min(0.9, max(0.5, confidence_score / 100))

        if regime == "LOW":
            if iv_hv_gap > 5 and dte < 10:
                strategy = "Butterfly Spread"
                reason = "Low vol & short expiry favors pinning strategies"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.0
            else:
                strategy = "Iron Fly"
                reason = "Low volatility and time decay favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta", "Range Bound"]

        elif regime == "MEDIUM":
            if iv_hv_gap > 3 and iv_skew > 2:
                strategy = "Iron Condor"
                reason = "Medium vol and skew favor wide-range Iron Condor"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8
            else:
                strategy = "Short Strangle"
                reason = "Balanced vol, no events, and premium-rich environment"
                tags = ["Neutral", "Premium Selling", "Volatility Harvest"]

        elif regime == "HIGH":
            if iv_hv_gap > 10:
                strategy = "Jade Lizard"
                reason = "High IV + call skew = Jade Lizard for defined upside risk"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.2
            else:
                strategy = "Iron Condor"
                reason = "High vol favors wide-range Iron Condor for premium collection"
                tags = ["Neutral", "Theta", "Range Bound"]

        elif regime == "EVENT-DRIVEN":
            if iv > 30 and dte < 5:
                strategy = "Short Straddle"
                reason = "Event + near expiry + IV spike → high premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.5
            else:
                strategy = "Calendar Spread"
                reason = "Event-based uncertainty favors term structure opportunity"
                tags = ["Volatility", "Event", "Calendar"]

        capital_alloc = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc.get(regime, 0.2) * position_size
        max_loss = deploy * 0.2
        total_exposure = deploy / capital

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
            "Behavior_Score": 8 if deploy < 0.5 * capital else 6,
            "Behavior_Warnings": ["Consider reducing position size"] if deploy >= 0.5 * capital else []
        }
    except Exception as e:
        st.error(f"Error generating strategy: {e}")
        logger.error(f"Error generating strategy: {e}")
        return None

# Trading Functions
def prepare_trade_orders(strategy, real_data, capital):
    """Prepare trade orders based on strategy and market data."""
    try:
        logger.info(f"Preparing trade orders: {strategy['Strategy']}")
        if not real_data or "option_chain_near" not in real_data or "atm_strike" not in real_data:
            return None, "Invalid real-time data from 5paisa API"

        option_chain_near = real_data["option_chain_near"]
        option_chain_far = real_data["option_chain_far"]
        atm_strike = real_data["atm_strike"]
        lot_size = 25
        deploy = strategy["Deploy"]
        max_loss = strategy["Max_Loss"]
        near_expiry = real_data["near_expiry"]
        far_expiry = real_data["far_expiry"]

        premium_per_lot = real_data["straddle_price"] * lot_size
        lots = max(1, min(int(deploy / premium_per_lot), int(max_loss / (premium_per_lot * 0.2))))
        orders = []

        if strategy["Strategy"] == "Short Straddle":
            strikes = [(atm_strike, "CE", "S", near_expiry), (atm_strike, "PE", "S", near_expiry)]
        elif strategy["Strategy"] == "Short Strangle":
            call_strike = atm_strike + 100
            put_strike = atm_strike - 100
            strikes = [(call_strike, "CE", "S", near_expiry), (put_strike, "PE", "S", near_expiry)]
        elif strategy["Strategy"] == "Iron Condor":
            call_sell_strike = atm_strike + 100
            call_buy_strike = call_sell_strike + 100
            put_sell_strike = atm_strike - 100
            put_buy_strike = put_sell_strike - 100
            strikes = [
                (call_sell_strike, "CE", "S", near_expiry),
                (call_buy_strike, "CE", "B", near_expiry),
                (put_sell_strike, "PE", "S", near_expiry),
                (put_buy_strike, "PE", "B", near_expiry)
            ]
        elif strategy["Strategy"] == "Iron Fly":
            call_sell_strike = atm_strike
            call_buy_strike = atm_strike + 100
            put_sell_strike = atm_strike
            put_buy_strike = atm_strike - 100
            strikes = [
                (call_sell_strike, "CE", "S", near_expiry),
                (call_buy_strike, "CE", "B", near_expiry),
                (put_sell_strike, "PE", "S", near_expiry),
                (put_buy_strike, "PE", "B", near_expiry)
            ]
        elif strategy["Strategy"] == "Butterfly Spread":
            call_sell_strike = atm_strike
            call_buy_strike = atm_strike + 200
            put_buy_strike = atm_strike - 200
            strikes = [
                (call_sell_strike, "CE", "S", near_expiry),
                (call_buy_strike, "CE", "B", near_expiry),
                (put_buy_strike, "PE", "B", near_expiry)
            ]
        elif strategy["Strategy"] == "Jade Lizard":
            call_sell_strike = atm_strike + 100
            put_sell_strike = atm_strike - 100
            put_buy_strike = atm_strike - 200
            strikes = [
                (call_sell_strike, "CE", "S", near_expiry),
                (put_sell_strike, "PE", "S", near_expiry),
                (put_buy_strike, "PE", "B", near_expiry)
            ]
        elif strategy["Strategy"] == "Calendar Spread":
            strikes = [
                (atm_strike, "CE", "S", near_expiry),
                (atm_strike, "CE", "B", far_expiry)
            ]
        else:
            return None, "Unsupported strategy"

        for strike, cp_type, buy_sell, expiry in strikes:
            option_chain = option_chain_far if expiry == far_expiry else option_chain_near
            opt_data = option_chain[(option_chain["StrikeRate"] == strike) & (option_chain["CPType"] == cp_type)]
            if opt_data.empty:
                return None, f"No option data for {cp_type} at strike {strike}"
            scrip_code = int(opt_data["ScripCode"].iloc[0])
            latest_price = float(opt_data["LastRate"].iloc[0])
            proposed_price = latest_price * (1.02 if buy_sell == "B" else 0.98)  # Limit order with 2% buffer
            orders.append({
                "Exchange": "N",
                "ExchangeType": "D",
                "ScripCode": scrip_code,
                "Quantity": lot_size * lots,
                "Price": proposed_price,
                "OrderType": "SELL" if buy_sell == "S" else "BUY",
                "IsIntraday": False
            })

        return orders, f"Prepared {len(orders)} orders for {strategy['Strategy']} with {lots} lots"
    except Exception as e:
        logger.error(f"Error preparing trade orders: {e}")
        return None, f"Order preparation failed: {e}"

def execute_trade_orders(client, orders):
    """Execute prepared trade orders."""
    try:
        logger.info("Executing trade orders")
        for order in orders:
            margin = client.margin()
            if not margin or order["Quantity"] * order["Price"] > sum(m.get("AvailableMargin", 0) for m in margin):
                return False, f"Insufficient margin for order: {order['ScripCode']}"
            response = client.place_order(
                OrderType=order["OrderType"],
                Exchange=order["Exchange"],
                ExchangeType=order["ExchangeType"],
                ScripCode=order["ScripCode"],
                Qty=order["Quantity"],
                Price=order["Price"],
                IsIntraday=order["IsIntraday"]
            )
            if response.get("Status") != 0:
                return False, f"Order failed for ScripCode {order['ScripCode']}: {response.get('Message', 'Unknown error')}"
        return True, "All orders executed successfully"
    except Exception as e:
        logger.error(f"Error executing trade orders: {e}")
        return False, f"Trade execution failed: {e}"

def square_off_positions(client):
    """Square off all open positions."""
    try:
        logger.info("Squaring off all positions")
        response = client.squareoff_all()
        if response.get("Status") == 0:
            logger.info("All positions squared off successfully")
            return True
        logger.error(f"Failed to square off positions: {response.get('Message', 'Unknown error')}")
        return False
    except Exception as e:
        logger.error(f"Error squaring off positions: {e}")
        return False

# Sidebar Login and Controls
with st.sidebar:
    st.header("🔐 5paisa Login")
    totp_code = st.text_input("TOTP (from Authenticator App)", type="password")
    if st.button("Login"):
        with st.spinner("Logging in..."):
            client = initialize_5paisa_client(totp_code)
            if client:
                st.session_state.client = client
                st.success("✅ Logged in successfully")
            else:
                st.error("❌ Login failed")

    if st.session_state.logged_in:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.client = None
            st.success("✅ Logged out successfully")
        st.header("⚙️ Trading Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
        dte_preference = st.slider("DTE Preference (days)", 7, 30, 15)
        st.markdown("**Backtest Parameters**")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-29"))
        strategy_choice = st.selectbox("Strategy", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Short Straddle"])
        run_button = st.button("Run Analysis")
        if st.button("Square Off All Positions"):
            with st.spinner("Squaring off positions..."):
                success = square_off_positions(st.session_state.client)
                if success:
                    st.success("✅ All positions squared off successfully")
                else:
                    st.error("❌ Failed to square off positions")

# Main Execution
if not st.session_state.logged_in:
    st.info("Please login to 5paisa from the sidebar to proceed")
else:
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)
    tabs = st.tabs(["Snapshot", "Forecast", "Strategy", "Portfolio", "Journal", "Backtest"])

    if run_button:
        with st.spinner("Running VolGuard Analysis..."):
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.violations = 0
            st.session_state.journal_complete = False

            df, real_data, data_source = load_data(st.session_state.client)
            if df is not None:
                df = generate_features(df, real_data, capital)
                if df is not None:
                    backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(
                        df, capital, strategy_choice, start_date, end_date
                    )
                    st.session_state.backtest_run = True
                    st.session_state.backtest_results = {
                        "backtest_df": backtest_df,
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                        "max_drawdown": max_drawdown,
                        "sharpe_ratio": sharpe_ratio,
                        "sortino_ratio": sortino_ratio,
                        "calmar_ratio": calmar_ratio,
                        "strategy_perf": strategy_perf,
                        "regime_perf": regime_perf
                    }

                    # Snapshot Tab
                    with tabs[0]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📊 Market Snapshot")
                        last_date = df.index[-1].strftime("%d-%b-%Y")
                        last_nifty = df["NIFTY_Close"].iloc[-1]
                        prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 else last_nifty
                        last_vix = df["VIX"].iloc[-1]
                        regime = "LOW" if last_vix < 15 else "MEDIUM" if last_vix < 20 else "HIGH"
                        regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high"}[regime]
                        st.markdown(f'<div class="gauge">{regime}</div><div style="text-align: center;">Market Regime</div>', unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("NIFTY 50", f"{last_nifty:,.2f}", f"{(last_nifty - prev_nifty)/prev_nifty*100:+.2f}%")
                        with col2:
                            st.metric("India VIX", f"{last_vix:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%")
                        with col3:
                            st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}")
                        with col4:
                            st.metric("Straddle Price", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}")
                        st.markdown(f"**Last Updated**: {last_date} | **Source**: {data_source}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Forecast Tab
                    with tabs[1]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📈 Volatility Forecast")
                        with st.spinner("Predicting market volatility..."):
                            forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df, forecast_horizon)
                        if forecast_log is not None:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Blended Volatility", f"{np.mean(blended_vols):.2f}%")
                            with col2:
                                st.metric("Realized Volatility", f"{realized_vol:.2f}%")
                            with col3:
                                st.metric("Model RMSE", f"{rmse:.2f}%")
                                st.markdown(f'<div class="gauge">{int(confidence_score)}%</div><div style="text-align: center;">Confidence</div>', unsafe_allow_html=True)
                            st.line_chart(pd.DataFrame({
                                "GARCH": garch_vols,
                                "XGBoost": xgb_vols,
                                "Blended": blended_vols
                            }, index=forecast_log["Date"]), color=["#e94560", "#00d4ff", "#ffcc00"])
                            st.markdown("### Feature Importance")
                            feature_importance = pd.DataFrame({
                                'Feature': feature_cols,
                                'Importance': feature_importances
                            }).sort_values(by='Importance', ascending=False)
                            st.dataframe(feature_importance, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Strategy Tab
                    with tabs[2]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("🎯 Trading Strategies")
                        strategy = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital)
                        if strategy is None:
                            st.markdown('<div class="alert-banner">🚨 Discipline Lock: Complete Journaling to Unlock Trading</div>', unsafe_allow_html=True)
                        else:
                            regime_class = {
                                "LOW": "regime-low",
                                "MEDIUM": "regime-medium",
                                "HIGH": "regime-high",
                                "EVENT-DRIVEN": "regime-event"
                            }.get(strategy["Regime"], "regime-low")
                            st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True)
                            st.markdown(f"""
                                <div class="strategy-card">
                                    <h4>{strategy["Strategy"]}</h4>
                                    <span class="regime-badge {regime_class}">{strategy["Regime"]}</span>
                                    <p><b>Reason:</b> {strategy["Reason"]}</p>
                                    <p><b>Confidence:</b> {strategy["Confidence"]:.2f}</p>
                                    <p><b>Risk-Reward:</b> {strategy["Risk_Reward"]:.2f}:1</p>
                                    <p><b>Capital:</b> ₹{strategy["Deploy"]:,.0f}</p>
                                    <p><b>Max Loss:</b> ₹{strategy["Max_Loss"]:,.0f}</p>
                                    <p><b>Tags:</b> {', '.join(strategy["Tags"])}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            if strategy["Risk_Flags"]:
                                st.markdown(f'<div class="alert-banner">⚠️ Risk Flags: {", ".join(strategy["Risk_Flags"])}</div>', unsafe_allow_html=True)
                            if st.button("Trade Now"):
                                with st.spinner("Placing trade..."):
                                    orders, prep_message = prepare_trade_orders(strategy, real_data, capital)
                                    if orders:
                                        success, exec_message = execute_trade_orders(st.session_state.client, orders)
                                        if success:
                                            trade_log = {
                                                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                "Strategy": strategy["Strategy"],
                                                "Regime": strategy["Regime"],
                                                "Risk_Level": "High" if strategy["Risk_Flags"] else "Low",
                                                "Outcome": "Pending"
                                            }
                                            st.session_state.trades.append(trade_log)
                                            try:
                                                pd.DataFrame(st.session_state.trades).to_csv("trade_log.csv", index=False)
                                            except PermissionError:
                                                logger.error("Permission denied when writing to trade_log.csv")
                                                st.error("Cannot save trade_log.csv: Permission denied")
                                            st.success(f"✅ {exec_message}")
                                        else:
                                            st.error(f"❌ {exec_message}")
                                    else:
                                        st.error(f"❌ {prep_message}")

                    # Portfolio Tab
                    with tabs[3]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("💼 Portfolio Overview")
                        portfolio_data = fetch_all_api_portfolio_data(st.session_state.client, capital)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Weekly P&L", f"₹{portfolio_data['weekly_pnl']:,.2f}")
                        with col2:
                            st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                        with col3:
                            st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
                        st.markdown("### Open Positions")
                        pos_df = pd.DataFrame(portfolio_data["positions"]) if portfolio_data["positions"] else pd.DataFrame()
                        st.dataframe(pos_df, use_container_width=True)
                        st.markdown("### Holdings")
                        hold_df = pd.DataFrame(portfolio_data["holdings"]) if portfolio_data["holdings"] else pd.DataFrame()
                        st.dataframe(hold_df, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Journal Tab
                    with tabs[4]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📝 Discipline Hub")
                        with st.form(key="journal_form"):
                            reason_strategy = st.selectbox("Why did you choose this strategy?", ["High IV", "Low Risk", "Event Opportunity", "Other"])
                            override_risk = st.radio("Did you override any risk flags?", ("Yes", "No"))
                            expected_outcome = st.text_area("Expected Outcome")
                            submit_journal = st.form_submit_button("Submit Journal Entry")
                            if submit_journal:
                                score = (3 if override_risk == "No" else 0) + \
                                        (3 if reason_strategy != "Other" else 0) + \
                                        (3 if expected_outcome else 0) + \
                                        (1 if portfolio_data["weekly_pnl"] > 0 else 0)
                                score = min(score, 10)
                                journal_entry = {
                                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Strategy_Reason": reason_strategy,
                                    "Override_Risk": override_risk,
                                    "Expected_Outcome": expected_outcome,
                                    "Discipline_Score": score
                                }
                                journal_df = pd.DataFrame([journal_entry])
                                try:
                                    journal_df.to_csv("journal_log.csv", mode='a', header=not os.path.exists("journal_log.csv"), index=False)
                                except PermissionError:
                                    logger.error("Permission denied when writing to journal_log.csv")
                                    st.error("Cannot save journal_log.csv: Permission denied")
                                st.success(f"Journal Entry Saved! Discipline Score: {score}/10")
                                if score >= 8:
                                    st.markdown("""
                                        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
                                        <script>
                                            confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
                                        </script>
                                    """, unsafe_allow_html=True)
                                st.session_state.journal_complete = True
                                if st.session_state.violations > 0:
                                    st.session_state.violations = 0
                                    st.success("✅ Discipline Lock Removed")
                        st.markdown("### Past Entries")
                        if os.path.exists("journal_log.csv"):
                            journal_df = pd.read_csv("journal_log.csv")
                            st.dataframe(journal_df, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Backtest Tab
                    with tabs[5]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📉 Backtest Results")
                        if st.session_state.backtest_run and st.session_state.backtest_results:
                            results = st.session_state.backtest_results
                            if results["backtest_df"].empty:
                                st.warning("No trades generated for the selected parameters")
                            else:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total P&L", f"₹{results['total_pnl']:,.2f}")
                                with col2:
                                    st.metric("Win Rate", f"{results['win_rate']*100:.2f}%")
                                with col3:
                                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                                with col4:
                                    st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.2f}")
                                st.markdown("### Cumulative P&L")
                                st.line_chart(results["backtest_df"]["PnL"].cumsum(), color="#e94560")
                                st.markdown("### Strategy Performance")
                                st.dataframe(results["strategy_perf"].style.format({
                                    "sum": "₹{:,.2f}",
                                    "mean": "₹{:,.2f}",
                                    "Win_Rate": "{:.2%}"
                                }), use_container_width=True)
                                st.markdown("### Regime Performance")
                                st.dataframe(results["regime_perf"].style.format({
                                    "sum": "₹{:,.2f}",
                                    "mean": "₹{:,.2f}",
                                    "Win_Rate": "{:.2%}"
                                }), use_container_width=True)
                                st.markdown("### Detailed Backtest Results")
                                st.dataframe(results["backtest_df"].style.format({
                                    "PnL": "₹{:,.2f}",
                                    "Capital_Deployed": "₹{:,.2f}",
                                    "Max_Loss": "₹{:,.2f}",
                                    "Risk_Reward": "{:.2f}"
                                }), use_container_width=True)
                        else:
                            st.info("Run the analysis to view backtest results")
                        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True)
