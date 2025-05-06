import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta
import logging
from py5paisa import FivePaisaClient
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="VolGuard Pro - Option Seller's Copilot", page_icon="🛡️", layout="wide")

# Custom CSS for a Sensibull/Kite-like UI
st.markdown("""
    <style>
        .main { 
            background: #1a1a2e; 
            color: #e5e5e5; 
            font-family: 'Inter', sans-serif; 
        }
        .stTabs [data-baseweb="tab-list"] { 
            background: #0f1c2e; 
            border-radius: 8px; 
            padding: 10px; 
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); 
        }
        .stTabs [data-baseweb="tab"] { 
            color: #a0a0a0; 
            font-weight: 500; 
            padding: 10px 20px; 
            border-radius: 6px; 
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
            border-radius: 8px; 
            padding: 12px; 
            margin: 5px 0; 
            transition: all 0.3s ease; 
        }
        .sidebar .stButton>button:hover { 
            transform: scale(1.05); 
            background: #e94560; 
        }
        .card { 
            background: #16213e; 
            border-radius: 12px; 
            padding: 20px; 
            margin: 15px 0; 
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); 
            transition: all 0.3s ease; 
        }
        .card:hover { 
            transform: translateY(-5px); 
        }
        .strategy-carousel { 
            display: flex; 
            overflow-x: auto; 
            gap: 20px; 
            padding: 10px; 
        }
        .strategy-card { 
            flex: 0 0 auto; 
            width: 300px; 
            background: #1f2a44; 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); 
            transition: all 0.3s ease; 
        }
        .strategy-card:hover { 
            transform: scale(1.03); 
            background: #2a2a4a; 
        }
        .stMetric { 
            background: #0f3460; 
            border-radius: 12px; 
            padding: 15px; 
            text-align: center; 
        }
        .regime-badge { 
            padding: 8px 15px; 
            border-radius: 20px; 
            font-weight: bold; 
            font-size: 14px; 
            text-transform: uppercase; 
        }
        .regime-low { 
            background: #28a745; 
            color: white; 
        }
        .regime-medium { 
            background: #ffc107; 
            color: black; 
        }
        .regime-high { 
            background: #dc3545; 
            color: white; 
        }
        .regime-event { 
            background: #ff6f61; 
            color: white; 
        }
        .alert-banner { 
            background: #dc3545; 
            color: white; 
            padding: 15px; 
            border-radius: 10px; 
            position: sticky; 
            top: 0; 
            z-index: 100; 
        }
        .stButton>button { 
            background: #e94560; 
            color: white; 
            border-radius: 8px; 
            padding: 12px 25px; 
            font-size: 16px; 
            transition: all 0.3s ease; 
        }
        .stButton>button:hover { 
            transform: scale(1.05); 
            background: #ffcc00; 
            color: black; 
        }
        .footer { 
            text-align: center; 
            padding: 20px; 
            color: #a0a0a0; 
            font-size: 14px; 
            border-top: 1px solid rgba(255, 255, 255, 0.1); 
            margin-top: 30px; 
        }
        .trade-status { 
            background: rgba(255, 255, 255, 0.1); 
            border-radius: 10px; 
            padding: 15px; 
            margin-top: 15px; 
        }
        .data-tag { 
            font-size: 12px; 
            color: #a0a0a0; 
            margin-top: 5px; 
            font-style: italic; 
        }
        .expander-header { 
            background: #0f3460; 
            color: white; 
            padding: 10px; 
            border-radius: 8px; 
        }
        .stDataFrame { 
            background: #1f2a44; 
            border-radius: 8px; 
            padding: 10px; 
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
if "data_source_info" not in st.session_state:
    st.session_state.data_source_info = {"source": "", "timestamp": ""}


# 5paisa Client Initialization
def initialize_5paisa_client(totp_code):
    try:
        logger.info("Initializing 5paisa client")
        cred = {
            "APP_NAME": st.secrets["fivepaisa"].get("APP_NAME", ""),
            "APP_SOURCE": st.secrets["fivepaisa"].get("APP_SOURCE", ""),
            "USER_ID": st.secrets["fivepaisa"].get("USER_ID", ""),
            "PASSWORD": st.secrets["fivepaisa"].get("PASSWORD", ""),
            "USER_KEY": st.secrets["fivepaisa"].get("USER_KEY", ""),
            "ENCRYPTION_KEY": st.secrets["fivepaisa"].get("ENCRYPTION_KEY", "")
        }
        client = FivePaisaClient(cred=cred)
        client.get_totp_session(
            st.secrets["fivepaisa"].get("CLIENT_CODE", ""),
            totp_code,
            st.secrets["fivepaisa"].get("PIN", "")
        )
        if client.get_access_token():
            logger.info("5paisa client initialized successfully")
            return client
        else:
            logger.error("Failed to get access token")
            return None
    except Exception as e:
        logger.error(f"Error initializing 5paisa client: {str(e)}")
        return None

# Data Fetching with Source Tagging
def max_pain(df, nifty_spot):
    try:
        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
        strikes = df["StrikeRate"].unique()
        pain = []
        for K in strikes:
            total_loss = 0
            for s in strikes:
                if s in calls:
                    total_loss += max(0, s - K) * calls.get(s, 0)
                if s in puts:
                    total_loss += max(0, K - s) * puts.get(s, 0)
            pain.append((K, total_loss))
        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None

def fetch_nifty_data(client):
    try:
        logger.info("Fetching real-time data from 5paisa API")
        if not client.get_access_token():
            logger.error("Client session invalid or expired")
            st.error("❌ 5paisa API session invalid. Please re-login.")
            return None

        req_list = [
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920000},  # NIFTY 50
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920005}   # India VIX
        ]
        max_retries = 3
        market_feed = None
        for attempt in range(max_retries):
            try:
                market_feed = client.fetch_market_feed(req_list)
                if market_feed is None or not isinstance(market_feed, dict) or "Data" not in market_feed:
                    raise Exception("Invalid market feed response: Response is None or missing 'Data' key")
                if not market_feed["Data"]:
                    raise Exception("No data returned for NIFTY 50")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch market feed after {max_retries} attempts: {str(e)}")
                    st.error(f"❌ Failed to fetch NIFTY 50 and VIX data from 5paisa API: {str(e)}")
                    return None
                logger.warning(f"Market feed attempt {attempt + 1} failed: {str(e)}. Retrying...")
                continue

        if market_feed is None or not market_feed.get("Data"):
            logger.error("Market feed is None or contains no data")
            st.error("❌ Failed to fetch NIFTY 50 and VIX data from 5paisa API: No data received")
            return None

        nifty_data = next((item for item in market_feed["Data"] if item["ScripCode"] == 999920000), None)
        vix_data = next((item for item in market_feed["Data"] if item["ScripCode"] == 999920005), None)

        if not nifty_data:
            logger.error("NIFTY 50 data missing in market feed")
            st.error("❌ NIFTY 50 data missing in 5paisa API response")
            return None

        nifty_spot = nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0))
        if not nifty_spot:
            logger.error("Missing NIFTY 50 price")
            st.error("❌ Missing NIFTY 50 price in 5paisa API response")
            return None

        vix = vix_data.get("LastRate", vix_data.get("LastTradedPrice", 0)) if vix_data else 0
        if not vix_data:
            logger.warning("India VIX data missing; setting VIX to 0 and proceeding")

        for attempt in range(max_retries):
            try:
                expiries = client.get_expiry("N", "NIFTY")
                if not expiries or "Data" not in expiries or not expiries["Data"]:
                    raise Exception("Empty or invalid expiry response")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch expiries after {max_retries} attempts: {str(e)}")
                    st.error(f"❌ Failed to fetch expiry dates from 5paisa API: {str(e)}")
                    return None
                logger.warning(f"Expiry fetch attempt {attempt + 1} failed: {str(e)}. Retrying...")
                continue

        expiry_timestamp = expiries["Data"][0]["Timestamp"]
        expiry_date = expiries["Data"][0]["ExpiryDate"]

        for attempt in range(max_retries):
            try:
                option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
                if not option_chain or "Options" not in option_chain:
                    raise Exception("Empty or invalid option chain response")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch option chain after {max_retries} attempts: {str(e)}")
                    st.error(f"❌ Failed to fetch NIFTY 50 option chain from 5paisa API: {str(e)}")
                    return None
                logger.warning(f"Option chain fetch attempt {attempt + 1} failed: {str(e)}. Retrying...")
                continue

        df = pd.DataFrame(option_chain["Options"])
        required_columns = ["StrikeRate", "CPType", "LastRate", "OpenInterest", "ScripCode"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Option chain missing required columns: {missing_columns}")
            st.error(f"❌ Option chain data missing required columns: {missing_columns}")
            return None

        df["StrikeRate"] = df["StrikeRate"].astype(float)
        atm_strike = df["StrikeRate"].iloc[(df["StrikeRate"] - nifty_spot).abs().argmin()]
        atm_data = df[df["StrikeRate"] == atm_strike]
        atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
        atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
        straddle_price = atm_call + atm_put

        calls = df[df["CPType"] == "CE"]
        puts = df[df["CPType"] == "PE"]
        call_oi = calls["OpenInterest"].sum()
        put_oi = puts["OpenInterest"].sum()
        pcr = put_oi / call_oi if call_oi != 0 else float("inf")

        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)
        if max_pain_strike is None:
            logger.warning("Max pain calculation failed; setting default values")
            max_pain_strike = atm_strike
            max_pain_diff_pct = 0

        vix_change_pct = 0
        iv_file = "vix_history.csv"
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            prev_vix = iv_history["VIX"].iloc[-1] if not iv_history.empty else vix
            vix_change_pct = ((vix - prev_vix) / prev_vix * 100) if prev_vix != 0 else 0
        pd.DataFrame({"Date": [datetime.now().date()], "VIX": [vix]}).to_csv(iv_file, mode='a', header=not os.path.exists(iv_file), index=False)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.data_source_info = {
            "source": "5paisa API",
            "timestamp": timestamp
        }

        logger.info("Real-time data fetched successfully from 5paisa API")
        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "vix_change_pct": vix_change_pct,
            "option_chain": df,
            "expiry": expiry_date
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa API data: {str(e)}")
        st.error(f"❌ Error fetching data from 5paisa API: {str(e)}")
        return None

def load_data(client):
    try:
        logger.info("Loading data")
        with st.spinner("Fetching market data..."):
            real_data = fetch_nifty_data(client) if client else None
        data_source = st.session_state.data_source_info["source"] if real_data else "CSV Fallback"
        timestamp = st.session_state.data_source_info["timestamp"] if real_data else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Data source: {data_source}")

        if real_data is None:
            logger.warning("Falling back to GitHub CSV")
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty["Date"] = nifty["Date"].dt.normalize()
            if nifty["Date"].duplicated().sum() > 0:
                logger.warning(f"Found {nifty['Date'].duplicated().sum()} duplicate dates in nifty_50.csv. Aggregating by last value.")
                nifty = nifty.groupby("Date").last().reset_index()
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix["Date"] = vix["Date"].dt.normalize()
            if vix["Date"].duplicated().sum() > 0:
                logger.warning(f"Found {vix['Date'].duplicated().sum()} duplicate dates in india_vix.csv. Aggregating by last value.")
                vix = vix.groupby("Date").last().reset_index()
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates)
            df = df.groupby(df.index).last()
            df = df.sort_index()
            df = df.ffill().bfill()
        else:
            latest_date = datetime.now().date()
            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["vix"]]
            }, index=[pd.to_datetime(latest_date).normalize()])

            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty["Date"] = nifty["Date"].dt.normalize()
            if nifty["Date"].duplicated().sum() > 0:
                logger.warning(f"Found {nifty['Date'].duplicated().sum()} duplicate dates in nifty_50.csv. Aggregating by last value.")
                nifty = nifty.groupby("Date").last().reset_index()
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix["Date"] = vix["Date"].dt.normalize()
            if vix["Date"].duplicated().sum() > 0:
                logger.warning(f"Found {vix['Date'].duplicated().sum()} duplicate dates in india_vix.csv. Aggregating by last value.")
                vix = vix.groupby("Date").last().reset_index()
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})

            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"],
                "VIX": vix["VIX"]
            }).dropna()
            historical_df = historical_df[historical_df.index < pd.to_datetime(latest_date).normalize()]

            historical_df.index = historical_df.index.normalize()
            df.index = df.index.normalize()

            df = pd.concat([historical_df, df])
            df = df.groupby(df.index).last()
            df = df.sort_index()

        logger.debug(f"Data loaded successfully from {data_source}. Shape: {df.shape}")
        return df, real_data, {"source": data_source, "timestamp": timestamp}
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None, None, {"source": "Error", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def fetch_portfolio_data(client, capital):
    try:
        logger.info("Fetching portfolio data from 5paisa API")
        with st.spinner("Fetching portfolio data..."):
            positions = client.positions()
        if not positions:
            logger.warning("No positions found")
            return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "source": "5paisa API", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        total_pnl = 0
        total_margin = 0
        total_exposure = 0
        for position in positions:
            total_pnl += position.get("ProfitLoss", 0)
            total_margin += position.get("MarginUsed", 0)
            total_exposure += position.get("Exposure", 0)

        logger.info("Portfolio data fetched successfully")
        return {
            "weekly_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure": total_exposure / capital * 100 if capital > 0 else 0,
            "source": "5paisa API",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        st.error(f"❌ Error fetching portfolio data: {str(e)}")
        return {
            "weekly_pnl": 0,
            "margin_used": 0,
            "exposure": 0,
            "source": "Error",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# Feature Generation
@st.cache_data
def generate_features(df, real_data, capital):
    try:
        logger.info("Generating features")
        df = df.copy()
        df.index = df.index.normalize()
        n_days = len(df)
        np.random.seed(42)

        if real_data:
            base_pcr = real_data["pcr"]
            base_straddle_price = real_data["straddle_price"]
            base_max_pain_diff_pct = real_data["max_pain_diff_pct"]
            base_vix_change_pct = real_data["vix_change_pct"]
        else:
            base_pcr = 1.0
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

        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
        df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
        df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)
        if real_data:
            df["ATM_IV"].iloc[-1] = real_data["vix"]

        def dynamic_ivp(x):
            if len(x) >= 5:
                return (np.sum(x.iloc[:-1] <= x.iloc[-1]) / (len(x) - 1)) * 100
            return 50.0
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp)
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

        df["Event_Flag"] = np.where((df.index.month % 3 == 0) & (df.index.day < 5) | (df["Days_to_Expiry"] <= 3), 1, 0)
        df["FII_Index_Fut_Pos"] = np.cumsum(np.random.normal(0, 10000, n_days)).astype(int)
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
        df["Straddle_Price"] = np.clip(np.random.normal(base_straddle_price, 50, n_days), 50, 400)
        if real_data:
            df["Straddle_Price"].iloc[-1] = base_straddle_price

        if df.isna().sum().sum() > 0:
            df = df.interpolate().fillna(method='bfill')

        try:
            df.to_csv("volguard_hybrid_data.csv")
        except PermissionError:
            logger.error("Permission denied when writing to volguard_hybrid_data.csv")
            st.error("Cannot save volguard_hybrid_data.csv: Permission denied")

        logger.debug("Features generated successfully")
        return df
    except Exception as e:
        st.error(f"❌ Error generating features: {str(e)}")
        logger.error(f"Error generating features: {str(e)}")
        return None

# Volatility Forecasting
feature_cols = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos'
]

@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    try:
        logger.info("Forecasting volatility")
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df_garch = df.tail(len(df))
        if len(df_garch) < 200:
            st.error(f"Insufficient data for GARCH: {len(df_garch)} days")
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
        logger.debug("Volatility forecast completed")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        st.error(f"❌ Error in volatility forecasting: {str(e)}")
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None

# Backtesting
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.info(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")
        if df.empty:
            st.error("Backtest failed: No data available")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        df = df.groupby(df.index).last()
        df = df.loc[start_date:end_date].copy()
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
            except Exception as e:
                logger.error(f"Error in strategy engine: {str(e)}")
                return None, None, f"Strategy engine failed: {str(e)}", [], 0, 0, 0

        def get_dynamic_slippage(strategy, iv, dte):
            base_slippage = 0.005
            iv_multiplier = min(iv / 20, 2.5)
            dte_factor = 1.5 if dte < 5 else 1.0
            strategy_multipliers = {
                "Iron Condor": 1.8,
                "Butterfly Spread": 2.2,
                "Iron Fly": 1.5,
                "Short Strangle": 1.6,
                "Calendar Spread": 1.3,
                "Jade Lizard": 1.4,
                "Short Straddle": 1.5
            }
            return base_slippage * strategy_multipliers.get(strategy, 1.0) * iv_multiplier * dte_factor

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

                max_strategy_loss = premium * 0.6 if strategy in ["Iron Fly", "Iron Condor"] else premium * 0.8
                loss = min(loss, max_strategy_loss)
                pnl = premium - loss

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
            logger.warning(f"No trades generated for {strategy_choice} from {start_date} to {end_date}")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max() if len(backtest_df) > 0 else 0

        backtest_df.set_index("Date", inplace=True)
        df = df.groupby(df.index).last()
        returns = backtest_df["PnL"] / df["Total_Capital"].reindex(backtest_df.index, method="ffill").fillna(capital)
        nifty_returns = df["NIFTY_Close"].pct_change().reindex(backtest_df.index, method="ffill").fillna(0)
        excess_returns = returns - nifty_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(126) if excess_returns.std() != 0 else 0
        sortino_ratio = excess_returns.mean() / excess_returns[excess_returns < 0].std() * np.sqrt(126) if len(excess_returns[excess_returns < 0]) > 0 and excess_returns[excess_returns < 0].std() != 0 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 else float('inf')

        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_perf["Win_Rate"] = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_perf["Win_Rate"] = backtest_df.groupby("Regime")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)

        logger.debug("Backtest completed successfully")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"❌ Error running backtest: {str(e)}")
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

# Strategy Generation (Multiple Strategies)
@st.cache_data
def generate_trading_strategies(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
    try:
        logger.info("Generating trading strategies")
        df = df.copy()
        df.index = df.index.normalize()
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
                return []

        if event_flag == 1:
            regime = "EVENT-DRIVEN"
        elif avg_vol < 15:
            regime = "LOW"
        elif avg_vol < 20:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        strategies = []
        confidence = 0.5 + 0.5 * (1 - np.abs(forecast_log["GARCH_Vol"].iloc[0] - forecast_log["XGBoost_Vol"].iloc[0]) / max(forecast_log["GARCH_Vol"].iloc[0], forecast_log["XGBoost_Vol"].iloc[0]))
        capital_alloc = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]

        # Strategy 1: Primary Strategy
        if regime == "LOW":
            strategy = "Iron Fly"
            reason = "Low volatility and time decay favors delta-neutral Iron Fly"
            tags = ["Neutral", "Theta", "Range Bound"]
            risk_reward = 1.5
            if iv_hv_gap > 5 and dte < 10:
                strategy = "Butterfly Spread"
                reason = "Low vol & short expiry favors pinning strategies"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.0
        elif regime == "MEDIUM":
            strategy = "Short Strangle"
            reason = "Balanced vol, no events, and premium-rich environment"
            tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
            risk_reward = 1.5
            if iv_hv_gap > 3 and iv_skew > 2:
                strategy = "Iron Condor"
                reason = "Medium vol and skew favor wide-range Iron Condor"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8
        elif regime == "HIGH":
            strategy = "Iron Condor"
            reason = "High vol favors wide-range Iron Condor for premium collection"
            tags = ["Neutral", "Theta", "Range Bound"]
            risk_reward = 1.5
            if iv_hv_gap > 10:
                strategy = "Jade Lizard"
                reason = "High IV + call skew = Jade Lizard for defined upside risk"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.2
        elif regime == "EVENT-DRIVEN":
            strategy = "Calendar Spread"
            reason = "Event-based uncertainty favors term structure opportunity"
            tags = ["Volatility", "Event", "Calendar"]
            risk_reward = 1.3
            if iv > 30 and dte < 5:
                strategy = "Short Straddle"
                reason = "Event + near expiry + IV spike → high premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.5

        deploy = capital * capital_alloc.get(regime, 0.2) * position_size
        max_loss = deploy * 0.2
        total_exposure = deploy / capital
        behavior_score = 8 if deploy < 0.5 * capital else 6
        behavior_warnings = ["Consider reducing position size"] if behavior_score < 7 else []

        strategies.append({
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
        })

        # Strategy 2: Alternative for the Regime
        if regime == "LOW":
            strategy = "Short Strangle"
            reason = "Low vol alternative for premium selling"
            tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
            risk_reward = 1.4
        elif regime == "MEDIUM":
            strategy = "Iron Fly"
            reason = "Medium vol alternative with tighter range"
            tags = ["Neutral", "Theta", "Range Bound"]
            risk_reward = 1.5
        elif regime == "HIGH":
            strategy = "Short Straddle"
            reason = "High vol alternative for aggressive premium capture"
            tags = ["Volatility", "Neutral"]
            risk_reward = 1.3
        elif regime == "EVENT-DRIVEN":
            strategy = "Jade Lizard"
            reason = "Event-driven alternative with defined upside risk"
            tags = ["Skewed", "Volatility", "Defined Risk"]
            risk_reward = 1.2

        deploy = capital * capital_alloc.get(regime, 0.2) * position_size * 0.8
        max_loss = deploy * 0.2
        total_exposure = deploy / capital
        strategies.append({
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence * 0.95,
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure,
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings
        })

        # Strategy 3: Defensive Option
        strategy = "Iron Condor"
        reason = "Defensive strategy with wide range for stability"
        tags = ["Neutral", "Theta", "Range Bound"]
        risk_reward = 1.8
        deploy = capital * capital_alloc.get(regime, 0.2) * position_size * 0.6
        max_loss = deploy * 0.2
        total_exposure = deploy / capital
        strategies.append({
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence * 0.9,
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure,
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings
        })

        # Strategy 4: High Confidence Option (if applicable)
        if iv_hv_gap > 5:
            strategy = "Short Straddle"
            reason = "High IV-HV gap favors aggressive premium capture"
            tags = ["Volatility", "Neutral"]
            risk_reward = 1.5
            deploy = capital * capital_alloc.get(regime, 0.2) * position_size * 0.5
            max_loss = deploy * 0.2
            total_exposure = deploy / capital
            strategies.append({
                "Regime": regime,
                "Strategy": strategy,
                "Reason": reason,
                "Tags": tags,
                "Confidence": confidence * 0.85,
                "Risk_Reward": risk_reward,
                "Deploy": deploy,
                "Max_Loss": max_loss,
                "Exposure": total_exposure,
                "Risk_Flags": risk_flags,
                "Behavior_Score": behavior_score,
                "Behavior_Warnings": behavior_warnings
            })

        logger.debug(f"Generated {len(strategies)} trading strategies")
        return strategies
    except Exception as e:
        st.error(f"❌ Error generating strategies: {str(e)}")
        logger.error(f"Error generating strategies: {str(e)}")
        return []

# Trading Functions
def place_trade(client, strategy, real_data, capital):
    try:
        logger.info(f"Attempting to place trade: {strategy['Strategy']}")
        if not real_data or "option_chain" not in real_data or "atm_strike" not in real_data or "expiry" not in real_data:
            logger.error("Invalid or missing real-time data")
            return False, "Trade rejected: Invalid or missing market data from 5paisa API"

        if not client.get_access_token():
            logger.error("Client session invalid or expired")
            return False, "Trade rejected: 5paisa API session invalid. Please re-login."

        option_chain = real_data["option_chain"]
        atm_strike = real_data["atm_strike"]
        lot_size = 25
        deploy = strategy["Deploy"]
        max_loss = strategy["Max_Loss"]
        expiry = real_data["expiry"]

        required_columns = ["ScripCode", "StrikeRate", "CPType", "LastRate"]
        if not all(col in option_chain.columns for col in required_columns):
            logger.error(f"Option chain missing required columns: {required_columns}")
            return False, f"Trade rejected: Option chain missing required columns: {required_columns}"

        try:
            margin_data = client.margin()
            if not isinstance(margin_data, dict):
                logger.error("Invalid margin data response")
                return False, "Trade rejected: Unable to fetch margin data"
            available_margin = margin_data.get("AvailableMargin", 0)
            estimated_margin = deploy * 1.2
            if available_margin < estimated_margin:
                logger.error(f"Insufficient margin: Available={available_margin}, Required={estimated_margin}")
                return False, f"Trade rejected: Insufficient margin (Available: ₹{available_margin:,.2f}, Required: ₹{estimated_margin:,.2f})"
        except Exception as e:
            logger.error(f"Error checking margin: {str(e)}")
            return False, f"Trade rejected: Unable to verify margin requirements ({str(e)})"

        premium_per_lot = real_data["straddle_price"] * lot_size
        if premium_per_lot <= 0:
            logger.error("Invalid straddle price for lot calculation")
            return False, "Trade rejected: Invalid straddle price for lot calculation"
        lots = max(1, min(int(deploy / premium_per_lot), int(max_loss / (premium_per_lot * 0.2))))
        if lots <= 0:
            logger.error("Calculated lots is zero or negative")
            return False, "Trade rejected: Invalid lot size calculated"

        orders = []
        strikes = []
        if strategy["Strategy"] == "Short Straddle":
            strikes = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")]
        elif strategy["Strategy"] == "Short Strangle":
            call_strike = atm_strike + 100
            put_strike = atm_strike - 100
            strikes = [(call_strike, "CE", "S"), (put_strike, "PE", "S")]
        elif strategy["Strategy"] == "Iron Condor":
            call_sell_strike = atm_strike + 100
            call_buy_strike = call_sell_strike + 100
            put_sell_strike = atm_strike - 100
            put_buy_strike = put_sell_strike - 100
            strikes = [
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B"),
                (put_sell_strike, "PE", "S"),
                (put_buy_strike, "PE", "B")
            ]
        elif strategy["Strategy"] == "Iron Fly":
            call_sell_strike = atm_strike
            call_buy_strike = atm_strike + 100
            put_sell_strike = atm_strike
            put_buy_strike = atm_strike - 100
            strikes = [
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B"),
                (put_sell_strike, "PE", "S"),
                (put_buy_strike, "PE", "B")
            ]
        elif strategy["Strategy"] == "Butterfly Spread":
            call_sell_strike = atm_strike
            call_buy_strike = atm_strike + 200
            put_buy_strike = atm_strike - 200
            strikes = [
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B"),
                (put_buy_strike, "PE", "B")
            ]
        elif strategy["Strategy"] == "Jade Lizard":
            call_sell_strike = atm_strike + 100
            put_sell_strike = atm_strike - 100
            put_buy_strike = atm_strike - 200
            strikes = [
                (call_sell_strike, "CE", "S"),
                (put_sell_strike, "PE", "S"),
                (put_buy_strike, "PE", "B")
            ]
        elif strategy["Strategy"] == "Calendar Spread":
            call_sell_strike = atm_strike
            call_buy_strike = atm_strike
            strikes = [
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B")
            ]
        else:
            logger.error(f"Unsupported strategy: {strategy['Strategy']}")
            return False, f"Trade rejected: Unsupported strategy ({strategy['Strategy']})"

        for strike, cp_type, buy_sell in strikes:
            opt_data = option_chain[(option_chain["StrikeRate"] == strike) & (option_chain["CPType"] == cp_type)]
            if opt_data.empty:
                logger.error(f"No option data for {cp_type} at strike {strike}")
                return False, f"Trade rejected: No option data for {cp_type} at strike {strike}"

            scrip_code = opt_data["ScripCode"].iloc[0]
            price = opt_data["LastRate"].iloc[0]
            if not isinstance(scrip_code, (int, float)) or pd.isna(scrip_code):
                logger.error(f"Invalid ScripCode for {cp_type} at strike {strike}")
                return False, f"Trade rejected: Invalid ScripCode for {cp_type} at strike {strike}"
            if not isinstance(price, (int, float)) or pd.isna(price) or price <= 0:
                logger.error(f"Invalid LastRate for {cp_type} at strike {strike}")
                return False, f"Trade rejected: Invalid LastRate for {cp_type} at strike {strike}"

            order = {
                "Exchange": "N",
                "ExchangeType": "D",
                "ScripCode": int(scrip_code),
                "Quantity": lot_size * lots,
                "Price": 0,
                "OrderType": "SELL" if buy_sell == "S" else "BUY",
                "IsIntraday": False
            }
            orders.append(order)

        max_retries = 3
        for order in orders:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Placing order: {order}")
                    response = client.place_order(
                        OrderType=order["OrderType"],
                        Exchange=order["Exchange"],
                        ExchangeType=order["ExchangeType"],
                        ScripCode=order["ScripCode"],
                        Qty=order["Quantity"],
                        Price=order["Price"],
                        IsIntraday=order["IsIntraday"]
                    )
                    if not isinstance(response, dict) or "Status" not in response:
                        raise Exception(f"Invalid API response for ScripCode {order['ScripCode']}")
                    if response.get("Status") != 0:
                        error_message = response.get("Message", "Unknown error")
                        logger.error(f"Order failed for ScripCode {order['ScripCode']}: {error_message}")
                        return False, f"Trade rejected: Order failed for ScripCode {order['ScripCode']} ({error_message})"
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to place order for ScripCode {order['ScripCode']} after {max_retries} attempts: {str(e)}")
                        return False, f"Trade rejected: Failed to place order for ScripCode {order['ScripCode']} ({str(e)})"
                    logger.warning(f"Order placement attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    continue

        logger.info(f"Trade placed successfully: {strategy['Strategy']} with {lots} lots")
        return True, f"Trade placed successfully: {strategy['Strategy']} with {lots} lots"
    except Exception as e:
        logger.error(f"Error placing trade: {str(e)}")
        return False, f"Trade rejected: Unexpected error ({str(e)})"

def square_off_positions(client):
    try:
        logger.info("Squaring off all positions")
        with st.spinner("Squaring off positions..."):
            positions = client.positions()
            if not positions:
                logger.info("No positions to square off")
                return True, "No open positions to square off"

            for position in positions:
                scrip_code = position.get("ScripCode")
                qty = position.get("OpenQty", 0)
                if qty == 0:
                    continue

                order = {
                    "Exchange": position.get("Exchange", "N"),
                    "ExchangeType": position.get("ExchangeType", "D"),
                    "ScripCode": scrip_code,
                    "Quantity": qty,
                    "Price": 0,
                    "OrderType": "SELL" if position.get("BuySell") == "B" else "BUY",
                    "IsIntraday": False
                }
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
                    error_message = response.get("Message", "Unknown error")
                    logger.error(f"Failed to square off position for ScripCode {scrip_code}: {error_message}")
                    return False, f"Failed to square off position for ScripCode {scrip_code}: {error_message}"

            logger.info("All positions squared off successfully")
            return True, "All positions squared off successfully"
    except Exception as e:
        logger.error(f"Error squaring off positions: {str(e)}")
        return False, f"Error squaring off positions: {str(e)}"

# Main App
st.title("🛡️ VolGuard Pro - Option Seller's Copilot")

# Sidebar for Authentication and Settings
with st.sidebar:
    st.header("Account Setup")
    totp_code = st.text_input("Enter 5paisa TOTP Code", type="password")
    if st.button("Login"):
        client = initialize_5paisa_client(totp_code)
        if client:
            st.session_state.client = client
            st.session_state.logged_in = True
            st.success("✅ Logged in successfully")
        else:
            st.session_state.logged_in = False
            st.session_state.client = None

    if st.session_state.logged_in:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.client = None
            st.success("✅ Logged out successfully")

    st.header("Settings")
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
    risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
    forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 10, 5)

# Main Content
if not st.session_state.logged_in:
    st.warning("Please login to continue")
else:
    client = st.session_state.client
    df, real_data, data_source_info = load_data(client)
    if df is None:
        st.error("Failed to load data. Please try again.")
        st.stop()

    df = generate_features(df, real_data, capital)
    if df is None:
        st.error("Failed to generate features. Please try again.")
        st.stop()

    forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df, forecast_horizon)
    if forecast_log is None:
        st.error("Volatility forecasting failed. Please try again.")
        st.stop()

    portfolio_data = fetch_portfolio_data(client, capital)

    # Tabs
    tabs = st.tabs(["📊 Snapshot", "🔮 Forecast", "🎯 Strategies", "💼 Portfolio", "📓 Journal", "📉 Backtest"])

    # Snapshot Tab
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Market Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("NIFTY 50", f"₹{df['NIFTY_Close'].iloc[-1]:,.2f}")
        with col2:
            st.metric("India VIX", f"{df['VIX'].iloc[-1]:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:.2f}%")
        with col3:
            st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}")
        with col4:
            st.metric("Straddle Price", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}")
        st.markdown(f'<div class="data-tag">Data Source: {data_source_info["source"]} | Last Updated: {data_source_info["timestamp"]}</div>',
        unsafe_allow_html=True
        )

        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Max Pain Strike", f"₹{real_data['max_pain_strike']:,.2f}" if real_data else "N/A")
        with col6:
            st.metric("Max Pain Diff", f"{real_data['max_pain_diff_pct']:.2f}%" if real_data else "N/A")
        with col7:
            st.metric("Days to Expiry", f"{int(df['Days_to_Expiry'].iloc[-1])}")

        st.markdown('<div class="regime-badge regime-' + ('event' if df['Event_Flag'].iloc[-1] == 1 else 'low' if blended_vols[0] < 15 else 'medium' if blended_vols[0] < 20 else 'high') + '">' +
                    ('Event-Driven' if df['Event_Flag'].iloc[-1] == 1 else 'Low Vol' if blended_vols[0] < 15 else 'Medium Vol' if blended_vols[0] < 20 else 'High Vol') + '</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Market Trends")
        # Ensure data is valid before plotting
        df_last_30 = df.iloc[-30:].copy()
        if len(df_last_30) < 2 or df_last_30['NIFTY_Close'].isna().all() or df_last_30['VIX'].isna().all():
            st.warning("Insufficient or invalid data to plot market trends. Please ensure NIFTY 50 and VIX data are available.")
        else:
            trend_fig = go.Figure()
            trend_fig.add_trace(go.Scatter(x=df_last_30.index, y=df_last_30['NIFTY_Close'], mode='lines', name='NIFTY 50', line=dict(color='#e94560')))
            trend_fig.add_trace(go.Scatter(x=df_last_30.index, y=df_last_30['VIX'], mode='lines', name='India VIX', line=dict(color='#ffcc00'), yaxis='y2'))
            trend_fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title='NIFTY 50', titlefont=dict(color='#e94560'), tickfont=dict(color='#e94560')),
                yaxis2=dict(title='India VIX', titlefont=dict(color='#ffcc00'), tickfont=dict(color='#ffcc00'), overlaying='y', side='right'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                plot_bgcolor='#1a1a2e',
                paper_bgcolor='#1a1a2e',
                font=dict(color='#e5e5e5')
            )
            st.plotly_chart(trend_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Forecast Tab
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Volatility Forecast")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Realized Vol", f"{realized_vol:.2f}%")
        with col2:
            st.metric("Forecasted Vol (Blended)", f"{blended_vols[0]:.2f}%")
        with col3:
            st.metric("Confidence Score", f"{confidence_score:.2f}%")

        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(x=forecast_log['Date'], y=forecast_log['GARCH_Vol'], mode='lines', name='GARCH', line=dict(color='#e94560')))
        forecast_fig.add_trace(go.Scatter(x=forecast_log['Date'], y=forecast_log['XGBoost_Vol'], mode='lines', name='XGBoost', line=dict(color='#ffcc00')))
        forecast_fig.add_trace(go.Scatter(x=forecast_log['Date'], y=forecast_log['Blended_Vol'], mode='lines', name='Blended', line=dict(color='#28a745')))
        forecast_fig.update_layout(
            title="Volatility Forecast",
            template='plotly_dark',
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(title='Volatility (%)'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(forecast_fig, use_container_width=True)
        st.markdown(f'<div class="data-tag">Model RMSE: {rmse:.2f} | Data Source: {data_source_info["source"]} | Last Updated: {data_source_info["timestamp"]}</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': feature_importances})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="XGBoost Feature Importance")
        importance_fig.update_layout(template='plotly_dark', height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(importance_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Strategies Tab
    with tabs[2]:
        strategies = generate_trading_strategies(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital)
        if not strategies:
            st.error("No strategies available. Please check your inputs or journal requirements.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Trading Strategies")
            st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True)
            for idx, strategy in enumerate(strategies):
                with st.container():
                    st.markdown(f'<div class="strategy-card">', unsafe_allow_html=True)
                    st.markdown(f"**{strategy['Strategy']}**")
                    st.markdown(f"**Regime:** {strategy['Regime']}")
                    st.markdown(f"**Reason:** {strategy['Reason']}")
                    st.markdown(f"**Tags:** {', '.join(strategy['Tags'])}")
                    st.markdown(f"**Confidence:** {strategy['Confidence']:.2%}")
                    st.markdown(f"**Risk/Reward:** {strategy['Risk_Reward']:.2f}")
                    st.markdown(f"**Capital to Deploy:** ₹{strategy['Deploy']:,.2f}")
                    st.markdown(f"**Max Loss:** ₹{strategy['Max_Loss']:,.2f}")
                    st.markdown(f"**Exposure:** {strategy['Exposure']:.2%}")
                    if strategy['Risk_Flags']:
                        st.markdown(f"**Risk Flags:** {', '.join(strategy['Risk_Flags'])}")
                    if strategy['Behavior_Warnings']:
                        st.markdown(f"**Behavior Warnings:** {', '.join(strategy['Behavior_Warnings'])}")
                    st.markdown(f"**Behavior Score:** {strategy['Behavior_Score']}/10")

                    if st.button(f"Execute {strategy['Strategy']}", key=f"execute_{idx}"):
                        success, message = place_trade(client, strategy, real_data, capital)
                        if success:
                            st.success(message)
                            st.session_state.trades.append({
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "Strategy": strategy['Strategy'],
                                "Capital_Deployed": strategy['Deploy'],
                                "Max_Loss": strategy['Max_Loss'],
                                "Status": "Open"
                            })
                        else:
                            st.error(message)
                    st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Square Off All Positions"):
                success, message = square_off_positions(client)
                if success:
                    st.success(message)
                    for trade in st.session_state.trades:
                        trade["Status"] = "Closed"
                else:
                    st.error(message)

            if st.session_state.trades:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Active Trades")
                trades_df = pd.DataFrame(st.session_state.trades)
                st.dataframe(trades_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    # Portfolio Tab
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Portfolio Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Weekly PnL", f"₹{portfolio_data['weekly_pnl']:,.2f}")
        with col2:
            st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
        with col3:
            st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
        st.markdown(f'<div class="data-tag">Data Source: {portfolio_data["source"]} | Last Updated: {portfolio_data["timestamp"]}</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("PnL Trend")
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'])
            trades_df = trades_df.sort_values('Timestamp')
            trades_df['Cumulative_PnL'] = trades_df['Capital_Deployed'].cumsum() - trades_df['Max_Loss'].cumsum()
            pnl_fig = go.Figure()
            pnl_fig.add_trace(go.Scatter(x=trades_df['Timestamp'], y=trades_df['Cumulative_PnL'], mode='lines', name='Cumulative PnL', line=dict(color='#e94560')))
            pnl_fig.update_layout(
                title="Cumulative PnL Over Time",
                template='plotly_dark',
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(title='PnL (₹)'),
            )
            st.plotly_chart(pnl_fig, use_container_width=True)
        else:
            st.info("No trades executed yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Journal Tab
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Trading Journal")
        with st.form("journal_form"):
            trade_reflection = st.text_area("Reflect on your recent trade: What went well? What could be improved?")
            lessons_learned = st.text_area("Lessons Learned from this Trade")
            emotional_state = st.selectbox("Emotional State During Trade", ["Calm", "Stressed", "Excited", "Anxious"])
            discipline_score = st.slider("Discipline Score (1-10)", 1, 10, 5)
            submitted = st.form_submit_button("Submit Journal Entry")
            if submitted:
                st.session_state.journal_complete = True
                st.session_state.violations = 0
                st.success("Journal entry submitted successfully! Trading unlocked.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Backtest Tab
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Backtest Strategies")
        strategy_choice = st.selectbox("Select Strategy to Backtest", ["All Strategies", "Iron Condor", "Iron Fly", "Short Strangle", "Short Straddle", "Butterfly Spread", "Jade Lizard", "Calendar Spread"])
        start_date = st.date_input("Start Date", value=df.index.min().date(), min_value=df.index.min().date(), max_value=df.index.max().date())
        end_date = st.date_input("End Date", value=df.index.max().date(), min_value=df.index.min().date(), max_value=df.index.max().date())

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(
                    df, capital, strategy_choice, pd.Timestamp(start_date), pd.Timestamp(end_date)
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

        if st.session_state.backtest_run and st.session_state.backtest_results:
            results = st.session_state.backtest_results
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total PnL", f"₹{results['total_pnl']:,.2f}")
            with col2:
                st.metric("Win Rate", f"{results['win_rate']:.2%}")
            with col3:
                st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.2f}")
            with col4:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")

            col5, col6 = st.columns(2)
            with col5:
                st.metric("Sortino Ratio", f"{results['sortino_ratio']:.2f}")
            with col6:
                st.metric("Calmar Ratio", f"{results['calmar_ratio']:.2f}")

            st.markdown('<div class="expander-header">Backtest PnL Trend</div>', unsafe_allow_html=True)
            backtest_fig = go.Figure()
            backtest_fig.add_trace(go.Scatter(x=results['backtest_df'].index, y=results['backtest_df']['PnL'].cumsum(), mode='lines', name='Cumulative PnL', line=dict(color='#e94560')))
            backtest_fig.update_layout(
                title="Backtest Cumulative PnL",
                template='plotly_dark',
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis=dict(title='PnL (₹)'),
            )
            st.plotly_chart(backtest_fig, use_container_width=True)

            st.markdown('<div class="expander-header">Strategy Performance</div>', unsafe_allow_html=True)
            st.dataframe(results['strategy_perf'], use_container_width=True)

            st.markdown('<div class="expander-header">Regime Performance</div>', unsafe_allow_html=True)
            st.dataframe(results['regime_perf'], use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">VolGuard Pro © 2025 | Built with ❤ by Shritish & Salman</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.write("Running VolGuard Pro...")
