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
from scipy.stats import norm
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG)
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
if "last_monitor_time" not in st.session_state:
    st.session_state.last_monitor_time = 0
if "total_exposure" not in st.session_state:
    st.session_state.total_exposure = 0

# 5paisa Client Initialization
def initialize_5paisa_client(totp_code):
    try:
        logger.info("Initializing 5paisa client")
        cred = {
            "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
            "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
            "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
            "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
            "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
            "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
        }
        client = FivePaisaClient(cred=cred)
        client.get_totp_session(
            st.secrets["fivepaisa"]["CLIENT_CODE"],
            totp_code,
            st.secrets["fivepaisa"]["PIN"]
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

# Data Fetching
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
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None

def fetch_nifty_data(client):
    try:
        logger.info("Fetching real-time data from 5paisa API")
        req_list = [
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920000},  # NIFTY 50
            {"Exch": "N", "ExchType": "C", "ScripCode": 999920005}   # India VIX
        ]
        market_feed = client.fetch_market_feed(req_list)
        if not market_feed or "Data" not in market_feed or len(market_feed["Data"]) < 2:
            raise Exception("Failed to fetch NIFTY 50 or India VIX")

        nifty_data = market_feed["Data"][0]
        vix_data = market_feed["Data"][1]
        nifty_spot = nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0))
        vix = vix_data.get("LastRate", vix_data.get("LastTradedPrice", 0))
        if not nifty_spot or not vix:
            raise Exception("Missing NIFTY or VIX price")

        expiries = client.get_expiry("N", "NIFTY")
        if not expiries or "Data" not in expiries:
            raise Exception("Failed to fetch expiries")
        expiry_timestamp = expiries["Data"][0]["Timestamp"]
        option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
        if not option_chain or "Options" not in option_chain:
            raise Exception("Failed to fetch option chain")

        df = pd.DataFrame(option_chain["Options"])
        if not all(col in df.columns for col in ["StrikeRate", "CPType", "LastRate", "OpenInterest"]):
            raise Exception("Required columns missing")

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
            raise Exception("Max pain calculation failed")

        vix_change_pct = 0
        iv_file = "vix_history.csv"
        if os.path.exists(iv_file):
            iv_history = pd.read_csv(iv_file)
            prev_vix = iv_history["VIX"].iloc[-1] if not iv_history.empty else vix
            vix_change_pct = ((vix - prev_vix) / prev_vix * 100) if prev_vix != 0 else 0
        pd.DataFrame({"Date": [datetime.now()], "VIX": [vix]}).to_csv(iv_file, mode='a', header=not os.path.exists(iv_file), index=False)

        logger.info("Real-time data fetched successfully")
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
            "expiry": expiries["Data"][0]["ExpiryDate"]
        }
    except Exception as e:
        logger.error(f"Error fetching 5paisa API data: {str(e)}")
        return None

@st.cache_data
def load_data(client):
    try:
        logger.info("Loading data")
        real_data = fetch_nifty_data(client) if client else None
        data_source = "5paisa API (LIVE)" if real_data else "CSV (FALLBACK)"
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
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})
            nifty = nifty[~nifty.index.duplicated(keep='last')]

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)
            vix = vix[~vix.index.duplicated(keep='last')]

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates)
            df = df.ffill().bfill()
        else:
            latest_date = datetime.now().date()
            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["vix"]]
            }, index=[pd.to_datetime(latest_date)])

            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})
            nifty = nifty[~nifty.index.duplicated(keep='last')]

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)
            vix = vix[~vix.index.duplicated(keep='last')]

            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"],
                "VIX": vix["VIX"]
            }).dropna()
            historical_df = historical_df[historical_df.index < pd.to_datetime(latest_date)]
            historical_df = historical_df[~historical_df.index.duplicated(keep='last')]
            df = pd.concat([historical_df, df])
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

        logger.debug(f"Data loaded successfully from {data_source}, rows: {len(df)}")
        return df, real_data, data_source
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None

def fetch_portfolio_data(client, capital):
    try:
        logger.info("Fetching portfolio data")
        positions = client.positions()
        if not positions:
            logger.warning("No positions found")
            return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "delta": 0}

        total_pnl = 0
        total_margin = 0
        total_exposure = 0
        total_delta = 0
        for position in positions:
            total_pnl += position.get("ProfitLoss", 0)
            total_margin += position.get("MarginUsed", 0)
            total_exposure += position.get("Exposure", 0)
            total_delta += position.get("Delta", 0)
        total_exposure = total_exposure / capital * 100 if capital > 0 else 0

        logger.info("Portfolio data fetched successfully")
        return {
            "weekly_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure": total_exposure,
            "delta": total_delta
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "delta": 0}

# Risk Management Functions
def calculate_option_delta(S, K, T, r, sigma, option_type):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == "CE":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        return delta
    except Exception as e:
        logger.error(f"Error calculating delta: {str(e)}")
        return 0

def dynamic_stop_loss(vix):
    if vix < 15:
        return 0.25
    elif vix < 20:
        return 0.20
    else:
        return 0.15

def dynamic_max_loss(vix):
    return dynamic_stop_loss(vix)

def hedge_delta(client, total_delta, nifty_spot, capital):
    try:
        futures_lot_size = 25
        contracts_needed = int(-total_delta)
        if abs(contracts_needed * futures_lot_size * nifty_spot) > 0.1 * capital:
            contracts_needed = int(0.1 * capital / (futures_lot_size * nifty_spot)) * np.sign(contracts_needed)

        if abs(contracts_needed) < 1:
            return True, "Delta within neutral band (±50)"

        order = {
            "Exchange": "N",
            "ExchangeType": "C",
            "ScripCode": 999920000,
            "Quantity": abs(contracts_needed) * futures_lot_size,
            "Price": 0,
            "OrderType": "BUY" if contracts_needed > 0 else "SELL",
            "IsIntraday": False
        }
        response = client.place_order(**order)
        if response.get("Status") == 0:
            logger.info(f"Delta hedge: {order['OrderType']} {order['Quantity']} futures")
            return True, f"Hedged: {order['OrderType']} {order['Quantity']}"
        return False, f"Hedge failed: {response.get('Message', 'Unknown error')}"
    except Exception as e:
        logger.error(f"Error hedging delta: {str(e)}")
        return False, f"Error: {str(e)}"

def monitor_positions(client, capital, real_data, strategy):
    try:
        current_time = time.time()
        if current_time - st.session_state.last_monitor_time < 900:
            return True, "Monitoring interval not reached"
        st.session_state.last_monitor_time = current_time

        logger.info("Monitoring positions")
        positions = client.positions()
        if not positions:
            return True, "No open positions"

        total_pnl = sum(pos.get("ProfitLoss", 0) for pos in positions)
        total_exposure = sum(pos.get("Exposure", 0) for pos in positions) / capital * 100
        vix = real_data["vix"] if real_data else 15
        vix_change_pct = real_data["vix_change_pct"] if real_data else 0

        stop_loss = strategy["Deploy"] * dynamic_stop_loss(vix) if strategy else 0.2 * capital
        if total_pnl < -stop_loss:
            success = square_off_positions(client)
            st.session_state.violations += 1
            return False, f"Stop-loss triggered: Squared off (P&L: ₹{total_pnl:,.2f})"

        if total_pnl < -0.1 * capital:
            success = square_off_positions(client)
            st.session_state.violations += 1
            return False, f"Drawdown exceeded: Squared off (P&L: ₹{total_pnl:,.2f})"

        if vix_change_pct > 20:
            success = square_off_positions(client)
            st.session_state.violations += 1
            return False, f"VIX spike: Squared off ({vix_change_pct:.2f}%)"

        now = datetime.now()
        if now.hour == 15 and now.minute >= 25 and now.weekday() < 5:
            success = square_off_positions(client)
            return success, "End of day: Squared off"

        total_delta = 0
        if real_data and "option_chain" in real_data:
            nifty_spot = real_data["nifty_spot"]
            for pos in positions:
                opt_data = real_data["option_chain"][real_data["option_chain"]["ScripCode"] == pos.get("ScripCode")]
                if not opt_data.empty:
                    strike = opt_data["StrikeRate"].iloc[0]
                    option_type = opt_data["CPType"].iloc[0]
                    T = max(1/365, (pd.to_datetime(real_data["expiry"]) - now).days / 365)
                    sigma = vix / 100
                    delta = calculate_option_delta(nifty_spot, strike, T, 0.06, sigma, option_type)
                    total_delta += delta * pos.get("Quantity", 0) * (-1 if pos.get("OrderType") == "SELL" else 1)
            if abs(total_delta) > 50:
                success, message = hedge_delta(client, total_delta, nifty_spot, capital)
                if not success:
                    return False, f"Delta hedging failed: {message}"

        return True, "Positions monitored successfully"
    except Exception as e:
        logger.error(f"Error monitoring positions: {str(e)}")
        return False, f"Error: {str(e)}"

def square_off_positions(client):
    try:
        logger.info("Squaring off all positions")
        response = client.squareoff_all()
        if response.get("Status") == 0:
            st.session_state.total_exposure = 0
            logger.info("All positions squared off")
            return True
        return False
    except Exception as e:
        logger.error(f"Error squaring off: {str(e)}")
        return False

# Feature Generation
@st.cache_data
def generate_features(df, real_data, capital):
    try:
        logger.info("Generating features")
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
        df["Straddle_Price"] = np.clip(np.random.normal(base_straddle_price, 50, n_days), 50, 400)
        if real_data:
            df["Straddle_Price"].iloc[-1] = base_straddle_price

        if df.isna().sum().sum() > 0:
            df = df.interpolate().fillna(method='bfill')

        df.to_csv("volguard_hybrid_data.csv")
        logger.debug("Features generated successfully")
        return df
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
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
        df.index = pd.to_datetime(df.index)
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
        st.error(f"Error in volatility forecasting: {str(e)}")
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None

# Backtesting
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.info(f"Starting backtest for {strategy_choice}")
        if df.empty:
            st.error("Backtest failed: No data available")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        df_backtest = df.loc[start_date:end_date].copy()
        if len(df_backtest) < 50:
            st.error(f"Backtest failed: Insufficient data ({len(df_backtest)} days)")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        required_cols = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price"]
        missing_cols = [col for col in required_cols if col not in df_backtest.columns]
        if missing_cols:
            st.error(f"Backtest failed: Missing columns {missing_cols}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        backtest_results = []
        lot_size = 25
        base_transaction_cost = 0.002
        stt = 0.0005
        portfolio_pnl = 0
        risk_free_rate = 0.06 / 126
        nifty_returns = df_backtest["NIFTY_Close"].pct_change()

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
                        reason = "Balanced vol, premium-rich environment"
                        tags = ["Neutral", "Premium Selling"]

                elif regime == "HIGH":
                    if iv_hv_gap > 10:
                        strategy = "Jade Lizard"
                        reason = "High IV + call skew = Jade Lizard"
                        tags = ["Skewed", "Volatility"]
                        risk_reward = 1.2
                    else:
                        strategy = "Iron Condor"
                        reason = "High vol favors wide-range Iron Condor"
                        tags = ["Neutral", "Theta"]

                elif regime == "EVENT-DRIVEN":
                    if iv > 30 and dte < 5:
                        strategy = "Short Straddle"
                        reason = "Event + IV spike → high premium capture"
                        tags = ["Volatility", "Event"]
                        risk_reward = 1.5
                    else:
                        strategy = "Calendar Spread"
                        reason = "Event uncertainty favors term structure"
                        tags = ["Volatility", "Calendar"]

                capital = day_data["Total_Capital"]
                capital_alloc = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.06}
                deploy = capital * capital_alloc.get(regime, 0.06)
                max_loss = deploy * dynamic_max_loss(iv)
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

        for i in range(1, len(df_backtest)):
            try:
                day_data = df_backtest.iloc[i]
                prev_day = df_backtest.iloc[i-1]
                date = day_data.name
                avg_vol = df_backtest["Realized_Vol"].iloc[max(0, i-5):i].mean()

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
                iv_factor = min(day_data["ATM_IV"] / avg_vol, 1.5) if avg_vol != 0 else 1.0
                breakeven_factor = 0.04 if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) else 0.06
                breakeven = entry_price * (1 + iv_factor * breakeven_factor)
                nifty_move = abs(day_data["NIFTY_Close"] - prev_day["NIFTY_Close"])
                loss = max(0, nifty_move - breakeven) * lot_size * lots

                max_strategy_loss = premium * 0.6 if strategy in ["Iron Fly", "Iron Condor"] else premium * 0.8
                loss = min(loss, max_strategy_loss)
                pnl = premium - loss
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
            logger.warning(f"No trades generated for {strategy_choice}")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max() if len(backtest_df) > 0 else 0

        backtest_df.set_index("Date", inplace=True)
        returns = backtest_df["PnL"] / df_backtest["Total_Capital"].reindex(backtest_df.index, method="ffill")
        nifty_returns = df_backtest["NIFTY_Close"].pct_change().reindex(backtest_df.index, method="ffill").fillna(0)
        excess_returns = returns - nifty_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(126) if excess_returns.std() != 0 else 0
        sortino_ratio = excess_returns.mean() / excess_returns[excess_returns < 0].std() * np.sqrt(126) if len(excess_returns[excess_returns < 0]) > 0 and excess_returns[excess_returns < 0].std() != 0 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 else 0

        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_perf["Win_Rate"] = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_perf["Win_Rate"] = backtest_df.groupby("Regime")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)

        logger.debug("Backtest completed successfully")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

# Strategy Generation
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
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

        portfolio_data = fetch_portfolio_data(st.session_state.client, capital)
        risk_flags = []
        if latest["VIX"] > 25:
            risk_flags.append("VIX > 25%")
        if st.session_state.total_exposure > 50:
            risk_flags.append("Exposure > 50%")
        if latest["PnL_Day"] < -0.05 * capital:
            risk_flags.append("Weekly Loss > 5%")
        if latest["VIX_Change_Pct"] > 10:
            risk_flags.append("High VIX Spike")
        if abs(portfolio_data["delta"]) > 100:
            risk_flags.append("High Delta Exposure")
        if portfolio_data["weekly_pnl"] < -dynamic_stop_loss(latest["VIX"]) * capital:
            risk_flags.append("Stop-Loss Breached")

        if risk_flags:
            st.session_state.violations += 1
            if st.session_state.violations >= 3 and not st.session_state.journal_complete:
                st.error("🚨 Discipline Lock: Complete Journaling to Unlock Trading")
                return None, "Discipline lock activated"

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
                reason = "Low vol & short expiry favors pinning strategies"
                tags = ["Neutral", "Theta"]
                risk_reward = 2.0
            else:
                strategy = "Iron Fly"
                reason = "Low volatility favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta"]

        elif regime == "MEDIUM":
            if iv_hv_gap > 3 and iv_skew > 2:
                strategy = "Iron Condor"
                reason = "Medium vol and skew favor wide-range Iron Condor"
                tags = ["Neutral", "Theta"]
                risk_reward = 1.8
            else:
                strategy = "Short Strangle"
                reason = "Balanced vol, premium-rich environment"
                tags = ["Neutral", "Premium Selling"]

        elif regime == "HIGH":
            if iv_hv_gap > 10:
                strategy = "Jade Lizard"
                reason = "High IV + call skew = Jade Lizard"
                tags = ["Skewed", "Volatility"]
                risk_reward = 1.2
            else:
                strategy = "Iron Condor"
                reason = "High vol favors wide-range Iron Condor"
                tags = ["Neutral", "Theta"]

        elif regime == "EVENT-DRIVEN":
            if iv > 30 and dte < 5:
                strategy = "Short Straddle"
                reason = "Event + IV spike → high premium capture"
                tags = ["Volatility", "Event"]
                risk_reward = 1.5
            else:
                strategy = "Calendar Spread"
                reason = "Event uncertainty favors term structure"
                tags = ["Volatility", "Calendar"]

        capital_alloc = {"LOW": 0.30, "MEDIUM": 0.20, "HIGH": 0.10, "EVENT-DRIVEN": 0.15}
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc.get(regime, 0.15) * position_size
        total_exposure = st.session_state.total_exposure + (deploy / capital * 100)
        if total_exposure > 50:
            deploy = (50 - st.session_state.total_exposure) * capital / 100
            if deploy < capital * 0.01:
                return None, "Exposure limit reached (50% of capital)"
        max_loss = deploy * dynamic_max_loss(iv)

        behavior_score = 8 if deploy < 0.3 * capital else 6
        behavior_warnings = ["Consider reducing position size"] if behavior_score < 7 else []

        logger.debug("Trading strategy generated successfully")
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
        return None, f"Error: {str(e)}"

# Trading Functions
def place_trade(client, strategy, real_data, capital):
    try:
        logger.info(f"Placing trade: {strategy['Strategy']}")
        if not real_data or "option_chain" not in real_data or "atm_strike" not in real_data:
            return False, "Invalid real-time data"

        option_chain = real_data["option_chain"]
        atm_strike = real_data["atm_strike"]
        lot_size = 25
        deploy = strategy["Deploy"]
        vix = real_data["vix"]
        nifty_spot = real_data["nifty_spot"]
        expiry = real_data["expiry"]

        premium_per_lot = real_data["straddle_price"] * lot_size
        lots = max(1, int(deploy / premium_per_lot))
        strikes = []
        if strategy["Strategy"] == "Short Straddle":
            strikes = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")]
        elif strategy["Strategy"] == "Short Strangle":
            strikes = [(atm_strike + 100, "CE", "S"), (atm_strike - 100, "PE", "S")]
        elif strategy["Strategy"] == "Iron Condor":
            strikes = [
                (atm_strike + 100, "CE", "S"),
                (atm_strike + 200, "CE", "B"),
                (atm_strike - 100, "PE", "S"),
                (atm_strike - 200, "PE", "B")
            ]
        elif strategy["Strategy"] == "Iron Fly":
            strikes = [
                (atm_strike, "CE", "S"),
                (atm_strike + 100, "CE", "B"),
                (atm_strike, "PE", "S"),
                (atm_strike - 100, "PE", "B")
            ]
        elif strategy["Strategy"] == "Butterfly Spread":
            strikes = [
                (atm_strike, "CE", "S"),
                (atm_strike + 200, "CE", "B"),
                (atm_strike - 200, "PE", "B")
            ]
        elif strategy["Strategy"] == "Jade Lizard":
            strikes = [
                (atm_strike + 100, "CE", "S"),
                (atm_strike - 100, "PE", "S"),
                (atm_strike - 200, "PE", "B")
            ]
        elif strategy["Strategy"] == "Calendar Spread":
            strikes = [(atm_strike, "CE", "S"), (atm_strike, "CE", "B")]
        else:
            return False, "Unsupported strategy"

        total_delta = 0
        T = max(1/365, (pd.to_datetime(expiry) - datetime.now()).days / 365)
        for strike, cp_type, buy_sell in strikes:
            opt_data = option_chain[(option_chain["StrikeRate"] == strike) & (option_chain["CPType"] == cp_type)]
            if opt_data.empty:
                return False, f"No data for {cp_type} at {strike}"
            scrip_code = int(opt_data["ScripCode"].iloc[0])
            delta = calculate_option_delta(nifty_spot, strike, T, 0.06, vix/100, cp_type)
            total_delta += delta * lot_size * lots * (-1 if buy_sell == "S" else 1)
            order = {
                "Exchange": "N",
                "ExchangeType": "D",
                "ScripCode": scrip_code,
                "Quantity": lot_size * lots,
                "Price": 0,
                "OrderType": "SELL" if buy_sell == "S" else "BUY",
                "IsIntraday": False
            }
            response = client.place_order(**order)
            if response.get("Status") != 0:
                return False, f"Order failed: {response.get('Message', 'Unknown error')}"

        if abs(total_delta) > 50:
            success, message = hedge_delta(client, total_delta, nifty_spot, capital)
            if not success:
                return False, f"Trade placed but hedge failed: {message}"

        st.session_state.total_exposure += deploy / capital * 100
        st.session_state.trades.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Strategy": strategy["Strategy"],
            "Regime": strategy["Regime"],
            "Risk_Level": risk_tolerance,
            "Outcome": "Pending"
        })
        return True, f"Trade placed: {strategy['Strategy']} with {lots} lots"
    except Exception as e:
        logger.error(f"Error placing trade: {str(e)}")
        return False, f"Error: {str(e)}"

# Sidebar Login and Controls
with st.sidebar:
    st.header("🔐 5paisa Login")
    totp_code = st.text_input("TOTP (from Authenticator App)", type="password")
    if st.button("Login"):
        client = initialize_5paisa_client(totp_code)
        if client:
            st.session_state.client = client
            st.session_state.logged_in = True
            st.success("✅ Logged in successfully")
        else:
            st.error("❌ Login failed")

    if st.session_state.logged_in:
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

                    strategy = None
                    success, message = monitor_positions(st.session_state.client, capital, real_data, strategy)
                    if not success:
                        st.error(f"❌ {message}")

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

                    with tabs[2]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("🎯 Trading Strategies")
                        strategy = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital)
                        if strategy is None or isinstance(strategy, tuple):
                            st.markdown('<div class="alert-banner">🚨 Discipline Lock: Complete Journaling to Unlock Trading</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f"**Recommended Strategy**: {strategy['Strategy']}")
                            st.markdown(f"**Regime**: {strategy['Regime']}")
                            st.markdown(f"**Reason**: {strategy['Reason']}")
                            st.markdown(f"**Tags**: {', '.join(strategy['Tags'])}")
                            st.markdown(f"**Confidence**: {strategy['Confidence']:.2%}")
                            st.markdown(f"**Risk/Reward**: {strategy['Risk_Reward']:.2f}")
                            st.markdown(f"**Capital to Deploy**: ₹{strategy['Deploy']:,.2f}")
                            st.markdown(f"**Max Loss**: ₹{strategy['Max_Loss']:,.2f}")
                            st.markdown(f"**Exposure**: {strategy['Exposure']:.2f}%")
                            if strategy["Risk_Flags"]:
                                st.markdown("**Risk Flags**: " + "; ".join(strategy["Risk_Flags"]))
                            if strategy["Behavior_Warnings"]:
                                st.markdown("**Behavior Warnings**: " + "; ".join(strategy["Behavior_Warnings"]))
                            if st.button("Execute Trade"):
                                success, message = place_trade(st.session_state.client, strategy, real_data, capital)
                                if success:
                                    st.success(f"✅ {message}")
                                else:
                                    st.error(f"❌ {message}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with tabs[3]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("💼 Portfolio Overview")
                        portfolio_data = fetch_portfolio_data(st.session_state.client, capital)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Weekly P&L", f"₹{portfolio_data['weekly_pnl']:,.2f}")
                        with col2:
                            st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                        with col3:
                            st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
                        with col4:
                            st.metric("Portfolio Delta", f"{portfolio_data['delta']:.2f}")
                        st.markdown("### Open Positions")
                        positions = st.session_state.client.positions()
                        if positions:
                            pos_df = pd.DataFrame(positions)
                            st.dataframe(pos_df[["ScripCode", "Quantity", "ProfitLoss", "Exposure"]], use_container_width=True)
                        else:
                            st.info("No open positions")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with tabs[4]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📝 Trading Journal")
                        if st.session_state.violations >= 3 and not st.session_state.journal_complete:
                            st.markdown('<div class="alert-banner">🚨 Discipline Lock: Complete Journaling to Unlock Trading</div>', unsafe_allow_html=True)
                        with st.form("journal_form"):
                            st.write("Log your trade details")
                            trade_date = st.date_input("Trade Date", value=datetime.now())
                            strategy_used = st.text_input("Strategy Used")
                            outcome = st.selectbox("Outcome", ["Profit", "Loss", "Breakeven"])
                            lessons = st.text_area("Lessons Learned")
                            submitted = st.form_submit_button("Submit Journal Entry")
                            if submitted:
                                st.session_state.trades.append({
                                    "Date": trade_date.strftime("%Y-%m-%d"),
                                    "Strategy": strategy_used,
                                    "Outcome": outcome,
                                    "Lessons": lessons
                                })
                                st.session_state.journal_complete = True
                                st.success("✅ Journal entry submitted. Trading unlocked!")
                        st.markdown("### Trade History")
                        if st.session_state.trades:
                            trades_df = pd.DataFrame(st.session_state.trades)
                            st.dataframe(trades_df, use_container_width=True)
                        else:
                            st.info("No trades logged yet")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with tabs[5]:
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📉 Backtest Results")
                        if st.session_state.backtest_run and st.session_state.backtest_results:
                            results = st.session_state.backtest_results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total P&L", f"₹{results['total_pnl']:,.2f}")
                                st.metric("Win Rate", f"{results['win_rate']:.2%}")
                            with col2:
                                st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.2f}")
                                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                            with col3:
                                st.metric("Sortino Ratio", f"{results['sortino_ratio']:.2f}")
                                st.metric("Calmar Ratio", f"{results['calmar_ratio']:.2f}")
                            st.markdown("### Performance by Strategy")
                            st.dataframe(results["strategy_perf"], use_container_width=True)
                            st.markdown("### Performance by Regime")
                            st.dataframe(results["regime_perf"], use_container_width=True)
                            st.markdown("### P&L Over Time")
                            st.line_chart(results["backtest_df"]["PnL"].cumsum())
                        else:
                            st.info("Run analysis to see backtest results")
                        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim</div>', unsafe_allow_html=True)
