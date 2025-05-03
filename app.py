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
from py5paisa import FivePaisaClient
import time
import os

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Page config
st.set_page_config(page_title="VolGuard", page_icon="🛡️", layout="wide")

# Custom CSS for modern UI (unchanged from your original code)
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
st.markdown("**Protection First, Edge Always** | Built by Shritish Shukla & Salman Azim")

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

# 5paisa Client Setup (Using Streamlit Secrets)
cred = {
    "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
    "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
    "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
    "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
    "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
    "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
}
client = FivePaisaClient(cred=cred)

import streamlit as st
from py5paisa import FivePaisaClient
from py5paisa.order import Order, OrderType, Exchange
import pandas as pd
import logging
import json

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("5paisa")
logging.getLogger("urllib3").setLevel(logging.DEBUG)  # Capture HTTP requests

# Set page config FIRST to avoid StreamlitSetPageConfigMustBeFirstCommandError
st.set_page_config(page_title="5paisa Option Seller Dashboard", layout="wide")

# ---------------------------- Login Section ----------------------------
with st.sidebar:
    st.header("🔐 Login to 5paisa")
    totp_code = st.text_input("Enter TOTP Code (from Authenticator App)", type="password")

    if st.button("Login"):
        try:
            cred = {
                "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
                "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
                "USER_ID": Taliban secrets["fivepaisa"]["USER_ID"],
                "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
                "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
                "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
            }

            logger.debug("Initializing FivePaisaClient with credentials")
            client = FivePaisaClient(cred=cred)
            client_code = st.secrets["fivepaisa"]["CLIENT_CODE"]
            pin = st.secrets["fivepaisa"]["PIN"]

            # Log inputs (mask sensitive data)
            st.write(f"Client Code: {client_code[:4]}****")
            st.write(f"TOTP Code: {totp_code[:2]}****")
            st.write(f"PIN: {'*' * len(pin)}")

            logger.debug("Calling get_totp_session with client_code=%s, totp_code=%s, pin=%s",
                         client_code[:4] + "****", totp_code[:2] + "****", "*" * len(pin))
            response = client.get_totp_session(client_code, totp_code, pin)

            # Debug: Display response
            st.write(f"Response Type: {type(response)}")
            st.write(f"Response: {response}")

            access_token = client.get_access_token()
            st.write(f"Access Token: {access_token}")

            if access_token:
                st.success("✅ Logged in successfully")
                st.session_state.client = client
                st.session_state.logged_in = True
                logger.info("Login successful, access token received")
            else:
                st.error("❌ Login Failed: No token received")
                st.session_state.logged_in = False
                logger.error("Login failed: No access token received")

        except Exception as e:
            st.error(f"Login Error: {str(e)}")
            logger.exception("Exception during login")
            st.session_state.logged_in = False

# ---------------------------- Main Dashboard ----------------------------
if st.session_state.get("logged_in", False):
    client = st.session_state.client
    st.title("5paisa Option Seller Dashboard")

    # ---------------- Account Summary ----------------
    st.subheader("💰 Account Summary")
    try:
        logger.debug("Fetching margin")
        margin = client.margin()
        logger.debug("Margin response: %s", json.dumps(margin, indent=2))

        # Handle list response
        if isinstance(margin, list):
            margin_data = margin[0] if margin else {}
        else:
            margin_data = margin.get('Data', [{}])[0]

        st.metric("Available Margin", f"₹{margin_data.get('CashMarginAvailable', 0):,.2f}")
        st.metric("Used Margin", f"₹{margin_data.get('MarginUsed', 0):,.2f}")
        st.metric("Net Margin", f"₹{margin_data.get('NetCashAvailable', 0):,.2f}")
        logger.info("Margin fetched successfully")
    except Exception as e:
        st.warning(f"Could not fetch margin info: {str(e)}")
        logger.exception("Error fetching margin")

    # ---------------- Holdings ----------------
    st.subheader("📦 Holdings")
    try:
        logger.debug("Fetching holdings")
        holdings = client.holdings()
        logger.debug("Holdings response: %s", json.dumps(holdings, indent=2))

        # Handle list response
        if isinstance(holdings, list):
            df_hold = pd.DataFrame(holdings)
        else:
            df_hold = pd.DataFrame(holdings.get('Data', []))

        if not df_hold.empty:
            st.dataframe(
                df_hold[['ScripName', 'Qty', 'BuyAvgRate', 'LTP', 'CurrentValue']],
                use_container_width=True,
                column_config={
                    "Qty": st.column_config.NumberColumn(format="%d"),
                    "BuyAvgRate": st.column_config.NumberColumn(format="₹%.2f"),
                    "LTP": st.column_config.NumberColumn(format="₹%.2f"),
                    "CurrentValue": st.column_config.NumberColumn(format="₹%.2f")
                }
            )
            logger.info("Holdings fetched successfully")
        else:
            st.info("No holdings found.")
    except Exception as e:
        st.warning(f"Could not fetch holdings: {str(e)}")
        logger.exception("Error fetching holdings")

    # ---------------- Positions ----------------
    st.subheader("📊 Open Positions")
    try:
        logger.debug("Fetching positions")
        pos = client.positions()
        logger.debug("Positions response: %s", json.dumps(pos, indent=2))

        # Handle list response
        if isinstance(pos, list):
            df_pos = pd.DataFrame(pos)
        else:
            df_pos = pd.DataFrame(pos.get('Data', []))

        if not df_pos.empty:
            df_pos['Unrealized P&L'] = df_pos.get('MTOM', 0)
            df_pos['Realized P&L'] = df_pos.get('BookedPL', 0)
            df_pos['Total P&L'] = df_pos['Unrealized P&L'] + df_pos['Realized P&L']
            st.dataframe(
                df_pos[['ScripName', 'BuyQty', 'SellQty', 'NetQty', 'BuyAvgRate', 'SellAvgRate', 'Unrealized P&L', 'Realized P&L', 'Total P&L']],
                use_container_width=True,
                column_config={
                    "BuyQty": st.column_config.NumberColumn(format="%d"),
                    "SellQty": st.column_config.NumberColumn(format="%d"),
                    "NetQty": st.column_config.NumberColumn(format="%d"),
                    "BuyAvgRate": st.column_config.NumberColumn(format="₹%.2f"),
                    "SellAvgRate": st.column_config.NumberColumn(format="₹%.2f"),
                    "Unrealized P&L": st.column_config.NumberColumn(format="₹%.2f"),
                    "Realized P&L": st.column_config.NumberColumn(format="₹%.2f"),
                    "Total P&L": st.column_config.NumberColumn(format="₹%.2f")
                }
            )
            st.success(f"Total P&L: ₹{df_pos['Total P&L'].sum():,.2f}")
            logger.info("Positions fetched successfully")
        else:
            st.info("No open positions.")
    except Exception as e:
        st.warning(f"Could not fetch positions: {str(e)}")
        logger.exception("Error fetching positions")

    # ---------------- Order Book ----------------
    st.subheader("📋 Order Book")
    try:
        logger.debug("Fetching order book")
        order_book = client.order_book()
        logger.debug("Order book response: %s", json.dumps(order_book, indent=2))

        # Handle list response
        if isinstance(order_book, list):
            df_orders = pd.DataFrame(order_book)
        else:
            df_orders = pd.DataFrame(order_book.get('Data', []))

        if not df_orders.empty:
            st.dataframe(
                df_orders[['ScripName', 'OrderType', 'Qty', 'Rate', 'Status', 'ExchOrderID', 'OrderTime']],
                use_container_width=True,
                column_config={
                    "Qty": st.column_config.NumberColumn(format="%d"),
                    "Rate": st.column_config.NumberColumn(format="₹%.2f"),
                    "OrderTime": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")
                }
            )
            logger.info("Order book fetched successfully")
        else:
            st.info("No orders found.")
    except Exception as e:
        st.warning(f"Could not fetch order book: {str(e)}")
        logger.exception("Error fetching order book")

    # ---------------- Trade Book ----------------
    st.subheader("📜 Trade Book")
    try:
        logger.debug("Fetching trade book")
        trade_book = client.get_tradebook()
        logger.debug("Trade book response: %s", json.dumps(trade_book, indent=2))

        # Handle list response
        if isinstance(trade_book, list):
            df_trades = pd.DataFrame(trade_book)
        else:
            df_trades = pd.DataFrame(trade_book.get('Data', []))

        if not df_trades.empty:
            st.dataframe(
                df_trades[['ScripName', 'TradeType', 'Qty', 'Rate', 'TradeID', 'TradeTime']],
                use_container_width=True,
                column_config={
                    "Qty": st.column_config.NumberColumn(format="%d"),
                    "Rate": st.column_config.NumberColumn(format="₹%.2f"),
                    "TradeTime": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss")
                }
            )
            logger.info("Trade book fetched successfully")
        else:
            st.info("No trades found.")
    except Exception as e:
        st.warning(f"Could not fetch trade book: {str(e)}")
        logger.exception("Error fetching trade book")

    # ---------------- Place Derivative Order ----------------
    st.subheader("🛒 Place Derivative Order")
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbol (e.g., NIFTY, BANKNIFTY)", value="NIFTY").upper()
        exchange = st.selectbox("Exchange", ["N", "B"], format_func=lambda x: {"N": "NSE", "B": "BSE"}[x])
        qty = st.number_input("Quantity", value=50, step=25, min_value=1)
    with col2:
        price = st.number_input("Price (0 for Market)", value=1.5, min_value=0.0, step=0.05)
        order_type = st.selectbox("Order Type", ["B", "S"], format_func=lambda x: {"B": "Buy", "S": "Sell"}[x])
        is_intraday = st.checkbox("Intraday", value=True)
        is_stop_loss = st.checkbox("Stop Loss Order", value=False)
        stop_loss_price = st.number_input("Stop Loss Price", value=0.0, min_value=0.0, step=0.05) if is_stop_loss else 0

    if st.button("Place Order"):
        try:
            logger.debug("Searching scrip for symbol: %s", symbol)
            scrip_data = client.search_scrip(symbol)
            if not scrip_data or not scrip_data[0].get("ScripCode"):
                raise ValueError(f"Scrip code not found for symbol: {symbol}")

            scrip_code = scrip_data[0]["ScripCode"]
            logger.debug("Placing order: Symbol=%s, ScripCode=%s, Exchange=%s, Qty=%s, Price=%s, OrderType=%s, IsIntraday=%s, StopLossPrice=%s",
                         symbol, scrip_code, exchange, qty, price, order_type, is_intraday, stop_loss_price)

            order = client.place_order(
                OrderType=order_type,
                Exchange=exchange,
                ExchangeType="D",  # Derivatives
                ScripCode=scrip_code,
                Qty=qty,
                Price=price if price > 0 else 0,  # Use 0 for Market orders
                IsIntraday=is_intraday,
                IsStopLossOrder=is_stop_loss,
                StopLossPrice=stop_loss_price,
                IsVTCOrder=False
            )
            st.success(f"Order Placed: {json.dumps(order, indent=2)}")
            logger.info("Order placed successfully: %s", order)
        except Exception as e:
            st.error(f"Order Error: {str(e)}")
            logger.exception("Error placing order")

    # ---------------- Position Conversion ----------------
    st.subheader("🔄 Position Conversion")
    conv_symbol = st.text_input("Conversion Symbol (e.g., BPCL_EQ)", value="BPCL_EQ")
    conv_exchange = st.selectbox("Conversion Exchange", ["N", "B"], key="conv_exch")
    conv_qty = st.number_input("Conversion Quantity", value=5, step=1, min_value=1)
    conv_type = st.selectbox("Conversion Type", ["B", "S"], format_func=lambda x: {"B": "Buy", "S": "Sell"}[x])
    conv_from = st.selectbox("From", ["D", "I"], format_func=lambda x: {"D": "Delivery", "I": "Intraday"}[x])
    conv_to = st.selectbox("To", ["D", "I"], format_func=lambda x: {"D": "Delivery", "I": "Intraday"}[x])

    if st.button("Convert Position'):
        try:
            logger.debug("Converting position: Symbol=%s, Exchange=%s, Qty=%s, Type=%s, From=%s, To=%s",
                         conv_symbol, conv_exchange, conv_qty, conv_type, conv_from, conv_to)
            result = client.position_convertion(
                Exchange=conv_exchange,
                ExchangeType="C",  # Cash
                ScripName=conv_symbol,
                BuySell=conv_type,
                Qty=conv_qty,
                From=conv_from,
                To=conv_to
            )
            st.success(f"Position Converted: {json.dumps(result, indent=2)}")
            logger.info("Position conversion successful: %s", result)
        except Exception as e:
            st.error(f"Position Conversion Error: {str(e)}")
            logger.exception("Error converting position")

    # ---------------- Option Chain Placeholder ----------------
    st.subheader("📂 Option Chain Viewer")
    opt_symbol = st.text_input("Enter Symbol for Option Chain (e.g., NIFTY, BANKNIFTY)", value="NIFTY").upper()
    if st.button("Fetch Option Chain"):
        try:
            logger.debug("Attempting to fetch option chain for symbol: %s", opt_symbol)
            st.info(f"Option chain for {opt_symbol} — (API not available in py5paisa SDK. Contact 5paisa support for REST API access.)")
            logger.info("Option chain fetch attempted (placeholder)")
        except Exception as e:
            st.error(f"Option Chain Error: {str(e)}")
            logger.exception("Error fetching option chain")

    # ---------------- Logout ----------------
    if st.button("🔒 Logout"):
        st.session_state.logged_in = False
        st.session_state.client = None
        st.success("Logged out successfully!")
        logger.info("User logged out")

else:
    st.info("Please login from the sidebar to continue.")


# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Trading Controls")
    forecast_horizon = st.slider("Forecast Horizon (days)", 1, 10, 7, key="horizon_slider")
    capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000, key="capital_input")
    risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1, key="risk_select")
    st.markdown("**Backtest Parameters**")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"), key="start_date")
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-29"), key="end_date")
    strategy_choice = st.selectbox("Strategy", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard"], key="strategy_select")
    run_button = st.button("Run Analysis", key="run_button")
    st.markdown("---")
    st.markdown("**Motto:** Deploy with edge, survive, outlast.")

# Black-Scholes IV calculation (from your 5paisa script)
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, T, r, market_price, option_type='call', tol=1e-5, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        if option_type == 'call':
            model_price = black_scholes_call(S, K, T, r, sigma)
        else:
            model_price = black_scholes_call(S, K, T, r, sigma) + K * np.exp(-r * T) - S
        diff = model_price - market_price
        if abs(diff) < tol:
            return sigma * 100
        vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        if vega == 0:
            return np.nan
        sigma -= diff / vega
    return np.nan

# Max Pain calculation (from your 5paisa script)
def max_pain(df, nifty_spot):
    calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["LastRate"]
    puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["LastRate"]
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

# Function to fetch real data from 5paisa (modified from your 5paisa script)
def fetch_nifty_data():
    try:
        # Fetch NIFTY 50 spot price
        nifty_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920000,
            "Symbol": "NIFTY",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        nifty_data = client.fetch_market_feed(nifty_req)
        if not nifty_data or "Data" not in nifty_data or not nifty_data["Data"]:
            raise Exception("Failed to fetch Nifty 50 index price")
        nifty_spot = nifty_data["Data"][0].get("LastRate", nifty_data["Data"][0].get("LastTradedPrice", 0))
        if not nifty_spot:
            raise Exception("Nifty price key not found")

        # Fetch option chain
        expiry_timestamp = 1746694800000  # May 8, 2025
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

        # Fetch IV for ATM ±100 strikes
        iv_df = df[(df["StrikeRate"] >= atm_strike - 100) & (df["StrikeRate"] <= atm_strike + 100)].copy()
        T = (datetime(2025, 5, 8) - datetime.now()).days / 365.0
        r = 0.06
        iv_df["IV (%)"] = iv_df.apply(
            lambda row: implied_volatility(
                S=nifty_spot,
                K=row["StrikeRate"],
                T=T,
                r=r,
                market_price=row["LastRate"],
                option_type='call' if row["CPType"] == "CE" else 'put'
            ),
            axis=1
        )

        # Calculate PCR
        calls = df[df["CPType"] == "CE"]
        puts = df[df["CPType"] == "PE"]
        call_oi = calls["OpenInterest"].sum()
        put_oi = puts["OpenInterest"].sum()
        pcr = put_oi / call_oi if call_oi != 0 else float("inf")

        # Calculate Max Pain
        max_pain_strike, max_pain_diff_pct = max_pain(df, nifty_spot)

        # Calculate VIX Change (using ATM IV history)
        atm_iv = iv_df[iv_df["StrikeRate"] == atm_strike]["IV (%)"].mean()
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
    except Exception as e:
        logger.error(f"Error fetching 5paisa data: {str(e)}")
        return None

# Function to load data (updated to prioritize 5paisa data)
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    try:
        # Step 1: Fetch real data from 5paisa
        real_data = fetch_nifty_data()
        if real_data is None:
            logger.warning("Falling back to GitHub CSV due to 5paisa failure")
            # Fallback to GitHub-hosted NIFTY CSV
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            response = requests.get(nifty_url)
            response.raise_for_status()
            nifty = pd.read_csv(io.StringIO(response.text), encoding="utf-8-sig")
            nifty.columns = nifty.columns.str.strip()
            if "Date" not in nifty.columns or "Close" not in nifty.columns:
                st.error(f"NIFTY CSV missing required columns 'Date' or 'Close'. Found: {nifty.columns.tolist()}")
                return None, None
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"])
            nifty = nifty[["Date", "Close"]].set_index("Date")
            nifty.index = pd.to_datetime(nifty.index)
            nifty = nifty.rename(columns={"Close": "NIFTY_Close"})
            nifty_series = nifty["NIFTY_Close"].squeeze()

            # Fetch India VIX from GitHub
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
            vix.columns = vix.columns.str.strip()
            if "Date" not in vix.columns or "Close" not in vix.columns:
                st.error(f"india_vix.csv is missing required columns. Found: {vix.columns.tolist()}")
                return pd.DataFrame({"NIFTY_Close": nifty_series}, index=nifty.index)
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            vix = vix.dropna(subset=["Date"])
            vix = vix[["Date", "Close"]].set_index("Date").rename(columns={"Close": "VIX"})
            vix.index = pd.to_datetime(vix.index)
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
                return None, None
        else:
            # Use 5paisa data for latest day
            latest_date = datetime.now().date()
            df = pd.DataFrame({
                "NIFTY_Close": [real_data["nifty_spot"]],
                "VIX": [real_data["atm_iv"]]
            }, index=[pd.to_datetime(latest_date)])
            
            # Load historical data from GitHub to extend the dataset
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

            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            response = requests.get(vix_url)
            response.raise_for_status()
            vix = pd.read_csv(io.StringIO(response.text))
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
            
            df = pd.concat([historical_df, df])
            df = df[~df.index.duplicated(keep='last')]
            df = df.sort_index()

        logger.debug("Data loaded successfully.")
        return df, real_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Error loading data: {str(e)}")
        return None, None

# Function to generate synthetic features (calibrated with real data)
@st.cache_data
def generate_synthetic_features(df, real_data, capital):
    try:
        n_days = len(df)
        np.random.seed(42)
        risk_free_rate = 0.06
        strike_step = 100

        # Calibrate synthetic data using real data (if available)
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
                    return max(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), 0)
                else:
                    return max(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0)
            except:
                return 0

        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        event_spike = np.where((df.index.month % 3 == 0) & (df.index.day < 5), 1.2, 1.0)
        df["ATM_IV"] = df["VIX"] * (1 + np.random.normal(0, 0.1, n_days)) * event_spike
        df["ATM_IV"] = np.clip(df["ATM_IV"], 5, 50)

        # Calibrate ATM_IV with real data for the latest day
        if real_data:
            df["ATM_IV"].iloc[-1] = base_iv

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
        
        df.to_csv("volguard_hybrid_data.csv")
        logger.debug("Synthetic features generated successfully.")
        return df
    except Exception as e:
        st.error(f"Error generating features: {str(e)}")
        logger.error(f"Error generating features: {str(e)}")
        return None

# Function to fetch portfolio data from 5paisa
def fetch_portfolio_data():
    try:
        # Fetch positions (open trades)
        positions = client.positions()
        if not positions:
            raise Exception("Failed to fetch positions")

        total_pnl = 0
        total_margin = 0
        total_exposure = 0
        for position in positions:
            total_pnl += position.get("ProfitLoss", 0)
            total_margin += position.get("MarginUsed", 0)
            total_exposure += position.get("Exposure", 0)

        return {
            "weekly_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure": total_exposure / capital * 100 if capital > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0}

# Function to forecast volatility (unchanged except for minor optimizations)
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

# Function to generate trading strategy (added risk checks)
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
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

        # Risk Checks
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

        capital_alloc = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc.get(regime, 0.2) * position_size
        max_loss = deploy * 0.2
        total_exposure = deploy / capital

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

# Function for backtesting (unchanged)
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.debug(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")
        if df.empty:
            st.error("Backtest failed: No data available.")
            logger.error("Backtest failed: Empty DataFrame")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()
        
        df_backtest = df.loc[start_date:end_date].copy()
        if len(df_backtest) < 50:
            st.error(f"Backtest failed: Insufficient data ({len(df_backtest)} days). Need at least 50 days.")
            logger.error(f"Backtest failed: Insufficient data ({len(df_backtest)} days)")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()
        
        required_cols = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price"]
        missing_cols = [col for col in required_cols if col not in df_backtest.columns]
        if missing_cols:
            st.error(f"Backtest failed: Missing columns {missing_cols}")
            logger.error(f"Backtest failed: Missing columns {missing_cols}")
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
                "Jade Lizard": 1.4
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

        logger.debug("Backtest completed successfully.")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

# Main execution
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.info("Please login to 5paisa from the sidebar to proceed.")
else:
    if run_button:
        with st.spinner("Running VolGuard Analysis..."):
            # Reset previous backtest results
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.violations = 0
            st.session_state.journal_complete = False

            df, real_data = load_data()
            if df is not None:
                df = generate_synthetic_features(df, real_data, capital)
                if df is not None:
                    # Display Latest Market Data
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📊 Latest Market Snapshot")
                    last_date = df.index[-1].strftime("%d-%b-%Y")
                    last_nifty = df["NIFTY_Close"].iloc[-1]
                    prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 else "N/A"
                    last_vix = df["VIX"].iloc[-1]
                    prev_vix = df["VIX"].iloc[-2] if len(df) >= 2 else "N/A"
                    col1, col2 = st.columns(2)
                    with col1:
                        nifty_change = last_nifty - prev_nifty if prev_nifty != "N/A" else "N/A"
                        nifty_change_display = f"{nifty_change:+,.2f}" if nifty_change != "N/A" else "N/A"
                        st.metric(
                            label="NIFTY 50 Last Close",
                            value=f"{last_nifty:,.2f}",
                            delta=nifty_change_display,
                            delta_color="normal",
                            help="Latest NIFTY 50 closing price"
                        )
                    with col2:
                        vix_change = last_vix - prev_vix if prev_vix != "N/A" else "N/A"
                        vix_change_display = f"{vix_change:+.2f}" if vix_change != "N/A" else "N/A"
                        st.metric(
                            label="India VIX",
                            value=f"{last_vix:.2f}%",
                            delta=vix_change_display,
                            delta_color="normal",
                            help="Latest India VIX value"
                        )
                    st.markdown(f"**Last Updated**: {last_date}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Portfolio Tracking
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("💼 Portfolio Overview")
                    portfolio_data = fetch_portfolio_data()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Weekly P&L", f"₹{portfolio_data['weekly_pnl']:,.2f}")
                    with col2:
                        st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                    with col3:
                        st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
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
                        strategy = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital)
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

                            # Trade Execution
                            if st.button("Trade Now"):
                                try:
                                    # Simplified trade execution (Iron Condor example)
                                    option_chain = real_data["option_chain"]
                                    atm_strike = real_data["atm_strike"]
                                    # Define strikes for Iron Condor
                                    call_sell_strike = atm_strike + 100
                                    call_buy_strike = call_sell_strike + 100
                                    put_sell_strike = atm_strike - 100
                                    put_buy_strike = put_sell_strike - 100

                                    # Place orders
                                    orders = [
                                        {
                                            "Exch": "N",
                                            "ExchType": "C",
                                            "ScripCode": option_chain[(option_chain["StrikeRate"] == call_sell_strike) & (option_chain["CPType"] == "CE")]["ScripCode"].iloc[0],
                                            "BuySell": "S",
                                            "Qty": 25,
                                            "Price": option_chain[(option_chain["StrikeRate"] == call_sell_strike) & (option_chain["CPType"] == "CE")]["LastRate"].iloc[0],
                                            "OrderType": "LIMIT"
                                        },
                                        {
                                            "Exch": "N",
                                            "ExchType": "C",
                                            "ScripCode": option_chain[(option_chain["StrikeRate"] == call_buy_strike) & (option_chain["CPType"] == "CE")]["ScripCode"].iloc[0],
                                            "BuySell": "B",
                                            "Qty": 25,
                                            "Price": option_chain[(option_chain["StrikeRate"] == call_buy_strike) & (option_chain["CPType"] == "CE")]["LastRate"].iloc[0],
                                            "OrderType": "LIMIT"
                                        },
                                        {
                                            "Exch": "N",
                                            "ExchType": "C",
                                            "ScripCode": option_chain[(option_chain["StrikeRate"] == put_sell_strike) & (option_chain["CPType"] == "PE")]["ScripCode"].iloc[0],
                                            "BuySell": "S",
                                            "Qty": 25,
                                            "Price": option_chain[(option_chain["StrikeRate"] == put_sell_strike) & (option_chain["CPType"] == "PE")]["LastRate"].iloc[0],
                                            "OrderType": "LIMIT"
                                        },
                                        {
                                            "Exch": "N",
                                            "ExchType": "C",
                                            "ScripCode": option_chain[(option_chain["StrikeRate"] == put_buy_strike) & (option_chain["CPType"] == "PE")]["ScripCode"].iloc[0],
                                            "BuySell": "B",
                                            "Qty": 25,
                                            "Price": option_chain[(option_chain["StrikeRate"] == put_buy_strike) & (option_chain["CPType"] == "PE")]["LastRate"].iloc[0],
                                            "OrderType": "LIMIT"
                                        }
                                    ]

                                    for order in orders:
                                        client.place_order(
                                            OrderType=order["BuySell"],
                                            Exchange=order["Exch"],
                                            ExchangeType=order["ExchType"],
                                            ScripCode=order["ScripCode"],
                                            Qty=order["Qty"],
                                            Price=order["Price"],
                                            IsIntraday=False
                                        )

                                    # Log the trade
                                    trade_log = {
                                        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        "Strategy": strategy["Strategy"],
                                        "Regime": strategy["Regime"],
                                        "Risk_Level": "High" if strategy["Risk_Flags"] else "Low",
                                        "Outcome": "Pending"
                                    }
                                    st.session_state.trades.append(trade_log)
                                    pd.DataFrame(st.session_state.trades).to_csv("trade_log.csv", index=False)

                                    st.success("✅ Trade Placed Successfully!")
                                except Exception as e:
                                    st.error(f"Trade Failed: {str(e)}")

                        st.markdown('</div>', unsafe_allow_html=True)

                        # Backtest Section
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.subheader("📊 Backtest Performance")
                        st.markdown(f"Evaluating historical performance for {strategy_choice} from {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')} (Option Selling Strategies Only).")
                        
                        with st.spinner("Running backtest..."):
                            if start_date >= end_date:
                                st.error("Backtest failed: Start date must be before end date.")
                            elif start_date < df.index[0].date() or end_date > df.index[-1].date():
                                st.error("Backtest failed: Selected date range is outside available data.")
                            else:
                                st.session_state.backtest_run = True
                                backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(df, capital, strategy_choice, start_date, end_date)
                                st.session_state.backtest_results = (backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf)

                        if st.session_state.backtest_run and st.session_state.backtest_results is not None:
                            backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = st.session_state.backtest_results
                            if len(backtest_df) == 0:
                                st.error("No trades generated. Try selecting 'All Strategies' or a wider date range.")
                            else:
                                st.success("Backtest completed successfully!")
                                st.markdown(f"**Period**: {start_date.strftime('%d-%b-%Y')} to {end_date.strftime('%d-%b-%Y')}")
                                st.markdown(f"**Strategy Tested**: {strategy_choice}")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total P&L", f"₹{total_pnl:,.2f}")
                                    st.metric("Win Rate", f"{win_rate*100:.2f}%")
                                with col2:
                                    st.metric("Max Drawdown", f"₹{max_drawdown:,.2f}")
                                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                                with col3:
                                    st.metric("Sortino Ratio", f"{sortino_ratio:.2f}")
                                    st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")

                                # Performance Charts
                                st.markdown("### Cumulative P&L Over Time")
                                cum_pnl = backtest_df["PnL"].cumsum()
                                st.line_chart(cum_pnl, color="#e94560", use_container_width=True)

                                st.markdown("### Strategy Performance Breakdown")
                                st.dataframe(strategy_perf, use_container_width=True)

                                st.markdown("### Regime Performance Breakdown")
                                st.dataframe(regime_perf, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Journaling Section
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Discipline Hub: Trading Journal")
st.markdown("Reflect on your trades to improve discipline and consistency.")
with st.form(key="journal_form"):
    reason_strategy = st.selectbox("Why did you choose this strategy?", ["High IV", "Low Risk", "Event Opportunity", "Other"], key="journal_reason")
    override_risk = st.radio("Did you override any risk flags?", ("Yes", "No"), key="journal_override")
    expected_outcome = st.text_area("What is your expected outcome for this trade?", key="journal_outcome")
    submit_journal = st.form_submit_button("Submit Journal Entry")

    if submit_journal:
        # Calculate Discipline Score
        score = 0
        if override_risk == "No":
            score += 3
        if reason_strategy != "Other":
            score += 3
        if expected_outcome:
            score += 3
        if portfolio_data["weekly_pnl"] > 0:
            score += 1
        score = min(score, 10)

        # Save journal entry
        journal_entry = {
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Strategy_Reason": reason_strategy,
            "Override_Risk": override_risk,
            "Expected_Outcome": expected_outcome,
            "Discipline_Score": score
        }
        # Save to CSV
        journal_df = pd.DataFrame([journal_entry])
        journal_file = "journal_log.csv"
        journal_df.to_csv(journal_file, mode='a', header=not os.path.exists(journal_file), index=False)
        st.success(f"Journal Entry Saved! Discipline Score: {score}/10")
        st.session_state.journal_complete = True
        # Reset violations if journaling is complete
        if st.session_state.violations > 0:
            st.session_state.violations = 0
            st.success("✅ Discipline Lock Removed - You can now trade!")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True)
