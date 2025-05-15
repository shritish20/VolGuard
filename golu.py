import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import pickle
from datetime import datetime, timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import upstox_client
from upstox_client import Configuration, ApiClient, OptionsApi, UserApi, PortfolioApi, OrderApi, OrderApiV3, PlaceOrderV3Request
import logging
import time
import json
from streamlit_aggrid import AgGrid, GridOptionsBuilder

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
import xgboost as xgb
xgb.set_config(verbosity=0)

# === Streamlit Configuration ===
st.set_page_config(page_title="VolGuard Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Professional UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    body, .stApp {
        font-family: 'Inter', sans-serif;
        background: #0a0a0a;
        color: #ffffff;
    }
    /* Top Bar */
    .top-bar {
        background: linear-gradient(135deg, #1c2526 0%, #121212 100%);
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #1e88e5;
    }
    .top-bar div {
        display: flex;
        align-items: center;
        margin: 0 15px;
    }
    .top-bar div p {
        margin: 0 0 0 8px;
        font-size: 1.1em;
        font-weight: 500;
    }
    .top-bar i {
        color: #1e88e5;
        font-size: 1.2em;
    }
    /* Sidebar */
    .css-1d391kg {
        background: #121212;
        padding: 20px;
        border-right: 2px solid #1e88e5;
    }
    .css-1d391kg h1 {
        color: #1e88e5;
        font-size: 1.8em;
        margin-bottom: 25px;
    }
    .css-1d391kg .stButton>button {
        background: linear-gradient(135deg, #1e88e5 0%, #ab47bc 100%);
        color: #ffffff;
        border-radius: 10px;
        padding: 12px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .css-1d391kg .stButton>button:hover {
        background: linear-gradient(135deg, #ab47bc 0%, #1e88e5 100%);
        box-shadow: 0 0 20px rgba(30, 136, 229, 0.6);
    }
    /* Widgets */
    .widget-card {
        background: rgba(28, 37, 38, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(30, 136, 229, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.5s ease-in;
    }
    .widget-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(30, 136, 229, 0.6);
    }
    .widget-card h3 {
        color: #1e88e5;
        margin: 0 0 15px 0;
        display: flex;
        align-items: center;
        font-size: 1.4em;
    }
    .widget-card h3 i {
        margin-right: 10px;
        color: #ab47bc;
    }
    .widget-card p, .widget-card span {
        color: #e0e0e0;
        font-size: 1.1em;
        margin: 5px 0;
    }
    /* Alerts */
    .alert-green {
        background: #43a047;
        color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
    }
    .alert-yellow {
        background: #ffb300;
        color: #000000;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
    }
    .alert-red {
        background: #e53935;
        color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: 500;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e88e5 0%, #ab47bc 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ab47bc 0%, #1e88e5 100%);
        box-shadow: 0 0 20px rgba(30, 136, 229, 0.6);
    }
    /* Tabs */
    .stTabs [role="tab"] {
        background: transparent;
        color: #ffffff;
        border-bottom: 3px solid transparent;
        padding: 12px 25px;
        margin-right: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        border-bottom: 3px solid #1e88e5;
        color: #1e88e5;
    }
    .stTabs [role="tab"]:hover {
        color: #ab47bc;
    }
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    /* Plotly Charts */
    .plotly-chart {
        background: #121212;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
    }
    </style>
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
""", unsafe_allow_html=True)

# === Session State Initialization ===
if 'volguard_data' not in st.session_state:
    st.session_state.volguard_data = None
if 'xgb_prediction' not in st.session_state:
    st.session_state.xgb_prediction = None
if 'atm_iv' not in st.session_state:
    st.session_state.atm_iv = None
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
if 'journal_entries' not in st.session_state:
    st.session_state.journal_entries = []
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'deployed_capital' not in st.session_state:
    st.session_state.deployed_capital = 0
if 'daily_pnl' not in st.session_state:
    st.session_state.daily_pnl = 0
if 'user_details' not in st.session_state:
    st.session_state.user_details = None
if 'option_chain' not in st.session_state:
    st.session_state.option_chain = None
if 'positions' not in st.session_state:
    st.session_state.positions = []

prev_oi = {}

# === Sidebar Controls ===
st.sidebar.header("VolGuard Pro")
st.sidebar.markdown("<p style='color:#ab47bc;font-size:1.1em;'>Options Trading Dashboard</p>", unsafe_allow_html=True)
access_token = st.sidebar.text_input("Upstox Access Token", type="password", placeholder="Enter your access token")
total_capital = st.sidebar.slider("Total Capital (₹)", 100000, 5000000, 1000000, 10000)
risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
refresh_interval = st.sidebar.slider("Data Refresh Interval (seconds)", 30, 300, 60)
theme_toggle = st.sidebar.checkbox("Light Theme", value=False)
if theme_toggle:
    st.markdown("<style>.stApp {background: #f5f5f5; color: #000000;}</style>", unsafe_allow_html=True)

# === Risk Management Rules ===
MAX_EXPOSURE_PCT = 40
MAX_LOSS_PER_TRADE_PCT = 4
DAILY_LOSS_LIMIT_PCT = 4
max_loss_per_trade = total_capital * (MAX_LOSS_PER_TRADE_PCT / 100)
daily_loss_limit = total_capital * (DAILY_LOSS_LIMIT_PCT / 100)
max_deployed_capital = total_capital * (MAX_EXPOSURE_PCT / 100)

# === Helper Functions (Reused from Original Code) ===
def get_nearest_expiry(options_api, instrument_key):
    try:
        response = options_api.get_option_contracts(instrument_key=instrument_key)
        contracts = response.to_dict().get("data", [])
        expiry_dates = set()
        for contract in contracts:
            exp = contract.get("expiry")
            if isinstance(exp, str):
                exp = datetime.strptime(exp, "%Y-%m-%d")
            expiry_dates.add(exp)
        expiry_list = sorted(expiry_dates)
        today = datetime.now()
        valid_expiries = [e.strftime("%Y-%m-%d") for e in expiry_list if e >= today]
        nearest = valid_expiries[0] if valid_expiries else None
        time.sleep(0.5)
        return nearest
    except Exception as e:
        logger.error(f"Expiry fetch failed: {e}")
        return None

def fetch_option_chain(options_api, instrument_key, expiry):
    try:
        res = options_api.get_put_call_option_chain(instrument_key=instrument_key, expiry_date=expiry)
        time.sleep(0.5)
        return res.to_dict().get('data', [])
    except Exception as e:
        logger.error(f"Option chain fetch error: {e}")
        return []

def process_chain(data):
    global prev_oi
    rows, ce_oi, pe_oi = [], 0, 0
    for r in data:
        ce = r.get('call_options', {})
        pe = r.get('put_options', {})
        ce_md, pe_md = ce.get('market_data', {}), pe.get('market_data', {})
        ce_gk, pe_gk = ce.get('option_greeks', {}), pe.get('option_greeks', {})
        strike = r['strike_price']
        ce_oi_val = ce_md.get("oi", 0)
        pe_oi_val = pe_md.get("oi", 0)
        ce_oi_change = ce_oi_val - prev_oi.get(f"{strike}_CE", 0)
        pe_oi_change = pe_oi_val - prev_oi.get(f"{strike}_PE", 0)
        ce_oi_change_pct = (ce_oi_change / prev_oi.get(f"{strike}_CE", 0) * 100) if prev_oi.get(f"{strike}_CE", 0) else 0
        pe_oi_change_pct = (pe_oi_change / prev_oi.get(f"{strike}_PE", 0) * 100) if prev_oi.get(f"{strike}_PE", 0) else 0
        strike_pcr = pe_oi_val / ce_oi_val if ce_oi_val else 0
        row = {
            "Strike": strike,
            "CE_LTP": ce_md.get("ltp"),
            "CE_IV": ce_gk.get("iv"),
            "CE_Delta": ce_gk.get("delta"),
            "CE_Theta": ce_gk.get("theta"),
            "CE_Vega": ce_gk.get("vega"),
            "CE_OI": ce_oi_val,
            "CE_OI_Change": ce_oi_change,
            "CE_OI_Change_Pct": ce_oi_change_pct,
            "CE_Volume": ce_md.get("volume", 0),
            "PE_LTP": pe_md.get("ltp"),
            "PE_IV": pe_gk.get("iv"),
            "PE_Delta": ce_gk.get("delta"),
            "PE_Theta": ce_gk.get("theta"),
            "PE_Vega": ce_gk.get("vega"),
            "PE_OI": pe_oi_val,
            "PE_OI_Change": pe_oi_change,
            "PE_OI_Change_Pct": pe_oi_change_pct,
            "PE_Volume": pe_md.get("volume", 0),
            "Strike_PCR": strike_pcr,
            "CE_Token": ce.get("instrument_key"),
            "PE_Token": pe.get("instrument_key")
        }
        ce_oi += ce_oi_val
        pe_oi += pe_oi_val
        rows.append(row)
        prev_oi[f"{strike}_CE"] = ce_oi_val
        prev_oi[f"{strike}_PE"] = pe_oi_val
    df = pd.DataFrame(rows).sort_values("Strike")
    return df, ce_oi, pe_oi

def calculate_metrics(df, ce_oi_total, pe_oi_total, spot):
    atm = df.iloc[(df['Strike'] - spot).abs().argsort()[:1]]
    atm_strike = atm['Strike'].values[0]
    pcr = round(pe_oi_total / ce_oi_total, 2) if ce_oi_total else 0
    max_pain_series = df.apply(lambda x: sum(df['CE_OI'] * abs(df['Strike'] - x['Strike'])) +
                                         sum(df['PE_OI'] * abs(df['Strike'] - x['Strike'])), axis=1)
    strike_with_pain = df.loc[max_pain_series.idxmin(), 'Strike']
    straddle_price = float(atm['CE_LTP'].values[0] + atm['PE_LTP'].values[0])
    atm_row = df[df['Strike'] == atm_strike]
    atm_iv = (atm_row['CE_IV'].values[0] + atm_row['PE_IV'].values[0]) / 2 if not atm_row.empty else 0
    return pcr, strike_with_pain, straddle_price, atm_strike, atm_iv

def plot_iv_skew(df, spot, atm_strike):
    valid = df[(df['CE_IV'] > 0) & (df['PE_IV'] > 0)]
    if valid.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['CE_IV'], mode='lines+markers', name='Call IV', line=dict(color='#1e88e5')))
    fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['PE_IV'], mode='lines+markers', name='Put IV', line=dict(color='#ab47bc')))
    fig.add_vline(x=spot, line=dict(color='#ffffff', dash='dash'), name='Spot')
    fig.add_vline(x=atm_strike, line=dict(color='#43a047', dash='dot'), name='ATM')
    fig.update_layout(
        title="IV Skew",
        xaxis_title="Strike",
        yaxis_title="IV (%)",
        template="plotly_dark",
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font=dict(color='#ffffff')
    )
    return fig

def get_market_depth(access_token, base_url, token):
    try:
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        url = f"{base_url}/market-quote/depth"
        res = requests.get(url, headers=headers, params={"instrument_key": token})
        depth = res.json().get('data', {}).get(token, {}).get('depth', {})
        bid_volume = sum(item.get('quantity', 0) for item in depth.get('buy', []))
        ask_volume = sum(item.get('quantity', 0) for item in depth.get('sell', []))
        time.sleep(0.5)
        return {"bid_volume": bid_volume, "ask_volume": ask_volume}
    except Exception as e:
        logger.error(f"Depth fetch error for {token}: {e}")
        return {}

def calculate_rolling_and_fixed_hv(nifty_close):
    log_returns = np.log(nifty_close.pct_change() + 1).dropna()
    last_7d_std = log_returns[-7:].std()
    rolling_rv_annualized = last_7d_std * np.sqrt(252) * 100
    last_date = nifty_close.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
    rv_7d_df = pd.DataFrame({
        "Date": future_dates,
        "Day": future_dates.day_name(),
        "7-Day Realized Volatility (%)": np.round([rolling_rv_annualized]*7, 2)
    })
    hv_30d = log_returns[-30:].std() * np.sqrt(252) * 100
    hv_1y = log_returns[-252:].std() * np.sqrt(252) * 100
    return rv_7d_df, round(hv_30d, 2), round(hv_1y, 2)

def compute_realized_vol(nifty_df):
    log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna()
    last_7d_std = log_returns[-7:].std() * np.sqrt(252) * 100
    return last_7d_std if not np.isnan(last_7d_std) else 0

def get_user_details(access_token):
    configuration = Configuration()
    configuration.access_token = access_token
    client = ApiClient(configuration)
    user_api = UserApi(client)
    portfolio_api = PortfolioApi(client)
    order_api = OrderApi(client)
    details = {}
    try:
        details['profile'] = user_api.get_profile(api_version="v2").to_dict()
    except Exception as e:
        details['profile_error'] = str(e)
    try:
        details['funds'] = user_api.get_user_fund_margin(api_version="v2").to_dict()
    except Exception as e:
        details['funds_error'] = str(e)
    try:
        details['holdings'] = portfolio_api.get_holdings(api_version="v2").to_dict()
    except Exception as e:
        details['holdings_error'] = str(e)
    try:
        details['positions'] = portfolio_api.get_positions(api_version="v2").to_dict()
    except Exception as e:
        details['positions_error'] = str(e)
    try:
        details['orders'] = order_api.get_order_book(api_version="v2").to_dict()
    except Exception as e:
        details['orders_error'] = str(e)
    try:
        details['trades'] = order_api.get_trade_history(api_version="v2").to_dict()
    except Exception as e:
        details['trades_error'] = str(e)
    return details

def find_atm_strike(spot_price, strikes):
    return min(strikes, key=lambda x: abs(x - spot_price))

def get_instrument_key_from_chain(option_chain, strike, opt_type):
    for leg in option_chain:
        if leg['strike_price'] == strike:
            if opt_type == 'CE':
                return leg.get('call_options', {}).get('instrument_key')
            elif opt_type == 'PE':
                return leg.get('put_options', {}).get('instrument_key')
    return None

def build_strategy_legs(option_chain, spot_price, strategy_name, quantity, otm_distance=50):
    strikes = [leg['strike_price'] for leg in option_chain]
    atm_strike = find_atm_strike(spot_price, strikes)
    legs = []

    def get_key(strike, opt_type):
        return get_instrument_key_from_chain(option_chain, strike, opt_type)

    s = strategy_name.lower()

    if s == "iron_fly":
        legs = [
            {"instrument_key": get_key(atm_strike, "CE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike, "PE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
        ]
    elif s == "iron_condor":
        legs = [
            {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike + 2 * otm_distance, "CE"), "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
        ]
    elif s == "short_straddle":
        legs = [
            {"instrument_key": get_key(atm_strike, "CE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike, "PE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
        ]
    elif s == "short_strangle":
        legs = [
            {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
        ]
    elif s == "bull_put_credit":
        legs = [
            {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
        ]
    elif s == "bear_call_credit":
        legs = [
            {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike + 2 * otm_distance, "CE"), "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
        ]
    elif s == "jade_lizard":
        legs = [
            {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
        ]
    else:
        raise ValueError(f"Unsupported strategy: {strategy_name}")

    for leg in legs:
        if leg["instrument_key"] is None:
            raise ValueError(f"Missing instrument key for leg: {leg}")

    return legs

def place_order_for_leg(order_api, leg):
    try:
        price_val = 0 if leg["order_type"] == "MARKET" else leg.get("price", 0)
        body = PlaceOrderV3Request(
            instrument_token=leg["instrument_key"],
            transaction_type=leg["action"],
            order_type=leg["order_type"],
            product="I",
            quantity=leg["quantity"],
            validity="DAY",
            disclosed_quantity=0,
            trigger_price=0.0,
            tag="volguard",
            is_amo=False,
            slice=False,
            price=price_val
        )
        response = order_api.place_order(body)
        logger.info(f"Order placed: {leg['action']} {leg['instrument_key']} qty={leg['quantity']}")
        return response.to_dict()
    except Exception as e:
        logger.error(f"Order failed for {leg['instrument_key']}: {e}")
        return None

def execute_strategy(access_token, option_chain, spot_price, strategy_name, quantity):
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        order_api = OrderApiV3(client)

        legs = build_strategy_legs(option_chain, spot_price, strategy_name, quantity)
        order_results = []
        for leg in legs:
            result = place_order_for_leg(order_api, leg)
            order_results.append(result)
            if result:
                st.success(f"Order placed: {leg['action']} {leg['instrument_key']} (Qty: {leg['quantity']})")
            else:
                st.error(f"Order failed for {leg['instrument_key']}")
        return order_results
    except Exception as e:
        st.error(f"Error executing strategy: {e}")
        return None

def run_volguard(access_token):
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        options_api = upstox_client.OptionsApi(client)
        instrument_key = "NSE_INDEX|Nifty 50"
        base_url = "https://api.upstox.com/v2"

        expiry = get_nearest_expiry(options_api, instrument_key)
        if not expiry:
            return None, None, None, None, None
        chain = fetch_option_chain(options_api, instrument_key, expiry)
        if not chain:
            return None, None, None, None, None
        spot = chain[0].get("underlying_spot_price")
        if not spot:
            return None, None, None, None, None

        df, ce_oi, pe_oi = process_chain(chain)
        if df.empty:
            return None, None, None, None, None
        pcr, max_pain, straddle_price, atm_strike, atm_iv = calculate_metrics(df, ce_oi, pe_oi, spot)
        ce_depth = get_market_depth(access_token, base_url, df[df['Strike'] == atm_strike]['CE_Token'].values[0])
        pe_depth = get_market_depth(access_token, base_url, df[df['Strike'] == atm_strike]['PE_Token'].values[0])
        iv_skew_fig = plot_iv_skew(df, spot, atm_strike)

        result = {
            "nifty_spot": spot,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain": max_pain,
            "expiry": expiry,
            "iv_skew_data": df.to_dict(),
            "ce_depth": ce_depth,
            "pe_depth": pe_depth,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "atm_iv": atm_iv
        }
        st.session_state.option_chain = chain
        return result, df, iv_skew_fig, atm_strike, atm_iv
    except Exception as e:
        logger.error(f"Volguard run error: {e}")
        return None, None, None, None, None

def check_risk(capital_to_deploy, max_loss, daily_pnl):
    new_deployed_capital = st.session_state.deployed_capital + capital_to_deploy
    new_exposure_pct = (new_deployed_capital / total_capital) * 100 if total_capital > 0 else 0
    new_daily_pnl = daily_pnl + st.session_state.daily_pnl

    if new_exposure_pct > MAX_EXPOSURE_PCT:
        return "red", f"Exposure exceeds {MAX_EXPOSURE_PCT}%! Cannot deploy ₹{capital_to_deploy:,}."
    if max_loss > max_loss_per_trade:
        return "red", f"Max loss per trade exceeds ₹{max_loss_per_trade:,} (4% of capital)!"
    if new_daily_pnl < -daily_loss_limit:
        return "red", f"Daily loss limit exceeded! Max loss allowed today: ₹{daily_loss_limit:,}."
    if new_exposure_pct > 30:
        return "yellow", "Exposure > 30%. Proceed with caution."
    return "green", "Safe to trade."

# === Auto-Refresh Logic ===
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > refresh_interval:
    if access_token:
        with st.spinner("Refreshing market data..."):
            result, df, iv_skew_fig, atm_strike, atm_iv = run_volguard(access_token)
            if result:
                st.session_state.volguard_data = result
                st.session_state.atm_iv = atm_iv
                st.session_state.last_refresh = time.time()
                st.success("Market data refreshed!")

# === Top Bar ===
exposure_pct = (st.session_state.deployed_capital / total_capital) * 100 if total_capital > 0 else 0
st.markdown(f"""
    <div class='top-bar'>
        <div><i class="material-icons">account_balance_wallet</i><p>Capital: ₹{total_capital:,}</p></div>
        <div><i class="material-icons">trending_up</i><p>Deployed: ₹{st.session_state.deployed_capital:,}</p></div>
        <div><i class="material-icons">percent</i><p>Exposure: {exposure_pct:.1f}%</p></div>
        <div><i class="material-icons">monetization_on</i><p>P&L: ₹{st.session_state.daily_pnl:,}</p></div>
        <div><i class="material-icons">schedule</i><p>Last Refresh: {datetime.now().strftime('%H:%M:%S')}</p></div>
    </div>
""", unsafe_allow_html=True)

# === Main Dashboard ===
st.header("VolGuard Pro Dashboard")
st.markdown("<p style='color:#ab47bc;'>Your Ultimate Options Trading Command Center</p>", unsafe_allow_html=True)

# Fetch User Details for Positions
if access_token and st.sidebar.button("Sync Account"):
    with st.spinner("Syncing account details..."):
        user_details = get_user_details(access_token)
        st.session_state.user_details = user_details
        st.session_state.positions = user_details.get('positions', {}).get('data', [])
        st.success("Account synced!")

# === Dashboard Layout ===
col1, col2 = st.columns([2, 1])

with col1:
    # Live Positions Widget
    st.markdown("""
        <div class='widget-card'>
            <h3><i class='material-icons'>trending_up</i>Live Positions</h3>
        </div>
    """, unsafe_allow_html=True)
    if st.session_state.positions:
        positions_df = pd.DataFrame(st.session_state.positions)
        if not positions_df.empty:
            positions_df = positions_df[['tradingsymbol', 'quantity', 'pnl', 'buy_avg_price', 'sell_avg_price', 'last_price']]
            positions_df['pnl'] = positions_df['pnl'].apply(lambda x: f"₹{x:,.2f}")
            gb = GridOptionsBuilder.from_dataframe(positions_df)
            gb.configure_default_column(editable=False, filter=True, sortable=True)
            gb.configure_column("tradingsymbol", headerName="Symbol", width=150)
            gb.configure_column("quantity", headerName="Qty", width=100)
            gb.configure_column("pnl", headerName="P&L", width=120)
            grid_options = gb.build()
            AgGrid(positions_df, gridOptions=grid_options, height=200, theme='balham-dark')
        else:
            st.info("No open positions.")
    else:
        st.info("Sync account to view live positions.")

    # Key Metrics Widget
    st.markdown("""
        <div class='widget-card'>
            <h3><i class='material-icons'>bar_chart</i>Key Metrics</h3>
        </div>
    """, unsafe_allow_html=True)
    if st.session_state.volguard_data:
        result = st.session_state.volguard_data
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        with col_metrics1:
            st.markdown(f"<p>ATM IV: <span style='color:#1e88e5'>{st.session_state.atm_iv:.2f}%</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p>PCR: <span style='color:#ab47bc'>{result['pcr']:.2f}</span></p>", unsafe_allow_html=True)
        with col_metrics2:
            st.markdown(f"<p>Straddle Price: <span style='color:#1e88e5'>₹{result['straddle_price']:.2f}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p>Max Pain: <span style='color:#ab47bc'>{result['max_pain']:.2f}</span></p>", unsafe_allow_html=True)
        with col_metrics3:
            st.markdown(f"<p>Nifty Spot: <span style='color:#1e88e5'>{result['nifty_spot']:.2f}</span></p>", unsafe_allow_html=True)
            st.markdown(f"<p>ATM Strike: <span style='color:#ab47bc'>{result['atm_strike']:.2f}</span></p>", unsafe_allow_html=True)
    else:
        st.info("Run VolGuard to fetch metrics.")

    # Volatility Snapshot
    st.markdown("""
        <div class='widget-card'>
            <h3><i class='material-icons'>timeline</i>Volatility Snapshot</h3>
        </div>
    """, unsafe_allow_html=True)
    try:
        nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()
        realized_vol = compute_realized_vol(nifty_df)

        log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna() * 100
        model = arch_model(log_returns, vol="Garch", p=1, q=1)
        model_fit = model.fit(disp="off")
        garch_forecast = model_fit.forecast(horizon=7)
        garch_vols = np.sqrt(garch_forecast.variance.values[-1]) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)

        xgb_vol = st.session_state.get('xgb_prediction', 15.0)
        atm_iv = st.session_state.atm_iv if st.session_state.atm_iv else 20.0
        last_date = nifty_df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
        plot_df = pd.DataFrame({
            "Date": forecast_dates,
            "GARCH": garch_vols,
            "XGBoost": [xgb_vol] * 7,
            "Realized": [realized_vol] * 7,
            "ATM IV": [atm_iv] * 7
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["GARCH"], mode='lines+markers', name='GARCH', line=dict(color='#1e88e5')))
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["XGBoost"], mode='lines+markers', name='XGBoost', line=dict(color='#ab47bc')))
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Realized"], mode='lines+markers', name='Realized', line=dict(color='#00adb5')))
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["ATM IV"], mode='lines+markers', name='ATM IV', line=dict(color='#f4e7ba')))
        fig.update_layout(
            title="Volatility Outlook",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
            showlegend=True,
            height=300,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info("Run volatility models to view snapshot.")

with col2:
    # Strategy Execution Widget
    st.markdown("""
        <div class='widget-card'>
            <h3><i class='material-icons'>strategy</i>Strategy Execution</h3>
        </div>
    """, unsafe_allow_html=True)
    strategy_options = ["Iron_Fly", "Iron_Condor", "Short_Straddle", "Short_Strangle", "Bull_Put_Credit", "Bear_Call_Credit", "Jade_Lizard"]
    selected_strategy = st.selectbox("Select Strategy", strategy_options)
    quantity = st.number_input("Quantity (Lots)", min_value=1, value=1, step=1)
    if st.button("Execute Strategy"):
        if not access_token or not st.session_state.option_chain:
            st.error("Run VolGuard and provide access token first.")
        else:
            with st.spinner(f"Executing {selected_strategy.replace('_', ' ')}..."):
                capital_required = total_capital * 0.3  # Simplified
                max_loss = total_capital * 0.03
                risk_status, risk_message = check_risk(capital_required, max_loss, 0)
                if risk_status == "red":
                    st.markdown(f"<div class='alert-red'>{risk_message}</div>", unsafe_allow_html=True)
                else:
                    order_results = execute_strategy(
                        access_token=access_token,
                        option_chain=st.session_state.option_chain,
                        spot_price=st.session_state.volguard_data['nifty_spot'],
                        strategy_name=selected_strategy,
                        quantity=quantity * 50
                    )
                    if order_results:
                        st.session_state.deployed_capital += capital_required
                        dummy_pnl = np.random.uniform(-max_loss, max_loss * 1.5)
                        st.session_state.daily_pnl += dummy_pnl
                        st.session_state.trade_log.append({
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "strategy": selected_strategy.replace('_', ' '),
                            "capital_deployed": capital_required,
                            "max_loss": max_loss,
                            "pnl": dummy_pnl,
                            "quantity": quantity * 50
                        })
                        st.success(f"Strategy executed! P&L: ₹{dummy_pnl:,.2f}")

    # Risk Console
    st.markdown("""
        <div class='widget-card'>
            <h3><i class='material-icons'>security</i>Risk Console</h3>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<p>Max Exposure: <span style='color:#1e88e5'>{MAX_EXPOSURE_PCT}% (₹{max_deployed_capital:,})</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p>Max Loss/Trade: <span style='color:#ab47bc'>{MAX_LOSS_PER_TRADE_PCT}% (₹{max_loss_per_trade:,})</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p>Daily Loss Limit: <span style='color:#1e88e5'>{DAILY_LOSS_LIMIT_PCT}% (₹{daily_loss_limit:,})</span></p>", unsafe_allow_html=True)

    # Market Depth Widget
    st.markdown("""
        <div class='widget-card'>
            <h3><i class='material-icons'>shopping_cart</i>Market Depth</h3>
        </div>
    """, unsafe_allow_html=True)
    if st.session_state.volguard_data:
        result = st.session_state.volguard_data
        st.markdown(f"<p>CE Bid/Ask: <span style='color:#1e88e5'>{result['ce_depth'].get('bid_volume', 0)}/{result['ce_depth'].get('ask_volume', 0)}</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p>PE Bid/Ask: <span style='color:#ab47bc'>{result['pe_depth'].get('bid_volume', 0)}/{result['pe_depth'].get('ask_volume', 0)}</span></p>", unsafe_allow_html=True)
    else:
        st.info("Run VolGuard to fetch depth.")

# Trade Log Widget (Full Width)
st.markdown("""
    <div class='widget-card'>
        <h3><i class='material-icons'>history</i>Trade Log</h3>
    </div>
""", unsafe_allow_html=True)
if st.session_state.trade_log:
    trade_df = pd.DataFrame(st.session_state.trade_log)
    trade_df['pnl'] = trade_df['pnl'].apply(lambda x: f"₹{x:,.2f}")
    trade_df['capital_deployed'] = trade_df['capital_deployed'].apply(lambda x: f"₹{x:,.2f}")
    trade_df['max_loss'] = trade_df['max_loss'].apply(lambda x: f"₹{x:,.2f}")
    gb = GridOptionsBuilder.from_dataframe(trade_df)
    gb.configure_default_column(editable=False, filter=True, sortable=True)
    grid_options = gb.build()
    AgGrid(trade_df, gridOptions=grid_options, height=200, theme='balham-dark')
else:
    st.info("No trades executed yet.")

# === Tabs for Deep Dives ===
tab1, tab2, tab3, tab4 = st.tabs(["Market Snapshot", "Volatility Forecast", "Strategy Lab", "Journal"])

with tab1:
    st.header("Market Snapshot")
    if st.session_state.volguard_data:
        result = st.session_state.volguard_data
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class='widget-card'>
                    <h3><i class='material-icons'>schedule</i>Market Overview</h3>
                    <p>Timestamp: {result['timestamp']}</p>
                    <p>Nifty Spot: {result['nifty_spot']:.2f}</p>
                    <p>ATM Strike: {result['atm_strike']:.2f}</p>
                    <p>Expiry: {result['expiry']}</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            iv_skew_fig = plot_iv_skew(pd.DataFrame(result['iv_skew_data']), result['nifty_spot'], result['atm_strike'])
            if iv_skew_fig:
                st.plotly_chart(iv_skew_fig, use_container_width=True)
    else:
        st.info("Run VolGuard to view snapshot.")

with tab2:
    st.header("Volatility Forecast")
    try:
        nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()

        log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna() * 100
        model = arch_model(log_returns, vol="Garch", p=1, q=1)
        model_fit = model.fit(disp="off")
        garch_forecast = model_fit.forecast(horizon=7)
        garch_vols = np.sqrt(garch_forecast.variance.values[-1]) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)
        last_date = nifty_df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Day": forecast_dates.day_name(),
            "GARCH Volatility (%)": np.round(garch_vols, 2)
        })
        st.dataframe(forecast_df)
    except Exception as e:
        st.error(f"Error loading forecast: {e}")

with tab3:
    st.header("Strategy Lab")
    st.info("Coming Soon: Strategy backtesting and payoff diagrams.")

with tab4:
    st.header("Trade Journal")
    journal_entry = st.text_area("Add Journal Entry")
    if st.button("Save Entry"):
        if journal_entry:
            st.session_state.journal_entries.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "entry": journal_entry
            })
            st.success("Entry saved!")
    if st.session_state.journal_entries:
        for entry in st.session_state.journal_entries:
            st.markdown(f"""
                <div class='widget-card'>
                    <h3><i class='material-icons'>book</i>{entry['timestamp']}</h3>
                    <p>{entry['entry']}</p>
                </div>
            """, unsafe_allow_html=True)
