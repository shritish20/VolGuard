import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime, timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import upstox_client
from upstox_client import Configuration, ApiClient, OptionsApi, UserApi, PortfolioApi, OrderApi, OrderApiV3, PlaceOrderV3Request
from upstox_client.rest import ApiException
import logging
import time
import plotly.graph_objects as go
import json
import retrying
import math

# Configure Logging
logging.basicConfig(filename='volguard.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
import xgboost as xgb
xgb.set_config(verbosity=0)

# === Streamlit Configuration ===
st.set_page_config(page_title="VolGuard Capital - Trading Platform", layout="wide")

# Custom CSS for Professional Web App UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    
    :root {
        --primary-bg: #0A0A0A;
        --secondary-bg: #1A1A1A;
        --primary-green: #00A86B;
        --alert-red: #FF4D4D;
        --highlight-gold: #FFD700;
        --text-primary: #F5F5F5;
        --text-secondary: #A0A0A0;
        --border-color: #2A2A2A;
    }
    
    body, .stApp {
        font-family: 'Inter', sans-serif;
        background: var(--primary-bg);
        color: var(--text-primary);
        margin: 0;
        padding: 0;
    }
    
    /* Top Navigation Bar */
    .nav-bar {
        background: var(--secondary-bg);
        padding: 12px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-color);
        position: fixed;
        top: 0;
        width: 100%;
        z-index: 1000;
    }
    .nav-bar .logo {
        display: flex;
        align-items: center;
    }
    .nav-bar .logo img {
        height: 32px;
        margin-right: 8px;
    }
    .nav-bar .logo span {
        font-size: 20px;
        font-weight: 700;
        color: var(--primary-green);
    }
    .nav-bar .nav-links a {
        color: var(--text-primary);
        text-decoration: none;
        margin: 0 16px;
        font-size: 14px;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    .nav-bar .nav-links a:hover, .nav-bar .nav-links a.active {
        color: var(--primary-green);
        border-bottom: 2px solid var(--primary-green);
        padding-bottom: 4px;
    }
    .nav-bar .user-profile {
        display: flex;
        align-items: center;
    }
    .nav-bar .user-profile select {
        background: var(--secondary-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        padding: 6px;
        font-size: 14px;
    }
    
    /* Sidebar */
    .sidebar {
        background: var(--secondary-bg);
        width: 280px;
        position: fixed;
        top: 60px;
        left: 0;
        height: calc(100vh - 60px);
        padding: 24px;
        border-right: 1px solid var(--border-color);
        transition: transform 0.3s ease;
    }
    .sidebar.collapsed {
        transform: translateX(-280px);
    }
    .sidebar h3 {
        color: var(--primary-green);
        font-size: 18px;
        margin-bottom: 16px;
    }
    .sidebar .stSlider, .sidebar .stSelectbox, .sidebar .stTextInput, .sidebar .stButton {
        margin-bottom: 16px;
    }
    .sidebar .stButton>button {
        background: var(--primary-green);
        color: var(--text-primary);
        border-radius: 6px;
        padding: 10px;
        font-weight: 500;
        width: 100%;
        transition: background 0.3s ease;
    }
    .sidebar .stButton>button:hover {
        background: #008B5A;
    }
    
    /* Main Content */
    .main-content {
        margin-top: 60px;
        margin-left: 280px;
        padding: 24px;
        transition: margin-left 0.3s ease;
    }
    .main-content.full-width {
        margin-left: 0;
    }
    .section {
        margin-bottom: 32px;
        background: var(--secondary-bg);
        border-radius: 8px;
        padding: 24px;
        border: 1px solid var(--border-color);
    }
    .section h2 {
        font-size: 24px;
        font-weight: 700;
        color: var(--primary-green);
        margin-bottom: 16px;
        display: flex;
        align-items: center;
    }
    .section h2 svg {
        margin-right: 8px;
        fill: var(--highlight-gold);
    }
    
    /* Market Pulse */
    .market-pulse {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin-bottom: 24px;
    }
    .market-pulse .metric {
        background: var(--primary-bg);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 12px;
        flex: 1;
        min-width: 150px;
        transition: transform 0.3s ease;
    }
    .market-pulse .metric:hover {
        transform: translateY(-2px);
    }
    .market-pulse .metric h4 {
        font-size: 14px;
        color: var(--text-secondary);
        margin: 0 0 4px 0;
        display: flex;
        align-items: center;
    }
    .market-pulse .metric h4 svg {
        margin-right: 4px;
        fill: var(--primary-green);
    }
    .market-pulse .metric p {
        font-size: 18px;
        font-weight: 500;
        color: var(--text-primary);
        margin: 0;
    }
    
    /* Tables */
    .stDataFrame table {
        background: var(--primary-bg);
        border-collapse: collapse;
        width: 100%;
    }
    .stDataFrame th {
        background: var(--secondary-bg);
        color: var(--primary-green);
        padding: 12px;
        text-align: left;
        position: sticky;
        top: 0;
        z-index: 10;
    }
    .stDataFrame td {
        padding: 12px;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-primary);
    }
    .stDataFrame tr:hover {
        background: #2A2A2A;
    }
    
    /* Buttons and Inputs */
    .stButton>button {
        background: var(--primary-green);
        color: var(--text-primary);
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        border: none;
        transition: background 0.3s ease;
    }
    .stButton>button:hover {
        background: #008B5A;
    }
    .stButton>button:disabled {
        background: #4A4A4A;
        cursor: not-allowed;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: var(--primary-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 10px;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: var(--text-secondary);
        font-size: 14px;
    }
    
    /* Alerts and Toasts */
    .alert {
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        font-size: 14px;
    }
    .alert-success {
        background: #1A3C34;
        color: var(--primary-green);
    }
    .alert-warning {
        background: #4A3C1A;
        color: var(--highlight-gold);
    }
    .alert-error {
        background: #4A1A1A;
        color: var(--alert-red);
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background: var(--secondary-bg);
        color: var(--text-primary);
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 0.3s ease-in;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .sidebar {
            width: 100%;
            height: auto;
            transform: translateX(0);
            position: relative;
        }
        .sidebar.collapsed {
            display: none;
        }
        .main-content {
            margin-left: 0;
        }
        .market-pulse .metric {
            min-width: 100%;
        }
        .nav-bar .nav-links {
            display: none;
        }
        .nav-bar .hamburger {
            display: block;
            cursor: pointer;
        }
    }
    </style>
    <!-- Heroicons CDN -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- JavaScript for Keyboard Shortcuts -->
    <script>
        document.addEventListener('keydown', function(event) {
            if (event.ctrlKey && event.key === 'r') {
                event.preventDefault();
                document.getElementById('refresh-data').click();
            }
            if (event.ctrlKey && event.key === 'e') {
                event.preventDefault();
                document.getElementById('execute-strategy').click();
            }
        });
    </script>
""", unsafe_allow_html=True)

# === Session State Initialization ===
# Existing logic unchanged
if 'volguard_data' not in st.session_state:
    st.session_state.volguard_data = None
if 'xgb_prediction' not in st.session_state:
    st.session_state.xgb_prediction = None
if 'atm_iv' not in st.session_state:
    st.session_state.atm_iv = 0.0
if 'realized_vol' not in st.session_state:
    st.session_state.realized_vol = 0.0
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
if 'journal_entries' not in st.session_state:
    st.session_state.journal_entries = []
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'deployed_capital' not in st.session_state:
    st.session_state.deployed_capital = 0.0
if 'daily_pnl' not in st.session_state:
    st.session_state.daily_pnl = 0.0
if 'user_details' not in st.session_state:
    st.session_state.user_details = None
if 'option_chain' not in st.session_state:
    st.session_state.option_chain = None
if 'trade_metrics' not in st.session_state:
    st.session_state.trade_metrics = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0.0,
        'pnl_history': []
    }
if 'risk_settings' not in st.session_state:
    st.session_state.risk_settings = {
        'max_exposure_pct': 50.0,
        'max_loss_per_trade_pct': 5.0,
        'daily_loss_limit_pct': 5.0
    }
if 'risk_status' not in st.session_state:
    st.session_state.risk_status = 'green'

# Initialize prev_oi globally
prev_oi = {}

# === Live Updates ===
# Check market hours (9:15 AM to 3:30 PM IST)
now = datetime.now()
market_open = now.time() >= datetime.strptime("09:15", "%H:%M").time() and now.time() <= datetime.strptime("15:30", "%H:%M").time0
refresh_counter = st_autorefresh(interval=5000, key="data-refresh")  # Refresh every 5 seconds

# === Top Navigation Bar ===
st.markdown(f"""
    <div class="nav-bar">
        <div class="logo">
            <img src="https://via.placeholder.com/32" alt="VolGuard Capital">
            <span>VolGuard Capital</span>
        </div>
        <div class="nav-links">
            <a href="#snapshot" class="{'' if st.session_state.get('page', 'snapshot') == 'snapshot' else 'active'}">Snapshot</a>
            <a href="#forecast">Forecast</a>
            <a href="#prediction">Prediction</a>
            <a href="#strategies">Strategies</a>
            <a href="#dashboard">Dashboard</a>
            <a href="#journal">Journal</a>
        </div>
        <div class="user-profile">
            <select onchange="alert('Profile action: ' + this.value)">
                <option value="profile">Profile</option>
                <option value="settings">Settings</option>
                <option value="logout">Logout</option>
            </select>
        </div>
    </div>
""", unsafe_allow_html=True)

# === Sidebar ===
sidebar_visible = st.session_state.get('sidebar_visible', True)
if st.button("☰ Toggle Sidebar", key="toggle-sidebar"):
    st.session_state.sidebar_visible = not sidebar_visible

st.markdown(f"""
    <div class="sidebar {'collapsed' if not sidebar_visible else ''}">
        <h3>Settings</h3>
""", unsafe_allow_html=True)

total_capital = st.slider("Total Capital (₹)", 100000, 5000000, 1000000, 10000, help="Your total trading capital.")
risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], help="Your risk tolerance.")
st.subheader("Risk Management")
max_exposure_pct = st.slider("Max Exposure (%)", 10.0, 100.0, st.session_state.risk_settings['max_exposure_pct'], 1.0)
max_loss_per_trade_pct = st.slider("Max Loss per Trade (%)", 1.0, 10.0, st.session_state.risk_settings['max_loss_per_trade_pct'], 0.1)
daily_loss_limit_pct = st.slider("Daily Loss Limit (%)", 1.0, 10.0, st.session_state.risk_settings['daily_loss_limit_pct'], 0.1)
access_token = st.text_input("Upstox Access Token", type="password", help="Enter your Upstox access token.")
run_engine = st.button("Run Engine", key="run-engine")
st.markdown("</div>", unsafe_allow_html=True)

# Update risk settings
st.session_state.risk_settings.update({
    'max_exposure_pct': max_exposure_pct,
    'max_loss_per_trade_pct': max_loss_per_trade_pct,
    'daily_loss_limit_pct': daily_loss_limit_pct
})

# Calculate risk limits
max_loss_per_trade = total_capital * (max_loss_per_trade_pct / 100)
daily_loss_limit = total_capital * (daily_loss_limit_pct / 100)
max_deployed_capital = total_capital * (max_exposure_pct / 100)

# === Main Content ===
st.markdown(f"""
    <div class="main-content {'full-width' if not sidebar_visible else ''}">
""", unsafe_allow_html=True)

# === Helper Functions ===
# Existing logic unchanged
@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_nearest_expiry(_options_api, instrument_key):
    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_expiry():
        try:
            response = _options_api.get_option_contracts(instrument_key=instrument_key)
            return response.to_dict().get("data", [])
        except ApiException as e:
            logger.error(f"Expiry fetch failed: {e}")
            raise

    try:
        contracts = fetch_expiry()
        expiry_dates = set()
        for contract in contracts:
            exp = contract.get("expiry")
            if isinstance(exp, str):
                exp = datetime.strptime(exp, "%Y-%m-%d")
            expiry_dates.add(exp)
        expiry_list = sorted(expiry_dates)
        today = datetime.now()
        valid_expiries = [e.strftime("%Y-%m-%d") for e in expiry_list if e >= today]
        return valid_expiries[0] if valid_expiries else None
    except Exception as e:
        logger.error(f"Expiry fetch error: {e}")
        return None

def fetch_option_chain(_options_api, instrument_key, expiry):
    @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_chain():
        try:
            res = _options_api.get_put_call_option_chain(instrument_key=instrument_key, expiry_date=expiry)
            return res.to_dict().get('data', [])
        except ApiException as e:
            logger.error(f"Option chain fetch failed: {e}")
            raise

    try:
        return fetch_chain()
    except Exception as e:
        logger.error(f"Option chain fetch error: {e}")
        return []

def process_chain(data):
    try:
        rows, ce_oi, pe_oi = [], 0, 0
        for r in data:
            ce = r.get('call_options', {})
            pe = r.get('put_options', {})
            ce_md, pe_md = ce.get('market_data', {}), pe.get('market_data', {})
            ce_gk, pe_gk = ce.get('option_greeks', {}), pe.get('option_greeks', {})
            strike = r.get('strike_price', 0)
            ce_oi_val = ce_md.get("oi", 0) or 0
            pe_oi_val = pe_md.get("oi", 0) or 0
            ce_oi_change = ce_oi_val - prev_oi.get(f"{strike}_CE", 0)
            pe_oi_change = pe_oi_val - prev_oi.get(f"{strike}_PE", 0)
            ce_oi_change_pct = (ce_oi_change / prev_oi.get(f"{strike}_CE", 1) * 100) if prev_oi.get(f"{strike}_CE", 0) else 0
            pe_oi_change_pct = (pe_oi_change / prev_oi.get(f"{strike}_PE", 1) * 100) if prev_oi.get(f"{strike}_PE", 0) else 0
            strike_pcr = pe_oi_val / (ce_oi_val or 1)
            row = {
                "Strike": strike,
                "CE_LTP": ce_md.get("ltp", 0) or 0,
                "CE_IV": ce_gk.get("iv", 0) or 0,
                "CE_Delta": ce_gk.get("delta", 0) or 0,
                "CE_Theta": ce_gk.get("theta", 0) or 0,
                "CE_Vega": ce_gk.get("vega", 0) or 0,
                "CE_OI": ce_oi_val,
                "CE_OI_Change": ce_oi_change,
                "CE_OI_Change_Pct": ce_oi_change_pct,
                "CE_Volume": ce_md.get("volume", 0) or 0,
                "PE_LTP": pe_md.get("ltp", 0) or 0,
                "PE_IV": pe_gk.get("iv", 0) or 0,
                "PE_Delta": pe_gk.get("delta", 0) or 0,
                "PE_Theta": pe_gk.get("theta", 0) or 0,
                "PE_Vega": pe_gk.get("vega", 0) or 0,
                "PE_OI": pe_oi_val,
                "PE_OI_Change": pe_oi_change,
                "PE_OI_Change_Pct": pe_oi_change_pct,
                "PE_Volume": pe_md.get("volume", 0) or 0,
                "Strike_PCR": strike_pcr,
                "CE_Token": ce.get("instrument_key", ""),
                "PE_Token": pe.get("instrument_key", "")
            }
            ce_oi += ce_oi_val
            pe_oi += pe_oi_val
            rows.append(row)
            prev_oi[f"{strike}_CE"] = ce_oi_val
            prev_oi[f"{strike}_PE"] = pe_oi_val
        df = pd.DataFrame(rows).sort_values("Strike")
        if not df.empty:
            df['OI_Skew'] = (df['PE_OI'] - df['CE_OI']) / (df['PE_OI'] + df['CE_OI'] + 1)
            valid_iv = df[(df['CE_IV'] > 0) & (df['PE_IV'] > 0)]
            if len(valid_iv) > 2:
                iv_diff = (valid_iv['PE_IV'] - valid_iv['CE_IV']).abs()
                df['IV_Skew_Slope'] = iv_diff.rolling(window=3).mean().reindex(df.index, fill_value=0)
            else:
                df['IV_Skew_Slope'] = 0
        return df, ce_oi, pe_oi
    except Exception as e:
        logger.error(f"Option chain processing error: {e}")
        return pd.DataFrame(), 0, 0

def calculate_metrics(df, ce_oi_total, pe_oi_total, spot):
    try:
        if df.empty:
            return 0, 0, 0, 0, 0
        atm = df.iloc[(df['Strike'] - spot).abs().argsort()[:1]]
        atm_strike = atm['Strike'].values[0] if not atm.empty else spot
        pcr = pe_oi_total / (ce_oi_total or 1)
        min_pain = float('inf')
        max_pain = spot
        for strike in df['Strike']:
            pain = 0
            for s in df['Strike']:
                if s <= strike:
                    pain += df[df['Strike'] == s]['CE_OI'].iloc[0] * max(0, strike - s)
                if s >= strike:
                    pain += df[df['Strike'] == s]['PE_OI'].iloc[0] * max(0, s - strike)
            if pain < min_pain:
                min_pain = pain
                max_pain = strike
        straddle_price = float(atm['CE_LTP'].values[0] + atm['PE_LTP'].values[0]) if not atm.empty else 0
        atm_iv = (atm['CE_IV'].values[0] + atm['PE_IV'].values[0]) / 2 if not atm.empty else 0
        return pcr, max_pain, straddle_price, atm_strike, atm_iv
    except Exception as e:
        logger.error(f"Metrics calculation error: {e}")
        return 0, 0, 0, 0, 0

def calculate_regime_score(atm_iv, realized_vol, pcr, vix=15.0, iv_skew_slope=0):
    try:
        score = 0
        if realized_vol > 0:
            iv_rv_ratio = atm_iv / realized_vol
            if iv_rv_ratio > 1.5:
                score += 30
            elif iv_rv_ratio > 1.2:
                score += 20
            elif iv_rv_ratio < 0.8:
                score -= 10
        ivp = 50.0
        if atm_iv > 25:
            ivp = 75
            score += 20
        elif atm_iv < 15:
            ivp = 25
            score -= 10
        if pcr > 1.5:
            score += 20
        elif pcr < 0.8:
            score -= 10
        if vix > 20:
            score += 20
        elif vix < 12:
            score -= 10
        if iv_skew_slope > 0.1:
            score += 10
        elif iv_skew_slope < -0.1:
            score -= 5
        score = max(0, min(100, score))
        if score > 80:
            regime = "High Vol Trend"
            explanation = "Market expects significant volatility. Consider hedged strategies like Iron Fly with long options."
        elif score > 60:
            regime = "Elevated Volatility"
            explanation = "Volatility is above average. Defensive strategies like Iron Condor are suitable."
        elif score > 40:
            regime = "Neutral Volatility"
            explanation = "Market is balanced. Explore strategies like Jade Lizard or Bull Put Credit."
        else:
            regime = "Low Volatility"
            explanation = "Market is calm. Aggressive strategies like Short Straddle may work, but monitor closely."
        return score, regime, explanation
    except Exception as e:
        logger.error(f"Regime score error: {e}")
        return 50, "Neutral Volatility", "Unable to classify regime due to data issues."

def plot_iv_skew(df, spot, atm_strike):
    try:
        valid = df[(df['CE_IV'] > 0) & (df['PE_IV'] > 0)]
        if valid.empty:
            return None
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['CE_IV'], mode='lines+markers', name='Call IV', line=dict(color='#00A86B')))
        fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['PE_IV'], mode='lines+markers', name='Put IV', line=dict(color='#FFD700')))
        fig.add_vline(x=spot, line=dict(color='#F5F5F5', dash='dash'), name='Spot')
        fig.add_vline(x=atm_strike, line=dict(color='#008B5A', dash='dot'), name='ATM')
        fig.update_layout(
            title="IV Skew",
            xaxis_title="Strike",
            yaxis_title="IV (%)",
            template="plotly_dark",
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='#0A0A0A',
            paper_bgcolor='#0A0A0A',
            font=dict(color='#F5F5F5')
        )
        return fig
    except Exception as e:
        logger.error(f"IV skew plot error: {e}")
        return None

@st.cache_data(ttl=5)
def get_market_depth(access_token, base_url, token):
    try:
        @retrying.retry(stop_max_attempt_number=3, wait_fixed=2000)
        def fetch_depth():
            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
            url = f"{base_url}/market-quote/depth"
            res = requests.get(url, headers=headers, params={"instrument_key": token})
            res.raise_for_status()
            return res.json().get('data', {}).get(token, {}).get('depth', {})

        depth = fetch_depth()
        bid_volume = sum(item.get('quantity', 0) for item in depth.get('buy', []))
        ask_volume = sum(item.get('quantity', 0) for item in depth.get('sell', []))
        return {"bid_volume": bid_volume, "ask_volume": ask_volume}
    except Exception as e:
        logger.error(f"Depth fetch error for {token}: {e}")
        return {"bid_volume": 0, "ask_volume": 0}

def compute_realized_vol(nifty_df):
    try:
        required_cols = ['NIFTY_Close']
        if not all(col in nifty_df.columns for col in required_cols):
            raise ValueError("CSV missing required column: 'NIFTY_Close'")
        log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna()
        last_7d_std = log_returns[-7:].std() * np.sqrt(252) * 100
        return last_7d_std if not np.isnan(last_7d_std) else 0
    except Exception as e:
        logger.error(f"Realized vol error: {e}")
        return 0

def calculate_rolling_and_fixed_hv(nifty_close):
    try:
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
        return rv_7d_df, hv_30d, hv_1y
    except Exception as e:
        logger.error(f"HV calculation error: {e}")
        return pd.DataFrame(), 0, 0

@st.cache_data(ttl=300)
def get_user_details(access_token):
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        user_api = UserApi(client)
        portfolio_api = PortfolioApi(client)
        order_api = OrderApi(client)
        details = {}
        details['profile'] = user_api.get_profile(api_version="v2").to_dict()
        details['funds'] = user_api.get_user_fund_margin(api_version="v2").to_dict()
        details['holdings'] = portfolio_api.get_holdings(api_version="v2").to_dict()
        details['positions'] = portfolio_api.get_positions(api_version="v2").to_dict()
        details['orders'] = order_api.get_order_book(api_version="v2").to_dict()
        details['trades'] = order_api.get_trade_history(api_version="v2").to_dict()
        return details
    except Exception as e:
        logger.error(f"User details fetch error: {e}")
        return {'error': str(e)}

def find_atm_strike(spot_price, strikes):
    try:
        return min(strikes, key=lambda x: abs(x - spot_price))
    except Exception as e:
        logger.error(f"ATM strike error: {e}")
        return spot_price

def build_strategy_legs(option_chain, spot_price, strategy_name, quantity, otm_distance=50):
    try:
        quantity = int(float(quantity))
        strikes = [leg['strike_price'] for leg in option_chain]
        atm_strike = find_atm_strike(spot_price, strikes)
        legs = []

        def get_key(strike, opt_type):
            for leg in option_chain:
                if leg['strike_price'] == strike:
                    if opt_type == 'CE':
                        return leg.get('call_options', {}).get('instrument_key')
                    elif opt_type == 'PE':
                        return leg.get('put_options', {}).get('instrument_key')
            return None

        s = strategy_name.lower()
        if s == "iron_fly":
            legs = [
                {"instrument_key": get_key(atm_strike, "CE"), "strike": atm_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike, "PE"), "strike": atm_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "iron_condor":
            legs = [
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike + 2 * otm_distance, "CE"), "strike": atm_strike + 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "strike": atm_strike - 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "short_straddle":
            legs = [
                {"instrument_key": get_key(atm_strike, "CE"), "strike": atm_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike, "PE"), "strike": atm_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "short_strangle":
            legs = [
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "bull_put_credit":
            legs = [
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "strike": atm_strike - 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "bear_call_credit":
            legs = [
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike + 2 * otm_distance, "CE"), "strike": atm_strike + 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "jade_lizard":
            legs = [
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "strike": atm_strike - 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        else:
            raise ValueError(f"Unsupported strategy: {strategy_name}")

        legs = [leg for leg in legs if leg["instrument_key"]]
        if not legs:
            raise ValueError("No valid legs generated due to missing instrument keys.")
        return legs
    except Exception as e:
        logger.error(f"Strategy legs error: {e}")
        return []

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
        logger.info(f"Placing order: {body.to_dict()}")
        response = order_api.place_order(body)
        st.toast(f"Order placed: {leg['action']} {leg['instrument_key']} qty={leg['quantity']}", icon="✅")
        return response.to_dict()
    except ApiException as e:
        error_msg = str(e)
        try:
            error_json = json.loads(e.body)
            reason = error_json.get("error", {}).get("message", "Unknown API error")
        except:
            reason = error_msg
        logger.error(f"Order fetal: {leg['instrument_key']}: {reason}")
        logger.error(f"Payload used: {body.to_dict()}")
        st.toast(f"Order failed for {leg['instrument_key']}: {reason}", icon="❌")
        return None

def fetch_trade_pnl(order_api, order_id):
    try:
        trade_details = order_api.get_order_details(order_id=order_id, api_version="v2").to_dict()
        trade_pnl = trade_details.get('data', {}).get('pnl', 0) or 0
        return trade_pnl
    except Exception as e:
        logger.error(f"Failed to fetch P&L for order {order_id}: {e}")
        return 0

def update_trade_metrics(pnl):
    try:
        metrics = st.session_state.trade_metrics
        metrics['total_trades'] += 1
        metrics['total_pnl'] += pnl
        if pnl > 0:
            metrics['winning_trades'] += 1
        elif pnl < 0:
            metrics['losing_trades'] += 1
        metrics['pnl_history'].append({"timestamp": datetime.now(), "pnl": pnl})
    except Exception as e:
        logger.error(f"Trade metrics update error: {e}")

def check_risk(capital_to_deploy, max_loss, daily_pnl, atm_iv, realized_vol):
    try:
        new_deployed_capital = st.session_state.deployed_capital + capital_to_deploy
        new_exposure_pct = (new_deployed_capital / total_capital) * 100 if total_capital > 0 else 0
        new_daily_pnl = daily_pnl + st.session_state.daily_pnl
        vol_factor = 1.0
        if atm_iv > 0 and realized_vol > 0:
            iv_rv_ratio = atm_iv / realized_vol
            if iv_rv_ratio > 1.5:
                vol_factor = 0.7
            elif iv_rv_ratio < 0.8:
                vol_factor = 1.2
        adjusted_max_exposure = max_deployed_capital * vol_factor
        adjusted_exposure_pct = (new_deployed_capital / adjusted_max_exposure) * 100 if adjusted_max_exposure > 0 else 0
        if new_exposure_pct > max_exposure_pct or new_deployed_capital > adjusted_max_exposure:
            return "red", f"Exposure exceeds {max_exposure_pct:.1f}% (adjusted: {adjusted_exposure_pct:.1f}%)!"
        if max_loss > max_loss_per_trade:
            return "red", f"Max loss per trade exceeds ₹{max_loss_per_trade:,.2f}!"
        if new_daily_pnl < -daily_loss_limit:
            return "red", f"Daily loss limit exceeded! Max: ₹{daily_loss_limit:,.2f}."
        if new_exposure_pct > max_exposure_pct * 0.8:
            return "yellow", f"Exposure nearing {max_exposure_pct}% (current: {new_exposure_pct:.1f}%)."
        return "green", "Safe to trade."
    except Exception as e:
        logger.error(f"Risk check error: {e}")
        return "red", "Risk calculation failed."

def execute_strategy(access_token, option_chain, spot_price, strategy_name, quantity, df):
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        order_api = OrderApiV3(client)
        quantity = int(float(quantity))
        legs = build_strategy_legs(option_chain, spot_price, strategy_name, quantity)
        if not legs:
            st.toast(f"Failed to build legs for {strategy_name}.", icon="❌")
            logger.error(f"No valid legs generated for {strategy_name}")
            return None, 0, 0, 0

        max_loss = 0
        entry_price = 0
        for leg in legs:
            try:
                strike = leg.get('strike', 0)
                qty = leg['quantity']
                opt_type = 'CE' if 'CALL' in leg['instrument_key'].upper() else 'PE'
                row = df[df['Strike'] == strike]
                if not row.empty:
                    ltp = float(row[f'{opt_type}_LTP'].iloc[0])
                    if leg['action'] == 'SELL':
                        max_loss += ltp * qty
                        entry_price += ltp * qty
                    else:
                        max_loss -= ltp * qty
                        entry_price -= ltp * qty
                else:
                    st.toast(f"No data for strike {strike} ({opt_type}).", icon="⚠️")
                    logger.warning(f"No data for strike {strike} ({opt_type})")
                    return None, 0, 0, 0
            except Exception as e:
                logger.error(f"Error calculating leg metrics for {leg['instrument_key']}: {e}")
                st.toast(f"Could not calculate metrics: {e}", icon="❌")
                return None, 0, 0, 0

        max_loss = abs(max_loss)
        capital_to_deploy = max_loss * 1.5
        risk_status, risk_message = check_risk(capital_to_deploy, max_loss, 0, st.session_state.atm_iv, st.session_state.realized_vol)
        if risk_status == "red":
            st.toast(risk_message, icon="❌")
            logger.error(f"Risk check failed: {risk_message}")
            return None, 0, 0, 0
        elif risk_status == "yellow":
            st.toast(risk_message, icon="⚠️")

        st.markdown("**Strategy Legs:**")
        for leg in legs:
            st.markdown(f"- {leg['action']} {leg['instrument_key']} (Strike: {leg.get('strike', 'N/A')}, Qty: {leg['quantity']})")

        order_results = []
        total_pnl = 0
        for leg in legs:
            result = place_order_for_leg(order_api, leg)
            if result:
                order_results.append(result)
                order_id = result.get('data', {}).get('order_id')
                if order_id:
                    time.sleep(2)
                    pnl = fetch_trade_pnl(order_api, order_id)
                    total_pnl += pnl
            else:
                return None, 0, 0, 0

        st.session_state.deployed_capital += capital_to_deploy
        st.session_state.daily_pnl += total_pnl
        update_trade_metrics(total_pnl)
        st.session_state.trade_log.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy_name.replace('_', ' '),
            "capital": capital_to_deploy,
            "pnl": total_pnl,
            "quantity": quantity * 75,
            "regime_score": regime_score if 'regime_score' in globals() else 0,
            "entry_price": entry_price,
            "max_loss": max_loss
        })

        logger.info(f"Strategy executed: {strategy_name}, P&L: {total_pnl}, Capital: {capital_to_deploy}")
        st.toast(f"Executed {strategy_name.replace('_', ' ')}! P&L: ₹{total_pnl:,.2f}", icon="✅")
        return order_results, total_pnl, entry_price, max_loss
    except Exception as e:
        logger.error(f"Strategy execution error: {e}")
        st.toast(f"Error executing strategy: {e}", icon="❌")
        return None, 0, 0, 0

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
            st.toast("Unable to fetch expiry date.", icon="❌")
            return None, None, None, None, None

        chain = fetch_option_chain(options_api, instrument_key, expiry)
        if not chain:
            st.toast("Unable to fetch option chain.", icon="❌")
            return None, None, None, None, None

        spot = chain[0].get("underlying_spot_price") or 0
        if not spot:
            st.toast("Unable to fetch spot price.", icon="❌")
            return None, None, None, None, None

        df, ce_oi, pe_oi = process_chain(chain)
        if df.empty:
            st.toast("Option chain data is empty.", icon="❌")
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
        st.toast("Failed to fetch options data.", icon="❌")
        return None, None, None, None, None

# === Snapshot Section ===
st.markdown("""
    <div class="section" id="snapshot">
        <h2><i class="fas fa-chart-line"></i> Market Snapshot</h2>
""", unsafe_allow_html=True)

if market_open:
    st.markdown(f'<span style="color: #00A86B; font-size: 14px;">Market Open</span>', unsafe_allow_html=True)
else:
    st.markdown(f'<span style="color: #FF4D4D; font-size: 14px;">Market Closed</span>', unsafe_allow_html=True)

if st.button("Refresh Data (Ctrl+R)", key="refresh-data"):
    if not access_token:
        st.toast("Please enter a valid Upstox access token.", icon="❌")
    else:
        with st.spinner("Fetching options data..."):
            result, df, iv_skew_fig, atm_strike, atm_iv = run_volguard(access_token)
            if result:
                st.session_state.volguard_data = result
                st.session_state.atm_iv = atm_iv
                st.toast("Data fetched successfully!", icon="✅")

if st.session_state.volguard_data:
    exposure_pct = (st.session_state.deployed_capital / total_capital) * 100 if total_capital > 0 else 0
    risk_status, risk_message = check_risk(0, 0, 0, st.session_state.atm_iv, st.session_state.realized_vol)
    st.session_state.risk_status = risk_status
    st.markdown(f"""
        <div class="market-pulse">
            <div class="metric">
                <h4><i class="fas fa-percentage"></i> Exposure</h4>
                <p>{exposure_pct:.1f}%</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-money-bill-wave"></i> Daily P&L</h4>
                <p>₹{st.session_state.daily_pnl:,.2f}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-exclamation-triangle"></i> Risk Status</h4>
                <p>{risk_status.capitalize()}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-clock"></i> Last Updated</h4>
                <p>{st.session_state.volguard_data['timestamp']}</p>
            </div>
        </div>
        <div class="alert alert-{risk_status}">
            {risk_message}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="market-pulse">
            <div class="metric">
                <h4><i class="fas fa-chart-bar"></i> Nifty Spot</h4>
                <p>{:.2f}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-percentage"></i> ATM IV</h4>
                <p>{:.2f}%</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-calendar"></i> Expiry</h4>
                <p>{}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-money-check-alt"></i> ATM Strike</h4>
                <p>{:.2f}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-dollar-sign"></i> Straddle Price</h4>
                <p>{:.2f}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-balance-scale"></i> PCR</h4>
                <p>{:.2f}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-bullseye"></i> Max Pain</h4>
                <p>{:.2f}</p>
            </div>
        </div>
    """.format(
        st.session_state.volguard_data['nifty_spot'],
        st.session_state.atm_iv,
        st.session_state.volguard_data['expiry'],
        st.session_state.volguard_data['atm_strike'],
        st.session_state.volguard_data['straddle_price'],
        st.session_state.volguard_data['pcr'],
        st.session_state.volguard_data['max_pain']
    ), unsafe_allow_html=True)

    if iv_skew_fig:
        st.subheader("IV Skew")
        st.plotly_chart(iv_skew_fig, use_container_width=True)

    st.subheader("Key Strikes")
    atm_idx = df[df['Strike'] == atm_strike].index[0]
    key_strikes = df.iloc[max(0, atm_idx-6):atm_idx+7][[
        'Strike', 'CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega', 'CE_OI',
        'CE_OI_Change', 'CE_OI_Change_Pct', 'CE_Volume', 'PE_LTP', 'PE_IV', 'PE_Delta',
        'PE_Theta', 'PE_Vega', 'PE_OI', 'PE_OI_Change', 'PE_OI_Change_Pct', 'PE_Volume',
        'Strike_PCR', 'OI_Skew', 'IV_Skew_Slope'
    ]]
    key_strikes['CE_OI_Change'] = key_strikes['CE_OI_Change'].apply(
        lambda x: f"{x:.1f}*" if x > 500000 else f"{x:.1f}"
    )
    key_strikes['PE_OI_Change'] = key_strikes['PE_OI_Change'].apply(
        lambda x: f"{x:.1f}*" if x > 500000 else f"{x:.1f}"
    )
    st.dataframe(key_strikes, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# === Forecast Section ===
st.markdown("""
    <div class="section" id="forecast">
        <h2><i class="fas fa-chart-area"></i> Volatility Forecast</h2>
""", unsafe_allow_html=True)

try:
    nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
    nifty_df.columns = nifty_df.columns.str.strip()
    nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
    nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
    nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
    if 'NIFTY_Close' not in nifty_df.columns:
        raise ValueError("CSV missing 'NIFTY_Close' column")
    nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()

    log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna() * 100
    model = arch_model(log_returns, vol="Garch", p=1, q=1)
    model_fit = model.fit(disp="off")
    forecast_horizon = 7
    garch_forecast = model_fit.forecast(horizon=forecast_horizon)
    garch_vols = np.sqrt(garch_forecast.variance.values[-1]) * np.sqrt(252)
    garch_vols = np.clip(garch_vols, 5, 50)
    last_date = nifty_df.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Day": forecast_dates.day_name(),
        "Forecasted Volatility (%)": np.round(garch_vols, 2)
    })

    st.session_state.realized_vol = compute_realized_vol(nifty_df)

    st.subheader("GARCH Forecast")
    st.dataframe(forecast_df, use_container_width=True)

    avg_vol = forecast_df["Forecasted Volatility (%)"].mean()
    st.subheader("Insight")
    if avg_vol > 20:
        st.markdown('<div class="alert alert-warning">High volatility (>20%). Consider defensive strategies like Iron Condor.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert alert-success">Moderate volatility. Explore strategies like Jade Lizard.</div>', unsafe_allow_html=True)

    rv_7d_df, hv_30d, hv_1y = calculate_rolling_and_fixed_hv(nifty_df["NIFTY_Close"])
    st.subheader("Historical Volatility")
    st.markdown(f"""
        <div class="market-pulse">
            <div class="metric">
                <h4><i class="fas fa-history"></i> 30-Day HV</h4>
                <p>{hv_30d:.2f}%</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-history"></i> 1-Year HV</h4>
                <p>{hv_1y:.2f}%</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

except Exception as e:
    logger.error(f"GARCH error: {e}")
    st.toast(f"Error loading GARCH data: {e}", icon="❌")

st.markdown("</div>", unsafe_allow_html=True)

# === Prediction Section ===
st.markdown("""
    <div class="section" id="prediction">
        <h2><i class="fas fa-brain"></i> Volatility Prediction</h2>
""", unsafe_allow_html=True)

xgb_model_url = "https://drive.google.com/uc?export=download&id=1Gs86p1p8wsGe1lp498KC-OVn0e87Gv-R"
xgb_csv_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/synthetic_volguard_dataset.csv"

st.subheader("Model Evaluation")
if st.button("Run Evaluation"):
    try:
        with st.spinner("Loading XGBoost data..."):
            xgb_df = pd.read_csv(xgb_csv_url)
            xgb_df = xgb_df.dropna()
            features = ['ATM_IV', 'Realized_Vol', 'IVP', 'Event_Impact_Score', 'FII_DII_Net_Long', 'PCR', 'VIX']
            target = 'Next_5D_Realized_Vol'
            if not all(col in xgb_df.columns for col in features + [target]):
                st.toast("CSV missing required columns!", icon="❌")
                st.stop()
            X = xgb_df[features]
            y = xgb_df[target] * 100

            response = requests.get(xgb_model_url)
            if response.status_code != 200:
                st.toast("Failed to load XGBoost model.", icon="❌")
                st.stop()
            xgb_model = pickle.loads(response.content)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            y_pred_train = xgb_model.predict(X_train)
            y_pred_test = xgb_model.predict(X_test)

            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            st.markdown("""
                <div class="market-pulse">
                    <div class="metric">
                        <h4><i class="fas fa-chart-bar"></i> Train RMSE</h4>
                        <p>{:.4f}%</p>
                    </div>
                    <div class="metric">
                        <h4><i class="fas fa-chart-bar"></i> Train MAE</h4>
                        <p>{:.4f}%</p>
                    </div>
                    <div class="metric">
                        <h4><i class="fas fa-chart-bar"></i> Train R²</h4>
                        <p>{:.4f}</p>
                    </div>
                    <div class="metric">
                        <h4><i class="fas fa-chart-bar"></i> Test RMSE</h4>
                        <p>{:.4f}%</p>
                    </div>
                    <div class="metric">
                        <h4><i class="fas fa-chart-bar"></i> Test MAE</h4>
                        <p>{:.4f}%</p>
                    </div>
                    <div class="metric">
                        <h4><i class="fas fa-chart-bar"></i> Test R²</h4>
                        <p>{:.4f}</p>
                    </div>
                </div>
            """.format(rmse_train, mae_train, r2_train, rmse_test, mae_test, r2_test), unsafe_allow_html=True)

            fig = go.Figure()
            importances = xgb_model.feature_importances_
            sorted_idx = np.argsort(importances)
            fig.add_trace(go.Bar(
                y=np.array(features)[sorted_idx],
                x=importances[sorted_idx],
                orientation='h',
                marker=dict(color='#00A86B')
            ))
            fig.update_layout(
                title="Feature Importances",
                xaxis_title="Importance",
                yaxis_title="Feature",
                template="plotly_dark",
                margin=dict(l=40, r=40, t=40, b=40),
                plot_bgcolor='#0A0A0A',
                paper_bgcolor='#0A0A0A',
                font=dict(color='#F5F5F5')
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")
        st.toast(f"Error running evaluation: {e}", icon="❌")

st.subheader("Predict Volatility")
try:
    nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
    nifty_df.columns = nifty_df.columns.str.strip()
    nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
    nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
    nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
    if 'NIFTY_Close' not in nifty_df.columns:
        raise ValueError("CSV missing 'NIFTY_Close' column")
    nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()
    realized_vol = compute_realized_vol(nifty_df)
except Exception as e:
    realized_vol = 0
    st.toast(f"Could not compute Realized Volatility: {e}", icon="⚠️")

atm_iv = st.session_state.volguard_data.get('atm_iv', 0) if st.session_state.volguard_data else 0
pcr = st.session_state.volguard_data.get('pcr', 0) if st.session_state.volguard_data else 0

col1, col2 = st.columns(2)
with col1:
    atm_iv_input = st.number_input("ATM IV (%)", value=float(atm_iv), min_value=0.0, max_value=100.0, step=0.1)
    realized_vol_input = st.number_input("Realized Volatility (%)", value=float(realized_vol), min_value=0.0, max_value=100.0, step=0.1)
    ivp_input = st.number_input("IV Percentile (0–100)", value=50.0, min_value=0.0, max_value=100.0, step=1.0)
with col2:
    event_score_input = st.number_input("Event Impact Score (0–2)", value=1.0, min_value=0.0, max_value=2.0, step=0.1)
    fii_dii_input = st.number_input("FII/DII Net Long (₹ Cr)", value=0.0, step=100.0)
    pcr_input = st.number_input("Put-Call Ratio", value=float(pcr), min_value=0.0, max_value=5.0, step=0.01)
    vix_input = st.number_input("VIX (%)", value=15.0, min_value=0.0, max_value=100.0, step=0.1)

if st.button("Predict"):
    try:
        with st.spinner("Predicting..."):
            response = requests.get(xgb_model_url)
            if response.status_code != 200:
                st.toast("Failed to load XGBoost model.", icon="❌")
                st.stop()
            xgb_model = pickle.loads(response.content)

            new_data = pd.DataFrame({
                'ATM_IV': [atm_iv_input],
                'Realized_Vol': [realized_vol_input],
                'IVP': [ivp_input],
                'Event_Impact_Score': [event_score_input],
                'FII_DII_Net_Long': [fii_dii_input],
                'PCR': [pcr_input],
                'VIX': [vix_input]
            })

            prediction = xgb_model.predict(new_data)[0]
            st.session_state.xgb_prediction = prediction
            st.markdown(f"""
                <div class="market-pulse">
                    <div class="metric">
                        <h4><i class="fas fa-brain"></i> Predicted Volatility</h4>
                        <p>{prediction:.2f}%</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            last_date = nifty_df.index[-1]
            xgb_forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
            xgb_forecast_df = pd.DataFrame({
                "Date": xgb_forecast_dates,
                "Day": xgb_forecast_dates.day_name(),
                "Predicted Volatility (%)": np.round([prediction]*7, 2)
            })
            st.dataframe(xgb_forecast_df, use_container_width=True)

            st.subheader("Insight")
            if prediction > 20:
                st.markdown('<div class="alert alert-warning">High volatility (>20%). Consider defensive strategies.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert alert-success">Moderate volatility. Explore strategies like Jade Lizard.</div>', unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.toast(f"Error predicting volatility: {e}", icon="❌")

st.markdown("</div>", unsafe_allow_html=True)

# === Strategies Section ===
st.markdown("""
    <div class="section" id="strategies">
        <h2><i class="fas fa-chess-knight"></i> Strategies</h2>
""", unsafe_allow_html=True)

strategy_options = [
    "Iron_Fly", "Iron_Condor", "Short_Straddle", "Short_Strangle",
    "Bull_Put_Credit", "Bear_Call_Credit", "Jade_Lizard"
]

strategy_tooltips = {
    "Iron_Fly": "Profits if market stays at the same price. Defined risk, suitable for calm markets.",
    "Iron_Condor": "Profits in a range-bound market. Defined risk, ideal for low volatility.",
    "Short_Straddle": "High reward for minimal movement, but unlimited risk.",
    "Short_Strangle": "Wider range than Straddle, high reward, unlimited risk.",
    "Bull_Put_Credit": "Profits if market rises. Limited reward, safer than directional bets.",
    "Bear_Call_Credit": "Profits if market falls. Limited reward, safer than directional bets.",
    "Jade_Lizard": "Profits if market rises slightly or stays flat. No upside risk."
}

if run_engine or st.session_state.strategies:
    try:
        nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        if 'NIFTY_Close' not in nifty_df.columns:
            raise ValueError("CSV missing 'NIFTY_Close' column")
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()
        realized_vol = compute_realized_vol(nifty_df)
    except Exception as e:
        realized_vol = 0
        logger.error(f"Realized vol fetch error: {e}")
        st.toast(f"Could not compute Realized Volatility: {e}", icon="⚠️")

    atm_iv = st.session_state.atm_iv
    pcr = st.session_state.volguard_data.get('pcr', 0) if st.session_state.volguard_data else 0
    iv_skew_slope = pd.DataFrame(st.session_state.volguard_data.get('iv_skew_data', {})).get('IV_Skew_Slope', pd.Series([0])).iloc[0] if st.session_state.volguard_data else 0

    regime_score, regime, regime_explanation = calculate_regime_score(atm_iv, realized_vol, pcr, vix=15.0, iv_skew_slope=iv_skew_slope)

    strategies = []
    if risk_profile == "Conservative":
        if regime in ["Neutral Volatility", "Low Volatility"]:
            strategies.append({
                "name": "Iron_Condor",
                "logic": "Ideal for range-bound markets with low volatility.",
                "capital_required": total_capital * 0.3,
                "max_loss": total_capital * 0.03,
                "confidence": 0.8
            })
        strategies.append({
            "name": "Iron_Fly",
            "logic": "Safe for stable markets with defined risk.",
            "capital_required": total_capital * 0.25,
            "max_loss": total_capital * 0.025,
            "confidence": 0.75
        })
    elif risk_profile == "Moderate":
        if regime == "Low Volatility":
            strategies.append({
                "name": "Bull_Put_Credit",
                "logic": "Suitable for bullish or stable markets.",
                "capital_required": total_capital * 0.3,
                "max_loss": total_capital * 0.03,
                "confidence": 0.7
            })
        elif regime == "Elevated Volatility":
            strategies.append({
                "name": "Bear_Call_Credit",
                "logic": "Suitable for bearish or volatile markets.",
                "capital_required": total_capital * 0.3,
                "max_loss": total_capital * 0.03,
                "confidence": 0.7
            })
        strategies.append({
            "name": "Jade_Lizard",
            "logic": "Balanced for slight upward movement.",
            "capital_required": total_capital * 0.35,
            "max_loss": total_capital * 0.035,
            "confidence": 0.8
        })
    elif risk_profile == "Aggressive":
        if regime in ["Low Volatility", "Neutral Volatility"]:
            strategies.append({
                "name": "Short_Straddle",
                "logic": "High reward in low volatility, but unlimited risk.",
                "capital_required": total_capital * 0.4,
                "max_loss": total_capital * 0.05,
                "confidence": 0.65
            })
            strategies.append({
                "name": "Short_Strangle",
                "logic": "Wider range, high reward in stable markets.",
                "capital_required": total_capital * 0.4,
                "max_loss": total_capital * 0.05,
                "confidence": 0.65
            })
        elif regime == "Elevated Volatility":
            strategies.append({
                "name": "Jade_Lizard",
                "logic": "Balanced for volatile markets with no upside risk.",
                "capital_required": total_capital * 0.35,
                "max_loss": total_capital * 0.035,
                "confidence": 0.75
            })

    st.session_state.strategies = strategies
    st.markdown(f"""
        <div class="market-pulse">
            <div class="metric">
                <h4><i class="fas fa-tachometer-alt"></i> Regime Score</h4>
                <p>{regime_score:.1f}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-signal"></i> Volatility Regime</h4>
                <p>{regime}</p>
            </div>
        </div>
        <div class="alert alert-info">{regime_explanation}</div>
    """, unsafe_allow_html=True)

    st.subheader("Recommended Strategies")
    for strategy in strategies:
        st.markdown(f"""
            <div class="section" style="padding: 16px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="tooltip">
                        <h4 style="margin: 0;">{strategy['name'].replace('_', ' ')}</h4>
                        <span class="tooltiptext">{strategy_tooltips[strategy['name']]}</span>
                    </div>
                    <button id="execute-strategy" onclick="document.getElementById('execute-{strategy['name']}').click();" style="background: var(--primary-green); color: var(--text-primary); border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer;">Execute (Ctrl+E)</button>
                </div>
                <p style="color: var(--text-secondary);">{strategy['logic']}</p>
                <div class="market-pulse">
                    <div class="metric">
                        <h4>Capital Required</h4>
                        <p>₹{strategy['capital_required']:,.2f}</p>
                    </div>
                    <div class="metric">
                        <h4>Max Loss</h4>
                        <p>₹{strategy['max_loss']:,.2f}</p>
                    </div>
                    <div class="metric">
                        <h4>Confidence</h4>
                        <p>{strategy['confidence']*100:.0f}%</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        with st.form(key=f"execute_form_{strategy['name']}"):
            quantity = st.number_input(f"Quantity (lots) for {strategy['name'].replace('_', ' ')}", min_value=1, value=1, step=1, key=f"qty_{strategy['name']}")
            submit = st.form_submit_button(f"Execute {strategy['name'].replace('_', ' ')}", key=f"execute-{strategy['name']}")
            if submit:
                if not access_token or not st.session_state.option_chain:
                    st.toast("Please run the engine with a valid access token.", icon="❌")
                else:
                    spot_price = st.session_state.volguard_data['nifty_spot']
                    with st.spinner(f"Executing {strategy['name'].replace('_', ' ')}..."):
                        order_results, total_pnl, entry_price, max_loss = execute_strategy(
                            access_token, st.session_state.option_chain, spot_price, strategy['name'], quantity, df
                        )
                        if order_results:
                            st.markdown(f"""
                                <div class="alert alert-success">
                                    Successfully executed {strategy['name'].replace('_', ' ')}! P&L: ₹{total_pnl:,.2f}
                                </div>
                            """, unsafe_allow_html=True)

st.subheader("Manual Strategy Execution")
with st.form("manual_strategy_form"):
    strategy_name = st.selectbox("Select Strategy", strategy_options, help="Choose a strategy to execute manually.")
    quantity_manual = st.number_input("Quantity (lots)", min_value=1, value=1, step=1)
    submit_manual = st.form_submit_button("Execute Strategy")
    if submit_manual:
        if not access_token or not st.session_state.option_chain:
            st.toast("Please run the engine with a valid access token.", icon="❌")
        else:
            spot_price = st.session_state.volguard_data['nifty_spot']
            with st.spinner(f"Executing {strategy_name.replace('_', ' ')}..."):
                order_results, total_pnl, entry_price, max_loss = execute_strategy(
                    access_token, st.session_state.option_chain, spot_price, strategy_name, quantity_manual, df
                )
                if order_results:
                    st.markdown(f"""
                        <div class="alert alert-success">
                            Successfully executed {strategy_name.replace('_', ' ')}! P&L: ₹{total_pnl:,.2f}
                        </div>
                    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# === Dashboard Section ===
st.markdown("""
    <div class="section" id="dashboard">
        <h2><i class="fas fa-tachometer-alt"></i> Trading Dashboard</h2>
""", unsafe_allow_html=True)

if st.button("Fetch Account Details"):
    if not access_token:
        st.toast("Please enter a valid Upstox access token.", icon="❌")
    else:
        with st.spinner("Fetching account details..."):
            st.session_state.user_details = get_user_details(access_token)
            if 'error' not in st.session_state.user_details:
                st.toast("Account details fetched successfully!", icon="✅")
            else:
                st.toast("Failed to fetch account details.", icon="❌")

if st.session_state.user_details and 'error' not in st.session_state.user_details:
    profile = st.session_state.user_details.get('profile', {}).get('data', {})
    funds = st.session_state.user_details.get('funds', {}).get('data', {})
    holdings = st.session_state.user_details.get('holdings', {}).get('data', [])
    positions = st.session_state.user_details.get('positions', {}).get('data', [])
    orders = st.session_state.user_details.get('orders', {}).get('data', [])
    trades = st.session_state.user_details.get('trades', {}).get('data', [])

    st.markdown("""
        <div class="market-pulse">
            <div class="metric">
                <h4><i class="fas fa-user"></i> User</h4>
                <p>{}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-wallet"></i> Available Margin</h4>
                <p>₹{:.2f}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-hand-holding-usd"></i> Used Margin</h4>
                <p>₹{:.2f}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-money-bill-wave"></i> Total P&L</h4>
                <p>₹{:.2f}</p>
            </div>
        </div>
    """.format(
        profile.get('user_name', 'N/A'),
        funds.get('equity', {}).get('available_margin', 0),
        funds.get('equity', {}).get('used_margin', 0),
        st.session_state.trade_metrics['total_pnl']
    ), unsafe_allow_html=True)

    st.subheader("Performance Metrics")
    metrics = st.session_state.trade_metrics
    st.markdown("""
        <div class="market-pulse">
            <div class="metric">
                <h4><i class="fas fa-exchange-alt"></i> Total Trades</h4>
                <p>{}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-trophy"></i> Winning Trades</h4>
                <p>{}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-exclamation-triangle"></i> Losing Trades</h4>
                <p>{}</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-chart-line"></i> Win Rate</h4>
                <p>{:.1f}%</p>
            </div>
        </div>
    """.format(
        metrics['total_trades'],
        metrics['winning_trades'],
        metrics['losing_trades'],
        (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
    ), unsafe_allow_html=True)

    st.subheader("Risk Analysis")
    exposure_pct = (st.session_state.deployed_capital / total_capital) * 100 if total_capital > 0 else 0
    st.markdown("""
        <div class="market-pulse">
            <div class="metric">
                <h4><i class="fas fa-percentage"></i> Current Exposure</h4>
                <p>{:.1f}%</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-exclamation-circle"></i> Max Exposure</h4>
                <p>{:.1f}%</p>
            </div>
            <div class="metric">
                <h4><i class="fas fa-shield-alt"></i> Daily Loss Limit</h4>
                <p>₹{:.2f}</p>
            </div>
        </div>
    """.format(
        exposure_pct,
        max_exposure_pct,
        daily_loss_limit
    ), unsafe_allow_html=True)

    st.subheader("Cumulative P&L")
    if metrics['pnl_history']:
        pnl_df = pd.DataFrame(metrics['pnl_history'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pnl_df['timestamp'],
            y=pnl_df['pnl'].cumsum(),
            mode='lines+markers',
            line=dict(color='#00A86B'),
            name='Cumulative P&L'
        ))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="P&L (₹)",
            template="plotly_dark",
            plot_bgcolor='#0A0A0A',
            paper_bgcolor='#0A0A0A',
            font=dict(color='#F5F5F5'),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# === Journal Section ===
st.markdown("""
    <div class="section" id="journal">
        <h2><i class="fas fa-book"></i> Trading Journal</h2>
""", unsafe_allow_html=True)

st.subheader("Trade Log")
if st.session_state.trade_log:
    trade_log_df = pd.DataFrame(st.session_state.trade_log)
    trade_log_df['pnl'] = trade_log_df['pnl'].apply(lambda x: f"₹{x:,.2f}")
    trade_log_df['capital'] = trade_log_df['capital'].apply(lambda x: f"₹{x:,.2f}")
    trade_log_df['entry_price'] = trade_log_df['entry_price'].apply(lambda x: f"₹{x:,.2f}")
    trade_log_df['max_loss'] = trade_log_df['max_loss'].apply(lambda x: f"₹{x:,.2f}")
    st.dataframe(trade_log_df, use_container_width=True)

    if st.button("Export Trade Log as CSV"):
        trade_log_df.to_csv("trade_log.csv", index=False)
        st.markdown("""
            <a href="data:text/csv;base64,{}" download="trade_log.csv" style="color: var(--primary-green); text-decoration: none;">
                Download Trade Log
            </a>
        """.format(
            base64.b64encode(trade_log_df.to_csv(index=False).encode()).decode()
        ), unsafe_allow_html=True)
        st.toast("Trade log exported!", icon="✅")

st.subheader("Journal Entries")
with st.form("journal_form"):
    journal_entry = st.text_area("New Journal Entry", height=100, placeholder="Record your thoughts, strategy rationale, or market observations...")
    submit_journal = st.form_submit_button("Save Entry")
    if submit_journal and journal_entry:
        st.session_state.journal_entries.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry": journal_entry
        })
        st.toast("Journal entry saved!", icon="✅")

if st.session_state.journal_entries:
    journal_df = pd.DataFrame(st.session_state.journal_entries)
    for idx, row in journal_df.iterrows():
        st.markdown(f"""
            <div class="section" style="padding: 16px; margin-bottom: 16px;">
                <p style="color: var(--text-secondary); font-size: 12px;">{row['date']}</p>
                <p>{row['entry']}</p>
                <button onclick="document.getElementById('delete-{idx}').click();" style="background: var(--alert-red); color: var(--text-primary); border: none; border-radius: 6px; padding: 8px 16px; cursor: pointer;">Delete</button>
            </div>
        """, unsafe_allow_html=True)
        if st.button(f"Delete Entry {idx}", key=f"delete-{idx}", help="Delete this journal entry"):
            st.session_state.journal_entries.pop(idx)
            st.experimental_rerun()

    if st.button("Export Journal as CSV"):
        journal_df.to_csv("journal.csv", index=False)
        st.markdown("""
            <a href="data:text/csv;base64,{}" download="journal.csv" style="color: var(--primary-green); text-decoration: none;">
                Download Journal
            </a>
        """.format(
            base64.b64encode(journal_df.to_csv(index=False).encode()).decode()
        ), unsafe_allow_html=True)
        st.toast("Journal exported!", icon="✅")

st.markdown("</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown("""
    <div style="padding: 16px; text-align: center; color: var(--text-secondary); font-size: 12px;">
        VolGuard Pro 2.0 | VolGuard Capital | <a href="mailto:support@volguardcapital.com" style="color: var(--primary-green); text-decoration: none;">support@volguardcapital.com</a>
    </div>
""", unsafe_allow_html=True)

# === Close Main Content ===
st.markdown("</div>", unsafe_allow_html=True)

# === Automatic Refresh Logic ===
if market_open and access_token and refresh_counter:
    try:
        result, df, iv_skew_fig, atm_strike, atm_iv = run_volguard(access_token)
        if result:
            st.session_state.volguard_data = result
            st.session_state.atm_iv = atm_iv
            st.toast("Data refreshed!", icon="✅")
        else:
            st.toast("Using cached data due to API issue.", icon="⚠️")
    except Exception as e:
        logger.error(f"Auto-refresh error: {e}")
        st.toast("Failed to refresh data.", icon="❌")
