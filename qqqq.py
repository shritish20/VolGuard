import streamlit as st
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

# === Streamlit Configuration ===
st.set_page_config(page_title="VolGuard Pro 2.0 - AI Trading Copilot", layout="wide")

# Custom CSS for Polished UI
st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
.metric-card {
    padding: 8px;
    margin-bottom: 12px;
    background-color: #1e1e1e;
    border-radius: 10px;
    box-shadow: 0 0 6px rgba(0,0,0,0.4);
}
.metric-card h4 {
    font-size: 16px;
}
.tooltiptext {
    font-size: 12px;
}
.stButton>button {
    width: 100% !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
body {
    font-family: 'Segoe UI', sans-serif;
}
.metric-card, .highlight-card {
    padding: 10px;
    margin-bottom: 10px;
    background-color: #1e1e1e;
    border-radius: 12px;
    box-shadow: 0 0 6px rgba(0,0,0,0.3);
    color: #f1f1f1;
}
.metric-card h4, .highlight-card h4 {
    font-size: 15px;
    margin-bottom: 4px;
}
.metric-card p, .highlight-card p {
    font-size: 14px;
    margin: 0;
}
@media screen and (max-width: 600px) {
    .metric-card, .highlight-card {
        font-size: 13px;
        padding: 8px;
    }
    .metric-card h4, .highlight-card h4 {
        font-size: 14px;
    }
}
.stButton>button {
    width: 100% !important;
    font-size: 16px !important;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background: #121212;
        color: #FAFAFA;
    }
    .css-1d391kg {
        background: #1E1E1E;
        padding: 20px;
        border-right: 1px solid #4CAF50;
    }
    .css-1d391kg h1 {
        color: #4CAF50;
        font-size: 1.6em;
        margin-bottom: 20px;
    }
    .css-1d391kg .stButton>button {
        background: #4CAF50;
        color: #FAFAFA;
        border-radius: 8px;
        padding: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .css-1d391kg .stButton>button:hover {
        background: #388E3C;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.5);
    }
    .top-bar {
        background: #1E1E1E;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #4CAF50;
    }
    .top-bar div {
        margin: 0 15px;
        display: flex;
        align-items: center;
    }
    .top-bar div p {
        margin: 0 0 0 8px;
        font-size: 1.1em;
        color: #FAFAFA;
    }
    .top-bar i {
        color: #4CAF50;
    }
    .stTabs [role="tab"] {
        background: transparent;
        color: #FAFAFA;
        border-bottom: 2px solid transparent;
        padding: 10px 20px;
        margin-right: 5px;
        transition: all 0.3s ease;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        border-bottom: 2px solid #4CAF50;
        color: #4CAF50;
    }
    .stTabs [role="tab"]:hover {
        color: #FFA726;
    }
    .stButton>button {
        background: #4CAF50;
        color: #FAFAFA;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #388E3C;
        box-shadow: 0 0 15px rgba(76, 175, 80, 0.5);
    }
    .metric-card {
        background: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);
        border: 1px solid #4CAF50;
        transition: transform 0.3s ease;
        width: 100%;
        max-width: 600px;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    .metric-card h4 {
        color: #4CAF50;
        margin: 0;
        display: flex;
        align-items: center;
    }
    .metric-card h4 i {
        margin-right: 8px;
        color: #FFA726;
    }
    .metric-card p {
        color: #FAFAFA;
        font-size: 1.1em;
        margin: 5px 0 0 0;
    }
    .highlight-card {
        background: #4CAF50;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.5);
        border: 1px solid #388E3C;
        width: 100%;
        max-width: 600px;
    }
    .highlight-card h4 {
        color: #FAFAFA;
        margin: 0;
        display: flex;
        align-items: center;
    }
    .highlight-card h4 i {
        margin-right: 8px;
        color: #FAFAFA;
    }
    .highlight-card p {
        color: #FAFAFA;
        font-size: 1.3em;
        margin: 5px 0 0 0;
    }
    .alert-green {
        background-color: #388E3C;
        color: #FAFAFA;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-yellow {
        background-color: #FFA726;
        color: #121212;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-red {
        background-color: #EF5350;
        color: #FAFAFA;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    h1, h2, h3, h4 {
        color: #4CAF50;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes scaleUp {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1E1E1E;
        color: #FAFAFA;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    @media (max-width: 600px) {
        .metric-card, .highlight-card {
            max-width: 100%;
        }
        .top-bar {
            flex-direction: column;
            align-items: flex-start;
        }
        .top-bar div {
            margin: 5px 0;
        }
    }
    </style>
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
""", unsafe_allow_html=True)

# Configure Logging
logging.basicConfig(filename='volguard.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
import xgboost as xgb
xgb.set_config(verbosity=0)

# === Session State Initialization ===
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

# === Sidebar Controls ===
st.sidebar.header("VolGuard Pro 2.0 - Trading Copilot")
total_capital = st.sidebar.slider("Total Capital (₹)", 100000, 5000000, 1000000, 10000, help="Your total trading capital.")
risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], help="Your risk tolerance for strategy recommendations.")
st.sidebar.subheader("Risk Management Settings")
max_exposure_pct = st.sidebar.slider("Max Exposure (%)", 10.0, 100.0, st.session_state.risk_settings['max_exposure_pct'], 1.0, help="Maximum capital to deploy at any time.")
max_loss_per_trade_pct = st.sidebar.slider("Max Loss per Trade (%)", 1.0, 10.0, st.session_state.risk_settings['max_loss_per_trade_pct'], 0.1, help="Maximum loss allowed per trade.")
daily_loss_limit_pct = st.sidebar.slider("Daily Loss Limit (%)", 1.0, 10.0, st.session_state.risk_settings['daily_loss_limit_pct'], 0.1, help="Maximum daily loss allowed.")
run_engine = st.sidebar.button("Run Engine", help="Generate strategy recommendations.")

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

# === Top Bar (Quick Stats) ===
exposure_pct = (st.session_state.deployed_capital / total_capital) * 100 if total_capital > 0 else 0
st.markdown(f"""
    <div class='top-bar'>
        <div><i class="material-icons">percent</i><p>Exposure: {exposure_pct:.1f}%</p></div>
        <div><i class="material-icons">monetization_on</i><p>Daily P&L: ₹{st.session_state.daily_pnl:,.2f}</p></div>
        <div><i class="material-icons">warning</i><p>Risk Status: {st.session_state.risk_status.capitalize()}</p></div>
    </div>
""", unsafe_allow_html=True)

# === Robust Risk Manager ===
def check_risk(capital_to_deploy, max_loss, daily_pnl, atm_iv, realized_vol):
    try:
        new_deployed_capital = st.session_state.deployed_capital + capital_to_deploy
        new_exposure_pct = (new_deployed_capital / total_capital) * 100 if total_capital > 0 else 0
        new_daily_pnl = daily_pnl + st.session_state.daily_pnl

        # Volatility adjustment
        vol_factor = 1.0
        if atm_iv > 0 and realized_vol > 0:
            iv_rv_ratio = atm_iv / realized_vol
            if iv_rv_ratio > 1.5:
                vol_factor = 0.7  # Reduce exposure by 30% in high vol
            elif iv_rv_ratio < 0.8:
                vol_factor = 1.2  # Increase exposure in low vol

        adjusted_max_exposure = max_deployed_capital * vol_factor
        adjusted_exposure_pct = (new_deployed_capital / adjusted_max_exposure) * 100 if adjusted_max_exposure > 0 else 0

        if new_exposure_pct > max_exposure_pct or new_deployed_capital > adjusted_max_exposure:
            return "red", f"Exposure exceeds {max_exposure_pct:.1f}% (adjusted: {adjusted_exposure_pct:.1f}%)! Cannot deploy ₹{capital_to_deploy:,.2f}."
        if max_loss > max_loss_per_trade:
            return "red", f"Max loss per trade exceeds ₹{max_loss_per_trade:,.2f} ({max_loss_per_trade_pct}% of capital)!"
        if new_daily_pnl < -daily_loss_limit:
            return "red", f"Daily loss limit exceeded! Max loss allowed today: ₹{daily_loss_limit:,.2f}."
        if new_exposure_pct > max_exposure_pct * 0.8:
            return "yellow", f"Exposure nearing {max_exposure_pct}% (current: {new_exposure_pct:.1f}%). Proceed with caution."
        return "green", "Safe to trade."
    except Exception as e:
        logger.error(f"Risk check error: {e}")
        return "red", "Risk calculation failed. Please check inputs and try again."

# Initial risk check
risk_status, risk_message = check_risk(0, 0, 0, st.session_state.atm_iv, st.session_state.realized_vol)
st.session_state.risk_status = risk_status
if risk_status == "green":
    st.markdown(f"<div class='alert-green'>{risk_message}</div>", unsafe_allow_html=True)
elif risk_status == "yellow":
    st.markdown(f"<div class='alert-yellow'>{risk_message}</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='alert-red'>{risk_message}</div>", unsafe_allow_html=True)

# === Helper Functions ===
@st.cache_data(ttl=300)
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
    global prev_oi
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
        fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['CE_IV'], mode='lines+markers', name='Call IV', line=dict(color='#4CAF50')))
        fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['PE_IV'], mode='lines+markers', name='Put IV', line=dict(color='#FFA726')))
        fig.add_vline(x=spot, line=dict(color='#FAFAFA', dash='dash'), name='Spot')
        fig.add_vline(x=atm_strike, line=dict(color='#388E3C', dash='dot'), name='ATM')
        fig.update_layout(
            title="IV Skew",
            xaxis_title="Strike",
            yaxis_title="IV (%)",
            template="plotly_dark",
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font=dict(color='#FAFAFA')
        )
        return fig
    except Exception as e:
        logger.error(f"IV skew plot error: {e}")
        return None

@st.cache_data(ttl=300)
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
        # Ensure quantity is an integer
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

        # Filter out legs with missing instrument keys
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
            product="I",  # Use "D" if "I" fails
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

        st.success(f"✅ Order placed: {leg['action']} {leg['instrument_key']} qty={leg['quantity']}")
        return response.to_dict()

    except ApiException as e:
        error_msg = str(e)

        # Try extracting exact reason from error body
        try:
            error_json = json.loads(e.body)
            reason = error_json.get("error", {}).get("message", "Unknown API error")
        except:
            reason = error_msg

        logger.error(f"❌ Order failed for {leg['instrument_key']}: {reason}")
        logger.error(f"Payload used: {body.to_dict()}")

        st.error(f"❌ Order failed for {leg['instrument_key']}.\n\n**Reason:** {reason}")
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

def generate_payout_chart(df, legs, spot):
    try:
        strikes = df["Strike"].tolist()
        spot_range = np.linspace(spot * 0.95, spot * 1.05, 100)
        pnl = []

        for s in spot_range:
            total = 0
            for leg in legs:
                strike = leg.get('strike')
                qty = int(leg.get('quantity', 0)) * 75  # 75 contracts per Nifty lot
                opt_type = 'CE' if 'CE' in leg['instrument_key'] else 'PE'
                action = leg['action']

                intrinsic = max(0, s - strike) if opt_type == "CE" else max(0, strike - s)
                payoff = -intrinsic if action == "SELL" else intrinsic
                total += payoff * qty
            pnl.append(total)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=pnl, mode="lines", name="P/L", line=dict(color="#4CAF50")))
        fig.add_vline(x=spot, line=dict(color="white", dash="dash"))
        fig.update_layout(
            title="Strategy Payout at Expiry",
            xaxis_title="Spot Price at Expiry",
            yaxis_title="Net P&L (₹)",
            template="plotly_dark",
            plot_bgcolor="#121212",
            paper_bgcolor="#121212",
            font=dict(color="#FAFAFA"),
            height=400
        )
        return fig
    except Exception as e:
        st.error(f"Failed to generate payout chart: {e}")
        return None

def monte_carlo_expiry_simulation(legs, spot_price, num_simulations=1000, days_to_expiry=5, volatility=0.2):
    try:
        results = []
        for _ in range(num_simulations):
            daily_returns = np.random.normal(loc=0, scale=volatility / np.sqrt(252), size=days_to_expiry)
            simulated_spot = spot_price * np.prod(1 + daily_returns)
            total_pnl = 0
            for leg in legs:
                strike = leg['strike']
                qty = int(leg.get('quantity', 0)) * 75
                opt_type = 'CE' if 'CE' in leg['instrument_key'] else 'PE'
                action = leg['action']
                intrinsic = max(0, simulated_spot - strike) if opt_type == "CE" else max(0, strike - simulated_spot)
                payoff = -intrinsic if action == "SELL" else intrinsic
                total_pnl += payoff * qty
            results.append(total_pnl)
        return results
    except Exception as e:
        st.error(f"Monte Carlo simulation failed: {e}")
        return []

def execute_strategy(access_token, option_chain, spot_price, strategy_name, quantity, df):
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        order_api = OrderApiV3(client)

        # Convert quantity to integer (fix for float issue)
        quantity = int(float(quantity))

        # Build strategy legs
        legs = build_strategy_legs(option_chain, spot_price, strategy_name, quantity)
        if not legs:
            st.error(f"Failed to build legs for {strategy_name}.")
            logger.error(f"No valid legs generated for {strategy_name}")
            return None, 0, 0, 0, []

        # Display strategy legs
        st.write("**Strategy Legs:**")
        for leg in legs:
            st.write(f"- {leg['action']} {leg['instrument_key']} (Strike: {leg.get('strike', 'N/A')}, Qty: {leg['quantity']})")

        # Calculate max loss and entry price
        max_loss = 0
        entry_price = 0

        for leg in legs:
            try:
                strike = leg.get('strike', 0)
                qty = leg['quantity']  # Already an integer from build_strategy_legs
                opt_type = 'CE' if 'CALL' in leg['instrument_key'].upper() else 'PE'
                row = df[df['Strike'] == strike]
                if not row.empty:
                    ltp = float(row[f'{opt_type}_LTP'].iloc[0])
                    if leg['action'] == 'SELL':
                        max_loss += ltp * qty
                        entry_price += ltp * qty
                    else:  # BUY
                        max_loss -= ltp * qty
                        entry_price -= ltp * qty
                else:
                    st.warning(f"No data found for strike {strike} ({opt_type}).")
                    logger.warning(f"No data for strike {strike} ({opt_type})")
            except Exception as e:
                logger.error(f"Error calculating leg metrics for {leg['instrument_key']}: {e}")
                st.error(f"Could not calculate metrics for a leg: {e}")
                return None, 0, 0, 0, []

        max_loss = abs(max_loss)

        # Risk check
        capital_to_deploy = max_loss * 1.5  # Conservative buffer
        risk_status, risk_message = check_risk(capital_to_deploy, max_loss, 0, st.session_state.atm_iv, st.session_state.realized_vol)
        if risk_status == "red":
            st.error(risk_message)
            logger.error(f"Risk check failed: {risk_message}")
            return None, 0, 0, 0, []
        elif risk_status == "yellow":
            st.warning(risk_message)

        # Place orders
        st.write("\n**Placing Orders...**")
        order_results = []
        total_pnl = 0
        for leg in legs:
            result = place_order_for_leg(order_api, leg)
            if result:
                order_results.append(result)
                order_id = result.get('data', {}).get('order_id')
                if order_id:
                    time.sleep(2)  # Prevent API rate limiting
                    pnl = fetch_trade_pnl(order_api, order_id)
                    total_pnl += pnl
                st.success(f"Order placed: {leg['action']} {leg['instrument_key']} (Qty: {leg['quantity']})")
                logger.info(f"Order placed: {leg['action']} {leg['instrument_key']} qty={leg['quantity']}")
            else:
                st.error(f"Order failed for {leg['instrument_key']}")
                logger.error(f"Order failed for {leg['instrument_key']}")
                return None, 0, 0, 0, []

        # Update session state
        st.session_state.deployed_capital += capital_to_deploy
        st.session_state.daily_pnl += total_pnl
        update_trade_metrics(total_pnl)
        st.session_state.trade_log.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy_name.replace('_', ' '),
            "capital": capital_to_deploy,
            "pnl": total_pnl,
            "quantity": quantity * 75,  # Assuming 1 lot = 75 contracts for Nifty
            "regime_score": regime_score if 'regime_score' in globals() else 0,
            "entry_price": entry_price,
            "max_loss": max_loss
        })

        logger.info(f"Strategy executed: {strategy_name}, P&L: {total_pnl}, Capital: {capital_to_deploy}")
        st.markdown(f"<div class='alert-green'>Successfully executed {strategy_name.replace('_', ' ')}! P&L: ₹{total_pnl:,.2f}</div>", unsafe_allow_html=True)

        return order_results, total_pnl, entry_price, max_loss, legs

    except Exception as e:
        logger.error(f"Strategy execution error: {e}")
        st.error(f"Error executing strategy: {e}. Please check your inputs and try again.")
        return None, 0, 0, 0, []

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
            st.error("Unable to fetch the nearest expiry date. Please check your access token.")
            return None, None, None, None, None

        chain = fetch_option_chain(options_api, instrument_key, expiry)
        if not chain:
            st.error("Unable to fetch option chain data. Please check your access token.")
            return None, None, None, None, None

        spot = chain[0].get("underlying_spot_price") or 0
        if not spot:
            st.error("Unable to fetch spot price. Please check your access token.")
            return None, None, None, None, None

        df, ce_oi, pe_oi = process_chain(chain)
        if df.empty:
            st.error("Option chain data is empty. Please try again later.")
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
        st.error("Failed to fetch options data. Please check your Upstox access token.")
        return None, None, None, None, None

def calculate_discipline_score(trade_log, regime_score_threshold=60, max_trades_per_day=3):
    violations = []
    score = 100

    if not trade_log:
        return 100, violations

    trades_df = pd.DataFrame(trade_log)
    trades_df['date_only'] = pd.to_datetime(trades_df['date']).dt.date

    # Rule 1: Avoid trading in Risk-Red Regimes (Regime Score < threshold)
    risk_trades = trades_df[trades_df['regime_score'] < regime_score_threshold]
    if not risk_trades.empty:
        violations.append(f"{len(risk_trades)} trade(s) executed during Risk-Red regime.")
        score -= len(risk_trades) * 10

    # Rule 2: Avoid Overtrading (more than 3 trades per day)
    trade_counts = trades_df.groupby('date_only').size()
    over_trades = trade_counts[trade_counts > max_trades_per_day]
    if not over_trades.empty:
        violations.append(f"{len(over_trades)} day(s) with overtrading.")
        score -= len(over_trades) * 10

    # Rule 3: Risk-Reward Violation (Max Loss too high)
    high_risk = trades_df[trades_df['max_loss'] > 0.05 * 1000000]  # e.g. > 5% capital
    if not high_risk.empty:
        violations.append(f"{len(high_risk)} high-risk trade(s) exceeding 5% capital loss.")
        score -= len(high_risk) * 5

    score = max(score, 0)
    return score, violations

# === Streamlit Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Snapshot", "Forecast", "Prediction", "Strategies", "Dashboard", "Journal", "Backtest"
])

# === Tab 1: Snapshot ===
# === Tab 1: Snapshot ===
with tab1:
    st.header("Market Snapshot")
    access_token = st.text_input("Enter Upstox Access Token", type="password", help="Enter your Upstox access token to fetch live market data.")
    # Store the access token in session state for reuse across tabs
    st.session_state.access_token = access_token

    if st.button("Run VolGuard"):
        if not access_token:
            st.error("Please enter a valid Upstox access token.")
        else:
            with st.spinner("Fetching options data..."):
                result, df, iv_skew_fig, atm_strike, atm_iv = run_volguard(access_token)
                if result:
                    st.session_state.volguard_data = result
                    st.session_state.atm_iv = atm_iv
                    st.success("Data fetched successfully!")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("Market Snapshot")
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>schedule</i> Timestamp</h4><p>{result['timestamp']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>trending_up</i> Nifty Spot</h4><p>{result['nifty_spot']:.2f}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='highlight-card'><h4><i class='material-icons'>percent</i> ATM IV</h4><p>{atm_iv:.2f}%</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>event</i> Expiry</h4><p>{result['expiry']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>attach_money</i> ATM Strike</h4><p>{result['atm_strike']:.2f}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>monetization_on</i> Straddle Price</h4><p>{result['straddle_price']:.2f}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>balance</i> PCR</h4><p>{result['pcr']:.2f}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>warning</i> Max Pain</h4><p>{result['max_pain']:.2f}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>shopping_cart</i> CE Depth</h4><p>Bid: {result['ce_depth'].get('bid_volume', 0)}, Ask: {result['ce_depth'].get('ask_volume', 0)}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>shopping_cart</i> PE Depth</h4><p>Bid: {result['pe_depth'].get('bid_volume', 0)}, Ask: {result['pe_depth'].get('ask_volume', 0)}</p></div>", unsafe_allow_html=True)
                    with col2:
                        if iv_skew_fig:
                            st.subheader("IV Skew Plot")
                            st.plotly_chart(iv_skew_fig, use_container_width=True)

                    # Wrapped in expander for mobile-friendliness
                    with st.expander("Key Strikes (ATM ± 6)"):
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
                        for idx, row in key_strikes.iterrows():
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h4><i class='material-icons'>attach_money</i> Strike: {row['Strike']:.2f}</h4>
                                    <p>CE LTP: {row['CE_LTP']:.2f} | CE IV: {row['CE_IV']:.2f}% | CE OI: {row['CE_OI']:.0f} | OI Change: {row['CE_OI_Change']}</p>
                                    <p>PE LTP: {row['PE_LTP']:.2f} | PE IV: {row['PE_IV']:.2f}% | PE OI: {row['PE_OI']:.0f} | OI Change: {row['PE_OI_Change']}</p>
                                    <p>Strike PCR: {row['Strike_PCR']:.2f} | OI Skew: {row['OI_Skew']:.2f} | IV Skew Slope: {row['IV_Skew_Slope']:.2f}</p>
                                </div>
                            """, unsafe_allow_html=True)
                         

# === Tab 2: Forecast ===
with tab2:
    st.header("GARCH: 7-Day Volatility Forecast")
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

        st.subheader("GARCH Volatility Forecast")
        for idx, row in forecast_df.iterrows():
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>event</i> {row['Date'].strftime('%Y-%m-%d')} ({row['Day']})</h4>
                    <p>Forecasted Volatility: {row['Forecasted Volatility (%)']}%</p>
                </div>
            """, unsafe_allow_html=True)

        avg_vol = forecast_df["Forecasted Volatility (%)"].mean()
        st.subheader("Trading Insight")
        if avg_vol > 20:
            st.markdown("<div class='alert-yellow'>Volatility is high (>20%). Consider defensive strategies like Iron Condor.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-green'>Volatility is moderate. Explore strategies like Jade Lizard.</div>", unsafe_allow_html=True)

        rv_7d_df, hv_30d, hv_1y = calculate_rolling_and_fixed_hv(nifty_df["NIFTY_Close"])
        st.subheader("Historical Volatility")
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>history</i> 30-Day HV</h4><p>{hv_30d:.2f}%</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>history</i> 1-Year HV</h4><p>{hv_1y:.2f}%</p></div>", unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"GARCH error: {e}")
        st.error(f"Error loading GARCH data: {e}. Please check the CSV source.")

# === Tab 3: Prediction ===
with tab3:
    st.header("XGBoost: 7-Day Volatility Prediction")
    xgb_model_url = "https://drive.google.com/uc?export=download&id=1Gs86p1p8wsGe1lp498KC-OVn0e87Gv-R"
    xgb_csv_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/synthetic_volguard_dataset.csv"

    st.subheader("Evaluate Trained Model")
    if st.button("Run Model Evaluation"):
        try:
            with st.spinner("Loading XGBoost data and model..."):
                xgb_df = pd.read_csv(xgb_csv_url)
                xgb_df = xgb_df.dropna()
                features = ['ATM_IV', 'Realized_Vol', 'IVP', 'Event_Impact_Score', 'FII_DII_Net_Long', 'PCR', 'VIX']
                target = 'Next_5D_Realized_Vol'
                if not all(col in xgb_df.columns for col in features + [target]):
                    st.error("CSV missing required columns!")
                    st.stop()
                X = xgb_df[features]
                y = xgb_df[target] * 100

                response = requests.get(xgb_model_url)
                if response.status_code != 200:
                    st.error("Failed to load XGBoost model.")
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

                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h3>Training Metrics</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>bar_chart</i> RMSE</h4><p>{rmse_train:.4f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>bar_chart</i> MAE</h4><p>{mae_train:.4f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>bar_chart</i> R²</h4><p>{r2_train:.4f}</p></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<h3>Test Metrics</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>bar_chart</i> RMSE</h4><p>{rmse_test:.4f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>bar_chart</i> MAE</h4><p>{mae_test:.4f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>bar_chart</i> R²</h4><p>{r2_test:.4f}</p></div>", unsafe_allow_html=True)

                fig = go.Figure()
                importances = xgb_model.feature_importances_
                sorted_idx = np.argsort(importances)
                fig.add_trace(go.Bar(
                    y=np.array(features)[sorted_idx],
                    x=importances[sorted_idx],
                    orientation='h',
                    marker=dict(color='#4CAF50')
                ))
                fig.update_layout(
                    title="XGBoost Feature Importances",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    template="plotly_dark",
                    margin=dict(l=40, r=40, t=40, b=40),
                    plot_bgcolor='#121212',
                    paper_bgcolor='#121212',
                    font=dict(color='#FAFAFA')
                )
                st.subheader("Feature Importances")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Model evaluation error: {e}")
            st.error(f"Error running model evaluation: {e}. Please check the CSV source.")

    st.subheader("Predict with New Data")
    st.info("Use VolGuard data or enter values manually.")
    
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
        st.warning(f"Could not compute Realized Volatility: {e}")

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

    if st.button("Predict Volatility"):
        try:
            with st.spinner("Loading model and predicting..."):
                response = requests.get(xgb_model_url)
                if response.status_code != 200:
                    st.error("Failed to load XGBoost model.")
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
                st.markdown(f"<div class='highlight-card'><h4><i class='material-icons'>trending_up</i> Predicted 7-Day Volatility</h4><p>{prediction:.2f}%</p></div>", unsafe_allow_html=True)

                last_date = nifty_df.index[-1]
                xgb_forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
                xgb_forecast_df = pd.DataFrame({
                    "Date": xgb_forecast_dates,
                    "Day": xgb_forecast_dates.day_name(),
                    "Predicted Volatility (%)": np.round([prediction]*7, 2)
                })
                st.subheader("XGBoost Prediction")
                for idx, row in xgb_forecast_df.iterrows():
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4><i class='material-icons'>event</i> {row['Date'].strftime('%Y-%m-%d')} ({row['Day']})</h4>
                            <p>Predicted Volatility: {row['Predicted Volatility (%)']}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.subheader("Trading Insight")
                if prediction > 20:
                    st.markdown("<div class='alert-yellow'>Predicted volatility is high (>20%). Consider defensive strategies like Iron Condor.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-green'>Predicted volatility is moderate. Explore strategies like Jade Lizard.</div>", unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            st.error(f"Error predicting volatility: {e}. Please check the model source.")

# === Tab 4: Strategies ===
with tab4:
    st.header("Strategy Recommendations")
    st.subheader("Risk Guard Settings")
    max_iv_allowed = st.slider("Max IV Allowed (%)", 10.0, 35.0, 22.0, step=0.5)
    min_regime_score = st.slider("Minimum Regime Score", 0, 100, 50, step=5, help="Minimum regime score to allow trading.")
    max_trades_per_day = st.slider("Max Trades per Day", 1, 10, 3, step=1, help="Maximum number of trades allowed per day.")

    if run_engine:
        with st.spinner("Generating strategy recommendations..."):
            try:
                if st.session_state.volguard_data is None:
                    st.error("No market data available. Please run VolGuard in the Snapshot tab first.")
                    st.stop()

                spot_price = st.session_state.volguard_data.get('nifty_spot', 0)
                atm_iv = st.session_state.volguard_data.get('atm_iv', 0)
                pcr = st.session_state.volguard_data.get('pcr', 0)
                regime_score, regime, explanation = calculate_regime_score(
                    atm_iv, st.session_state.realized_vol, pcr, vix=15.0,
                    iv_skew_slope=st.session_state.volguard_data.get('iv_skew_data', {}).get('IV_Skew_Slope', 0)
                )

                strategies = [
                    {"name": "Iron_Fly", "max_loss": 0.03 * total_capital, "confidence": 0.85, "suitable_regime": "High Vol Trend"},
                    {"name": "Iron_Condor", "max_loss": 0.02 * total_capital, "confidence": 0.90, "suitable_regime": "Elevated Volatility"},
                    {"name": "Short_Straddle", "max_loss": 0.05 * total_capital, "confidence": 0.75, "suitable_regime": "Low Volatility"},
                    {"name": "Short_Strangle", "max_loss": 0.04 * total_capital, "confidence": 0.80, "suitable_regime": "Low Volatility"},
                    {"name": "Bull_Put_Credit", "max_loss": 0.02 * total_capital, "confidence": 0.88, "suitable_regime": "Neutral Volatility"},
                    {"name": "Bear_Call_Credit", "max_loss": 0.02 * total_capital, "confidence": 0.88, "suitable_regime": "Neutral Volatility"},
                    {"name": "Jade_Lizard", "max_loss": 0.025 * total_capital, "confidence": 0.87, "suitable_regime": "Neutral Volatility"}
                ]

                st.session_state.strategies = strategies
                st.subheader("Strategy Recommendations")
                st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>insights</i> Regime: {regime}</h4><p>{explanation}</p></div>", unsafe_allow_html=True)

                for strat in strategies:
                    confidence = strat['confidence'] * 100
                    suitability = "✅ Suitable" if strat['suitable_regime'] == regime else "⚠️ Less Suitable"
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4><i class='material-icons'>strategy</i> {strat['name'].replace('_', ' ')}</h4>
                            <p>Max Loss: ₹{strat['max_loss']:,.2f} | Confidence: {confidence:.1f}%</p>
                            <p>{suitability}</p>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                logger.error(f"Strategy recommendation error: {e}")
                st.error(f"Error generating strategies: {e}")

    st.subheader("Execute Strategy")
    # Check if strategies are available before showing the dropdown
    if not st.session_state.strategies:
        st.warning("No strategies available. Please generate strategy recommendations first by running the engine.")
        st.stop()

    selected_strategy = st.selectbox("Select Strategy", [s['name'] for s in st.session_state.strategies], help="Choose a strategy to execute.")
    quantity = st.number_input("Quantity (Lots)", min_value=1, max_value=100, value=1, step=1, help="Number of lots (1 lot = 75 contracts for Nifty).")
    otm_distance = st.slider("OTM Distance (₹)", 50, 500, 50, step=50, help="Distance for OTM strikes in strategies.")

    if st.button("Execute Strategy"):
        # Check if access token exists in session state
        if 'access_token' not in st.session_state or not st.session_state.access_token:
            st.error("Please enter your Upstox access token in the Snapshot tab first.")
            st.stop()
        if not selected_strategy:
            st.error("Please select a strategy.")
            st.stop()
        if not st.session_state.option_chain:
            st.error("No option chain data available. Please run VolGuard in the Snapshot tab first.")
            st.stop()
        if not st.session_state.volguard_data:
            st.error("No market data available. Please run VolGuard in the Snapshot tab first.")
            st.stop()

        spot_price = st.session_state.volguard_data.get('nifty_spot', 0)
        df = pd.DataFrame(st.session_state.volguard_data.get('iv_skew_data', {}))
        if df.empty:
            st.error("Option chain data is empty.")
            st.stop()

        # Guard 1: IV Check
        if st.session_state.atm_iv > max_iv_allowed:
            st.error(f"Trade blocked: ATM IV ({st.session_state.atm_iv}%) exceeds allowed max ({max_iv_allowed}%).")
            st.stop()

        # Guard 2: Regime Score
        regime_score, _, _ = calculate_regime_score(
            st.session_state.atm_iv, st.session_state.realized_vol,
            st.session_state.volguard_data.get('pcr', 0), vix=15.0
        )
        if regime_score < min_regime_score:
            st.error(f"Trade blocked: Regime score ({regime_score}) below minimum ({min_regime_score}).")
            st.stop()

        # Guard 3: Max Loss Check
        estimated_loss = total_capital * (max_loss_per_trade_pct / 100)
        if selected_strategy and any(s for s in st.session_state.strategies if s['name'] == selected_strategy and s['max_loss'] > estimated_loss):
            st.error(f"Trade blocked: Max loss exceeds {max_loss_per_trade_pct}% of capital.")
            st.stop()

        # Guard 4: Max Trades per Day
        trades_today = len([t for t in st.session_state.trade_log if pd.to_datetime(t['date']).date() == datetime.now().date()])
        if trades_today >= max_trades_per_day:
            st.error(f"Trade blocked: Max trades per day ({max_trades_per_day}) reached.")
            st.stop()

        with st.spinner("Executing strategy..."):
            order_results, trade_pnl, entry_price, max_loss, legs = execute_strategy(
                st.session_state.access_token, st.session_state.option_chain, spot_price, selected_strategy, quantity, df
            )

        # Show Payout Chart
        fig = generate_payout_chart(df, legs, spot_price)
        if fig:
            st.subheader("Payout Simulation")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Monte Carlo Risk Simulation")
            sim_days = st.slider("Days to Expiry", 1, 10, 5)
            sim_vol = st.slider("Assumed IV (%)", 10.0, 40.0, float(st.session_state.atm_iv), step=0.5)
            sim_button = st.button("Run Monte Carlo Simulation")

            if sim_button:
                with st.spinner("Simulating expiry outcomes..."):
                    simulated_pnl = monte_carlo_expiry_simulation(legs, spot_price, 1000, sim_days, sim_vol / 100)
                    pnl_df = pd.DataFrame(simulated_pnl, columns=["P&L"])
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Median P&L", f"₹{np.median(simulated_pnl):,.0f}")
                    col2.metric("Win %", f"{(pnl_df['P&L'] > 0).mean()*100:.1f}%")
                    col3.metric("Risk of Loss", f"{(pnl_df['P&L'] < 0).mean()*100:.1f}%")
                    st.subheader("Simulated Expiry P&L Distribution")
                    st.bar_chart(pnl_df["P&L"].value_counts().sort_index())

# === Tab 5: Dashboard ===
with tab5:
    st.header("Performance Dashboard")
    st.subheader("Trade Metrics")
    metrics = st.session_state.trade_metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", metrics['total_trades'])
    col2.metric("Winning Trades", metrics['winning_trades'])
    col3.metric("Losing Trades", metrics['losing_trades'])
    col4.metric("Total P&L (₹)", f"{metrics['total_pnl']:,.2f}")

    st.subheader("P&L History")
    if metrics['pnl_history']:
        pnl_df = pd.DataFrame(metrics['pnl_history'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pnl_df['timestamp'], y=pnl_df['pnl'],
            mode='lines+markers', name='P&L',
            line=dict(color='#4CAF50')
        ))
        fig.update_layout(
            title="P&L Over Time",
            xaxis_title="Timestamp",
            yaxis_title="P&L (₹)",
            template="plotly_dark",
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font=dict(color='#FAFAFA')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No P&L history available.")

    st.subheader("Discipline Score")
    discipline_score, violations = calculate_discipline_score(st.session_state.trade_log)
    st.markdown(f"<div class='highlight-card'><h4><i class='material-icons'>verified</i> Discipline Score</h4><p>{discipline_score}/100</p></div>", unsafe_allow_html=True)
    if violations:
        st.subheader("Violations")
        for v in violations:
            st.markdown(f"<div class='alert-red'>{v}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-green'>No trading violations detected.</div>", unsafe_allow_html=True)

# === Tab 6: Journal ===
with tab6:
    st.header("Trading Journal")
    st.subheader("Trade Log")
    if st.session_state.trade_log:
        trade_df = pd.DataFrame(st.session_state.trade_log)
        trade_df = trade_df.rename(columns={
            'date': 'Date',
            'strategy': 'Strategy',
            'capital': 'Capital (₹)',
            'pnl': 'P&L (₹)',
            'quantity': 'Quantity',
            'regime_score': 'Regime Score',
            'entry_price': 'Entry Price (₹)',
            'max_loss': 'Max Loss (₹)'
        })
        for idx, row in trade_df.iterrows():
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>history</i> {row['Date']}</h4>
                    <p>Strategy: {row['Strategy']}</p>
                    <p>Capital: ₹{row['Capital (₹)']:,.2f}</p>
                    <p>P&L: ₹{row['P&L (₹)']:,.2f}</p>
                    <p>Quantity: {row['Quantity']}</p>
                    <p>Regime Score: {row['Regime Score']}</p>
                    <p>Entry Price: ₹{row['Entry Price (₹)']:,.2f}</p>
                    <p>Max Loss: ₹{row['Max Loss (₹)']:,.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        st.subheader("Download Trade Log")
        csv = trade_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "trade_log.csv", "text/csv")
    else:
        st.info("No trades logged yet.")

    st.subheader("Journal Entries")
    journal_entry = st.text_area("Add Journal Entry", help="Record your thoughts, strategies, or market observations.")
    if st.button("Save Journal Entry"):
        if journal_entry:
            st.session_state.journal_entries.append({
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'entry': journal_entry
            })
            st.success("Journal entry saved!")
        else:
            st.error("Please enter a journal entry.")

    if st.session_state.journal_entries:
        st.subheader("Previous Entries")
        for entry in st.session_state.journal_entries:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>notes</i> {entry['date']}</h4>
                    <p>{entry['entry']}</p>
                </div>
            """, unsafe_allow_html=True)

# === Tab 7: Backtest ===
with tab7:
    st.header("Backtesting")
    st.subheader("Historical Strategy Performance")
    backtest_strategy = st.selectbox("Select Strategy to Backtest", [s['name'] for s in st.session_state.strategies], key="backtest_strategy")
    backtest_quantity = st.number_input("Backtest Quantity (Lots)", min_value=1, max_value=100, value=1, step=1, key="backtest_qty")
    backtest_period = st.slider("Backtest Period (Days)", 30, 365, 90, step=30, key="backtest_period")

    if st.button("Run Backtest"):
        try:
            with st.spinner("Running backtest..."):
                nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
                nifty_df.columns = nifty_df.columns.str.strip()
                nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
                nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
                nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
                if 'NIFTY_Close' not in nifty_df.columns:
                    raise ValueError("CSV missing 'NIFTY_Close' column")
                nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()

                backtest_data = nifty_df.tail(backtest_period)
                simulated_pnl = []
                for idx, row in backtest_data.iterrows():
                    spot = row['NIFTY_Close']
                    strikes = np.arange(spot - 500, spot + 500, 50)
                    atm_strike = find_atm_strike(spot, strikes)
                    legs = build_strategy_legs(
                        [{'strike_price': s, 'call_options': {'instrument_key': f'NSE_FO|CALL_{s}'}, 'put_options': {'instrument_key': f'NSE_FO|PUT_{s}'}} for s in strikes],
                        spot, backtest_strategy, backtest_quantity
                    )
                    expiry_spot = spot * (1 + np.random.normal(0, 0.02))
                    total_pnl = 0
                    for leg in legs:
                        strike = leg['strike']
                        qty = int(leg.get('quantity', 0)) * 75
                        opt_type = 'CE' if 'CE' in leg['instrument_key'] else 'PE'
                        action = leg['action']
                        intrinsic = max(0, expiry_spot - strike) if opt_type == "CE" else max(0, strike - expiry_spot)
                        payoff = -intrinsic if action == "SELL" else intrinsic
                        total_pnl += payoff * qty
                    simulated_pnl.append(total_pnl)

                backtest_df = pd.DataFrame({
                    'Date': backtest_data.index,
                    'P&L': simulated_pnl
                })
                total_pnl = backtest_df['P&L'].sum()
                win_rate = (backtest_df['P&L'] > 0).mean() * 100
                avg_pnl = backtest_df['P&L'].mean()
                max_drawdown = (backtest_df['P&L'].cumsum().cummax() - backtest_df['P&L'].cumsum()).max()

                st.subheader("Backtest Results")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total P&L (₹)", f"{total_pnl:,.2f}")
                col2.metric("Win Rate (%)", f"{win_rate:.1f}")
                col3.metric("Avg P&L per Trade (₹)", f"{avg_pnl:,.2f}")
                col4.metric("Max Drawdown (₹)", f"{max_drawdown:,.2f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=backtest_df['Date'], y=backtest_df['P&L'].cumsum(),
                    mode='lines', name='Cumulative P&L',
                    line=dict(color='#4CAF50')
                ))
                fig.update_layout(
                    title=f"Backtest: {backtest_strategy.replace('_', ' ')}",
                    xaxis_title="Date",
                    yaxis_title="Cumulative P&L (₹)",
                    template="plotly_dark",
                    plot_bgcolor='#121212',
                    paper_bgcolor='#121212',
                    font=dict(color='#FAFAFA')
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Backtest error: {e}")
            st.error(f"Error running backtest: {e}. Please check the CSV source.")

# === Footer ===
st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <p style='color: #FAFAFA;'>VolGuard Pro 2.0 - Built with ❤️ by Shritish Shukla & Salman Azim</p>
        <p style='color: #FAFAFA;'>For support, contact: shritish@example.com</p>
    </div>
""", unsafe_allow_html=True)
