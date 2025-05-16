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

# Configure Logging
logging.basicConfig(filename='volguard.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
import xgboost as xgb
xgb.set_config(verbosity=0)

# === Streamlit Configuration ===
st.set_page_config(page_title="VolGuard Pro 2.0 - AI Trading Copilot", layout="wide")

# Custom CSS for Polished UI
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
        import plotly.graph_objects as go
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
            return None, 0, 0, 0

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
                return None, 0, 0, 0

        max_loss = abs(max_loss)

        # Risk check
        capital_to_deploy = max_loss * 1.5  # Conservative buffer
        risk_status, risk_message = check_risk(capital_to_deploy, max_loss, 0, st.session_state.atm_iv, st.session_state.realized_vol)
        if risk_status == "red":
            st.error(risk_message)
            logger.error(f"Risk check failed: {risk_message}")
            return None, 0, 0, 0
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
                return None, 0, 0, 0

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

        return order_results, total_pnl, entry_price, max_loss

    except Exception as e:
        logger.error(f"Strategy execution error: {e}")
        st.error(f"Error executing strategy: {e}. Please check your inputs and try again.")
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

# === Streamlit Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Snapshot", "Forecast", "Prediction", "Strategies", "Dashboard", "Journal", "Backtest"
])

# === Tab 1: Snapshot ===
with tab1:
    st.header("Market Snapshot")
    access_token = st.text_input("Enter Upstox Access Token", type="password", help="Enter your Upstox access token to fetch live market data.")

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

                    st.subheader("Key Strikes (ATM ± 6)")
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
    strategy_options = [
        "Iron_Fly", "Iron_Condor", "Short_Straddle", "Short_Strangle",
        "Bull_Put_Credit", "Bear_Call_Credit", "Jade_Lizard"
    ]

    strategy_tooltips = {
        "Iron_Fly": "Profits if market stays at the same price. Defined risk, suitable for calm markets. Max loss is wings' cost minus premium.",
        "Iron_Condor": "Profits in a range-bound market. Defined risk, ideal for low volatility. Max loss is strike difference minus premium.",
        "Short_Straddle": "High reward for minimal movement, but unlimited risk. Use in low volatility with caution.",
        "Short_Strangle": "Wider range than Straddle, high reward, unlimited risk. Monitor volatility closely.",
        "Bull_Put_Credit": "Profits if market rises. Limited reward, safer than directional bets. Max loss is strike difference minus premium.",
        "Bear_Call_Credit": "Profits if market falls. Limited reward, safer than directional bets. Max loss is strike difference minus premium.",
        "Jade_Lizard": "Profits if market rises slightly or stays flat. No upside risk, but downside risk exists."
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
            st.warning(f"Could not compute Realized Volatility: {e}")

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
                "confidence": 0.75
            })
        elif risk_profile == "Aggressive":
            if regime in ["Neutral Volatility", "Low Volatility"]:
                strategies.append({
                    "name": "Short_Straddle",
                    "logic": "High reward for calm markets but risky.",
                    "capital_required": total_capital * 0.4,
                    "max_loss": total_capital * 0.04,
                    "confidence": 0.65
                })
                strategies.append({
                    "name": "Short_Strangle",
                    "logic": "Wider range for stable markets, high risk.",
                    "capital_required": total_capital * 0.35,
                    "max_loss": total_capital * 0.035,
                    "confidence": 0.6
                })

        st.session_state.strategies = strategies
        st.markdown(f"""
            <div class='highlight-card'>
                <h4><i class='material-icons'>assessment</i> Market Regime (Score: {regime_score})</h4>
                <p>Regime: {regime}</p>
                <p>Why: {regime_explanation}</p>
            </div>
        """, unsafe_allow_html=True)

        for strategy in strategies:
            strategy_name = strategy['name'].replace('_', ' ')
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>
                        <i class='material-icons'>strategy</i> 
                        {strategy_name}
                        <span class='tooltip'>
                            <i class='material-icons' style='font-size: 16px; margin-left: 5px;'>info</i>
                            <span class='tooltiptext'>{strategy_tooltips.get(strategy['name'], 'No explanation available.')}</span>
                        </span>
                    </h4>
                    <p>Logic: {strategy['logic']}</p>
                    <p>Capital Required: ₹{strategy['capital_required']:,.2f}</p>
                    <p>Max Loss: ₹{strategy['max_loss']:,.2f}</p>
                    <p>Confidence: {strategy['confidence']*100:.1f}%</p>
                    <p>Market Regime: {regime}</p>
                </div>
            """, unsafe_allow_html=True)

        st.subheader("Execute Strategy")
        selected_strategy = st.selectbox("Select Strategy to Execute", strategy_options, help="Choose a strategy to execute.")
        quantity = st.number_input("Quantity (Lots)", min_value=1, max_value=100, value=1, step=1, help="Number of lots to trade (1 lot = 75 contracts for Nifty).")
        if st.button("Execute Strategy"):
            if not access_token:
                st.error("Please enter a valid Upstox access token in the Snapshot tab.")
            elif not st.session_state.option_chain:
                st.error("Please run VolGuard in the Snapshot tab to fetch option chain data.")
            elif not st.session_state.volguard_data:
                st.error("No market data available. Please run VolGuard in the Snapshot tab.")
            else:
                try:
                    spot_price = st.session_state.volguard_data.get('nifty_spot', 0)
                    option_chain = st.session_state.option_chain
                    df = pd.DataFrame(st.session_state.volguard_data.get('iv_skew_data', {}))
                    if df.empty:
                        st.error("Option chain data is empty. Please try again.")
                    else:
                        order_results, trade_pnl, entry_price, max_loss, legs = execute_strategy(
                        access_token, option_chain, spot_price, selected_strategy, quantity, df
                        )

                        # Show Payout Chart
                        fig = generate_payout_chart(df, legs, spot_price)
                        if fig:
                            st.subheader("Payout Simulation")
                            st.plotly_chart(fig, use_container_width=True)
                        if order_results:
                            st.session_state.deployed_capital += max_loss * 1.5
                            st.session_state.daily_pnl += trade_pnl
                            update_trade_metrics(trade_pnl)
                            st.session_state.trade_log.append({
                                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "strategy": selected_strategy.replace('_', ' '),
                                "capital": max_loss * 1.5,
                                "pnl": trade_pnl,
                                "quantity": quantity * 75,
                                "regime_score": regime_score,
                                "entry_price": entry_price,
                                "max_loss": max_loss
                            })
                            logger.info(f"Trade executed: {selected_strategy}, P&L: {trade_pnl}, Capital: {max_loss * 1.5}")
                            st.markdown(f"<div class='alert-green'>Successfully executed {selected_strategy.replace('_', ' ')}! P&L: ₹{trade_pnl:,.2f}</div>", unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Strategy execution error: {e}")
                    st.error(f"Error executing strategy: {e}. Please check your inputs and try again.")
    else:
        st.warning("Run the engine from the sidebar to generate strategies.")

# === Tab 5: Dashboard ===
with tab5:
    st.header("Trading Dashboard")
    if access_token:
        try:
            user_details = get_user_details(access_token)
            if 'error' in user_details:
                st.error(f"Failed to fetch user details: {user_details['error']}")
            else:
                st.subheader("Account Details")
                col1, col2 = st.columns(2)
                with col1:
                    profile = user_details.get('profile', {}).get('data', {})
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>person</i> Name</h4><p>{profile.get('name', 'N/A')}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>email</i> Email</h4><p>{profile.get('email', 'N/A')}</p></div>", unsafe_allow_html=True)
                with col2:
                    funds = user_details.get('funds', {}).get('data', {})
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>account_balance_wallet</i> Equity Margin</h4><p>₹{funds.get('equity', {}).get('available_margin', 0):,.2f}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>monetization_on</i> Used Margin</h4><p>₹{funds.get('equity', {}).get('used_margin', 0):,.2f}</p></div>", unsafe_allow_html=True)

                st.subheader("Performance Metrics")
                metrics = st.session_state.trade_metrics
                win_rate = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
                sharpe_ratio = (np.mean([x['pnl'] for x in metrics['pnl_history']]) / np.std([x['pnl'] for x in metrics['pnl_history']]) * np.sqrt(252)) if metrics['pnl_history'] and np.std([x['pnl'] for x in metrics['pnl_history']]) != 0 else 0
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>bar_chart</i> Total Trades</h4><p>{metrics['total_trades']}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>trending_up</i> Win Rate</h4><p>{win_rate:.2f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>show_chart</i> Sharpe Ratio</h4><p>{sharpe_ratio:.2f}</p></div>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>monetization_on</i> Total P&L</h4><p>₹{metrics['total_pnl']:,.2f}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>thumb_up</i> Winning Trades</h4><p>{metrics['winning_trades']}</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>thumb_down</i> Losing Trades</h4><p>{metrics['losing_trades']}</p></div>", unsafe_allow_html=True)

                # Risk Analysis
                st.subheader("Risk Analysis")
                try:
                    pnl_series = pd.Series([x['pnl'] for x in metrics['pnl_history']])
                    drawdown = (pnl_series.cumsum() - pnl_series.cumsum().cummax()).min() if not pnl_series.empty else 0
                    var_95 = np.percentile(pnl_series, 5) if not pnl_series.empty else 0
                    stress_loss = var_95 * 2  # Simulate 2x volatility spike
                    col5, col6 = st.columns(2)
                    with col5:
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>warning</i> Max Drawdown</h4><p>₹{drawdown:,.2f}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>warning</i> VaR (95%)</h4><p>₹{var_95:,.2f}</p></div>", unsafe_allow_html=True)
                    with col6:
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>warning</i> Stress Test Loss</h4><p>₹{stress_loss:,.2f}</p></div>", unsafe_allow_html=True)
                        risk_status = st.session_state.risk_status
                        status_text = "Safe" if risk_status == "green" else "Warning" if risk_status == "yellow" else "Critical"
                        status_class = "alert-green" if risk_status == "green" else "alert-yellow" if risk_status == "yellow" else "alert-red"
                        st.markdown(f"<div class='{status_class}'><h4><i class='material-icons'>warning</i> Risk Status</h4><p>{status_text}</p></div>", unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Risk analysis error: {e}")
                    st.warning("Unable to compute risk metrics due to insufficient trade data.")

                # P&L Plot
                if metrics['pnl_history']:
                    st.subheader("Cumulative P&L")
                    pnl_df = pd.DataFrame(metrics['pnl_history'])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pnl_df['timestamp'],
                        y=pnl_df['pnl'].cumsum(),
                        mode='lines',
                        name='Cumulative P&L',
                        line=dict(color='#4CAF50')
                    ))
                    fig.update_layout(
                        title="Cumulative P&L Over Time",
                        xaxis_title="Date",
                        yaxis_title="P&L (₹)",
                        template="plotly_dark",
                        plot_bgcolor='#121212',
                        paper_bgcolor='#121212',
                        font=dict(color='#FAFAFA')
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            st.error(f"Error loading dashboard: {e}. Please check your access token.")
    else:
        st.warning("Enter your Upstox access token in the Snapshot tab to view account details.")

# === Tab 6: Journal ===
with tab6:
    st.header("Trading Journal")
    st.subheader("Add Journal Entry")
    journal_text = st.text_area("Journal Entry", height=100, help="Record your thoughts, strategy reflections, or market observations.")
    journal_regime_score = st.number_input("Regime Score (Optional)", min_value=0, max_value=100, step=1, value=regime_score if 'regime_score' in locals() else 50)
    if st.button("Save Journal Entry"):
        if journal_text.strip():
            entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": journal_text.strip(),
                "regime_score": journal_regime_score
            }
            st.session_state.journal_entries.append(entry)
            logger.info(f"Journal entry saved: {entry['timestamp']}")
            st.markdown("<div class='alert-green'>Journal entry saved!</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter some text before saving.")

    st.subheader("Trade Log")
    if st.session_state.trade_log:
        trade_df = pd.DataFrame(st.session_state.trade_log)
        trade_df = trade_df[['date', 'strategy', 'capital', 'pnl', 'quantity', 'regime_score', 'entry_price', 'max_loss']]
        trade_df.columns = ['Date', 'Strategy', 'Capital (₹)', 'P&L (₹)', 'Quantity', 'Regime Score', 'Entry Price (₹)', 'Max Loss (₹)']
        trade_df = trade_df.sort_values('Date', ascending=False)
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
                </div
            """, unsafe_allow_html=True)
        if st.button("Export Trade Log to CSV"):
            trade_df.to_csv("trade_log.csv", index=False)
            st.markdown("<div class='alert-green'>Trade log exported to trade_log.csv!</div>", unsafe_allow_html=True)
    else:
        st.info("No trades logged yet. Execute a strategy to populate the trade log.")

    st.subheader("Journal Entries")
    if st.session_state.journal_entries:
        for idx, entry in enumerate(st.session_state.journal_entries):
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>event_note</i> {entry['timestamp']}</h4>
                    <p>{entry['text']}</p>
                    <p>Regime Score: {entry['regime_score']}</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button(f"Delete Entry {idx + 1}", key=f"delete_{idx}"):
                st.session_state.journal_entries.pop(idx)
                logger.info(f"Journal entry deleted: {entry['timestamp']}")
                st.markdown("<div class='alert-green'>Journal entry deleted!</div>", unsafe_allow_html=True)
                st.experimental_rerun()
    else:
        st.info("No journal entries yet. Add one above.")

#tab 7 Backtest
with tab7:
    st.header("Strategy Backtest Simulator (Beta)")

    try:
        nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()

        st.success(f"Nifty history loaded: {nifty_df.index.min().date()} to {nifty_df.index.max().date()}")
    except Exception as e:
        st.error(f"Failed to load Nifty data: {e}")
        st.stop()

    strategy_choice = st.selectbox("Select Backtest Strategy", [
        "Iron_Fly", "Iron_Condor", "Bull_Put_Credit", "Bear_Call_Credit"
    ])
    threshold_vol = st.slider("Volatility Threshold (%)", 10.0, 30.0, 18.0, 0.5)

    st.markdown("Backtest assumes entry if GARCH 7D Vol > threshold.")

    if st.button("Run Backtest"):
        try:
            log_returns = np.log(nifty_df["NIFTY_Close"].pct_change() + 1).dropna() * 100
            model = arch_model(log_returns, vol="Garch", p=1, q=1)
            model_fit = model.fit(disp="off")
            forecast = model_fit.forecast(horizon=1, start=252)
            dates = forecast.variance.index
            vol_series = np.sqrt(forecast.variance.values.flatten()) * np.sqrt(252)
            vol_series = pd.Series(vol_series, index=dates)
            vol_series = vol_series[~vol_series.isna()]
            pnl = []
            equity = 0
            capital = 1000000  # virtual
            for date, vol in vol_series.iteritems():
                if vol > threshold_vol:
                    # Simulate strategy payoff range (simplified)
                    ret = np.random.normal(loc=0.002, scale=0.01)  # avg 0.2% return/day
                    pl = capital * ret
                else:
                    pl = 0
                equity += pl
                pnl.append({"Date": date, "Vol": vol, "P&L": pl, "Equity": equity})

            pnl_df = pd.DataFrame(pnl).set_index("Date")

            st.subheader("Backtest Results")
            st.line_chart(pnl_df["Equity"])

            total_trades = (pnl_df["P&L"] != 0).sum()
            wins = (pnl_df["P&L"] > 0).sum()
            losses = (pnl_df["P&L"] < 0).sum()

            st.markdown(f"- Total Trades: {total_trades}")
            st.markdown(f"- Win Rate: {wins / total_trades * 100:.2f}%")
            st.markdown(f"- Net P&L: ₹{pnl_df['P&L'].sum():,.2f}")
        except Exception as e:
            st.error(f"Backtest failed: {e}")

# === Final Footer ===
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4CAF50;'>Built by Shritish Shukla and Salman Azimuddin </p>", unsafe_allow_html=True)
