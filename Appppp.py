# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import pickle
from datetime import datetime, timedelta
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import upstox_client
from upstox_client.rest import ApiException
import logging
import time
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb

# Suppress XGBoost warnings
xgb.set_config(verbosity=0)

# === Streamlit Configuration ===
st.set_page_config(page_title="VolGuard Pro", layout="wide")

# Custom CSS for Sexy Look
st.markdown("""
    <style>
    body {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
    }
    .stTabs [role="tab"] {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        color: #e0e0e0;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
        border: 1px solid #ff6b6b;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
        color: white;
        border: 1px solid #ff6b6b;
    }
    .stButton>button {
        background: linear-gradient(135deg, #e94560 0%, #ff6b6b 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff6b6b 0%, #e94560 100%);
        box-shadow: 0 0 10px rgba(233, 69, 96, 0.5);
    }
    .metric-card {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    .metric-card h4 {
        color: #ff6b6b;
        margin: 0;
    }
    .metric-card p {
        color: #e0e0e0;
        font-size: 1.2em;
        margin: 5px 0 0 0;
    }
    .highlight-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #e94560 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.5);
        animation: fadeIn 0.5s ease-in;
    }
    .highlight-card h4 {
        color: white;
        margin: 0;
    }
    .highlight-card p {
        color: white;
        font-size: 1.5em;
        margin: 5px 0 0 0;
    }
    .alert-green {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-in;
    }
    .alert-yellow {
        background-color: #ffc107;
        color: black;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-in;
    }
    .alert-red {
        background-color: #dc3545;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-in;
    }
    .top-bar {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .top-bar div {
        margin: 0 10px;
    }
    .top-bar div p {
        margin: 0;
        font-size: 1.1em;
    }
    h1, h2, h3, h4 {
        color: #ff6b6b;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# === Session State to Store Data ===
if 'volguard_data' not in st.session_state:
    st.session_state.volguard_data = None
if 'xgb_prediction' not in st.session_state:
    st.session_state.xgb_prediction = None
if 'atm_iv' not in st.session_state:
    st.session_state.atm_iv = None
if 'strategies' not in st.session_state:
    st.session_state.strategies = None
if 'journal_entries' not in st.session_state:
    st.session_state.journal_entries = []
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = []
if 'deployed_capital' not in st.session_state:
    st.session_state.deployed_capital = 0
if 'daily_pnl' not in st.session_state:
    st.session_state.daily_pnl = 0

# === Sidebar Controls ===
st.sidebar.header("VolGuard Pro Controls")
total_capital = st.sidebar.slider("Total Capital (₹)", 100000, 5000000, 1000000, 10000)
risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
run_engine = st.sidebar.button("Run Engine")

# === Risk Management Rules ===
MAX_EXPOSURE_PCT = 40  # Max 40% of total capital can be deployed
MAX_LOSS_PER_TRADE_PCT = 4  # Max 4% of total capital per trade
DAILY_LOSS_LIMIT_PCT = 4  # Max 4% of total capital loss in a day
max_loss_per_trade = total_capital * (MAX_LOSS_PER_TRADE_PCT / 100)
daily_loss_limit = total_capital * (DAILY_LOSS_LIMIT_PCT / 100)
max_deployed_capital = total_capital * (MAX_EXPOSURE_PCT / 100)

# === Top Bar (Quick Stats) ===
exposure_pct = (st.session_state.deployed_capital / total_capital) * 100 if total_capital > 0 else 0
st.markdown(f"""
    <div class='top-bar'>
        <div><p>Total Capital: ₹{total_capital:,}</p></div>
        <div><p>Deployed Capital: ₹{st.session_state.deployed_capital:,}</p></div>
        <div><p>Exposure: {exposure_pct:.1f}%</p></div>
        <div><p>Daily P&L: ₹{st.session_state.daily_pnl:,}</p></div>
    </div>
""", unsafe_allow_html=True)

# === Risk Manager ===
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

# IV-RV will be calculated after VolGuard runs for additional risk insight
iv_rv = 0  # Default value, will be updated after VolGuard
risk_status, risk_message = check_risk(0, 0, 0)  # Initial check
if risk_status == "green":
    st.markdown(f"<div class='alert-green'>{risk_message}</div>", unsafe_allow_html=True)
elif risk_status == "yellow":
    st.markdown(f"<div class='alert-yellow'>{risk_message}</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='alert-red'>{risk_message}</div>", unsafe_allow_html=True)

# === Helper Functions ===
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
        ce_oi_change_pct = (ce_oi_change / prev_oi[f"{strike}_CE"] * 100) if prev_oi.get(f"{strike}_CE", 0) else 0
        pe_oi_change_pct = (pe_oi_change / prev_oi[f"{strike}_PE"] * 100) if prev_oi.get(f"{strike}_PE", 0) else 0
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
            "PE_Delta": pe_gk.get("delta"),
            "PE_Theta": pe_gk.get("theta"),
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
    fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['CE_IV'], mode='lines+markers', name='Call IV', line=dict(color='#e94560')))
    fig.add_trace(go.Scatter(x=valid['Strike'], y=valid['PE_IV'], mode='lines+markers', name='Put IV', line=dict(color='#ff6b6b')))
    fig.add_vline(x=spot, line=dict(color='gray', dash='dash'), name='Spot')
    fig.add_vline(x=atm_strike, line=dict(color='green', dash='dot'), name='ATM')
    fig.update_layout(
        title="IV Skew",
        xaxis_title="Strike",
        yaxis_title="IV",
        template="plotly_dark",
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40)
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
        return result, df, iv_skew_fig, atm_strike, atm_iv
    except Exception as e:
        logger.error(f"Volguard run error: {e}")
        return None, None, None, None, None

# === Streamlit Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Snapshot", "Forecast", "Prediction", "Strategies", "Dashboard"])

# === Tab 1: VolGuard ===
with tab1:
    st.header("Market Snapshot")
    st.warning("Upstox API is available from 5:30 AM to 12:00 AM IST. Data may not load outside these hours.")
    access_token = st.text_input("Enter Upstox Access Token", type="password")
    
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
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Market Snapshot")
                        st.markdown(f"<div class='metric-card'><h4>Timestamp</h4><p>{result['timestamp']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4>Nifty Spot</h4><p>{result['nifty_spot']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='highlight-card'><h4>ATM IV</h4><p>{atm_iv:.2f}%</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4>Expiry</h4><p>{result['expiry']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4>ATM Strike</h4><p>{result['atm_strike']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4>Straddle Price</h4><p>{result['straddle_price']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4>PCR</h4><p>{result['pcr']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4>Max Pain</h4><p>{result['max_pain']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4>CE Depth</h4><p>Bid Volume={result['ce_depth'].get('bid_volume', 0)}, Ask Volume={result['ce_depth'].get('ask_volume', 0)}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4>PE Depth</h4><p>Bid Volume={result['pe_depth'].get('bid_volume', 0)}, Ask Volume={result['pe_depth'].get('ask_volume', 0)}</p></div>", unsafe_allow_html=True)
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
                        'Strike_PCR'
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
                                <h4>Strike: {row['Strike']}</h4>
                                <p>CE LTP: {row['CE_LTP']:.2f} | CE IV: {row['CE_IV']:.2f} | CE OI: {row['CE_OI']:.2f}</p>
                                <p>PE LTP: {row['PE_LTP']:.2f} | PE IV: {row['PE_IV']:.2f} | PE OI: {row['PE_OI']:.2f}</p>
                                <p>Strike PCR: {row['Strike_PCR']:.2f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error("Failed to fetch options data. Check your access token or API availability.")

# === Tab 2: GARCH ===
with tab2:
    st.header("GARCH: 7-Day Volatility Forecast")
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

        st.subheader("GARCH Volatility Forecast")
        for idx, row in forecast_df.iterrows():
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>{row['Date'].strftime('%Y-%m-%d')} ({row['Day']})</h4>
                    <p>Forecasted Volatility: {row['Forecasted Volatility (%)']}%</p>
                </div>
            """, unsafe_allow_html=True)

        avg_vol = forecast_df["Forecasted Volatility (%)"].mean()
        st.subheader("Trading Insight")
        if avg_vol > 20:
            st.markdown("<div class='alert-yellow'>Volatility is high (>20%). Consider defensive strategies like Iron Condor.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-green'>Volatility is moderate. You can explore strategies like Jade Lizard.</div>", unsafe_allow_html=True)

        rv_7d_df, hv_30d, hv_1y = calculate_rolling_and_fixed_hv(nifty_df["NIFTY_Close"])
        st.subheader("Historical Volatility")
        st.markdown(f"<div class='metric-card'><h4>30-Day HV (Annualized)</h4><p>{hv_30d}%</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>1-Year HV (Annualized)</h4><p>{hv_1y}%</p></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading GARCH data: {e}")

# === Tab 3: XGBoost ===
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
                features = ['ATM_IV', 'Realized_Vol', 'IVP', 'Event_Impact_Score', 'FII_DII_Net_Long', 'PCR']
                target = 'Next_5D_Realized_Vol'
                if not all(col in xgb_df.columns for col in features + [target]):
                    st.error("CSV missing required columns!")
                    st.stop()
                X = xgb_df[features]
                y = xgb_df[target] * 100

                response = requests.get(xgb_model_url)
                if response.status_code != 200:
                    st.error("Failed to load xgb_model.pkl from Google Drive.")
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
                    st.markdown(f"<div class='metric-card'><h4>RMSE</h4><p>{rmse_train:.4f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4>MAE</h4><p>{mae_train:.4f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4>R²</h4><p>{r2_train:.4f}</p></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<h3>Test Metrics</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4>RMSE</h4><p>{rmse_test:.4f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4>MAE</h4><p>{mae_test:.4f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-card'><h4>R²</h4><p>{r2_test:.4f}</p></div>", unsafe_allow_html=True)

                fig = go.Figure()
                importances = xgb_model.feature_importances_
                sorted_idx = np.argsort(importances)
                fig.add_trace(go.Bar(
                    y=np.array(features)[sorted_idx],
                    x=importances[sorted_idx],
                    orientation='h',
                    marker=dict(color='#e94560')
                ))
                fig.update_layout(
                    title="XGBoost Feature Importances",
                    xaxis_title="Feature Importance",
                    yaxis_title="Feature",
                    template="plotly_dark",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.subheader("Feature Importances")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error running model evaluation: {e}")

    st.subheader("Predict with New Data")
    st.info("Use VolGuard data (if available) or enter values manually.")
    
    try:
        nifty_df = pd.read_csv("https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv")
        nifty_df.columns = nifty_df.columns.str.strip()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date")
        nifty_df = nifty_df.rename(columns={"Close": "NIFTY_Close"})
        nifty_df = nifty_df[["NIFTY_Close"]].dropna().sort_index()
        realized_vol = compute_realized_vol(nifty_df)
    except Exception as e:
        realized_vol = 0
        st.warning(f"Could not compute Realized Volatility: {e}")

    atm_iv = st.session_state.volguard_data['atm_iv'] if st.session_state.volguard_data else 0
    pcr = st.session_state.volguard_data['pcr'] if st.session_state.volguard_data else 0

    col1, col2 = st.columns(2)
    with col1:
        atm_iv_input = st.number_input("ATM IV (%)", value=float(atm_iv), min_value=0.0, step=0.1)
        realized_vol_input = st.number_input("Realized Volatility (%)", value=float(realized_vol), min_value=0.0, step=0.1)
        ivp_input = st.number_input("IV Percentile (0–100)", value=50.0, min_value=0.0, max_value=100.0, step=1.0)
    with col2:
        event_score_input = st.number_input("Event Impact Score (0–2)", value=1.0, min_value=0.0, max_value=2.0, step=1.0)
        fii_dii_input = st.number_input("FII/DII Net Long (₹ Cr)", value=0.0, step=100.0)
        pcr_input = st.number_input("Put-Call Ratio", value=float(pcr), min_value=0.0, step=0.01)

    if st.button("Predict Volatility"):
        try:
            with st.spinner("Loading model and predicting..."):
                response = requests.get(xgb_model_url)
                if response.status_code != 200:
                    st.error("Failed to load xgb_model.pkl from Google Drive.")
                    st.stop()
                xgb_model = pickle.loads(response.content)

                new_data = pd.DataFrame({
                    'ATM_IV': [atm_iv_input],
                    'Realized_Vol': [realized_vol_input],
                    'IVP': [ivp_input],
                    'Event_Impact_Score': [event_score_input],
                    'FII_DII_Net_Long': [fii_dii_input],
                    'PCR': [pcr_input]
                })

                prediction = xgb_model.predict(new_data)[0]
                st.session_state.xgb_prediction = prediction
                st.markdown(f"<div class='metric-card'><h4>Predicted Next 7-Day Realized Volatility</h4><p>{prediction:.2f}%</p></div>", unsafe_allow_html=True)

                last_date = nifty_df.index[-1]
                xgb_forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
                xgb_forecast_df = pd.DataFrame({
                    "Date": xgb_forecast_dates,
                    "Day": xgb_forecast_dates.day_name(),
                    "Predicted Volatility (%)": np.round([prediction]*7, 2)
                })
                st.subheader("XGBoost Prediction Dates")
                for idx, row in xgb_forecast_df.iterrows():
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h4>{row['Date'].strftime('%Y-%m-%d')} ({row['Day']})</h4>
                            <p>Predicted Volatility: {row['Predicted Volatility (%)']}%</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.subheader("Trading Insight")
                if prediction > 20:
                    st.markdown("<div class='alert-yellow'>Predicted volatility is high (>20%). Consider defensive strategies like Iron Condor.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-green'>Predicted volatility is moderate. You can explore strategies like Jade Lizard.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error predicting volatility: {e}")

# === Tab 4: Strategies ===
with tab4:
    st.header("Strategy Recommendations")
    if run_engine or st.session_state.get('strategies', None):
        # Market Regime Classification (Simplified)
        iv_rv = st.session_state.atm_iv - realized_vol if st.session_state.atm_iv else 0
        if iv_rv > 10:
            regime = "High"
        elif iv_rv > 5:
            regime = "Medium"
        else:
            regime = "Low"

        # Strategy Recommendation Logic
        strategies = []
        if risk_profile == "Conservative":
            strategies.append({
                "name": "Iron Condor",
                "logic": "Neutral strategy for low volatility regime",
                "capital_required": total_capital * 0.3,
                "max_loss": total_capital * 0.03,
                "confidence": 0.8
            })
        elif risk_profile == "Moderate":
            strategies.append({
                "name": "Jade Lizard",
                "logic": "Skewed risk-reward for medium volatility",
                "capital_required": total_capital * 0.35,
                "max_loss": total_capital * 0.035,
                "confidence": 0.75
            })
        elif risk_profile == "Aggressive":
            strategies.append({
                "name": "Ratio Backspread",
                "logic": "High risk, high reward for high volatility regime",
                "capital_required": total_capital * 0.4,
                "max_loss": total_capital * 0.04,
                "confidence": 0.65
            })

        st.session_state.strategies = strategies
        for strategy in strategies:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>{strategy['name']}</h4>
                    <p>Logic: {strategy['logic']}</p>
                    <p>Capital Required: ₹{strategy['capital_required']:.2f}</p>
                    <p>Max Loss: ₹{strategy['max_loss']:.2f}</p>
                    <p>Confidence: {strategy['confidence']*100:.1f}%</p>
                    <p>Market Regime: {regime}</p>
                </div>
            """, unsafe_allow_html=True)
            # Risk Check Before Trade
            risk_status, risk_message = check_risk(strategy['capital_required'], strategy['max_loss'], 0)
            if risk_status == "red":
                st.markdown(f"<div class='alert-red'>{risk_message}</div>", unsafe_allow_html=True)
            else:
                if st.button(f"Trade Now - {strategy['name']}", key=strategy['name']):
                    # Dummy Trade Execution
                    dummy_pnl = np.random.uniform(-strategy['max_loss'], strategy['max_loss'] * 1.5)
                    st.session_state.deployed_capital += strategy['capital_required']
                    st.session_state.daily_pnl += dummy_pnl
                    st.session_state.trade_log.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "strategy": strategy['name'],
                        "capital_deployed": strategy['capital_required'],
                        "max_loss": strategy['max_loss'],
                        "pnl": dummy_pnl
                    })
                    st.success(f"Trade executed: {strategy['name']} | Capital Deployed: ₹{strategy['capital_required']:,} | P&L: ₹{dummy_pnl:,.2f}")

# === Tab 5: Dashboard ===
with tab5:
    st.header("Dashboard: Risk & Performance")
    st.subheader("Volatility Insights")
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
        xgb_vols = [xgb_vol] * 7

        atm_iv = st.session_state.atm_iv if st.session_state.atm_iv is not None else 20.0
        atm_iv_vols = [atm_iv] * 7
        pcr = st.session_state.volguard_data['pcr'] if st.session_state.volguard_data else 1.0
        straddle_price = st.session_state.volguard_data['straddle_price'] if st.session_state.volguard_data else 0
        max_pain = st.session_state.volguard_data['max_pain'] if st.session_state.volguard_data else 0
        iv_rv = atm_iv - realized_vol

        rv_vols = [realized_vol] * 7

        last_date = nifty_df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=7)
        plot_df = pd.DataFrame({
            "Date": forecast_dates,
            "GARCH Forecast": garch_vols,
            "XGBoost Prediction": xgb_vols,
            "Realized Volatility": rv_vols,
            "ATM IV": atm_iv_vols
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["GARCH Forecast"], mode='lines+markers', name='GARCH Forecast', line=dict(color='#e94560')))
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["XGBoost Prediction"], mode='lines+markers', name='XGBoost Prediction', line=dict(color='#ff6b6b')))
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["Realized Volatility"], mode='lines+markers', name='Realized Volatility', line=dict(color='#00adb5')))
        fig.add_trace(go.Scatter(x=plot_df["Date"], y=plot_df["ATM IV"], mode='lines+markers', name='ATM IV', line=dict(color='#f4e7ba')))
        fig.update_layout(
            title=f"Volatility Comparison ({forecast_dates[0].strftime('%b %d')}–{forecast_dates[-1].strftime('%b %d, %Y')})",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='highlight-card'><h4>ATM IV</h4><p>{atm_iv:.2f}%</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><h4>IV-RV</h4><p>{iv_rv:.2f}%</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h4>PCR</h4><p>{pcr:.2f}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><h4>Straddle Price</h4><p>{straddle_price:.2f}</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h4>Max Pain</h4><p>{max_pain:.2f}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><h4>Realized Volatility</h4><p>{realized_vol:.2f}%</p></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading volatility insights: {e}")
        st.write("Run VolGuard and XGBoost tabs first to populate data.")

    st.subheader("Risk Management Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h4>Total Capital</h4><p>₹{total_capital:,}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Deployed Capital</h4><p>₹{st.session_state.deployed_capital:,}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Exposure</h4><p>{exposure_pct:.1f}%</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>Max Exposure Allowed</h4><p>{MAX_EXPOSURE_PCT}% (₹{max_deployed_capital:,})</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Max Loss Per Trade</h4><p>{MAX_LOSS_PER_TRADE_PCT}% (₹{max_loss_per_trade:,})</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4>Daily Loss Limit</h4><p>{DAILY_LOSS_LIMIT_PCT}% (₹{daily_loss_limit:,})</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>Daily P&L</h4><p>₹{st.session_state.daily_pnl:,}</p></div>", unsafe_allow_html=True)

    st.subheader("Trade Log")
    if st.session_state.trade_log:
        for trade in st.session_state.trade_log:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4>{trade['date']}</h4>
                    <p>Strategy: {trade['strategy']}</p>
                    <p>Capital Deployed: ₹{trade['capital_deployed']:,.2f}</p>
                    <p>Max Loss: ₹{trade['max_loss']:,.2f}</p>
                    <p>P&L: ₹{trade['pnl']:,.2f}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trades executed yet.")

# === Journal Section ===
st.header("Journal: Trading Notes")
st.subheader("Add Your Thoughts")
journal_entry = st.text_area("Write your trading thoughts, strategy rationale, or reflections:")
if st.button("Save Journal Entry"):
    if journal_entry:
        st.session_state.journal_entries.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "entry": journal_entry
        })
        st.success("Journal entry saved!")
    else:
        st.warning("Please write something before saving.")

st.subheader("Your Journal Entries")
if st.session_state.journal_entries:
    for entry in st.session_state.journal_entries:
        st.markdown(f"""
            <div class='metric-card'>
                <h4>{entry['timestamp']}</h4>
                <p>{entry['entry']}</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("No journal entries yet. Start writing your thoughts!")
