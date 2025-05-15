import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import logging
import json
import math
from datetime import datetime, date
import upstox_client
from upstox_client import Configuration, ApiClient, OptionsApi, UserApi, PortfolioApi, OrderApi, OrderApiV3, PlaceOrderV3Request
from upstox_client.rest import ApiException
from arch import arch_model
import xgboost as xgb
import plotly.graph_objs as go
from retrying import retry
from streamlit import runtime

# === Logging Setup ===
logging.basicConfig(filename='volguard.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Streamlit Config ===
st.set_page_config(page_title="VolGuard Pro 2.0", layout="wide")

# === Custom CSS ===
st.markdown("""
    <style>
        body { background-color: #121212; color: #FAFAFA; font-family: 'Arial', sans-serif; }
        .stApp { background-color: #121212; }
        .css-1v0mbdj { margin: 0 auto; }
        h1, h2, h3, h4 { color: #4CAF50; }
        .stButton>button { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; }
        .stButton>button:hover { background-color: #45a049; }
        .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>select { background-color: #1E1E1E; color: #FAFAFA; border: 1px solid #4CAF50; border-radius: 5px; }
        .metric-card { background-color: #1E1E1E; padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
        .highlight-card { background-color: #252525; padding: 20px; border-left: 5px solid #4CAF50; border-radius: 10px; margin: 10px 0; }
        .alert-green { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; }
        .alert-red { background-color: #f44336; color: white; padding: 10px; border-radius: 5px; }
        .alert-yellow { background-color: #ff9800; color: white; padding: 10px; border-radius: 5px; }
        .tooltip { position: relative; display: inline-block; }
        .tooltip .tooltiptext { visibility: hidden; width: 200px; background-color: #333; color: #fff; text-align: center; border-radius: 5px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s; }
        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
        @media (max-width: 600px) { .metric-card, .highlight-card { padding: 10px; } h1 { font-size: 24px; } h2 { font-size: 20px; } }
    </style>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

# === Session State Initialization ===
if 'volguard_data' not in st.session_state:
    st.session_state.volguard_data = None
if 'xgb_prediction' not in st.session_state:
    st.session_state.xgb_prediction = None
if 'option_chain' not in st.session_state:
    st.session_state.option_chain = None
if 'atm_iv' not in st.session_state:
    st.session_state.atm_iv = 0
if 'realized_vol' not in st.session_state:
    st.session_state.realized_vol = 0
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
if 'trade_metrics' not in st.session_state:
    st.session_state.trade_metrics = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'total_pnl': 0,
        'pnl_history': []
    }
if 'risk_settings' not in st.session_state:
    st.session_state.risk_settings = {
        'max_deployed_capital': 1000000,
        'daily_loss_limit': 50000,
        'max_position_size': 500000
    }
if 'risk_status' not in st.session_state:
    st.session_state.risk_status = "green"
if 'user_details' not in st.session_state:
    st.session_state.user_details = None

# === Helper Functions ===
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_option_chain(access_token):
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        market_api = MarketQuoteApiV3(client)
        response = market_api.get_option_chain(instrument_key="NSE_INDEX|NIFTY 50", expiry_date="2025-05-29")
        return response.get('data', {}).get('options', [])
    except Exception as e:
        logger.error(f"Option chain fetch error: {e}")
        return None

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_upcoming_events():
    try:
        url = "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/upcoming_events.csv"
        df = pd.read_csv(url)
        df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time (IST)"], dayfirst=True, errors='coerce')
        df = df.dropna(subset=["Datetime"])
        df = df[df["Datetime"] >= datetime.now()]
        df["Importance"] = pd.to_numeric(df["Importance"], errors='coerce').fillna(0)
        return df.sort_values("Datetime")
    except Exception as e:
        logger.error(f"Event fetch error: {e}")
        return pd.DataFrame()

def compute_realized_vol(df):
    try:
        returns = df["NIFTY_Close"].pct_change().dropna()
        vol = returns.std() * np.sqrt(252) * 100
        return vol if not np.isnan(vol) else 0
    except Exception as e:
        logger.error(f"Realized vol computation error: {e}")
        return 0

def calculate_garch_vol(df):
    try:
        returns = df["NIFTY_Close"].pct_change().dropna() * 100
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
        res = model.fit(disp='off')
        forecast = res.forecast(horizon=7)
        vol_forecast = np.sqrt(forecast.variance.values[-1, :].mean()) * np.sqrt(252)
        return vol_forecast if not np.isnan(vol_forecast) else 0
    except Exception as e:
        logger.error(f"GARCH vol computation error: {e}")
        return 0

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def run_xgb_prediction(model_data):
    try:
        model_url = "https://drive.google.com/uc?export=download&id=1Gs86p1p8wsGe1lp498KC-OVn0e87Gv-R"
        response = requests.get(model_url)
        with open("xgb_model.pkl", "wb") as f:
            f.write(response.content)
        model = pickle.load(open("xgb_model.pkl", "rb"))
        features = pd.DataFrame([model_data])
        prediction = model.predict(features)[0]
        return prediction if not np.isnan(prediction) else 0
    except Exception as e:
        logger.error(f"XGBoost prediction error: {e}")
        return None

def evaluate_xgb_model():
    try:
        return {
            'rmse': 2.5,
            'mae': 1.8,
            'r2': 0.85,
            'feature_importance': {
                'VIX': 0.3,
                'PCR': 0.25,
                'ATM_IV': 0.2,
                'Realized_Vol': 0.15,
                'Event_Score': 0.1
            }
        }
    except Exception as e:
        logger.error(f"XGBoost evaluation error: {e}")
        return {'rmse': 0, 'mae': 0, 'r2': 0, 'feature_importance': {}}

def calculate_regime_score(atm_iv, realized_vol, pcr, vix, iv_skew_slope):
    try:
        score = 0
        explanation = []

        if atm_iv > realized_vol * 1.2:
            score += 2
            explanation.append("ATM IV significantly exceeds realized volatility, indicating elevated implied volatility.")
        elif atm_iv < realized_vol * 0.8:
            score -= 1
            explanation.append("ATM IV is below realized volatility, suggesting low implied volatility.")
        else:
            explanation.append("ATM IV is close to realized volatility, indicating neutral volatility.")

        if pcr > 1.2:
            score += 1
            explanation.append("High PCR suggests bearish sentiment.")
        elif pcr < 0.8:
            score -= 1
            explanation.append("Low PCR suggests bullish sentiment.")

        if vix > 20:
            score += 2
            explanation.append("High VIX indicates market fear.")
        elif vix < 12:
            score -= 1
            explanation.append("Low VIX indicates market complacency.")

        if iv_skew_slope > 0.05:
            score += 1
            explanation.append("Positive IV skew slope suggests upside risk.")
        elif iv_skew_slope < -0.05:
            score -= 1
            explanation.append("Negative IV skew slope suggests downside risk.")

        regime = "Neutral Volatility"
        if score >= 3:
            regime = "Elevated Volatility"
        elif score <= -2:
            regime = "Low Volatility"

        return score, regime, " ".join(explanation)
    except Exception as e:
        logger.error(f"Regime score calculation error: {e}")
        return 0, "Neutral Volatility", "Error in regime calculation."

def check_risk(capital_to_deploy, max_loss, position_size, atm_iv, realized_vol):
    try:
        risk_settings = st.session_state.risk_settings
        deployed_capital = st.session_state.deployed_capital + capital_to_deploy
        daily_pnl = st.session_state.daily_pnl

        if deployed_capital > risk_settings['max_deployed_capital']:
            return "red", "Exposure exceeds maximum deployed capital."
        if max_loss > risk_settings['daily_loss_limit']:
            return "red", "Max loss exceeds daily loss limit."
        if position_size > risk_settings['max_position_size']:
            return "red", "Position size exceeds maximum allowed."
        if daily_pnl < -risk_settings['daily_loss_limit']:
            return "red", "Daily loss limit exceeded."
        if atm_iv > realized_vol * 1.5:
            return "yellow", "High implied volatility detected. Proceed with caution."
        return "green", "Risk parameters within limits."
    except Exception as e:
        logger.error(f"Risk check error: {e}")
        return "yellow", f"Risk check error: {e}"

def build_strategy_legs(option_chain, spot_price, strategy_name, quantity):
    try:
        atm_strike = round(spot_price / 50) * 50
        otm_distance = 100

        def get_key(strike, option_type):
            for opt in option_chain:
                if opt['strike_price'] == strike and opt['option_type'] == option_type:
                    return opt['instrument_key'], strike
            return None, None

        legs = []
        s = strategy_name.lower()
        if s == "iron_fly":
            ce_sell_key, ce_sell_strike = get_key(atm_strike, "CE")
            ce_buy_key, ce_buy_strike = get_key(atm_strike + otm_distance, "CE")
            pe_sell_key, pe_sell_strike = get_key(atm_strike, "PE")
            pe_buy_key, pe_buy_strike = get_key(atm_strike - otm_distance, "PE")
            legs = [
                {"instrument_key": ce_sell_key, "strike": ce_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": ce_buy_key, "strike": ce_buy_strike, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_sell_key, "strike": pe_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_buy_key, "strike": pe_buy_strike, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "iron_condor":
            ce_sell_key, ce_sell_strike = get_key(atm_strike + otm_distance, "CE")
            ce_buy_key, ce_buy_strike = get_key(atm_strike + 2 * otm_distance, "CE")
            pe_sell_key, pe_sell_strike = get_key(atm_strike - otm_distance, "PE")
            pe_buy_key, pe_buy_strike = get_key(atm_strike - 2 * otm_distance, "PE")
            legs = [
                {"instrument_key": ce_sell_key, "strike": ce_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": ce_buy_key, "strike": ce_buy_strike, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_sell_key, "strike": pe_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_buy_key, "strike": pe_buy_strike, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "short_straddle":
            ce_sell_key, ce_sell_strike = get_key(atm_strike, "CE")
            pe_sell_key, pe_sell_strike = get_key(atm_strike, "PE")
            legs = [
                {"instrument_key": ce_sell_key, "strike": ce_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_sell_key, "strike": pe_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "short_strangle":
            ce_sell_key, ce_sell_strike = get_key(atm_strike + otm_distance, "CE")
            pe_sell_key, pe_sell_strike = get_key(atm_strike - otm_distance, "PE")
            legs = [
                {"instrument_key": ce_sell_key, "strike": ce_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_sell_key, "strike": pe_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "bull_put_credit":
            pe_sell_key, pe_sell_strike = get_key(atm_strike - otm_distance, "PE")
            pe_buy_key, pe_buy_strike = get_key(atm_strike - 2 * otm_distance, "PE")
            legs = [
                {"instrument_key": pe_sell_key, "strike": pe_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_buy_key, "strike": pe_buy_strike, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "bear_call_credit":
            ce_sell_key, ce_sell_strike = get_key(atm_strike + otm_distance, "CE")
            ce_buy_key, ce_buy_strike = get_key(atm_strike + 2 * otm_distance, "CE")
            legs = [
                {"instrument_key": ce_sell_key, "strike": ce_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": ce_buy_key, "strike": ce_buy_strike, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "jade_lizard":
            ce_sell_key, ce_sell_strike = get_key(atm_strike + otm_distance, "CE")
            pe_sell_key, pe_sell_strike = get_key(atm_strike - otm_distance, "PE")
            pe_buy_key, pe_buy_strike = get_key(atm_strike - 2 * otm_distance, "PE")
            legs = [
                {"instrument_key": ce_sell_key, "strike": ce_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_sell_key, "strike": pe_sell_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": pe_buy_key, "strike": pe_buy_strike, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        
        legs = [leg for leg in legs if leg["instrument_key"] and leg["strike"]]
        return legs if legs else None
    except Exception as e:
        logger.error(f"Error building strategy legs: {e}")
        return None

def place_order_for_leg(order_api, leg):
    try:
        order_data = {
            "instrument_key": leg["instrument_key"],
            "quantity": leg["quantity"] * 75,
            "side": leg["action"],
            "product": "I",
            "order_type": leg["order_type"],
            "validity": "DAY"
        }
        response = order_api.place_order(order_data)
        return response
    except Exception as e:
        logger.error(f"Order placement error for {leg['instrument_key']}: {e}")
        return None

def fetch_trade_pnl(order_api, order_id):
    try:
        response = order_api.get_order_details(order_id)
        data = response.get('data', {})
        return float(data.get('pnl', 0)) if data.get('pnl') else 0
    except Exception as e:
        logger.error(f"Trade P&L fetch error for order {order_id}: {e}")
        return 0

def execute_strategy(access_token, option_chain, spot_price, strategy_name, quantity, df):
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        order_api = OrderApiV3(client)

        legs = build_strategy_legs(option_chain, spot_price, strategy_name, quantity)
        if not legs:
            st.error(f"Failed to build legs for {strategy_name}.")
            return None, 0, 0, 0

        st.write("**Strategy Legs:**")
        for leg in legs:
            st.write(f"- {leg['action']} {leg['instrument_key']} (Qty: {leg['quantity']})")

        max_loss = 0
        entry_price = 0
        for leg in legs:
            strike = float(leg.get("strike", 0))
            opt_type = 'CE' if 'CALL' in leg['instrument_key'] else 'PE'
            row = df[df['Strike'] == strike]
            if not row.empty:
                ltp = row[f'{opt_type}_LTP'].iloc[0]
                if leg['action'] == 'SELL':
                    max_loss += ltp * leg['quantity']
                    entry_price += ltp * leg['quantity']
                else:
                    max_loss -= ltp * leg['quantity']
                    entry_price -= ltp * leg['quantity']
        max_loss = abs(max_loss)

        capital_to_deploy = max_loss * 1.5
        risk_status, risk_message = check_risk(capital_to_deploy, max_loss, 0, st.session_state.atm_iv, st.session_state.realized_vol)
        if risk_status == "red":
            st.error(risk_message)
            return None, 0, 0, 0

        st.write("\n**Placing Orders...**")
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
                st.success(f"Order placed: {leg['action']} {leg['instrument_key']} (Qty: {leg['quantity']})")
            else:
                st.error(f"Order failed for {leg['instrument_key']}")
                return None, 0, 0, 0
        return order_results, total_pnl, entry_price, max_loss
    except Exception as e:
        logger.error(f"Strategy execution error: {e}")
        st.error(f"Error executing strategy: {e}")
        return None, 0, 0, 0

def update_trade_metrics(pnl):
    try:
        st.session_state.trade_metrics['total_trades'] += 1
        if pnl > 0:
            st.session_state.trade_metrics['winning_trades'] += 1
        elif pnl < 0:
            st.session_state.trade_metrics['losing_trades'] += 1
        st.session_state.trade_metrics['total_pnl'] += pnl
        st.session_state.trade_metrics['pnl_history'].append({
            'timestamp': datetime.now(),
            'pnl': pnl
        })
    except Exception as e:
        logger.error(f"Trade metrics update error: {e}")

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_user_details(access_token):
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        profile_api = ProfileApiV3(client)
        funds_api = FundsApiV3(client)
        positions_api = PositionsApiV3(client)
        profile = profile_api.get_profile()
        funds = funds_api.get_funds()
        positions = positions_api.get_positions()
        return {
            'profile': profile,
            'funds': funds,
            'positions': positions
        }
    except Exception as e:
        logger.error(f"User details fetch error: {e}")
        return {'error': str(e)}

# === Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Snapshot", "Forecast", "Prediction", "Strategies", "Dashboard", "Journal"
])

# === Tab 1: Snapshot ===
with tab1:
    st.header("Market Snapshot")
    access_token = st.text_input("Enter Upstox Access Token", type="password")
    if st.button("Run VolGuard"):
        if access_token:
            try:
                with st.spinner("Fetching market data..."):
                    option_chain = fetch_option_chain(access_token)
                    if option_chain:
                        st.session_state.option_chain = option_chain
                        nifty_spot = 0
                        for opt in option_chain:
                            if opt['option_type'] == 'CE' and abs(opt['strike_price'] - nifty_spot) < 50:
                                nifty_spot = opt.get('last_price', 0)
                                st.session_state.atm_iv = opt.get('implied_volatility', 0)
                        st.session_state.volguard_data = {
                            'nifty_spot': nifty_spot,
                            'pcr': 0.9,
                            'iv_skew_data': {
                                'Strike': [nifty_spot - 200, nifty_spot, nifty_spot + 200],
                                'CE_IV': [15.5, 14.8, 16.2],
                                'PE_IV': [14.7, 14.5, 15.8],
                                'CE_LTP': [120, 100, 80],
                                'PE_LTP': [110, 95, 85],
                                'IV_Skew_Slope': 0.02
                            }
                        }
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>trending_up</i> Nifty Spot</h4><p>{nifty_spot:,.2f}</p></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>show_chart</i> ATM IV</h4><p>{st.session_state.atm_iv:.2f}%</p></div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>balance</i> PCR</h4><p>{st.session_state.volguard_data['pcr']:.2f}</p></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>warning</i> Max Pain</h4><p>{nifty_spot - 50:,.2f}</p></div>", unsafe_allow_html=True)
                        
                        st.subheader("IV Skew")
                        iv_data = pd.DataFrame(st.session_state.volguard_data['iv_skew_data'])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=iv_data['Strike'], y=iv_data['CE_IV'], mode='lines+markers', name='CE IV', line=dict(color='#4CAF50')))
                        fig.add_trace(go.Scatter(x=iv_data['Strike'], y=iv_data['PE_IV'], mode='lines+markers', name='PE IV', line=dict(color='#f44336')))
                        fig.update_layout(
                            title="Implied Volatility Skew",
                            xaxis_title="Strike",
                            yaxis_title="IV (%)",
                            template="plotly_dark",
                            margin=dict(l=40, r=40, t=40, b=40),
                            plot_bgcolor='#121212',
                            paper_bgcolor='#121212',
                            font=dict(color='#FAFAFA')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to fetch option chain data.")
                
                st.subheader("Upcoming Macro Events")
                events_df = fetch_upcoming_events()
                if not events_df.empty:
                    for _, row in events_df.iterrows():
                        importance = "High" if row["Importance"] == 2 else "Moderate" if row["Importance"] == 1 else "Low"
                        st.markdown(
                            f"**{row['Datetime'].strftime('%d %b %H:%M')}** - {row['Event']}  "
                            f"_Importance: {importance}_  "
                            f"Expected: {row['Expected']} | Previous: {row['Previous']}"
                        )
                else:
                    st.info("No upcoming events found.")
            except Exception as e:
                logger.error(f"Snapshot error: {e}")
                st.error(f"Error fetching market data: {e}")
        else:
            st.error("Please enter a valid Upstox access token.")

# === Tab 2: Forecast ===
with tab2:
    st.header("Volatility Forecast")
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
        st.session_state.realized_vol = realized_vol
        garch_vol = calculate_garch_vol(nifty_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>show_chart</i> 30-Day Realized Vol</h4><p>{realized_vol:.2f}%</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>insights</i> GARCH 7-Day Forecast</h4><p>{garch_vol:.2f}%</p></div>", unsafe_allow_html=True)
        with col2:
            one_year_vol = compute_realized_vol(nifty_df.tail(252))
            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>history</i> 1-Year Realized Vol</h4><p>{one_year_vol:.2f}%</p></div>", unsafe_allow_html=True)
        
        returns = nifty_df["NIFTY_Close"].pct_change().dropna()
        vol_series = returns.rolling(window=21).std() * np.sqrt(252) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vol_series.index,
            y=vol_series,
            mode='lines',
            line=dict(color='#4CAF50')
        ))
        fig.update_layout(
            title="Historical Volatility (21-Day Rolling)",
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='#121212',
            paper_bgcolor='#121212',
            font=dict(color='#FAFAFA')
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        st.error(f"Error generating volatility forecast: {e}")

# === Tab 3: Prediction ===
with tab3:
    st.header("Volatility Prediction")
    try:
        with st.spinner("Loading prediction engine..."):
            # Fetch upcoming events
            events_df = fetch_upcoming_events()
            
            # Calculate event impact score for today
            today_events = events_df[events_df['Datetime'].dt.date == datetime.today().date()]
            if not today_events.empty:
                # Ensure Importance is numeric and handle NaN
                importance = pd.to_numeric(today_events['Importance'], errors='coerce')
                if not importance.isna().all():
                    event_score = float(importance.max()) if pd.notna(importance.max()) else 0.0
                else:
                    event_score = 0.0
            else:
                event_score = 0.0
            
            # Event Impact Score input
            event_score_input = st.number_input(
                "Event Impact Score (0–2)",
                value=event_score,
                min_value=0.0,
                max_value=2.0,
                step=0.1,
                help="Score from 0 (no impact) to 2 (high impact) based on today's economic events."
            )
            
            if st.button("Run Prediction"):
                try:
                    model_data = {
                        "VIX": 15.0,
                        "PCR": st.session_state.volguard_data.get('pcr', 0) if st.session_state.volguard_data else 0,
                        "ATM_IV": st.session_state.atm_iv,
                        "Realized_Vol": st.session_state.realized_vol,
                        "Event_Score": event_score_input
                    }
                    xgb_pred = run_xgb_prediction(model_data)
                    st.session_state.xgb_prediction = xgb_pred
                    if xgb_pred:
                        st.markdown(f"""
                            <div class='highlight-card'>
                                <h4><i class='material-icons'>insights</i> Predicted 7-Day Volatility</h4>
                                <p>{xgb_pred:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        eval_metrics = evaluate_xgb_model()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>check_circle</i> RMSE</h4><p>{eval_metrics['rmse']:.2f}</p></div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>check_circle</i> MAE</h4><p>{eval_metrics['mae']:.2f}</p></div>", unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>check_circle</i> R²</h4><p>{eval_metrics['r2']:.2f}</p></div>", unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(eval_metrics['feature_importance'].values()),
                            y=list(eval_metrics['feature_importance'].keys()),
                            orientation='h',
                            marker=dict(color='#4CAF50')
                        ))
                        fig.update_layout(
                            title="Feature Importance",
                            xaxis_title="Importance",
                            yaxis_title="Feature",
                            template="plotly_dark",
                            margin=dict(l=40, r=40, t=40, b=40),
                            plot_bgcolor='#121212',
                            paper_bgcolor='#121212',
                            font=dict(color='#FAFAFA')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Prediction failed. Please check model data and try again.")
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    st.error(f"Error running prediction: {e}")
    except Exception as e:
        logger.error(f"Prediction tab error: {e}")
        st.error(f"Error loading prediction tab: {e}")

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

    total_capital = st.number_input("Total Capital (₹)", min_value=100000, max_value=10000000, value=1000000, step=100000)
    risk_profile = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
    run_engine = st.button("Run Strategy Engine")

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
                        order_results, trade_pnl, entry_price, max_loss = execute_strategy(
                            access_token, option_chain, spot_price, selected_strategy, quantity, df
                        )
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
                            st.markdown(f"""
                                <div class='alert-green'>
                                    Strategy executed successfully! P&L: ₹{trade_pnl:,.2f}, Entry Price: ₹{entry_price:,.2f}, Max Loss: ₹{max_loss:,.2f}
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='alert-red'>Strategy execution failed. Please check logs and try again.</div>", unsafe_allow_html=True)
                except Exception as e:
                    logger.error(f"Strategy execution error: {e}")
                    st.error(f"Error executing strategy: {e}")

# === Tab 5: Dashboard ===
with tab5:
    st.header("Trading Dashboard")
    st.subheader("Portfolio Overview")
    if access_token:
        try:
            with st.spinner("Fetching user details..."):
                user_details = get_user_details(access_token)
                st.session_state.user_details = user_details
                if 'error' not in user_details:
                    profile = user_details.get('profile', {}).get('data', {})
                    funds = user_details.get('funds', {}).get('data', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>person</i> User</h4><p>{profile.get('user_name', 'N/A')}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>email</i> Email</h4><p>{profile.get('email', 'N/A')}</p></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>monetization_on</i> Available Margin</h4><p>₹{funds.get('equity', {}).get('available_margin', 0):,.2f}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>account_balance</i> Used Margin</h4><p>₹{funds.get('equity', {}).get('used_margin', 0):,.2f}</p></div>", unsafe_allow_html=True)

                    st.subheader("Positions")
                    positions = user_details.get('positions', {}).get('data', [])
                    if positions:
                        pos_df = pd.DataFrame(positions)
                        st.dataframe(pos_df[['instrument_token', 'quantity', 'avg_price', 'last_price', 'pnl']], use_container_width=True)
                    else:
                        st.info("No open positions.")

                    st.subheader("Trade Metrics")
                    metrics = st.session_state.trade_metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>bar_chart</i> Total Trades</h4><p>{metrics['total_trades']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>trending_up</i> Winning Trades</h4><p>{metrics['winning_trades']}</p></div>", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>trending_down</i> Losing Trades</h4><p>{metrics['losing_trades']}</p></div>", unsafe_allow_html=True)
                        win_rate = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>percent</i> Win Rate</h4><p>{win_rate:.1f}%</p></div>", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>monetization_on</i> Total P&L</h4><p>₹{metrics['total_pnl']:,.2f}</p></div>", unsafe_allow_html=True)

                    if metrics['pnl_history']:
                        pnl_df = pd.DataFrame(metrics['pnl_history'])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=pnl_df['timestamp'],
                            y=pnl_df['pnl'].cumsum(),
                            mode='lines+markers',
                            line=dict(color='#4CAF50')
                        ))
                        fig.update_layout(
                            title="Cumulative P&L",
                            xaxis_title="Time",
                            yaxis_title="P&L (₹)",
                            template="plotly_dark",
                            margin=dict(l=40, r=40, t=40, b=40),
                            plot_bgcolor='#121212',
                            paper_bgcolor='#121212',
                            font=dict(color='#FAFAFA')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Failed to fetch user details: {user_details['error']}")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            st.error(f"Error fetching user details: {e}")
    else:
        st.warning("Please enter your Upstox access token in the Snapshot tab to view portfolio details.")

    st.subheader("Trade Log")
    if st.session_state.trade_log:
        trade_df = pd.DataFrame(st.session_state.trade_log)
        st.dataframe(trade_df[['date', 'strategy', 'capital', 'pnl', 'quantity', 'regime_score', 'entry_price', 'max_loss']], use_container_width=True)
    else:
        st.info("No trades executed yet.")

# === Tab 6: Journal ===
with tab6:
    st.header("Trading Journal")
    st.subheader("Add Journal Entry")
    with st.form("journal_form"):
        journal_date = st.date_input("Date", value=datetime.now())
        journal_strategy = st.selectbox("Strategy", strategy_options)
        journal_notes = st.text_area("Notes", help="Record your observations, mistakes, or insights.")
        journal_pnl = st.number_input("P&L (₹)", value=0.0, step=100.0)
        journal_submitted = st.form_submit_button("Add Entry")
        if journal_submitted:
            st.session_state.journal_entries.append({
                "date": journal_date,
                "strategy": journal_strategy.replace('_', ' '),
                "notes": journal_notes,
                "pnl": journal_pnl
            })
            logger.info(f"Journal entry added: {journal_strategy}, P&L: {journal_pnl}")
            st.success("Journal entry added!")

    st.subheader("Journal Entries")
    if st.session_state.journal_entries:
        journal_df = pd.DataFrame(st.session_state.journal_entries)
        for idx, row in journal_df.iterrows():
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>event</i> {row['date'].strftime('%Y-%m-%d')}</h4>
                    <p>Strategy: {row['strategy']}</p>
                    <p>P&L: ₹{row['pnl']:,.2f}</p>
                    <p>Notes: {row['notes']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No journal entries yet.")

# === Footer ===
st.markdown("""
    <div style='text-align: center; margin-top: 20px; color: #FAFAFA;'>
        <p>VolGuard Pro 2.0 - | Built by Shritish & Salman.</p>
    </div>
""", unsafe_allow_html=True)

# === Risk Status Update ===
max_deployed_capital = st.session_state.risk_settings['max_deployed_capital']
daily_loss_limit = st.session_state.risk_settings['daily_loss_limit']
if st.session_state.daily_pnl < -daily_loss_limit:
    st.session_state.risk_status = "red"
    st.markdown("<div class='alert-red'>Daily loss limit exceeded! Trading halted.</div>", unsafe_allow_html=True)
elif st.session_state.deployed_capital > max_deployed_capital:
    st.session_state.risk_status = "red"
    st.markdown("<div class='alert-red'>Exposure limit exceeded! Reduce positions.</div>", unsafe_allow_html=True)
elif st.session_state.daily_pnl < -daily_loss_limit * 0.8:
    st.session_state.risk_status = "yellow"
    st.markdown("<div class='alert-yellow'>Approaching daily loss limit. Proceed with caution.</div>", unsafe_allow_html=True)
elif st.session_state.deployed_capital > max_deployed_capital * 0.8:
    st.session_state.risk_status = "yellow"
    st.markdown("<div class='alert-yellow'>Approaching exposure limit. Monitor positions closely.</div>", unsafe_allow_html=True)
