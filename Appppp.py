# main.py (Final Updated Version)
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
import plotly.graph_objects as go

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress XGBoost warnings
import xgboost as xgb
xgb.set_config(verbosity=0)

# === Streamlit Configuration ===
st.set_page_config(page_title="VolGuard Pro", layout="wide")

# === Custom CSS for Premium Look ===
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

        body {
            font-family: 'Montserrat', sans-serif;
            background: linear-gradient(135deg, #1E3A8A 0%, #111827 100%);
            color: #F3F4F6;
        }

        .stApp {
            background: transparent;
        }

        .top-bar {
            display: flex;
            justify-content: space-around;
            background: linear-gradient(90deg, #1E3A8A, #3B82F6);
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
            animation: slideIn 0.5s ease-in-out;
        }

        .top-bar div {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .top-bar p {
            margin: 0;
            font-size: 16px;
            font-weight: 600;
            color: #FBBF24;
        }

        .metric-card {
            background: #1F2937;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        .metric-card h4 {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0 0 10px 0;
            font-size: 18px;
            color: #FBBF24;
        }

        .metric-card p {
            margin: 5px 0;
            font-size: 14px;
            color: #D1D5DB;
        }

        .alert-red {
            background: #EF4444;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }

        .alert-yellow {
            background: #F59E0B;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }

        .stButton>button {
            background: linear-gradient(90deg, #3B82F6, #1E3A8A);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: 600;
            transition: background 0.3s ease;
        }

        .stButton>button:hover {
            background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        }

        h1, h2, h3, h4, h5, h6 {
            color: #FBBF24;
            font-weight: 700;
        }

        .sidebar .sidebar-content {
            background: #111827;
            color: #F3F4F6;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-20px); }
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
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'holdings' not in st.session_state:
    st.session_state.holdings = []
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'funds_margin' not in st.session_state:
    st.session_state.funds_margin = {}
if 'live_pnl' not in st.session_state:
    st.session_state.live_pnl = 0
if 'pnl_data' not in st.session_state:
    st.session_state.pnl_data = []

# Initialize prev_oi globally
prev_oi = {}

# === Logo and Tagline ===
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <img src='https://via.placeholder.com/150x50.png?text=VolGuard+Logo' alt='VolGuard Logo' style='height: 50px;'>
        <h1>VolGuard Pro</h1>
        <p style='color: #D1D5DB; font-style: italic; font-size: 16px;'>Shield Your Trades, Amplify Your Gains</p>
    </div>
""", unsafe_allow_html=True)

# === Sidebar Controls ===
st.sidebar.header("VolGuard Pro Controls")
total_capital = st.sidebar.slider("Total Capital (₹)", 100000, 5000000, 1000000, 10000)
risk_profile = st.sidebar.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"])
run_engine = st.sidebar.button("Run Engine")

# === Risk Management Rules ===
MAX_EXPOSURE_PCT = 40
MAX_LOSS_PER_TRADE_PCT = 4
DAILY_LOSS_LIMIT_PCT = 4
max_loss_per_trade = total_capital * (MAX_LOSS_PER_TRADE_PCT / 100)
daily_loss_limit = total_capital * (DAILY_LOSS_LIMIT_PCT / 100)
max_deployed_capital = total_capital * (MAX_EXPOSURE_PCT / 100)

# === Top Bar (Quick Stats) ===
exposure_pct = (st.session_state.deployed_capital / total_capital) * 100 if total_capital > 0 else 0
st.markdown(f"""
    <div class='top-bar'>
        <div><i class="material-icons">account_balance_wallet</i><p>Total Capital: ₹{total_capital:,}</p></div>
        <div><i class="material-icons">trending_up</i><p>Deployed Capital: ₹{st.session_state.deployed_capital:,}</p></div>
        <div><i class="material-icons">percent</i><p>Exposure: {exposure_pct:.1f}%</p></div>
        <div><i class="material-icons">monetization_on</i><p>Live P&L: ₹{st.session_state.live_pnl:,}</p></div>
    </div>
""", unsafe_allow_html=True)

# === Risk Manager ===
def check_risk(capital_to_deploy, max_loss, daily_pnl):
    new_deployed_capital = st.session_state.deployed_capital + capital_to_deploy
    new_exposure_pct = (new_deployed_capital / total_capital) * 100 if total_capital > 0 else 0
    new_daily_pnl = daily_pnl + st.session_state.live_pnl

    if new_exposure_pct > MAX_EXPOSURE_PCT:
        return "red", f"Exposure exceeds {MAX_EXPOSURE_PCT}%! Cannot deploy ₹{capital_to_deploy:,}."
    if max_loss > max_loss_per_trade:
        return "red", f"Max loss per trade exceeds ₹{max_loss_per_trade:,} (4% of capital)!"
    if new_daily_pnl < -daily_loss_limit:
        return "red", f"Daily loss limit exceeded! Max loss allowed today: ₹{daily_loss_limit:,}."
    if new_exposure_pct > 30:
        return "yellow", "Exposure > 30%. Proceed with caution."
    return "green", "Safe to trade."

# === Helper Functions ===
def fetch_historical_data():
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = st.session_state.access_token
        client = upstox_client.ApiClient(configuration)
        history_api = upstox_client.HistoryV3Api(client)
        
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        response = history_api.get_historical_candle_data1(
            instrument_key="NSE_INDEX|Nifty 50",
            unit="day",
            interval="1d",
            to_date=to_date,
            from_date=from_date
        )
        data = response.to_dict().get('data', [])
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['returns'] = df['close'].pct_change().dropna()
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def calculate_realized_vol(df, window=30):
    if df.empty:
        return 0
    returns = df['returns'].dropna()
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return rolling_vol[-1] if not rolling_vol.empty else 0

def get_nearest_expiry():
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = st.session_state.access_token
        client = upstox_client.ApiClient(configuration)
        options_api = upstox_client.OptionsApi(client)
        
        response = options_api.get_option_contracts(
            instrument_key="NSE_INDEX|Nifty 50",
            option_type="XX"
        )
        contracts = response.to_dict().get('data', [])
        
        today = datetime.now().date()
        expiries = [datetime.strptime(contract.get('expiry_date'), "%Y-%m-%d").date() for contract in contracts]
        future_expiries = [exp for exp in expiries if exp >= today]
        if not future_expiries:
            raise ValueError("No future expiries found.")
        
        nearest_expiry = min(future_expiries)
        return nearest_expiry.strftime("%Y-%m-%d")
    except Exception as e:
        logger.error(f"Error fetching nearest expiry: {e}")
        return (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

def fetch_option_chain(access_token, expiry_date):
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        options_api = upstox_client.OptionsApi(client)
        
        response = options_api.get_put_call_option_chain(
            instrument_key="NSE_INDEX|Nifty 50",
            expiry_date=expiry_date
        )
        option_data = response.to_dict().get('data', [])
        
        market_quote_api = upstox_client.MarketQuoteV3Api(client)
        spot_response = market_quote_api.get_ltp(instrument_key="NSE_INDEX|Nifty 50")
        spot_price = spot_response.to_dict().get('data', {}).get('NSE_INDEX|Nifty 50', {}).get('last_price', 0)
        
        if not option_data or not spot_price:
            raise ValueError("Failed to fetch option chain or spot price.")
        
        return option_data, spot_price
    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return [], 0

def calculate_iv_skew(option_data, spot_price):
    if not option_data:
        return None, 0, 0
    
    strikes = []
    ce_ltp = []
    pe_ltp = []
    ce_iv = []
    pe_iv = []
    ce_token = []
    pe_token = []
    
    for item in option_data:
        strike = item.get('strike_price', 0)
        call_data = item.get('call_option', {})
        put_data = item.get('put_option', {})
        
        strikes.append(strike)
        ce_ltp.append(call_data.get('last_price', 0))
        pe_ltp.append(put_data.get('last_price', 0))
        ce_iv.append(call_data.get('iv', 0))
        pe_iv.append(put_data.get('iv', 0))
        ce_token.append(call_data.get('instrument_key', ''))
        pe_token.append(put_data.get('instrument_key', ''))
    
    df = pd.DataFrame({
        'Strike': strikes,
        'CE_LTP': ce_ltp,
        'PE_LTP': pe_ltp,
        'CE_IV': ce_iv,
        'PE_IV': pe_iv,
        'CE_Token': ce_token,
        'PE_Token': pe_token
    })
    
    atm_strike = df.iloc[(df['Strike'] - spot_price).abs().argsort()[:1]]['Strike'].values[0]
    atm_row = df[df['Strike'] == atm_strike]
    atm_iv = (atm_row['CE_IV'].values[0] + atm_row['PE_IV'].values[0]) / 2 if atm_row['CE_IV'].values[0] and atm_row['PE_IV'].values[0] else 0
    
    df['IV_Skew'] = df['CE_IV'] - df['PE_IV']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Strike'], y=df['CE_IV'], mode='lines+markers', name='Call IV', line=dict(color='#FBBF24')))
    fig.add_trace(go.Scatter(x=df['Strike'], y=df['PE_IV'], mode='lines+markers', name='Put IV', line=dict(color='#3B82F6')))
    fig.add_trace(go.Scatter(x=df['Strike'], y=df['IV_Skew'], mode='lines', name='IV Skew', line=dict(color='#EF4444')))
    fig.update_layout(
        title='IV Skew Across Strikes',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        template='plotly_dark',
        title_font=dict(size=20, color='#FBBF24'),
        xaxis=dict(title_font=dict(color='#D1D5DB'), tickfont=dict(color='#D1D5DB')),
        yaxis=dict(title_font=dict(color='#D1D5DB'), tickfont=dict(color='#D1D5DB')),
    )
    
    return df, atm_strike, atm_iv, fig

def run_volguard(access_token):
    try:
        expiry_date = get_nearest_expiry()
        if not expiry_date:
            raise ValueError("Could not determine expiry date.")
        
        option_data, spot_price = fetch_option_chain(access_token, expiry_date)
        if not option_data or not spot_price:
            raise ValueError("Failed to fetch option chain or spot price.")
        
        df, atm_strike, atm_iv, iv_skew_fig = calculate_iv_skew(option_data, spot_price)
        if df.empty:
            raise ValueError("Failed to process option chain data.")
        
        result = {
            'nifty_spot': spot_price,
            'atm_strike': atm_strike,
            'iv_skew_data': df.to_dict(),
            'expiry_date': expiry_date
        }
        
        return result, df, iv_skew_fig, atm_strike, atm_iv
    except Exception as e:
        logger.error(f"Error in run_volguard: {e}")
        st.error(f"Failed to run VolGuard: {str(e)}")
        return None, pd.DataFrame(), None, 0, 0

def fetch_portfolio(access_token):
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        portfolio_api = upstox_client.PortfolioApi(client)
        
        holdings = portfolio_api.get_holdings()
        holdings_data = holdings.to_dict().get('data', [])
        
        positions = portfolio_api.get_positions()
        positions_data = positions.to_dict().get('data', [])
        
        return holdings_data, positions_data
    except ApiException as e:
        logger.error(f"Portfolio fetch error: {e}")
        return [], []

def fetch_funds_and_margin(access_token):
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        user_api = upstox_client.UserApi(client)
        
        response = user_api.get_user_fund_margin()
        fund_margin = response.to_dict().get('data', {})
        return fund_margin
    except ApiException as e:
        logger.error(f"Funds and margin fetch error: {e}")
        return {}

def fetch_live_pnl(access_token, from_date, to_date):
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        pnl_api = upstox_client.TradeProfitAndLossApi(client)
        
        response = pnl_api.get_trade_wise_profit_and_loss_data(
            segment="EQ",
            financial_year="2425",
            from_date=from_date,
            to_date=to_date
        )
        pnl_data = response.to_dict().get('data', [])
        total_pnl = sum(trade.get('pnl', 0) for trade in pnl_data)
        return total_pnl, pnl_data
    except ApiException as e:
        logger.error(f"P&L fetch error: {e}")
        return 0, []

def calculate_iron_fly(df, spot, atm_strike, capital, risk_profile):
    try:
        atm_row = df[df['Strike'] == atm_strike]
        if atm_row.empty:
            raise ValueError("ATM strike not found in option chain.")
        
        short_call_strike = atm_strike
        short_put_strike = atm_strike
        short_call_price = atm_row['CE_LTP'].values[0]
        short_put_price = atm_row['PE_LTP'].values[0]
        
        strike_diff = spot * 0.05
        long_call_strike = atm_strike + strike_diff
        long_put_strike = atm_strike - strike_diff
        
        long_call_row = df.iloc[(df['Strike'] - long_call_strike).abs().argsort()[:1]]
        long_put_row = df.iloc[(df['Strike'] - long_put_strike).abs().argsort()[:1]]
        if long_call_row.empty or long_put_row.empty:
            raise ValueError("Could not find OTM strikes for wings.")
        
        long_call_strike = long_call_row['Strike'].values[0]
        long_put_strike = long_put_row['Strike'].values[0]
        long_call_price = long_call_row['CE_LTP'].values[0]
        long_put_price = long_put_row['PE_LTP'].values[0]
        
        lot_size = 25
        position_size = int((capital * 0.3) / (short_call_price + short_put_price)) // lot_size
        quantity = position_size * lot_size
        
        net_credit = (short_call_price + short_put_price - long_call_price - long_put_price) * quantity
        max_profit = net_credit
        
        wing_width = long_call_strike - short_call_strike
        max_loss = (wing_width * quantity) - net_credit
        
        breakeven_upper = short_call_strike + (net_credit / quantity)
        breakeven_lower = short_put_strike - (net_credit / quantity)
        
        iv_rv = st.session_state.atm_iv - realized_vol if st.session_state.atm_iv else 0
        confidence = 0.8 if iv_rv > 10 else 0.6
        if risk_profile == "Conservative":
            confidence *= 0.5
        elif risk_profile == "Moderate":
            confidence *= 0.8
        
        reasoning = f"""
        - **Market Regime**: IV-RV = {iv_rv:.2f}% (High IV, suitable for Iron Fly).
        - **Short Strikes**: ATM Call and Put at {atm_strike} (Spot: {spot}).
        - **Long Strikes**: OTM Call at {long_call_strike}, OTM Put at {long_put_strike} (5% away).
        - **Position Size**: {quantity} units ({position_size} lots).
        """
        
        return {
            "name": "Iron Fly",
            "logic": "Neutral strategy for high IV regime",
            "short_call_strike": short_call_strike,
            "short_put_strike": short_put_strike,
            "long_call_strike": long_call_strike,
            "long_put_strike": long_put_strike,
            "short_call_price": short_call_price,
            "short_put_price": short_put_price,
            "long_call_price": long_call_price,
            "long_put_price": long_put_price,
            "quantity": quantity,
            "capital_required": capital * 0.3,
            "max_loss": max_loss,
            "max_profit": max_profit,
            "breakeven_upper": breakeven_upper,
            "breakeven_lower": breakeven_lower,
            "confidence": confidence,
            "reasoning": reasoning
        }
    except Exception as e:
        logger.error(f"Error in calculate_iron_fly: {e}")
        return {
            "name": "Iron Fly",
            "logic": "Neutral strategy for high IV regime",
            "capital_required": capital * 0.3,
            "max_loss": capital * 0.05,
            "confidence": 0.5,
            "reasoning": f"Failed to calculate details: {str(e)}"
        }

def place_iron_fly_orders(access_token, strategy, df):
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        order_api = upstox_client.OrderApiV3(client)
        
        atm_row = df[df['Strike'] == strategy['short_call_strike']]
        long_call_row = df[df['Strike'] == strategy['long_call_strike']]
        long_put_row = df[df['Strike'] == strategy['long_put_strike']]
        
        short_call_token = atm_row['CE_Token'].values[0]
        short_put_token = atm_row['PE_Token'].values[0]
        long_call_token = long_call_row['CE_Token'].values[0]
        long_put_token = long_put_row['PE_Token'].values[0]
        
        quantity = strategy['quantity']
        
        orders = [
            upstox_client.PlaceOrderV3Request(
                quantity=quantity,
                product="D",
                validity="DAY",
                price=strategy['short_call_price'],
                tag="VolGuard_IronFly",
                instrument_token=short_call_token,
                order_type="LIMIT",
                transaction_type="SELL",
                disclosed_quantity=0,
                trigger_price=0.0,
                is_amo=False,
                slice=False
            ),
            upstox_client.PlaceOrderV3Request(
                quantity=quantity,
                product="D",
                validity="DAY",
                price=strategy['short_put_price'],
                tag="VolGuard_IronFly",
                instrument_token=short_put_token,
                order_type="LIMIT",
                transaction_type="SELL",
                disclosed_quantity=0,
                trigger_price=0.0,
                is_amo=False,
                slice=False
            ),
            upstox_client.PlaceOrderV3Request(
                quantity=quantity,
                product="D",
                validity="DAY",
                price=strategy['long_call_price'],
                tag="VolGuard_IronFly",
                instrument_token=long_call_token,
                order_type="LIMIT",
                transaction_type="BUY",
                disclosed_quantity=0,
                trigger_price=0.0,
                is_amo=False,
                slice=False
            ),
            upstox_client.PlaceOrderV3Request(
                quantity=quantity,
                product="D",
                validity="DAY",
                price=strategy['long_put_price'],
                tag="VolGuard_IronFly",
                instrument_token=long_put_token,
                order_type="LIMIT",
                transaction_type="BUY",
                disclosed_quantity=0,
                trigger_price=0.0,
                is_amo=False,
                slice=False
            )
        ]
        
        order_responses = []
        for order in orders:
            response = order_api.place_order(order)
            order_responses.append(response.to_dict())
            time.sleep(0.5)
        
        net_credit = strategy['max_profit']
        
        st.session_state.trade_log.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy['name'],
            "capital_deployed": strategy['capital_required'],
            "max_loss": strategy['max_loss'],
            "pnl": net_credit,
            "order_details": order_responses
        })
        
        st.session_state.deployed_capital += strategy['capital_required']
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
        live_pnl, pnl_data = fetch_live_pnl(access_token, from_date, to_date)
        st.session_state.live_pnl = live_pnl
        st.session_state.pnl_data = pnl_data
        
        return True, "Trade executed successfully!"
    except ApiException as e:
        logger.error(f"Order placement error: {e}")
        return False, f"Failed to place orders: {e}"

# === Streamlit Tabs ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Snapshot", "Forecast", "Prediction", "Strategies", "Dashboard"])

# === Tab 1: VolGuard ===
with tab1:
    st.header("Market Snapshot")
    access_token = st.text_input("Enter Upstox Access Token", type="password", help="Enter your Upstox access token to fetch live market data.")
    
    if st.button("Run VolGuard"):
        if not access_token:
            st.error("Please enter a valid Upstox access token.")
        else:
            with st.spinner("Fetching options data... Please wait."):
                st.session_state.access_token = access_token
                result, df, iv_skew_fig, atm_strike, atm_iv = run_volguard(access_token)
                if result:
                    st.session_state.volguard_data = result
                    st.session_state.atm_iv = atm_iv
                    
                    holdings, positions = fetch_portfolio(access_token)
                    funds_margin = fetch_funds_and_margin(access_token)
                    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                    to_date = datetime.now().strftime("%Y-%m-%d")
                    live_pnl, pnl_data = fetch_live_pnl(access_token, from_date, to_date)
                    
                    st.session_state.holdings = holdings
                    st.session_state.positions = positions
                    st.session_state.funds_margin = funds_margin
                    st.session_state.live_pnl = live_pnl
                    st.session_state.pnl_data = pnl_data
                    st.session_state.deployed_capital = sum(pos.get('quantity', 0) * pos.get('last_price', 0) for pos in positions)
                    st.session_state.daily_pnl = live_pnl
                    
                    st.success("Data fetched successfully!")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.plotly_chart(iv_skew_fig, use_container_width=True)
                    with col2:
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>trending_up</i> NIFTY Spot</h4><p>₹{result['nifty_spot']}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>strike</i> ATM Strike</h4><p>{atm_strike}</p></div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>volatility</i> ATM IV</h4><p>{atm_iv:.2f}%</p></div>", unsafe_allow_html=True)

# === Tab 2: Forecast ===
with tab2:
    st.header("Volatility Forecast")
    historical_data = fetch_historical_data()
    realized_vol = calculate_realized_vol(historical_data)
    if not historical_data.empty:
        returns = historical_data['returns'].dropna() * 100
        model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_fit = model.fit(disp='off')
        forecast = garch_fit.forecast(horizon=5)
        forecast_vol = np.sqrt(forecast.variance.values[-1, :]) * np.sqrt(252)
        
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>volatility</i> Realized Volatility (30D)</h4><p>{realized_vol:.2f}%</p></div>", unsafe_allow_html=True)
        st.subheader("5-Day Volatility Forecast")
        forecast_df = pd.DataFrame({
            'Day': [f"Day {i+1}" for i in range(5)],
            'Forecasted Volatility (%)': forecast_vol
        })
        st.dataframe(forecast_df, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df['Day'], y=forecast_df['Forecasted Volatility (%)'], mode='lines+markers', line=dict(color='#FBBF24')))
        fig.update_layout(
            title='5-Day Volatility Forecast',
            xaxis_title='Day',
            yaxis_title='Volatility (%)',
            template='plotly_dark',
            title_font=dict(size=20, color='#FBBF24'),
            xaxis=dict(title_font=dict(color='#D1D5DB'), tickfont=dict(color='#D1D5DB')),
            yaxis=dict(title_font=dict(color='#D1D5DB'), tickfont=dict(color='#D1D5DB')),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to fetch historical data for forecasting.")

# === Tab 3: Prediction ===
with tab3:
    st.header("NIFTY Prediction")
    if not historical_data.empty:
        data = historical_data[['close']].copy()
        data['lag1'] = data['close'].shift(1)
        data['lag2'] = data['close'].shift(2)
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['volatility'] = data['close'].pct_change().rolling(window=5).std() * np.sqrt(252)
        data = data.dropna()
        
        X = data[['lag1', 'lag2', 'ma5', 'volatility']]
        y = data['close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        y_pred = xgb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        latest_data = X.iloc[-1:].copy()
        prediction = xgb_model.predict(latest_data)[0]
        st.session_state.xgb_prediction = prediction
        
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>prediction</i> Predicted NIFTY Close</h4><p>₹{prediction:.2f}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>error</i> Model MSE</h4><p>{mse:.2f}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>error</i> Model MAE</h4><p>{mae:.2f}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>score</i> Model R²</h4><p>{r2:.2f}</p></div>", unsafe_allow_html=True)
    else:
        st.error("Failed to fetch historical data for prediction.")

# === Tab 4: Strategies ===
with tab4:
    st.header("Strategy Recommendations")
    if run_engine or st.session_state.get('strategies', None):
        iv_rv = st.session_state.atm_iv - realized_vol if st.session_state.atm_iv else 0
        if iv_rv > 10:
            regime = "High"
        elif iv_rv > 5:
            regime = "Medium"
        else:
            regime = "Low"

        strategies = []
        df = pd.DataFrame(st.session_state.volguard_data['iv_skew_data']) if st.session_state.volguard_data else pd.DataFrame()
        spot = st.session_state.volguard_data['nifty_spot'] if st.session_state.volguard_data else 0
        atm_strike = st.session_state.volguard_data['atm_strike'] if st.session_state.volguard_data else 0
        
        if not df.empty:
            if regime == "High" and risk_profile in ["Moderate", "Aggressive"]:
                iron_fly = calculate_iron_fly(df, spot, atm_strike, total_capital, risk_profile)
                strategies.append(iron_fly)
            elif risk_profile == "Conservative":
                strategies.append({
                    "name": "Iron Condor",
                    "logic": "Neutral strategy for low volatility regime",
                    "capital_required": total_capital * 0.3,
                    "max_loss": total_capital * 0.03,
                    "confidence": 0.8,
                    "reasoning": "Conservative profile prefers low-risk strategies."
                })
            elif risk_profile == "Moderate":
                strategies.append({
                    "name": "Jade Lizard",
                    "logic": "Skewed risk-reward for medium volatility",
                    "capital_required": total_capital * 0.35,
                    "max_loss": total_capital * 0.035,
                    "confidence": 0.75,
                    "reasoning": "Moderate profile suits Jade Lizard in medium volatility."
                })
            elif risk_profile == "Aggressive":
                strategies.append({
                    "name": "Ratio Backspread",
                    "logic": "High risk, high reward for high volatility regime",
                    "capital_required": total_capital * 0.4,
                    "max_loss": total_capital * 0.04,
                    "confidence": 0.65,
                    "reasoning": "Aggressive profile suits high-risk strategies."
                })

        st.session_state.strategies = strategies
        for strategy in strategies:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>strategy</i> {strategy['name']}</h4>
                    <p>Logic: {strategy['logic']}</p>
                    <p>Capital Required: ₹{strategy['capital_required']:.2f}</p>
                    <p>Max Loss: ₹{strategy['max_loss']:.2f}</p>
                    <p>Max Profit: ₹{strategy.get('max_profit', 'N/A'):.2f}</p>
                    <p>Breakeven Points: {strategy.get('breakeven_lower', 'N/A'):.2f} - {strategy.get('breakeven_upper', 'N/A'):.2f}</p>
                    <p>Confidence: {strategy['confidence']*100:.1f}%</p>
                    <p>Market Regime: {regime}</p>
                    <p>Reasoning: {strategy['reasoning']}</p>
                </div>
            """, unsafe_allow_html=True)
            risk_status, risk_message = check_risk(strategy['capital_required'], strategy['max_loss'], 0)
            if risk_status == "red":
                st.markdown(f"<div class='alert-red'>{risk_message}</div>", unsafe_allow_html=True)
            else:
                if st.button(f"Trade Now - {strategy['name']}", key=strategy['name']):
                    if not st.session_state.access_token:
                        st.error("Please run VolGuard with a valid access token to place trades.")
                    else:
                        success, message = place_iron_fly_orders(st.session_state.access_token, strategy, df)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

# === Tab 5: Dashboard ===
with tab5:
    st.header("Dashboard: Risk & Performance")
    st.subheader("Volatility Insights")
    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>volatility</i> Realized Volatility (30D)</h4><p>{realized_vol:.2f}%</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>volatility</i> ATM Implied Volatility</h4><p>{st.session_state.atm_iv if st.session_state.atm_iv else 0:.2f}%</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>difference</i> IV-RV Spread</h4><p>{(st.session_state.atm_iv - realized_vol) if st.session_state.atm_iv else 0:.2f}%</p></div>", unsafe_allow_html=True)

    st.subheader("Portfolio Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>account_balance_wallet</i> Total Funds</h4><p>₹{st.session_state.funds_margin.get('equity', {}).get('available_margin', 0):,}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>trending_up</i> Deployed Capital</h4><p>₹{st.session_state.deployed_capital:,}</p></div>", unsafe_allow_html=True)
        exposure_pct = (st.session_state.deployed_capital / total_capital) * 100 if total_capital > 0 else 0
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>percent</i> Exposure</h4><p>{exposure_pct:.1f}%</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>security</i> Max Exposure Allowed</h4><p>{MAX_EXPOSURE_PCT}% (₹{max_deployed_capital:,})</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>warning</i> Max Loss Per Trade</h4><p>{MAX_LOSS_PER_TRADE_PCT}% (₹{max_loss_per_trade:,})</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>warning</i> Daily Loss Limit</h4><p>{DAILY_LOSS_LIMIT_PCT}% (₹{daily_loss_limit:,})</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4><i class='material-icons'>monetization_on</i> Live P&L</h4><p>₹{st.session_state.live_pnl:,}</p></div>", unsafe_allow_html=True)

    st.subheader("Holdings")
    if st.session_state.holdings:
        for holding in st.session_state.holdings:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>store</i> {holding.get('tradingsymbol', 'N/A')}</h4>
                    <p>Quantity: {holding.get('quantity', 0)}</p>
                    <p>Avg Price: ₹{holding.get('average_price', 0):.2f}</p>
                    <p>Current Value: ₹{holding.get('last_price', 0) * holding.get('quantity', 0):,.2f}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No holdings data available.")

    st.subheader("Positions")
    if st.session_state.positions:
        for position in st.session_state.positions:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>trending_up</i> {position.get('tradingsymbol', 'N/A')}</h4>
                    <p>Quantity: {position.get('quantity', 0)}</p>
                    <p>Buy Avg: ₹{position.get('buy_avg', 0):.2f}</p>
                    <p>Unrealized P&L: ₹{position.get('unrealised_pnl', 0):,.2f}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No positions data available.")

    st.subheader("Trade Log")
    if st.session_state.trade_log:
        for trade in st.session_state.trade_log:
            st.markdown(f"""
                <div class='metric-card'>
                    <h4><i class='material-icons'>history</i> {trade['date']}</h4>
                    <p>Strategy: {trade['strategy']}</p>
                    <p>Capital Deployed: ₹{trade['capital_deployed']:,.2f}</p>
                    <p>Max Loss: ₹{trade['max_loss']:,.2f}</p>
                    <p>P&L: ₹{trade['pnl']:,.2f}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trades executed yet.")
