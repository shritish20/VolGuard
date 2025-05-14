# main.py (Updated)
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
# Existing CSS and styling remains the same...

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
# Existing helper functions (get_nearest_expiry, fetch_option_chain, etc.) remain the same...

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

def fetch_live_market_data(access_token):
    try:
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        market_quote_api = upstox_client.MarketQuoteV3Api(client)
        
        response = market_quote_api.get_ltp(instrument_key="NSE_INDEX|Nifty 50")
        ltp_data = response.to_dict().get('data', {}).get('NSE_INDEX|Nifty 50', {}).get('last_price', 0)
        return ltp_data
    except ApiException as e:
        logger.error(f"Live market data fetch error: {e}")
        return 0

def calculate_iron_fly(df, spot, atm_strike, capital, risk_profile):
    atm_row = df[df['Strike'] == atm_strike]
    short_call_strike = atm_strike
    short_put_strike = atm_strike
    short_call_price = atm_row['CE_LTP'].values[0]
    short_put_price = atm_row['PE_LTP'].values[0]
    
    strike_diff = atm_strike * 0.05
    long_call_strike = atm_strike + strike_diff
    long_put_strike = atm_strike - strike_diff
    
    long_call_row = df.iloc[(df['Strike'] - long_call_strike).abs().argsort()[:1]]
    long_put_row = df.iloc[(df['Strike'] - long_put_strike).abs().argsort()[:1]]
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
                result, df, iv_skew_fig, atm_strike, atm_iv = run_volguard(access_token)
                if result:
                    st.session_state.volguard_data = result
                    st.session_state.atm_iv = atm_iv
                    st.session_state.access_token = access_token
                    
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
                    # Existing Snapshot tab display code...

    if st.session_state.volguard_data:
        if st.button("Refresh Live Market Data"):
            with st.spinner("Fetching live market data..."):
                live_spot = fetch_live_market_data(st.session_state.access_token)
                if live_spot:
                    st.session_state.volguard_data['nifty_spot'] = live_spot
                    st.success(f"Updated NIFTY Spot Price: ₹{live_spot}")

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
    # Existing volatility insights code...

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
