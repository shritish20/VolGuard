import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from smartbhai_gpt import SmartBhaiGPT
from upstox_api import initialize_upstox_client, fetch_all_api_portfolio_data, prepare_trade_orders, execute_trade_orders, square_off_positions, fetch_market_depth_by_scrip
from data_processing import load_data, generate_features, FEATURE_COLS
from volatility_forecasting import forecast_volatility_future
from backtesting import run_backtest
from strategy_generation import generate_trading_strategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        .smartbhai-container {
            background: #1e2a44;
            border-radius: 12px;
            padding: 15px;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        .smartbhai-input > div > div > input {
            background: #2a3b5a;
            border: 2px solid #00cc00;
            border-radius: 10px;
            padding: 12px;
            color: #e5e5e5;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        .smartbhai-input > div > div > input:focus {
            border-color: #ffcc00;
            outline: none;
            box-shadow: 0 0 5px rgba(255, 204, 0, 0.5);
        }
        .smartbhai-button > button {
            width: 100%;
            background: #e94560;
            color: white;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        .smartbhai-button > button:hover {
            background: #ffcc00;
            color: #1a1a2e;
            transform: translateY(-2px);
        }
        .smartbhai-chat {
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            margin-top: 15px;
            background: #16213e;
            border-radius: 10px;
        }
        .chat-bubble {
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
            font-size: 14px;
            line-height: 1.4;
        }
        .user-bubble {
            background: #e94560;
            color: white;
            margin-left: auto;
            text-align: right;
            border-bottom-right-radius: 2px;
        }
        .smartbhai-bubble {
            background: #00cc00;
            color: #1a1a2e;
            margin-right: auto;
            border-bottom-left-radius: 2px;
        }
        .smartbhai-title {
            color: #ffcc00;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            text-align: center;
        }
        .smartbhai-subtitle {
            color: #a0a0a0;
            font-size: 14px;
            margin-bottom: 15px;
            text-align: center;
        }
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
if "client" not in st.session_state:
    st.session_state.client = None
if "real_time_market_data" not in st.session_state:
    st.session_state.real_time_market_data = None
if "api_portfolio_data" not in st.session_state:
    st.session_state.api_portfolio_data = {}
if "prepared_orders" not in st.session_state:
    st.session_state.prepared_orders = None
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "forecast_log" not in st.session_state:
    st.session_state.forecast_log = None
if "forecast_metrics" not in st.session_state:
    st.session_state.forecast_metrics = None
if "generated_strategy" not in st.session_state:
    st.session_state.generated_strategy = None
if "backtest_cumulative_pnl_chart_data" not in st.session_state:
    st.session_state.backtest_cumulative_pnl_chart_data = None
if "active_strategy_details" not in st.session_state:
    st.session_state.active_strategy_details = None
if "order_placement_errors" not in st.session_state:
    st.session_state.order_placement_errors = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "capital" not in st.session_state:
    st.session_state.capital = 1000000  # Default capital to prevent AttributeError
if "risk_tolerance" not in st.session_state:
    st.session_state.risk_tolerance = "Moderate"  # Default risk tolerance
if "forecast_horizon" not in st.session_state:
    st.session_state.forecast_horizon = 7  # Default forecast horizon

# Initialize SmartBhai GPT
smartbhai_gpt = None
try:
    smartbhai_gpt = SmartBhaiGPT(responses_file="responses.csv")
except Exception as e:
    st.sidebar.error(f"Bhai, SmartBhai GPT load nahi hua: {str(e)}")
    logger.error(f"SmartBhai GPT initialization failed: {str(e)}")

# Check Upstox API hours
def is_market_hours():
    now = datetime.now().time()
    start = datetime.strptime("09:15", "%H:%M").time()
    end = datetime.strptime("15:30", "%H:%M").time()
    return start <= now <= end

# Fetch portfolio data
def fetch_portfolio_data(upstox_client, capital):
    portfolio_summary = {
        "weekly_pnl": 0.0,
        "margin_used": 0.0,
        "exposure": 0.0,
        "total_capital": capital
    }
    if not upstox_client or not upstox_client.get("access_token"):
        logger.warning("Client not available for portfolio data")
        return portfolio_summary
    try:
        portfolio_data = fetch_all_api_portfolio_data(upstox_client)
        st.session_state.api_portfolio_data = portfolio_data
        margin_data = portfolio_data.get("margin", {}).get("data", {})
        positions_data = portfolio_data.get("positions", {}).get("data", [])
        portfolio_summary["margin_used"] = margin_data.get("utilized_margin", 0.0) if isinstance(margin_data, dict) else 0.0
        portfolio_summary["weekly_pnl"] = sum(pos.get("unrealized_mtm", 0.0) + pos.get("realized_profit", 0.0) for pos in positions_data if isinstance(pos, dict))
        portfolio_summary["exposure"] = (portfolio_summary["margin_used"] / capital * 100) if capital > 0 else 0.0
        logger.info("Portfolio data fetched successfully")
        return portfolio_summary
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        return portfolio_summary

# Calculate position PnL with LTP
def calculate_position_pnl_with_ltp(upstox_client, positions_data):
    if not upstox_client or not upstox_client.get("access_token") or not positions_data:
        return positions_data
    updated_positions = []
    for pos in positions_data:
        if not isinstance(pos, dict):
            updated_positions.append(pos)
            continue
        instrument_key = pos.get("instrument_key")
        buy_avg_price = pos.get("buy_avg_price", 0.0)
        sell_avg_price = pos.get("sell_avg_price", 0.0)
        quantity = pos.get("quantity", 0)
        if instrument_key and quantity != 0:
            try:
                market_data = fetch_market_depth_by_scrip(upstox_client, instrument_key=instrument_key)
                ltp = market_data["Data"][0].get("LastTradedPrice", 0.0) if market_data and market_data.get("Data") else 0.0
                position_pnl = quantity * (ltp - buy_avg_price) if quantity > 0 else abs(quantity) * (sell_avg_price - ltp)
                pos['CurrentPnL'] = position_pnl
                pos['LTP'] = ltp
            except Exception as e:
                logger.warning(f"Could not fetch LTP for {instrument_key}: {e}")
                pos['CurrentPnL'] = pos.get("unrealized_mtm", 0.0)
                pos['LTP'] = 0.0
        updated_positions.append(pos)
    return updated_positions

# Sidebar
with st.sidebar:
    st.header("🔑 Upstox Login")
    access_token = st.text_input("Access Token", type="password")
    if st.button("Login to Upstox"):
        st.session_state.client = initialize_upstox_client(access_token)
        if st.session_state.client and st.session_state.client.get("access_token"):
            st.session_state.logged_in = True
            st.success("✅ Logged in to Upstox!")
            logger.info("Upstox login successful")
        else:
            st.session_state.logged_in = False
            st.error("❌ Login failed. Check access token.")
            logger.error("Upstox login failed")

    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=st.session_state.capital, step=100000, format="%d")
        st.session_state.capital = capital
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=["Conservative", "Moderate", "Aggressive"].index(st.session_state.risk_tolerance))
        st.session_state.risk_tolerance = risk_tolerance
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, st.session_state.forecast_horizon)
        st.session_state.forecast_horizon = forecast_horizon
        st.markdown("---")
        st.markdown("**Backtest Parameters**")
        default_start_date = datetime.now().date() - timedelta(days=365)
        default_end_date = datetime.now().date()
        start_date = st.date_input("Start Date", value=default_start_date)
        st.session_state.backtest_start_date = start_date
        end_date = st.date_input("End Date", value=default_end_date)
        st.session_state.backtest_end_date = end_date
        strategy_options = ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Short Straddle", "Short Put Vertical Spread", "Short Call Vertical Spread", "Short Put", "Long Put"]
        strategy_choice = st.selectbox("Backtest Strategy Filter", strategy_options, index=0)
        st.session_state.backtest_strategy = strategy_choice
        if st.button("Run Backtest"):
            st.session_state.backtest_run = True
            st.session_state.backtest_results = None  # Reset previous results
            st.success("Backtest started! Check the Backtest tab for results.")
            logger.info("Backtest initiated")

        st.markdown("---")
        if st.button("Square Off All Positions"):
            if st.session_state.logged_in and st.session_state.client:
                success = square_off_positions(st.session_state.client)
                if success:
                    st.success("All positions squared off successfully!")
                    logger.info("All positions squared off")
                else:
                    st.error("Failed to square off positions. Check logs.")
                    logger.error("Failed to square off positions")
            else:
                st.error("Not logged in to Upstox.")
                logger.warning("Square off attempted without login")

# Main App
st.title("🛡️ VolGuard Pro")
st.markdown("**Your AI-powered options trading cockpit for NIFTY 50**")

# Check market hours
if not is_market_hours():
    st.warning("🕒 Currently outside Upstox market depth hours (9:15 AM–3:30 PM IST). Some data (e.g., option chain) may be limited. Using cached or fallback data.")

# Load Data
if st.session_state.logged_in:
    try:
        df, real_data, data_source = load_data(st.session_state.client)
        if df is None or real_data is None:
            st.error("Failed to load data. Check Upstox API or CSV files.")
            logger.error("Data loading failed from Upstox")
            st.stop()
        st.session_state.real_time_market_data = real_data
        analysis_df = generate_features(df, real_data, st.session_state.capital)
        if analysis_df is None:
            st.error("Failed to generate features. Check data processing.")
            logger.error("Feature generation failed")
            st.stop()
        st.session_state.analysis_df = analysis_df
        logger.info(f"Data loaded successfully from {data_source}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.error(f"Data loading error: {str(e)}")
        st.stop()
else:
    st.warning("Please log in to Upstox to access real-time data.")
    try:
        df, real_data, data_source = load_data(None)  # Fallback to CSV
        if df is None:
            st.error("Failed to load fallback data. Check CSV files.")
            logger.error("Fallback data loading failed")
            st.stop()
        st.session_state.real_time_market_data = real_data or {}
        analysis_df = generate_features(df, real_data, st.session_state.capital)
        if analysis_df is None:
            st.error("Failed to generate features. Check data processing.")
            logger.error("Feature generation failed")
            st.stop()
        st.session_state.analysis_df = analysis_df
        logger.info(f"Data loaded successfully from {data_source}")
    except Exception as e:
        st.error(f"Error loading fallback data: {str(e)}")
        logger.error(f"Fallback data loading error: {str(e)}")
        st.stop()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Market Snapshot",
    "🔮 Volatility Forecast",
    "🤖 Trading Strategy",
    "💼 Portfolio",
    "📝 Journal",
    "📈 Backtest",
    "⚠️ Risk Dashboard"
])

# Tab 1: Market Snapshot
with tab1:
    st.header("📊 Market Snapshot")
    if st.session_state.real_time_market_data:
        market_data = st.session_state.real_time_market_data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("NIFTY Spot", f"{market_data.get('nifty_spot', 'N/A'):.2f}")
        with col2:
            st.metric("VIX", f"{market_data.get('vix', 'N/A'):.2f}")
        with col3:
            st.metric("PCR", f"{market_data.get('pcr', 'N/A'):.2f}")
        with col4:
            st.metric("ATM Straddle", f"{market_data.get('straddle_price', 'N/A'):.2f}")
        
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("ATM Strike", f"{market_data.get('atm_strike', 'N/A'):.2f}")
        with col6:
            st.metric("Max Pain", f"{market_data.get('max_pain_strike', 'N/A'):.2f}")
        with col7:
            st.metric("Max Pain Diff %", f"{market_data.get('max_pain_diff_pct', 'N/A'):.2f}%")
        
        st.subheader("Option Chain")
        if "option_chain" in market_data and not market_data["option_chain"].empty:
            st.dataframe(market_data["option_chain"][["StrikeRate", "CPType", "LastRate", "IV", "OpenInterest", "Volume"]].head(10))
        else:
            st.warning("Option chain data not available. Try during market hours (9:15 AM–3:30 PM IST).")
    else:
        st.warning("No real-time market data available.")
    st.markdown(f"**Data Source**: {data_source}")

# Tab 2: Volatility Forecast
with tab2:
    st.header("🔮 Volatility Forecast")
    if st.session_state.analysis_df is not None:
        try:
            forecast_df, forecast_metrics = forecast_volatility_future(
                st.session_state.analysis_df,
                horizon=st.session_state.forecast_horizon
            )
            st.session_state.forecast_log = forecast_df
            st.session_state.forecast_metrics = forecast_metrics
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Forecasted VIX", f"{forecast_metrics.get('forecasted_vix', 'N/A'):.2f}")
            with col2:
                st.metric("VIX Range Low", f"{forecast_metrics.get('vix_range_low', 'N/A'):.2f}")
            with col3:
                st.metric("VIX Range High", f"{forecast_metrics.get('vix_range_high', 'N/A'):.2f}")
            
            st.subheader("Forecast Trend")
            if not forecast_df.empty:
                st.line_chart(forecast_df[["VIX", "VIX_Forecast"]])
            else:
                st.warning("No forecast data available.")
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            logger.error(f"Volatility forecast error: {str(e)}")
    else:
        st.warning("No analysis data available for forecasting.")

# Tab 3: Trading Strategy
with tab3:
    st.header("🤖 Trading Strategy")
    if st.session_state.analysis_df is not None and st.session_state.real_time_market_data:
        try:
            strategy = generate_trading_strategy(
                st.session_state.analysis_df,
                st.session_state.real_time_market_data,
                st.session_state.capital,
                st.session_state.risk_tolerance
            )
            st.session_state.generated_strategy = strategy
            
            if strategy:
                st.markdown(f"**Recommended Strategy**: {strategy['Strategy']}")
                st.markdown(f"**Confidence**: {strategy['Confidence']:.2%}")
                st.markdown(f"**Deploy Amount**: ₹{strategy['Deploy']:.2f}")
                
                if st.button("Prepare Orders"):
                    orders = prepare_trade_orders(
                        strategy,
                        st.session_state.real_time_market_data,
                        st.session_state.capital
                    )
                    st.session_state.prepared_orders = orders
                    if orders:
                        st.success("Orders prepared successfully!")
                        st.dataframe(pd.DataFrame(orders))
                        logger.info("Orders prepared successfully")
                    else:
                        st.error("Failed to prepare orders.")
                        logger.error("Order preparation failed")
                
                if st.session_state.prepared_orders:
                    if st.button("Execute Orders"):
                        success, response = execute_trade_orders(
                            st.session_state.client,
                            st.session_state.prepared_orders
                        )
                        if success:
                            st.success("Orders executed successfully!")
                            st.session_state.trades.extend(st.session_state.prepared_orders)
                            st.session_state.prepared_orders = None
                            logger.info("Orders executed successfully")
                        else:
                            st.error("Order execution failed.")
                            st.session_state.order_placement_errors.append(response)
                            st.json(response)
                            logger.error(f"Order execution failed: {response}")
            else:
                st.warning("No strategy generated.")
        except Exception as e:
            st.error(f"Error generating strategy: {str(e)}")
            logger.error(f"Trading strategy error: {str(e)}")
    else:
        st.warning("No data available for strategy generation.")

# Tab 4: Portfolio
with tab4:
    st.header("💼 Portfolio")
    portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Weekly PnL", f"₹{portfolio_summary['weekly_pnl']:.2f}")
    with col2:
        st.metric("Margin Used", f"₹{portfolio_summary['margin_used']:.2f}")
    with col3:
        st.metric("Exposure", f"{portfolio_summary['exposure']:.2f}%")
    
    st.subheader("Open Positions")
    positions_data = st.session_state.api_portfolio_data.get("positions", {}).get("data", [])
    positions_data = calculate_position_pnl_with_ltp(st.session_state.client, positions_data)
    if positions_data:
        positions_df = pd.DataFrame(positions_data)
        if not positions_df.empty:
            st.dataframe(positions_df[["instrument_key", "quantity", "buy_avg_price", "LTP", "CurrentPnL"]])
        else:
            st.info("No open positions.")
    else:
        st.info("No open positions.")

# Tab 5: Journal
with tab5:
    st.header("📝 Trading Journal")
    journal_file = "journal_log.csv"
    if os.path.exists(journal_file):
        journal_df = pd.read_csv(journal_file)
    else:
        journal_df = pd.DataFrame(columns=["Date", "Strategy", "PnL", "Notes"])
    
    with st.form("journal_form"):
        date = st.date_input("Trade Date", value=datetime.now().date())
        strategy = st.text_input("Strategy")
        pnl = st.number_input("PnL (₹)", format="%.2f")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Log Trade")
        if submitted:
            new_entry = pd.DataFrame({
                "Date": [date],
                "Strategy": [strategy],
                "PnL": [pnl],
                "Notes": [notes]
            })
            journal_df = pd.concat([journal_df, new_entry], ignore_index=True)
            journal_df.to_csv(journal_file, index=False)
            st.session_state.journal_complete = True
            st.success("Trade logged successfully!")
            logger.info("Trade logged successfully")
    
    st.subheader("Trade History")
    if not journal_df.empty:
        st.dataframe(journal_df)
    else:
        st.info("No trades logged yet.")

# Tab 6: Backtest
with tab6:
    st.header("📈 Backtest Results")
    if st.session_state.backtest_run and st.session_state.analysis_df is not None:
        try:
            backtest_results = run_backtest(
                st.session_state.analysis_df,
                st.session_state.backtest_start_date,
                st.session_state.backtest_end_date,
                st.session_state.capital,
                st.session_state.backtest_strategy
            )
            st.session_state.backtest_results = backtest_results
            st.session_state.backtest_run = False
            
            if backtest_results:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{backtest_results['total_return']:.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.2%}")
                
                st.subheader("Performance Chart")
                if "cumulative_pnl" in backtest_results:
                    st.session_state.backtest_cumulative_pnl_chart_data = backtest_results["cumulative_pnl"]
                    st.line_chart(backtest_results["cumulative_pnl"])
            else:
                st.error("Backtest failed. Check data or parameters.")
                logger.error("Backtest failed")
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            logger.error(f"Backtest error: {str(e)}")
    else:
        st.info("Run a backtest from the sidebar to see results.")

# Tab 7: Risk Dashboard
with tab7:
    st.header("⚠️ Risk Dashboard")
    if st.session_state.analysis_df is not None:
        try:
            latest_data = st.session_state.analysis_df.iloc[-1]
            risk_metrics = {
                "VaR_95": latest_data.get("PnL_Day", 0) * -1.65,  # Simplified VaR
                "Max_Loss": latest_data.get("PnL_Day", 0) * -2.33,  # 99% confidence
                "Volatility_Regime": "High" if latest_data.get("VIX", 0) > 20 else "Medium" if latest_data.get("VIX", 0) > 15 else "Low"
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VaR (95%)", f"₹{risk_metrics['VaR_95']:.2f}")
            with col2:
                st.metric("Max Loss (99%)", f"₹{risk_metrics['Max_Loss']:.2f}")
            with col3:
                regime_class = {
                    "Low": "regime-low",
                    "Medium": "regime-medium",
                    "High": "regime-high"
                }.get(risk_metrics["Volatility_Regime"], "regime-medium")
                st.markdown(f"<span class='regime-badge {regime_class}'>{risk_metrics['Volatility_Regime']}</span>", unsafe_allow_html=True)
            
            st.subheader("Risk Exposure")
            st.gauge = f"<div class='gauge'>{portfolio_summary['exposure']:.1f}%</div>"
            st.markdown(st.gauge, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating risk dashboard: {str(e)}")
            logger.error(f"Risk dashboard error: {str(e)}")
    else:
        st.warning("No data available for risk dashboard.")

# SmartBhai GPT Chat Interface
st.markdown("---")
st.markdown("<div class='smartbhai-container'>", unsafe_allow_html=True)
st.markdown("<div class='smartbhai-title'>💬 SmartBhai GPT</div>", unsafe_allow_html=True)
st.markdown("<div class='smartbhai-subtitle'>Ask about IV, strategies, or market buzz!</div>", unsafe_allow_html=True)

if smartbhai_gpt:
    with st.container():
        # Chat history
        st.markdown("<div class='smartbhai-chat'>", unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"<div class='chat-bubble user-bubble'>{chat['message']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-bubble smartbhai-bubble'>{chat['message']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Input form
        with st.form("smartbhai_form", clear_on_submit=True):
            query = st.text_input("Ask SmartBhai:", key="query_input", placeholder="Bhai, IV kya hai? Strategy kya lagau?")
            submit = st.form_submit_button("Send", use_container_width=True)
            if submit and query:
                response = smartbhai_gpt.generate_response(query)
                st.session_state.chat_history.append({"role": "user", "message": query})
                st.session_state.chat_history.append({"role": "assistant", "message": response})
                st.rerun()
else:
    st.error("SmartBhai GPT is not available.")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        VolGuard Pro | Built with ❤️ by Shritish | Powered by Upstox API & Streamlit
        <br>
        <small>Disclaimer: Trading involves risks. Do your own research.</small>
    </div>
""", unsafe_allow_html=True)
