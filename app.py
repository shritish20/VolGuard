Import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from smartbhai_gpt import SmartBhaiGPT  # Import SmartBhai GPT class

# Import modularized components
from fivepaisa_api import initialize_5paisa_client, fetch_all_api_portfolio_data, prepare_trade_orders, execute_trade_orders, square_off_positions, fetch_market_depth_by_scrip
from data_processing import load_data, generate_features, FEATURE_COLS
from volatility_forecasting import forecast_volatility_future
from backtesting import run_backtest
from strategy_generation import generate_trading_strategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# Custom CSS (updated to style SmartBhai GPT widget)
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
        /* SmartBhai GPT Widget Styling */
        .smartbhai-input > div > div > input {
            border: 2px solid #00cc00;
            border-radius: 8px;
            padding: 10px;
            background: #16213e;
            color: #e5e5e5;
        }
        .smartbhai-button > button {
            width: 100%;
            background: #e94560;
            color: white;
            border-radius: 10px;
            padding: 12px;
            margin: 10px 0;
        }
        .smartbhai-button > button:hover {
            transform: scale(1.05);
            background: #00cc00;
        }
        .smartbhai-chat {
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
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
# Initialize chat history and query input for SmartBhai GPT
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Initialize SmartBhai GPT
smartbhai_gpt = None
try:
    smartbhai_gpt = SmartBhaiGPT(responses_file="responses.csv")
except Exception as e:
    st.sidebar.error(f"Bhai, SmartBhai GPT load nahi hua: {str(e)}")

# Fetch portfolio data
def fetch_portfolio_data(client, capital):
    portfolio_summary = {
        "weekly_pnl": 0.0,
        "margin_used": 0.0,
        "exposure": 0.0,
        "total_capital": capital
    }
    if client is None or not client.get_access_token():
        logger.warning("Client not available for portfolio data")
        return portfolio_summary
    try:
        portfolio_data = fetch_all_api_portfolio_data(client)
        st.session_state.api_portfolio_data = portfolio_data
        margin_data = portfolio_data.get("margin", {})
        positions_data = portfolio_data.get("positions", [])
        if isinstance(margin_data, dict):
            portfolio_summary["margin_used"] = margin_data.get("UtilizedMargin", 0.0)
        if isinstance(positions_data, list):
            portfolio_summary["weekly_pnl"] = sum(pos.get("BookedPL", 0.0) + pos.get("UnrealizedMTM", 0.0) for pos in positions_data if isinstance(pos, dict))
        portfolio_summary["exposure"] = (portfolio_summary["margin_used"] / capital * 100) if capital > 0 else 0.0
        return portfolio_summary
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        return portfolio_summary

# Calculate position PnL with LTP
def calculate_position_pnl_with_ltp(client, positions_data):
    if not client or not client.get_access_token() or not positions_data:
        return positions_data
    updated_positions = []
    for pos in positions_data:
        if not isinstance(pos, dict):
            updated_positions.append(pos)
            continue
        scrip_code = pos.get("ScripCode")
        exchange = pos.get("Exch")
        exchange_type = pos.get("ExchType")
        buy_avg_price = pos.get("BuyAvgPrice", 0.0)
        sell_avg_price = pos.get("SellAvgPrice", 0.0)
        buy_qty = pos.get("BuyQty", 0)
        sell_qty = pos.get("SellQty", 0)
        net_qty = buy_qty - sell_qty
        if scrip_code and exchange and exchange_type and net_qty != 0:
            try:
                market_data = fetch_market_depth_by_scrip(client, Exchange=exchange, ExchangeType=exchange_type, ScripCode=scrip_code)
                ltp = market_data["Data"][0].get("LastTradedPrice", 0.0) if market_data and market_data.get("Data") else 0.0
                position_pnl = net_qty * (ltp - buy_avg_price) if net_qty > 0 else abs(net_qty) * (sell_avg_price - ltp)
                pos['CurrentPnL'] = position_pnl
                pos['LTP'] = ltp
            except Exception as e:
                logger.warning(f"Could not fetch LTP for ScripCode {scrip_code}: {e}")
                pos['CurrentPnL'] = pos.get("UnrealizedMTM", 0.0)
                pos['LTP'] = 0.0
        updated_positions.append(pos)
    return updated_positions

# Sidebar
with st.sidebar:
    st.header("🔑 5paisa Login")
    totp_code = st.text_input("TOTP (from Authenticator App)", type="password")
    if st.button("Login to 5paisa"):
        st.session_state.client = initialize_5paisa_client(st.secrets, totp_code)
        if st.session_state.client and st.session_state.client.get_access_token():
            st.session_state.logged_in = True
            st.success("✅ Logged in to 5paisa!")
        else:
            st.session_state.logged_in = False
            st.error("❌ Login failed. Check credentials and TOTP.")
    
    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000, format="%d")
        st.session_state.capital = capital
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
        st.session_state.risk_tolerance = risk_tolerance
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
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
        st.session_state.backtest_strategy_choice = strategy_choice
        st.markdown("---")
        st.header("⚠️ Emergency Actions")
        st.warning("Use with EXTREME CAUTION!")
        if st.button("🚨 Square Off All Positions"):
            if square_off_positions(st.session_state.client):
                st.success("✅ All positions squared off")
            else:
                st.error("❌ Failed to square off positions")
    
    # SmartBhai GPT Chat Widget
    st.markdown("---")
    st.header("🗣️ SmartBhai GPT")
    st.markdown("Ask your trading copilot about options!")
    # Use a session state variable to control the input field
    query = st.text_input(
        "Type your query:",
        value=st.session_state.query_input,
        key="gpt_query_input",
        help="E.g., 'What is IV?' or 'Check my straddle at 21000'"
    )
    if st.button("Ask SmartBhai", key="smartbhai_button"):
        if query and smartbhai_gpt:
            with st.spinner("SmartBhai is thinking..."):
                try:
                    response = smartbhai_gpt.generate_response(query)
                    st.session_state.chat_history.append({"query": query, "response": response})
                    st.session_state.query_input = ""  # Clear the input by updating the controlled state
                except Exception as e:
                    st.error(f"Bhai, kuch gadbad ho gaya: {str(e)}")
        else:
            st.error("Bhai, query ya SmartBhai GPT load nahi hua!")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown('<div class="smartbhai-chat">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            st.markdown(f"**You**: {chat['query']}")
            st.markdown(f"**SmartBhai**: {chat['response']} 😎")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # SEBI disclaimer
    st.markdown("**Disclaimer**: SmartBhai is a decision-support tool, not financial advice. Do your own research!")

# Main Execution
if not st.session_state.logged_in:
    st.info("Please login to 5paisa from the sidebar.")
else:
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)
    run_button = st.button("📈 Run Analysis")
    if run_button:
        with st.spinner("Running Analysis..."):
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.backtest_cumulative_pnl_chart_data = None
            st.session_state.prepared_orders = None
            st.session_state.analysis_df = None
            st.session_state.real_time_market_data = None
            st.session_state.forecast_log = None
            st.session_state.forecast_metrics = None
            st.session_state.generated_strategy = None
            st.session_state.api_portfolio_data = {}
            st.session_state.active_strategy_details = None
            st.session_state.order_placement_errors = []
            df, real_data, data_source = load_data(st.session_state.client)
            st.session_state.analysis_df = df
            st.session_state.real_time_market_data = real_data
            st.session_state.data_source = data_source
            if st.session_state.client:
                st.session_state.api_portfolio_data = fetch_all_api_portfolio_data(st.session_state.client)
            if df is not None:
                df_featured = generate_features(st.session_state.analysis_df, st.session_state.real_time_market_data, st.session_state.capital)
                if df_featured is not None:
                    st.session_state.analysis_df = df_featured
                    backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf, cumulative_pnl_chart_data = run_backtest(
                        st.session_state.analysis_df, st.session_state.capital, st.session_state.backtest_strategy_choice, st.session_state.backtest_start_date, st.session_state.backtest_end_date
                    )
                    st.session_state.backtest_run = True
                    st.session_state.backtest_results = {
                        "backtest_df": backtest_df,
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                        "max_drawdown": max_drawdown,
                        "sharpe_ratio": sharpe_ratio,
                        "sortino_ratio": sortino_ratio,
                        "calmar_ratio": calmar_ratio,
                        "strategy_perf": strategy_perf,
                        "regime_perf": regime_perf
                    }
                    st.session_state.backtest_cumulative_pnl_chart_data = cumulative_pnl_chart_data
                    with st.spinner("Predicting volatility..."):
                        forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(st.session_state.analysis_df, st.session_state.forecast_horizon)
                        st.session_state.forecast_log = forecast_log
                        st.session_state.forecast_metrics = {
                            "garch_vols": garch_vols, "xgb_vols": xgb_vols, "blended_vols": blended_vols,
                            "realized_vol": realized_vol, "confidence_score": confidence_score,
                            "rmse": rmse, "feature_importances": feature_importances
                        }
                    st.session_state.generated_strategy = generate_trading_strategy(
                        st.session_state.analysis_df,
                        st.session_state.forecast_log,
                        st.session_state.forecast_metrics.get("realized_vol"),
                        st.session_state.risk_tolerance,
                        st.session_state.forecast_metrics.get("confidence_score"),
                        st.session_state.capital,
                        st.session_state.violations,
                        st.session_state.journal_complete
                    )
                    if st.session_state.generated_strategy and not st.session_state.generated_strategy.get("Discipline_Lock", False):
                        st.session_state.active_strategy_details = st.session_state.generated_strategy
                else:
                    st.error("Analysis failed: Feature generation error")
            else:
                st.error("Analysis failed: Data loading error")

    tabs = st.tabs(["📊 Snapshot", "📈 Forecast", "🧪 Strategy", "💰 Portfolio", "📝 Journal", "📉 Backtest", "🛡️ Risk Dashboard"])

    # Snapshot Tab
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Market Snapshot")
        if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
            df = st.session_state.analysis_df
            latest_date = df.index[-1].strftime("%d-%b-%Y")
            last_nifty = df["NIFTY_Close"].iloc[-1]
            prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 else last_nifty
            last_vix = df["VIX"].iloc[-1]
            regime = "LOW" if last_vix < 15 else "MEDIUM" if last_vix < 20 else "HIGH"
            if st.session_state.generated_strategy:
                regime = st.session_state.generated_strategy["Regime"]
            regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"}.get(regime, "regime-low")
            st.markdown(f'<div style="text-align: center;"><span class="regime-badge {regime_class}">{regime} Market Regime</span></div>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("NIFTY 50", f"{last_nifty:,.2f}", f"{(last_nifty - prev_nifty)/prev_nifty*100:+.2f}%")
            with col2:
                st.metric("India VIX", f"{last_vix:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%")
            with col3:
                st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}" if 'PCR' in df.columns else "N/A")
            with col4:
                st.metric("Straddle Price", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}" if 'Straddle_Price' in df.columns else "N/A")
            st.markdown(f"**Last Updated**: {latest_date} | **Source**: {st.session_state.get('data_source', 'Unknown')}")
            if st.session_state.real_time_market_data:
                with st.expander("Raw 5paisa API Data"):
                    st.json(st.session_state.real_time_market_data)
        else:
            st.info("Run analysis to see market snapshot")
        st.markdown('</div>', unsafe_allow_html=True)

    # Forecast Tab
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Volatility Forecast")
        if st.session_state.forecast_log is not None and not st.session_state.forecast_log.empty:
            forecast_log = st.session_state.forecast_log
            forecast_metrics = st.session_state.forecast_metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Blended Volatility", f"{np.mean(forecast_log['Blended_Vol']):.2f}%")
            with col2:
                st.metric("Realized Volatility", f"{forecast_metrics['realized_vol']:.2f}%")
            with col3:
                st.metric("Model RMSE", f"{forecast_metrics['rmse']:.2f}%")
                st.markdown(f'<div class="gauge">{int(forecast_metrics["confidence_score"])}%</div><div style="text-align: center;">Confidence</div>', unsafe_allow_html=True)
            st.line_chart(pd.DataFrame({
                "GARCH": forecast_log["GARCH_Vol"],
                "XGBoost": forecast_log["XGBoost_Vol"],
                "Blended": forecast_log["Blended_Vol"]
            }).set_index(forecast_log["Date"]), color=["#e94560", "#00d4ff", "#ffcc00"])
            st.markdown("### Feature Importance")
            if forecast_metrics.get("feature_importances") is not None:
                feature_importance = pd.DataFrame({
                    'Feature': FEATURE_COLS,
                    'Importance': forecast_metrics["feature_importances"]
                }).sort_values(by='Importance', ascending=False)
                st.dataframe(feature_importance, use_container_width=True)
        else:
            st.info("Run analysis to see volatility forecast")
        st.markdown('</div>', unsafe_allow_html=True)

    # Strategy Tab
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧪 Trading Strategy")
        if st.session_state.generated_strategy and st.session_state.generated_strategy.get("Discipline_L-Div-ock", False):
            st.markdown('<div class="alert-banner">⚠️ Discipline Lock: Complete Journaling</div>', unsafe_allow_html=True)
        elif st.session_state.generated_strategy:
            strategy = st.session_state.generated_strategy
            real_data = st.session_state.real_time_market_data
            capital = st.session_state.capital
            regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"}.get(strategy["Regime"], "regime-low")
            st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True)
            st.markdown(f"""
                <div class="strategy-card">
                    <h4>{strategy["Strategy"]}</h4>
                    <span class="regime-badge {regime_class}">{strategy["Regime"]} Regime</span>
                    <p><b>Reason:</b> {strategy["Reason"]}</p>
                    <p><b>Confidence:</b> {strategy["Confidence"]:.2f}</p>
                    <p><b>Risk-Reward:</b> {strategy["Risk_Reward"]:.2f}:1</p>
                    <p><b>Capital Deploy:</b> ₹{strategy["Deploy"]:,.0f}</p>
                    <p><b>Max Loss:</b> ₹{strategy["Max_Loss"]:,.0f}</p>
                    <p><b>Exposure:</b> {strategy["Exposure"]:.2f}%</p>
                    <p><b>Tags:</b> {', '.join(strategy["Tags"])}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if strategy["Risk_Flags"]:
                st.markdown(f'<div class="alert-banner">⚠️ Risk Flags: {", ".join(strategy["Risk_Flags"])}</div>', unsafe_allow_html=True)
            if strategy.get("Behavior_Warnings"):
                for warning in strategy["Behavior_Warnings"]:
                    st.warning(f"⚠️ Behavioral Warning: {warning}")
            st.markdown("---")
            st.subheader("Ready to Trade?")
            st.session_state.active_strategy_details = strategy
            if st.button("📝 Prepare Orders"):
                st.session_state.prepared_orders = prepare_trade_orders(strategy, real_data, capital)
                st.session_state.order_placement_errors = []
            if st.session_state.prepared_orders:
                st.markdown("### Proposed Order Details")
                st.warning("REVIEW CAREFULLY BEFORE PLACING!")
                orders_df = pd.DataFrame(st.session_state.prepared_orders)
                orders_display_cols = ['Leg_Type', 'Strike', 'Expiry', 'Quantity_Lots', 'Quantity_Units', 'Proposed_Price', 'Last_Price_API', 'Stop_Loss_Price', 'Take_Profit_Price', 'ScripCode']
                st.dataframe(orders_df[orders_display_cols], use_container_width=True)
                if st.session_state.order_placement_errors:
                    st.error("Previous order placement failed:")
                    for error in st.session_state.order_placement_errors:
                        st.write(f"- {error}")
                if st.button("✅ Confirm and Place Orders"):
                    success, details = execute_trade_orders(st.session_state.client, st.session_state.prepared_orders)
                    if success:
                        st.success("✅ Orders placed successfully!")
                    else:
                        st.session_state.order_placement_errors = [resp["Response"].get("Message", "Unknown error") for resp in details["responses"] if resp["Response"].get("Status") != 0]
                        st.error("❌ Order placement failed. See errors above.")
            else:
                st.info("Click 'Prepare Orders' to see order details")
        else:
            st.info("Run analysis to generate a strategy")
        st.markdown('</div>', unsafe_allow_html=True)

    # Portfolio Tab
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💰 Portfolio Overview")
        portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current P&L", f"₹{portfolio_summary['weekly_pnl']:,.2f}")
        with col2:
            st.metric("Margin Used", f"₹{portfolio_summary['margin_used']:,.2f}")
        with col3:
            st.metric("Exposure", f"{portfolio_summary['exposure']:.2f}%")
        st.markdown("---")
        if st.session_state.api_portfolio_data:
            st.subheader("Account Data")
            with st.expander("📂 Holdings"):
                holdings_data = st.session_state.api_portfolio_data.get("holdings")
                if holdings_data and isinstance(holdings_data, list):
                    st.dataframe(pd.DataFrame(holdings_data), use_container_width=True)
                else:
                    st.info("No holdings found")
            with st.expander("💲 Margin Details"):
                margin_data = st.session_state.api_portfolio_data.get("margin")
                if margin_data and isinstance(margin_data, dict):
                    st.dataframe(pd.DataFrame([{"Metric": k, "Value": v} for k, v in margin_data.items()]), use_container_width=True)
                else:
                    st.info("No margin data found")
            with st.expander("💹 Open Positions"):
                positions_data = st.session_state.api_portfolio_data.get("positions")
                if positions_data and isinstance(positions_data, list):
                    positions_with_pnl = calculate_position_pnl_with_ltp(st.session_state.client, positions_data)
                    positions_df = pd.DataFrame(positions_with_pnl)
                    format_mapping = {col: '₹{:,.2f}' for col in ['BuyAvgPrice', 'SellAvgPrice', 'LTP', 'CurrentPnL', 'BookedPL', 'UnrealizedMTM']}
                    cols_to_format = {col: fmt for col, fmt in format_mapping.items() if col in positions_df.columns}
                    st.dataframe(positions_df.style.format(cols_to_format), use_container_width=True)
                else:
                    st.info("No open positions found")
            with st.expander("📋 Order Book"):
                order_book_data = st.session_state.api_portfolio_data.get("order_book")
                if order_book_data and isinstance(order_book_data, list):
                    st.dataframe(pd.DataFrame(order_book_data), use_container_width=True)
                else:
                    st.info("No open orders found")
            with st.expander("📜 Trade Book"):
                trade_book_data = st.session_state.api_portfolio_data.get("trade_book")
                if trade_book_data and isinstance(trade_book_data, list):
                    st.dataframe(pd.DataFrame(trade_book_data), use_container_width=True)
                else:
                    st.info("No executed trades found")
            with st.expander("📰 Market Status"):
                market_status_data = st.session_state.api_portfolio_data.get("market_status")
                if market_status_data:
                    st.json(market_status_data)
                else:
                    st.info("Market status not available")
        else:
            st.info("Run analysis to fetch portfolio data")
        st.markdown('</div>', unsafe_allow_html=True)

    # Journal Tab
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Discipline Hub")
        with st.form(key="journal_form"):
            st.markdown("Reflect on your trading decisions")
            reason_strategy = st.selectbox("Why this strategy?", ["High IV", "Low Risk", "Event Opportunity", "Bullish Bias", "Bearish Bias", "Range Bound", "Expiry Play", "Other"])
            override_risk = st.radio("Override risk flags?", ("No", "Yes"), index=0)
            expected_outcome = st.text_area("Trade plan and outcome")
            lessons_learned = st.text_area("Lessons learned (optional)")
            submit_journal = st.form_submit_button("💾 Save Journal Entry")
            if submit_journal:
                score = (3 if override_risk == "No" else 0) + (2 if reason_strategy != "Other" else 0) + (3 if expected_outcome else 0) + (1 if lessons_learned else 0)
                portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
                if portfolio_summary['weekly_pnl'] > 0:
                    score += 1
                score = min(score, 10)
                journal_entry = {
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Strategy_Reason": reason_strategy,
                    "Override_Risk": override_risk,
                    "Expected_Outcome": expected_outcome,
                    "Lessons_Learned": lessons_learned,
                    "Discipline_Score": score
                }
                try:
                    pd.DataFrame([journal_entry]).to_csv("journal_log.csv", mode='a', header=not os.path.exists("journal_log.csv"), index=False, encoding='utf-8')
                    st.success(f"✅ Journal Entry Saved! Score: {score}/10")
                    if score >= 7 and st.session_state.violations > 0:
                        st.session_state.violations = 0
                        st.session_state.journal_complete = True
                        st.success("🔓 Discipline Lock Removed!")
                except Exception as e:
                    st.error(f"❌ Error saving journal: {e}")
        st.markdown("### Past Entries")
        if os.path.exists("journal_log.csv"):
            try:
                journal_df = pd.read_csv("journal_log.csv", encoding='utf-8')
                journal_df['Date'] = pd.to_datetime(journal_df['Date']).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(journal_df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error reading journal: {e}")
        else:
            st.info("No journal entries found")
        st.markdown('</div>', unsafe_allow_html=True)

    # Backtest Tab
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📉 Backtest Results")
        if st.session_state.backtest_run and st.session_state.backtest_results:
            results = st.session_state.backtest_results
            if st.session_state.backtest_cumulative_pnl_chart_data is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total P&L", f"₹{results['total_pnl']:,.2f}")
                with col2:
                    st.metric("Win Rate", f"{results['win_rate']*100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.2f}")
                st.markdown("### Cumulative P&L")
                st.line_chart(st.session_state.backtest_cumulative_pnl_chart_data, color="#e94560")
                st.markdown("### Performance by Strategy")
                if not results["strategy_perf"].empty:
                    st.dataframe(results["strategy_perf"].style.format({"sum": "₹{:,.2f}", "mean": "₹{:,.2f}", "Win_Rate": "{:.2%}"}), use_container_width=True)
                st.markdown("### Performance by Regime")
                if not results["regime_perf"].empty:
                    st.dataframe(results["regime_perf"].style.format({"sum": "₹{:,.2f}", "mean": "₹{:,.2f}", "Win_Rate": "{:.2%}"}), use_container_width=True)
                st.markdown("### Detailed Backtest Trades")
                if not results["backtest_df"].empty:
                    display_cols = ['Date', 'Event', 'Regime', 'Strategy', 'PnL', 'Cumulative_PnL', 'Strategy_Cum_PnL', 'Capital_Deployed', 'Max_Loss', 'Max_Profit', 'Risk_Reward', 'Notes']
                    display_cols_filtered = [col for col in display_cols if col in results["backtest_df"].columns]
                    st.dataframe(results["backtest_df"][display_cols_filtered].style.format({
                        "PnL": "₹{:,.2f}",
                        "Cumulative_PnL": "₹{:,.2f}",
                        "Strategy_Cum_PnL": "₹{:,.2f}",
                        "Capital_Deployed": "₹{:,.2f}",
                        "Max_Loss": "₹{:,.2f}",
                        "Max_Profit": lambda x: f'₹{x:,.2f}' if x != float('inf') else 'Unlimited',
                        "Risk_Reward": "{:.2f}"
                    }), use_container_width=True)
        else:
            st.info("Run analysis to view backtest results")
        st.markdown('</div>', unsafe_allow_html=True)

    # Risk Dashboard Tab
    with tabs[6]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🛡️ Live Risk Management Dashboard")
        st.markdown("### Portfolio Risk Summary")
        portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Capital", f"₹{portfolio_summary.get('total_capital', st.session_state.capital):,.2f}")
        with col2:
            st.metric("Today's P&L", f"₹{portfolio_summary['weekly_pnl']:,.2f}")
        with col3:
            st.metric("Margin Used", f"₹{portfolio_summary['margin_used']:,.2f}")
        st.metric("Current Exposure", f"{portfolio_summary['exposure']:.2f}%")
        st.markdown("---")
        st.markdown("### Open Positions")
        positions_data = st.session_state.api_portfolio_data.get("positions", [])
        if positions_data:
            positions_with_pnl = calculate_position_pnl_with_ltp(st.session_state.client, positions_data)
            if positions_with_pnl:
                positions_df = pd.DataFrame(positions_with_pnl)
                format_mapping = {col: '₹{:,.2f}' for col in ['BuyAvgPrice', 'SellAvgPrice', 'LTP', 'CurrentPnL', 'BookedPL', 'UnrealizedMTM']}
                cols_to_format = {col: fmt for col, fmt in format_mapping.items() if col in positions_df.columns}
                st.dataframe(positions_df.style.format(cols_to_format), use_container_width=True)
            else:
                st.info("Could not calculate PnL for positions")
        else:
            st.info("No open positions found")
        st.markdown("---")
        st.markdown("### Active Strategy Risk")
        if st.session_state.active_strategy_details:
            strategy = st.session_state.active_strategy_details
            regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"}.get(strategy["Regime"], "regime-low")
            st.markdown(f"""
                <div class="strategy-card">
                    <h4>{strategy["Strategy"]}</h4>
                    <span class="regime-badge {regime_class}">{strategy["Regime"]} Regime</span>
                    <p><b>Reason:</b> {strategy["Reason"]}</p>
                    <p><b>Risk-Reward:</b> {strategy["Risk_Reward"]:.2f}:1</p>
                    <p><b>Capital Deploy:</b> ₹{strategy["Deploy"]:,.0f}</p>
                    <p><b>Max Loss:</b> ₹{strategy["Max_Loss"]:,.0f}</p>
                    <p><b>Tags:</b> {', '.join(strategy["Tags"])}</p>
                </div>
            """, unsafe_allow_html=True)
            if strategy["Risk_Flags"]:
                st.warning(f'⚠️ Risk Flags: {", ".join(strategy["Risk_Flags"])}')
            if strategy.get("Behavior_Warnings"):
                for warning in strategy["Behavior_Warnings"]:
                    st.warning(f"⚠️ Behavioral Warning: {warning}")
        st.markdown("---")
        st.markdown("### Value at Risk (VaR)")
        st.info("Simplified VaR for illustration")
        if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
            df_var = st.session_state.analysis_df.copy()
            df_var['NIFTY_Daily_Return'] = df_var['NIFTY_Close'].pct_change()
            df_var = df_var.dropna(subset=['NIFTY_Daily_Return'])
            if not df_var.empty:
                confidence_level = 0.99
                worst_loss_pct = np.percentile(df_var['NIFTY_Daily_Return'], (1 - confidence_level) * 100)
                current_value = portfolio_summary.get('total_capital', st.session_state.capital) + portfolio_summary.get('weekly_pnl', 0.0)
                var_absolute = current_value * abs(worst_loss_pct)
                st.write(f"**Historical 1-Day VaR ({confidence_level*100:.0f}%):** ₹{var_absolute:,.2f}")
                st.caption("Assumes portfolio moves with NIFTY. Does not account for option greeks.")
            else:
                st.info("Insufficient data for VaR")
        else:
            st.info("Run analysis for VaR calculation")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True)
import logging
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from data_processing import FEATURE_COLS # Import the feature columns definition

# Setup logging
logger = logging.getLogger(__name__)

def forecast_volatility_xgboost(df_xgb, forecast_horizon):
    """
    Forecasts volatility using an XGBoost Regressor.
    Takes a DataFrame with features and 'Realized_Vol' and returns XGBoost volatility forecasts.
    """
    try:
        logger.info("Forecasting volatility using XGBoost")

        # Ensure target variable exists and drop NaNs caused by shift
        if "Realized_Vol" not in df_xgb.columns:
             logger.error("Realized_Vol feature is missing for XGBoost target.")
             # Return default values or handle appropriately in the caller
             return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), 0.0, None

        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna(subset=['Target_Vol'] + FEATURE_COLS) # Drop NaNs in target or features

        if len(df_xgb) < 50: # Minimum data for XGBoost training
            logger.warning(f"Insufficient data ({len(df_xgb)} rows) for XGBoost training after dropping NaNs.")
            # Return default values or handle appropriately in the caller
            return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), 0.0, None


        X = df_xgb[FEATURE_COLS]
        y = df_xgb['Target_Vol']

        # Ensure split_index is valid
        split_index = int(len(X) * 0.8)
        if split_index < 1 or split_index >= len(X):
             split_index = max(1, len(X) - 50) # Ensure some test data, at least 50 points

        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # Ensure there's data in train and test sets
        if X_train.empty or X_test.empty:
             logger.warning("Training or testing data is empty after split.")
             return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), 0.0, None


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) # Use transform on test set

        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.debug(f"XGBoost training completed. RMSE: {rmse:.2f}")


        # Forecast using XGBoost
        xgb_vols = []
        # Start forecasting from the last known feature state
        if df_xgb.empty:
            logger.warning("DataFrame is empty for XGBoost forecasting.")
            return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), rmse, model.feature_importances_

        current_row = df_xgb[FEATURE_COLS].iloc[-1].copy()

        for i in range(forecast_horizon):
            # Prepare the current feature row for prediction
            current_row_df = pd.DataFrame([current_row], columns=FEATURE_COLS)
            current_row_scaled = scaler.transform(current_row_df)

            # Predict the next volatility
            next_vol = model.predict(current_row_scaled)[0]
            xgb_vols.append(next_vol)

            # Simulate feature changes for the next day's prediction
            # These simulations should ideally be more sophisticated (e.g., based on predicted price move)
            # For now, keep the current simulation logic as it was, but apply bounds and handle potential NaNs
            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] = np.clip(current_row["VIX"] * np.random.uniform(0.98, 1.02), 5, 50) # Add bounds
            current_row["Straddle_Price"] = np.clip(current_row["Straddle_Price"] * np.random.uniform(0.98, 1.02), 50, 400) # Add bounds
            # VIX_Change_Pct is a daily change, recalculate based on simulated VIX change
            prev_vix = df_xgb["VIX"].iloc[-1] if len(df_xgb)>0 else current_row["VIX"]
            current_row["VIX_Change_Pct"] = ((current_row["VIX"] / (prev_vix if prev_vix > 0 else current_row["VIX"])) - 1) * 100 if prev_vix > 0 else 0

            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.05)) # Reduced noise
            current_row["Realized_Vol"] = np.clip(next_vol * np.random.uniform(0.98, 1.02), 5, 50) # Use predicted vol with noise
            current_row["IVP"] = np.clip(current_row["IVP"] * np.random.uniform(0.99, 1.01), 0, 100) # Add bounds
            current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.02), 0.7, 2.0) # Reduced noise
            current_row["Spot_MaxPain_Diff_Pct"] = np.clip(current_row["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0.98, 1.02), 0.1, 5.0) # Add bounds
            # Event Flag simulation could be improved - maybe base on remaining DTE to next known expiry
            current_row["Event_Flag"] = 1 if current_row["Days_to_Expiry"] <= 3 else 0 # Simple rule based on DTE
            current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 500) # Reduced noise
            current_row["FII_Option_Pos"] += np.random.normal(0, 200) # Reduced noise
            current_row["IV_Skew"] = np.clip(current_row["IV_Skew"] + np.random.normal(0, 0.05), -3, 3) # Reduced noise

            # Ensure no NaNs creep in during simulation - use backward/forward fill as a fallback
            current_row = current_row.fillna(method='bfill').fillna(method='ffill')


        xgb_vols = np.clip(xgb_vols, 5, 50)
        # Apply event spike to XGBoost forecast if the last known day was an event day
        if df_xgb["Event_Flag"].iloc[-1] == 1:
            xgb_vols = [v * 1.05 for v in xgb_vols] # Reduced spike effect


        logger.debug("XGBoost forecast completed.")
        return xgb_vols, rmse, model.feature_importances_

    except Exception as e:
        logger.error(f"Error in XGBoost volatility forecasting: {str(e)}", exc_info=True)
        # Return default values or handle appropriately in the caller
        return np.full(forecast_horizon, df_xgb["VIX"].iloc[-1] if "VIX" in df_xgb.columns and len(df_xgb) > 0 else 15.0), 0.0, None

import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from garch_forecasting import forecast_volatility_garch # Import GARCH function
from xgboost_forecasting import forecast_volatility_xgboost # Import XGBoost function
from data_processing import FEATURE_COLS # Import FEATURE_COLS

# Setup logging
logger = logging.getLogger(__name__)

def forecast_volatility_future(df, forecast_horizon):
    """
    Blends GARCH and XGBoost forecasts and calculates confidence.
    """
    try:
        logger.info("Starting volatility forecasting")
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()

        # Ensure df has enough data and required columns for both models
        required_cols = list(FEATURE_COLS) + ['NIFTY_Close', 'Realized_Vol', 'VIX']
        if len(df) < 200 or not all(col in df.columns for col in required_cols):
            logger.error(f"Insufficient data ({len(df)} days) or missing columns ({[col for col in required_cols if col not in df.columns]}) for volatility forecasting.")
            return None, None, None, None, None, None, None, None


        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        # Ensure realized_vol calculation handles potential NaNs
        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean() if not df["Realized_Vol"].dropna().empty and len(df["Realized_Vol"].dropna().iloc[-5:]) >= 1 else df["VIX"].iloc[-1] if "VIX" in df.columns and len(df) > 0 and not pd.isna(df["VIX"].iloc[-1]) else 15.0 # Fallback realized vol


        # --- Run GARCH Forecasting ---
        garch_vols = forecast_volatility_garch(df.tail(max(200, forecast_horizon * 2)), forecast_horizon) # Use a reasonable window for GARCH

        # --- Run XGBoost Forecasting ---
        # Pass the full DataFrame for XGBoost as it handles its own data splitting
        xgb_vols, rmse, feature_importances = forecast_volatility_xgboost(df, forecast_horizon)

        # Ensure garch_vols and xgb_vols are lists/arrays of the same length
        if not isinstance(garch_vols, (list, np.ndarray)):
             garch_vols = np.full(forecast_horizon, realized_vol) # Fallback if GARCH failed
        if not isinstance(xgb_vols, (list, np.ndarray)):
             xgb_vols = np.full(forecast_horizon, realized_vol) # Fallback if XGBoost failed

        # Ensure lengths match before blending
        min_horizon = min(len(garch_vols), len(xgb_vols), forecast_horizon)
        garch_vols = garch_vols[:min_horizon]
        xgb_vols = xgb_vols[:min_horizon]
        future_dates = future_dates[:min_horizon]


        # Blending GARCH and XGBoost
        if min_horizon > 0:
             # Calculate initial difference for weighting based on the first forecast day
             initial_garch_vol = garch_vols[0] if len(garch_vols) > 0 else realized_vol
             initial_xgb_vol = xgb_vols[0] if len(xgb_vols) > 0 else realized_vol

             garch_diff = np.abs(initial_garch_vol - realized_vol)
             xgb_diff = np.abs(initial_xgb_vol - realized_vol)

             # Avoid division by zero or very small numbers
             total_diff = garch_diff + xgb_diff
             garch_weight = xgb_diff / total_diff if total_diff > 0 else 0.5
             xgb_weight = 1 - garch_weight

             # Apply weights across the forecast horizon
             blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]

             # Confidence based on the agreement between models and deviation from realized
             model_agreement = 1 - (np.abs(initial_garch_vol - initial_xgb_vol) / max(initial_garch_vol, initial_xgb_vol) if max(initial_garch_vol, initial_xgb_vol) > 0 else 0)
             deviation_from_realized = 1 - (min(garch_diff, xgb_diff) / realized_vol if realized_vol > 0 else 0)
             confidence_score = min(100, max(30, (model_agreement * 0.6 + deviation_from_realized * 0.4) * 100)) # Adjust weighting and min confidence
        else:
             blended_vols = []
             confidence_score = 50 # Low confidence if no forecast days
             logger.warning("No forecast days available for blending.")


        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * min_horizon if min_horizon > 0 else [] # Assign confidence per day
        })
        logger.debug("Volatility forecasting completed")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances
    except Exception as e:
        logger.error(f"Error in volatility forecasting: {str(e)}", exc_info=True)
        return None, None, None, None, None, None, None, None
import logging
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital, current_violations, journal_complete):
    """
    Generates a trading strategy based on market conditions and forecasts.
    """
    try:
        logger.info("Generating trading strategy")
        df = df.copy()
        df.index = df.index.normalize()
        if df.empty:
             logger.error("Cannot generate strategy: Input DataFrame is empty.")
             return None

        # Discipline Lock check
        if current_violations >= 2 and not journal_complete:
             logger.info("Discipline Lock active. Cannot generate strategy.")
             return {"Discipline_Lock": True} # Indicate lock is active


        latest = df.iloc[-1]
        # Ensure required columns exist in the latest row
        required_latest_cols = ["ATM_IV", "Realized_Vol", "IV_Skew", "PCR", "Days_to_Expiry", "Event_Flag", "VIX", "Spot_MaxPain_Diff_Pct", "PnL_Day", "VIX_Change_Pct"]
        if not all(col in latest.index for col in required_latest_cols):
             missing = [col for col in required_latest_cols if col not in latest.index]
             logger.error(f"Missing required columns in latest data for strategy generation: {missing}")
             return None # Indicate failure


        # Use realized_vol as a fallback if forecast_log is missing or empty
        avg_vol = np.mean(forecast_log["Blended_Vol"]) if forecast_log is not None and not forecast_log.empty and 'Blended_Vol' in forecast_log.columns and not forecast_log["Blended_Vol"].isna().all() else realized_vol

        iv = latest["ATM_IV"]
        hv = latest["Realized_Vol"]
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"]
        pcr = latest["PCR"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]
        latest_vix = latest["VIX"]
        spot_max_pain_diff_pct = latest["Spot_MaxPain_Diff_Pct"]
        pnl_day = latest["PnL_Day"]
        vix_change_pct = latest["VIX_Change_Pct"]


        risk_flags = []
        if latest_vix > 25:
            risk_flags.append("VIX > 25% - High Volatility Risk")
        if spot_max_pain_diff_pct > 3: # Adjust threshold
            risk_flags.append(f"Spot-Max Pain Diff > {spot_max_pain_diff_pct:.1f}% - Potential Pinning Risk")
        if pnl_day < -0.01 * capital: # Daily loss > 1% of capital
            risk_flags.append(f"Recent Daily Loss ({pnl_day:,.0f} â‚¹ ) - Consider reducing size")
        if vix_change_pct > 8: # Adjust threshold
            risk_flags.append(f"High VIX Spike Detected ({vix_change_pct:+.1f}%)")


        # Regime determination based on average forecast volatility
        if avg_vol < 15:
            regime = "LOW"
        elif avg_vol < 20:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        # Add Event-Driven regime check
        if event_flag == 1 or dte <= 3: # Within 3 days of expiry or explicit event flag
             regime = "EVENT-DRIVEN"


        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0

        # Strategy selection logic - refine based on combined factors
        if regime == "LOW":
            if iv_hv_gap > 2 and dte < 10:
                strategy = "Butterfly Spread"
                reason = "Low forecast vol, IV > HV, near expiry favors pinning"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.5
            elif iv_skew < -1: # Negative skew
                 strategy = "Short Put" # Simple directional play
                 reason = "Low forecast vol, negative IV skew suggests put selling opportunity"
                 tags = ["Directional", "Bullish", "Premium Selling"]
                 risk_reward = 1.5
            else:
                strategy = "Iron Fly"
                reason = "Low forecast volatility favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8

        elif regime == "MEDIUM":
            if iv_hv_gap > 1.5 and iv_skew > 0.5:
                strategy = "Iron Condor"
                reason = "Medium forecast vol, IV > HV, positive skew favors Iron Condor"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.5
            elif pcr > 1.1:
                 strategy = "Short Put Vertical Spread" # Bullish bias
                 reason = "Medium forecast vol, bullish PCR suggests defined risk put spread"
                 tags = ["Directional", "Bullish", "Defined Risk"]
                 risk_reward = 1.2
            elif pcr < 0.9:
                 strategy = "Short Call Vertical Spread" # Bearish bias
                 reason = "Medium forecast vol, bearish PCR suggests defined risk call spread"
                 tags = ["Directional", "Bearish", "Defined Risk"]
                 risk_reward = 1.2
            else:
                strategy = "Short Strangle"
                reason = "Medium forecast vol, balanced indicators, premium capture"
                tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                risk_reward = 1.6

        elif regime == "HIGH":
            if iv_hv_gap > 5 or latest_vix > 28: # High IV spike
                strategy = "Jade Lizard"
                reason = "High IV spike, IV > HV favors Jade Lizard for defined upside"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.0
            elif iv_skew < -2: # Extreme negative skew
                 strategy = "Long Put" # Protective or bearish play
                 reason = "High forecast vol, extreme negative skew suggests downside protection"
                 tags = ["Directional", "Bearish", "Protection"]
                 risk_reward = 2.0
            else:
                 strategy = "Iron Condor" # Can still work in high vol if range is wide enough
                 reason = "High forecast vol, wide expected range favors Iron Condor premium"
                 tags = ["Neutral", "Theta", "Range Bound"]
                 risk_reward = 1.3

        elif regime == "EVENT-DRIVEN":
            if iv > 30 and dte < 3:
                strategy = "Short Straddle"
                reason = "High IV, very near expiry event â€” max premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.8
            elif dte < 7: # Near expiry event
                strategy = "Calendar Spread"
                reason = "Event-based uncertainty and near expiry favors term structure opportunity"
                tags = ["Volatility", "Event", "Calendar"]
                risk_reward = 1.5
            else: # Event further out, but still impactful
                 strategy = "Iron Condor" # Capture premium before the event
                 reason = "Event anticipation favors capturing premium with Iron Condor"
                 tags = ["Neutral", "Event", "Range Bound"]
                 risk_reward = 1.4


        # Confidence score from forecast function
        confidence_score_from_forecast = confidence_score if confidence_score is not None else 50 # Use calculated confidence, default to 50

        # Capital allocation based on regime and risk tolerance
        capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.08) # Match backtest allocation
        position_size_multiplier = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc_pct * position_size_multiplier # Scale by risk tolerance

        # Max loss calculation matching backtest logic
        max_loss_pct = {"Iron Condor": 0.02, "Butterfly Spread": 0.015, "Iron Fly": 0.02, "Short Strangle": 0.03, "Calendar Spread": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04, "Short Put Vertical Spread": 0.015, "Long Put": 0.03, "Short Put": 0.02, "Short Call Vertical Spread": 0.015}.get(strategy, 0.025)
        max_loss = deploy * max_loss_pct


        total_exposure = deploy / capital if capital > 0 else 0

        # Behavior Score
        behavior_score = 8 if total_exposure < 0.10 else 6 # Adjust threshold to 10% exposure
        behavior_warnings = ["Consider reducing position size (Exposure > 10%)"] if behavior_score < 8 else []


        logger.debug(f"Trading strategy generated: {strategy}")
        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence_score_from_forecast, # Use calculated confidence
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure * 100, # Display as percentage
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings,
            "Discipline_Lock": False # Indicate lock is NOT active
        }
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}", exc_info=True)
        return None
import pandas as pd
import re
import streamlit as st
import csv
from fivepaisa_api import fetch_market_depth_by_scrip

class SmartBhaiGPT:
    def __init__(self, responses_file="responses.csv"):
        # Load response templates
        try:
            self.responses = pd.read_csv(
                responses_file,
                quoting=csv.QUOTE_ALL,  # Force quoting to handle commas in text
                encoding='utf-8'
            )
        except FileNotFoundError:
            raise FileNotFoundError("Bhai, responses.csv nahi mila! Check kar project folder mein.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Bhai, responses.csv ka format galat hai! Error: {str(e)}")
    
    def fetch_app_data(self, context_needed):
        """
        Fetch real-time data (e.g., IV, gamma) from VolGuard Pro's data pipeline.
        Primary: st.session_state.analysis_df (from generate_features).
        Fallback 1: 5paisa API via fetch_market_depth_by_scrip.
        Fallback 2: Static fallback_data.csv.
        """
        try:
            # Primary source: st.session_state.analysis_df from generate_features
            if "analysis_df" in st.session_state and st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
                df = st.session_state.analysis_df
                latest_data = df.iloc[-1]  # Get the latest row
                data = {
                    "iv": latest_data.get("IV", 30.0),  # Implied Volatility
                    "gamma": latest_data.get("Gamma", 0.05),
                    "delta": latest_data.get("Delta", 0.4),
                    "vix": latest_data.get("VIX", 25.0),  # India VIX
                    "margin": (st.session_state.get("api_portfolio_data", {}).get("margin", {}).get("UtilizedMargin", 0.0) / st.session_state.get("capital", 1000000) * 100) or 85.0
                }
            else:
                raise ValueError("Analysis DataFrame not available")
        
        except Exception as e:
            # Fallback 1: Try 5paisa API
            try:
                client = st.session_state.get("client")
                if client and client.get_access_token():
                    # Example: Fetch NIFTY options data (adjust ScripCode as needed)
                    market_data = fetch_market_depth_by_scrip(client, Exchange="N", ExchangeType="D", ScripCode=999920000)  # NIFTY index placeholder
                    ltp = market_data["Data"][0].get("LastTradedPrice", 0.0) if market_data and market_data.get("Data") else 0.0
                    # Mock option greeks (replace with actual option chain data if available)
                    data = {
                        "iv": 30.0,  # Replace with actual IV from option chain
                        "gamma": 0.05,
                        "delta": 0.4,
                        "vix": 25.0,  # Replace with actual VIX from API
                        "margin": (st.session_state.get("api_portfolio_data", {}).get("margin", {}).get("UtilizedMargin", 0.0) / st.session_state.get("capital", 1000000) * 100) or 85.0
                    }
                else:
                    raise ValueError("5paisa client not available")
            except Exception as e2:
                # Fallback 2: Load from fallback_data.csv
                try:
                    fallback_df = pd.read_csv("fallback_data.csv", encoding='utf-8')
                    latest_fallback = fallback_df.iloc[-1]
                    data = {
                        "iv": latest_fallback.get("iv", 30.0),
                        "gamma": latest_fallback.get("gamma", 0.05),
                        "delta": latest_fallback.get("delta", 0.4),
                        "vix": latest_fallback.get("vix", 25.0),
                        "margin": latest_fallback.get("margin", 85.0)
                    }
                except Exception as e3:
                    # Last resort: Hardcoded defaults
                    data = {
                        "iv": "N/A",
                        "gamma": "N/A",
                        "delta": "N/A",
                        "vix": "N/A",
                        "margin": "N/A"
                    }
                    print(f"Error fetching data: Primary failed ({e}), API failed ({e2}), CSV failed ({e3})")
        
        # Return only the needed context
        return {key: data.get(key, "N/A") for key in context_needed.split(",")}
    
    def generate_response(self, user_query):
        """
        Match user query to a response template and fill with app data.
        """
        user_query = user_query.lower().strip()
        
        # Find matching response
        for _, row in self.responses.iterrows():
            pattern = row["query_pattern"]
            if re.search(pattern, user_query):
                # Fetch required context (e.g., IV, gamma)
                context = self.fetch_app_data(row["context_needed"])
                
                # Fill response template
                try:
                    response = row["response_template"].format(**context)
                    return response
                except KeyError:
                    return "Bhai, data thoda off lag raha hai. Try again! Do your own research!"
        
        # Fallback response for unmatched queries
        return "Bhai, yeh query thodi alag hai. Kya bol raha hai, thoda clearly bata? 😜 Do your own research!"

# Test the class
if __name__ == "__main__":
    gpt = SmartBhaiGPT()
    test_queries = [
        "What is IV?",
        "Check my straddle at 21000",
        "Should I hedge?",
        "Random query"
    ]
    for query in test_queries:
        print(f"Query: {query}")
        print(f"Response: {gpt.generate_response(query)}\n")import logging
import pandas as pd
import numpy as np
from arch import arch_model

# Setup logging
logger = logging.getLogger(__name__)

def forecast_volatility_garch(df_garch, forecast_horizon):
    """
    Forecasts volatility using a GARCH(1,1) model.
    Takes a DataFrame with 'NIFTY_Close' and returns GARCH volatility forecasts.
    """
    try:
        logger.info("Forecasting volatility using GARCH(1,1)")

        if len(df_garch) < 100: # Minimum data for GARCH stability
             logger.warning(f"Insufficient data ({len(df_garch)} rows) for GARCH model. Skipping GARCH forecast.")
             # Return default values or handle appropriately in the caller
             return np.full(forecast_horizon, df_garch["VIX"].iloc[-1] if "VIX" in df_garch.columns and len(df_garch) > 0 else 15.0)


        # Ensure Log_Returns calculation handles potential NaNs at the beginning
        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'].pct_change() + 1).dropna() * 100

        if df_garch['Log_Returns'].empty or len(df_garch['Log_Returns']) < 100:
             logger.warning(f"Insufficient historical returns data ({len(df_garch['Log_Returns'])}) for GARCH.")
             # Return default values or handle appropriately in the caller
             return np.full(forecast_horizon, df_garch["VIX"].iloc[-1] if "VIX" in df_garch.columns and len(df_garch) > 0 else 15.0)


        garch_model = arch_model(df_garch['Log_Returns'], vol='Garch', p=1, q=1, rescale=False)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)

        # Convert conditional standard deviation to annualized volatility (%)
        garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50) # Apply reasonable bounds

        logger.debug("GARCH forecast completed.")
        return garch_vols

    except Exception as e:
        logger.error(f"Error in GARCH volatility forecasting: {str(e)}", exc_info=True)
        # Return default values or handle appropriately in the caller
        return np.full(forecast_horizon, df_garch["VIX"].iloc[-1] if "VIX" in df_garch.columns and len(df_garch) > 0 else 15.0)
import logging
import io
import re
from datetime import datetime
from py5paisa import FivePaisaClient
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Helper function to parse 5paisa date string
def parse_5paisa_date_string(date_string):
    if not isinstance(date_string, str):
        return None
    match = re.search(r'/Date\((\d+)[+-]\d+\)/', date_string)
    if match:
        return int(match.group(1))
    return None

# Helper function to format timestamp
def format_timestamp_to_date_str(timestamp_ms):
    if timestamp_ms is None:
        return "N/A"
    try:
        timestamp_s = timestamp_ms / 1000
        dt_object = datetime.fromtimestamp(timestamp_s)
        return dt_object.strftime("%Y-%m-%d")
    except Exception:
        return "N/A"

# Max Pain calculation
def calculate_max_pain(df: pd.DataFrame, nifty_spot: float):
    try:
        if df.empty or "StrikeRate" not in df.columns or "CPType" not in df.columns or "OpenInterest" not in df.columns:
            logger.warning("Option chain data incomplete for max pain")
            return None, None
        df["StrikeRate"] = pd.to_numeric(df["StrikeRate"], errors='coerce')
        df["OpenInterest"] = pd.to_numeric(df["OpenInterest"], errors='coerce').fillna(0)
        df = df.dropna(subset=["StrikeRate", "OpenInterest"]).copy()
        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
        strikes = df["StrikeRate"].unique()
        strikes.sort()
        pain = []
        for K in strikes:
            total_loss = 0
            for s in strikes:
                if s in calls:
                    total_loss += max(0, K - s) * calls.get(s, 0)
                if s in puts:
                    total_loss += max(0, s - K) * puts.get(s, 0)
            pain.append((K, total_loss))
        if not pain:
            logger.warning("No valid strikes for max pain")
            return None, None
        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0
        logger.debug(f"Max Pain: Strike={max_pain_strike}, Diff%={max_pain_diff_pct:.2f}")
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None

# Initialize 5paisa client
def initialize_5paisa_client(secrets, totp_code):
    try:
        logger.info("Initializing 5paisa client")
        cred = {
            "APP_NAME": secrets["fivepaisa"]["APP_NAME"],
            "APP_SOURCE": secrets["fivepaisa"]["APP_SOURCE"],
            "USER_ID": secrets["fivepaisa"]["USER_ID"],
            "PASSWORD": secrets["fivepaisa"]["PASSWORD"],
            "USER_KEY": secrets["fivepaisa"]["USER_KEY"],
            "ENCRYPTION_KEY": secrets["fivepaisa"]["ENCRYPTION_KEY"]
        }
        client = FivePaisaClient(cred=cred)
        client.get_totp_session(
            secrets["fivepaisa"]["CLIENT_CODE"],
            totp_code,
            secrets["fivepaisa"]["PIN"]
        )
        if client.get_access_token():
            logger.info("5paisa client initialized successfully")
            return client
        else:
            logger.error("Failed to get access token")
            return None
    except Exception as e:
        logger.error(f"Error initializing 5paisa client: {str(e)}")
        return None

# Fetch real-time market data
def fetch_real_time_market_data(client):
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available")
        return None
    logger.info("Fetching real-time market data")
    nifty_spot = None
    vix = None
    atm_strike = None
    straddle_price = 0
    pcr = 0
    max_pain_strike = None
    max_pain_diff_pct = None
    expiry_date_str = None
    df_option_chain = pd.DataFrame()
    expiry_timestamp = None
    try:
        # Fetch NIFTY 50
        nifty_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920000,
            "Symbol": "NIFTY",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        nifty_market_feed = client.fetch_market_feed(nifty_req)
        if nifty_market_feed and nifty_market_feed.get("Data"):
            nifty_data = nifty_market_feed["Data"][0]
            nifty_spot = float(nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0)))
            logger.info(f"Fetched NIFTY Spot: {nifty_spot}")

        # Fetch India VIX
        vix_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920005,
            "Symbol": "INDIAVIX",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        vix_market_feed = client.fetch_market_feed(vix_req)
        if vix_market_feed and vix_market_feed.get("Data"):
            vix_data = vix_market_feed["Data"][0]
            vix = float(vix_data.get("LTP", vix_data.get("LastRate", 0)))
            logger.info(f"Fetched VIX: {vix}")

        # Fetch NIFTY expiries
        expiries = client.get_expiry("N", "NIFTY")
        if expiries and expiries.get("Expiry"):
            first_expiry = expiries["Expiry"][0]
            expiry_date_string = first_expiry.get("ExpiryDate")
            expiry_timestamp = parse_5paisa_date_string(expiry_date_string)
            expiry_date_str = format_timestamp_to_date_str(expiry_timestamp)
            logger.info(f"Fetched expiry: {expiry_date_str}")

        # Fetch Option Chain
        if expiry_timestamp:
            option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
            if option_chain and option_chain.get("Options"):
                df_option_chain = pd.DataFrame(option_chain["Options"])
                df_option_chain["StrikeRate"] = pd.to_numeric(df_option_chain["StrikeRate"], errors='coerce')
                df_option_chain["OpenInterest"] = pd.to_numeric(df_option_chain["OpenInterest"], errors='coerce').fillna(0)
                df_option_chain["LastRate"] = pd.to_numeric(df_option_chain["LastRate"], errors='coerce')
                df_option_chain = df_option_chain.dropna(subset=["StrikeRate", "OpenInterest", "LastRate"]).copy()
                logger.info(f"Option chain fetched: {len(df_option_chain)} rows")

        # Calculate ATM, Straddle, PCR, Max Pain
        if nifty_spot and not df_option_chain.empty:
            atm_strike_iloc = (df_option_chain["StrikeRate"] - nifty_spot).abs().argmin()
            atm_strike = df_option_chain["StrikeRate"].iloc[atm_strike_iloc]
            atm_data = df_option_chain[df_option_chain["StrikeRate"] == atm_strike]
            atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
            atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
            straddle_price = atm_call + atm_put
            calls_oi_sum = df_option_chain[df_option_chain["CPType"] == "CE"]["OpenInterest"].sum()
            puts_oi_sum = df_option_chain[df_option_chain["CPType"] == "PE"]["OpenInterest"].sum()
            pcr = puts_oi_sum / calls_oi_sum if calls_oi_sum != 0 else float("inf")
            max_pain_strike, max_pain_diff_pct = calculate_max_pain(df_option_chain, nifty_spot)
            logger.info(f"Calculated: ATM={atm_strike}, Straddle={straddle_price}, PCR={pcr}")

        vix_change_pct = ((vix / (df_option_chain["IV"].iloc[-2] if "IV" in df_option_chain.columns and len(df_option_chain) >= 2 else vix)) - 1) * 100 if vix else 0

        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "vix_change_pct": vix_change_pct,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "expiry": expiry_date_str,
            "option_chain": df_option_chain,
            "source": "5paisa API (LIVE)"
        }
    except Exception as e:
        logger.error(f"Error fetching real-time data: {str(e)}")
        return None

# Fetch portfolio data
def fetch_all_api_portfolio_data(client):
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available")
        return {}
    logger.info("Fetching portfolio data")
    portfolio_data = {}
    try:
        portfolio_data["holdings"] = client.holdings()
        portfolio_data["margin"] = client.margin()
        portfolio_data["positions"] = client.positions()
        portfolio_data["order_book"] = client.order_book()
        portfolio_data["trade_book"] = client.get_tradebook()
        portfolio_data["market_status"] = client.get_market_status()
        logger.info("Portfolio data fetched")
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
    return portfolio_data

# Fetch market depth
def fetch_market_depth_by_scrip(client, Exchange, ExchangeType, ScripCode):
    if client is None or not client.get_access_token():
        logger.warning("Client not available for market depth")
        return None
    try:
        req = [{
            "Exch": Exchange,
            "ExchType": ExchangeType,
            "ScripCode": ScripCode
        }]
        depth = client.fetch_market_depth(req)
        if depth and depth.get("Data"):
            return depth
        logger.warning(f"No market depth for ScripCode {ScripCode}")
        return None
    except Exception as e:
        logger.error(f"Error fetching market depth for ScripCode {ScripCode}: {str(e)}")
        return None

# Prepare trade orders
def prepare_trade_orders(strategy, real_data, capital):
    logger.info(f"Preparing orders for: {strategy['Strategy']}")
    if not real_data or "option_chain" not in real_data or real_data["option_chain"].empty:
        logger.error("Invalid real-time data")
        return None
    option_chain = real_data["option_chain"]
    atm_strike = real_data["atm_strike"]
    expiry_date_str = real_data["expiry"]
    straddle_price_live = real_data["straddle_price"]
    lot_size = 75  # NIFTY lot size
    deploy = strategy["Deploy"]
    premium_per_lot = straddle_price_live * lot_size if straddle_price_live > 0 else 200 * lot_size
    lots = max(1, min(10, int(deploy / premium_per_lot)))
    orders_to_place = []
    strategy_legs = []

    if strategy["Strategy"] == "Short Straddle":
        strategy_legs = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")]
    elif strategy["Strategy"] == "Short Strangle":
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        call_strike = next((s for s in strikes_sorted if s >= atm_strike + 100), None)
        put_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)
        if call_strike and put_strike:
            strategy_legs = [(call_strike, "CE", "S"), (put_strike, "PE", "S")]
        else:
            logger.error("No suitable strikes for Short Strangle")
            return None
    elif strategy["Strategy"] == "Iron Condor":
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None)
        put_buy_strike = next((s for s in reversed(strikes_sorted) if s < put_sell_strike - 100), None)
        call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 50), None)
        call_buy_strike = next((s for s in strikes_sorted if s > call_sell_strike + 100), None)
        if all([put_sell_strike, put_buy_strike, call_sell_strike, call_buy_strike]):
            strategy_legs = [
                (put_buy_strike, "PE", "B"),
                (put_sell_strike, "PE", "S"),
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B")
            ]
        else:
            logger.error("No suitable strikes for Iron Condor")
            return None

    for leg in strategy_legs:
        strike, cp_type, buy_sell = leg
        opt_data = option_chain[
            (option_chain["StrikeRate"] == strike) &
            (option_chain["CPType"] == cp_type)
        ]
        if opt_data.empty:
            logger.error(f"No data for {cp_type} at strike {strike} for expiry {expiry_date_str}")
            return None
        scrip_code = int(opt_data["ScripCode"].iloc[0])
        latest_price = float(opt_data["LastRate"].iloc[0]) if not pd.isna(opt_data["LastRate"].iloc[0]) else 0.0
        proposed_price = 0  # Market order
        stop_loss_price = latest_price * 0.9 if buy_sell == "B" else latest_price * 1.1
        take_profit_price = latest_price * 1.1 if buy_sell == "B" else latest_price * 0.9
        orders_to_place.append({
            "Strategy": strategy["Strategy"],
            "Leg_Type": f"{buy_sell} {cp_type}",
            "Strike": strike,
            "Expiry": expiry_date_str,
            "Exchange": "N",
            "ExchangeType": "D",
            "ScripCode": scrip_code,
            "Quantity_Lots": lots,
            "Quantity_Units": lots * lot_size,
            "Proposed_Price": proposed_price,
            "Last_Price_API": latest_price,
            "Stop_Loss_Price": stop_loss_price,
            "Take_Profit_Price": take_profit_price
        })
    logger.info(f"Prepared {len(orders_to_place)} orders")
    return orders_to_place

# Execute trade orders
def execute_trade_orders(client, prepared_orders):
    logger.info(f"Executing {len(prepared_orders)} orders")
    if client is None or not client.get_access_token():
        logger.error("5paisa client not available")
        return False, {"error": "Invalid client session"}
    if not prepared_orders:
        logger.warning("No orders to execute")
        return False, {"error": "No orders provided"}
    market_status = client.get_market_status()
    if not market_status.get("MarketStatus", {}).get("IsOpen", False):
        logger.error("Market is closed")
        return False, {"error": "Market is closed"}
    all_successful = True
    responses = []
    for order in prepared_orders:
        try:
            logger.info(f"Placing order: {order}")
            if not isinstance(order["ScripCode"], int) or order["ScripCode"] <= 0:
                logger.error(f"Invalid ScripCode: {order['ScripCode']}")
                all_successful = False
                responses.append({"Order": order, "Response": {"Status": -1, "Message": "Invalid ScripCode"}})
                continue
            response = client.place_order(
                OrderType=order["Leg_Type"].split(" ")[0].upper(),
                Exchange=order["Exchange"],
                ExchangeType=order["ExchangeType"],
                ScripCode=order["ScripCode"],
                Qty=order["Quantity_Units"],
                Price=order["Proposed_Price"],
                IsIntraday=False
            )
            logger.debug(f"Response for ScripCode {order['ScripCode']}: {response}")
            responses.append({"Order": order, "Response": response})
            if response.get("Status") != 0:
                all_successful = False
                error_message = response.get("Message", "Unknown error")
                logger.error(f"Order failed for ScripCode {order['ScripCode']}: {error_message}")
            else:
                logger.info(f"Order placed for ScripCode {order['ScripCode']}")
        except Exception as e:
            all_successful = False
            logger.error(f"Error for ScripCode {order['ScripCode']}: {str(e)}")
            responses.append({"Order": order, "Response": {"Status": -1, "Message": f"Exception: {e}"}})
    return all_successful, {"responses": responses}

# Square off positions
def square_off_positions(client):
    try:
        if client is None or not client.get_access_token():
            logger.error("5paisa client not available")
            return False
        logger.info("Squaring off all positions")
        response = client.squareoff_all()
        if response.get("Status") == 0:
            logger.info("Square off successful")
            return True
        else:
            logger.error(f"Square off failed: {response.get('Message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error squaring off: {str(e)}")
        return Falseimport logging
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fivepaisa_api import fetch_real_time_market_data # Import from the API module

# Setup logging
logger = logging.getLogger(__name__)

# Define feature columns used in modeling
FEATURE_COLS = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos'
]


# Data Loading (API First, then CSV Fallback)
def load_data(client):
    """
    Attempts to load data from 5paisa API first, falls back to CSV if API fails.
    Returns DataFrame for analysis, real-time data dict, and data source tag.
    """
    df = None
    real_data = None
    data_source = "CSV (FALLBACK)" # Default source

    # Attempt to fetch real-time data from API
    # Pass the client to the API fetching function
    real_data = fetch_real_time_market_data(client)

    if real_data and real_data.get("nifty_spot") is not None and real_data.get("vix") is not None:
        logger.info("Successfully fetched real-time data from 5paisa API.")
        data_source = real_data["source"] # Use the source tag from fetch function

        latest_date = datetime.now().date()
        live_df_row = pd.DataFrame({
            "NIFTY_Close": [real_data["nifty_spot"]],
            "VIX": [real_data["vix"]]
        }, index=[pd.to_datetime(latest_date).normalize()])

        # Load historical CSV data
        try:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))

            nifty.columns = nifty.columns.str.strip()
            vix.columns = vix.columns.str.strip()

            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")

            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            # Normalize dates and remove duplicates, keeping last
            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna()

            # Ensure historical data is before the live data date to avoid duplicates on the same day
            historical_df = historical_df[historical_df.index < live_df_row.index[0]]

            # Combine historical and real-time data
            df = pd.concat([historical_df, live_df_row])
            df = df.groupby(df.index).last() # Final deduplication if any
            df = df.sort_index()
            df = df.ffill().bfill() # Fill any remaining gaps

            logger.debug(f"Combined historical and live data. Shape: {df.shape}")

        except Exception as e:
             logger.error(f"Error loading historical CSV data while having live data: {str(e)}")
             # If CSV loading fails but live data is there, just use the live data point
             df = live_df_row
             logger.warning("Proceeding with only live data point due to CSV error.")


    else:
        logger.warning("Failed to fetch real-time data from 5paisa API. Falling back to CSV.")
        real_data = None # Ensure real_data is None on fallback

        try:
            # Load historical CSV data (full range)
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))

            nifty.columns = nifty.columns.str.strip()
            vix.columns = vix.columns.str.strip()

            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")

            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            # Normalize dates and remove duplicates, keeping last
            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna()

            df = df.groupby(df.index).last() # Final deduplication
            df = df.sort_index()
            df = df.ffill().bfill() # Fill any remaining gaps

            logger.debug(f"Loaded data from CSV fallback. Shape: {df.shape}")

        except Exception as e:
            logger.error(f"Fatal error loading data from CSV fallback: {str(e)}")
            return None, None, "Data Load Failed"


    # Ensure data is sufficient for analysis
    if df is None or len(df) < 2:
         logger.error("Insufficient data loaded for analysis.")
         return None, None, data_source


    logger.debug(f"Data loading successful. Final DataFrame shape: {df.shape}. Source: {data_source}")
    return df, real_data, data_source


# Feature Generation
def generate_features(df, real_data, capital):
    try:
        logger.info("Generating features")
        df = df.copy()
        df.index = df.index.normalize()
        n_days = len(df)

        # Use actual real_data values if available, fallback to historical patterns or defaults
        # Fallback to latest historical value if available, otherwise a default
        base_pcr = real_data["pcr"] if real_data and real_data.get("pcr") is not None and not np.isnan(real_data["pcr"]) else df["PCR"].iloc[-1] if "PCR" in df.columns and len(df) > 1 and not pd.isna(df["PCR"].iloc[-1]) else 1.0
        base_straddle_price = real_data["straddle_price"] if real_data and real_data.get("straddle_price") is not None and not np.isnan(real_data["straddle_price"]) else df["Straddle_Price"].iloc[-1] if "Straddle_Price" in df.columns and len(df) > 1 and not pd.isna(df["Straddle_Price"].iloc[-1]) else 200.0
        base_max_pain_diff_pct = real_data["max_pain_diff_pct"] if real_data and real_data.get("max_pain_diff_pct") is not None and not np.isnan(real_data["max_pain_diff_pct"]) else df["Spot_MaxPain_Diff_Pct"].iloc[-1] if "Spot_MaxPain_Diff_Pct" in df.columns and len(df) > 1 and not pd.isna(df["Spot_MaxPain_Diff_Pct"].iloc[-1]) else 0.5
        base_vix_change_pct = real_data["vix_change_pct"] if real_data and real_data.get("vix_change_pct") is not None and not np.isnan(real_data["vix_change_pct"]) else df["VIX_Change_Pct"].iloc[-1] if "VIX_Change_Pct" in df.columns and len(df) > 1 and not pd.isna(df["VIX_Change_Pct"].iloc[-1]) else 0.0


        def calculate_days_to_expiry(dates):
            days_to_expiry = []
            fetched_expiry = None
            if real_data and real_data.get("expiry") and real_data["expiry"] != "N/A":
                 try:
                      fetched_expiry = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date()
                 except ValueError:
                      logger.warning(f"Could not parse fetched expiry date string: {real_data['expiry']}")
                      fetched_expiry = None


            for date in dates:
                date_only = date.date()
                if fetched_expiry and date_only >= fetched_expiry:
                     dte = 0 # On or after expiry day
                elif fetched_expiry and date_only < fetched_expiry:
                     dte = (fetched_expiry - date_only).days
                else:
                    # Fallback for historical data or failed API expiry fetch
                    days_ahead = (3 - date_only.weekday()) % 7 # Days to next Thursday
                    if days_ahead == 0: # If today is Thursday, next expiry is in 7 days
                        days_ahead = 7
                    next_expiry_approx = date_only + timedelta(days=days_ahead)
                    dte = (next_expiry_approx - date_only).days
                days_to_expiry.append(dte)
            return np.array(days_to_expiry)


        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index)
        # Ensure DTE is non-negative
        df["Days_to_Expiry"] = np.clip(df["Days_to_Expiry"], 0, None)


        # These synthetic features need care when mixed with live data
        # For the latest row (live data), use actual fetched values where possible
        df["ATM_IV"] = df["VIX"] # Use VIX as ATM IV approximation if not directly fetched
        if real_data and real_data.get("vix") is not None and not np.isnan(real_data["vix"]):
             df.loc[df.index[-1], "ATM_IV"] = real_data["vix"] # Use live VIX for latest ATM_IV

        # Event Flag based on fetched expiry date if available
        if real_data and real_data.get("expiry") and real_data["expiry"] != "N/A":
             try:
                  fetched_expiry_dt = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date()
                  df["Event_Flag"] = np.where(
                       (df.index.date == fetched_expiry_dt) | # Expiry day
                       (df["Days_to_Expiry"] <= 3), # Near expiry (within 3 days)
                       1, 0
                  )
             except ValueError:
                  logger.warning(f"Could not parse fetched expiry date string for Event Flag: {real_data['expiry']}")
                  # Fallback event flag based on approximate Thursday expiry
                  df["Event_Flag"] = np.where(
                       (df.index.weekday == 3) | # Thursdays
                       (df["Days_to_Expiry"] <= 3),
                       1, 0
                  )
        else:
             # Fallback event flag based on approximate Thursday expiry
             df["Event_Flag"] = np.where(
                  (df.index.weekday == 3) | # Thursdays
                  (df["Days_to_Expiry"] <= 3),
                  1, 0
             )


        # Re-calculate features that depend on previous rows or live data
        def dynamic_ivp(x):
            if len(x) >= 5 and x.iloc[-1] is not None and not pd.isna(x.iloc[-1]):
                # Only consider non-nan historical values for percentile rank
                historical_values = x.iloc[:-1].dropna()
                if not historical_values.empty:
                    return (np.sum(historical_values <= x.iloc[-1]) / len(historical_values)) * 100
            return 50.0 # Default if not enough data or latest is NaN/None
        # Apply IVP calculation, ensuring it handles potential NaNs at the end
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp, raw=False) # Use raw=False to pass series
        # Fill initial NaNs and potential NaNs from apply
        df["IVP"] = df["IVP"].interpolate(method='linear').fillna(50.0) # Use linear interpolation

        market_trend = df["NIFTY_Close"].pct_change().rolling(5).mean().fillna(0)
        # Ensure PCR calculation uses the base_pcr for the latest value
        df["PCR"] = np.clip(base_pcr + np.random.normal(0, 0.05, n_days) + market_trend * -5, 0.7, 2.0) # Reduced random noise
        df.loc[df.index[-1], "PCR"] = base_pcr # Use live PCR for latest


        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100
        # Ensure VIX_Change_Pct calculation uses the base_vix_change_pct for the latest value
        df.loc[df.index[-1], "VIX_Change_Pct"] = base_vix_change_pct # Use live VIX change for latest


        # These synthetic features are less tied to live data directly, keep approximation
        # Ensure Spot_MaxPain_Diff_Pct uses the base_max_pain_diff_pct for the latest value
        df["Spot_MaxPain_Diff_Pct"] = np.clip(base_max_pain_diff_pct + np.random.normal(0, 0.1, n_days) + df["Days_to_Expiry"]*0.01, 0.1, 5.0) # Slightly adjust logic
        df.loc[df.index[-1], "Spot_MaxPain_Diff_Pct"] = base_max_pain_diff_pct # Use live max pain diff for latest


        # FII data, Skew, Realized Vol, etc. are synthetic or historical approximations
        # If live FII/Skew were available from API, they'd be used here.
        # Keeping the random generation for now as they aren't in notebook API calls
        fii_trend = np.random.normal(0, 5000, n_days) # Reduced random noise
        fii_trend[::10] *= -1.5 # More frequent directional changes
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 2000, n_days)).astype(int) # Reduced random noise

        df["IV_Skew"] = np.clip(np.random.normal(0, 0.5, n_days) + (df["VIX"] / 20 - 1) * 2 + (df["Days_to_Expiry"] / 15 - 1)*0.5, -3, 3) # Slightly adjust logic

        # Realized Vol needs historical data window
        df["Realized_Vol"] = df["NIFTY_Close"].pct_change().rolling(5, min_periods=5).std() * np.sqrt(252) * 100
        # Fill initial NaNs and potential NaNs by using VIX or a reasonable default
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"]).fillna(15.0) # Fill with VIX, then a default

        df["Advance_Decline_Ratio"] = np.clip(1.0 + np.random.normal(0, 0.1, n_days) + market_trend * 5, 0.7, 1.5) # Reduced random noise and range
        df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 5e4 + df["FII_Option_Pos"] / 2e4 + df["PCR"]-1) / 3 # Scale FII impact
        df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -1.5, 1.5) # Reduced range
        # Ensure DTE is > 0 for division where needed, or handle division by zero
        df["Gamma_Bias"] = np.clip(df["IV_Skew"] * (30 - np.clip(df["Days_to_Expiry"], 1, 30)) / 30, -2, 2)


        df["Total_Capital"] = capital # Assign capital to every row for backtest scaling

        # Synthetic PnL Day - used in strategy evaluation but not for backtest PnL calculation
        df["PnL_Day"] = np.random.normal(0, capital * 0.005, n_days) * (1 - df["Event_Flag"] * 0.2) # Scale PnL to capital

        # Ensure Straddle_Price uses the base_straddle_price for the latest value
        df["Straddle_Price"] = np.clip(base_straddle_price + np.random.normal(0, base_straddle_price*0.1, n_days), base_straddle_price*0.5, base_straddle_price*1.5) # Base around base_straddle_price
        df.loc[df.index[-1], "Straddle_Price"] = base_straddle_price # Use live straddle for latest


        # Final check and interpolate any remaining NaNs
        if df.isna().sum().sum() > 0:
            logger.warning(f"NaNs found after initial feature generation: {df.isna().sum().sum()}")
            # Use a combination of interpolation and backward/forward fill
            df = df.apply(lambda x: x.interpolate(method='linear')).fillna(method='bfill').fillna(method='ffill')
            if df.isna().sum().sum() > 0:
                 logger.error(f"NaNs still present after interpolation/fill: {df.isna().sum().sum()}")
                 # Potentially raise an error or return None if critical NaNs remain
                 # For now, let's log and continue, assuming downstream handles some NaNs
                 pass


        # Ensure all FEATURE_COLS are present after feature generation and cleaning
        if not all(col in df.columns for col in FEATURE_COLS):
             missing = [col for col in FEATURE_COLS if col not in df.columns]
             logger.error(f"FATAL ERROR: Missing required FEATURE_COLS after generation: {missing}")
             return None # Return None if critical features are missing


        logger.debug("Features generated successfully")
        return df
    except Exception as e:
        logger.error(f"Error generating features: {str(e)}")
        return None
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import math # Import math for log and sqrt

from data_processing import FEATURE_COLS # Import FEATURE_COLS

# Setup logging
logger = logging.getLogger(__name__)

# --- Constants for Transaction Costs (Illustrative - Adjust Based on Broker) ---
# These are examples; check your broker's charges
BROKERAGE_PER_LOT = 20 # Example: ₹20 per lot per side (buy/sell)
EXCHANGE_TRANSACTION_CHARGE_PCT = 0.00053 # Example: NSE F&O percentage
SEBI_TURNOVER_FEE_PCT = 0.0001 # Example: SEBI fee percentage
STT_SELL_OPTIONS_PCT = 0.017 # Example: 0.017% on Sell side (Premium Value)
CLEARING_CHARGE_PER_LOT = 1 # Example: ₹1 per lot per side
GST_ON_TOTAL_COSTS_PCT = 0.18 # 18% GST on Brokerage + Exchange Charges + SEBI Fee + Clearing Charge
STAMP_DUTY_PCT = 0.003 # Example: 0.003% on Buy side (Premium Value)

# --- Black-Scholes Option Pricing Model ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the theoretical price of a European option using the Black-Scholes model.

    Parameters:
    S (float): Underlying asset price
    K (float): Strike price
    T (float): Time to expiry in years (e.g., 30 days / 365)
    r (float): Risk-free interest rate (annual, decimal)
    sigma (float): Volatility (annual, decimal)
    option_type (str): 'call' for a call option, 'put' for a put option.

    Returns:
    float: The theoretical option price.
    """
    # Prevent division by zero or log of zero/negative
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    return max(0.0, price) # Option price cannot be negative

# --- Calculate Transaction Costs ---
def calculate_transaction_costs(strategy, trade_type, num_lots, premium_per_lot):
    """
    Calculates estimated transaction costs for a trade leg.

    Parameters:
    strategy (str): The trading strategy name.
    trade_type (str): "BUY" or "SELL"
    num_lots (int): Number of lots traded.
    premium_per_lot (float): Premium paid/received per lot.

    Returns:
    float: Estimated total transaction cost for this leg.
    """
    if num_lots <= 0 or premium_per_lot < 0:
        return 0.0

    brokerage = BROKERAGE_PER_LOT * num_lots
    # Exchange charges, SEBI fee based on turnover (lots * lot_size * underlying_price_approx - or use premium?)
    # Using premium value is more accurate for options transaction costs
    turnover = premium_per_lot * num_lots * 25 # Assuming Nifty lot size 25 - make this dynamic?
    exchange_charge = turnover * EXCHANGE_TRANSACTION_CHARGE_PCT
    sebi_fee = turnover * SEBI_TURNOVER_FEE_PCT
    clearing_charge = CLEARING_CHARGE_PER_LOT * num_lots

    total_statutory_charges_base = exchange_charge + sebi_fee + clearing_charge
    gst = (brokerage + total_statutory_charges_base) * GST_ON_TOTAL_COSTS_PCT

    stt = 0.0
    stamp_duty = 0.0

    if trade_type == "SELL":
        # STT on premium value on the sell side for options
        stt = premium_per_lot * num_lots * 25 * STT_SELL_OPTIONS_PCT
    elif trade_type == "BUY":
        # Stamp duty on premium value on the buy side
        stamp_duty = premium_per_lot * num_lots * 25 * STAMP_DUTY_PCT # Stamp duty is often very small/negligible

    total_costs = brokerage + total_statutory_charges_base + gst + stt + stamp_duty

    return total_costs


def run_strategy_engine(day_data, avg_vol_forecast, portfolio_pnl, capital):
    """
    Determines the trading strategy based on market regime and indicators.
    (Logic remains similar, but deployment/max_loss are based on strategy type)
    """
    # ... (Strategy engine logic is the same as before) ...
    try:
        # Use day_data (real historical/live features) for strategy decision
        iv = day_data.get("ATM_IV", 0.0)
        hv = day_data.get("Realized_Vol", 0.0)
        iv_hv_gap = iv - hv
        iv_skew = day_data.get("IV_Skew", 0.0)
        dte = day_data.get("Days_to_Expiry", 0)
        event_flag = day_data.get("Event_Flag", 0)
        pcr = day_data.get("PCR", 1.0)
        vix_change_pct = day_data.get("VIX_Change_Pct", 0.0)

        # Drawdown limit check based on total capital
        if portfolio_pnl < -0.10 * capital: # 10% drawdown limit
            return None, None, "Portfolio drawdown limit reached", [], 0, 0, 0, [] # Added empty list for legs


        # Determine regime based on blended forecast volatility (avg_vol_forecast)
        if avg_vol_forecast is None:
             regime = "MEDIUM" # Default if forecast failed
        elif avg_vol_forecast < 15:
            regime = "LOW"
        elif avg_vol_forecast < 20:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        # Add Event-Driven regime check
        if event_flag == 1 or dte <= 3: # Within 3 days of expiry or explicit event flag
             regime = "EVENT-DRIVEN"


        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0 # Base risk-reward
        strategy_legs_definition = [] # Define the legs for the chosen strategy


        # Strategy selection logic based on regime and real-time indicators
        # Define legs with (strike_offset_from_atm, type, buy_sell, quantity_multiplier)
        # strike_offset_from_atm: 0 for ATM, +ve for OTM Calls / ITM Puts, -ve for ITM Calls / OTM Puts
        # This is a simplification; actual strikes would need to be looked up in the option chain data.
        # For backtesting simulation, we'll derive strikes relative to the day's Nifty close.
        # Let's use a simple points offset for strike selection in simulation for now.
        # A more advanced backtest would find actual tradable strikes near these offsets.

        strike_step = 50 # Nifty strike increments

        if regime == "LOW":
            if iv_hv_gap > 3 and dte < 15:
                strategy = "Butterfly Spread (Call)"
                reason = "Low vol & moderate expiry favors pinning strategies"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.5
                # Example Call Butterfly: Buy ITM (ATM - 100), Sell 2x ATM (ATM), Buy OTM (ATM + 100)
                strike_offset = 100
                strategy_legs_definition = [
                    (-strike_offset, "CE", "B", 1), # Buy ITM Call
                    (0, "CE", "S", 2),           # Sell 2x ATM Call
                    (strike_offset, "CE", "B", 1)            # Buy OTM Call
                ]

            elif iv_skew < -1:
                 strategy = "Short Put"
                 reason = "Low forecast vol, negative IV skew suggests put selling opportunity"
                 tags = ["Directional", "Bullish", "Premium Selling"]
                 risk_reward = 1.5
                 # Short OTM Put
                 strike_offset = -100 # Sell 100 points below ATM
                 strategy_legs_definition = [
                     (strike_offset, "PE", "S", 1)
                 ]

            else:
                strategy = "Iron Fly (Short Straddle + Bought Wings)"
                reason = "Low volatility environment favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8
                # Sell ATM Straddle, Buy OTM wings (e.g., +/- 100 points)
                strike_offset = 100
                strategy_legs_definition = [
                    (0, "CE", "S", 1),           # Sell ATM Call
                    (0, "PE", "S", 1),           # Sell ATM Put
                    (strike_offset, "CE", "B", 1), # Buy OTM Call wing
                    (-strike_offset, "PE", "B", 1) # Buy OTM Put wing
                ]

        elif regime == "MEDIUM":
            if iv_hv_gap > 2 and iv_skew > 0.5:
                strategy = "Iron Condor"
                reason = "Medium vol and skew favor wide-range Iron Condor"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.5
                # Sell OTM Strangle, Buy further OTM wings (e.g., Sell +/- 100, Buy +/- 200)
                sell_offset = 100
                buy_offset = 200
                strategy_legs_definition = [
                    (sell_offset, "CE", "S", 1),    # Sell OTM Call
                    (-sell_offset, "PE", "S", 1),   # Sell OTM Put
                    (buy_offset, "CE", "B", 1),     # Buy further OTM Call wing
                    (-buy_offset, "PE", "B", 1)     # Buy further OTM Put wing
                ]

            elif pcr > 1.1 and dte < 10:
                 strategy = "Short Put Vertical Spread"
                 reason = "Medium vol, bullish PCR, and short expiry"
                 tags = ["Directional", "Bullish", "Defined Risk"]
                 risk_reward = 1.2
                 # Sell OTM Put, Buy further OTM Put (e.g., Sell -100, Buy -200)
                 sell_offset = -100
                 buy_offset = -200
                 strategy_legs_definition = [
                     (sell_offset, "PE", "S", 1),
                     (buy_offset, "PE", "B", 1)
                 ]

            elif pcr < 0.9 and dte < 10:
                 strategy = "Short Call Vertical Spread"
                 reason = "Medium vol, bearish PCR, and short expiry"
                 tags = ["Directional", "Bearish", "Defined Risk"]
                 risk_reward = 1.2
                 # Sell OTM Call, Buy further OTM Call (e.g., Sell +100, Buy +200)
                 sell_offset = 100
                 buy_offset = 200
                 strategy_legs_definition = [
                     (sell_offset, "CE", "S", 1),
                     (buy_offset, "CE", "B", 1)
                 ]

            else:
                strategy = "Short Strangle"
                reason = "Balanced vol, premium-rich environment for Short Strangle"
                tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                risk_reward = 1.6
                # Sell OTM Strangle (e.g., +/- 100 points)
                strike_offset = 100
                strategy_legs_definition = [
                    (strike_offset, "CE", "S", 1),
                    (-strike_offset, "PE", "S", 1)
                ]

        elif regime == "HIGH":
            if iv_hv_gap > 5 or vix_change_pct > 5:
                strategy = "Jade Lizard"
                reason = "High IV spike/gap favors Jade Lizard for defined upside risk"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.0
                # Short OTM Call, Short OTM Put, Long further OTM Put (e.g., Short +100 CE, Short -100 PE, Long -200 PE)
                call_sell_offset = 100
                put_sell_offset = -100
                put_buy_offset = -200
                strategy_legs_definition = [
                    (call_sell_offset, "CE", "S", 1),
                    (put_sell_offset, "PE", "S", 1),
                    (put_buy_offset, "PE", "B", 1)
                ]

            elif dte < 10 and iv_hv_gap > 5:
                 strategy = "Iron Condor" # Still viable for premium capture
                 reason = "High vol and near expiry favors wide premium collection"
                 tags = ["Neutral", "Theta", "Range Bound"]
                 risk_reward = 1.3
                 # Use wider strikes in high vol (e.g., Sell +/- 150, Buy +/- 300)
                 sell_offset = 150
                 buy_offset = 300
                 strategy_legs_definition = [
                    (sell_offset, "CE", "S", 1),
                    (-sell_offset, "PE", "S", 1),
                    (buy_offset, "CE", "B", 1),
                    (-buy_offset, "PE", "B", 1)
                 ]
            elif iv_skew < -2:
                 strategy = "Long Put"
                 reason = "High vol suggests potential downside risk, protective put"
                 tags = ["Directional", "Bearish", "Protection"]
                 risk_reward = 2.0
                 # Buy OTM Put (e.g., -100 points)
                 strike_offset = -100
                 strategy_legs_definition = [
                     (strike_offset, "PE", "B", 1)
                 ]
            else:
                 strategy = "Short Strangle"
                 reason = "High volatility offers rich premium for Short Strangle (wider strikes needed)"
                 tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                 risk_reward = 1.5
                 # Use wider strikes (e.g., +/- 150 points)
                 strike_offset = 150
                 strategy_legs_definition = [
                    (strike_offset, "CE", "S", 1),
                    (-strike_offset, "PE", "S", 1)
                 ]


        elif regime == "EVENT-DRIVEN":
            if iv > 35 and dte < 3:
                strategy = "Short Straddle"
                reason = "Extreme IV + very near expiry event — max premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.8
                 # Sell ATM Straddle
                strategy_legs_definition = [
                    (0, "CE", "S", 1),
                    (0, "PE", "S", 1)
                ]
            elif dte < 7:
                strategy = "Calendar Spread (Call)" # Simplified for simulation - assumes buying longer expiry
                reason = "Event-based uncertainty and near expiry favors term structure opportunity"
                tags = ["Volatility", "Event", "Calendar"]
                risk_reward = 1.5
                # Sell Near ATM Call, Buy Far ATM Call (Far not simulated yet, so this is indicative)
                strategy_legs_definition = [
                    (0, "CE", "S", 1) # Only near leg is defined for now
                ]
            else:
                 strategy = "Iron Condor" # Capture premium before the event
                 reason = "Event anticipation favors capturing premium with Iron Condor"
                 tags = ["Neutral", "Event", "Range Bound"]
                 risk_reward = 1.4
                 # Use standard medium vol strikes for now
                 sell_offset = 100
                 buy_offset = 200
                 strategy_legs_definition = [
                    (sell_offset, "CE", "S", 1),
                    (-sell_offset, "PE", "S", 1),
                    (buy_offset, "CE", "B", 1),
                    (-buy_offset, "PE", "B", 1)
                 ]


        # Capital allocation based on regime
        capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.07)
        deploy_capital_base = capital * capital_alloc_pct # Base capital for the strategy


        # Determine the number of lots based on the most expensive leg or total premium
        # In a real scenario, you'd estimate the premium/max loss of the chosen strategy
        # For simulation, let's approximate based on Straddle price and number of legs
        approx_premium_per_lot_total = day_data.get("Straddle_Price", 200.0) * len(strategy_legs_definition) / 2 # Rough estimate
        lots = max(1, int(deploy_capital_base / (approx_premium_per_lot_total * 25)) if approx_premium_per_lot_total > 0 else 1) # Target deploying approx base capital

        # Ensure lots is a reasonable number
        lots = min(lots, 20) # Cap at 20 lots for simulation

        # Calculate actual deployed capital based on number of lots and estimated premium (rough)
        deployed = lots * approx_premium_per_lot_total * 25 # Rough deployed value for reporting

        # Max loss calculation (can be refined per strategy based on strikes)
        # For now, use a simplified max loss percentage of deployed capital
        max_loss_pct = {"Iron Condor": 0.02, "Butterfly Spread (Call)": 0.015, "Iron Fly (Short Straddle + Bought Wings)": 0.02, "Short Strangle": 0.03, "Calendar Spread (Call)": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04, "Short Put Vertical Spread": 0.015, "Long Put": 0.03, "Short Put": 0.02, "Short Call Vertical Spread": 0.015}.get(strategy, 0.025)
        max_loss = deployed * max_loss_pct # Max loss absolute value


        return regime, strategy, reason, tags, deployed, max_loss, risk_reward, strategy_legs_definition, lots # Return lots and legs definition

    except Exception as e:
        logger.error(f"Error in backtest strategy engine: {str(e)}")
        return None, None, f"Strategy engine failed: {str(e)}", [], 0, 0, 0, [], 0


def calculate_trade_pnl(strategy, day_data_start, day_data_end, strategy_legs_definition, num_lots, initial_capital, risk_free_rate_daily):
     """
     Simulates daily PnL for a given strategy trade using Black-Scholes.
     Calculates PnL based on option price changes and transaction costs.
     """
     try:
        lot_size = 25
        total_daily_pnl = 0.0
        total_transaction_costs = 0.0

        # Black-Scholes requires volatility as an annual decimal, time to expiry in years
        volatility_annual_decimal = day_data_start.get("ATM_IV", 15.0) / 100.0 # Use ATM_IV as proxy, convert to decimal
        days_to_expiry_start = day_data_start.get("Days_to_Expiry", 1)
        days_to_expiry_end = day_data_end.get("Days_to_Expiry", max(0, days_to_expiry_start - 1)) # Ensure DTE doesn't increase

        time_to_expiry_start_years = days_to_expiry_start / 365.0 if days_to_expiry_start > 0 else 0.0001 # Avoid T=0 initially
        time_to_expiry_end_years = days_to_expiry_end / 365.0 if days_to_expiry_end > 0 else 0.0001 # Avoid T=0 initially

        underlying_start = day_data_start.get("NIFTY_Close", 0.0)
        underlying_end = day_data_end.get("NIFTY_Close", underlying_start)

        # Simulate volatility change for the day (optional, adds realism)
        # Could use VIX change or a random walk around the forecast
        # For simplicity, let's use the average forecast volatility for pricing in this step
        # A more complex model might use different vols for different strikes (skew)
        volatility_start_of_day = day_data_start.get("ATM_IV", 15.0) / 100.0
        volatility_end_of_day = day_data_end.get("ATM_IV", 15.0) / 100.0 # Use end-of-day ATM_IV from features

        # Handle potential division by zero or log(0) in Black-Scholes parameters
        if underlying_start <= 0 or underlying_end <= 0 or volatility_start_of_day <= 0:
            logger.warning(f"Invalid data for option pricing on {day_data_start.name}. Underlying={underlying_start}/{underlying_end}, Vol={volatility_start_of_day}")
            return 0.0 # Return 0 PnL if pricing inputs are invalid


        for leg in strategy_legs_definition:
            strike_offset_points, option_type_short, buy_sell, quantity_multiplier = leg
            quantity_units = num_lots * lot_size * quantity_multiplier

            # Calculate the strike price for this leg based on the day's ATM price and offset
            # Find the nearest tradable strike to (underlying_start + strike_offset_points)
            # In a real backtest, you'd need historical option chain data to find actual strikes.
            # For this simulation, let's approximate by rounding to the nearest 50 or 100.
            target_strike = underlying_start + strike_offset_points
            strike_price = round(target_strike / strike_step) * strike_step # Round to nearest strike_step

            option_type_bs = 'call' if option_type_short == 'CE' else 'put'

            # Calculate option price at the start of the day
            try:
                 price_start = black_scholes(underlying_start, strike_price, time_to_expiry_start_years, risk_free_rate_daily*365, volatility_start_of_day, option_type_bs)
            except ValueError as e:
                 logger.error(f"BS Error (Start) for {strategy} {option_type_short} K={strike_price}: {e}")
                 price_start = 0.0 # Default to 0 if BS fails


            # Calculate option price at the end of the day
            # Use end-of-day underlying and potentially end-of-day volatility
            try:
                 price_end = black_scholes(underlying_end, strike_price, time_to_expiry_end_years, risk_free_rate_daily*365, volatility_end_of_day, option_type_bs)
            except ValueError as e:
                 logger.error(f"BS Error (End) for {strategy} {option_type_short} K={strike_price}: {e}")
                 price_end = 0.0 # Default to 0 if BS fails


            # Calculate PnL for this leg for the day
            # For a BUY trade, PnL = (Price_End - Price_Start) * Quantity
            # For a SELL trade, PnL = (Price_Start - Price_End) * Quantity (you profit if the price goes down)
            if buy_sell == "B":
                leg_pnl = (price_end - price_start) * quantity_units
            elif buy_sell == "S":
                leg_pnl = (price_start - price_end) * quantity_units
            else:
                leg_pnl = 0.0
                logger.warning(f"Unknown buy_sell type: {buy_sell}")


            total_daily_pnl += leg_pnl

            # Calculate transaction costs for entering and potentially exiting (assuming daily PnL closure for simplicity)
            # A more realistic backtest would track positions and apply costs on open/close events.
            # For simplicity here, let's apply costs as if the position was opened and closed daily.
            # This overestimates costs but provides a conservative estimate.
            # A better approach: apply cost when strategy is ENTERED and when EXITED (either by time or stop-loss/target)
            # Let's refactor to apply costs only on the first day a strategy is active.

            # This simplified model applies costs on the first day of the strategy only.
            # This requires tracking if this is the first day of this strategy instance.
            # For now, let's stick to applying costs per leg, perhaps scaled down.
            # A truly realistic backtest needs to manage position entry/exit explicitly.

            # Let's calculate costs per leg for opening the position
            # We need the premium at the time of opening. Use price_start as initial premium proxy.
            transaction_cost_leg = calculate_transaction_costs(strategy, buy_sell, num_lots * quantity_multiplier, price_start / lot_size) # Cost per unit

            total_transaction_costs += transaction_cost_leg


        # Subtract total transaction costs (applied once per strategy instance in a real backtest)
        # For this simplified daily PnL calculation, let's apply a fraction of the costs daily
        # or just apply full costs on day 1. Let's apply full costs on the first day for simplicity.
        # This needs logic in the main backtest loop to know the first day.

        # Let's refine the backtest loop instead to handle strategy entry/exit and apply costs there.
        # For NOW, within this simplified daily PnL, let's just return the gross PnL.
        # Transaction costs will be applied in the main backtest loop on the day the strategy starts.


        return total_daily_pnl # Return gross PnL for the day


     except Exception as e: # <--- This was the misplaced block
        logger.error(f"Error calculating trade PnL using Black-Scholes for {strategy} on {day_data_start.name}: {str(e)}", exc_info=True)
        return 0.0 # Return 0 PnL on error


# Backtesting Function
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.info(f"Starting robust backtest for {strategy_choice} from {start_date} to {end_date}")
        if df is None or df.empty:
            logger.error("Backtest failed: No data available")
            # Need to return 10 values including the empty DataFrame for chart data
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Ensure index is datetime and unique, then slice and copy
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep='first')] # Remove duplicates keeping the first
        df = df.sort_index() # Sort by date
        df = df.loc[start_date:end_date].copy() # Slice and copy

        if len(df) < 50:
            logger.warning(f"Backtest failed: Insufficient data ({len(df)} days) in selected range.")
            # Need to return 10 values including the empty DataFrame for chart data
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


        required_cols = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price", "PCR", "VIX_Change_Pct", "Spot_MaxPain_Diff_Pct"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Backtest failed: Missing required columns after date slicing: {missing_cols}")
             # Need to return 10 values including the empty DataFrame for chart data
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


        backtest_results = []
        portfolio_pnl = 0.0
        risk_free_rate_annual = 0.06 # Annual risk-free rate (adjust as needed)

        current_strategy = None
        strategy_entry_date = None
        strategy_legs_active = []
        strategy_lots = 0
        deployed_capital_for_strategy = 0 # Initialize deployed capital
        max_loss_for_strategy = 0 # Initialize max loss
        risk_reward_for_strategy = 0 # Initialize risk reward


        # Backtest loop - iterates through days in the selected range
        for i in range(1, len(df)):
            try:
                day_data_end = df.iloc[i] # Data at the end of the current day
                day_data_start = df.iloc[i-1] # Data at the start of the current day (previous day's close)
                date = day_data_end.name

                # Use the historical realized vol from the past few days
                avg_vol_for_strategy = df["Realized_Vol"].iloc[max(0, i-5):i].mean()

                # --- Strategy Decision and Entry ---
                # Only run strategy engine if no strategy is currently active
                if current_strategy is None:
                    regime, strategy_name, reason, tags, deploy, max_loss, risk_reward, strategy_legs_definition, lots = run_strategy_engine(
                        day_data_start, avg_vol_for_strategy, portfolio_pnl, capital # Use start-of-day data for decision
                    )

                    # Filter strategies if a specific one is chosen
                    if strategy_name is not None and (strategy_choice == "All Strategies" or strategy_name == strategy_choice):
                         # Decide to enter the strategy
                         current_strategy = strategy_name
                         strategy_entry_date = date # Strategy starts today
                         strategy_legs_active = strategy_legs_definition # Store legs definition
                         strategy_lots = lots # Store number of lots
                         deployed_capital_for_strategy = deploy # Store deployed capital for this instance
                         max_loss_for_strategy = max_loss # Store max loss for this instance
                         risk_reward_for_strategy = risk_reward # Store risk reward for this instance


                         # Calculate and apply transaction costs on entry
                         entry_costs = 0.0
                         lot_size = 25 # Assuming Nifty lot size
                         strike_step = 50 # Nifty strike increments

                         for leg in strategy_legs_active:
                             strike_offset_points, option_type_short, buy_sell, quantity_multiplier = leg
                             quantity_units = strategy_lots * lot_size * quantity_multiplier

                             # Need to get the estimated premium at entry (use start-of-day Black-Scholes price)
                             target_strike = day_data_start.get("NIFTY_Close", 0.0) + strike_offset_points
                             # Ensure strike price is a reasonable value > 0
                             strike_price = round(target_strike / strike_step) * strike_step if target_strike > 0 else 50 # Round to nearest strike_step, default to 50 if target is <= 0
                             strike_price = max(strike_price, 1) # Ensure strike is at least 1

                             option_type_bs = 'call' if option_type_short == 'CE' else 'put'
                             days_to_expiry_entry = day_data_start.get("Days_to_Expiry", 1)
                             time_to_expiry_entry_years = days_to_expiry_entry / 365.0 if days_to_expiry_entry > 0 else 0.0001
                             volatility_entry = day_data_start.get("ATM_IV", 15.0) / 100.0
                             # Ensure volatility is positive and reasonable
                             volatility_entry = max(volatility_entry, 0.0001) # Ensure volatility is > 0

                             try:
                                  premium_per_unit_entry = black_scholes(
                                      day_data_start.get("NIFTY_Close", 0.0),
                                      strike_price,
                                      time_to_expiry_entry_years,
                                      risk_free_rate_annual, # Use annual rate for BS
                                      volatility_entry,
                                      option_type_bs
                                  )
                             except ValueError:
                                  premium_per_unit_entry = 0.0
                                  logger.warning(f"BS Error at entry cost calculation for {strategy_name} {option_type_short} K={strike_price} on {date}")

                             # Ensure premium is not negative for cost calculation
                             premium_per_unit_entry = max(0.0, premium_per_unit_entry)

                             entry_costs += calculate_transaction_costs(strategy_name, buy_sell, strategy_lots * quantity_multiplier, premium_per_unit_entry * lot_size) # Pass premium per lot


                         portfolio_pnl -= entry_costs # Subtract entry costs from portfolio PnL


                         backtest_results.append({
                            "Date": date,
                            "Event": "ENTRY",
                            "Regime": regime,
                            "Strategy": current_strategy,
                            "PnL": -entry_costs, # Record entry costs as negative PnL on entry day
                            "Cumulative_PnL": portfolio_pnl,
                            "Capital_Deployed": deployed_capital_for_strategy,
                            "Max_Loss": max_loss_for_strategy,
                            "Risk_Reward": risk_reward_for_strategy,
                            "Notes": f"Entered {current_strategy} ({strategy_lots} lots)"
                         })
                         logger.debug(f"Entered strategy {current_strategy} on {date}. Entry Costs: {entry_costs:.2f}")


                # --- Daily PnL Calculation for Active Strategy ---
                if current_strategy is not None:
                    # Calculate PnL for the current day for the active strategy
                    # Pass the annual risk-free rate to calculate_trade_pnl, it will convert to daily for BS
                    daily_gross_pnl = calculate_trade_pnl(
                        current_strategy, day_data_start, day_data_end, strategy_legs_active, strategy_lots, capital, risk_free_rate_annual # Pass annual rate
                    )

                    portfolio_pnl += daily_gross_pnl # Add daily PnL to portfolio


                    # Record daily PnL
                    backtest_results.append({
                        "Date": date,
                        "Event": "DAILY_PNL",
                        "Regime": None, # Regime is determined at entry
                        "Strategy": current_strategy,
                        "PnL": daily_gross_pnl,
                        "Cumulative_PnL": portfolio_pnl,
                        "Capital_Deployed": deployed_capital_for_strategy, # Report deployed capital while active
                        "Max_Loss": max_loss_for_strategy,
                        "Risk_Reward": risk_reward_for_strategy,
                        "Notes": "Daily PnL"
                    })


                    # --- Strategy Exit Conditions (Simplified) ---
                    # Exit if DTE is 0 or 1 (end of expiry cycle for the primary legs)
                    # A real backtest would have stop-loss, target, and time-based exits.
                    if day_data_end.get("Days_to_Expiry", 0) <= 1: # Exit on expiry day or day before
                         exit_costs = 0.0 # Assuming minimal exit costs at expiry

                         portfolio_pnl -= exit_costs # Subtract exit costs

                         backtest_results.append({
                             "Date": date,
                             "Event": "EXIT (Expiry)",
                             "Regime": None,
                             "Strategy": current_strategy,
                             "PnL": -exit_costs, # Record exit costs
                             "Cumulative_PnL": portfolio_pnl,
                             "Capital_Deployed": 0, # Capital is freed up
                             "Max_Loss": 0,
                             "Risk_Reward": 0,
                             "Notes": "Exited due to expiry"
                         })
                         logger.debug(f"Exited strategy {current_strategy} on {date} due to expiry.")

                         # Reset active strategy
                         current_strategy = None
                         strategy_entry_date = None
                         strategy_legs_active = []
                         strategy_lots = 0
                         deployed_capital_for_strategy = 0
                         max_loss_for_strategy = 0
                         risk_reward_for_strategy = 0


            except Exception as e:
                # This except block is for errors *within* the daily loop iteration
                logger.error(f"Error in backtest loop at date {date}: {str(e)}", exc_info=True)
                # If an error occurs during trading simulation for an active strategy, exit it.
                if current_strategy is not None:
                     logger.warning(f"Force exiting strategy {current_strategy} on {date} due to error.")
                     backtest_results.append({
                         "Date": date,
                         "Event": "EXIT (Error)",
                         "Regime": None,
                         "Strategy": current_strategy,
                         "PnL": 0, # Assume 0 PnL for the exit on error day
                         "Cumulative_PnL": portfolio_pnl,
                         "Capital_Deployed": 0,
                         "Max_Loss": 0,
                         "Risk_Reward": 0,
                         "Notes": f"Exited due to error: {e}"
                     })
                     current_strategy = None
                     strategy_entry_date = None
                     strategy_legs_active = []
                     strategy_lots = 0
                     deployed_capital_for_strategy = 0
                     max_loss_for_strategy = 0
                     risk_reward_for_strategy = 0

                continue # Continue backtest even if one day fails


        backtest_df = pd.DataFrame(backtest_results)

        if backtest_df.empty:
            logger.warning(f"No trades generated for {strategy_choice} from {start_date} to {end_date}")
            # Need to return 10 values including the empty DataFrame for chart data
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


        # --- Performance Metrics Calculation ---
        # Need to calculate metrics based on DAILY returns on the *entire* capital
        # Create a daily PnL series, aligning with all dates in the backtest range
        # Filter for DAILY_PNL events to get the gross daily PnL from active strategies
        daily_gross_pnl_series = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].set_index('Date')['PnL']
        # Filter for cost events (ENTRY, EXIT, ERROR) to get the cost PnL
        costs_series = backtest_df[backtest_df['Event'].isin(['ENTRY', 'EXIT (Expiry)', 'EXIT (Error)'])].set_index('Date')['PnL']

        # Combine all PnL events into a single daily series
        # Use add with fill_value=0 to correctly sum PnL and costs that might happen on the same day
        all_daily_pnl_events = daily_gross_pnl_series.add(costs_series, fill_value=0).sort_index()

        # Reindex to include all dates in the backtest range, filling missing days with 0 PnL
        # Get the full date range from the original sliced df
        full_date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='B') # Business days
        # Align the combined PnL series to the full date range
        daily_total_pnl = all_daily_pnl_events.reindex(full_date_range, fill_value=0).sort_index()


        # Calculate cumulative PnL over the full range starting from initial capital
        cumulative_pnl_full = daily_total_pnl.cumsum() + capital # Add initial capital

        # Calculate daily returns based on capital at start of each day
        # Capital at start of day t = Capital at end of day t-1
        # Use .iloc to avoid potential issues with missing dates in index after shift
        capital_at_start_of_day = cumulative_pnl_full.shift(1).fillna(capital)
        daily_returns = daily_total_pnl / capital_at_start_of_day

        # Drop the first day's return if it's NaN (due to shift)
        daily_returns = daily_returns.dropna()

        # Total PnL is the last value of the cumulative PnL relative to initial capital
        total_pnl = cumulative_pnl_full.iloc[-1] - capital if not cumulative_pnl_full.empty else 0


        # Calculate Max Drawdown correctly from the full cumulative PnL series
        if not cumulative_pnl_full.empty:
            cumulative_pnl_values = cumulative_pnl_full.values
            peak_values = np.maximum.accumulate(cumulative_pnl_values)
            drawdown_values = peak_values - cumulative_pnl_values
            max_drawdown_abs = np.max(drawdown_values) # Absolute max drawdown
            # We report Max Drawdown as an absolute negative number or 0
            max_drawdown = -max_drawdown_abs if max_drawdown_abs > 0 else 0
        else:
            max_drawdown = 0


        # Ensure NIFTY returns are aligned for comparison
        # Reindex the original sliced df to align with the dates present in daily_returns
        df_aligned_for_returns = df.reindex(daily_returns.index)
        # Calculate Nifty returns for the aligned dates
        nifty_daily_returns = df_aligned_for_returns["NIFTY_Close"].pct_change()
        # Reindex again to ensure perfect alignment and drop any NaNs
        nifty_daily_returns = nifty_daily_returns.reindex(daily_returns.index).dropna()

        # Reindex daily_returns to match nifty_daily_returns for excess return calculation
        daily_returns_aligned = daily_returns.reindex(nifty_daily_returns.index).fillna(0)

        # Calculate excess returns
        risk_free_rate_daily = risk_free_rate_annual / 252 # Daily risk-free rate
        # Ensure risk_free_rate_daily is broadcast correctly
        excess_returns = daily_returns_aligned - nifty_daily_returns - risk_free_rate_daily


        # Ensure there are enough data points for standard deviation for Sharpe/Sortino (>1)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 and len(excess_returns) > 1 else 0
        # Calculate standard deviation of negative excess returns only
        sortino_std_negative = excess_returns[excess_returns < 0].std()
        sortino_ratio = excess_returns.mean() / sortino_std_negative * np.sqrt(252) if sortino_std_negative != 0 and len(excess_returns[excess_returns < 0]) > 1 else 0
        # Calmar Ratio = Annualized Return / Max Drawdown (%)
        annualized_return = (cumulative_pnl_full.iloc[-1] / capital)**(252/len(cumulative_pnl_full)) - 1 if capital > 0 and len(cumulative_pnl_full) > 0 else 0 # Approx annualized return
        calmar_ratio = annualized_return / (max_drawdown_abs / capital) if (max_drawdown_abs / capital) > 0 else float('inf') # Use absolute max drawdown %


        # Win Rate (calculated based on trades, not daily PnL)
        # A "trade" starts with ENTRY and ends with EXIT
        # Need to pair ENTRY and EXIT events to determine individual trade outcomes
        # This is more complex than counting daily PnL days. Let's keep the simplified daily win rate for now.
        # Let's calculate win rate based on days where the active strategy had positive gross PnL
        days_with_active_strategy = backtest_df[backtest_df['Event'] == 'DAILY_PNL']
        win_rate = (days_with_active_strategy['PnL'] > 0).sum() / len(days_with_active_strategy) if len(days_with_active_strategy) > 0 else 0


        # Performance by Strategy and Regime (recalculate based on the PnL recorded in backtest_df)
        # Filter for DAILY_PNL events to aggregate strategy performance
        strategy_perf = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        # Win rate by strategy is trickier with daily PnL - counts days with positive PnL while that strategy was active
        strategy_win_rates = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].groupby("Strategy")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        strategy_perf = pd.merge(strategy_perf, strategy_win_rates, on="Strategy")

        regime_perf = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        # Win rate by regime - counts days with positive PnL while in that regime (during active strategy)
        regime_win_rates = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].groupby("Regime")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        regime_perf = pd.merge(regime_perf, regime_win_rates, on="Regime")


        logger.debug("Robust backtest completed successfully")

        # Return the refined cumulative PnL for charting
        cumulative_pnl_chart_data = pd.DataFrame({'Cumulative_PnL': cumulative_pnl_full})


        # Ensure the backtest_df index is datetime for display in Streamlit
        if 'Date' in backtest_df.columns:
             backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
             # backtest_df.set_index('Date', inplace=True) # Set index if needed for display, but often better to keep 'Date' as a column

        # Ensure return values are correct even if metrics are not calculated (e.g., insufficient trades)
        total_pnl = total_pnl if pd.notna(total_pnl) else 0
        win_rate = win_rate if pd.notna(win_rate) else 0
        max_drawdown = max_drawdown if pd.notna(max_drawdown) else 0
        sharpe_ratio = sharpe_ratio if pd.notna(sharpe_ratio) and sharpe_ratio != float('inf') and sharpe_ratio != float('-inf') else 0
        sortino_ratio = sortino_ratio if pd.notna(sortino_ratio) and sortino_ratio != float('inf') and sortino_ratio != float('-inf') else 0
        calmar_ratio = calmar_ratio if pd.notna(calmar_ratio) and calmar_ratio != float('inf') and calmar_ratio != float('-inf') else 0


        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf, cumulative_pnl_chart_data # Return cumulative PnL for charting

    except Exception as e:
        # This except block is for errors *outside* the daily loop, like data loading issues
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        # Return default empty values in case of an error
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
