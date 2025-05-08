import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

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
        if st.session_state.generated_strategy and st.session_state.generated_strategy.get("Discipline_Lock", False):
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
                    pd.DataFrame([journal_entry]).to_csv("journal_log.csv", mode='a', header=not os.path.exists("journal_log.csv"), index=False)
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
                journal_df = pd.read_csv("journal_log.csv")
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
