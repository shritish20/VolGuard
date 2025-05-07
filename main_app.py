import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import logging
import os
from datetime import datetime, timedelta

# Import modularized components
from fivepaisa_api import initialize_5paisa_client, fetch_all_api_portfolio_data, prepare_trade_orders, execute_trade_orders, square_off_positions
from data_processing import load_data, generate_features, FEATURE_COLS
from volatility_forecasting import forecast_volatility_future
from backtesting import run_backtest # Import the updated run_backtest
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
if "backtest_cumulative_pnl_chart_data" not in st.session_state: # Added for the new chart data
     st.session_state.backtest_cumulative_pnl_chart_data = None


# Fetch portfolio data (can be called on tab switch or button click)
def fetch_portfolio_data(client, capital):
     """
     Fetches and summarizes relevant portfolio data from API with type checking.
     """
     portfolio_summary = {
         "weekly_pnl": 0.0,
         "margin_used": 0.0,
         "exposure": 0.0
     }
     if client is None or not client.get_access_token():
          logger.warning("Client not available to fetch portfolio data.")
          return portfolio_summary

     try:
          # Fetch raw data from API
          portfolio_data = fetch_all_api_portfolio_data(client)
          st.session_state.api_portfolio_data = portfolio_data # Store raw data in session state

          # Summarize relevant metrics from fetched data
          # Ensure the main portfolio_data is a dictionary
          if not isinstance(portfolio_data, dict):
              logger.error(f"Fetched portfolio_data is not a dictionary: {type(portfolio_data)}")
              st.warning("Could not summarize portfolio data. Raw API data might be in unexpected format.")
              return portfolio_summary # Return the initial empty summary

          # Access and check types of nested data
          margin_data = portfolio_data.get("margin", {})
          positions_data = portfolio_data.get("positions", []) # Default to empty list if 'positions' key is missing

          # Check if margin_data is a dictionary before using .get() on it
          if isinstance(margin_data, dict):
               portfolio_summary["margin_used"] = margin_data.get("UtilizedMargin", 0.0)
          else:
               logger.warning(f"'margin' data from API is not a dictionary: {type(margin_data)}. Expected a dict.")
               # Keep margin_used as 0.0

          # Calculate rough PnL from positions (Today's PnL and MTM)
          today_pnl = 0.0
          # Check if positions_data is a list and iterate safely
          if isinstance(positions_data, list):
               # Ensure each item in the list is a dictionary before calling .get()
               today_pnl = sum(pos.get("BookedPL", 0.0) + pos.get("UnrealizedMTM", 0.0) for pos in positions_data if isinstance(pos, dict))
          else:
               logger.warning(f"'positions' data from API is not a list: {type(positions_data)}. Expected a list.")
               # Keep today_pnl as 0.0


          portfolio_summary["weekly_pnl"] = today_pnl # Using 'weekly_pnl' key from original code, update label in UI

          # Calculate exposure - simplified as margin used relative to capital
          portfolio_summary["exposure"] = (portfolio_summary["margin_used"] / capital * 100) if capital > 0 else 0.0

          logger.info("Portfolio summary fetched and updated.")
          return portfolio_summary

     except Exception as e:
          logger.error(f"Error fetching or summarizing portfolio data: {str(e)}", exc_info=True)
          st.error(f"Error fetching portfolio data: {str(e)}")
          st.session_state.api_portfolio_data = {} # Clear potentially incomplete data
          return portfolio_summary # Return default summary on error


# Sidebar Login and Controls
with st.sidebar:
    st.header("🔑 5paisa Login")
    totp_code = st.text_input("TOTP (from Authenticator App)", type="password")

    if st.button("Login to 5paisa"):
        # Pass st.secrets to the initialization function
        st.session_state.client = initialize_5paisa_client(st.secrets, totp_code)
        if st.session_state.client and st.session_state.client.get_access_token():
            st.session_state.logged_in = True
            st.success("✅ Logged in to 5paisa!")
        else:
             st.session_state.logged_in = False
             st.error("❌ Login failed. Check credentials and TOTP.")


    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        capital = st.number_input("Capital (₹)", min_value=100000, value=st.session_state.get('capital', 1000000), step=100000, format="%d", key="capital_input") # Added key and default
        st.session_state.capital = capital # Store capital in session state
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=st.session_state.get('risk_tolerance_index', 1), key="risk_tolerance_input") # Added key and default
        st.session_state.risk_tolerance_index = ["Conservative", "Moderate", "Aggressive"].index(risk_tolerance) # Store index

        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, st.session_state.get('forecast_horizon', 7), key="forecast_horizon_input") # Added key and default
        st.session_state.forecast_horizon = forecast_horizon # Store forecast horizon

        st.markdown("---")
        st.markdown("**Backtest Parameters**")
        default_start_date = datetime.now().date() - timedelta(days=365)
        default_end_date = datetime.now().date() # Today
        start_date = st.date_input("Start Date", value=st.session_state.get('backtest_start_date', default_start_date), key="backtest_start_date_input") # Added key and default
        st.session_state.backtest_start_date = start_date
        end_date = st.date_input("End Date", value=st.session_state.get('backtest_end_date', default_end_date), key="backtest_end_date_input") # Added key and default
        st.session_state.backtest_end_date = end_date
        strategy_options = ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Short Straddle", "Short Put Vertical Spread", "Short Call Vertical Spread", "Short Put", "Long Put"]
        strategy_choice = st.selectbox("Backtest Strategy Filter", strategy_options, index=st.session_state.get('backtest_strategy_choice_index', 0), key="backtest_strategy_choice_input") # Added key and default index
        st.session_state.backtest_strategy_choice_index = strategy_options.index(strategy_choice)
        st.session_state.backtest_strategy_choice = strategy_choice


        st.markdown("---")
        st.header("⚠️ Emergency Actions")
        st.warning("Use with EXTREME CAUTION!")
        # Confirmation for Square Off is handled in the function
        if st.button("🚨 Square Off All Positions"):
             square_off_positions(st.session_state.client)


# Main Execution Area
if not st.session_state.logged_in:
    st.info("Please login to 5paisa from the sidebar to proceed. You need a secrets.toml file with your API credentials.")
else:
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)

    # Run Analysis button moved inside the main logged-in area for clarity
    run_button = st.button("📈 Run Analysis")


    if run_button:
        with st.spinner("Running VolGuard Analysis... Fetching data and generating insights."):
            # Clear previous run results relevant to analysis
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.backtest_cumulative_pnl_chart_data = None # Clear previous chart data
            # st.session_state.violations = 0 # Keep violations state
            # st.session_state.journal_complete = False # Keep journal state
            st.session_state.prepared_orders = None # Clear prepared orders from previous run
            st.session_state.analysis_df = None
            st.session_state.real_time_market_data = None
            st.session_state.forecast_log = None
            st.session_state.forecast_metrics = None
            st.session_state.generated_strategy = None
            st.session_state.api_portfolio_data = {} # Clear portfolio data, will refetch


            # Load Data (API first) - uses client from session state
            df, real_data, data_source = load_data(st.session_state.client)
            st.session_state.analysis_df = df # Store loaded df
            st.session_state.real_time_market_data = real_data # Store real_data
            st.session_state.data_source = data_source # Store data source


            # Fetch all API portfolio data if client is available
            if st.session_state.client:
                 st.session_state.api_portfolio_data = fetch_all_api_portfolio_data(st.session_state.client)


            if df is not None:
                # Generate Features - uses df and real_data from session state
                df_featured = generate_features(st.session_state.analysis_df, st.session_state.real_time_market_data, st.session_state.capital)

                if df_featured is not None:
                     st.session_state.analysis_df = df_featured # Update df in session state with features

                     # Run Backtest - uses featured df, capital, dates, and strategy choice from session state
                     # IMPORTANT: Capture the new cumulative PnL chart data returned
                     backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf, cumulative_pnl_chart_data = run_backtest(
                        st.session_state.analysis_df, st.session_state.capital, st.session_state.backtest_strategy_choice, st.session_state.backtest_start_date, st.session_state.backtest_end_date
                     )

                     # Store backtest results
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
                     # Store the dedicated cumulative PnL chart data
                     st.session_state.backtest_cumulative_pnl_chart_data = cumulative_pnl_chart_data


                     # Volatility Forecasting - uses featured df and forecast horizon from session state
                     with st.spinner("Predicting market volatility..."):
                         forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(st.session_state.analysis_df, st.session_state.forecast_horizon)
                         st.session_state.forecast_log = forecast_log # Store forecast log
                         st.session_state.forecast_metrics = { # Store other forecast results
                             "garch_vols": garch_vols, "xgb_vols": xgb_vols, "blended_vols": blended_vols,
                             "realized_vol": realized_vol, "confidence_score": confidence_score,
                             "rmse": rmse, "feature_importances": feature_importances
                         }

                     # Generate Trading Strategy - uses featured df, forecast results, risk tolerance, capital, violations, and journal state from session state
                     st.session_state.generated_strategy = generate_trading_strategy(
                         st.session_state.analysis_df,
                         st.session_state.forecast_log,
                         st.session_state.forecast_metrics["realized_vol"] if st.session_state.forecast_metrics else None, # Pass realized_vol safely
                         st.session_state.risk_tolerance_input, # Use key for risk tolerance
                         st.session_state.forecast_metrics["confidence_score"] if st.session_state.forecast_metrics else None, # Pass confidence safely
                         st.session_state.capital,
                         st.session_state.violations, # Pass current violations
                         st.session_state.journal_complete # Pass journal state
                     )
                else:
                     st.error("Analysis could not be completed due to feature generation failure.")

            else:
                st.error("Analysis could not be completed due to data loading failure.")


    # Define tabs
    tabs = st.tabs(["📊 Snapshot", "📈 Forecast", "🧪 Strategy", "💰 Portfolio", "📝 Journal", "📉 Backtest"])


    # --- Snapshot Tab ---
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Market Snapshot")
        if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
            df = st.session_state.analysis_df
            latest_date = df.index[-1].strftime("%d-%b-%Y")
            last_nifty = df["NIFTY_Close"].iloc[-1]
            prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 else last_nifty
            last_vix = df["VIX"].iloc[-1]

            # Determine regime based on latest VIX value from the df
            regime = "LOW" if last_vix < 15 else "MEDIUM" if last_vix < 20 else "HIGH"
            # Use generated strategy regime if available and not locked
            if 'generated_strategy' in st.session_state and st.session_state.generated_strategy is not None and not st.session_state.generated_strategy.get("Discipline_Lock", False):
                 regime = st.session_state.generated_strategy["Regime"]

            regime_class = {
                 "LOW": "regime-low",
                 "MEDIUM": "regime-medium",
                 "HIGH": "regime-high",
                 "EVENT-DRIVEN": "regime-event"
            }.get(regime, "regime-low")

            st.markdown(f'<div style="text-align: center;"><span class="regime-badge {regime_class}">{regime} Market Regime</span></div>', unsafe_allow_html=True)
            # Gauge Placeholder
            # st.markdown('<div class="gauge" style="margin: 20px auto;">Gauge Here</div>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("NIFTY 50", f"{last_nifty:,.2f}", f"{(last_nifty - prev_nifty)/prev_nifty*100:+.2f}%" if prev_nifty != 0 else "N/A")
            with col2:
                st.metric("India VIX", f"{last_vix:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%" if 'VIX_Change_Pct' in df.columns and len(df) >= 2 else "N/A")
            with col3:
                st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}" if 'PCR' in df.columns and len(df) > 0 and not pd.isna(df['PCR'].iloc[-1]) else "N/A")
            with col4:
                st.metric("Straddle Price", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}" if 'Straddle_Price' in df.columns and len(df) > 0 and not pd.isna(df['Straddle_Price'].iloc[-1]) else "N/A")

            # Display data source
            source_tag = st.session_state.get('data_source', 'Unknown')
            st.markdown(f"**Last Updated**: {latest_date} | **Source**: {source_tag}")

            # Display raw real-time data fetched if available
            if st.session_state.real_time_market_data and st.session_state.real_time_market_data.get("source") == "5paisa API (LIVE)":
                 with st.expander("Raw 5paisa API Data"):
                      st.json(st.session_state.real_time_market_data)
            elif source_tag == "CSV (FALLBACK)":
                 st.info("Data loaded from CSV fallback. Real-time API data could not be fetched.")

        else:
            st.info("Run the analysis to see the market snapshot.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Forecast Tab ---
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Volatility Forecast")
        if 'forecast_log' in st.session_state and st.session_state.forecast_log is not None and not st.session_state.forecast_log.empty:
             forecast_log = st.session_state.forecast_log
             forecast_metrics = st.session_state.forecast_metrics

             col1, col2, col3 = st.columns(3)
             with col1:
                 st.metric("Avg Blended Volatility", f"{np.mean(forecast_log['Blended_Vol']):.2f}%" if not forecast_log['Blended_Vol'].empty else "N/A")
             with col2:
                 st.metric("Realized Volatility (5-Day Avg)", f"{forecast_metrics['realized_vol']:.2f}%" if forecast_metrics and forecast_metrics.get('realized_vol') is not None else "N/A")
             with col3:
                 st.metric("Model RMSE", f"{forecast_metrics['rmse']:.2f}%" if forecast_metrics and forecast_metrics.get('rmse') is not None else "N/A")
                 confidence_score = forecast_metrics.get('confidence_score', 50) if forecast_metrics else 50
                 st.markdown(f'<div class="gauge" style="margin: 0 auto;">{int(confidence_score)}%</div><div style="text-align: center;">Confidence</div>', unsafe_allow_html=True)

             if not forecast_log.empty:
                 st.line_chart(pd.DataFrame({
                     "GARCH": forecast_log["GARCH_Vol"],
                     "XGBoost": forecast_log["XGBoost_Vol"],
                     "Blended": forecast_log["Blended_Vol"]
                 }).set_index(forecast_log["Date"]), color=["#e94560", "#00d4ff", "#ffcc00"])


             st.markdown("### Feature Importance")
             if forecast_metrics and forecast_metrics.get("feature_importances") is not None and FEATURE_COLS:
                 feature_importance = pd.DataFrame({
                     'Feature': FEATURE_COLS,
                     'Importance': forecast_metrics["feature_importances"]
                 }).sort_values(by='Importance', ascending=False)
                 st.dataframe(feature_importance, use_container_width=True)
             else:
                 st.info("Feature importance data not available.")

        else:
             st.info("Run the analysis to see the volatility forecast.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Strategy Tab ---
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧪 Trading Strategy")

        if 'generated_strategy' in st.session_state and st.session_state.generated_strategy is not None and st.session_state.generated_strategy.get("Discipline_Lock", False):
            st.markdown('<div class="alert-banner">⚠️ Discipline Lock: Complete Journaling to Unlock Trading</div>', unsafe_allow_html=True)
        elif 'generated_strategy' in st.session_state and st.session_state.generated_strategy is not None:
            strategy = st.session_state.generated_strategy
            real_data = st.session_state.real_time_market_data
            capital = st.session_state.capital

            regime_class = {
                "LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"
            }.get(strategy["Regime"], "regime-low")

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

            if strategy["Behavior_Warnings"]:
                 for warning in strategy["Behavior_Warnings"]:
                      st.warning(f"⚠️ Behavioral Warning: {warning}")

            st.markdown("---")
            st.subheader("Ready to Trade?")

            if st.button("📝 Prepare Orders for this Strategy"):
                 st.session_state.prepared_orders = prepare_trade_orders(strategy, real_data, capital)

            if st.session_state.prepared_orders:
                 st.markdown("### Proposed Order Details:")
                 st.warning("REVIEW THESE ORDERS CAREFULLY BEFORE PLACING!")

                 orders_df = pd.DataFrame(st.session_state.prepared_orders)
                 orders_display_cols = ['Leg_Type', 'Strike', 'Expiry', 'Quantity_Lots', 'Quantity_Units', 'Proposed_Price', 'Last_Price_API', 'ScripCode']
                 st.dataframe(orders_df[orders_display_cols], use_container_width=True)

                 st.markdown("---")

                 if st.button("✅ Confirm and Place Orders"):
                     success = execute_trade_orders(st.session_state.client, st.session_state.prepared_orders)


                 else:
                      st.info("Click 'Confirm and Place Orders' to send the orders to the broker.")

            elif 'generated_strategy' in st.session_state and st.session_state.generated_strategy is not None:
                 st.info("Click 'Prepare Orders for this Strategy' to see the order details before trading.")


        else:
            st.info("Run the analysis to generate a trading strategy.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Portfolio Tab ---
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💰 Portfolio Overview")

        # Fetch and display portfolio summary metrics
        # Pass client and capital from session state
        portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current P&L (Today/Holding)", f"₹{portfolio_summary['weekly_pnl']:,.2f}")
        with col2:
            st.metric("Margin Used", f"₹{portfolio_summary['margin_used']:,.2f}")
        with col3:
            st.metric("Exposure", f"{portfolio_summary['exposure']:.2f}%")

        st.markdown("---")

        # Display detailed portfolio data fetched from API if available
        if st.session_state.api_portfolio_data:
            st.subheader("Comprehensive Account Data (from 5paisa API)")

            # Holdings
            with st.expander("📂 Holdings"):
                 holdings_data = st.session_state.api_portfolio_data.get("holdings")
                 if holdings_data and isinstance(holdings_data, list):
                      holdings_df = pd.DataFrame(holdings_data)
                      st.dataframe(holdings_df, use_container_width=True)
                 else:
                      st.info("No holdings found or could not fetch.")

            # Margin
            with st.expander("💲 Margin Details"):
                 margin_data = st.session_state.api_portfolio_data.get("margin")
                 if margin_data and isinstance(margin_data, dict):
                      for key, value in margin_data.items():
                           st.write(f"**{key}**: {value}")
                 else:
                      st.info("No margin data found or could not fetch.")

            # Positions
            with st.expander("💹 Open Positions"):
                 positions_data = st.session_state.api_portfolio_data.get("positions")
                 if positions_data and isinstance(positions_data, list):
                      positions_df = pd.DataFrame(positions_data)
                      st.dataframe(positions_df, use_container_width=True)
                 else:
                      st.info("No open positions found or could not fetch.")

            # Order Book (Open Orders)
            with st.expander("📋 Order Book"):
                 order_book_data = st.session_state.api_portfolio_data.get("order_book")
                 if order_book_data and isinstance(order_book_data, list):
                      order_book_df = pd.DataFrame(order_book_data)
                      st.dataframe(order_book_df, use_container_width=True)
                 else:
                      st.info("No open orders found or could not fetch.")

            # Trade Book (Executed Trades)
            with st.expander("📜 Trade Book"):
                 trade_book_data = st.session_state.api_portfolio_data.get("trade_book")
                 if trade_book_data and isinstance(trade_book_data, list):
                      trade_book_df = pd.DataFrame(trade_book_data)
                      st.dataframe(trade_book_df, use_container_width=True)
                 else:
                      st.info("No executed trades found or could not fetch.")

            # Market Status
            with st.expander("📰 Market Status"):
                 market_status_data = st.session_state.api_portfolio_data.get("market_status")
                 if market_status_data:
                      st.json(market_status_data)
                 else:
                      st.info("Market status not available or could not fetch.")


        else:
             st.info("Connect to 5paisa and run analysis to fetch detailed portfolio data.")

        st.markdown('</div>', unsafe_allow_html=True)


    # --- Journal Tab ---
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Discipline Hub")
        with st.form(key="journal_form"):
            st.markdown("Reflect on your trading decisions and build discipline.")
            reason_strategy = st.selectbox("Why did you choose this strategy?", ["High IV", "Low Risk", "Event Opportunity", "Bullish Bias", "Bearish Bias", "Range Bound Expectation", "Expiry Play", "Other"])
            override_risk = st.radio("Did you override any system risk flags?", ("No", "Yes"), index=0)
            expected_outcome = st.text_area("Describe your trade plan, entry/exit criteria, and expected outcome.")
            lessons_learned = st.text_area("After the trade, what were the lessons learned (optional, for review)?")

            submit_journal = st.form_submit_button("💾 Save Journal Entry")

            if submit_journal:
                score = 0
                if override_risk == "No":
                    score += 3
                if reason_strategy != "Other":
                    score += 2
                if expected_outcome:
                    score += 3
                if lessons_learned:
                    score += 1
                latest_portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
                if latest_portfolio_summary['weekly_pnl'] > 0:
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
                journal_df_entry = pd.DataFrame([journal_entry])

                journal_file = "journal_log.csv"
                try:
                    journal_df_entry.to_csv(journal_file, mode='a', header=not os.path.exists(journal_file), index=False)
                    st.success(f"✅ Journal Entry Saved! Discipline Score: {score}/10")
                    logger.info(f"Journal entry saved. Score: {score}")

                    if score >= 7 and st.session_state.violations > 0:
                         st.session_state.violations = 0
                         st.session_state.journal_complete = True
                         st.success("🔓 Discipline Lock Removed! Keep up the good work.")
                         logger.info("Discipline lock removed.")

                except PermissionError:
                    logger.error("Permission denied when writing to journal_log.csv")
                    st.error("❌ Cannot save journal_log.csv: Permission denied")
                except Exception as e:
                    logger.error(f"Error saving journal entry: {e}")
                    st.error(f"❌ Error saving journal entry: {e}")


        st.markdown("### Past Entries")
        journal_file = "journal_log.csv"
        if os.path.exists(journal_file):
            try:
                journal_df = pd.read_csv(journal_file)
                if 'Date' in journal_df.columns:
                    journal_df['Date'] = pd.to_datetime(journal_df['Date']).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(journal_df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error reading journal_log.csv: {e}")
        else:
            st.info("No journal entries found yet.")
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Backtest Tab ---
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📉 Backtest Results")
        if st.session_state.backtest_run and st.session_state.backtest_results is not None:
            results = st.session_state.backtest_results
            # Check if the cumulative PnL chart data is available and not empty
            if st.session_state.backtest_cumulative_pnl_chart_data is not None and not st.session_state.backtest_cumulative_pnl_chart_data.empty:
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
                # Use the dedicated cumulative PnL chart data for plotting
                st.line_chart(st.session_state.backtest_cumulative_pnl_chart_data, color="#e94560")

                st.markdown("### Performance by Strategy")
                if not results["strategy_perf"].empty:
                     strategy_perf_formatted = results["strategy_perf"].style.format({
                         "sum": "₹{:,.2f}",
                         "mean": "₹{:,.2f}",
                         "Win_Rate": "{:.2%}"
                     })
                     st.dataframe(strategy_perf_formatted, use_container_width=True)
                else:
                     st.info("No strategy performance data available.")


                st.markdown("### Performance by Regime")
                if not results["regime_perf"].empty:
                     regime_perf_formatted = results["regime_perf"].style.format({
                         "sum": "₹{:,.2f}",
                         "mean": "₹{:,.2f}",
                         "Win_Rate": "{:.2%}"
                     })
                     st.dataframe(regime_perf_formatted, use_container_width=True)
                else:
                     st.info("No regime performance data available.")


                st.markdown("### Detailed Backtest Trades")
                # The backtest_df now contains ENTRY, DAILY_PNL, and EXIT events
                if not results["backtest_df"].empty:
                     detailed_trades_formatted = results["backtest_df"].style.format({
                         "PnL": "₹{:,.2f}",
                         "Cumulative_PnL": "₹{:,.2f}",
                         "Capital_Deployed": "₹{:,.2f}",
                         "Max_Loss": "₹{:,.2f}",
                         "Risk_Reward": "{:.2f}"
                     })
                     st.dataframe(detailed_trades_formatted, use_container_width=True)
                else:
                     st.info("No detailed backtest trade data available.")

            else:
                 # Display a warning if no cumulative PnL data is available
                 st.warning("No backtest results generated for the selected parameters.")

        else:
            st.info("Run the analysis to view backtest results.")
        st.markdown('</div>', unsafe_allow_html=True)


    st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True)
