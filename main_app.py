import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, time
import logging
from pathlib import Path
import html
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Assuming these modules are available
try:
    from smartbhai_gpt import SmartBhaiGPT
    from upstox_api import (
        initialize_upstox_client,
        fetch_all_api_portfolio_data,
        prepare_trade_orders,
        execute_trade_orders,
        square_off_positions,
        fetch_market_depth_by_scrip,
    )
    from data_processing import load_data, generate_features, FEATURE_COLS
    from volatility_forecasting import forecast_volatility_future
    from backtesting import run_backtest
    from strategy_generation import generate_trading_strategy
except ImportError as e:
    st.error(f"âŒ Missing required module: {e}. Please check your dependencies.")
    logging.error(f"Module import failed: {e}")
    st.stop()

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "volguard.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# === Streamlit Page Configuration ===
st.set_page_config(page_title="VolGuard Pro", page_icon="ðŸ›¡ï¸", layout="wide")

# === Custom CSS Styling ===
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(135deg, #1a1a2e, #0f1c2e);
            color: #e5e5e5;
            font-family: 'Inter', sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: #16213e;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        .stTabs [data-baseweb="tab"] {
            color: #a0a0a0;
            font-weight: 500;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #e94560;
            color: white;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #2a2a4a;
            color: white;
        }
        .sidebar .stButton>button {
            width: 100%;
            background: #0f3460;
            color: white;
            border-radius: 10px;
            padding: 12px;
            margin: 5px 0;
        }
        .sidebar .stButton>button:hover {
            transform: scale(1.05);
            background: #e94560;
        }
        .card {
            background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9));
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .stMetric {
            background: rgba(15, 52, 96, 0.7);
            border-radius: 15px;
            padding: 15px;
            text-align: center;
        }
        .regime-badge {
            padding: 8px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
        }
        .regime-low {
            background: #28a745;
            color: white;
        }
        .regime-medium {
            background: #ffc107;
            color: black;
        }
        .regime-high {
            background: #dc3545;
            color: white;
        }
        .regime-event {
            background: #ff6f61;
            color: white;
        }
        .alert-banner {
            background: #dc3545;
            color: white;
            padding: 15px;
            border-radius: 10px;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .stButton>button {
            background: #e94560;
            color: white;
            border-radius: 10px;
            padding: 12px 25px;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: #ffcc00;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #a0a0a0;
            font-size: 14px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 30px;
        }
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
            word-wrap: break-word;
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
        @media (max-width: 600px) {
            .stTabs [data-baseweb="tab"] {
                padding: 8px 10px;
                font-size: 14px;
            }
            .card {
                padding: 15px;
                margin: 10px 0;
            }
            .stMetric {
                padding: 10px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# === Initialize Session State ===
default_session_state = {
    "backtest_run": False,
    "backtest_results": None,
    "trades": [],
    "logged_in": False,
    "client": None,
    "real_time_market_data": {},
    "api_portfolio_data": {},
    "prepared_orders": None,
    "analysis_df": None,
    "forecast_log": None,
    "forecast_metrics": None,
    "generated_strategy": None,
    "backtest_cumulative_pnl_chart_data": None,
    "order_placement_errors": [],
    "chat_history": [],
    "capital": 1000000,
    "risk_tolerance": "Moderate",
    "forecast_horizon": 7,
    "backtest_start_date": datetime.now().date() - timedelta(days=365),
    "backtest_end_date": datetime.now().date(),
    "backtest_strategy": "All Strategies",
    "journal_date_input": datetime.now().date(),
    "journal_strategy_input": "",
    "journal_pnl_input": 0.0,
    "journal_notes_input": "",
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# === Initialize SmartBhai GPT ===
smartbhai_gpt = None
try:
    smartbhai_gpt = SmartBhaiGPT(responses_file="responses.csv")
    logger.info("SmartBhai GPT initialized successfully.")
except FileNotFoundError:
    st.sidebar.error("Bhai, SmartBhai GPT responses.csv file not found. Chat will be unavailable.")
    logger.error("SmartBhai GPT initialization failed: responses.csv not found.")
except Exception as e:
    st.sidebar.error(f"Bhai, SmartBhai GPT load nahi hua: {str(e)}. Chat will be unavailable.")
    logger.error(f"SmartBhai GPT initialization failed: {str(e)}", exc_info=True)

# === Helper Functions ===
def is_market_hours():
    """Checks if current time is within standard market hours (9:15 AM to 3:30 PM IST) and it's a weekday."""
    now = datetime.now()
    market_open_time = time(9, 15)
    market_close_time = time(15, 30)
    is_weekday = now.weekday() < 5
    # TODO: Add holiday check using Upstox MarketHolidaysAndTimingsApi
    return is_weekday and (market_open_time <= now.time() <= market_close_time)

@st.cache_data(show_spinner=False)
def fetch_portfolio_data(upstox_client, capital):
    """
    Fetches and summarizes user portfolio data using the Upstox API.
    Args:
        upstox_client (dict): Initialized Upstox API client dictionary.
        capital (float): User's total trading capital.
    Returns:
        dict: Portfolio metrics (weekly_pnl, margin_used, exposure, total_capital).
    """
    portfolio_summary = {
        "weekly_pnl": 0.0,
        "margin_used": 0.0,
        "exposure": 0.0,
        "total_capital": capital,
    }
    if not upstox_client:
        logger.warning("Upstox client not available for fetching portfolio data summary.")
        return portfolio_summary

    try:
        portfolio_data = fetch_all_api_portfolio_data(upstox_client)
        st.session_state.api_portfolio_data = portfolio_data
        margin_data = portfolio_data.get("margin", {}).get("data", {})
        positions_data = portfolio_data.get("positions", {}).get("data", [])

        if isinstance(margin_data, dict):
            portfolio_summary["margin_used"] = pd.to_numeric(
                margin_data.get("utilized_margin"), errors="coerce"
            ).fillna(0.0)
        else:
            logger.warning("Margin data not in expected dictionary format.")
            portfolio_summary["margin_used"] = 0.0

        if isinstance(positions_data, list):
            portfolio_summary["weekly_pnl"] = sum(
                pd.to_numeric(pos.get("unrealized_mtm"), errors="coerce").fillna(0.0)
                + pd.to_numeric(pos.get("realized_profit"), errors="coerce").fillna(0.0)
                for pos in positions_data
                if isinstance(pos, dict)
            )
        else:
            logger.warning("Positions data not in expected list format.")
            portfolio_summary["weekly_pnl"] = 0.0

        capital_numeric = pd.to_numeric(capital, errors="coerce").fillna(0.0)
        portfolio_summary["exposure"] = (
            portfolio_summary["margin_used"] / capital_numeric * 100
            if capital_numeric > 0
            else 0.0
        )
        logger.info("Portfolio data fetched and summarized successfully.")
        return portfolio_summary

    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}", exc_info=True)
        return portfolio_summary

def calculate_position_pnl_with_ltp(upstox_client, positions_data):
    """
    Updates PnL for each position using latest LTP from market depth.
    Args:
        upstox_client (dict): Initialized Upstox API client dictionary.
        positions_data (list): List of position dictionaries.
    Returns:
        list: Updated positions with 'CurrentPnL' and 'LTP' keys.
    """
    if not upstox_client or not positions_data or not isinstance(positions_data, list):
        logger.warning("Invalid inputs for PnL calculation with LTP.")
        return positions_data

    updated_positions = []
    for pos in positions_data:
        if not isinstance(pos, dict):
            updated_positions.append(pos)
            continue

        instrument_key = pos.get("instrument_key")
        buy_avg_price = pd.to_numeric(pos.get("buy_avg_price"), errors="coerce").fillna(0.0)
        sell_avg_price = pd.to_numeric(pos.get("sell_avg_price"), errors="coerce").fillna(0.0)
        quantity = pd.to_numeric(pos.get("quantity"), errors="coerce").fillna(0)

        if instrument_key and quantity != 0:
            try:
                market_data_response = fetch_market_depth_by_scrip(
                    upstox_client, instrument_key=instrument_key
                )
                ltp = (
                    pd.to_numeric(
                        market_data_response.get("Data", [{}])[0].get("LastTradedPrice"),
                        errors="coerce",
                    ).fillna(0.0)
                    if market_data_response
                    else 0.0
                )

                if ltp > 0:
                    if quantity > 0:
                        pos["CurrentPnL"] = quantity * (ltp - buy_avg_price)
                    elif quantity < 0:
                        pos["CurrentPnL"] = abs(quantity) * (sell_avg_price - ltp)
                    pos["LTP"] = ltp
                else:
                    logger.warning(
                        f"LTP invalid for {instrument_key}. Using 'unrealized_mtm'. LTP: {ltp}"
                    )
                    pos["CurrentPnL"] = pd.to_numeric(
                        pos.get("unrealized_mtm"), errors="coerce"
                    ).fillna(0.0)
                    pos["LTP"] = 0.0
            except Exception as e:
                logger.warning(
                    f"Could not fetch LTP for {instrument_key}: {e}. Using 'unrealized_mtm'."
                )
                pos["CurrentPnL"] = pd.to_numeric(
                    pos.get("unrealized_mtm"), errors="coerce"
                ).fillna(0.0)
                pos["LTP"] = 0.0
        else:
            pos["CurrentPnL"] = pd.to_numeric(
                pos.get("unrealized_mtm"), errors="coerce"
            ).fillna(0.0)
            pos["LTP"] = 0.0

        updated_positions.append(pos)
    return updated_positions

# === Streamlit App Layout ===
with st.sidebar:
    st.header("ðŸ”‘ Upstox Login")
    access_token = st.text_input(
        "Access Token", type="password", key="access_token_input"
    )
    if st.button("Login to Upstox", key="login_button"):
        if not access_token:
            st.error("âŒ Access token cannot be empty.")
            logger.error("Login attempted with empty access token.")
        else:
            client_objects = initialize_upstox_client(access_token)
            if client_objects:
                st.session_state.client = client_objects
                st.session_state.logged_in = True
                st.success("âœ… Logged in to Upstox!")
                logger.info("Upstox login successful.")
                st.rerun()
            else:
                st.session_state.logged_in = False
                st.session_state.client = None
                st.error("âŒ Login failed. Invalid or expired access token.")
                logger.error("Upstox login failed: Invalid or expired token.")

    if st.session_state.logged_in:
        st.header("âš™ï¸ Trading Controls")
        capital = st.number_input(
            "Capital (â‚¹)",
            min_value=100000,
            value=st.session_state.capital,
            step=100000,
            format="%d",
            key="capital_input",
        )
        st.session_state.capital = capital
        risk_options = ["Conservative", "Moderate", "Aggressive"]
        risk_tolerance = st.selectbox(
            "Risk Profile",
            risk_options,
            index=risk_options.index(st.session_state.risk_tolerance)
            if st.session_state.risk_tolerance in risk_options
            else 1,
            key="risk_tolerance_input",
        )
        st.session_state.risk_tolerance = risk_tolerance
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            1,
            30,
            st.session_state.forecast_horizon,
            key="forecast_horizon_input",
        )
        st.session_state.forecast_horizon = forecast_horizon

        st.markdown("---")
        st.markdown("**Backtest Parameters**")
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.backtest_start_date,
            key="backtest_start_date_input",
        )
        st.session_state.backtest_start_date = start_date
        end_date = st.date_input(
            "End Date",
            value=st.session_state.backtest_end_date,
            key="backtest_end_date_input",
        )
        st.session_state.backtest_end_date = end_date
        strategy_options = [
            "All Strategies",
            "Short Straddle",
            "Short Strangle",
            "Iron Condor",
            "Butterfly Spread",
            "Iron Fly",
            "Calendar Spread",
            "Jade Lizard",
            "Short Put Vertical Spread",
            "Short Call Vertical Spread",
            "Short Put",
            "Long Put",
        ]
        strategy_choice = st.selectbox(
            "Backtest Strategy Filter",
            strategy_options,
            index=strategy_options.index(st.session_state.backtest_strategy)
            if st.session_state.backtest_strategy in strategy_options
            else 0,
            key="backtest_strategy_filter_input",
        )
        st.session_state.backtest_strategy = strategy_choice
        if st.button("Run Backtest", key="run_backtest_button"):
            st.session_state.backtest_run = True
            st.session_state.backtest_results = None
            st.session_state.backtest_cumulative_pnl_chart_data = None
            st.info("Backtest started! Check the Backtest tab for results.")
            logger.info("Backtest initiated from sidebar.")
            st.rerun()

        st.markdown("---")
        if st.button("Square Off All Positions", key="square_off_button"):
            if st.session_state.logged_in and st.session_state.client:
                if is_market_hours():
                    st.info("Attempting to square off all open positions...")
                    success = square_off_positions(st.session_state.client)
                    if success:
                        st.success("âœ… Square off process initiated successfully!")
                        logger.info("Square off process initiated.")
                    else:
                        st.error("âŒ Failed to initiate square off process.")
                        logger.error("Failed to initiate square off process.")
                else:
                    st.warning("â° Market is closed. Cannot square off positions.")
                    logger.warning("Square off attempted outside market hours.")
            else:
                st.error("Not logged in to Upstox.")
                logger.warning("Square off attempted without login.")

# === Main App ===
st.title("ðŸ›¡ï¸ VolGuard Pro")
st.markdown("**Your AI-powered options trading cockpit for NIFTY 50**")

if not is_market_hours():
    st.warning(
        "âš ï¸ Outside standard market hours (9:15 AMâ€“3:30 PM IST, Mon-Fri). Live trading actions may be limited."
    )
    logger.info("App running outside standard market hours.")

# === Data Loading ===
data_load_success = False
data_source_tag = "Unknown"
logger.info("Starting data loading process in Streamlit app.")

try:
    client_for_load = st.session_state.client if st.session_state.logged_in else None
    df_loaded, real_data_loaded, data_source_tag = load_data(client_for_load)
    if df_loaded is not None and not df_loaded.empty:
        logger.info(f"Raw data loaded successfully from {data_source_tag}. Generating features.")
        st.session_state.real_time_market_data = real_data_loaded if real_data_loaded else {}
        analysis_df = generate_features(
            df_loaded.copy(), st.session_state.real_time_market_data, st.session_state.capital
        )
        if analysis_df is not None and not analysis_df.empty:
            st.session_state.analysis_df = analysis_df
            data_load_success = True
            logger.info("Features generated successfully.")
        else:
            st.error("âŒ Failed to generate features from loaded data.")
            logger.error("Feature generation failed during app load.")
            st.session_state.analysis_df = None
    else:
        logger.error(
            f"Data loading failed: load_data returned None or empty DataFrame from {data_source_tag}."
        )
        st.error(f"âŒ Could not load necessary data from {data_source_tag}.")
        st.session_state.analysis_df = None
        st.session_state.real_time_market_data = real_data_loaded if real_data_loaded else {}
except Exception as e:
    logger.critical(f"Critical error during data loading: {str(e)}", exc_info=True)
    st.error(f"âŒ A critical error occurred during data loading: {str(e)}.")
    data_load_success = False
    st.session_state.analysis_df = None
    st.session_state.real_time_market_data = {}

if not data_load_success and not st.session_state.real_time_market_data:
    st.error("âŒ VolGuard Pro cannot function without essential market data.")
    st.stop()
elif not data_load_success:
    st.warning("âš ï¸ Historical data or feature generation failed. Some sections may not be available.")
    logger.warning("App continuing with only real-time data.")

# === Tabs ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "ðŸ“Š Market Snapshot",
        "ðŸ”® Volatility Forecast",
        "ðŸ¤– Trading Strategy",
        "ðŸ’¼ Portfolio",
        "ðŸ“ Journal",
        "ðŸ“ˆ Backtest",
        "âš ï¸ Risk Dashboard",
        "ðŸ’¬ SmartBhai",
    ]
)

with tab1:
    st.header("ðŸ“Š Market Snapshot")
    if st.session_state.real_time_market_data:
        market_data = st.session_state.real_time_market_data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            nifty_spot = market_data.get("nifty_spot", "N/A")
            st.metric(
                "NIFTY Spot",
                f"{nifty_spot:.2f}" if isinstance(nifty_spot, (int, float)) else str(nifty_spot),
            )
        with col2:
            vix = market_data.get("vix", "N/A")
            st.metric(
                "India VIX",
                f"{vix:.2f}" if isinstance(vix, (int, float)) else str(vix),
            )
        with col3:
            pcr = market_data.get("pcr", "N/A")
            st.metric(
                "PCR",
                f"{pcr:.2f}" if isinstance(pcr, (int, float)) else str(pcr),
            )
        with col4:
            straddle_price = market_data.get("straddle_price", "N/A")
            st.metric(
                "ATM Straddle",
                f"{straddle_price:.2f}"
                if isinstance(straddle_price, (int, float))
                else str(straddle_price),
            )

        col5, col6, col7 = st.columns(3)
        with col5:
            atm_strike = market_data.get("atm_strike", "N/A")
            st.metric(
                "ATM Strike",
                f"{atm_strike:.2f}" if isinstance(atm_strike, (int, float)) else str(atm_strike),
            )
        with col6:
            max_pain_strike = market_data.get("max_pain_strike", "N/A")
            st.metric(
                "Max Pain",
                f"{max_pain_strike:.2f}"
                if isinstance(max_pain_strike, (int, float))
                else str(max_pain_strike),
            )
        with col7:
            max_pain_diff_pct = market_data.get("max_pain_diff_pct", "N/A")
            st.metric(
                "Max Pain Diff %",
                f"{max_pain_diff_pct:.2f}%"
                if isinstance(max_pain_diff_pct, (int, float))
                else str(max_pain_diff_pct),
            )

        st.subheader("Option Chain Preview")
        option_chain_df_preview = market_data.get("option_chain")
        if (
            option_chain_df_preview is not None
            and isinstance(option_chain_df_preview, pd.DataFrame)
            and not option_chain_df_preview.empty
        ):
            display_cols = [
                "StrikeRate",
                "CPType",
                "LastRate",
                "IV",
                "OpenInterest",
                "Volume",
                "ScripCode",
            ]
            display_cols = [col for col in display_cols if col in option_chain_df_preview.columns]
            st.dataframe(option_chain_df_preview[display_cols].head(20))
        else:
            st.info("Option chain data not available in real-time data.")
            logger.warning("Option chain data missing in real_time_market_data.")
    else:
        st.info("No real-time market data available. Please check login status.")
        logger.warning("real_time_market_data unavailable for Market Snapshot.")
    st.markdown(f"**Data Source**: {data_source_tag}")

with tab2:
    st.header("ðŸ”® Volatility Forecast")
    if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
        try:
            logger.info("Running volatility forecast.")
            forecast_log_df, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = (
                forecast_volatility_future(
                    st.session_state.analysis_df.copy(), st.session_state.forecast_horizon
                )
            )
            st.session_state.forecast_log = forecast_log_df
            st.session_state.forecast_metrics = {
                "forecasted_vix": blended_vols[-1] if blended_vols is not None and len(blended_vols) > 0 else None,
                "vix_range_low": np.min(blended_vols) if blended_vols is not None and len(blended_vols) > 0 else None,
                "vix_range_high": np.max(blended_vols) if blended_vols is not None and len(blended_vols) > 0 else None,
                "confidence": confidence_score,
                "rmse": rmse,
                "feature_importances": feature_importances,
                "realized_vol": realized_vol,
            }
            logger.info("Volatility forecast completed successfully.")

            metrics = st.session_state.forecast_metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                forecasted_vix = metrics.get("forecasted_vix", "N/A")
                st.metric(
                    "Forecasted VIX",
                    f"{forecasted_vix:.2f}"
                    if isinstance(forecasted_vix, (int, float))
                    else str(forecasted_vix),
                )
            with col2:
                vix_range_low = metrics.get("vix_range_low", "N/A")
                st.metric(
                    "VIX Range Low",
                    f"{vix_range_low:.2f}"
                    if isinstance(vix_range_low, (int, float))
                    else str(vix_range_low),
                )
            with col3:
                vix_range_high = metrics.get("vix_range_high", "N/A")
                st.metric(
                    "VIX Range High",
                    f"{vix_range_high:.2f}"
                    if isinstance(vix_range_high, (int, float))
                    else str(vix_range_high),
                )
            with col4:
                confidence = metrics.get("confidence", "N/A")
                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                    if isinstance(confidence, (int, float))
                    else str(confidence),
                )

            st.subheader("Forecast Trend (Blended)")
            if (
                st.session_state.forecast_log is not None
                and not st.session_state.forecast_log.empty
                and "Blended_Vol" in st.session_state.forecast_log.columns
            ):
                try:
                    historical_vix = st.session_state.analysis_df["VIX"].tail(30).copy()
                    forecast_chart_data = st.session_state.forecast_log.set_index("Date")[
                        "Blended_Vol"
                    ].copy()
                    combined_vols = pd.concat([historical_vix, forecast_chart_data]).rename(
                        "VIX_Combined"
                    )
                    st.line_chart(combined_vols)
                except Exception as chart_e:
                    st.warning(f"Could not generate forecast trend chart: {chart_e}.")
                    logger.error(f"Error generating forecast trend chart: {chart_e}")
            else:
                st.info("Forecast data not available for charting.")
                logger.warning("Forecast log data missing for charting.")

            st.subheader("Model Metrics")
            if metrics.get("rmse") is not None:
                st.write(f"XGBoost RMSE: {metrics['rmse']:.2f}")
            if metrics.get("realized_vol") is not None:
                st.write(f"Recent Realized Volatility: {metrics['realized_vol']:.2f}")

            st.subheader("Feature Importances (XGBoost)")
            if metrics.get("feature_importances") is not None and st.session_state.analysis_df is not None:
                try:
                    feature_importances_df = pd.DataFrame(
                        {
                            "Feature": FEATURE_COLS,
                            "Importance": metrics["feature_importances"],
                        }
                    ).sort_values("Importance", ascending=False)
                    st.bar_chart(feature_importances_df.set_index("Feature"))
                except Exception as fi_e:
                    st.warning(f"Could not display feature importances: {fi_e}.")
                    logger.error(f"Error displaying feature importances: {fi_e}")
            else:
                st.info("Feature importances data not available.")
                logger.warning("Feature importances data missing.")

        except Exception as e:
            st.error(f"âŒ Error generating volatility forecast: {str(e)}.")
            logger.error(f"Volatility forecast error: {str(e)}.", exc_info=True)
            st.session_state.forecast_log = None
            st.session_state.forecast_metrics = None
    else:
        st.info("No analysis data available for volatility forecasting.")
        logger.warning("Analysis DataFrame not available for volatility forecasting.")

with tab3:
    st.header("ðŸ¤– Trading Strategy")
    if (
        st.session_state.analysis_df is not None
        and not st.session_state.analysis_df.empty
        and st.session_state.real_time_market_data
        and st.session_state.forecast_metrics is not None
    ):
        try:
            logger.info("Generating trading strategy.")
            strategy = generate_trading_strategy(
                st.session_state.analysis_df.copy(),
                st.session_state.real_time_market_data.copy(),
                st.session_state.forecast_metrics.copy(),
                st.session_state.capital,
                st.session_state.risk_tolerance,
            )
            st.session_state.generated_strategy = strategy

            if strategy and isinstance(strategy, dict):
                st.subheader("Recommended Strategy")
                st.markdown(f"**Strategy**: {strategy.get('Strategy', 'N/A')}")
                st.markdown(f"**Confidence**: {strategy.get('Confidence', 0.0):.2%}")
                deploy_amount = strategy.get("Deploy", 0.0)
                st.markdown(
                    f"**Deploy Amount**: â‚¹{deploy_amount:.2f}"
                    if isinstance(deploy_amount, (int, float))
                    else f"Deploy Amount: {deploy_amount}"
                )
                st.markdown(f"**Reasoning**: {strategy.get('Reasoning', 'N/A')}")

                st.subheader("Order Actions")
                if st.button("Prepare Orders", key="prepare_orders_button"):
                    if is_market_hours():
                        st.info("Preparing orders based on the recommended strategy...")
                        prepared_orders = prepare_trade_orders(
                            st.session_state.generated_strategy.copy(),
                            st.session_state.real_time_market_data.copy(),
                            st.session_state.capital,
                        )
                        st.session_state.prepared_orders = prepared_orders
                        if prepared_orders and isinstance(prepared_orders, list) and len(prepared_orders) > 0:
                            st.success(f"âœ… {len(prepared_orders)} Orders prepared successfully!")
                            st.subheader("Prepared Orders Preview")
                            st.dataframe(pd.DataFrame(prepared_orders))
                            logger.info(f"{len(prepared_orders)} Orders prepared successfully.")
                        else:
                            st.warning("âš ï¸ Failed to prepare orders.")
                            logger.warning("Order preparation failed.")
                            st.session_state.prepared_orders = None
                    else:
                        st.warning("â° Market is closed. Cannot prepare live orders.")
                        st.session_state.prepared_orders = None

                if (
                    st.session_state.prepared_orders is not None
                    and st.session_state.logged_in
                    and st.session_state.client
                ):
                    if is_market_hours():
                        with st.form("execute_orders_form", clear_on_submit=False):
                            st.subheader("Ready to Execute")
                            st.warning("ðŸš¨ Clicking 'Execute Orders' will place REAL orders.")
                            confirm_execution = st.checkbox(
                                "Confirm: I understand this will place live orders."
                            )
                            execute_button_clicked = st.form_submit_button("Execute Orders")
                            if execute_button_clicked and confirm_execution:
                                st.info("Attempting to execute orders...")
                                success, response_details = execute_trade_orders(
                                    st.session_state.client,
                                    st.session_state.prepared_orders.copy(),
                                )
                                if success:
                                    st.success("âœ… Orders executed successfully!")
                                    logger.info("Orders executed successfully via API.")
                                    st.session_state.trades.extend(st.session_state.prepared_orders)
                                    st.session_state.prepared_orders = None
                                    st.session_state.order_placement_errors.append(response_details)
                                else:
                                    st.error("âŒ Order execution failed.")
                                    logger.error(f"Order execution failed: {response_details}")
                                    st.session_state.order_placement_errors.append(response_details)
                                st.subheader("Execution Response Details")
                                st.json(response_details)
                            elif execute_button_clicked:
                                st.error("âŒ Please confirm before executing orders.")
                    else:
                        st.warning("â° Market is closed. Cannot execute live orders.")
                elif st.session_state.prepared_orders is not None:
                    st.info("Please log in to Upstox to execute prepared orders.")

                if st.session_state.order_placement_errors:
                    st.subheader("Recent Order Placement Errors")
                    for error in st.session_state.order_placement_errors[-3:]:  # Limit to last 3
                        st.json(error)
            else:
                st.info("âš ï¸ No strategy generated based on current conditions.")
                st.session_state.generated_strategy = None
        except Exception as e:
            st.error(f"âŒ Error generating strategy: {str(e)}.")
            logger.error(f"Trading strategy generation error: {str(e)}.", exc_info=True)
            st.session_state.generated_strategy = None
            st.session_state.prepared_orders = None
    else:
        st.info("No data available for strategy generation.")
        logger.warning("Essential data missing for trading strategy generation.")

with tab4:
    st.header("ðŸ’¼ Portfolio")
    if st.session_state.logged_in and st.session_state.client:
        logger.info("Fetching portfolio data for display.")
        portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
        col1, col2, col3 = st.columns(3)
        with col1:
            weekly_pnl = portfolio_summary.get("weekly_pnl", 0.0)
            st.metric("Weekly PnL", f"â‚¹{weekly_pnl:.2f}")
        with col2:
            margin_used = portfolio_summary.get("margin_used", 0.0)
            st.metric("Margin Used", f"â‚¹{margin_used:.2f}")
        with col3:
            exposure = portfolio_summary.get("exposure", 0.0)
            st.metric("Exposure", f"{exposure:.2f}%")

        st.subheader("Open Positions")
        positions_data_raw = st.session_state.api_portfolio_data.get("positions", {}).get(
            "data", []
        )
        positions_data_updated = calculate_position_pnl_with_ltp(
            st.session_state.client, positions_data_raw
        )
        if positions_data_updated and isinstance(positions_data_updated, list):
            positions_df = pd.DataFrame(positions_data_updated)
            if not positions_df.empty:
                display_cols = [
                    "instrument_key",
                    "quantity",
                    "buy_avg_price",
                    "sell_avg_price",
                    "LTP",
                    "CurrentPnL",
                    "unrealized_mtm",
                    "realized_profit",
                    "product",
                ]
                display_cols = [col for col in display_cols if col in positions_df.columns]
                st.dataframe(positions_df[display_cols])
            else:
                st.info("No open positions.")
        else:
            st.info("No open positions or position data unavailable.")
            logger.warning("Position data missing for display.")

        st.subheader("Fund Summary")
        fund_data = st.session_state.api_portfolio_data.get("margin", {}).get("data", {})
        if fund_data and isinstance(fund_data, dict):
            fund_keys = [
                "available_margin",
                "utilized_margin",
                "total_margin",
                "payin_amount",
                "payout_amount",
            ]
            fund_display_data = {key: fund_data.get(key, "N/A") for key in fund_keys}
            if fund_display_data:
                fund_df = pd.DataFrame([fund_display_data])
                for col in fund_df.columns:
                    if pd.api.types.is_numeric_dtype(fund_df[col]):
                        fund_df[col] = pd.to_numeric(fund_df[col], errors="coerce")
                        fund_df[col] = fund_df[col].apply(
                            lambda x: f"â‚¹{x:.2f}" if pd.notna(x) else "N/A"
                        )
                st.dataframe(fund_df)
            else:
                st.info("Fund data not available.")
                logger.warning("Fund data missing in API portfolio data.")
        else:
            st.info("Fund data not available.")
            logger.warning("Fund data missing.")

        st.subheader("Order Book")
        order_data = st.session_state.api_portfolio_data.get("order_book", {}).get("data", [])
        if order_data and isinstance(order_data, list):
            order_df = pd.DataFrame(order_data)
            if not order_df.empty:
                order_display_cols = [
                    "instrument_key",
                    "quantity",
                    "transaction_type",
                    "order_type",
                    "status",
                    "price",
                    "average_price",
                    "placed_by",
                    "order_id",
                    "exchange_timestamp",
                ]
                order_display_cols = [col for col in order_display_cols if col in order_df.columns]
                st.dataframe(order_df[order_display_cols])
            else:
                st.info("Order book data is empty.")
        else:
            st.info("Order book data not available.")
            logger.warning("Order book data missing.")

        st.subheader("Trade History")
        trade_data = st.session_state.api_portfolio_data.get("trade_book", {}).get("data", [])
        if trade_data and isinstance(trade_data, list):
            trade_df = pd.DataFrame(trade_data)
            if not trade_df.empty:
                trade_display_cols = [
                    "instrument_key",
                    "quantity",
                    "transaction_type",
                    "exchange_timestamp",
                    "price",
                    "order_id",
                    "trade_id",
                ]
                trade_display_cols = [col for col in trade_display_cols if col in trade_df.columns]
                st.dataframe(trade_df[trade_display_cols])
            else:
                st.info("Trade history data is empty.")
        else:
            st.info("Trade history data not available.")
            logger.warning("Trade history data missing.")
    else:
        st.info("Please log in to Upstox to view your portfolio.")
        logger.info("User not logged in, Portfolio tab showing login message.")

with tab5:
    st.header("ðŸ“ Trading Journal")
    journal_file = Path("journal_log.csv")
    try:
        if journal_file.exists():
            journal_df = pd.read_csv(journal_file, encoding="utf-8")
            journal_df["Date"] = pd.to_datetime(journal_df["Date"], errors="coerce")
            journal_df = journal_df.dropna(subset=["Date"])
        else:
            journal_df = pd.DataFrame(columns=["Date", "Strategy", "PnL", "Notes"])
        logger.info(f"Journal loaded successfully. {len(journal_df)} entries found.")
    except Exception as e:
        st.error(f"âŒ Error loading journal file: {str(e)}")
        logger.error(f"Error loading journal file: {str(e)}", exc_info=True)
        journal_df = pd.DataFrame(columns=["Date", "Strategy", "PnL", "Notes"])

    with st.form("journal_form", clear_on_submit=True):
        st.subheader("Log a New Trade")
        date_log = st.date_input(
            "Trade Date",
            value=st.session_state.journal_date_input,
            key="journal_date_input",
        )
        st.session_state.journal_date_input = date_log
        strategy_log = st.text_input(
            "Strategy",
            value=st.session_state.journal_strategy_input,
            key="journal_strategy_input",
        )
        st.session_state.journal_strategy_input = strategy_log
        pnl_log = st.number_input(
            "PnL (â‚¹)",
            format="%.2f",
            value=st.session_state.journal_pnl_input,
            key="journal_pnl_input",
        )
        st.session_state.journal_pnl_input = pnl_log
        notes_log = st.text_area(
            "Notes",
            value=st.session_state.journal_notes_input,
            key="journal_notes_input",
        )
        st.session_state.journal_notes_input = notes_log
        submitted = st.form_submit_button("Log Trade")
        if submitted:
            new_entry = pd.DataFrame(
                {
                    "Date": [date_log],
                    "Strategy": [strategy_log],
                    "PnL": [pnl_log],
                    "Notes": [notes_log],
                }
            )
            new_entry["Date"] = pd.to_datetime(new_entry["Date"])
            journal_df = pd.concat([journal_df, new_entry], ignore_index=True)
            logger.info("New journal entry added to DataFrame.")
            try:
                journal_df.to_csv(journal_file, index=False, encoding="utf-8")
                st.success("âœ… Trade logged successfully!")
                logger.info("Journal updated successfully saved to CSV.")
                st.session_state.journal_strategy_input = ""
                st.session_state.journal_pnl_input = 0.0
                st.session_state.journal_notes_input = ""
            except Exception as e:
                st.error(f"âŒ Error saving journal entry: {str(e)}")
                logger.error(f"Error saving journal entry to CSV: {str(e)}", exc_info=True)

    st.subheader("Trade History")
    if not journal_df.empty:
        if "Date" in journal_df.columns:
            journal_df_display = journal_df.sort_values(by="Date", ascending=False).copy()
            st.dataframe(journal_df_display)
        else:
            st.warning("Journal DataFrame is missing the 'Date' column.")
            st.dataframe(journal_df)
    else:
        st.info("No trades logged yet.")
        logger.info("Journal DataFrame is empty.")

with tab6:
    st.header("ðŸ“ˆ Backtest Results")
    if (
        st.session_state.backtest_run
        and st.session_state.analysis_df is not None
        and not st.session_state.analysis_df.empty
    ):
        st.info("Running backtest simulation...")
        logger.info("Backtest trigger detected. Starting backtest run.")
        try:
            start_date = st.session_state.backtest_start_date
            end_date = st.session_state.backtest_end_date
            analysis_df = st.session_state.analysis_df
            if start_date >= end_date:
                st.error("âŒ Backtest Error: Start date must be before end date.")
                logger.error("Backtest date range invalid: Start date >= End date.")
                st.session_state.backtest_run = False
                st.session_state.backtest_results = None
                st.session_state.backtest_cumulative_pnl_chart_data = None
            elif (
                pd.to_datetime(start_date) < analysis_df.index.min()
                or pd.to_datetime(end_date) > analysis_df.index.max()
            ):
                st.error(
                    f"âŒ Backtest Error: Selected date range is outside available data range ({analysis_df.index.min().date()} to {analysis_df.index.max().date()})."
                )
                logger.error("Backtest date range outside available data.")
                st.session_state.backtest_run = False
                st.session_state.backtest_results = None
                st.session_state.backtest_cumulative_pnl_chart_data = None
            else:
                backtest_results = run_backtest(
                    analysis_df.copy(),
                    start_date,
                    end_date,
                    st.session_state.capital,
                    st.session_state.backtest_strategy,
                )
                st.session_state.backtest_results = backtest_results
                st.session_state.backtest_run = False
                if backtest_results and isinstance(backtest_results, dict):
                    st.success("âœ… Backtest completed!")
                    logger.info("Backtest completed successfully.")
                    st.subheader("Key Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_return = backtest_results.get("total_return", 0.0)
                        st.metric("Total Return", f"{total_return:.2%}")
                    with col2:
                        sharpe_ratio = backtest_results.get("sharpe_ratio", 0.0)
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                    with col3:
                        max_drawdown = backtest_results.get("max_drawdown", 0.0)
                        st.metric("Max Drawdown", f"{max_drawdown:.2%}")

                    st.subheader("Cumulative PnL Chart")
                    cumulative_pnl_data = backtest_results.get("cumulative_pnl")
                    if cumulative_pnl_data is not None and isinstance(cumulative_pnl_data, pd.Series):
                        st.session_state.backtest_cumulative_pnl_chart_data = cumulative_pnl_data
                        st.line_chart(cumulative_pnl_data)
                    else:
                        st.warning("Cumulative PnL data not available.")
                        logger.warning("Cumulative PnL data missing.")

                    st.subheader("Trade Log Preview")
                    trade_log_df = backtest_results.get("trade_log")
                    if (
                        trade_log_df is not None
                        and isinstance(trade_log_df, pd.DataFrame)
                        and not trade_log_df.empty
                    ):
                        st.dataframe(trade_log_df.head())
                    else:
                        st.info("Backtest trade log is empty.")
                        logger.warning("Backtest trade log data missing.")
                else:
                    st.error("âŒ Backtest failed to return results.")
                    logger.error("Backtest failed to return results.")
                    st.session_state.backtest_results = None
                    st.session_state.backtest_cumulative_pnl_chart_data = None
        except Exception as e:
            st.error(f"âŒ An error occurred while running the backtest: {str(e)}.")
            logger.error(f"Backtest execution error: {str(e)}.", exc_info=True)
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.backtest_cumulative_pnl_chart_data = None
    elif st.session_state.backtest_results is not None:
        logger.info("Displaying existing backtest results.")
        metrics = st.session_state.backtest_results
        st.subheader("Key Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_return = metrics.get("total_return", 0.0)
            st.metric("Total Return", f"{total_return:.2%}")
        with col2:
            sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col3:
            max_drawdown = metrics.get("max_drawdown", 0.0)
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")

        st.subheader("Cumulative PnL Chart")
        if st.session_state.backtest_cumulative_pnl_chart_data is not None:
            st.line_chart(st.session_state.backtest_cumulative_pnl_chart_data)
        else:
            st.warning("Cumulative PnL data not available for charting.")
            logger.warning("Stored Cumulative PnL data not available.")

        st.subheader("Trade Log Preview")
        trade_log_df = metrics.get("trade_log")
        if (
            trade_log_df is not None
            and isinstance(trade_log_df, pd.DataFrame)
            and not trade_log_df.empty
        ):
            st.dataframe(trade_log_df.head())
        else:
            st.info("Backtest trade log is empty.")
            logger.warning("Stored backtest trade log data missing.")
    else:
        st.info("Run a backtest from the sidebar to see results here.")
        logger.info("Backtest tab: No backtest results available.")


with tab7:
    st.header("ðŸ“ˆ Option Analytics")
    client = st.session_state.client if st.session_state.logged_in else None

    if not client:
        st.warning("Please login to Upstox to fetch option analytics.")
    else:
        st.info("Fetching latest option chain and market data...")
        try:
            from upstox_api import fetch_real_time_market_data
            data = fetch_real_time_market_data(client)
            if not data:
                st.error("âŒ Failed to fetch option data.")
            else:
                st.success(f"Last updated: {data['timestamp']}")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("NIFTY Spot", f"{data['nifty_spot']:.2f}")
                col2.metric("India VIX", f"{data['vix']:.2f}" if data['vix'] else "N/A")
                col3.metric("PCR", f"{data['pcr']:.2f}")
                col4.metric("ATM Straddle", f"{data['straddle_price']:.2f}")

                col5, col6, col7 = st.columns(3)
                col5.metric("ATM Strike", f"{data['atm_strike']}")
                col6.metric("Max Pain", f"{data['max_pain_strike']}")
                col7.metric("Expiry", data["expiry"])

                st.subheader("Market Depth (ATM Strike)")
                ce = data["ce_depth"]
                pe = data["pe_depth"]
                st.markdown(f"**Call Bid/Ask**: {ce['bid_volume']} / {ce['ask_volume']} | **Put Bid/Ask**: {pe['bid_volume']} / {pe['ask_volume']}")

                df = data["option_chain"]
                atm_idx = df[df["Strike"] == data["atm_strike"]].index[0]
                subset = df.iloc[max(0, atm_idx - 6): atm_idx + 7]

                st.subheader("ATM Â± 6 Strikes")
                display_cols = [
                    "Strike", "CE_LTP", "CE_IV", "CE_OI", "CE_Volume",
                    "PE_LTP", "PE_IV", "PE_OI", "PE_Volume", "Strike_PCR"
                ]
                st.dataframe(subset[display_cols].reset_index(drop=True), use_container_width=True)

                st.subheader("Full Option Chain (Preview)")
                st.dataframe(df[display_cols].head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying analytics: {e}")


with tab8:
    st.header("ðŸ’¬ SmartBhai GPT")
    st.markdown("<div class='smartbhai-container'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='smartbhai-title'>Ask about IV, strategies, or market buzz!</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='smartbhai-subtitle'>Bhai ko pucho market ka haal!</div>",
        unsafe_allow_html=True,
    )

    if smartbhai_gpt:
        logger.info("SmartBhai GPT is available. Rendering chat interface.")
        with st.container():
            st.markdown("<div class='smartbhai-chat'>", unsafe_allow_html=True)
            for chat in st.session_state.chat_history:
                if isinstance(chat, dict) and "role" in chat and "message" in chat:
                    escaped_message = html.escape(chat["message"])
                    if chat["role"] == "user":
                        st.markdown(
                            f"<div class='chat-bubble user-bubble'>{escaped_message}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div class='chat-bubble smartbhai-bubble'>{escaped_message}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    logger.warning(f"Skipping invalid chat history entry: {chat}")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.form("smartbhai_form", clear_on_submit=True):
                query = st.text_input(
                    "Ask SmartBhai:",
                    key="smartbhai_query_input",
                    placeholder="Bhai, IV kya hai? Strategy kya lagau?",
                    label_visibility="collapsed",
                )
                submit_button = st.form_submit_button("Send", use_container_width=True)
                if submit_button and query:
                    st.session_state.chat_history.append({"role": "user", "message": query})
                    try:
                        response = smartbhai_gpt.generate_response(query)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "message": response}
                        )
                        logger.info(f"SmartBhai response for query: '{query[:50]}...'")
                    except Exception as e:
                        logger.error(f"Error generating SmartBhai response: {str(e)}", exc_info=True)
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "message": "Sorry, Bhai, kuch technical issue hai. Try again! ðŸ™",
                            }
                        )
                    st.rerun()
    else:
        st.error("SmartBhai GPT is not available. Check initialization logs.")
        logger.warning("SmartBhai GPT instance is None, chat interface not rendered.")
    st.markdown("</div>", unsafe_allow_html=True)

# === Footer ===
st.markdown(
    """
    <div class='footer'>
        VolGuard Pro | Built with â¤ï¸ by Shritish | Powered by Upstox API & Streamlit
        <br>
        <small>Disclaimer: Trading involves risks. Do your own research and consult a financial advisor.</small>
    </div>
    """,
    unsafe_allow_html=True,
)
