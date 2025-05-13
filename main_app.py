import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
# Assuming these modules are available in the same directory or Python path
from smartbhai_gpt import SmartBhaiGPT # Import SmartBhai GPT class
from upstox_api import initialize_upstox_client, fetch_all_api_portfolio_data, prepare_trade_orders, execute_trade_orders, square_off_positions, fetch_market_depth_by_scrip # Import Upstox API functions
from data_processing import load_data, generate_features, FEATURE_COLS # Import data processing functions and feature list
# Assuming volatility_forecasting, backtesting, strategy_generation exist and are imported if needed
# from volatility_forecasting import forecast_volatility_future # Keeping this import here
# from backtesting import run_backtest # Keeping this import here
# from strategy_generation import generate_trading_strategy # Keeping this import here


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
# Sets basic page settings for the Streamlit app
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# Custom CSS
# Applies custom styling to the Streamlit elements using HTML and CSS
st.markdown("""
    <style>
        /* Main container styling with gradient background */
        .main { background: linear-gradient(135deg, #1a1a2e, #0f1c2e); color: #e5e5e5; font-family: 'Inter', sans-serif; }
        /* Styling for tabs */
        .stTabs [data-baseweb="tab-list"] { background: #16213e; border-radius: 10px; padding: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
        .stTabs [data-baseweb="tab"] { color: #a0a0a0; font-weight: 500; padding: 10px 20px; border-radius: 8px; }
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #e94560; color: white; font-weight: 700; }
        .stTabs [data-baseweb="tab"]:hover { background: #2a2a4a; color: white; }
        /* Sidebar button styling */
        .sidebar .stButton>button { width: 100%; background: #0f3460; color: white; border-radius: 10px; padding: 12px; margin: 5px 0; }
        .sidebar .stButton>button:hover { transform: scale(1.05); background: #e94560; }
        /* Card styling for sections */
        .card { background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9)); border-radius: 15px; padding: 20px; margin: 15px 0; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); }
        .card:hover { transform: translateY(-5px); }
        /* Specific styling for strategy cards (horizontal scroll) */
        .strategy-carousel { display: flex; overflow-x: auto; gap: 20px; padding: 10px; }
        .strategy-card { flex: 0 0 auto; width: 300px; background: #16213e; border-radius: 15px; padding: 20px; }
        .strategy-card:hover { transform: scale(1.05); }
        /* Metric card styling */
        .stMetric { background: rgba(15, 52, 96, 0.7); border-radius: 15px; padding: 15px; text-align: center; }
        /* Gauge styling (example - needs more complex implementation for actual gauge) */
        /* .gauge { width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px; } */
        /* Volatility regime badge styling */
        .regime-badge { padding: 8px 15px; border-radius: 20px; font-weight: bold; font-size: 14px; text-transform: uppercase; }
        .regime-low { background: #28a745; color: white; } /* Green */
        .regime-medium { background: #ffc107; color: black; } /* Yellow */
        .regime-high { background: #dc3545; color: white; } /* Red */
        .regime-event { background: #ff6f61; color: white; } /* Coral */
        /* Alert banner styling (e.g., for market hours) */
        .alert-banner { background: #dc3545; color: white; padding: 15px; border-radius: 10px; position: sticky; top: 0; z-index: 100; }
        /* Main button styling */
        .stButton>button { background: #e94560; color: white; border-radius: 10px; padding: 12px 25px; font-size: 16px; border: none; }
        .stButton>button:hover { transform: scale(1.05); background: #ffcc00; } /* Hover effect */
        /* Footer styling */
        .footer { text-align: center; padding: 20px; color: #a0a0a0; font-size: 14px; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 30px; }
        /* SmartBhai GPT Chat Styling */
        .smartbhai-container {
            background: #1e2a44;
            border-radius: 12px;
            padding: 15px;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        .smartbhai-input > div > div > input {
            background: #2a3b5a;
            border: 2px solid #00cc00; /* Green border */
            border-radius: 10px;
            padding: 12px;
            color: #e5e5e5;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        .smartbhai-input > div > div > input:focus {
            border-color: #ffcc00; /* Yellow border on focus */
            outline: none;
            box-shadow: 0 0 5px rgba(255, 204, 0, 0.5);
        }
        .smartbhai-button > button {
            width: 100%;
            background: #e94560; /* Reddish background */
            color: white;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        .smartbhai-button > button:hover {
            background: #ffcc00; /* Yellow background on hover */
            color: #1a1a2e; /* Dark text on hover */
            transform: translateY(-2px); /* Slight lift effect */
        }
        .smartbhai-chat {
            max-height: 300px; /* Fixed height with scroll */
            overflow-y: auto;
            padding: 10px;
            margin-top: 15px;
            background: #16213e; /* Dark background for chat history */
            border-radius: 10px;
        }
        .chat-bubble {
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%; /* Limit bubble width */
            font-size: 14px;
            line-height: 1.4;
            word-wrap: break-word; /* Ensure long words wrap */
        }
        .user-bubble {
            background: #e94560; /* Reddish background for user */
            color: white;
            margin-left: auto; /* Align user bubble to the right */
            text-align: right;
            border-bottom-right-radius: 2px; /* Slightly different corner */
        }
        .smartbhai-bubble {
            background: #00cc00; /* Green background for SmartBhai */
            color: #1a1a2e; /* Dark text for SmartBhai */
            margin-right: auto; /* Align SmartBhai bubble to the left */
            border-bottom-left-radius: 2px; /* Slightly different corner */
        }
        .smartbhai-title {
            color: #ffcc00; /* Yellow title color */
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            text-align: center;
        }
        .smartbhai-subtitle {
            color: #a0a0a0; /* Gray subtitle color */
            font-size: 14px;
            margin-bottom: 15px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True) # unsafe_allow_html=True is needed to render raw HTML/CSS

# Initialize session state
# Session state is crucial in Streamlit to preserve variables across reruns
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False # Flag to trigger backtest
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None # Store backtest results
if "violations" not in st.session_state:
    st.session_state.violations = 0 # Example state variable (risk violations?) - not fully used in provided code
if "journal_complete" not in st.session_state:
    st.session_state.journal_complete = False # Flag for journal entry - not fully used
if "trades" not in st.session_state:
    st.session_state.trades = [] # Store list of executed trades (prepared orders)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False # Login status flag
if "client" not in st.session_state:
    st.session_state.client = None # Store the initialized Upstox client object(s)
if "real_time_market_data" not in st.session_state:
    st.session_state.real_time_market_data = None # Store real-time market data
if "api_portfolio_data" not in st.session_state:
    st.session_state.api_portfolio_data = {} # Store portfolio data fetched from API
if "prepared_orders" not in st.session_state:
    st.session_state.prepared_orders = None # Store orders prepared by the strategy
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None # Store the main DataFrame with generated features
if "forecast_log" not in st.session_state:
    st.session_state.forecast_log = None # Store volatility forecast log data
if "forecast_metrics" not in st.session_state:
    st.session_state.forecast_metrics = None # Store volatility forecast key metrics
if "generated_strategy" not in st.session_state:
    st.session_state.generated_strategy = None # Store the generated trading strategy
if "backtest_cumulative_pnl_chart_data" not in st.session_state:
    st.session_state.backtest_cumulative_pnl_chart_data = None # Store data for backtest PnL chart
if "active_strategy_details" not in st.session_state:
    st.session_state.active_strategy_details = None # Store details of currently active strategy (for display?) - not fully used
if "order_placement_errors" not in st.session_state:
    st.session_state.order_placement_errors = [] # Store errors encountered during order placement
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Store chat history for SmartBhai GPT
if "query_input" not in st.session_state:
    st.session_state.query_input = "" # Store the current input for the chat box
# Initialize default configuration values in session state
if "capital" not in st.session_state:
    st.session_state.capital = 1000000  # Default trading capital
if "risk_tolerance" not in st.session_state:
    st.session_state.risk_tolerance = "Moderate"  # Default risk tolerance
if "forecast_horizon" not in st.session_state:
    st.session_state.forecast_horizon = 7  # Default forecast horizon in days

# --- Initialize SmartBhai GPT ---
# Initialize the GPT model once when the app starts
smartbhai_gpt = None
try:
    smartbhai_gpt = SmartBhaiGPT(responses_file="responses.csv")
    logger.info("SmartBhai GPT initialized successfully.")
except Exception as e:
    st.sidebar.error(f"Bhai, SmartBhai GPT load nahi hua: {str(e)}")
    logger.error(f"SmartBhai GPT initialization failed: {str(e)}")


# Check Upstox API hours for live market depth (only for trading actions)
# This function checks if the current time is within standard Indian market hours for equity derivatives
def is_market_depth_hours():
    """Checks if current time is within standard market hours (9:15 AM to 3:30 PM IST)."""
    now = datetime.now().time()
    start = datetime.strptime("09:15", "%H:%M").time()
    end = datetime.strptime("15:30", "%H:%M").time()
    return start <= now <= end

# Fetch portfolio data
# This function fetches various portfolio details using the Upstox API client
def fetch_portfolio_data(upstox_client, capital):
    """Fetches and summarizes user portfolio data."""
    portfolio_summary = {
        "weekly_pnl": 0.0,
        "margin_used": 0.0,
        "exposure": 0.0,
        "total_capital": capital
    }
    # Check if client is initialized and has an access token
    if not upstox_client or not upstox_client.get("access_token"):
        logger.warning("Upstox client not available for portfolio data.")
        return portfolio_summary # Return default summary if client is missing
    try:
        # Fetch all portfolio-related data using the API client
        portfolio_data = fetch_all_api_portfolio_data(upstox_client)
        # Store the raw API response in session state
        st.session_state.api_portfolio_data = portfolio_data

        # Extract and summarize relevant information
        margin_data = portfolio_data.get("margin", {}).get("data", {})
        positions_data = portfolio_data.get("positions", {}).get("data", [])

        # Ensure margin_data is a dictionary before accessing keys
        if isinstance(margin_data, dict):
            portfolio_summary["margin_used"] = margin_data.get("utilized_margin", 0.0)
        else:
            logger.warning("Margin data not in expected format.")
            portfolio_summary["margin_used"] = 0.0 # Default to 0 if format is wrong


        # Calculate total PnL from positions
        # Ensure positions_data is a list before iterating
        if isinstance(positions_data, list):
             portfolio_summary["weekly_pnl"] = sum(
                 pos.get("unrealized_mtm", 0.0) + pos.get("realized_profit", 0.0)
                 for pos in positions_data if isinstance(pos, dict) # Ensure each item is a dictionary
             )
        else:
            logger.warning("Positions data not in expected format.")
            portfolio_summary["weekly_pnl"] = 0.0 # Default to 0 if format is wrong


        # Calculate exposure percentage
        portfolio_summary["exposure"] = (portfolio_summary["margin_used"] / capital * 100) if capital and capital > 0 else 0.0 # Avoid division by zero

        logger.info("Portfolio data fetched and summarized successfully.")
        return portfolio_summary
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        # Return the default summary in case of any error
        return portfolio_summary


# Calculate position PnL with LTP
# This function attempts to update position PnL using the latest LTP by fetching market depth
def calculate_position_pnl_with_ltp(upstox_client, positions_data):
    """Updates positions data with current PnL based on live LTP if available."""
    # Return original data if client is not available or no positions
    if not upstox_client or not upstox_client.get("access_token") or not positions_data or not isinstance(positions_data, list):
        logger.warning("Client or positions data not available for PnL calculation with LTP.")
        return positions_data # Return original data


    updated_positions = []
    for pos in positions_data:
        # Ensure position item is a dictionary
        if not isinstance(pos, dict):
            updated_positions.append(pos)
            continue # Skip to next item if format is unexpected

        instrument_key = pos.get("instrument_key")
        buy_avg_price = pos.get("buy_avg_price", 0.0)
        sell_avg_price = pos.get("sell_avg_price", 0.0)
        quantity = pos.get("quantity", 0)

        # Proceed only if instrument_key and non-zero quantity are present
        if instrument_key and quantity != 0:
            try:
                # Fetch market data for the specific instrument key
                market_data = fetch_market_depth_by_scrip(upstox_client, instrument_key=instrument_key)
                # Extract LTP from the market data response
                # Check if market_data is valid and contains the expected structure
                ltp = market_data.get("Data", [{}])[0].get("LastTradedPrice", 0.0) if market_data else 0.0

                # Calculate position PnL based on quantity and LTP vs average price
                if ltp and ltp > 0: # Calculate PnL only if LTP is valid
                    position_pnl = quantity * (ltp - buy_avg_price) if quantity > 0 else abs(quantity) * (sell_avg_price - ltp)
                    pos['CurrentPnL'] = position_pnl
                    pos['LTP'] = ltp
                else:
                    # Fallback to unrealized_mtm if LTP is not available/valid
                    logger.warning(f"LTP not available for {instrument_key}. Using unrealized_mtm.")
                    pos['CurrentPnL'] = pos.get("unrealized_mtm", 0.0)
                    pos['LTP'] = 0.0 # Indicate LTP was not fetched

            except Exception as e:
                logger.warning(f"Could not fetch LTP or calculate PnL for {instrument_key}: {e}")
                # In case of any error during fetch or calculation, use unrealized_mtm as fallback
                pos['CurrentPnL'] = pos.get("unrealized_mtm", 0.0)
                pos['LTP'] = 0.0 # Indicate LTP was not fetched
        else:
             # If instrument_key or quantity is missing/zero, use existing PnL data
             pos['CurrentPnL'] = pos.get("unrealized_mtm", 0.0)
             pos['LTP'] = 0.0 # No LTP available

        updated_positions.append(pos)
    return updated_positions


# --- Streamlit App Layout and Logic ---

# Sidebar for Login and Controls
with st.sidebar:
    st.header("🔑 Upstox Login")
    # Input for access token, masked as password
    access_token = st.text_input("Access Token", type="password", key="access_token_input") # Added a unique key

    # Login button
    if st.button("Login to Upstox"):
        if not access_token:
            st.error("❌ Access token cannot be empty.")
            logger.error("Login attempted with empty access token.")
        else:
            # Attempt to initialize the Upstox client using the provided token
            client_objects = initialize_upstox_client(access_token)
            if client_objects:
                # Store the initialized client objects in session state
                st.session_state.client = client_objects
                st.session_state.logged_in = True
                st.success("✅ Logged in to Upstox!")
                logger.info("Upstox login successful.")
                # Optionally, trigger a rerun to refresh the UI with login status
                st.rerun()
            else:
                st.session_state.logged_in = False
                st.session_state.client = None
                st.error("❌ Login failed. Invalid or expired access token. Get a new token from Upstox.")
                logger.error("Upstox login failed: Invalid or expired token.")

    # Show trading controls only if logged in
    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        # Input for trading capital, linked to session state
        capital = st.number_input("Capital (₹)", min_value=100000, value=st.session_state.capital, step=100000, format="%d", key="capital_input")
        st.session_state.capital = capital # Update session state

        # Select box for risk tolerance, linked to session state
        risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=["Conservative", "Moderate", "Aggressive"].index(st.session_state.risk_tolerance), key="risk_tolerance_input")
        st.session_state.risk_tolerance = risk_tolerance # Update session state

        # Slider for forecast horizon, linked to session state
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, st.session_state.forecast_horizon, key="forecast_horizon_input")
        st.session_state.forecast_horizon = forecast_horizon # Update session state

        st.markdown("---")
        st.markdown("**Backtest Parameters**")
        # Date inputs for backtest period, linked to session state
        default_start_date = datetime.now().date() - timedelta(days=365)
        default_end_date = datetime.now().date()
        start_date = st.date_input("Start Date", value=default_start_date, key="backtest_start_date_input")
        st.session_state.backtest_start_date = start_date # Update session state
        end_date = st.date_input("End Date", value=default_end_date, key="backtest_end_date_input")
        st.session_state.backtest_end_date = end_date # Update session state

        # Select box for backtest strategy filter, linked to session state
        strategy_options = ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Short Straddle", "Short Put Vertical Spread", "Short Call Vertical Spread", "Short Put", "Long Put"]
        strategy_choice = st.selectbox("Backtest Strategy Filter", strategy_options, index=0, key="backtest_strategy_input")
        st.session_state.backtest_strategy = strategy_choice # Update session state

        # Button to trigger the backtest
        if st.button("Run Backtest"):
            # Set flag to True, the main script body will detect this and run the backtest
            st.session_state.backtest_run = True
            st.session_state.backtest_results = None  # Reset previous results
            st.success("Backtest started! Check the Backtest tab for results.")
            logger.info("Backtest initiated.")
            # Rerun the app to proceed to the backtest logic in the main body
            st.rerun()

        st.markdown("---")
        # Button to square off all positions
        if st.button("Square Off All Positions"):
            # Check if client is logged in and available
            if st.session_state.logged_in and st.session_state.client:
                # Call the square_off_positions function from upstox_api module
                success = square_off_positions(st.session_state.client)
                if success:
                    st.success("All positions squared off successfully!")
                    logger.info("All positions squared off.")
                    # Optionally, refresh portfolio data after squaring off
                    # fetch_portfolio_data(st.session_state.client, st.session_state.capital)
                else:
                    st.error("Failed to square off positions. Check logs.")
                    logger.error("Failed to square off positions.")
            else:
                st.error("Not logged in to Upstox.")
                logger.warning("Square off attempted without login.")


# Main App Title and Description
st.title("🛡️ VolGuard Pro")
st.markdown("**Your AI-powered options trading cockpit for NIFTY 50**")

# Check market depth hours for live trading data (informational banner)
# This banner informs the user if they are outside live market hours
if not is_market_depth_hours():
    st.warning("⚠️ Outside live market depth hours (9:15 AM–3:30 PM IST). Live LTP and order execution may be limited.")
    logger.info("App running outside live market hours.")

# --- Data Loading and Feature Generation ---
# This section runs every time the app reruns
data_load_success = False
data_source = "Unknown"

# Attempt to load data using the client if logged in, otherwise fallback to CSV
if st.session_state.logged_in and st.session_state.client:
    try:
        logger.info("Attempting to load data using Upstox API.")
        # Pass the client object to load_data
        df, real_data, data_source = load_data(st.session_state.client)
        if df is not None:
            # Store real-time data and generated features DataFrame in session state
            st.session_state.real_time_market_data = real_data if real_data else {}
            analysis_df = generate_features(df.copy(), st.session_state.real_time_market_data, st.session_state.capital) # Pass a copy
            if analysis_df is not None:
                st.session_state.analysis_df = analysis_df
                data_load_success = True
                logger.info(f"Data loaded successfully from {data_source}.")
            else:
                st.error("❌ Failed to generate features from loaded data. Check data processing module logs.")
                logger.error("Feature generation failed during app load.")
                # No st.stop() here, allow fallback attempt
        else:
             st.warning("Data loading from Upstox API failed. Attempting CSV fallback.")
             # API load failed, attempt CSV fallback below
             st.session_state.real_time_market_data = {} # Reset real-time data


    except Exception as e:
        st.error(f"❌ Error during Upstox API data loading: {str(e)}. Attempting CSV fallback.")
        logger.error(f"Error during Upstox API data loading: {str(e)}.")
        # API load failed, attempt CSV fallback below
        st.session_state.real_time_market_data = {} # Reset real-time data


# Fallback to CSV if API loading failed or not logged in
if not data_load_success:
    logger.info("Attempting to load data from CSV fallback.")
    try:
        # Pass None as client to indicate using CSV fallback
        df, real_data, data_source = load_data(None) # Fallback to CSV
        if df is not None:
            # Store fallback real-time data and generated features DataFrame in session state
            st.session_state.real_time_market_data = real_data if real_data else {} # Store fallback real_data
            analysis_df = generate_features(df.copy(), st.session_state.real_time_market_data, st.session_state.capital) # Pass a copy
            if analysis_df is not None:
                st.session_state.analysis_df = analysis_df
                data_load_success = True
                logger.info(f"Data loaded successfully from {data_source} (CSV Fallback).")
            else:
                st.error("❌ Failed to generate features from fallback data. Check data processing module logs.")
                logger.error("Feature generation failed during CSV fallback.")
                st.session_state.analysis_df = None # Ensure state is None on failure
                st.session_state.real_time_market_data = {} # Ensure state is empty on failure
        else:
            st.error("❌ Failed to load fallback data from CSV. Check CSV files and logs.")
            logger.error("Fallback data loading failed.")
            st.session_state.analysis_df = None # Ensure state is None on failure
            st.session_state.real_time_market_data = {} # Ensure state is empty on failure

    except Exception as e:
        st.error(f"❌ Error loading fallback data: {str(e)}")
        logger.error(f"Fallback data loading error: {str(e)}")
        st.session_state.analysis_df = None # Ensure state is None on failure
        st.session_state.real_time_market_data = {} # Ensure state is empty on failure


# Stop the app if data loading was completely unsuccessful
if not data_load_success or st.session_state.analysis_df is None:
    st.error("❌ VolGuard Pro could not load necessary data. Please ensure API credentials are correct (if using API) or CSV files are available.")
    logger.critical("App halted due to data loading failure.")
    st.stop() # Stop execution if essential data is missing


# --- Main Content Tabs ---
# Define the tabs for the main content area
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
    # Display market data if available
    if st.session_state.real_time_market_data:
        market_data = st.session_state.real_time_market_data
        # Use columns for a cleaner layout of metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            nifty_spot = market_data.get('nifty_spot', 'N/A')
            # Format metric value if it's a number
            st.metric("NIFTY Spot", f"{nifty_spot:.2f}" if isinstance(nifty_spot, (int, float)) else str(nifty_spot))
        with col2:
            vix = market_data.get('vix', 'N/A')
            st.metric("India VIX", f"{vix:.2f}" if isinstance(vix, (int, float)) else str(vix))
        with col3:
            pcr = market_data.get('pcr', 'N/A')
            st.metric("PCR", f"{pcr:.2f}" if isinstance(pcr, (int, float)) else str(pcr))
        with col4:
            straddle_price = market_data.get('straddle_price', 'N/A')
            st.metric("ATM Straddle", f"{straddle_price:.2f}" if isinstance(straddle_price, (int, float)) else str(straddle_price))

        # More metrics in another row of columns
        col5, col6, col7 = st.columns(3)
        with col5:
            atm_strike = market_data.get('atm_strike', 'N/A')
            st.metric("ATM Strike", f"{atm_strike:.2f}" if isinstance(atm_strike, (int, float)) else str(atm_strike))
        with col6:
            max_pain_strike = market_data.get('max_pain_strike', 'N/A')
            st.metric("Max Pain", f"{max_pain_strike:.2f}" if isinstance(max_pain_strike, (int, float)) else str(max_pain_strike))
        with col7:
            max_pain_diff_pct = market_data.get('max_pain_diff_pct', 'N/A')
            st.metric("Max Pain Diff %", f"{max_pain_diff_pct:.2f}%" if isinstance(max_pain_diff_pct, (int, float)) else str(max_pain_diff_pct))

        st.subheader("Option Chain Preview")
        # Display a preview of the option chain DataFrame if available
        if "option_chain" in market_data and isinstance(market_data["option_chain"], pd.DataFrame) and not market_data["option_chain"].empty:
            # Select relevant columns for display
            display_cols = ["StrikeRate", "CPType", "LastRate", "IV", "OpenInterest", "Volume"]
            # Ensure columns exist before displaying
            display_cols = [col for col in display_cols if col in market_data["option_chain"].columns]
            st.dataframe(market_data["option_chain"][display_cols].head(20)) # Show more rows
        else:
            st.info("Option chain data not available.")
    else:
        st.error("No market data available. Check data source and logs.")

    # Display data source information
    if data_load_success:
        st.markdown(f"**Data Source**: {data_source}")


# Tab 2: Volatility Forecast
with tab2:
    st.header("🔮 Volatility Forecast")
    # Run forecast only if analysis data is available
    if st.session_state.analysis_df is not None:
        try:
            # Import forecasting functions here if they aren't globally imported
            from volatility_forecasting import forecast_volatility_future

            # Run the volatility forecasting model
            # Pass a copy of the DataFrame to the forecasting function
            forecast_log_df, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(st.session_state.analysis_df.copy(), st.session_state.forecast_horizon)

            # Store results in session state
            st.session_state.forecast_log = forecast_log_df
            st.session_state.forecast_metrics = {
                'forecasted_vix': blended_vols[-1] if blended_vols else None, # Get the last forecast value
                'vix_range_low': np.min(blended_vols) if blended_vols else None,
                'vix_range_high': np.max(blended_vols) if blended_vols else None,
                'confidence': confidence_score,
                'rmse': rmse,
                'feature_importances': feature_importances,
                'realized_vol': realized_vol
            }

            # Display forecast metrics
            metrics = st.session_state.forecast_metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                forecasted_vix = metrics.get('forecasted_vix', 'N/A')
                st.metric("Forecasted VIX", f"{forecasted_vix:.2f}" if isinstance(forecasted_vix, (int, float)) else str(forecasted_vix))
            with col2:
                vix_range_low = metrics.get('vix_range_low', 'N/A')
                st.metric("VIX Range Low", f"{vix_range_low:.2f}" if isinstance(vix_range_low, (int, float)) else str(vix_range_low))
            with col3:
                vix_range_high = metrics.get('vix_range_high', 'N/A')
                st.metric("VIX Range High", f"{vix_range_high:.2f}" if isinstance(vix_range_high, (int, float)) else str(vix_range_high))
            with col4:
                 confidence = metrics.get('confidence', 'N/A')
                 st.metric("Confidence", f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else str(confidence))

            st.subheader("Forecast Trend (Blended)")
            # Display the forecast trend chart
            if st.session_state.forecast_log is not None and not st.session_state.forecast_log.empty:
                # Ensure 'Blended_Vol' column exists before charting
                if 'Blended_Vol' in st.session_state.forecast_log.columns:
                     # Combine historical VIX and forecast for the chart
                     historical_vix = st.session_state.analysis_df['VIX'].tail(30) # Show last 30 days of historical VIX
                     combined_vols = pd.concat([historical_vix, st.session_state.forecast_log.set_index('Date')['Blended_Vol']])
                     combined_vols = combined_vols.rename("VIX_Combined") # Rename series for chart legend
                     st.line_chart(combined_vols)
                else:
                     st.warning("Blended volatility data not available for charting.")
            else:
                st.warning("No forecast data available for charting.")

            st.subheader("Model Metrics")
            if metrics.get('rmse') is not None:
                 st.write(f"XGBoost RMSE: {metrics['rmse']:.2f}")
            if metrics.get('realized_vol') is not None:
                 st.write(f"Recent Realized Volatility: {metrics['realized_vol']:.2f}")


            st.subheader("Feature Importances (XGBoost)")
            if metrics.get('feature_importances') is not None and st.session_state.analysis_df is not None:
                feature_importances_df = pd.DataFrame({
                    'Feature': FEATURE_COLS, # Use FEATURE_COLS from data_processing
                    'Importance': metrics['feature_importances']
                }).sort_values('Importance', ascending=False)
                st.bar_chart(feature_importances_df.set_index('Feature'))
            else:
                 st.info("Feature importances not available.")

        except Exception as e:
            st.error(f"❌ Error generating volatility forecast: {str(e)}. Check forecasting module logs.")
            logger.error(f"Volatility forecast error: {str(e)}.")
            st.session_state.forecast_log = None # Reset state on error
            st.session_state.forecast_metrics = None # Reset state on error

    else:
        st.warning("No analysis data available for volatility forecasting.")


# Tab 3: Trading Strategy
with tab3:
    st.header("🤖 Trading Strategy")
    # Generate strategy only if analysis data, market data, and forecast metrics are available
    if st.session_state.analysis_df is not None and st.session_state.real_time_market_data and st.session_state.forecast_metrics is not None:
        try:
            # Import strategy generation function here if not globally imported
            from strategy_generation import generate_trading_strategy

            # Generate trading strategy using available data and user parameters
            strategy = generate_trading_strategy(
                st.session_state.analysis_df.copy(), # Pass a copy
                st.session_state.real_time_market_data,
                st.session_state.forecast_metrics, # Pass forecast metrics
                st.session_state.capital,
                st.session_state.risk_tolerance
            )
            st.session_state.generated_strategy = strategy # Store the generated strategy

            if strategy:
                st.subheader("Recommended Strategy")
                st.markdown(f"**Strategy**: {strategy.get('Strategy', 'N/A')}")
                st.markdown(f"**Confidence**: {strategy.get('Confidence', 0.0):.2%}")
                st.markdown(f"**Deploy Amount**: ₹{strategy.get('Deploy', 0.0):.2f}")
                st.markdown(f"**Reasoning**: {strategy.get('Reasoning', 'N/A')}")


                # Button to prepare orders based on the generated strategy
                if st.button("Prepare Orders"):
                    # Check if market is open before preparing live orders
                    if is_market_depth_hours():
                         orders = prepare_trade_orders(
                            strategy,
                            st.session_state.real_time_market_data,
                            st.session_state.capital
                        )
                         st.session_state.prepared_orders = orders # Store prepared orders
                         if orders:
                             st.success(f"✅ {len(orders)} Orders prepared successfully!")
                             st.subheader("Prepared Orders Preview")
                             # Display prepared orders in a DataFrame
                             st.dataframe(pd.DataFrame(orders))
                             logger.info("Orders prepared successfully.")
                         else:
                             st.error("❌ Failed to prepare orders. Check logs.")
                             logger.error("Order preparation failed.")
                             st.session_state.prepared_orders = None # Reset state on failure
                    else:
                         st.warning("⏰ Market is closed. Cannot prepare live orders outside market hours (9:15 AM - 3:30 PM IST).")


                # Show execute button only if orders have been prepared and user is logged in
                if st.session_state.prepared_orders and st.session_state.logged_in and st.session_state.client:
                    if st.button("Execute Orders"):
                        # Check if market is open before executing live orders
                        if is_market_depth_hours():
                             st.info("Attempting to execute orders...")
                             # Call the execute_trade_orders function from upstox_api module
                             success, response = execute_trade_orders(
                                 st.session_state.client,
                                 st.session_state.prepared_orders
                             )
                             if success:
                                 st.success("✅ Orders executed successfully!")
                                 # Add the executed orders to the journal/trades list
                                 st.session_state.trades.extend(st.session_state.prepared_orders)
                                 st.session_state.prepared_orders = None # Clear prepared orders after execution
                                 st.session_state.order_placement_errors = [] # Clear errors
                                 logger.info("Orders executed successfully.")
                                 # Optionally, trigger a rerun to refresh portfolio data
                                 # st.experimental_rerun() # Consider using st.rerun() in newer Streamlit versions
                             else:
                                 st.error("❌ Order execution failed.")
                                 # Store and display the error response
                                 st.session_state.order_placement_errors.append(response)
                                 st.subheader("Execution Response")
                                 st.json(response) # Display the raw response for debugging
                                 logger.error(f"Order execution failed: {response}.")
                        else:
                             st.warning("⏰ Market is closed. Cannot execute live orders outside market hours (9:15 AM - 3:30 PM IST).")

                # Display any order placement errors
                if st.session_state.order_placement_errors:
                     st.subheader("Recent Order Placement Errors")
                     for error in st.session_state.order_placement_errors:
                         st.json(error) # Display each error as JSON


            else:
                st.warning("⚠️ No strategy generated based on current market conditions and parameters.")
                st.session_state.generated_strategy = None # Ensure state is None if no strategy


        except Exception as e:
            st.error(f"❌ Error generating strategy: {str(e)}. Check strategy generation module logs.")
            logger.error(f"Trading strategy error: {str(e)}.")
            st.session_state.generated_strategy = None # Reset state on error

    else:
        st.warning("No analysis data, market data, or forecast available for strategy generation.")

# Tab 4: Portfolio
with tab4:
    st.header("💼 Portfolio")
    # Fetch and display portfolio summary and positions if logged in
    if st.session_state.logged_in and st.session_state.client:
        # Fetch portfolio data dynamically every time the tab is viewed or app reruns
        portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)

        # Display portfolio summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Weekly PnL", f"₹{portfolio_summary['weekly_pnl']:.2f}")
        with col2:
            st.metric("Margin Used", f"₹{portfolio_summary['margin_used']:.2f}")
        with col3:
            st.metric("Exposure", f"{portfolio_summary['exposure']:.2f}%")

        st.subheader("Open Positions")
        # Get raw positions data from session state
        positions_data_raw = st.session_state.api_portfolio_data.get("positions", {}).get("data", [])
        # Calculate PnL using live LTP if available and update the positions data
        positions_data_updated = calculate_position_pnl_with_ltp(st.session_state.client, positions_data_raw)

        # Display open positions in a DataFrame
        if positions_data_updated:
            positions_df = pd.DataFrame(positions_data_updated)
            if not positions_df.empty:
                # Select relevant columns for display
                display_cols = ["instrument_key", "quantity", "buy_avg_price", "sell_avg_price", "LTP", "CurrentPnL", "unrealized_mtm", "realized_profit"]
                display_cols = [col for col in display_cols if col in positions_df.columns] # Ensure columns exist

                st.dataframe(positions_df[display_cols])
            else:
                st.info("No open positions.")
        else:
            st.info("No open positions.")

        st.subheader("Fund Summary")
        fund_data = st.session_state.api_portfolio_data.get("margin", {}).get("data", {})
        if fund_data and isinstance(fund_data, dict):
             fund_df = pd.DataFrame([fund_data]) # Convert dictionary to DataFrame for display
             # Select relevant fund details
             fund_display_cols = ["available_margin", "utilized_margin", "total_margin", "payin_amount", "payout_amount"]
             fund_display_cols = [col for col in fund_display_cols if col in fund_df.columns] # Ensure columns exist
             if not fund_df.empty:
                  st.dataframe(fund_df[fund_display_cols].applymap(lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x)) # Format currency
             else:
                  st.info("Fund data not available.")
        else:
             st.info("Fund data not available.")


        st.subheader("Trade History")
        trade_data = st.session_state.api_portfolio_data.get("trade_book", {}).get("data", [])
        if trade_data and isinstance(trade_data, list):
            trade_df = pd.DataFrame(trade_data)
            if not trade_df.empty:
                 # Select and format relevant trade details
                 trade_display_cols = ["instrument_key", "quantity", "transaction_type", "exchange_timestamp", "price", "order_id"]
                 trade_display_cols = [col for col in trade_display_cols if col in trade_df.columns]
                 if not trade_df.empty:
                    st.dataframe(trade_df[trade_display_cols])
                 else:
                    st.info("No recent trades.")
            else:
                st.info("No recent trades.")
        else:
            st.info("Trade history not available.")

        st.subheader("Order Book")
        order_data = st.session_state.api_portfolio_data.get("order_book", {}).get("data", [])
        if order_data and isinstance(order_data, list):
            order_df = pd.DataFrame(order_data)
            if not order_df.empty:
                 # Select and format relevant order details
                 order_display_cols = ["instrument_key", "quantity", "transaction_type", "order_type", "status", "price", "average_price", "placed_by", "order_id"]
                 order_display_cols = [col for col in order_display_cols if col in order_df.columns]
                 if not order_df.empty:
                    st.dataframe(order_df[order_display_cols])
                 else:
                    st.info("No pending orders.")
            else:
                 st.info("No pending orders.")
        else:
            st.info("Order book data not available.")

    else:
        st.info("Please log in to Upstox to view your portfolio.")


# Tab 5: Journal
with tab5:
    st.header("📝 Trading Journal")
    journal_file = "journal_log.csv"
    # Load existing journal entries
    try:
        if os.path.exists(journal_file):
            journal_df = pd.read_csv(journal_file)
        else:
            # Create an empty DataFrame if the file doesn't exist
            journal_df = pd.DataFrame(columns=["Date", "Strategy", "PnL", "Notes"])
    except Exception as e:
        st.error(f"Error loading journal file: {str(e)}")
        logger.error(f"Error loading journal file: {str(e)}")
        journal_df = pd.DataFrame(columns=["Date", "Strategy", "PnL", "Notes"]) # Start with empty DF on error


    # Form for adding new journal entries
    with st.form("journal_form"):
        st.subheader("Log a New Trade")
        date = st.date_input("Trade Date", value=datetime.now().date(), key="journal_date")
        strategy = st.text_input("Strategy", key="journal_strategy")
        pnl = st.number_input("PnL (₹)", format="%.2f", key="journal_pnl")
        notes = st.text_area("Notes", key="journal_notes")
        submitted = st.form_submit_button("Log Trade")

        # Process form submission
        if submitted:
            # Create a new entry as a DataFrame row
            new_entry = pd.DataFrame({
                "Date": [date],
                "Strategy": [strategy],
                "PnL": [pnl],
                "Notes": [notes]
            })
            # Append the new entry to the existing journal DataFrame
            journal_df = pd.concat([journal_df, new_entry], ignore_index=True)
            try:
                # Save the updated DataFrame back to the CSV file
                journal_df.to_csv(journal_file, index=False)
                st.session_state.journal_complete = True # Flag success (though not strictly used)
                st.success("✅ Trade logged successfully!")
                logger.info("Trade logged successfully.")
                # Clear form inputs after submission by updating session state
                st.session_state.journal_date = datetime.now().date() # Reset date input
                st.session_state.journal_strategy = "" # Reset strategy input
                st.session_state.journal_pnl = 0.0 # Reset PnL input
                st.session_state.journal_notes = "" # Reset notes input
                st.rerun() # Rerun to clear form and display updated table
            except Exception as e:
                 st.error(f"Error saving journal entry: {str(e)}")
                 logger.error(f"Error saving journal entry: {str(e)}")


    st.subheader("Trade History")
    # Display the full trade history from the journal DataFrame
    if not journal_df.empty:
        # Sort journal entries by date descending
        journal_df['Date'] = pd.to_datetime(journal_df['Date'], errors='coerce') # Ensure Date is datetime
        journal_df = journal_df.sort_values(by='Date', ascending=False).dropna(subset=['Date']) # Sort and drop rows where Date couldn't be parsed
        st.dataframe(journal_df)
    else:
        st.info("No trades logged yet.")


# Tab 6: Backtest
with tab6:
    st.header("📈 Backtest Results")
    # Check if backtest should run based on the sidebar button flag
    if st.session_state.backtest_run and st.session_state.analysis_df is not None:
        st.info("Running backtest...")
        try:
            # Import backtesting function here if not globally imported
            from backtesting import run_backtest

            # Run the backtest using analysis data and parameters from session state
            backtest_results = run_backtest(
                st.session_state.analysis_df.copy(), # Pass a copy
                st.session_state.backtest_start_date,
                st.session_state.backtest_end_date,
                st.session_state.capital,
                st.session_state.backtest_strategy
            )
            st.session_state.backtest_results = backtest_results # Store results
            st.session_state.backtest_run = False # Reset the flag after running

            if backtest_results:
                st.success("✅ Backtest completed!")
                st.subheader("Key Performance Metrics")
                # Display key backtest metrics using columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{backtest_results.get('total_return', 0.0):.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{backtest_results.get('sharpe_ratio', 0.0):.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{backtest_results.get('max_drawdown', 0.0):.2%}")

                st.subheader("Cumulative PnL Chart")
                # Display the cumulative PnL chart
                if "cumulative_pnl" in backtest_results and isinstance(backtest_results["cumulative_pnl"], pd.Series):
                    st.session_state.backtest_cumulative_pnl_chart_data = backtest_results["cumulative_pnl"] # Store chart data
                    st.line_chart(backtest_results["cumulative_pnl"])
                else:
                     st.warning("Cumulative PnL data not available for charting.")

                st.subheader("Trade Log Preview")
                # Display a preview of the backtest trade log
                if "trade_log" in backtest_results and isinstance(backtest_results["trade_log"], pd.DataFrame) and not backtest_results["trade_log"].empty:
                     st.dataframe(backtest_results["trade_log"].head())
                else:
                     st.info("Backtest trade log is empty or not available.")

            else:
                st.error("❌ Backtest failed to return results. Check backtesting module logs.")
                logger.error("Backtest failed.")
                st.session_state.backtest_results = None # Reset state on failure

        except Exception as e:
            st.error(f"❌ Error running backtest: {str(e)}. Check backtesting module logs.")
            logger.error(f"Backtest error: {str(e)}.")
            st.session_state.backtest_results = None # Reset state on error

    # Display results if a backtest has already been run and results are stored
    elif st.session_state.backtest_results is not None:
        st.subheader("Key Performance Metrics")
        metrics = st.session_state.backtest_results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0.0):.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0):.2%}")

        st.subheader("Cumulative PnL Chart")
        if st.session_state.backtest_cumulative_pnl_chart_data is not None:
             st.line_chart(st.session_state.backtest_cumulative_pnl_chart_data)
        else:
             st.warning("Cumulative PnL data not available for charting.")

        st.subheader("Trade Log Preview")
        if "trade_log" in st.session_state.backtest_results and isinstance(st.session_state.backtest_results["trade_log"], pd.DataFrame) and not st.session_state.backtest_results["trade_log"].empty:
             st.dataframe(st.session_state.backtest_results["trade_log"].head())
        else:
             st.info("Backtest trade log is empty or not available.")


    else:
        st.info("Run a backtest from the sidebar to see results.")


# Tab 7: Risk Dashboard
with tab7:
    st.header("⚠️ Risk Dashboard")
    # Display risk metrics if analysis data is available
    if st.session_state.analysis_df is not None:
        try:
            # Get the latest data row for risk calculation
            if not st.session_state.analysis_df.empty:
                latest_data = st.session_state.analysis_df.iloc[-1]

                # Calculate simplified risk metrics
                # Ensure 'PnL_Day' exists and is numeric, default to 0 if not
                pnl_day = pd.to_numeric(latest_data.get("PnL_Day", 0), errors='coerce').fillna(0)

                risk_metrics = {
                    # Simplified VaR (Value at Risk) and Max Loss based on daily PnL distribution assumption
                    # These are rough estimates and not actual portfolio VaR calculations
                    "VaR_95": pnl_day * -1.65,  # Approx 95th percentile loss for normal distribution
                    "Max_Loss": pnl_day * -2.33,  # Approx 99th percentile loss for normal distribution
                    # Determine volatility regime based on VIX level
                    "Volatility_Regime": "High" if latest_data.get("VIX", 0) > 20 else ("Medium" if latest_data.get("VIX", 0) > 15 else "Low")
                }

                # Fetch portfolio summary again to get the latest exposure for the gauge
                # This might be slightly out of sync if trades just executed, but sufficient for dashboard
                portfolio_summary_for_risk = fetch_portfolio_data(st.session_state.client, st.session_state.capital)


                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("VaR (95%)", f"₹{risk_metrics['VaR_95']:.2f}")
                with col2:
                    st.metric("Max Loss (99%)", f"₹{risk_metrics['Max_Loss']:.2f}")
                with col3:
                    # Determine CSS class based on volatility regime
                    regime_class = {
                        "Low": "regime-low",
                        "Medium": "regime-medium",
                        "High": "regime-high"
                        # "Event": "regime-event" # Add if Event regime is used
                    }.get(risk_metrics["Volatility_Regime"], "regime-medium") # Default to medium

                    # Display volatility regime as a styled badge using HTML
                    st.markdown(f"<span class='regime-badge {regime_class}'>{risk_metrics['Volatility_Regime']} Regime</span>", unsafe_allow_html=True) # Added " Regime" for clarity

                st.subheader("Risk Exposure (Margin Used / Capital)")
                # Display exposure using a simple metric, the gauge graphic requires more complex implementation than simple CSS
                st.metric("Current Exposure", f"{portfolio_summary_for_risk['exposure']:.2f}%")
                # The original gauge div was commented out as it's just a static graphic placeholder in the CSS


            else:
                 st.warning("Analysis DataFrame is empty. Cannot calculate risk metrics.")

        except Exception as e:
            st.error(f"❌ Error generating risk dashboard: {str(e)}. Check logs.")
            logger.error(f"Risk dashboard error: {str(e)}.")

    else:
        st.warning("No data available for risk dashboard.")


# --- SmartBhai GPT Chat Interface ---
st.markdown("---") # Horizontal rule for separation
st.markdown("<div class='smartbhai-container'>", unsafe_allow_html=True) # Chat container div
st.markdown("<div class='smartbhai-title'>💬 SmartBhai GPT</div>", unsafe_allow_html=True) # Chat title
st.markdown("<div class='smartbhai-subtitle'>Ask about IV, strategies, or market buzz!</div>", unsafe_allow_html=True) # Chat subtitle

# Display chat interface only if SmartBhai GPT was initialized successfully
if smartbhai_gpt:
    with st.container():
        # Chat history display area with fixed height and scroll
        st.markdown("<div class='smartbhai-chat'>", unsafe_allow_html=True)
        # Iterate through chat history in session state and display bubbles
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                # User bubble aligned right
                st.markdown(f"<div class='chat-bubble user-bubble'>{chat['message']}</div>", unsafe_allow_html=True)
            else:
                # SmartBhai bubble aligned left
                st.markdown(f"<div class='chat-bubble smartbhai-bubble'>{chat['message']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) # Close chat history div

        # Input form for new messages
        with st.form("smartbhai_form", clear_on_submit=True):
            # Text input for user query, linked to session state and using custom CSS class
            query = st.text_input("Ask SmartBhai:", key="query_input", placeholder="Bhai, IV kya hai? Strategy kya lagau?", label_visibility="collapsed") # Use label_visibility="collapsed" to hide the label above the input box
            # Button to send the message, using custom CSS class
            submit = st.form_submit_button("Send", use_container_width=True, help="Send message to SmartBhai GPT") # Added help text


            # Process form submission if query is not empty
            if submit and query:
                st.session_state.query_input = "" # Clear the input box immediately via session state update
                st.session_state.chat_history.append({"role": "user", "message": query}) # Add user query to history

                # Generate response from SmartBhai GPT
                try:
                     response = smartbhai_gpt.generate_response(query)
                     st.session_state.chat_history.append({"role": "assistant", "message": response}) # Add bot response to history
                except Exception as e:
                     logger.error(f"Error generating SmartBhai response: {str(e)}")
                     st.session_state.chat_history.append({"role": "assistant", "message": "Sorry, Bhai, kuch technical issue hai. Try again! 🙏"}) # Add an error message


                st.rerun() # Rerun the app to display the updated chat history

else:
    # Show an error if SmartBhai GPT failed to initialize
    st.error("SmartBhai GPT is not available.")

st.markdown("</div>", unsafe_allow_html=True) # Close chat container div


# --- Footer ---
st.markdown("""
    <div class='footer'>
        VolGuard Pro | Built with ❤️ by Shritish | Powered by Upstox API & Streamlit
        <br>
        <small>Disclaimer: Trading involves risks. Do your own research.</small>
    </div>
""", unsafe_allow_html=True)

