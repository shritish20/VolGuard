import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
# Assuming these modules are available in the same directory or Python path
from smartbhai_gpt import SmartBhaiGPT # Import SmartBhai GPT class
# Import functions from the upstox_api module
from upstox_api import initialize_upstox_client, fetch_all_api_portfolio_data, prepare_trade_orders, execute_trade_orders, square_off_positions, fetch_market_depth_by_scrip
# Import functions and the FEATURE_COLS list from data_processing module
from data_processing import load_data, generate_features, FEATURE_COLS
# Import functions from other modules that will be used in tabs
# Assuming these files and functions exist and work as intended based on the backend logic
from volatility_forecasting import forecast_volatility_future
from backtesting import run_backtest
from strategy_generation import generate_trading_strategy


# Setup logging for the Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Streamlit Page Configuration ===
# Sets basic page settings for the Streamlit app
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# === Custom CSS Styling ===
# Applies custom styling to the Streamlit elements for look and feel
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

# === Initialize Session State ===
# Session state is crucial in Streamlit to preserve variables across reruns
# Initialize state variables with default values if they don't exist
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False # Flag to trigger backtest
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None # Store backtest results
# if "violations" not in st.session_state: # Not used in provided code, can be removed
#     st.session_state.violations = 0
# if "journal_complete" not in st.session_state: # Not used in provided code, can be removed
#     st.session_state.journal_complete = False
if "trades" not in st.session_state:
    st.session_state.trades = [] # Store list of executed trades (prepared orders)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False # Login status flag
if "client" not in st.session_state:
    st.session_state.client = None # Store the initialized Upstox client object(s)
if "real_time_market_data" not in st.session_state:
    st.session_state.real_time_market_data = None # Store real-time market data fetched from API
if "api_portfolio_data" not in st.session_state:
    st.session_state.api_portfolio_data = {} # Store portfolio data fetched from API
if "prepared_orders" not in st.session_state:
    st.session_state.prepared_orders = None # Store orders prepared by the strategy
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None # Store the main DataFrame with generated features
if "forecast_log" not in st.session_state:
    st.session_state.forecast_log = None # Store volatility forecast log data (e.g., daily forecast values)
if "forecast_metrics" not in st.session_state:
    st.session_state.forecast_metrics = None # Store volatility forecast key metrics
if "generated_strategy" not in st.session_state:
    st.session_state.generated_strategy = None # Store the generated trading strategy details
if "backtest_cumulative_pnl_chart_data" not in st.session_state:
    st.session_state.backtest_cumulative_pnl_chart_data = None # Store data for backtest Cumulative PnL chart
# if "active_strategy_details" not in st.session_state: # Not used in provided code, can be removed
#     st.session_state.active_strategy_details = None
if "order_placement_errors" not in st.session_state:
    st.session_state.order_placement_errors = [] # Store errors encountered during order placement
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # Store chat history for SmartBhai GPT chat interface
# Note: 'query_input' linked to st.text_input below manages its state automatically with key.
# if "query_input" not in st.session_state:
#     st.session_state.query_input = ""

# Initialize default configuration values in session state (linked to sidebar inputs)
if "capital" not in st.session_state:
    st.session_state.capital = 1000000  # Default trading capital
if "risk_tolerance" not in st.session_state:
    st.session_state.risk_tolerance = "Moderate"  # Default risk tolerance
if "forecast_horizon" not in st.session_state:
    st.session_state.forecast_horizon = 7  # Default forecast horizon in days
# Initialize backtest date inputs with defaults if not in state
if "backtest_start_date" not in st.session_state:
    st.session_state.backtest_start_date = datetime.now().date() - timedelta(days=365)
if "backtest_end_date" not in st.session_state:
    st.session_state.backtest_end_date = datetime.now().date()
# Initialize backtest strategy filter with default
if "backtest_strategy" not in st.session_state:
    st.session_state.backtest_strategy = "All Strategies"


# === Initialize SmartBhai GPT ===
# Initialize the GPT model once when the app starts. Wrap in try-except for safety.
smartbhai_gpt = None
try:
    # Pass responses_file path. Assumes responses.csv is in the same directory or accessible.
    smartbhai_gpt = SmartBhaiGPT(responses_file="responses.csv")
    logger.info("SmartBhai GPT initialized successfully.")
except FileNotFoundError:
    st.sidebar.error("Bhai, SmartBhai GPT responses.csv file not found. Chat will be unavailable.")
    logger.error("SmartBhai GPT initialization failed: responses.csv not found.")
except Exception as e:
    st.sidebar.error(f"Bhai, SmartBhai GPT load nahi hua: {str(e)}. Chat will be unavailable.")
    logger.error(f"SmartBhai GPT initialization failed: {str(e)}", exc_info=True)


# === Helper Function: Check Upstox API hours for live trading actions ===
# This function checks if the current time is within standard Indian market hours for equity derivatives
def is_market_hours():
    """Checks if current time is within standard market hours (9:15 AM to 3:30 PM IST) and it's a weekday."""
    now = datetime.now()
    market_open_time = datetime.time(9, 15)
    market_close_time = datetime.time(15, 30)
    is_weekday = now.weekday() < 5 # Monday=0, Sunday=6
    # Check if it's a weekday AND current time is between market open and close times
    return is_weekday and (market_open_time <= now.time() <= market_close_time)


# === Helper Function: Fetch and Summarize Portfolio Data ===
# This function fetches various portfolio details using the Upstox API client
def fetch_portfolio_data(upstox_client, capital):
    """
    Fetches and summarizes user portfolio data (funds, positions, orders, trades)
    using the Upstox API via the client dictionary.

    Args:
        upstox_client (dict): The initialized Upstox API client dictionary.
        capital (float): The user's total trading capital (from session state).

    Returns:
        dict: A dictionary summarizing key portfolio metrics (weekly_pnl, margin_used, exposure, total_capital).
              Returns default/zero values if client is invalid or data fetching fails.
    """
    portfolio_summary = {
        "weekly_pnl": 0.0,
        "margin_used": 0.0,
        "exposure": 0.0,
        "total_capital": capital
    }
    # Check if client is initialized and available
    if not upstox_client:
        logger.warning("Upstox client not available for fetching portfolio data summary.")
        return portfolio_summary # Return default summary if client is missing

    try:
        # Fetch all portfolio-related data using the API client functions from upstox_api
        # This function internally handles fetching holdings, margin, positions, order_book, trade_book
        portfolio_data = fetch_all_api_portfolio_data(upstox_client)
        # Store the raw API response data in session state so other parts can access details
        st.session_state.api_portfolio_data = portfolio_data

        # Extract and summarize relevant information for the dashboard metrics
        margin_data = portfolio_data.get("margin", {}).get("data", {})
        positions_data = portfolio_data.get("positions", {}).get("data", [])

        # Ensure margin_data is a dictionary before accessing keys
        if isinstance(margin_data, dict):
            # Get utilized margin, default to 0.0 if key missing or data invalid
            portfolio_summary["margin_used"] = pd.to_numeric(margin_data.get("utilized_margin"), errors='coerce').fillna(0.0)
        else:
            logger.warning("Margin data not in expected dictionary format.")
            portfolio_summary["margin_used"] = 0.0 # Default to 0 if format is wrong


        # Calculate total PnL from positions (unrealized MTM + realized profit)
        # Ensure positions_data is a list before iterating
        if isinstance(positions_data, list):
             # Sum PnL, ensuring each item is a dictionary and keys exist
             portfolio_summary["weekly_pnl"] = sum(
                 pd.to_numeric(pos.get("unrealized_mtm"), errors='coerce').fillna(0.0) +
                 pd.to_numeric(pos.get("realized_profit"), errors='coerce').fillna(0.0)
                 for pos in positions_data if isinstance(pos, dict)
             )
        else:
            logger.warning("Positions data not in expected list format.")
            portfolio_summary["weekly_pnl"] = 0.0 # Default to 0 if format is wrong


        # Calculate exposure percentage (Utilized Margin as a percentage of Total Capital)
        # Ensure capital is a valid number and not zero to avoid division errors
        capital_numeric = pd.to_numeric(capital, errors='coerce').fillna(0.0)
        portfolio_summary["exposure"] = (portfolio_summary["margin_used"] / capital_numeric * 100) if capital_numeric > 0 else 0.0

        logger.info("Portfolio data fetched and summarized successfully.")
        return portfolio_summary

    except Exception as e:
        logger.error(f"Error fetching and summarizing portfolio data: {str(e)}", exc_info=True)
        # Return the default summary in case of any error during the fetch/summary process
        return portfolio_summary


# === Helper Function: Calculate position PnL with latest LTP ===
# This function attempts to update position PnL using the latest LTP by fetching market depth
def calculate_position_pnl_with_ltp(upstox_client, positions_data):
    """
    Attempts to update the PnL for each position in the positions_data list
    by fetching the latest LTP using fetch_market_depth_by_scrip.

    Args:
        upstox_client (dict): The initialized Upstox API client dictionary.
        positions_data (list): A list of position dictionaries as fetched from the API.

    Returns:
        list: A new list of position dictionaries with 'CurrentPnL' and 'LTP' keys added/updated.
              Returns the original list if client is invalid or positions_data is not a list.
    """
    # Return original data if client is not available or input data is invalid/empty
    if not upstox_client or not positions_data or not isinstance(positions_data, list):
        logger.warning("Client or positions data not available or invalid for PnL calculation with LTP.")
        return positions_data # Return original data without modification

    updated_positions = []
    # Iterate through each position in the provided list
    for pos in positions_data:
        # Ensure the position item is a dictionary before processing
        if not isinstance(pos, dict):
            updated_positions.append(pos) # Append item as is if not a dictionary
            continue # Skip to the next item

        # Extract essential data from the position dictionary
        instrument_key = pos.get("instrument_key")
        buy_avg_price = pd.to_numeric(pos.get("buy_avg_price"), errors='coerce').fillna(0.0)
        sell_avg_price = pd.to_numeric(pos.get("sell_avg_price"), errors='coerce').fillna(0.0)
        quantity = pd.to_numeric(pos.get("quantity"), errors='coerce').fillna(0) # Quantity can be positive (long) or negative (short)

        # Proceed only if a valid instrument_key and non-zero quantity are present
        if instrument_key and quantity != 0:
            try:
                # Fetch market data (specifically LTP) for the specific instrument key
                # Assumes fetch_market_depth_by_scrip returns a dictionary with 'Data' key
                market_data_response = fetch_market_depth_by_scrip(upstox_client, instrument_key=instrument_key)

                # Extract LTP from the market data response structure
                # Check if market_data_response is valid and contains the expected structure
                # Safely access LTP using .get() and handle potential None/NaN
                ltp = pd.to_numeric(market_data_response.get("Data", [{}])[0].get("LastTradedPrice"), errors='coerce').fillna(0.0) if market_data_response else 0.0

                # Calculate position PnL based on quantity and LTP vs average price
                current_pnl = 0.0 # Initialize current PnL for this position
                if ltp is not None and not pd.isna(ltp) and ltp > 0: # Calculate PnL only if LTP is valid and positive
                    if quantity > 0: # Long position (bought)
                         current_pnl = quantity * (ltp - buy_avg_price)
                    elif quantity < 0: # Short position (sold)
                         # Note: abs(quantity) is used because quantity is negative for short positions
                         current_pnl = abs(quantity) * (sell_avg_price - ltp)
                    # Store the calculated current PnL and the fetched LTP in the position dictionary
                    pos['CurrentPnL'] = current_pnl
                    pos['LTP'] = ltp
                else:
                    # If LTP is not available, invalid, or zero, fall back to unrealized_mtm from original data
                    logger.warning(f"LTP not available or invalid for {instrument_key}. Using 'unrealized_mtm'. LTP: {ltp}")
                    pos['CurrentPnL'] = pd.to_numeric(pos.get("unrealized_mtm"), errors='coerce').fillna(0.0)
                    pos['LTP'] = 0.0 # Indicate LTP was not fetched or was invalid

            except Exception as e:
                # Log error if fetching LTP or calculating PnL for this specific position fails
                logger.warning(f"Could not fetch LTP or calculate PnL for {instrument_key}: {e}. Using 'unrealized_mtm'.")
                # In case of any error during fetch or calculation for this position, use unrealized_mtm as fallback
                pos['CurrentPnL'] = pd.to_numeric(pos.get("unrealized_mtm"), errors='coerce').fillna(0.0)
                pos['LTP'] = 0.0 # Indicate LTP was not fetched due to error

        else:
             # If instrument_key is missing or quantity is zero, use existing PnL data and set LTP to 0
             # Ensure unrealized_mtm is numeric, default to 0.0 if not
             pos['CurrentPnL'] = pd.to_numeric(pos.get("unrealized_mtm"), errors='coerce').fillna(0.0)
             pos['LTP'] = 0.0 # No LTP available or relevant

        updated_positions.append(pos) # Add the processed position (either updated or original) to the new list
    return updated_positions


# === Streamlit App Layout and Logic ===

# Sidebar for Login and Controls
with st.sidebar:
    st.header("🔑 Upstox Login")
    # Input for access token, masked as password. Added a unique key.
    access_token = st.text_input("Access Token", type="password", key="access_token_input") # Added key

    # Login button. Added a unique key.
    if st.button("Login to Upstox", key="login_button"): # Added key
        if not access_token:
            st.error("❌ Access token cannot be empty.")
            logger.error("Login attempted with empty access token.")
        else:
            # Attempt to initialize the Upstox client using the provided token
            client_objects = initialize_upstox_client(access_token)
            if client_objects:
                # Store the initialized client objects (including token) in session state
                st.session_state.client = client_objects
                st.session_state.logged_in = True
                st.success("✅ Logged in to Upstox!")
                logger.info("Upstox login successful.")
                # Trigger a rerun to update the UI based on login status
                st.rerun()
            else:
                st.session_state.logged_in = False
                st.session_state.client = None
                st.error("❌ Login failed. Invalid or expired access token. Get a new token from Upstox.")
                logger.error("Upstox login failed: Invalid or expired token.")

    # Show trading controls only if logged in
    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        # Input for trading capital, linked to session state. Added a unique key.
        capital = st.number_input("Capital (₹)", min_value=100000, value=st.session_state.capital, step=100000, format="%d", key="capital_input") # Added key
        st.session_state.capital = capital # Update session state

        # Select box for risk tolerance, linked to session state. Added a unique key.
        risk_options = ["Conservative", "Moderate", "Aggressive"]
        risk_tolerance = st.selectbox("Risk Profile", risk_options, index=risk_options.index(st.session_state.risk_tolerance), key="risk_tolerance_input") # Added key
        st.session_state.risk_tolerance = risk_tolerance # Update session state

        # Slider for forecast horizon, linked to session state. Added a unique key.
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, st.session_state.forecast_horizon, key="forecast_horizon_input") # Added key
        st.session_state.forecast_horizon = forecast_horizon # Update session state

        st.markdown("---")
        st.markdown("**Backtest Parameters**")
        # Date inputs for backtest period, linked to session state. Added unique keys.
        # Default dates are initialized in session state section above.
        start_date = st.date_input("Start Date", value=st.session_state.backtest_start_date, key="backtest_start_date_input") # Added key
        st.session_state.backtest_start_date = start_date # Update session state
        end_date = st.date_input("End Date", value=st.session_state.backtest_end_date, key="backtest_end_date_input") # Added key
        st.session_state.backtest_end_date = end_date # Update session state

        # Select box for backtest strategy filter, linked to session state. Added a unique key.
        strategy_options = ["All Strategies", "Short Straddle", "Short Strangle", "Iron Condor", "Butterfly Spread", "Iron Fly", "Calendar Spread", "Jade Lizard", "Short Put Vertical Spread", "Short Call Vertical Spread", "Short Put", "Long Put"] # Added more strategy options
        strategy_choice = st.selectbox("Backtest Strategy Filter", strategy_options, index=strategy_options.index(st.session_state.backtest_strategy), key="backtest_strategy_filter_input") # Added key
        st.session_state.backtest_strategy = strategy_choice # Update session state

        # Button to trigger the backtest. Added a unique key.
        if st.button("Run Backtest", key="run_backtest_button"): # Added key
            # Set flag to True in session state to signal the main script body to run the backtest
            st.session_state.backtest_run = True
            st.session_state.backtest_results = None  # Reset previous results
            st.session_state.backtest_cumulative_pnl_chart_data = None # Reset previous chart data
            st.info("Backtest started! Check the Backtest tab for results.")
            logger.info("Backtest initiated from sidebar.")
            # Rerun the app to proceed to the backtest logic in the main body
            st.rerun()

        st.markdown("---")
        # Button to square off all positions. Added a unique key.
        if st.button("Square Off All Positions", key="square_off_button"): # Added key
            # Check if client is logged in and available and market is open for trading actions
            if st.session_state.logged_in and st.session_state.client:
                if is_market_hours():
                     st.info("Attempting to square off all open positions...")
                     # Call the square_off_positions function from upstox_api module
                     # This function handles fetching positions and placing orders
                     success = square_off_positions(st.session_state.client)
                     if success:
                         st.success("✅ Square off process initiated successfully! Check Portfolio tab for status.")
                         logger.info("Square off process initiated.")
                         # Note: Squaring off is asynchronous. Orders are placed,
                         # their status needs to be monitored in the Order Book/Positions.
                     else:
                         st.error("❌ Failed to initiate square off process. Check logs for details.")
                         logger.error("Failed to initiate square off process.")
                else:
                    st.warning("⏰ Market is closed. Cannot square off positions outside market hours (9:15 AM - 3:30 PM IST).")

            else:
                st.error("Not logged in to Upstox.")
                logger.warning("Square off attempted without login.")


# === Main App Title and Description ===
st.title("🛡️ VolGuard Pro")
st.markdown("**Your AI-powered options trading cockpit for NIFTY 50**")

# Check market hours for live trading data/actions (informational banner)
# This banner informs the user if they are outside live market hours
if not is_market_hours():
    st.warning("⚠️ Outside standard market hours (9:15 AM–3:30 PM IST, Mon-Fri). Live trading actions (order placement/square off) and possibly live LTP updates may be limited.")
    logger.info("App running outside standard market hours.")

# === Data Loading and Feature Generation ===
# This section runs every time the app reruns. It loads fresh data.
data_load_success = False
data_source_tag = "Unknown"
logger.info("Starting data loading process in Streamlit app.")

# Attempt to load data using the client (which attempts API first) if logged in, otherwise fallback to CSV
try:
    # Pass the client dictionary if logged in, otherwise pass None to signal CSV fallback
    client_for_load = st.session_state.client if st.session_state.logged_in else None
    # The load_data function handles the API vs CSV logic internally and returns a DataFrame, real_data dict, and source tag
    df_loaded, real_data_loaded, data_source_tag = load_data(client_for_load)

    # Check if data was successfully loaded (df_loaded is not None and not empty)
    if df_loaded is not None and not df_loaded.empty:
        logger.info(f"Raw data loaded successfully from {data_source_tag}. Generating features.")
        # Store the fetched real-time data dictionary in session state
        st.session_state.real_time_market_data = real_data_loaded if real_data_loaded is not None else {}
        # Generate features using the loaded DataFrame and the latest real-time data
        # Pass a copy of the DataFrame to prevent potential modification issues
        analysis_df = generate_features(df_loaded.copy(), st.session_state.real_time_market_data, st.session_state.capital)

        # Check if feature generation was successful
        if analysis_df is not None and not analysis_df.empty:
            st.session_state.analysis_df = analysis_df # Store the final analysis DataFrame in session state
            data_load_success = True
            logger.info("Features generated successfully.")
        else:
            st.error("❌ Failed to generate features from loaded data. Check data processing module logs for details.")
            logger.error("Feature generation failed during app load.")
            st.session_state.analysis_df = None # Ensure state is None on feature generation failure
            st.session_state.real_time_market_data = {} # Ensure state is empty if feature gen fails

    else:
         # Data loading (API or CSV) failed to return a valid DataFrame
         logger.error(f"Data loading failed: load_data returned None or empty DataFrame from {data_source_tag}. Check data loading logic and sources.")
         st.error(f"❌ VolGuard Pro could not load necessary data from {data_source_tag}. Please ensure API credentials are correct (if using API) or CSV files are available.")
         st.session_state.analysis_df = None # Ensure state is None on data loading failure
         st.session_state.real_time_market_data = real_data_loaded if real_data_loaded is not None else {} # Still store real_data if fetched

except Exception as e:
    # Catch any unexpected errors during the data loading or feature generation process
    logger.critical(f"Critical error during data loading or feature generation: {str(e)}", exc_info=True)
    st.error(f"❌ A critical error occurred during data loading or processing: {str(e)}. Please check application logs.")
    data_load_success = False
    st.session_state.analysis_df = None # Ensure state is None on critical error
    st.session_state.real_time_market_data = {} # Ensure state is empty on critical error


# Stop the app execution if data loading and feature generation was completely unsuccessful
# Most parts of the app depend on st.session_state.analysis_df
if not data_load_success or st.session_state.analysis_df is None or st.session_state.analysis_df.empty:
    logger.critical("App halting due to data loading or feature generation failure. Analysis DataFrame is not available.")
    # Only stop if analysis_df is critical for proceeding. Market Snapshot can show real_data even if analysis_df fails.
    # Let's reconsider st.stop(). If real_time_market_data loaded, Market Snapshot can partially function.
    # Only truly stop if NO data at all is available.
    if st.session_state.real_time_market_data is None or not st.session_state.real_time_market_data:
         st.error("❌ VolGuard Pro cannot function without essential market data.")
         st.stop() # Stop execution if no data was loaded at all
    else:
         # Allow app to continue, but sections depending on analysis_df will show warnings.
         st.warning("⚠️ Historical data or feature generation failed. Some sections may not be available.")
         logger.warning("App continuing with only real-time data due to historical data/feature failure.")


# === Main Content Tabs ===
# Define the tabs for the main content area of the application
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Market Snapshot",
    "🔮 Volatility Forecast",
    "🤖 Trading Strategy",
    "💼 Portfolio",
    "📝 Journal",
    "📈 Backtest",
    "⚠️ Risk Dashboard",
    "💬 SmartBhai" # Added SmartBhai tab explicitly
])

# Tab 1: Market Snapshot
with tab1:
    st.header("📊 Market Snapshot")
    # Display market data if available in session state
    if st.session_state.real_time_market_data:
        market_data = st.session_state.real_time_market_data
        logger.debug(f"Displaying Market Snapshot using real_time_market_data: {market_data.keys()}")

        # Use columns for a cleaner layout of key market metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            nifty_spot = market_data.get('nifty_spot', 'N/A')
            # Format metric value if it's a number, otherwise display as string
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
        # Display a preview of the option chain DataFrame if available within real_time_market_data
        option_chain_df_preview = market_data.get("option_chain")
        if option_chain_df_preview is not None and isinstance(option_chain_df_preview, pd.DataFrame) and not option_chain_df_preview.empty:
            # Select relevant columns for display in the preview
            display_cols = ["StrikeRate", "CPType", "LastRate", "IV", "OpenInterest", "Volume", "ScripCode"]
            # Ensure columns exist before displaying
            display_cols = [col for col in display_cols if col in option_chain_df_preview.columns]
            st.dataframe(option_chain_df_preview[display_cols].head(20)) # Show the first 20 rows
        else:
            st.info("Option chain data not available in real-time data.")
            logger.warning("Option chain data missing or invalid in real_time_market_data for Market Snapshot.")

    else:
        # This block is shown if real_time_market_data is None or empty, even if analysis_df failed
        st.info("No real-time market data available. Please check login status and data loading.")
        logger.warning("real_time_market_data is not available for Market Snapshot.")

    # Display data source information if data loading was attempted
    st.markdown(f"**Data Source**: {data_source_tag}")


# Tab 2: Volatility Forecast
with tab2:
    st.header("🔮 Volatility Forecast")
    # Run forecast only if analysis data is available (forecast model needs historical features)
    if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
        try:
            logger.info("Running volatility forecast.")
            # Run the volatility forecasting model using the analysis DataFrame and forecast horizon from session state
            # Pass a copy of the DataFrame to the forecasting function to prevent modifications
            forecast_log_df, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(st.session_state.analysis_df.copy(), st.session_state.forecast_horizon)

            # Store the forecast results and metrics in session state
            st.session_state.forecast_log = forecast_log_df
            st.session_state.forecast_metrics = {
                'forecasted_vix': blended_vols[-1] if blended_vols is not None and len(blended_vols) > 0 else None, # Get the last forecast value
                'vix_range_low': np.min(blended_vols) if blended_vols is not None and len(blended_vols) > 0 else None,
                'vix_range_high': np.max(blended_vols) if blended_vols is not None and len(blended_vols) > 0 else None,
                'confidence': confidence_score,
                'rmse': rmse,
                'feature_importances': feature_importances,
                'realized_vol': realized_vol
            }
            logger.info("Volatility forecast completed successfully.")

            # Display forecast metrics using columns
            metrics = st.session_state.forecast_metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Safely get and format the forecasted VIX metric
                forecasted_vix = metrics.get('forecasted_vix', 'N/A')
                st.metric("Forecasted VIX", f"{forecasted_vix:.2f}" if isinstance(forecasted_vix, (int, float)) else str(forecasted_vix))
            with col2:
                 # Safely get and format the VIX Range Low metric
                vix_range_low = metrics.get('vix_range_low', 'N/A')
                st.metric("VIX Range Low", f"{vix_range_low:.2f}" if isinstance(vix_range_low, (int, float)) else str(vix_range_low))
            with col3:
                 # Safely get and format the VIX Range High metric
                vix_range_high = metrics.get('vix_range_high', 'N/A')
                st.metric("VIX Range High", f"{vix_range_high:.2f}" if isinstance(vix_range_high, (int, float)) else str(vix_range_high))
            with col4:
                 # Safely get and format the Confidence metric
                 confidence = metrics.get('confidence', 'N/A')
                 st.metric("Confidence", f"{confidence:.1f}%" if isinstance(confidence, (int, float)) else str(confidence))

            st.subheader("Forecast Trend (Blended)")
            # Display the forecast trend chart
            # Ensure forecast_log is available and contains the 'Blended_Vol' column
            if st.session_state.forecast_log is not None and not st.session_state.forecast_log.empty and 'Blended_Vol' in st.session_state.forecast_log.columns:
                try:
                    # Combine historical VIX and forecast for a complete trend line
                    # Show a recent portion of historical VIX (e.g., last 30 days)
                    historical_vix = st.session_state.analysis_df['VIX'].tail(30).copy() # Ensure 'VIX' column exists and is numeric
                    # Ensure the forecast log DataFrame has a datetime index for plotting
                    forecast_chart_data = st.session_state.forecast_log.set_index('Date')['Blended_Vol'].copy() # Ensure 'Date' and 'Blended_Vol' exist

                    # Combine the historical and forecasted series
                    combined_vols = pd.concat([historical_vix, forecast_chart_data])
                    combined_vols = combined_vols.rename("VIX_Combined") # Rename series for chart legend

                    st.line_chart(combined_vols)
                except Exception as chart_e:
                     st.warning(f"Could not generate forecast trend chart: {chart_e}. Check data format.")
                     logger.error(f"Error generating forecast trend chart: {chart_e}")

            else:
                st.info("Forecast data not available for charting.")
                logger.warning("Forecast log data missing or invalid for charting.")


            st.subheader("Model Metrics")
            # Safely display RMSE and Realized Volatility
            if metrics.get('rmse') is not None:
                 st.write(f"XGBoost RMSE: {metrics['rmse']:.2f}")
            if metrics.get('realized_vol') is not None:
                 st.write(f"Recent Realized Volatility: {metrics['realized_vol']:.2f}")


            st.subheader("Feature Importances (XGBoost)")
            # Display feature importances if available
            if metrics.get('feature_importances') is not None and st.session_state.analysis_df is not None:
                try:
                    # Create a DataFrame for plotting feature importances
                    feature_importances_df = pd.DataFrame({
                        'Feature': FEATURE_COLS, # Use the defined FEATURE_COLS list
                        'Importance': metrics['feature_importances']
                    }).sort_values('Importance', ascending=False)
                    st.bar_chart(feature_importances_df.set_index('Feature'))
                except Exception as fi_e:
                     st.warning(f"Could not display feature importances: {fi_e}. Check data format.")
                     logger.error(f"Error displaying feature importances: {fi_e}")
            else:
                 st.info("Feature importances data not available.")
                 logger.warning("Feature importances data missing or invalid.")


        except Exception as e:
            st.error(f"❌ Error generating volatility forecast: {str(e)}. Check forecasting module logs.")
            logger.error(f"Volatility forecast error: {str(e)}.", exc_info=True)
            # Reset session state variables related to forecast on error
            st.session_state.forecast_log = None
            st.session_state.forecast_metrics = None

    else:
        # Message shown if analysis_df is not available or empty
        st.info("No analysis data available for volatility forecasting. Ensure data loaded successfully.")
        logger.warning("Analysis DataFrame not available for volatility forecasting.")


# Tab 3: Trading Strategy
with tab3:
    st.header("🤖 Trading Strategy")
    # Generate strategy only if analysis data, real-time market data, and forecast metrics are available
    if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty and st.session_state.real_time_market_data and st.session_state.forecast_metrics is not None:
        try:
            logger.info("Generating trading strategy.")
            # Generate trading strategy using available data and user parameters from session state
            # Pass copies of DataFrames/dicts to prevent unintended modifications
            strategy = generate_trading_strategy(
                st.session_state.analysis_df.copy(),
                st.session_state.real_time_market_data.copy(),
                st.session_state.forecast_metrics.copy(),
                st.session_state.capital,
                st.session_state.risk_tolerance
            )
            st.session_state.generated_strategy = strategy # Store the generated strategy details

            if strategy and isinstance(strategy, dict):
                # Display the recommended strategy details
                st.subheader("Recommended Strategy")
                st.markdown(f"**Strategy**: {strategy.get('Strategy', 'N/A')}")
                st.markdown(f"**Confidence**: {strategy.get('Confidence', 0.0):.2%}")
                # Safely format Deploy amount as currency
                deploy_amount = strategy.get('Deploy', 0.0)
                st.markdown(f"**Deploy Amount**: ₹{deploy_amount:.2f}" if isinstance(deploy_amount, (int, float)) else f"Deploy Amount: {deploy_amount}")
                st.markdown(f"**Reasoning**: {strategy.get('Reasoning', 'N/A')}")

                # --- Order Preparation and Execution Buttons ---
                st.subheader("Order Actions")

                # Button to prepare orders based on the generated strategy. Added a unique key.
                if st.button("Prepare Orders", key="prepare_orders_button"): # Added key
                    # Check if market is open before preparing LIVE orders that use real-time data
                    if is_market_hours():
                         st.info("Preparing orders based on the recommended strategy...")
                         # Call the prepare_trade_orders function from upstox_api module
                         # Pass the generated strategy, latest real-time data, and capital
                         prepared_orders = prepare_trade_orders(
                            st.session_state.generated_strategy.copy(), # Pass a copy
                            st.session_state.real_time_market_data.copy(), # Pass a copy
                            st.session_state.capital
                        )
                         st.session_state.prepared_orders = prepared_orders # Store prepared orders in session state
                         st.session_state.order_placement_errors = [] # Clear any previous order errors

                         if prepared_orders and isinstance(prepared_orders, list) and len(prepared_orders) > 0:
                             st.success(f"✅ {len(prepared_orders)} Orders prepared successfully!")
                             st.subheader("Prepared Orders Preview")
                             # Display prepared orders in a DataFrame for review
                             st.dataframe(pd.DataFrame(prepared_orders))
                             logger.info(f"{len(prepared_orders)} Orders prepared successfully.")
                         else:
                             st.warning("⚠️ Failed to prepare orders. Check strategy details and logs.")
                             logger.warning("Order preparation resulted in no orders or failed.")
                             st.session_state.prepared_orders = None # Ensure state is None if preparation failed

                    else:
                         st.warning("⏰ Market is closed. Cannot prepare live orders outside market hours (9:15 AM - 3:30 PM IST).")
                         st.session_state.prepared_orders = None # Ensure state is None if market is closed


                # Show execute button only if orders have been prepared and user is logged in and market is open
                # Added a unique key.
                if st.session_state.prepared_orders is not None and st.session_state.logged_in and st.session_state.client:
                    # Check market hours again before displaying the execute button for live trading
                    if is_market_hours():
                         # Use a form for the execute button to allow clearing state or showing message after click
                         with st.form("execute_orders_form", clear_on_submit=False): # Use a form to manage button click state
                              st.subheader("Ready to Execute")
                              st.warning("🚨 IMPORTANT: Clicking 'Execute Orders' will place REAL orders in your Upstox account. Proceed with caution.")
                              execute_button_clicked = st.form_submit_button("Execute Orders", help="Click to place prepared orders in your Upstox account.") # Added key implicitly via form_submit_button

                              if execute_button_clicked:
                                  st.info("Attempting to execute orders...")
                                  # Call the execute_trade_orders function from upstox_api module
                                  # This function handles placing orders via API and collects responses/errors
                                  success, response_details = execute_trade_orders(
                                      st.session_state.client, # Pass the initialized client dictionary
                                      st.session_state.prepared_orders.copy() # Pass a copy of prepared orders
                                  )

                                  # Process the response from execute_trade_orders
                                  if success:
                                      st.success("✅ Orders executed successfully!")
                                      logger.info("Orders executed successfully via API.")
                                      # Add the executed orders (or details) to the journal/trades list in session state
                                      # Note: This just adds the *prepared* order details to a list,
                                      # actual trade confirmation should ideally come from the Trade Book/Order Book.
                                      st.session_state.trades.extend(st.session_state.prepared_orders)
                                      st.session_state.prepared_orders = None # Clear prepared orders after successful execution
                                      st.session_state.order_placement_errors = [] # Clear errors

                                      # Optionally, trigger a rerun to refresh portfolio data in the Portfolio tab
                                      # st.experimental_rerun() # Use st.rerun() in newer Streamlit versions if needed

                                  else:
                                      st.error("❌ Order execution failed. See details below.")
                                      logger.error(f"Order execution failed. Details: {response_details}")
                                      # Store the error details received from the execution function
                                      st.session_state.order_placement_errors.append(response_details)
                                      # The prepared orders are NOT cleared automatically on failure,
                                      # allowing the user to potentially try executing again or modify.

                                  # Display the API response details for debugging/confirmation
                                  st.subheader("Execution Response Details")
                                  st.json(response_details) # Display the raw response dictionary/list


                    else:
                         st.warning("⏰ Market is closed. Cannot execute live orders outside market hours (9:15 AM - 3:30 PM IST).")

                elif st.session_state.prepared_orders is not None:
                     # Show message if orders prepared but not logged in
                     st.info("Please log in to Upstox to execute prepared orders.")


                # Display any recent order placement errors stored in session state
                if st.session_state.order_placement_errors:
                     st.subheader("Recent Order Placement Errors")
                     for error in st.session_state.order_placement_errors:
                         st.json(error) # Display each error entry as JSON


            else:
                # Message shown if no strategy was generated
                st.info("⚠️ No strategy generated based on current market conditions and parameters.")
                st.session_state.generated_strategy = None # Ensure state is None if no strategy


        except Exception as e:
            st.error(f"❌ Error generating strategy: {str(e)}. Check strategy generation module logs.")
            logger.error(f"Trading strategy generation error: {str(e)}.", exc_info=True)
            # Reset session state variables related to strategy on error
            st.session_state.generated_strategy = None
            st.session_state.prepared_orders = None
            st.session_state.order_placement_errors = []


    else:
        # Message shown if essential data for strategy generation is missing
        st.info("No analysis data, real-time market data, or forecast available for strategy generation. Ensure data loaded and forecast completed.")
        logger.warning("Essential data missing for trading strategy generation.")


# Tab 4: Portfolio
with tab4:
    st.header("💼 Portfolio")
    # Fetch and display portfolio summary and positions if logged in
    if st.session_state.logged_in and st.session_state.client:
        logger.info("Fetching portfolio data for display.")
        # Fetch portfolio data dynamically every time the tab is viewed or app reruns
        # This updates st.session_state.api_portfolio_data internally
        portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)

        # Display portfolio summary metrics using columns
        col1, col2, col3 = st.columns(3)
        with col1:
            # Safely display Weekly PnL
            weekly_pnl = portfolio_summary.get('weekly_pnl', 0.0)
            st.metric("Weekly PnL", f"₹{weekly_pnl:.2f}")
        with col2:
            # Safely display Margin Used
            margin_used = portfolio_summary.get('margin_used', 0.0)
            st.metric("Margin Used", f"₹{margin_used:.2f}")
        with col3:
            # Safely display Exposure
            exposure = portfolio_summary.get('exposure', 0.0)
            st.metric("Exposure", f"{exposure:.2f}%")

        st.subheader("Open Positions")
        # Get raw positions data from session state (fetched by fetch_portfolio_data)
        positions_data_raw = st.session_state.api_portfolio_data.get("positions", {}).get("data", [])
        # Calculate PnL using live LTP if available and update the positions data list
        # This function calls fetch_market_depth_by_scrip internally for each position
        positions_data_updated = calculate_position_pnl_with_ltp(st.session_state.client, positions_data_raw)

        # Display open positions in a DataFrame
        if positions_data_updated and isinstance(positions_data_updated, list):
            positions_df = pd.DataFrame(positions_data_updated)
            if not positions_df.empty:
                # Select relevant columns for display and handle potential missing columns
                display_cols = ["instrument_key", "quantity", "buy_avg_price", "sell_avg_price", "LTP", "CurrentPnL", "unrealized_mtm", "realized_profit", "product"]
                display_cols = [col for col in display_cols if col in positions_df.columns] # Ensure columns exist

                st.dataframe(positions_df[display_cols])
            else:
                st.info("No open positions.")
        else:
            st.info("No open positions or position data unavailable.")
            logger.warning("Position data missing or invalid for display in Portfolio tab.")


        st.subheader("Fund Summary")
        fund_data = st.session_state.api_portfolio_data.get("margin", {}).get("data", {})
        if fund_data and isinstance(fund_data, dict):
             # Select relevant fund details keys
             fund_keys = ["available_margin", "utilized_margin", "total_margin", "payin_amount", "payout_amount"]
             # Filter fund_data to only include relevant keys and convert to DataFrame row
             fund_display_data = {key: fund_data.get(key) for key in fund_keys if key in fund_data}

             if fund_display_data:
                  fund_df = pd.DataFrame([fund_display_data]) # Convert dictionary to DataFrame row
                  # Ensure numeric columns before formatting, handle errors
                  for col in fund_df.columns:
                      if pd.api.types.is_numeric_dtype(fund_df[col]):
                           fund_df[col] = pd.to_numeric(fund_df[col], errors='coerce')
                           fund_df[col] = fund_df[col].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else 'N/A') # Format currency

                  st.dataframe(fund_df)
             else:
                  st.info("Fund data not available or empty.")
                  logger.warning("Fund data missing or empty in API portfolio data.")
        else:
             st.info("Fund data not available.")
             logger.warning("Fund data missing or not in dictionary format in API portfolio data.")


        st.subheader("Order Book")
        order_data = st.session_state.api_portfolio_data.get("order_book", {}).get("data", [])
        if order_data and isinstance(order_data, list):
            order_df = pd.DataFrame(order_data)
            if not order_df.empty:
                 # Select and format relevant order details, handle missing columns
                 order_display_cols = ["instrument_key", "quantity", "transaction_type", "order_type", "status", "price", "average_price", "placed_by", "order_id", "exchange_timestamp"]
                 order_display_cols = [col for col in order_display_cols if col in order_df.columns]
                 if not order_df.empty:
                    st.dataframe(order_df[order_display_cols])
                 else:
                    st.info("No pending or recent orders in the order book.")
            else:
                 st.info("Order book data is empty.")
        else:
            st.info("Order book data not available.")
            logger.warning("Order book data missing or not in list format in API portfolio data.")


        st.subheader("Trade History")
        trade_data = st.session_state.api_portfolio_data.get("trade_book", {}).get("data", [])
        if trade_data and isinstance(trade_data, list):
            trade_df = pd.DataFrame(trade_data)
            if not trade_df.empty:
                 # Select and format relevant trade details, handle missing columns
                 trade_display_cols = ["instrument_key", "quantity", "transaction_type", "exchange_timestamp", "price", "order_id", "trade_id"]
                 trade_display_cols = [col for col in trade_display_cols if col in trade_df.columns]
                 if not trade_df.empty:
                    st.dataframe(trade_df[trade_display_cols])
                 else:
                    st.info("No recent trades in the trade history.")
            else:
                st.info("Trade history data is empty.")
        else:
            st.info("Trade history data not available.")
            logger.warning("Trade history data missing or not in list format in API portfolio data.")


    else:
        # Message shown if not logged in
        st.info("Please log in to Upstox to view your portfolio.")
        logger.info("User not logged in, Portfolio tab showing login message.")


# Tab 5: Journal
with tab5:
    st.header("📝 Trading Journal")
    # Define the journal file name
    journal_file = "journal_log.csv"
    # Load existing journal entries from the CSV file
    try:
        if os.path.exists(journal_file):
            journal_df = pd.read_csv(journal_file)
            # Ensure 'Date' column is treated as datetime
            journal_df['Date'] = pd.to_datetime(journal_df['Date'], errors='coerce')
            # Drop rows where Date could not be parsed
            journal_df = journal_df.dropna(subset=['Date'])
        else:
            # Create an empty DataFrame with the expected columns if the file doesn't exist
            journal_df = pd.DataFrame(columns=["Date", "Strategy", "PnL", "Notes"])
        logger.info(f"Journal loaded successfully. {len(journal_df)} entries found.")
    except Exception as e:
        st.error(f"❌ Error loading journal file: {str(e)}")
        logger.error(f"Error loading journal file: {str(e)}", exc_info=True)
        # Start with an empty DataFrame on error to prevent further issues
        journal_df = pd.DataFrame(columns=["Date", "Strategy", "PnL", "Notes"])


    # Form for adding new journal entries
    # Use a unique key for the form
    with st.form("journal_form", clear_on_submit=True): # Use clear_on_submit=True to clear fields after submission
        st.subheader("Log a New Trade")
        # Input fields linked to session state with unique keys for persistence and clearing
        # Initialize default values in session state if needed, or use the form's default values
        date_input_value = st.session_state.get("journal_date_input", datetime.now().date())
        date_log = st.date_input("Trade Date", value=date_input_value, key="journal_date_input") # Added key
        st.session_state.journal_date_input = date_log # Update session state on change

        strategy_input_value = st.session_state.get("journal_strategy_input", "")
        strategy_log = st.text_input("Strategy", value=strategy_input_value, key="journal_strategy_input") # Added key
        st.session_state.journal_strategy_input = strategy_log # Update session state on change

        pnl_input_value = st.session_state.get("journal_pnl_input", 0.0)
        pnl_log = st.number_input("PnL (₹)", format="%.2f", value=pnl_input_value, key="journal_pnl_input") # Added key
        st.session_state.journal_pnl_input = pnl_log # Update session state on change

        notes_input_value = st.session_state.get("journal_notes_input", "")
        notes_log = st.text_area("Notes", value=notes_input_value, key="journal_notes_input") # Added key
        st.session_state.journal_notes_input = notes_log # Update session state on change

        # Submit button for the form
        submitted = st.form_submit_button("Log Trade")

        # Process form submission
        if submitted:
            # Create a new entry as a DataFrame row
            new_entry = pd.DataFrame({
                "Date": [date_log],
                "Strategy": [strategy_log],
                "PnL": [pnl_log],
                "Notes": [notes_log]
            })
            # Ensure 'Date' column in new entry is datetime
            new_entry['Date'] = pd.to_datetime(new_entry['Date'])

            # Append the new entry to the existing journal DataFrame
            # Use pd.concat to append DataFrames
            journal_df = pd.concat([journal_df, new_entry], ignore_index=True)
            logger.info("New journal entry added to DataFrame.")

            try:
                # Save the updated DataFrame back to the CSV file
                journal_df.to_csv(journal_file, index=False)
                st.success("✅ Trade logged successfully!")
                logger.info("Journal updated successfully saved to CSV.")
                # The form inputs are automatically cleared because clear_on_submit=True

                # Optional: Rerun to immediately display the updated table
                # st.rerun() # Rerun is not strictly needed with clear_on_submit=True but can force display update

            except Exception as e:
                 st.error(f"❌ Error saving journal entry: {str(e)}")
                 logger.error(f"Error saving journal entry to CSV: {str(e)}", exc_info=True)


    st.subheader("Trade History")
    # Display the full trade history from the journal DataFrame
    if not journal_df.empty:
        # Sort journal entries by date descending for display
        # Ensure 'Date' is datetime before sorting
        if 'Date' in journal_df.columns:
             journal_df_display = journal_df.sort_values(by='Date', ascending=False).copy()
             st.dataframe(journal_df_display)
        else:
             st.warning("Journal DataFrame is missing the 'Date' column. Cannot display or sort.")
             st.dataframe(journal_df) # Display without sorting if date column missing
    else:
        st.info("No trades logged yet.")
        logger.info("Journal DataFrame is empty, showing message.")


# Tab 6: Backtest
with tab6:
    st.header("📈 Backtest Results")
    # Check if backtest should run based on the sidebar button flag
    # This flag is set to True when the "Run Backtest" button in the sidebar is clicked.
    if st.session_state.backtest_run and st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
        st.info("Running backtest simulation...")
        logger.info("Backtest trigger detected. Starting backtest run.")
        try:
            # Ensure backtest start and end dates are valid and within the analysis data range
            start_date = st.session_state.backtest_start_date
            end_date = st.session_state.backtest_end_date
            analysis_df = st.session_state.analysis_df

            # Check if the selected backtest date range is valid
            if start_date >= end_date:
                 st.error("❌ Backtest Error: Start date must be before end date.")
                 logger.error("Backtest date range invalid: Start date is >= End date.")
                 st.session_state.backtest_run = False # Reset flag
                 st.session_state.backtest_results = None # Clear results
                 st.session_state.backtest_cumulative_pnl_chart_data = None
            elif pd.to_datetime(start_date) < analysis_df.index.min() or pd.to_datetime(end_date) > analysis_df.index.max():
                 st.error(f"❌ Backtest Error: Selected date range ({start_date} to {end_date}) is outside available data range ({analysis_df.index.min().date()} to {analysis_df.index.max().date()}).")
                 logger.error(f"Backtest date range outside available data: {start_date}-{end_date} vs {analysis_df.index.min().date()}-{analysis_df.index.max().date()}.")
                 st.session_state.backtest_run = False # Reset flag
                 st.session_state.backtest_results = None # Clear results
                 st.session_state.backtest_cumulative_pnl_chart_data = None
            else:
                 # Run the backtest using analysis data and parameters from session state
                 # Pass a copy of the DataFrame to prevent modifications by the backtest function
                 backtest_results = run_backtest(
                     analysis_df.copy(),
                     start_date,
                     end_date,
                     st.session_state.capital,
                     st.session_state.backtest_strategy
                 )
                 st.session_state.backtest_results = backtest_results # Store results in session state
                 st.session_state.backtest_run = False # Reset the flag after running the backtest

                 if backtest_results and isinstance(backtest_results, dict):
                     st.success("✅ Backtest completed!")
                     logger.info("Backtest completed successfully.")

                     st.subheader("Key Performance Metrics")
                     # Display key backtest metrics using columns, safely accessing dict keys
                     col1, col2, col3 = st.columns(3)
                     with col1:
                         total_return = backtest_results.get('total_return', 0.0)
                         st.metric("Total Return", f"{total_return:.2%}")
                     with col2:
                         sharpe_ratio = backtest_results.get('sharpe_ratio', 0.0)
                         st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                     with col3:
                         max_drawdown = backtest_results.get('max_drawdown', 0.0)
                         st.metric("Max Drawdown", f"{max_drawdown:.2%}")

                     st.subheader("Cumulative PnL Chart")
                     # Display the cumulative PnL chart if data is available and is a pandas Series
                     cumulative_pnl_data = backtest_results.get("cumulative_pnl")
                     if cumulative_pnl_data is not None and isinstance(cumulative_pnl_data, pd.Series):
                         st.session_state.backtest_cumulative_pnl_chart_data = cumulative_pnl_data # Store chart data in session state
                         st.line_chart(cumulative_pnl_data)
                     else:
                          st.warning("Cumulative PnL data not available or in unexpected format for charting.")
                          logger.warning("Cumulative PnL data missing or invalid for backtest chart.")


                     st.subheader("Trade Log Preview")
                     # Display a preview of the backtest trade log DataFrame if available
                     trade_log_df = backtest_results.get("trade_log")
                     if trade_log_df is not None and isinstance(trade_log_df, pd.DataFrame) and not trade_log_df.empty:
                          st.dataframe(trade_log_df.head()) # Show first few rows of the trade log
                     else:
                          st.info("Backtest trade log is empty or not available.")
                          logger.warning("Backtest trade log data missing or invalid.")


                 else:
                     st.error("❌ Backtest failed to return results. Check backtesting module logs for details.")
                     logger.error("Backtest failed to return results.")
                     st.session_state.backtest_results = None # Reset state on failure
                     st.session_state.backtest_cumulative_pnl_chart_data = None


        except Exception as e:
            st.error(f"❌ An error occurred while running the backtest: {str(e)}. Check backtesting module logs.")
            logger.error(f"Backtest execution error: {str(e)}.", exc_info=True)
            # Reset state variables on error
            st.session_state.backtest_run = False
            st.session_state.backtest_results = None
            st.session_state.backtest_cumulative_pnl_chart_data = None

    # Display results if a backtest has already been run and results are stored in session state
    elif st.session_state.backtest_results is not None:
        logger.info("Displaying existing backtest results.")
        # Retrieve stored results and display them
        metrics = st.session_state.backtest_results

        st.subheader("Key Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_return = metrics.get('total_return', 0.0)
            st.metric("Total Return", f"{total_return:.2%}")
        with col2:
            sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        with col3:
            max_drawdown = metrics.get('max_drawdown', 0.0)
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")

        st.subheader("Cumulative PnL Chart")
        # Use the stored chart data if available
        if st.session_state.backtest_cumulative_pnl_chart_data is not None:
             st.line_chart(st.session_state.backtest_cumulative_pnl_chart_data)
        else:
             st.warning("Cumulative PnL data not available for charting.")
             logger.warning("Stored Cumulative PnL data not available for charting.")


        st.subheader("Trade Log Preview")
        trade_log_df = metrics.get("trade_log")
        if trade_log_df is not None and isinstance(trade_log_df, pd.DataFrame) and not trade_log_df.empty:
             st.dataframe(trade_log_df.head())
        else:
             st.info("Backtest trade log is empty or not available.")
             logger.warning("Stored backtest trade log data missing or invalid.")


    else:
        # Message shown if no backtest has been triggered or results are not stored
        st.info("Run a backtest from the sidebar to see results here.")
        logger.info("Backtest tab: No backtest results available to display.")


# Tab 7: Risk Dashboard
with tab7:
    st.header("⚠️ Risk Dashboard")
    # Display risk metrics if analysis data is available
    if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
        logger.info("Displaying Risk Dashboard.")
        try:
            # Get the latest data row from the analysis DataFrame for risk calculation context
            # Ensure the DataFrame is not empty before accessing the last row
            latest_data = st.session_state.analysis_df.iloc[-1]
            logger.debug(f"Latest data for risk dashboard: {latest_data.index}")

            # Calculate simplified risk metrics based on available data
            # Ensure 'PnL_Day' exists and is numeric for VaR/Max Loss approximation
            # Use .get() with a default value and pd.to_numeric for safety
            pnl_day = pd.to_numeric(latest_data.get("PnL_Day"), errors='coerce').fillna(0.0)

        try:
            # Get the latest data row from the analysis DataFrame for risk calculation context
            # Ensure the DataFrame is not empty before accessing the last row
            if st.session_state.analysis_df is None or st.session_state.analysis_df.empty:
                logger.warning("Analysis DataFrame is empty. Cannot calculate risk metrics dependent on it.")
                # Return minimal risk metrics or None if analysis_df is essential
                pnl_day = 0.0
                latest_data = None # Ensure latest_data is None if DF is empty
            else:
                latest_data = st.session_state.analysis_df.iloc[-1]
                logger.debug(f"Latest data for risk dashboard: {latest_data.index}")
                # Calculate simplified risk metrics based on available data
                # Ensure 'PnL_Day' exists and is numeric for VaR/Max Loss approximation
                # Use .get() with a default value and pd.to_numeric for safety
                pnl_day = pd.to_numeric(latest_data.get("PnL_Day"), errors='coerce').fillna(0.0)


            # --- Calculate metrics *before* defining the dictionary ---
            # Calculate latest_vix value here using the corrected logic and pd.notna()
            temp_vix_real = pd.to_numeric(st.session_state.real_time_market_data.get("vix"), errors='coerce') if st.session_state.real_time_market_data else np.nan
            temp_vix_analysis = pd.to_numeric(latest_data.get("VIX"), errors='coerce') if latest_data is not None and latest_data.get("VIX") is not None else np.nan

            # Prioritize VIX from real_time_market_data, fallback to analysis_df, then default 15.0
            latest_vix = temp_vix_real if pd.notna(temp_vix_real) else (temp_vix_analysis if pd.notna(temp_vix_analysis) else 15.0)

            # Determine volatility regime based on the calculated latest_vix
            regime = "High" if latest_vix > 20 else ("Medium" if latest_vix > 15 else "Low")
            # You could add an "Event" regime check here if latest_data is available and has 'Event_Flag'
            # if latest_data is not None and int(latest_data.get('Event_Flag', 0)) == 1:
            #     regime = "Event"


            # --- Define the Risk_metrics dictionary using the calculated values ---
            risk_metrics = {
                # Simplified VaR (Value at Risk) and Max Loss approximations based on daily PnL std dev
                # Use the calculated pnl_day
                "VaR_95": abs(pnl_day) * 1.65,  # Approx 95th percentile potential loss
                "Max_Loss": abs(pnl_day) * 2.33,  # Approx 99th percentile potential loss
                # Use the calculated regime string for the Volatility_Regime key
                "Volatility_Regime": regime
                # You can add other calculated metrics here if needed
                # "Latest_VIX_Value": latest_vix # Optionally add the calculated VIX value itself
            }
            logger.debug(f"Calculated risk metrics: {risk_metrics}")

            # Fetch portfolio summary again to get the latest exposure for the gauge/metric display
            # This ensures the exposure metric is based on the most recent API data if logged in.
            # This call is necessary here because the Portfolio tab's display uses this function
            # and we need up-to-date exposure for the risk dashboard as well.
            portfolio_summary_for_risk = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
            logger.debug(f"Fetched portfolio summary for risk display: Exposure={portfolio_summary_for_risk.get('exposure', 0.0):.2f}%")


            # Display risk metrics using columns
            st.subheader("Key Risk Metrics") # Added subheader for clarity
            col1, col2, col3 = st.columns(3)
            with col1:
                # Safely display VaR (95%) - use .get() on the dictionary
                var_95 = risk_metrics.get('VaR_95', 0.0)
                st.metric("VaR (95%)", f"₹{var_95:.2f}")
            with col2:
                # Safely display Max Loss (99%) - use .get() on the dictionary
                max_loss = risk_metrics.get('Max_Loss', 0.0)
                st.metric("Max Loss (99%)", f"₹{max_loss:.2f}")
            with col3:
                # Display volatility regime as a styled badge - use the calculated 'regime' variable
                # Determine CSS class based on volatility regime
                regime_class = {
                    "Low": "regime-low",
                    "Medium": "regime-medium",
                    "High": "regime-high",
                    "Event": "regime-event" # Include if Event regime is used
                }.get(regime, "regime-medium") # Default to medium if regime is unknown

                # Use markdown with HTML to display the styled badge
                st.markdown(f"<span class='regime-badge {regime_class}'>{regime} Regime</span>", unsafe_allow_html=True) # Added " Regime" for clarity


            st.subheader("Risk Exposure (Margin Used / Capital)")
            # Display exposure using a simple metric
            # Safely get exposure from the portfolio summary fetched above
            current_exposure = portfolio_summary_for_risk.get('exposure', 0.0)
            st.metric("Current Exposure", f"{current_exposure:.2f}%")
            # The original gauge div was commented out as it's just a static graphic placeholder in the CSS


        except Exception as e:
            st.error(f"❌ Error generating risk dashboard: {str(e)}. Check logs.")
            logger.error(f"Risk dashboard error: {str(e)}.", exc_info=True)



                "Volatility_Regime": "High" if latest_vix > 20 else ("Medium" if latest_vix > 15 else "Low")
                # You could add an "Event" regime if there's an upcoming major event detected (e.g., using Event_Flag)
                # "Volatility_Regime": "Event" if latest_data.get("Event_Flag", 0) == 1 else ("High" if latest_vix > 20 else ("Medium" if latest_vix > 15 else "Low"))
            }
            logger.debug(f"Calculated risk metrics: {risk_metrics}")


            # Fetch portfolio summary again to get the latest exposure for the gauge/metric display
            # This ensures the exposure metric is based on the most recent API data if logged in.
            portfolio_summary_for_risk = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
            logger.debug(f"Fetched portfolio summary for risk display: Exposure={portfolio_summary_for_risk.get('exposure', 0.0):.2f}%")


            # Display risk metrics using columns
            col1, col2, col3 = st.columns(3)
            with col1:
                # Safely display VaR (95%)
                var_95 = risk_metrics.get('VaR_95', 0.0)
                st.metric("VaR (95%)", f"₹{var_95:.2f}")
            with col2:
                # Safely display Max Loss (99%)
                max_loss = risk_metrics.get('Max_Loss', 0.0)
                st.metric("Max Loss (99%)", f"₹{max_loss:.2f}")
            with col3:
                # Display volatility regime as a styled badge
                regime = risk_metrics.get('Volatility_Regime', 'N/A')
                # Determine CSS class based on volatility regime
                regime_class = {
                    "Low": "regime-low",
                    "Medium": "regime-medium",
                    "High": "regime-high",
                    "Event": "regime-event" # Include if Event regime is used
                }.get(regime, "regime-medium") # Default to medium if regime is unknown

                # Use markdown with HTML to display the styled badge
                st.markdown(f"<span class='regime-badge {regime_class}'>{regime} Regime</span>", unsafe_allow_html=True) # Added " Regime" for clarity


            st.subheader("Risk Exposure (Margin Used / Capital)")
            # Display exposure using a simple metric
            # Safely get exposure from the portfolio summary
            current_exposure = portfolio_summary_for_risk.get('exposure', 0.0)
            st.metric("Current Exposure", f"{current_exposure:.2f}%")
            # The original gauge div was commented out as it's just a static graphic placeholder in the CSS


        except Exception as e:
            st.error(f"❌ Error generating risk dashboard: {str(e)}. Check logs.")
            logger.error(f"Risk dashboard error: {str(e)}.", exc_info=True)

    else:
        # Message shown if analysis_df is not available
        st.info("No analysis data available for risk dashboard. Ensure data loaded successfully.")
        logger.warning("Analysis DataFrame not available for Risk Dashboard.")


# Tab 8: SmartBhai GPT Chat Interface
with tab8:
    st.header("💬 SmartBhai GPT")
    st.markdown("<div class='smartbhai-container'>", unsafe_allow_html=True) # Chat container div
    st.markdown("<div class='smartbhai-title'>Ask about IV, strategies, or market buzz!</div>", unsafe_allow_html=True) # Chat title
    st.markdown("<div class='smartbhai-subtitle'>Bhai ko pucho market ka haal!</div>", unsafe_allow_html=True) # Chat subtitle

    # Display chat interface ONLY if SmartBhai GPT was initialized successfully
    if smartbhai_gpt:
        logger.info("SmartBhai GPT is available. Rendering chat interface.")
        with st.container(): # Use a container for chat elements

            # Chat history display area with fixed height and scroll using custom CSS class
            st.markdown("<div class='smartbhai-chat'>", unsafe_allow_html=True)
            # Iterate through chat history stored in session state and display messages
            for chat in st.session_state.chat_history:
                # Check if the chat entry is a dictionary with required keys before displaying
                if isinstance(chat, dict) and "role" in chat and "message" in chat:
                    if chat["role"] == "user":
                        # User bubble aligned to the right using CSS class
                        st.markdown(f"<div class='chat-bubble user-bubble'>{chat['message']}</div>", unsafe_allow_html=True)
                    else: # Role is assumed to be "assistant" for SmartBhai's responses
                        # SmartBhai bubble aligned to the left using CSS class
                        st.markdown(f"<div class='chat-bubble smartbhai-bubble'>{chat['message']}</div>", unsafe_allow_html=True)
                else:
                    logger.warning(f"Skipping invalid chat history entry: {chat}") # Log invalid entries
            st.markdown("</div>", unsafe_allow_html=True) # Close chat history div


            # Input form for new messages. Use a form to manage input and button behavior.
            # Use clear_on_submit=True to automatically clear the text input after sending a message.
            with st.form("smartbhai_form", clear_on_submit=True):
                # Text input for user query. Linked to session state via key.
                # Using label_visibility="collapsed" to hide the label above the input box for cleaner look.
                query = st.text_input(
                    "Ask SmartBhai:", # Label (will be hidden)
                    key="smartbhai_query_input", # Unique key for this widget's state in session state
                    placeholder="Bhai, IV kya hai? Strategy kya lagau?", # Placeholder text
                    label_visibility="collapsed" # Hide the label
                )
                # Button to send the message. Use container_width to make it span the column.
                submit_button = st.form_submit_button(
                    "Send", # Button text
                    use_container_width=True, # Make button fill container width
                    help="Send message to SmartBhai GPT" # Tooltip text
                )

                # Process form submission if the submit button was clicked and the query is not empty
                if submit_button and query:
                    # Add user's query to the chat history stored in session state
                    st.session_state.chat_history.append({"role": "user", "message": query})

                    # Generate response from SmartBhai GPT instance
                    try:
                        # Pass the user's query to the generate_response method
                        response = smartbhai_gpt.generate_response(query)
                        # Add SmartBhai's response to the chat history
                        st.session_state.chat_history.append({"role": "assistant", "message": response})
                        logger.info(f"SmartBhai generated response for query: '{query[:50]}...'")
                    except Exception as e:
                        # Handle errors during response generation
                        logger.error(f"Error generating SmartBhai response for query '{query[:50]}...': {str(e)}", exc_info=True)
                        # Add a fallback error message to the chat history
                        st.session_state.chat_history.append({"role": "assistant", "message": "Sorry, Bhai, kuch technical issue hai. Try again! 🙏 (Error generating response)"})

                    # Rerun the app to update the chat history display with the new messages
                    # This is necessary because chat history display happens at the top of the form/container
                    st.rerun()

    else:
        # Show an error message if SmartBhai GPT failed to initialize
        st.error("SmartBhai GPT is not available. Check initialization logs.")
        logger.warning("SmartBhai GPT instance is None, chat interface not rendered.")

    st.markdown("</div>", unsafe_allow_html=True) # Close chat container div


# === Footer ===
st.markdown("""
    <div class='footer'>
        VolGuard Pro | Built with ❤️ by Shritish | Powered by Upstox API & Streamlit
        <br>
        <small>Disclaimer: Trading involves risks. Do your own research and consult a financial advisor.</small>
    </div>
""", unsafe_allow_html=True) # Allow HTML for the footer styling
