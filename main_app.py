# Open your file: streamlit_app.py
# Replace the ENTIRE content of this file with the code below.

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from smartbhai_gpt import SmartBhaiGPT  # Import SmartBhai GPT class (Make sure smartbhai_gpt.py is in the same directory or accessible)

# Import modularized components (Make sure these files are in the same directory or accessible)
from fivepaisa_api import initialize_5paisa_client, fetch_all_api_portfolio_data, prepare_trade_orders, execute_trade_orders, square_off_positions, fetch_market_depth_by_scrip
from data_processing import load_data, generate_features, FEATURE_COLS
from volatility_forecasting import forecast_volatility_future
from backtesting import run_backtest
from strategy_generation import generate_trading_strategy

# Setup logging for the main app file
# Configure logging to write to stdout so it appears in the Streamlit terminal logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for app-specific logging


# --- Page Configuration ---
# Set basic configuration for the Streamlit page
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")


# --- Custom CSS Styling ---
# Inject custom CSS to style the Streamlit components for a better look and feel.
# This includes styling for background, tabs, buttons, cards, chat bubbles, etc.
st.markdown("""
    <style>
        /* Main page background and text color */
        .main { background: linear-gradient(135deg, #1a1a2e, #0f1c2e); color: #e5e5e5; font-family: 'Inter', sans-serif; }

        /* Style for the tabs container */
        .stTabs [data-baseweb="tab-list"] { background: #16213e; border-radius: 10px; padding: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
        /* Style for individual tabs */
        .stTabs [data-baseweb="tab"] { color: #a0a0a0; font-weight: 500; padding: 10px 20px; border-radius: 8px; transition: all 0.3s ease; }
        /* Style for the currently selected tab */
        .stTabs [data-baseweb="tab"][aria-selected="true"] { background: #e94560; color: white; font-weight: 700; box-shadow: 0 2px 8px rgba(233, 69, 96, 0.5); }
        /* Style for tabs on hover */
        .stTabs [data-baseweb="tab"]:hover { background: #2a2a4a; color: white; }

        /* Style for sidebar buttons */
        .sidebar .stButton>button { width: 100%; background: #0f3460; color: white; border-radius: 10px; padding: 12px; margin: 5px 0; border: none; transition: all 0.3s ease; }
        /* Style for sidebar buttons on hover */
        .sidebar .stButton>button:hover { transform: scale(1.02); background: #e94560; }

        /* Style for main content cards/containers */
        .card { background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9)); border-radius: 15px; padding: 20px; margin: 15px 0; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4); transition: transform 0.3s ease; }
        /* Style for main content cards on hover */
        .card:hover { transform: translateY(-5px); }

        /* Style for strategy carousel container (if multiple strategy cards are used) */
        .strategy-carousel { display: flex; overflow-x: auto; gap: 20px; padding: 10px; -webkit-overflow-scrolling: touch; /* Enable momentum scrolling on iOS */ scrollbar-width: thin; scrollbar-color: #e94560 #16213e; }
        /* Custom scrollbar for the carousel */
        .strategy-carousel::-webkit-scrollbar { height: 8px; }
        .strategy-carousel::-webkit-scrollbar-track { background: #16213e; border-radius: 10px; }
        .strategy-carousel::-webkit-scrollbar-thumb { background: #e94560; border-radius: 10px; }

        /* Style for individual strategy cards within the carousel */
        .strategy-card { flex: 0 0 auto; width: 300px; background: #16213e; border-radius: 15px; padding: 20px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); transition: transform 0.3s ease; }
        /* Style for individual strategy cards on hover */
        .strategy-card:hover { transform: scale(1.03); }

        /* Style for Streamlit metrics */
        .stMetric { background: rgba(15, 52, 96, 0.7); border-radius: 15px; padding: 15px; text-align: center; margin: 5px; }
         /* Adjust font size for the delta value in metrics */
         .stMetric > div[data-testid="stMetricDelta"] { font-size: 1rem; }

        /* Style for gauge (used for confidence score) */
        .gauge { width: 100px; height: 100px; border-radius: 50%; background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px; margin: 10px auto; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3); }

        /* Style for market regime badges */
        .regime-badge { padding: 8px 15px; border-radius: 20px; font-weight: bold; font-size: 14px; text-transform: uppercase; margin-bottom: 15px; display: inline-block; }
        .regime-low { background: #28a745; color: white; } /* Green */
        .regime-medium { background: #ffc107; color: black; } /* Yellow */
        .regime-high { background: #dc3545; color: white; } /* Red */
        .regime-event { background: #ff6f61; color: white; } /* Orange */

        /* Style for general alert banners (like risk flags from strategy generator) */
        .alert-banner { background: #dc3545; color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px; font-weight: bold; }

        /* Style for main action buttons (Run Analysis, Prepare Orders, Place Orders) */
        .stButton>button { background: #e94560; color: white; border-radius: 10px; padding: 12px 25px; font-size: 16px; font-weight: 600; border: none; transition: all 0.3s ease; margin-top: 10px; }
        /* Style for main action buttons on hover */
        .stButton>button:hover { transform: scale(1.02); background: #ffcc00; color: #1a1a2e; }

        /* Style for the footer */
        .footer { text-align: center; padding: 20px; color: #a0a0a0; font-size: 14px; border-top: 1px solid rgba(255, 255, 255, 0.1); margin-top: 30px; }

        /* Enhanced SmartBhai GPT Widget Styling */
        .smartbhai-container {
            background: #1e2a44;
            border-radius: 12px;
            padding: 15px;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        /* Style for the SmartBhai input text box */
        .smartbhai-input > div > div > input {
            background: #2a3b5a;
            border: 2px solid #00cc00; /* Green border */
            border-radius: 10px;
            padding: 12px;
            color: #e5e5e5;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        /* Style for the SmartBhai input text box when focused */
        .smartbhai-input > div > div > input:focus {
            border-color: #ffcc00; /* Yellow border on focus */
            outline: none;
            box-shadow: 0 0 5px rgba(255, 204, 0, 0.5);
        }
        /* Style for the SmartBhai 'Ask' button */
        .smartbhai-button > button {
            width: 100%;
            background: #e94560; /* Red */
            color: white;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            margin-top: 10px;
        }
         /* Style for the SmartBhai 'Ask' button on hover */
        .smartbhai-button > button:hover {
            background: #ffcc00; /* Yellow */
            color: #1a1a2e;
            transform: translateY(-2px);
        }
        /* Style for the chat history container */
        .smartbhai-chat {
            max-height: 300px; /* Limit height and enable scrolling */
            overflow-y: auto;
            padding: 10px;
            margin-top: 15px;
            background: #16213e; /* Dark blue background */
            border-radius: 10px;
        }
        /* Base style for chat bubbles */
        .chat-bubble {
            margin: 10px 0;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%; /* Limit bubble width */
            font-size: 14px;
            line-height: 1.4;
            word-wrap: break-word; /* Ensure long words break */
            overflow-wrap: break-word; /* Ensure long words break */
        }
        /* Style for user's chat bubbles */
        .user-bubble {
            background: #e94560; /* Red */
            color: white;
            margin-left: auto; /* Align to the right */
            text-align: right;
            border-bottom-right-radius: 2px; /* Sharpen corner */
        }
        /* Style for SmartBhai's chat bubbles */
        .smartbhai-bubble {
            background: #00cc00; /* Green */
            color: #1a1a2e; /* Dark text */
            margin-right: auto; /* Align to the left */
            border-bottom-left-radius: 2px; /* Sharpen corner */
        }
        /* Style for the SmartBhai widget title */
        .smartbhai-title {
            color: #ffcc00; /* Yellow */
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            text-align: center;
        }
        /* Style for the SmartBhai widget subtitle */
        .smartbhai-subtitle {
            color: #a0a0a0; /* Grey */
            font-size: 14px;
            margin-bottom: 15px;
            text-align: center;
        }

         /* Style for the risk warning banner in the main content */
        .stAlert {
            border-radius: 10px;
            margin-bottom: 15px;
        }
        /* Override Streamlit's default alert background and text color */
        .stAlert > div[data-baseweb="alert"] {
             background-color: #ffc107 !important; /* Match warning color (Yellow) */
             color: #1a1a2e !important; /* Dark text for readability */
             border-color: #ffc107 !important; /* Border color matching background */
        }
         /* Style for the alert content text */
         .stAlert > div[data-baseweb="alert"] > div[data-testid="stAlertContent"] {
            font-weight: bold;
            line-height: 1.5;
         }
         /* Style for the expander within the risk alert */
         .stAlert .stExpander {
             margin-top: 10px;
             border: none; /* Remove default expander border */
             box-shadow: none; /* Remove default expander shadow */
         }
         .stAlert .stExpander button {
             background-color: rgba(0,0,0,0.1); /* Slightly darker background for expander header */
             color: #1a1a2e; /* Dark text */
             border-radius: 5px;
             padding: 5px 10px;
         }
         .stAlert .stExpander div[data-testid="stExpanderContent"] {
             background-color: rgba(0,0,0,0.05); /* Slightly darker background for expander content */
             padding: 10px;
             border-radius: 5px;
             margin-top: 5px;
         }


    </style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
# Streamlit uses session state to remember variable values across reruns.
# We initialize state variables if they don't exist to maintain the app's state.
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None
if "violations" not in st.session_state:
    st.session_state.violations = 0 # Counter for discipline violations
if "journal_complete" not in st.session_state:
    st.session_state.journal_complete = False # Flag for journal completion status
if "trades" not in st.session_state:
    st.session_state.trades = [] # List to potentially log trades over time (future use)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False # Authentication status
if "client" not in st.session_state:
    st.session_state.client = None # 5paisa client object for API interaction
if "real_time_market_data" not in st.session_state:
    st.session_state.real_time_market_data = None # Latest real-time market data from API
if "api_portfolio_data" not in st.session_state:
    st.session_state.api_portfolio_data = {} # Full portfolio data fetched from API
if "prepared_orders" not in st.session_state:
    st.session_state.prepared_orders = None # List of orders prepared for placing
if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None # DataFrame with historical and featured data for analysis
if "forecast_log" not in st.session_state:
    st.session_state.forecast_log = None # DataFrame containing the volatility forecast
if "forecast_metrics" not in st.session_state:
    st.session_state.forecast_metrics = None # Dictionary storing key forecast metrics (confidence, rmse, etc.)
if "generated_strategy" not in st.session_state:
    st.session_state.generated_strategy = None # Dictionary storing the recommended trading strategy details
if "backtest_cumulative_pnl_chart_data" not in st.session_state:
    st.session_state.backtest_cumulative_pnl_chart_data = None # Data for plotting the cumulative P&L chart from backtest
if "active_strategy_details" not in st.session_state:
    st.session_state.active_strategy_details = None # Details of the strategy currently being displayed/tracked
if "order_placement_errors" not in st.session_state:
    st.session_state.order_placement_errors = [] # List to store error messages from order placement attempts
# Initialize chat history and query input for SmartBhai GPT
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # List of chat messages (query-response pairs)
if "query_input" not in st.session_state:
    st.session_state.query_input = "" # State variable to control the value of the chat input box

# --- Initialize a state variable for SmartBhai's proactive risk warning ---
# This variable will store the result from the assess_trade_risk function.
# It can be None (no warning) or a dictionary {"main_message": str, "explanations": list[str]}.
if "risk_warning" not in st.session_state:
    st.session_state.risk_warning = None
    logger.info("Session state variable 'risk_warning' initialized.")

# --- Initialize a state variable to hold the journal data ---
# This will be used for behavioral nudges later.
if "journal_df" not in st.session_state:
     st.session_state.journal_df = None
     # Load journal data here on startup if the file exists
     journal_log_path = "journal_log.csv"
     if os.path.exists(journal_log_path):
          try:
               st.session_state.journal_df = pd.read_csv(journal_log_path, encoding='utf-8')
               st.session_state.journal_df['Date'] = pd.to_datetime(st.session_state.journal_df['Date']) # Ensure Date is datetime
               logger.info(f"Journal data loaded from {journal_log_path}")
          except Exception as e:
               logger.error(f"Error loading journal data from {journal_log_path}: {e}", exc_info=True)
               st.session_state.journal_df = pd.DataFrame() # Initialize empty if loading fails
     else:
          st.session_state.journal_df = pd.DataFrame() # Initialize empty if file doesn't exist
          logger.info(f"Journal log file not found at {journal_log_path}. Initializing empty DataFrame.")


# --- Initialize SmartBhai GPT ---
# Create an instance of your SmartBhaiGPT class. This happens once per session.
smartbhai_gpt = None # Initialize as None
try:
    # Pass the path to your responses CSV. Ensure responses.csv is in the same directory as smartbhai_gpt.py.
    smartbhai_gpt = SmartBhaiGPT(responses_file="responses.csv")
    logger.info("SmartBhaiGPT instance created successfully.")
except FileNotFoundError:
    st.sidebar.error("Error loading SmartBhai GPT: responses.csv not found. Please check your project files and path.")
    logger.error("Failed to create SmartBhaiGPT instance: responses.csv not found.")
except Exception as e:
    st.sidebar.error(f"Bhai, SmartBhai GPT load nahi hua: {str(e)}. Check logs for details.")
    logger.error(f"Failed to create SmartBhaiGPT instance: {e}", exc_info=True)


# --- Helper Functions ---
# These functions are used within the app to fetch or process data.

def fetch_portfolio_data(client, capital):
    """Fetches and summarizes key portfolio metrics from the API."""
    logger.info("Fetching portfolio summary data.")
    # Initialize a dictionary to store the summarized portfolio metrics
    portfolio_summary = {
        "weekly_pnl": 0.0, # Example PnL metric
        "margin_used": 0.0,
        "exposure": 0.0, # Calculated exposure percentage
        "total_capital": capital # User's defined capital
    }
    # Check if the 5paisa client is available and successfully logged in
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available for fetching portfolio data.")
        return portfolio_summary # Return the initialized summary if client is not ready

    try:
        # Fetch all detailed portfolio data using the API function from your module
        portfolio_data = fetch_all_api_portfolio_data(client)
        st.session_state.api_portfolio_data = portfolio_data # Store the full fetched data in session state

        # Populate the summary dictionary from the detailed fetched data
        margin_data = portfolio_data.get("margin", {}) # Get margin data dictionary
        positions_data = portfolio_data.get("positions", []) # Get positions data list

        if isinstance(margin_data, dict):
            portfolio_summary["margin_used"] = margin_data.get("UtilizedMargin", 0.0) # Get utilized margin

        if isinstance(positions_data, list):
             # Calculate total PnL by summing booked and unrealized PnL from all positions
            portfolio_summary["weekly_pnl"] = sum(pos.get("BookedPL", 0.0) + pos.get("UnrealizedMTM", 0.0) for pos in positions_data if isinstance(pos, dict))

        # Calculate exposure percentage based on utilized margin and total capital
        portfolio_summary["exposure"] = (portfolio_summary["margin_used"] / capital * 100) if capital > 0 else 0.0 # Avoid division by zero

        logger.info("Portfolio summary fetched successfully.")
        return portfolio_summary # Return the populated summary
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}", exc_info=True)
        return portfolio_summary # Return the initialized summary on error


def calculate_position_pnl_with_ltp(client, positions_data):
    """Calculates current PnL for open positions using latest traded price (LTP)."""
    logger.info("Calculating position PnL with LTP.")
    # Check if client is available, logged in, and there are positions to process
    if not client or not client.get_access_token() or not positions_data:
        logger.warning("Client not available or no positions for PnL calculation.")
        return positions_data # Return original data if cannot calculate PnL

    updated_positions = []
    # Iterate through each position in the provided list
    for pos in positions_data:
        if not isinstance(pos, dict):
            updated_positions.append(pos) # Add non-dictionary items directly
            continue

        # Extract necessary details from the position dictionary using .get() for safety
        scrip_code = pos.get("ScripCode")
        exchange = pos.get("Exch")
        exchange_type = pos.get("ExchType")
        buy_avg_price = pos.get("BuyAvgPrice", 0.0)
        sell_avg_price = pos.get("SellAvgPrice", 0.0)
        buy_qty = pos.get("BuyQty", 0)
        sell_qty = pos.get("SellQty", 0)
        net_qty = buy_qty - sell_qty # Calculate the net quantity (positive for long, negative for short)

        # Proceed with PnL calculation only if essential details are present and there is a net quantity
        if scrip_code and exchange and exchange_type and net_qty != 0:
            try:
                # Fetch the latest market depth for the specific scrip code to get LTP
                market_data = fetch_market_depth_by_scrip(client, Exchange=exchange, ExchangeType=exchange_type, ScripCode=scrip_code)
                # Extract the LastTradedPrice (LTP) safely from the market data response
                ltp = market_data["Data"][0].get("LastTradedPrice", 0.0) if market_data and market_data.get("Data") and market_data["Data"] else 0.0

                # Calculate the current PnL based on net quantity and LTP
                if net_qty > 0: # If net quantity is positive, it's a long position
                    position_pnl = net_qty * (ltp - buy_avg_price)
                else: # If net quantity is negative, it's a short position
                    position_pnl = abs(net_qty) * (sell_avg_price - ltp) # PnL = |Quantity| * (Sell Avg - LTP)

                pos['CurrentPnL'] = position_pnl # Add the calculated PnL to the position dictionary
                pos['LTP'] = ltp # Add the fetched LTP to the position dictionary
                logger.debug(f"Calculated PnL for ScripCode {scrip_code}: {position_pnl:.2f}")
            except Exception as e:
                logger.warning(f"Could not fetch LTP or calculate PnL for ScripCode {scrip_code}: {e}", exc_info=True)
                # If calculation fails, use the UnrealizedMTM provided by the API as a fallback PnL
                pos['CurrentPnL'] = pos.get("UnrealizedMTM", 0.0)
                pos['LTP'] = 0.0 # Set LTP to 0 if fetch failed

        updated_positions.append(pos) # Add the processed position (with or without calculated PnL) to the list

    logger.info("Position PnL calculation with LTP finished.")
    return updated_positions # Return the list of updated positions


# --- Sidebar Content ---
# Define the layout and content of the application's sidebar.
with st.sidebar:
    st.header("🔑 5paisa Login")
    # Input field for TOTP (Time-based One-Time Password) for 2FA
    totp_code = st.text_input("TOTP (from Authenticator App)", type="password", help="Enter the 6-digit TOTP from your authenticator app (e.g., Google Authenticator, Authy).")

    # Button to trigger the login process when clicked
    if st.button("Login to 5paisa"):
        logger.info("User clicked 'Login to 5paisa'. Attempting login.")
        # Call the initialize client function from your fivepaisa_api module
        # Pass the Streamlit secrets and the entered TOTP code
        st.session_state.client = initialize_5paisa_client(st.secrets, totp_code)
        # Check if the client was successfully initialized and an access token was obtained
        if st.session_state.client and st.session_state.client.get_access_token():
            st.session_state.logged_in = True # Set logged_in state to True
            st.success("✅ Logged in to 5paisa!") # Display success message
            logger.info("5paisa login successful.")
        else:
            st.session_state.logged_in = False # Set logged_in state to False
            st.error("❌ Login failed. Check credentials and TOTP.") # Display error message
            logger.warning("5paisa login failed.")

    # Trading Controls and Backtest Parameters section, visible only if the user is logged in
    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        # Number input for the user's total trading capital
        # Default value is retrieved from session state if available, otherwise 1,000,000
        capital = st.number_input("Capital (₹)", min_value=100000, value=st.session_state.get("capital", 1000000), step=100000, format="%d", help="Enter your total trading capital.")
        st.session_state.capital = capital # Store the entered capital in session state

        # Select box for the user's risk tolerance profile
        risk_profiles = ["Conservative", "Moderate", "Aggressive"]
        # Default value is retrieved from session state if available, otherwise "Moderate"
        risk_tolerance = st.selectbox("Risk Profile", risk_profiles, index=risk_profiles.index(st.session_state.get("risk_tolerance", "Moderate")), help="Select your risk tolerance. This can influence strategy suggestions.")
        st.session_state.risk_tolerance = risk_tolerance # Store the selected risk tolerance

        # Slider to select the horizon for volatility forecasting in days
        # Default value is retrieved from session state if available, otherwise 7 days
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, st.session_state.get("forecast_horizon", 7), help="Number of days ahead to forecast volatility.")
        st.session_state.forecast_horizon = forecast_horizon # Store the selected forecast horizon

        st.markdown("---") # Horizontal rule for visual separation
        st.markdown("**Backtest Parameters**")
        # Date input for the start date of the backtest period
        default_start_date = datetime.now().date() - timedelta(days=365) # Default to 1 year ago from today
        start_date = st.date_input("Start Date", value=st.session_state.get("backtest_start_date", default_start_date), help="Start date for historical backtesting.")
        st.session_state.backtest_start_date = start_date # Store the selected backtest start date
        # Date input for the end date of the backtest period
        default_end_date = datetime.now().date() # Default to today's date
        end_date = st.date_input("End Date", value=st.session_state.get("backtest_end_date", default_end_date), help="End date for historical backtesting.")
        st.session_state.backtest_end_date = end_date # Store the selected backtest end date

        # Select box to filter strategies during backtesting
        strategy_options = ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard", "Short Straddle", "Short Put Vertical Spread", "Short Call Vertical Spread", "Short Put", "Long Put"]
        # Default value is retrieved from session state if available, otherwise "All Strategies"
        strategy_choice = st.selectbox("Backtest Strategy Filter", strategy_options, index=strategy_options.index(st.session_state.get("backtest_strategy_choice", "All Strategies")), help="Select a specific strategy to backtest or 'All Strategies'.")
        st.session_state.backtest_strategy_choice = strategy_choice # Store the selected backtest strategy choice

        st.markdown("---") # Horizontal rule
        st.header("⚠️ Emergency Actions")
        st.warning("Use with EXTREME CAUTION!")
        # Button to square off all open positions via the API
        if st.button("🚨 Square Off All Positions"):
            logger.warning("User clicked 'Square Off All Positions'. Initiating square off.")
            # Call the square_off_positions function from your fivepaisa_api module
            if square_off_positions(st.session_state.client):
                st.success("✅ All positions squared off") # Display success message
                logger.info("Square off all positions API call successful.")
            else:
                st.error("❌ Failed to square off positions") # Display error message
                logger.error("Square off all positions API call failed.")

    # --- SmartBhai GPT Chat Widget in Sidebar ---
    st.markdown("---") # Horizontal rule to separate sections
    # Container div for the chat widget with custom styling
    st.markdown('<div class="smartbhai-container">', unsafe_allow_html=True)
    st.markdown('<div class="smartbhai-title">🗣️ SmartBhai GPT</div>', unsafe_allow_html=True)
    st.markdown('<div class="smartbhai-subtitle">Your trading copilot for options!</div>', unsafe_allow_html=True)

    # Input field for user queries to SmartBhai
    # Connects the input value to st.session_state.query_input for state management
    query = st.text_input(
        "Ask away...",
        value=st.session_state.query_input, # Initial value from session state
        key="gpt_query_input", # Unique key for this widget
        placeholder="E.g., 'What is IV?' or 'Check my straddle at 21000'", # Placeholder text when empty
        help="Ask SmartBhai about trading strategies, options, market insights, or your portfolio.", # Help text on hover
        label_visibility="collapsed" # Hide the default label as title/subtitle are used
    )

    # Button to send the user's query to SmartBhai when clicked
    if st.button("Ask SmartBhai", key="smartbhai_button"): # Unique key for this button
        logger.info(f"User clicked 'Ask SmartBhai'. Query: '{query}'")
        # Ensure there is text in the input field and the SmartBhaiGPT instance is available
        if query and smartbhai_gpt:
            with st.spinner("SmartBhai is thinking..."): # Show a spinner while the AI processes the query
                try:
                    # Call the generate_response method of the SmartBhaiGPT instance.
                    # Pass the current session state so SmartBhai has context about the app's data.
                    response = smartbhai_gpt.generate_response(query, st.session_state) # Pass st.session_state here!
                    # Append the user's query and SmartBhai's response as a dictionary to the chat history list
                    st.session_state.chat_history.append({"query": query, "response": response})
                    st.session_state.query_input = ""  # Clear the text input field after the query is sent
                    logger.info("SmartBhai response generated and added to chat history.")
                except Exception as e:
                    # Display an error message if there's an issue generating the response
                    st.error(f"Bhai, SmartBhai response generate nahi kar paya: {str(e)}")
                    logger.error(f"Error generating SmartBhai response for query '{query}': {e}", exc_info=True)
        else:
            # Provide feedback if the input was empty or SmartBhaiGPT instance is not available
            if not query:
                 st.error("Bhai, kuch type toh kar!")
                 logger.warning("Ask SmartBhai button clicked with empty query.")
            elif smartbhai_gpt is None:
                 st.error("Bhai, SmartBhai GPT load nahi hua. Check sidebar error messages.")
                 logger.warning("SmartBhaiGPT instance is None, cannot process chat query.")


    # Display the chat history using custom CSS for speech bubbles
    if st.session_state.chat_history:
        st.markdown('<div class="smartbhai-chat">', unsafe_allow_html=True) # Container for chat bubbles
        # Iterate through each chat entry (query-response pair) in the history
        for chat in st.session_state.chat_history:
            # Display the user's message in a bubble aligned to the right
            st.markdown(f'<div class="chat-bubble user-bubble">You: {chat["query"]}</div>', unsafe_allow_html=True)
            # Display SmartBhai's message in a bubble aligned to the left
            st.markdown(f'<div class="chat-bubble smartbhai-bubble">SmartBhai: {chat["response"]} 😎</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True) # Close the chat container div

    # Static SEBI disclaimer below the chat widget
    st.markdown('<div style="text-align: center; color: #a0a0a0; font-size: 12px; margin-top: 10px;">SmartBhai is a decision-support tool, not financial advice.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) # Close the smartbhai-container div


# --- Main Content Area ---
# This area is where the main analysis and dashboard tabs are displayed.
# It is visible only when the user is successfully logged in to the API.
if not st.session_state.logged_in:
    # Display a message prompting the user to log in if they are not
    st.info("Please login to 5paisa from the sidebar to access VolGuard Pro.")
else:
    # Display the main application title in the main content area
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)

    # Button to trigger the main analysis pipeline
    # This button initiates data loading, backtesting, forecasting, and strategy generation.
    run_button = st.button("📈 Run Analysis", help="Click to fetch live market data, portfolio details, run backtest, forecast volatility, and generate a strategy recommendation.")
    if run_button:
        logger.info("User clicked 'Run Analysis'. Starting full analysis pipeline.")
        # Reset key session state variables at the start of a new analysis run
        # This ensures a clean slate for the new analysis results.
        st.session_state.backtest_run = False
        st.session_state.backtest_results = None
        st.session_state.backtest_cumulative_pnl_chart_data = None
        st.session_state.prepared_orders = None # Clear any previously prepared orders
        st.session_state.analysis_df = None # Clear previous analysis data
        st.session_state.real_time_market_data = None # Clear previous real-time data
        st.session_state.forecast_log = None # Clear previous forecast
        st.session_state.forecast_metrics = None # Clear previous forecast metrics
        st.session_state.generated_strategy = None # Clear previous strategy recommendation
        st.session_state.api_portfolio_data = {} # Clear previous portfolio data
        st.session_state.active_strategy_details = None # Clear active strategy details
        st.session_state.order_placement_errors = [] # Clear previous order errors
        st.session_state.risk_warning = None # Clear any previous risk warning displayed by SmartBhai


        # --- 1. Data Loading ---
        with st.spinner("Fetching and processing data..."):
             logger.info("Analysis Step 1: Loading Data.")
             # Call load_data function from your data_processing module
             df, real_data, data_source = load_data(st.session_state.client)
             st.session_state.analysis_df = df # Store the main analysis DataFrame in session state
             st.session_state.real_time_market_data = real_data # Store the raw real-time data fetched
             st.session_state.data_source = data_source # Store information about the data source (API/CSV)
             logger.info(f"Data Loading Complete. Source: {data_source}. DataFrame shape: {df.shape if df is not None else 'None'}")


        # --- 2. Fetch Portfolio Data (if client is available) ---
        # Fetching portfolio data requires a logged-in client.
        if st.session_state.client:
             with st.spinner("Fetching portfolio data..."):
                 logger.info("Analysis Step 2: Fetching Portfolio Data.")
                 # Call fetch_all_api_portfolio_data function from your fivepaisa_api module
                 st.session_state.api_portfolio_data = fetch_all_api_portfolio_data(st.session_state.client)
                 logger.info(f"Portfolio Data Fetching Complete. Data available: {st.session_state.api_portfolio_data is not None}.")
        else:
             logger.warning("5paisa client not available, skipping portfolio data fetch during analysis run.")


        # --- 3. Feature Generation, Backtesting, Volatility Forecasting, Strategy Generation ---
        # These subsequent steps depend on successful data loading and feature generation.
        if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
            with st.spinner("Generating features and running backtest..."):
                 logger.info("Analysis Step 3: Feature Generation and Backtesting.")
                 # Generate features needed for forecasting, backtesting, and strategy generation
                 df_featured = generate_features(st.session_state.analysis_df, st.session_state.real_time_market_data, st.session_state.capital)
                 st.session_state.analysis_df = df_featured # Update analysis_df with the new featured data

                 # Proceed only if feature generation was successful and returned a non-empty DataFrame
                 if df_featured is not None and not df_featured.empty:
                     # Run the backtest using the featured data and selected parameters
                     backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf, cumulative_pnl_chart_data = run_backtest(
                         st.session_state.analysis_df,
                         st.session_state.capital,
                         st.session_state.backtest_strategy_choice,
                         st.session_state.backtest_start_date,
                         st.session_state.backtest_end_date
                     )
                     # Store all the results from the backtest
                     st.session_state.backtest_run = True # Flag that backtest has been run
                     st.session_state.backtest_results = {
                         "backtest_df": backtest_df, "total_pnl": total_pnl, "win_rate": win_rate,
                         "max_drawdown": max_drawdown, "sharpe_ratio": sharpe_ratio, "sortino_ratio": sortino_ratio,
                         "calmar_ratio": calmar_ratio, "strategy_perf": strategy_perf, "regime_perf": regime_perf
                     }
                     st.session_state.backtest_cumulative_pnl_chart_data = cumulative_pnl_chart_data # Store data for the PnL chart
                     logger.info("Backtesting Complete.")


                     # --- 4. Volatility Forecasting ---
                     with st.spinner("Predicting volatility..."):
                         logger.info("Analysis Step 4: Volatility Forecasting.")
                         # Perform volatility forecasting using the featured data and selected horizon
                         forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(st.session_state.analysis_df, st.session_state.forecast_horizon)
                         # Store the forecast results and associated metrics
                         st.session_state.forecast_log = forecast_log
                         st.session_state.forecast_metrics = {
                             "garch_vols": garch_vols, "xgb_vols": xgb_vols, "blended_vols": blended_vols,
                             "realized_vol": realized_vol, "confidence_score": confidence_score,
                             "rmse": rmse, "feature_importances": feature_importances
                         }
                         logger.info(f"Volatility Forecasting Complete. Confidence Score: {confidence_score:.1f}%")


                     # --- 5. Strategy Generation ---
                     with st.spinner("Generating trading strategy recommendation..."):
                         logger.info("Analysis Step 5: Strategy Generation.")
                         # Generate a trading strategy recommendation based on all analysis results and user parameters
                         st.session_state.generated_strategy = generate_trading_strategy(
                             st.session_state.analysis_df, # Pass analysis data
                             st.session_state.forecast_log, # Pass forecast data
                             st.session_state.forecast_metrics.get("realized_vol"), # Pass relevant metrics
                             st.session_state.risk_tolerance, # Pass user's risk tolerance
                             st.session_state.forecast_metrics.get("confidence_score"), # Pass forecast confidence
                             st.session_state.capital, # Pass user's capital
                             st.session_state.violations, # Pass user's discipline status (violations, journal complete)
                             st.session_state.journal_complete
                         )
                         # If a strategy was successfully generated and the discipline lock is not active
                         if st.session_state.generated_strategy and not st.session_state.generated_strategy.get("Discipline_Lock", False):
                             st.session_state.active_strategy_details = st.session_state.generated_strategy # Store the recommended strategy as the active one
                             logger.info(f"Strategy Generation Complete. Recommended Strategy: {st.session_state.generated_strategy.get('Strategy', 'None')}.")
                         elif st.session_state.generated_strategy and st.session_state.generated_strategy.get("Discipline_Lock", False):
                              # Display a warning if strategy generation is blocked by discipline lock
                              st.warning("⚠️ Strategy generation is currently locked due to discipline violations. Complete your journal entries.")
                              logger.warning("Strategy generation locked due to discipline violations.")
                         else:
                             # Log a warning if strategy generation failed or returned None
                             logger.warning("Strategy Generation failed or returned None.")


                 else:
                    # Error message if feature generation failed or returned empty data
                    st.error("Analysis failed: Feature generation error or returned empty featured data.")
                    logger.error("Feature generation failed or returned empty data.")
            # End of spinner for Feature Generation, Backtesting, Forecasting, Strategy
        else:
            # Error message if initial data loading failed or returned empty data
            st.error("Analysis failed: Data loading error or returned empty data.")
            logger.error("Initial data loading failed or returned empty data.")

    # --- Application Tabs ---
    # Define the tabs that organize the main content sections of the application.
    tabs = st.tabs(["📊 Snapshot", "📈 Forecast", "🧪 Strategy", "💰 Portfolio", "📝 Journal", "📉 Backtest", "🛡️ Risk Dashboard"])

    # --- Snapshot Tab Content ---
    # Content for the Market Snapshot tab.
    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True) # Container card with custom styling
        st.subheader("📊 Market Snapshot") # Tab header
        # Display key market metrics if analysis data is available (analysis_df)
        if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
            df = st.session_state.analysis_df # Get the analysis DataFrame
            latest_date = df.index[-1].strftime("%d-%b-%Y") # Get date of the latest data point and format it
            last_nifty = df["NIFTY_Close"].iloc[-1] # Get the latest NIFTY close price
            # Get previous day's NIFTY close for calculating daily change (handle case with less than 2 days data)
            prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 and "NIFTY_Close" in df.columns else last_nifty
            last_vix = df["VIX"].iloc[-1] if "VIX" in df.columns else "N/A" # Get the latest VIX, handle missing column

            # Determine the current market regime based on VIX or the generated strategy's regime
            # Prefer the regime identified by the strategy generator if available
            regime = "LOW" if (isinstance(last_vix, (int, float)) and last_vix < 15) else "MEDIUM" if (isinstance(last_vix, (int, float)) and last_vix < 20) else "HIGH"
            if st.session_state.generated_strategy and "Regime" in st.session_state.generated_strategy:
                regime = st.session_state.generated_strategy["Regime"]

            # Determine the appropriate CSS class for the regime badge based on the regime
            regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"}.get(regime, "regime-low")

            # Display the market regime badge centered using markdown and custom CSS
            st.markdown(f'<div style="text-align: center;"><span class="regime-badge {regime_class}">{regime} Market Regime</span></div>', unsafe_allow_html=True)

            # Display key market metrics in a row of columns for better layout
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Display NIFTY close and daily percentage change
                # Calculate percentage change safely
                nifty_change_pct = ((last_nifty - prev_nifty) / prev_nifty * 100) if prev_nifty != 0 and isinstance(last_nifty, (int, float)) and isinstance(prev_nifty, (int, float)) else "N/A"
                st.metric("NIFTY 50", f"{last_nifty:,.2f}", f"{nifty_change_pct:+.2f}%" if isinstance(nifty_change_pct, (int, float)) else "N/A")
            with col2:
                # Display India VIX and daily percentage change (handle insufficient data or missing column)
                vix_change_pct = df['VIX_Change_Pct'].iloc[-1] if 'VIX_Change_Pct' in df.columns and len(df) >= 2 and not pd.isna(df['VIX_Change_Pct'].iloc[-1]) else "N/A"
                st.metric("India VIX", f"{last_vix:.2f}%" if isinstance(last_vix, (int, float)) else "N/A", f"{vix_change_pct:+.2f}%" if isinstance(vix_change_pct, (int, float)) else "N/A")
            with col3:
                # Display PCR (Put-Call Ratio) if available in the data
                pcr_value = df['PCR'].iloc[-1] if 'PCR' in df.columns and len(df) >= 1 and not pd.isna(df['PCR'].iloc[-1]) else "N/A"
                st.metric("PCR", f"{pcr_value:.2f}" if isinstance(pcr_value, (int, float)) else "N/A")
            with col4:
                # Display Straddle Price if available in the data
                straddle_price = df['Straddle_Price'].iloc[-1] if 'Straddle_Price' in df.columns and len(df) >= 1 and not pd.isna(df['Straddle_Price'].iloc[-1]) else "N/A"
                st.metric("Straddle Price", f"₹{straddle_price:,.2f}" if isinstance(straddle_price, (int, float)) else "N/A")

            # Display data source and last updated date below the metrics
            st.markdown(f"**Last Updated**: {latest_date} | **Source**: {st.session_state.get('data_source', 'Unknown')}")

            # Expander to show raw real-time API data if available in session state
            if st.session_state.real_time_market_data:
                with st.expander("Raw 5paisa API Data"):
                    st.json(st.session_state.real_time_market_data) # Display raw data as JSON
        else:
            # Message displayed if analysis data is not available
            st.info("Run analysis to see market snapshot")
            logger.info("Snapshot tab: Analysis data not available to display.")
        st.markdown('</div>', unsafe_allow_html=True) # Closing tag for the card div


    # --- Forecast Tab Content ---
    # Content for the Volatility Forecast tab.
    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True) # Container card with custom styling
        st.subheader("📈 Volatility Forecast") # Tab header
        # Display forecast results if available in session state
        if st.session_state.forecast_log is not None and not st.session_state.forecast_log.empty and st.session_state.forecast_metrics is not None:
            forecast_log = st.session_state.forecast_log # Get the forecast log DataFrame
            forecast_metrics = st.session_state.forecast_metrics # Get the forecast metrics dictionary

            # Display key forecast metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                # Display average blended volatility forecast, handle empty data after dropna
                avg_blended_vol = np.mean(forecast_log['Blended_Vol'].dropna()) if 'Blended_Vol' in forecast_log.columns and not forecast_log['Blended_Vol'].dropna().empty else "N/A"
                st.metric("Avg Blended Volatility", f"{avg_blended_vol:.2f}%" if isinstance(avg_blended_vol, (int, float)) else "N/A")
            with col2:
                # Display historical realized volatility
                st.metric("Realized Volatility", f"{forecast_metrics.get('realized_vol', 0.0):.2f}%")
            with col3:
                # Display the RMSE of the forecast model
                st.metric("Model RMSE", f"{forecast_metrics.get('rmse', 0.0):.2f}%")

             # Display Confidence Gauge based on forecast confidence score
            confidence_score = forecast_metrics.get("confidence_score", 50) # Get confidence score, default to 50
            st.markdown(f'<div class="gauge">{int(confidence_score)}%</div><div style="text-align: center; margin-top: 5px;">Confidence Score</div>', unsafe_allow_html=True)


            # Display volatility forecast chart (GARCH, XGBoost, Blended)
            # Check if required columns for the chart exist in the forecast log and Date is present
            if all(col in forecast_log.columns for col in ['GARCH_Vol', 'XGBoost_Vol', 'Blended_Vol']) and 'Date' in forecast_log.columns:
                 # Prepare data for the line chart, setting Date as the DataFrame index
                 chart_data = pd.DataFrame({
                     "GARCH": forecast_log["GARCH_Vol"],
                     "XGBoost": forecast_log["XGBoost_Vol"],
                     "Blended": forecast_log["Blended_Vol"]
                 }).set_index(forecast_log["Date"])
                 st.line_chart(chart_data, color=["#e94560", "#00d4ff", "#ffcc00"]) # Display line chart with custom colors
            else:
                 st.warning("Volatility forecast chart data is incomplete or missing 'Date' column.")
                 logger.warning("Volatility forecast chart data missing required columns.")


            st.markdown("### Feature Importance (XGBoost)")
            # Display feature importance from the XGBoost model if available and features are defined
            if forecast_metrics.get("feature_importances") is not None and FEATURE_COLS and st.session_state.analysis_df is not None:
                # Ensure the number of importances matches the number of features used for training
                # FEATURE_COLS should list the names of the features
                if len(forecast_metrics["feature_importances"]) == len(FEATURE_COLS):
                     # Create a DataFrame to display feature importance
                     feature_importance = pd.DataFrame({
                         'Feature': FEATURE_COLS,
                         'Importance': forecast_metrics["feature_importances"]
                     }).sort_values(by='Importance', ascending=False) # Sort by importance in descending order
                     st.dataframe(feature_importance, use_container_width=True) # Display the feature importance DataFrame
                else:
                     st.warning(f"Feature importance data length mismatch: {len(forecast_metrics['feature_importances'])} importances for {len(FEATURE_COLS)} features.")
                     logger.warning("Feature importance data length does not match FEATURE_COLS length.")
            elif not FEATURE_COLS:
                st.info("FEATURE_COLS list is not defined or empty in data_processing.py.")
                logger.warning("FEATURE_COLS list is empty.")
            else:
                st.info("Feature importance data not available from forecast metrics.")
                logger.info("Feature importance data is None.")


        else:
            st.info("Run analysis to see volatility forecast") # Message displayed if forecast data is not available
            logger.info("Forecast tab: Data not available to display.")
        st.markdown('</div>', unsafe_allow_html=True) # Closing tag for the card div


    # --- Strategy Tab Content ---
    # Content for the Trading Strategy tab.
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True) # Container card with custom styling
        st.subheader("🧪 Trading Strategy") # Tab header
        # Check if a strategy was generated during the analysis run
        if st.session_state.generated_strategy:
            # Check if the discipline lock is active (example behavior based on violations/journal)
            # This logic needs to be implemented in generate_trading_strategy based on violations/journal state
            if st.session_state.generated_strategy.get("Discipline_Lock", False):
                st.markdown('<div class="alert-banner">⚠️ Discipline Lock Active: Complete Journaling to unlock strategy generation.</div>', unsafe_allow_html=True)
                logger.warning("Discipline Lock active, strategy generation is blocked.")
            else: # Strategy successfully generated and not locked by discipline
                strategy = st.session_state.generated_strategy # Get the generated strategy details dictionary
                real_data = st.session_state.real_time_market_data # Get real-time market data for order preparation
                capital = st.session_state.capital # Get user's capital from session state

                # Determine the appropriate CSS class for the strategy's regime badge
                regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"}.get(strategy.get("Regime", "LOW"), "regime-low")

                # Display strategy details in a card format using custom CSS
                st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True) # Container for strategy cards (currently only 1 displayed)
                # Display details for the primary recommended strategy card
                st.markdown(f"""
                    <div class="strategy-card">
                        <h4>{strategy.get("Strategy", "N/A")}</h4>
                        <span class="regime-badge {regime_class}">{strategy.get("Regime", "N/A")} Regime</span>
                        <p><b>Reason:</b> {strategy.get("Reason", "N/A")}</p>
                        <p><b>Confidence:</b> {strategy.get("Confidence", 0.0):.2f}</p>
                        <p><b>Risk-Reward:</b> {strategy.get("Risk_Reward", 0.0):.2f}:1</p>
                        <p><b>Capital Deploy:</b> ₹{strategy.get("Deploy", 0.0):,.0f}</p>
                        <p><b>Max Loss:</b> ₹{strategy.get("Max_Loss", 0.0):,.0f}</p>
                        <p><b>Exposure:</b> {strategy.get("Exposure", 0.0):.2f}%</p>
                        <p><b>Tags:</b> {', '.join(strategy.get("Tags", ["N/A"]))}</p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True) # Close carousel div

                # Display risk flags identified for the strategy by the generator in an alert banner
                if strategy.get("Risk_Flags"):
                    st.markdown(f'<div class="alert-banner">⚠️ Risk Flags from Strategy Analysis: {", ".join(strategy["Risk_Flags"])}</div>', unsafe_allow_html=True)

                # Display behavioral warnings from the strategy generator as standard warnings
                if strategy.get("Behavior_Warnings"):
                    for warning in strategy["Behavior_Warnings"]:
                        st.warning(f"⚠️ Behavioral Warning from Strategy Analysis: {warning}")

                st.markdown("---") # Horizontal rule for visual separation
                st.subheader("Ready to Trade?") # Section header

                # Store the generated strategy details in session state as the currently active/recommended strategy
                st.session_state.active_strategy_details = strategy

                # --- Prepare Orders Button ---
                # Button to trigger the preparation of concrete order details for the recommended strategy.
                if st.button("📝 Prepare Orders", help="Generate specific order details for the recommended strategy based on current market data."):
                    logger.info("User clicked 'Prepare Orders'. Initiating order preparation.")
                    # Call prepare_trade_orders function from your fivepaisa_api module
                    # This function needs the strategy details, real-time market data (for strikes, expiry, etc.), and user's capital.
                    st.session_state.prepared_orders = prepare_trade_orders(strategy, real_data, capital)
                    st.session_state.order_placement_errors = [] # Clear any previous order placement errors
                    st.session_state.risk_warning = None # Clear any previously displayed risk warning when preparing NEW orders
                    logger.info(f"Order preparation completed. Prepared orders available: {st.session_state.prepared_orders is not None and len(st.session_state.prepared_orders) > 0}.")


                    # --- START: CALL SMARTBHAI'S PROACTIVE RISK ASSESSMENT (Step 2 & 4 Integration) ---
                    # This is the core integration point for SmartBhai's risk officer function.
                    # It happens AFTER orders are prepared, but BEFORE they are displayed for user confirmation.
                    if smartbhai_gpt: # Ensure the smartbhai_gpt object was successfully initialized
                         logger.info("Calling SmartBhai's assess_trade_risk for the prepared orders.")
                         # Call the assess_trade_risk method from your SmartBhaiGPT instance.
                         # Pass the entire current session state so SmartBhai has all necessary context (market data, strategy, portfolio, journal, etc.)
                         st.session_state.risk_warning = smartbhai_gpt.assess_trade_risk(st.session_state)
                         # The result from assess_trade_risk (stored in st.session_state.risk_warning)
                         # will be either None (no warning) or a dictionary
                         # like {"main_message": "...", "explanations": [...]}.
                         logger.info(f"SmartBhai risk assessment completed. Warning result type: {type(st.session_state.risk_warning)}.")
                    else:
                         # Display a warning if SmartBhaiGPT failed to load, indicating risk assessment couldn't run
                         st.warning("⚠️ SmartBhai GPT is not loaded, cannot perform risk assessment. Check sidebar for errors.")
                         logger.warning("SmartBhai GPT instance is None, risk assessment skipped on Prepare Orders.")
                    # --- END: CALL SMARTBHAI'S PROACTIVE RISK ASSESSMENT ---


                # --- Display Proposed Order Details and SmartBhai's Risk Warning ---
                # This block displays the generated order details and SmartBhai's risk assessment result.
                # It is only visible if prepare_trade_orders successfully generated orders and stored them in session state.
                if st.session_state.prepared_orders:
                    st.markdown("### Proposed Order Details") # Header for proposed orders

                    # --- START: DISPLAY SMARTBHAI'S RISK WARNING (with explanations) (Step 4 Display) ---
                    # This displays the warning message and explanations returned by assess_trade_risk.
                    risk_result = st.session_state.risk_warning # Get the risk assessment result from session state

                    # Check if the risk_result is a dictionary (indicating a warning was generated)
                    if risk_result and isinstance(risk_result, dict):
                        main_message = risk_result.get("main_message") # Get the main warning message string
                        explanations = risk_result.get("explanations", []) # Get the list of explanation strings (default to empty list)

                        # Display the main warning message in a Streamlit warning box if it exists
                        if main_message:
                            st.warning(f"🚨 **SmartBhai Risk Alert:**\n\n{main_message}")
                            logger.warning("Displayed SmartBhai main risk warning in Strategy tab.")

                            # Display explanations in a clickable expander below the main message
                            if explanations:
                                # Use markdown for the expander header to include the lightbulb emoji and bold text
                                with st.expander("💡 **Why is this risky?** SmartBhai Explains:"):
                                    # Display each explanation as a bullet point within the expander content
                                    for i, exp in enumerate(explanations):
                                        st.markdown(f"- {exp}") # Use markdown for bullet points
                                logger.warning(f"Displayed {len(explanations)} risk explanations.")
                        # Handle case where main_message is None but explanations exist (less likely with current assess_trade_risk logic but good to handle)
                        elif explanations:
                             st.warning("🚨 **SmartBhai Alert (Details):**") # A generic warning header
                             with st.expander("💡 **Details:**"): # Expander for details
                                    for i, exp in enumerate(explanations):
                                        st.markdown(f"- {exp}")
                             logger.warning("SmartBhai risk explanations displayed without main message.")

                    # Fallback for if risk_warning is not None but not the expected dictionary format (indicates an error in assess_trade_risk return)
                    elif risk_result is not None:
                         st.warning(f"⚠️ **SmartBhai Alert (Unexpected Format):**\n\n{risk_result}")
                         logger.warning(f"SmartBhai risk warning in unexpected format: {type(risk_result)}.")

                    # If risk_result is None, no warning is displayed, and the code continues below.
                    # --- END: DISPLAY SMARTBHAI'S RISK WARNING (with explanations) ---


                    st.warning("REVIEW CAREFULLY BEFORE PLACING!") # Your existing message prompting careful review of orders

                    # Display the prepared orders in a DataFrame for user review
                    orders_df = pd.DataFrame(st.session_state.prepared_orders)
                    # Define columns to display for clarity and ensure they exist in the DataFrame
                    orders_display_cols = ['Leg_Type', 'Strike', 'Expiry', 'Quantity_Lots', 'Quantity_Units', 'Proposed_Price', 'Last_Price_API', 'Stop_Loss_Price', 'Take_Profit_Price', 'ScripCode']
                    orders_display_cols_filtered = [col for col in orders_display_cols if col in orders_df.columns]
                    st.dataframe(orders_df[orders_display_cols_filtered], use_container_width=True) # Display the DataFrame

                    # Display any order placement errors that occurred during a previous attempt
                    if st.session_state.order_placement_errors:
                        st.error("Previous order placement failed:")
                        for error in st.session_state.order_placement_errors:
                            st.write(f"- {error}") # Display each error message

                    # --- START: Confirmation step logic if Risk Warning is present (Step 3 Integration) ---
                    # Control the behavior (enabled/disabled state) of the "Confirm and Place Orders" button
                    confirm_risk_proceed = True # Initialize confirmation status as True (button enabled)

                    # Check if SmartBhai issued a risk warning that needs acknowledgment
                    # This is true if risk_result is a dictionary AND has a main message
                    if risk_result and isinstance(risk_result, dict) and risk_result.get("main_message"):
                        # If there's a main warning message, display a checkbox asking the user to confirm they understand the risk.
                        # The state of this checkbox (True/False) determines if the user has acknowledged the risk.
                        # The default value is False (unchecked), requiring the user to interact.
                        confirm_risk_proceed = st.checkbox("I understand the risks flagged by SmartBhai and wish to proceed with placing the order.", value=False)
                        logger.debug(f"Risk warning present. Confirmation checkbox displayed. Current state: {confirm_risk_proceed}.")

                    # Determine if the "Confirm and Place Orders" button should be enabled.
                    # The button should be enabled ONLY IF confirm_risk_proceed is True.
                    # confirm_risk_proceed is True if either:
                    # 1. There is NO risk warning (risk_result is None or doesn't have a main_message), so confirm_risk_proceed remains True by default.
                    # OR
                    # 2. There IS a risk warning, AND the user has explicitly checked the confirmation box, making confirm_risk_proceed True.
                    button_disabled = not confirm_risk_proceed # The button is disabled if confirm_risk_proceed is False
                    logger.debug(f"'Confirm and Place Orders' button disabled state: {button_disabled}.")
                    # --- END: Confirmation step logic ---


                    # --- Confirm and Place Orders Button ---
                    # This button calls the function to send the prepared orders to the broker API.
                    # Its 'disabled' state is controlled by the confirmation logic implemented above.
                    if st.button("✅ Confirm and Place Orders", help="Send the prepared orders to your 5paisa trading account.", disabled=button_disabled): # Added disabled parameter
                        logger.info("User clicked 'Confirm and Place Orders'. Attempting to execute orders.")
                        # This code block runs ONLY if the button is clicked (and therefore enabled,
                        # meaning risks were either absent or acknowledged).

                        # Call execute_trade_orders function from your fivepaisa_api module to send orders to 5paisa
                        success, details = execute_trade_orders(st.session_state.client, st.session_state.prepared_orders)

                        if success:
                            st.success("✅ Orders placed successfully!") # Display success message
                            # Clear state related to order preparation and risk warning on successful placement
                            st.session_state.risk_warning = None # Clear the risk warning
                            st.session_state.prepared_orders = None # Clear prepared orders
                            st.session_state.order_placement_errors = [] # Clear errors list
                            # Optional: Add the trade details to your journal or a separate trade log here (requires implementation)
                            # e.g., log the strategy details, orders, PnL, date, etc.
                            # st.session_state.trades.append({...})
                            logger.info("Orders placed successfully. State cleared.")
                        else:
                            # If order placement fails via the API, store and display the errors
                            # Extract error messages from the API response details
                            st.session_state.order_placement_errors = [resp.get("Response", {}).get("Message", "Unknown error") for resp in details.get("responses", []) if resp.get("Response", {}).get("Status") != 0]
                            st.error("❌ Order placement failed. See errors above.") # Display a general error message
                            logger.error(f"Order placement failed. Errors: {st.session_state.order_placement_errors}")
                            # The risk warning is intentionally NOT cleared here if placement fails, so the user still sees it alongside the API error.

                # This 'else' block handles the case where prepare_trade_orders returned None or an empty list
                # It remains outside the modified button block.
                else:
                    # This message is shown if 'Prepare Orders' was clicked but resulted in no orders.
                    st.info("Click 'Prepare Orders' to see order details")
                    logger.info("Prepared orders list is empty, not displaying order details or place button.")


        else: # This 'else' corresponds to the initial check if st.session_state.generated_strategy is None or locked
            # This message is shown if the analysis hasn't been run or strategy generation failed/is locked.
            st.info("Run analysis to generate a strategy first.")
            logger.info("Strategy tab: No strategy generated or discipline locked.")

        st.markdown('</div>', unsafe_allow_html=True) # Closing tag for the card div


    # --- Portfolio Tab Content ---
    # Content for the Portfolio tab.
    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True) # Container card with custom styling
        st.subheader("💰 Portfolio Overview") # Tab header
        # Fetch and display portfolio summary metrics for the dashboard
        # It's fetched again here to ensure the latest data is shown when the tab is active
        # Pass the client and capital from session state to the helper function
        portfolio_summary = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
        col1, col2, col3 = st.columns(3) # Use columns for layout
        with col1:
            # Display total capital (get from summary, fallback to session state capital)
            st.metric("Total Capital", f"₹{portfolio_summary.get('total_capital', st.session_state.capital):,.2f}")
        with col2:
            # Display P&L (using weekly PnL as 'today's' for simplicity in dashboard)
            st.metric("Today's P&L", f"₹{portfolio_summary.get('weekly_pnl', 0.0):,.2f}")
        with col3:
            # Display utilized margin
            st.metric("Margin Used", f"₹{portfolio_summary.get('margin_used', 0.0):,.2f}")
        # Display current exposure percentage below the columns
        st.metric("Current Exposure", f"{portfolio_summary.get('exposure', 0.0):.2f}%")

        st.markdown("---") # Horizontal rule

        st.markdown("### Account Data") # Section header for detailed account data
        # Display detailed account data if fetched and stored in session state during analysis run
        if st.session_state.api_portfolio_data:
            st.subheader("Account Data")
            # Expander to display Holdings data
            with st.expander("📂 Holdings"):
                holdings_data = st.session_state.api_portfolio_data.get("holdings")
                # Check if holdings data exists and is in the expected list format with nested 'Holding' key
                if holdings_data and isinstance(holdings_data, list):
                     # Some APIs return holdings as a list containing a dictionary with a 'Holding' key that is a list
                     # Check if the first item is a dict and has a 'Holding' key which is a list
                     if holdings_data and isinstance(holdings_data[0], dict) and isinstance(holdings_data[0].get('Holding'), list):
                         # Flatten the nested structure into a single list of dictionaries
                         holdings_list = []
                         for item in holdings_data:
                              if isinstance(item, dict) and item.get('Holding'):
                                   holdings_list.extend(item.get('Holding', [])) # Use .extend() to add elements from the inner list
                         if holdings_list:
                             st.dataframe(pd.DataFrame(holdings_list), use_container_width=True) # Display as DataFrame
                         else:
                             st.info("No holdings found in the expected nested format.")
                             logger.warning("Portfolio tab: Holdings data found but not in expected nested format for DataFrame conversion.")
                     # Handle case where holdings might be a flat list of dicts
                     elif holdings_data and isinstance(holdings_data, list) and holdings_data and isinstance(holdings_data[0], dict):
                         st.dataframe(pd.DataFrame(holdings_data), use_container_width=True) # Display as DataFrame
                     else:
                         st.info("No holdings found or data format unexpected.")
                         logger.warning("Portfolio tab: Holdings data found but format unexpected for DataFrame conversion.")
                else:
                    st.info("No holdings found")
                    logger.info("Portfolio tab: No holdings data available.")


            # Expander to display Margin Details
            with st.expander("💲 Margin Details"):
                margin_data = st.session_state.api_portfolio_data.get("margin")
                if margin_data and isinstance(margin_data, dict):
                     # Convert the margin dictionary to a DataFrame (list of dicts) for display
                    st.dataframe(pd.DataFrame([{"Metric": k, "Value": v} for k, v in margin_data.items()]), use_container_width=True)
                else:
                    st.info("No margin data found")
                    logger.info("Portfolio tab: No margin data available.")

            # Expander to display Open Positions
            with st.expander("💹 Open Positions"):
                positions_data = st.session_state.api_portfolio_data.get("positions")
                if positions_data and isinstance(positions_data, list):
                    # Calculate current PnL for positions using latest LTP before displaying them
                    positions_with_pnl = calculate_position_pnl_with_ltp(st.session_state.client, positions_data)
                    if positions_with_pnl:
                         positions_df = pd.DataFrame(positions_with_pnl)
                         # Define formatting for currency columns using Indian Rupee symbol and commas
                         format_mapping = {col: '₹{:,.2f}' for col in ['BuyAvgPrice', 'SellAvgPrice', 'LTP', 'CurrentPnL', 'BookedPL', 'UnrealizedMTM']}
                         # Apply formatting only to columns that actually exist in the DataFrame
                         cols_to_format = {col: fmt for col, fmt in format_mapping.items() if col in positions_df.columns}
                         st.dataframe(positions_df.style.format(cols_to_format), use_container_width=True) # Display positions DataFrame with formatting
                    else:
                         st.info("Could not calculate PnL for positions or no positions with necessary data.")
                         logger.warning("Portfolio tab: Positions data available but PnL calculation failed or returned empty.")
                else:
                    st.info("No open positions found")
                    logger.info("Portfolio tab: No open positions data available.")

            # Expander to display Order Book
            with st.expander("📋 Order Book"):
                order_book_data = st.session_state.api_portfolio_data.get("order_book")
                if order_book_data and isinstance(order_book_data, list):
                    st.dataframe(pd.DataFrame(order_book_data), use_container_width=True) # Display order book as DataFrame
                else:
                    st.info("No open orders found")
                    logger.info("Portfolio tab: No order book data available.")

            # Expander to display Trade Book
            with st.expander("📜 Trade Book"):
                trade_book_data = st.session_state.api_portfolio_data.get("trade_book")
                if trade_book_data and isinstance(trade_book_data, list):
                    st.dataframe(pd.DataFrame(trade_book_data), use_container_width=True) # Display trade book as DataFrame
                else:
                    st.info("No executed trades found")
                    logger.info("Portfolio tab: No trade book data available.")

            # Expander to display Market Status from the API
            with st.expander("📰 Market Status"):
                market_status_data = st.session_state.api_portfolio_data.get("market_status")
                if market_status_data:
                    st.json(market_status_data) # Display market status as JSON
                else:
                    st.info("Market status not available")
                    logger.info("Portfolio tab: Market status data not available.")

        else:
            st.info("Run analysis to fetch portfolio data") # Message displayed if API portfolio data is not in session state
            logger.info("Portfolio tab: No API portfolio data in session state.")
        st.markdown('</div>', unsafe_allow_html=True) # Closing tag for the card div


    # --- Journal Tab Content ---
    # Content for the Discipline Hub / Journal tab.
    with tabs[4]:
        st.markdown('<div class="card">', unsafe_allow_html=True) # Container card with custom styling
        st.subheader("📝 Discipline Hub") # Tab header
        st.markdown("Reflect on your trading decisions and discipline.")

        # Form for adding a new journal entry. Use a unique key for the form.
        with st.form(key="journal_form"):
            # Select box for the reason behind the chosen strategy
            reason_strategy = st.selectbox("Why this strategy?", ["High IV", "Low Risk", "Event Opportunity", "Bullish Bias", "Bearish Bias", "Range Bound", "Expiry Play", "Other"], help="Select the primary reason for choosing this strategy.")
            # Radio button to indicate if risk flags were overridden (Boolean choice)
            override_risk = st.radio("Did you override any SmartBhai/Strategy risk flags?", ("No", "Yes"), index=0, help="Be honest here! Did you go against any warnings?")
            # Text area for the user's trade plan and expected outcome description
            expected_outcome = st.text_area("Your trade plan and expected outcome:", help="Describe your plan before entering the trade.")
            # Text area for lessons learned from the trade (optional reflection)
            lessons_learned = st.text_area("Lessons learned from this trade (optional):", help="Reflect on what went well or wrong after the trade is closed.")

            # Button to submit the journal entry form
            submit_journal = st.form_submit_button("💾 Save Journal Entry")

            if submit_journal:
                logger.info("User submitted journal entry.")
                # Calculate a simple discipline score based on journal inputs
                score = 0
                if override_risk == "No": score += 3 # Reward not overriding risk flags
                if reason_strategy != "Other": score += 2 # Reward having a defined reason
                if expected_outcome: score += 3 # Reward having a plan (some text entered)
                if lessons_learned: score += 1 # Reward reflection (some text entered)

                # Fetch latest portfolio PnL to potentially influence discipline score (example behavioral feedback)
                # This adds a simple feedback loop to the score based on recent performance
                portfolio_summary_for_journal = fetch_portfolio_data(st.session_state.client, st.session_state.capital) # Fetch latest PnL
                if portfolio_summary_for_journal.get('weekly_pnl', 0) > 0:
                    score += 1 # Reward positive PnL (simple positive reinforcement)

                score = min(score, 10) # Cap the score between 0 and 10

                # Create a dictionary representing the journal entry
                journal_entry = {
                    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Timestamp the entry with current date and time
                    "Strategy_Reason": reason_strategy,
                    "Override_Risk": override_risk,
                    "Expected_Outcome": expected_outcome,
                    "Lessons_Learned": lessons_learned,
                    "Discipline_Score": score # Store the calculated discipline score
                }

                # Save the entry to a CSV file, appending to the existing file
                try:
                    journal_log_path = "journal_log.csv"
                    # Check if the journal log file exists to write the header only for the first entry
                    file_exists = os.path.exists(journal_log_path)
                    # Use pandas to append the entry to the CSV file. Header is written only if the file is new.
                    pd.DataFrame([journal_entry]).to_csv(journal_log_path, mode='a', header=not file_exists, index=False, encoding='utf-8')
                    st.success(f"✅ Journal Entry Saved! Your Discipline Score: {score}/10") # Display success message with score
                    logger.info(f"Journal entry saved successfully to {journal_log_path}. Score: {score}.")

                    # After saving, reload the journal data into session state so SmartBhai can access it
                    st.session_state.journal_df = pd.read_csv(journal_log_path, encoding='utf-8')
                    st.session_state.journal_df['Date'] = pd.to_datetime(st.session_state.journal_df['Date']) # Ensure Date is datetime objects

                    # Check if the discipline lock can be removed based on score and violations
                    # This logic needs to align with how violations are counted and the lock is applied in generate_trading_strategy
                    if score >= 7 and st.session_state.violations > 0: # Example condition to remove lock
                        st.session_state.violations = 0 # Reset the violation counter
                        st.session_state.journal_complete = True # Mark journaling as complete for lock removal
                        st.success("🔓 Discipline Lock Removed! You can now generate strategies again.")
                        logger.info("Discipline Lock removed due to high score and completed journaling.")
                    elif st.session_state.violations > 0:
                         # Nudge the user if they have violations but their score is not high enough to unlock
                         st.warning(f"Keep journaling! You need a Discipline Score of 7 or more in recent entries to help remove the Discipline Lock. Your last score: {score}/10.")

                except Exception as e:
                    st.error(f"❌ Error saving journal entry: {e}") # Display error message if saving fails
                    logger.error(f"Error saving journal entry to CSV: {e}", exc_info=True)

        st.markdown("### Past Entries") # Section header for displaying past entries
        # Display past journal entries from the CSV file
        journal_log_path = "journal_log.csv"
        if os.path.exists(journal_log_path):
            try:
                journal_df = pd.read_csv(journal_log_path, encoding='utf-8')
                # Convert the 'Date' column to datetime objects and format for display
                journal_df['Date'] = pd.to_datetime(journal_df['Date']).dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(journal_df, use_container_width=True) # Display the journal entries as a DataFrame
            except Exception as e:
                st.error(f"❌ Error reading journal log: {e}") # Display error if reading fails
                logger.error(f"Error reading journal log CSV: {e}", exc_info=True)
        else:
            st.info("No journal entries found yet. Save your first entry above!") # Message if the file doesn't exist
            logger.info("Journal tab: journal_log.csv does not exist.")
        st.markdown('</div>', unsafe_allow_html=True) # Closing tag for the card div


    # --- Backtest Tab Content ---
    # Content for the Backtest Results tab.
    with tabs[5]:
        st.markdown('<div class="card">', unsafe_allow_html=True) # Container card with custom styling
        st.subheader("📉 Backtest Results") # Tab header
        # Display backtest results if a backtest has been successfully run (checked via session state flag)
        if st.session_state.backtest_run and st.session_state.backtest_results:
            results = st.session_state.backtest_results # Get the backtest results dictionary from session state
            # Check if cumulative P&L chart data is available and not empty
            if st.session_state.backtest_cumulative_pnl_chart_data is not None and not st.session_state.backtest_cumulative_pnl_chart_data.empty:
                # Display key backtest performance metrics (Total P&L, Win Rate, Sharpe, Drawdown) in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    # Display total P&L from the backtest
                    st.metric("Total P&L", f"₹{results.get('total_pnl', 0.0):,.2f}")
                with col2:
                    # Display win rate percentage
                    st.metric("Win Rate", f"{results.get('win_rate', 0.0)*100:.2f}%")
                with col3:
                    # Display Sharpe Ratio
                    st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0.0):.2f}")
                with col4:
                    # Display Max Drawdown (format as currency, ensuring it's treated as a potential loss)
                    max_drawdown_display = results.get('max_drawdown', 0.0)
                    st.metric("Max Drawdown", f"₹{max_drawdown_display:,.2f}")

                # Display the cumulative P&L chart over time using the stored chart data
                st.markdown("### Cumulative P&L") # Chart header
                st.line_chart(st.session_state.backtest_cumulative_pnl_chart_data, color="#e94560") # Use a custom color for the line

                # Display performance broken down by strategy
                st.markdown("### Performance by Strategy") # Section header
                if results.get("strategy_perf") is not None and not results["strategy_perf"].empty:
                    # Format currency and percentage columns for better readability
                    st.dataframe(results["strategy_perf"].style.format({
                        "sum": "₹{:,.2f}",
                        "mean": "₹{:,.2f}",
                        "Win_Rate": "{:.2%}"
                    }), use_container_width=True) # Display as DataFrame
                else:
                    st.info("No strategy performance data available from backtest.")
                    logger.warning("Backtest tab: Strategy performance data is None or empty.")

                # Display performance broken down by market regime
                st.markdown("### Performance by Regime") # Section header
                if results.get("regime_perf") is not None and not results["regime_perf"].empty:
                     # Format currency and percentage columns
                    st.dataframe(results["regime_perf"].style.format({
                        "sum": "₹{:,.2f}",
                        "mean": "₹{:,.2f}",
                        "Win_Rate": "{:.2%}"
                    }), use_container_width=True) # Display as DataFrame
                else:
                    st.info("No regime performance data available from backtest.")
                    logger.warning("Backtest tab: Regime performance data is None or empty.")

                # Display detailed backtest trades log
                st.markdown("### Detailed Backtest Trades") # Section header
                if results.get("backtest_df") is not None and not results["backtest_df"].empty:
                    # Define the columns to display in the detailed trades DataFrame
                    display_cols = ['Date', 'Event', 'Regime', 'Strategy', 'PnL', 'Cumulative_PnL', 'Strategy_Cum_PnL', 'Capital_Deployed', 'Max_Loss', 'Max_Profit', 'Risk_Reward', 'Notes']
                    # Filter the display columns to include only those that actually exist in the DataFrame
                    display_cols_filtered = [col for col in display_cols if col in results["backtest_df"].columns]
                    # Apply formatting for currency, date, and risk-reward ratio for better readability
                    st.dataframe(results["backtest_df"][display_cols_filtered].style.format({
                        # Format Date column (handle potential datetime objects)
                        "Date": lambda x: x.strftime("%Y-%m-%d") if isinstance(x, datetime) else x,
                        "PnL": "₹{:,.2f}",
                        "Cumulative_PnL": "₹{:,.2f}",
                        "Strategy_Cum_PnL": "₹{:,.2f}",
                        "Capital_Deployed": "₹{:,.2f}",
                        "Max_Loss": "₹{:,.2f}",
                         # Custom formatting for Max_Profit, handling the possibility of infinite profit (for long options)
                        "Max_Profit": lambda x: f'₹{x:,.2f}' if isinstance(x, (int, float)) and x != float('inf') and not pd.isna(x) else ('Unlimited' if x == float('inf') else 'N/A'),
                        "Risk_Reward": "{:.2f}"
                    }), use_container_width=True) # Display the detailed trades DataFrame
                else:
                    st.info("No detailed backtest trades data available.")
                    logger.info("Backtest tab: Detailed backtest DataFrame is None or empty.")

            else:
                st.info("Backtest cumulative P&L chart data is not available or is empty.")
                logger.warning("Backtest cumulative PnL chart data is None or empty.")
        else:
            st.info("Run analysis to view backtest results") # Message displayed if backtest hasn't been run
            logger.info("Backtest tab: Backtest has not been run or results not available.")
        st.markdown('</div>', unsafe_allow_html=True) # Closing tag for the card div


    # --- Risk Dashboard Tab Content ---
    # Content for the Live Risk Management Dashboard tab.
    with tabs[6]:
        st.markdown('<div class="card">', unsafe_allow_html=True) # Container card with custom styling
        st.subheader("🛡️ Live Risk Management Dashboard") # Tab header
        st.markdown("### Portfolio Risk Summary") # Section header for portfolio risk summary
        # Fetch and display portfolio summary metrics for the dashboard
        # It's fetched again here to ensure the latest data is shown when the tab is active
        portfolio_summary_risk = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
        col1, col2, col3 = st.columns(3) # Use columns for layout
        with col1:
            # Display total capital (get from summary, fallback to session state capital)
            st.metric("Total Capital", f"₹{portfolio_summary_risk.get('total_capital', st.session_state.capital):,.2f}")
        with col2:
            # Display P&L (using weekly PnL as 'today's' for simplicity in dashboard)
            st.metric("Today's P&L", f"₹{portfolio_summary_risk.get('weekly_pnl', 0.0):,.2f}")
        with col3:
            # Display utilized margin
            st.metric("Margin Used", f"₹{portfolio_summary_risk.get('margin_used', 0.0):,.2f}")
        # Display current exposure percentage below the columns
        st.metric("Current Exposure", f"{portfolio_summary_risk.get('exposure', 0.0):.2f}%")

        st.markdown("---") # Horizontal rule

        st.markdown("### Open Positions") # Section header for open positions
        # Display open positions with calculated PnL using LTP
        positions_data_risk = st.session_state.api_portfolio_data.get("positions", []) # Get positions data from session state (default to empty list)
        if positions_data_risk:
            # Calculate PnL with LTP before displaying
            positions_with_pnl_risk = calculate_position_pnl_with_ltp(st.session_state.client, positions_data_risk)
            if positions_with_pnl_risk:
                positions_df_risk = pd.DataFrame(positions_with_pnl_risk)
                # Define and apply formatting for currency columns
                format_mapping_risk = {col: '₹{:,.2f}' for col in ['BuyAvgPrice', 'SellAvgPrice', 'LTP', 'CurrentPnL', 'BookedPL', 'UnrealizedMTM']}
                cols_to_format_risk = {col: fmt for col, fmt in format_mapping_risk.items() if col in positions_df_risk.columns}
                st.dataframe(positions_df_risk.style.format(cols_to_format_risk), use_container_width=True) # Display positions DataFrame
            else:
                st.info("Could not calculate PnL for positions or no positions with necessary data.")
                logger.warning("Risk Dashboard: Positions data available but PnL calculation failed or returned empty.")
        else:
            st.info("No open positions found") # Message displayed if no positions are in session state
            logger.info("Risk Dashboard: No open positions data available.")

        st.markdown("---") # Horizontal rule

        st.markdown("### Active Strategy Risk") # Section header for active strategy risk
        # Display details and risk flags of the currently active/recommended strategy
        if st.session_state.active_strategy_details:
            strategy_risk = st.session_state.active_strategy_details # Get active strategy details from session state
            # Determine CSS class for regime badge
            regime_class_risk = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high", "EVENT-DRIVEN": "regime-event"}.get(strategy_risk.get("Regime", "LOW"), "regime-low")
            # Display strategy card (similar format to Strategy tab)
            st.markdown(f"""
                <div class="strategy-card">
                    <h4>{strategy_risk.get("Strategy", "N/A")}</h4>
                    <span class="regime-badge {regime_class_risk}">{strategy_risk.get("Regime", "N/A")} Regime</span>
                    <p><b>Reason:</b> {strategy_risk.get("Reason", "N/A")}</p>
                    <p><b>Risk-Reward:</b> {strategy_risk.get("Risk_Reward", 0.0):.2f}:1</p>
                    <p><b>Capital Deploy:</b> ₹{strategy_risk.get("Deploy", 0.0):,.0f}</p>
                    <p><b>Max Loss:</b> ₹{strategy_risk.get("Max_Loss", 0.0):,.0f}</p>
                    <p><b>Tags:</b> {', '.join(strategy_risk.get("Tags", ["N/A"]))}</p>
                </div>
            """, unsafe_allow_html=True)
            # Display risk flags identified for this strategy
            if strategy_risk.get("Risk_Flags"):
                st.warning(f'⚠️ Risk Flags: {", ".join(strategy_risk["Risk_Flags"])}')
            # Display behavioral warnings
            if strategy_risk.get("Behavior_Warnings"):
                for warning in strategy_risk["Behavior_Warnings"]:
                    st.warning(f"⚠️ Behavioral Warning: {warning}")
        else:
            st.info("No active strategy details available. Run analysis to generate one.") # Message if no active strategy is in session state
            logger.info("Risk Dashboard: No active strategy details in session state.")

        st.markdown("---") # Horizontal rule

        st.markdown("### Value at Risk (VaR)") # Section header for VaR
        st.info("Simplified VaR for illustration (Historical, 1-Day, 99% Confidence)") # Info message about the VaR calculation
        # Calculate and display simplified Historical VaR if analysis data is available and not empty
        if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
            df_var = st.session_state.analysis_df.copy()
            # Calculate daily percentage returns for NIFTY Close price
            df_var['NIFTY_Daily_Return'] = df_var['NIFTY_Close'].pct_change()
            # Drop the first row which will have a NaN return after pct_change
            df_var = df_var.dropna(subset=['NIFTY_Daily_Return'])

            # Proceed only if there are historical returns data points to analyze
            if not df_var.empty:
                confidence_level = 0.99 # Set confidence level for VaR (e.g., 99%)
                # Calculate the percentile corresponding to the worst loss (e.g., the 1st percentile for 99% confidence)
                worst_loss_pct = np.percentile(df_var['NIFTY_Daily_Return'], (1 - confidence_level) * 100)
                # Estimate the current value of the portfolio (User's Capital + total P&L)
                current_value = portfolio_summary_risk.get('total_capital', st.session_state.capital) + portfolio_summary_risk.get('weekly_pnl', 0.0)
                # Calculate Value at Risk in absolute currency terms
                # VaR is typically reported as a positive value representing potential loss in worst-case scenario
                var_absolute = current_value * abs(worst_loss_pct) if isinstance(worst_loss_pct, (int, float)) and worst_loss_pct < 0 else 0 # Ensure worst_loss_pct is a valid number and negative

                # Display the calculated VaR formatted as currency
                st.write(f"**Historical 1-Day VaR ({confidence_level*100:.0f}%):** ₹{var_absolute:,.2f}")
                # Add a caption explaining the limitations of this simplified VaR calculation
                st.caption("Assumes portfolio moves proportionally with NIFTY. Does not account for option greeks, correlation, or tail risk beyond historical data.")
            else:
                st.info("Insufficient data (need historical returns) for VaR calculation.")
                logger.warning("Risk Dashboard: Insufficient data for VaR calculation.")
        else:
            st.info("Run analysis to calculate VaR") # Message displayed if analysis data is not available for VaR
            logger.info("Risk Dashboard: Analysis data not available for VaR calculation.")
        st.markdown('</div>', unsafe_allow_html=True) # Closing tag for the card div


    # --- Footer ---
    # Display the footer information at the bottom of the main content area.
    st.markdown('<div class="footer">Built with ❤️ by Shritish Shukla & Salman Azim | © 2025 VolGuard</div>', unsafe_allow_html=True)
