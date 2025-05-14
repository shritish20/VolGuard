import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
from pathlib import Path
import html
from dotenv import load_dotenv
from upstox_api import (
    initialize_upstox_client,
    fetch_all_api_portfolio_data,
    prepare_trade_orders,
    execute_trade_orders,
    square_off_positions,
    fetch_real_time_market_data,
)

# Load environment variables
load_dotenv()

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
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

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
        }
        .stTabs [data-baseweb="tab"] {
            color: #a0a0a0;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #e94560;
            color: white;
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
            background: #e94560;
        }
        .card {
            background: linear-gradient(145deg, rgba(22, 33, 62, 0.85), rgba(10, 25, 47, 0.9));
            border-radius: 15px;
            padding: 20px;
            margin: 15px 0;
        }
        .stMetric {
            background: rgba(15, 52, 96, 0.7);
            border-radius: 15px;
            padding: 15px;
            text-align: center;
        }
        .stButton>button {
            background: #e94560;
            color: white;
            border-radius: 10px;
            padding: 12px 25px;
        }
        .stButton>button:hover {
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
    </style>
    """,
    unsafe_allow_html=True,
)

# === Initialize Session State ===
default_session_state = {
    "logged_in": False,
    "client": None,
    "real_time_market_data": {},
    "api_portfolio_data": {},
    "prepared_orders": None,
    "trades": [],
    "order_placement_errors": [],
    "capital": 1000000,
    "risk_tolerance": "Moderate",
    "journal_date_input": datetime.now().date(),
    "journal_strategy_input": "",
    "journal_pnl_input": 0.0,
    "journal_notes_input": "",
}

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

# === Helper Functions ===
def is_market_hours():
    """Check if current time is within market hours (9:15 AM to 3:30 PM IST, Mon-Fri)."""
    now = datetime.now()
    market_open = time(9, 15)
    market_close = time(15, 30)
    is_weekday = now.weekday() < 5
    return is_weekday and (market_open <= now.time() <= market_close)

@st.cache_data(show_spinner=False)
def fetch_portfolio_data(upstox_client, capital):
    """Fetch and summarize portfolio data."""
    portfolio_summary = {
        "weekly_pnl": 0.0,
        "margin_used": 0.0,
        "exposure": 0.0,
        "total_capital": capital,
    }
    if not upstox_client:
        logger.warning("Upstox client not available.")
        return portfolio_summary

    try:
        portfolio_data = fetch_all_api_portfolio_data(upstox_client)
        st.session_state.api_portfolio_data = portfolio_data
        margin_data = portfolio_data.get("margin", {}).get("data", {})
        positions_data = portfolio_data.get("positions", {}).get("data", [])

        portfolio_summary["margin_used"] = float(margin_data.get("used_margin", 0)) if margin_data else 0.0
        portfolio_summary["weekly_pnl"] = sum(
            float(pos.get("unrealised", 0)) + float(pos.get("realised", 0))
            for pos in positions_data
        ) if positions_data else 0.0
        portfolio_summary["exposure"] = (
            portfolio_summary["margin_used"] / capital * 100 if capital > 0 else 0.0
        )
        return portfolio_summary
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {e}")
        return portfolio_summary

def generate_trading_strategy(analysis_df, real_time_market_data, forecast_metrics, capital, risk_tolerance):
    """Generate a mock trading strategy based on market data and risk tolerance."""
    date = datetime.now().date()
    signals = pd.DataFrame([
        {
            "Date": date,
            "Strategy": "Short Straddle" if risk_tolerance == "Aggressive" else "Iron Condor",
            "Confidence": 0.75 if risk_tolerance == "Moderate" else 0.9 if risk_tolerance == "Aggressive" else 0.6,
            "Deploy": capital * (0.3 if risk_tolerance == "Aggressive" else 0.2 if risk_tolerance == "Moderate" else 0.1),
            "Orders": [
                {"instrument_key": "NSE_INDEX|Nifty 50", "quantity": 50, "transaction_type": "SELL", "order_type": "MARKET"},
                {"instrument_key": "NSE_INDEX|Nifty 50", "quantity": 50, "transaction_type": "SELL", "order_type": "MARKET"}
            ],
            "Reasoning": f"Generated strategy for {risk_tolerance} risk profile based on current market conditions."
        }
    ])
    if not signals.empty and date in signals["Date"].values:
        signal = signals[signals["Date"] == date].iloc[0]
    else:
        signal = {
            "Strategy": "None",
            "Confidence": 0.0,
            "Deploy": 0.0,
            "Orders": [],
            "Reasoning": "No signal available for the date"
        }
    return {
        "Strategy": signal["Strategy"],
        "Confidence": signal["Confidence"],
        "Deploy": signal["Deploy"],
        "Orders": signal["Orders"],
        "Reasoning": signal["Reasoning"],
    }

# === Streamlit App Layout ===
with st.sidebar:
    st.header("🔑 Upstox Login")
    access_token = st.text_input("Access Token", type="password", key="access_token_input")
    if st.button("Login to Upstox"):
        if not access_token:
            st.error("❌ Access token cannot be empty.")
            logger.error("Login attempted with empty access token.")
        else:
            client_objects = initialize_upstox_client(access_token)
            if client_objects:
                st.session_state.client = client_objects
                st.session_state.logged_in = True
                st.success("✅ Logged in to Upstox!")
                logger.info("Upstox login successful.")
                st.rerun()
            else:
                st.error("❌ Login failed. Invalid or expired access token.")
                logger.error("Upstox login failed: Invalid or expired token.")

    if st.session_state.logged_in:
        st.header("⚙️ Trading Controls")
        st.session_state.capital = st.number_input(
            "Capital (₹)", min_value=100000, value=st.session_state.capital, step=100000
        )
        st.session_state.risk_tolerance = st.selectbox(
            "Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1
        )
        if st.button("Square Off All Positions"):
            if is_market_hours():
                success = square_off_positions(st.session_state.client)
                if success:
                    st.success("✅ Square off initiated!")
                    logger.info("Square off initiated.")
                else:
                    st.error("❌ Failed to square off.")
                    logger.error("Square off failed.")
            else:
                st.warning("⏰ Market is closed.")
                logger.warning("Square off attempted outside market hours.")

# === Main App ===
st.title("🛡️ VolGuard Pro")
st.markdown("**Your AI-powered options trading cockpit for NIFTY 50**")

if not is_market_hours():
    st.warning("⚠️ Outside market hours (9:15 AM–3:30 PM IST, Mon-Fri).")

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Market Snapshot", "🤖 Trading Strategy", "💼 Portfolio", "📝 Journal"]
)

with tab1:
    st.header("📊 Market Snapshot")
    if st.session_state.logged_in:
        data = fetch_real_time_market_data(st.session_state.client)
        st.session_state.real_time_market_data = data
        if data:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("NIFTY Spot", f"{data.get('nifty_spot', 'N/A'):.2f}")
            col2.metric("India VIX", f"{data.get('vix', 'N/A'):.2f}")
            col3.metric("PCR", f"{data.get('pcr', 'N/A'):.2f}")
            col4.metric("ATM Straddle", f"{data.get('straddle_price', 'N/A'):.2f}")

            st.subheader("Option Chain Preview")
            df = data.get("option_chain")
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.dataframe(df[["Strike", "CE_LTP", "CE_OI", "PE_LTP", "PE_OI"]].head(10))
            else:
                st.info("No option chain data available.")
        else:
            st.error("❌ Failed to fetch market data.")
    else:
        st.info("Please log in to Upstox.")

with tab2:
    st.header("🤖 Trading Strategy")
    if st.session_state.logged_in and st.session_state.real_time_market_data:
        strategy = generate_trading_strategy(
            pd.DataFrame(),
            st.session_state.real_time_market_data,
            {},
            st.session_state.capital,
            st.session_state.risk_tolerance
        )
        if strategy and strategy["Strategy"] != "None":
            st.markdown(f"**Strategy**: {strategy['Strategy']}")
            st.markdown(f"**Confidence**: {strategy['Confidence']:.2%}")
            st.markdown(f"**Deploy Amount**: ₹{strategy['Deploy']:.2f}")
            st.markdown(f"**Reasoning**: {strategy['Reasoning']}")

            if st.button("Prepare Orders"):
                if is_market_hours():
                    orders = prepare_trade_orders(strategy)
                    st.session_state.prepared_orders = orders
                    if orders:
                        st.success(f"✅ {len(orders)} orders prepared!")
                        st.dataframe(pd.DataFrame(orders))
                    else:
                        st.warning("⚠️ No orders prepared.")
                else:
                    st.warning("⏰ Market is closed.")

            if st.session_state.prepared_orders and is_market_hours():
                with st.form("execute_orders_form"):
                    st.warning("🚨 This will place REAL orders!")
                    confirm = st.checkbox("Confirm live order placement")
                    if st.form_submit_button("Execute Orders") and confirm:
                        success, response = execute_trade_orders(st.session_state.client, st.session_state.prepared_orders)
                        if success:
                            st.success("✅ Orders executed!")
                            st.session_state.trades.extend(st.session_state.prepared_orders)
                            st.session_state.prepared_orders = None
                        else:
                            st.error("❌ Order execution failed.")
                            st.json(response)
        else:
            st.info("No strategy available.")
    else:
        st.info("Please log in and fetch market data.")

with tab3:
    st.header("💼 Portfolio")
    if st.session_state.logged_in:
        portfolio = fetch_portfolio_data(st.session_state.client, st.session_state.capital)
        col1, col2, col3 = st.columns(3)
        col1.metric("Weekly PnL", f"₹{portfolio['weekly_pnl']:.2f}")
        col2.metric("Margin Used", f"₹{portfolio['margin_used']:.2f}")
        col3.metric("Exposure", f"{portfolio['exposure']:.2f}%")

        st.subheader("Open Positions")
        positions = st.session_state.api_portfolio_data.get("positions", {}).get("data", [])
        if positions:
            st.dataframe(pd.DataFrame(positions)[["instrument_token", "quantity", "average_price"]])
        else:
            st.info("No open positions.")

        st.subheader("Fund Summary")
        margin = st.session_state.api_portfolio_data.get("margin", {}).get("data", {})
        if margin:
            st.dataframe(pd.DataFrame([margin]))
        else:
            st.info("No fund data available.")
    else:
        st.info("Please log in to Upstox.")

with tab4:
    st.header("📝 Trading Journal")
    journal_file = Path("journal_log.csv")
    journal_df = pd.DataFrame(columns=["Date", "Strategy", "PnL", "Notes"])
    if journal_file.exists():
        journal_df = pd.read_csv(journal_file)
        journal_df["Date"] = pd.to_datetime(journal_df["Date"], errors="coerce")

    with st.form("journal_form"):
        date_log = st.date_input("Trade Date", st.session_state.journal_date_input)
        strategy_log = st.text_input("Strategy", st.session_state.journal_strategy_input)
        pnl_log = st.number_input("PnL (₹)", value=st.session_state.journal_pnl_input)
        notes_log = st.text_area("Notes", st.session_state.journal_notes_input)
        if st.form_submit_button("Log Trade"):
            new_entry = pd.DataFrame({
                "Date": [date_log],
                "Strategy": [strategy_log],
                "PnL": [pnl_log],
                "Notes": [notes_log]
            })
            journal_df = pd.concat([journal_df, new_entry], ignore_index=True)
            journal_df.to_csv(journal_file, index=False)
            st.success("✅ Trade logged!")
            st.session_state.journal_strategy_input = ""
            st.session_state.journal_pnl_input = 0.0
            st.session_state.journal_notes_input = ""

    st.subheader("Trade History")
    if not journal_df.empty:
        st.dataframe(journal_df.sort_values(by="Date", ascending=False))
    else:
        st.info("No trades logged.")

# === Footer ===
st.markdown(
    """
    <div class='footer'>
        VolGuard Pro | Built with ❤️ by Shritish | Powered by Upstox API & Streamlit
        <br>
        <small>Disclaimer: Trading involves risks. Consult a financial advisor.</small>
    </div>
    """,
    unsafe_allow_html=True,
)
