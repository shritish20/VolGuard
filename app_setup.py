import streamlit as st
from py5paisa import FivePaisaClient
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# Enhanced logging (from first code)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("volguard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="VolGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (from original code)
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #1a1a2e, #0f1c2e);
            color: #e5e5e5;
            font-family: 'Inter', 'Roboto', sans-serif;
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
            transition: background 0.3s;
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
            transition: transform 0.3s;
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
            transition: transform 0.3s;
            animation: fadeIn 0.6s;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .strategy-carousel {
            display: flex;
            overflow-x: auto;
            gap: 20px;
            padding: 10px;
            scrollbar-width: thin;
        }
        .strategy-card {
            flex: 0 0 auto;
            width: 300px;
            background: #16213e;
            border-radius: 15px;
            padding: 20px;
            transition: transform 0.3s;
        }
        .strategy-card:hover {
            transform: scale(1.05);
        }
        .stMetric {
            background: rgba(15, 52, 96, 0.7);
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        .gauge {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient(#e94560 0% 50%, #00d4ff 50% 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0 0 15px rgba(233, 69, 96, 0.5);
            animation: rotateIn 1s;
        }
        .alert-banner {
            background: #dc3545;
            color: white;
            padding: 15px;
            border-radius: 10px;
            position: sticky;
            top: 0;
            z-index: 100;
            animation: pulse 1.5s infinite;
        }
        .stButton>button {
            background: #e94560;
            color: white;
            border-radius: 10px;
            padding: 12px 25px;
            font-size: 16px;
            transition: transform 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: #ffcc00;
        }
        @keyframes fadeIn {
            from { opacity: 0; } to { opacity: 1; }
        }
        @keyframes rotateIn {
            from { transform: rotate(-180deg); opacity: 0; } to { transform: rotate(0); opacity: 1; }
        }
        @keyframes pulse {
            0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); }
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state (enhanced from first code)
def initialize_session_state():
    """Robust session state initialization"""
    defaults = {
        "backtest_run": False,
        "backtest_results": None,
        "violations": 0,
        "journal_complete": False,
        "trades": [],
        "logged_in": False,
        "trading_halted": False,
        "risk_alerts": [],
        "last_api_success": None,
        "api_retry_count": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# 5paisa Client Setup with retry logic (from first code, renamed to setup_client)
def setup_client():
    """Client setup with auto-retry"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            cred = {
                "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
                "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
                "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
                "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
                "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
                "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
            }
            client = FivePaisaClient(cred=cred)
            
            # Test connection
            test_req = [{"Exch": "N", "ExchType": "C", "ScripCode": 999920000}]
            client.fetch_market_feed(test_req)
            
            st.session_state.last_api_success = datetime.now()
            st.session_state.api_retry_count = 0
            return client
        except Exception as e:
            logger.warning(f"API Connection Attempt {attempt+1} failed: {str(e)}")
            st.session_state.api_retry_count += 1
            time.sleep(2)
    
    st.error("🔴 Critical: Failed to connect to 5paisa after 3 attempts")
    logger.error("Failed to connect to 5paisa after 3 attempts")
    return None

# Sidebar Login and Controls (merges original and first code)
def render_sidebar(client):
    """Enhanced sidebar with session checks and TOTP login"""
    with st.sidebar:
        st.header("🔐 5paisa Login")
        
        # Auto-refresh token if >30 mins old (from first code)
        if st.session_state.logged_in:
            last_success = st.session_state.last_api_success
            if last_success and (datetime.now() - last_success).seconds > 1800:
                st.warning("Session expired - refreshing...")
                client = setup_client()
                if client:
                    st.session_state.client = client
                    st.rerun()
        
        if not st.session_state.logged_in:
            totp_code = st.text_input("TOTP (from Authenticator App)", type="password")
            if st.button("Login"):
                try:
                    client = setup_client()
                    if client:
                        # TOTP login (from original code)
                        response = client.get_totp_session(
                            st.secrets["fivepaisa"]["CLIENT_CODE"],
                            totp_code,
                            st.secrets["fivepaisa"]["PIN"]
                        )
                        if client.get_access_token():
                            st.session_state.client = client
                            st.session_state.logged_in = True
                            st.session_state.last_api_success = datetime.now()
                            st.success("✅ Logged in successfully")
                            logger.info("User logged in successfully")
                            st.rerun()
                        else:
                            st.error("❌ Login failed: Invalid TOTP")
                            logger.error("Login failed: Invalid TOTP")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Login error: {str(e)}")
        
        if st.session_state.logged_in:
            st.header("⚙️ Trading Controls")
            capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
            risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
            forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
            st.markdown("**Backtest Parameters**")
            # Date validation (from first code)
            today = datetime.now().date()
            start_date = st.date_input("Start Date", value=today - timedelta(days=30))
            end_date = st.date_input("End Date", value=today)
            
            if start_date > end_date:
                st.error("Start date must be before end date")
                start_date, end_date = end_date, start_date
            
            strategy_choice = st.selectbox("Strategy", [
                "All Strategies", "Butterfly Spread", "Iron Condor",
                "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard"
            ])
            st.markdown("---")
            st.markdown("**Motto:** Deploy with edge, survive, outlast.")  # From original code
            return capital, risk_tolerance, forecast_horizon, start_date, end_date, strategy_choice
        return None, None, None, None, None, None

# Main UI Tabs (merges original and first code)
def render_main_ui():
    """UI with status indicators and alert banners"""
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)
    
    # Status bar (from first code)
    if st.session_state.get("last_api_success"):
        mins_ago = (datetime.now() - st.session_state.last_api_success).seconds // 60
        st.caption(f"🟢 API Connected ({mins_ago} mins ago) | Streamlit Cloud")
    else:
        st.caption("🔴 API Disconnected | Streamlit Cloud")
    
    # Alert banners (from original code)
    if st.session_state.trading_halted:
        st.markdown('<div class="alert-banner">🚨 Trading Halted: Risk Limits Breached!</div>', unsafe_allow_html=True)
    for alert in st.session_state.risk_alerts:
        st.markdown(f'<div class="alert-banner">⚠️ {alert}</div>', unsafe_allow_html=True)
    
    return st.tabs(["Snapshot", "Forecast", "Strategy", "Portfolio", "Journal", "Backtest"])
