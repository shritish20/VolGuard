import streamlit as st
from py5paisa import FivePaisaClient
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="VolGuard Pro", page_icon="🛡️", layout="wide")

# Custom CSS
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

# Initialize session state
def initialize_session_state():
    defaults = {
        "backtest_run": False,
        "backtest_results": None,
        "violations": 0,
        "journal_complete": False,
        "trades": [],
        "logged_in": False,
        "trading_halted": False,
        "risk_alerts": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# 5paisa Client Setup
def setup_client():
    cred = {
        "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
        "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
        "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
        "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
        "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
        "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
    }
    return FivePaisaClient(cred=cred)

# Sidebar Login and Controls
def render_sidebar(client):
    with st.sidebar:
        st.header("🔐 5paisa Login")
        totp_code = st.text_input("TOTP (from Authenticator App)", type="password")
        if st.button("Login"):
            try:
                response = client.get_totp_session(
                    st.secrets["fivepaisa"]["CLIENT_CODE"],
                    totp_code,
                    st.secrets["fivepaisa"]["PIN"]
                )
                if client.get_access_token():
                    st.session_state.client = client
                    st.session_state.logged_in = True
                    st.success("✅ Logged in successfully")
                else:
                    st.error("❌ Login failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")

        if st.session_state.logged_in:
            st.header("⚙️ Trading Controls")
            capital = st.number_input("Capital (₹)", min_value=100000, value=1000000, step=100000)
            risk_tolerance = st.selectbox("Risk Profile", ["Conservative", "Moderate", "Aggressive"], index=1)
            forecast_horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)
            st.markdown("**Backtest Parameters**")
            start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
            end_date = st.date_input("End Date", value=pd.to_datetime("2025-04-29"))
            strategy_choice = st.selectbox("Strategy", ["All Strategies", "Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard"])
            st.markdown("---")
            st.markdown("**Motto:** Deploy with edge, survive, outlast.")
            return capital, risk_tolerance, forecast_horizon, start_date, end_date, strategy_choice
    return None, None, None, None, None, None

# Main UI Tabs
def render_main_ui():
    st.markdown("<h1 style='color: #e94560; text-align: center;'>🛡️ VolGuard Pro: Your AI Trading Copilot</h1>", unsafe_allow_html=True)
    if st.session_state.trading_halted:
        st.markdown('<div class="alert-banner">🚨 Trading Halted: Risk Limits Breached!</div>', unsafe_allow_html=True)
    for alert in st.session_state.risk_alerts:
        st.markdown(f'<div class="alert-banner">⚠️ {alert}</div>', unsafe_allow_html=True)
    return st.tabs(["Snapshot", "Forecast", "Strategy", "Portfolio", "Journal", "Backtest"])
