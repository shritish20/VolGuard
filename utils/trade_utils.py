import streamlit as st
import pandas as pd
import logging
from py5paisa import FivePaisaClient
from utils.date_utils import parse_5paisa_date_string
from config.settings import LOT_SIZE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_5paisa_client(totp_code):
    """Initializes the 5paisa client with TOTP authentication."""
    try:
        logger.info("Initializing 5paisa client")
        cred = {
            "APP_NAME": st.secrets["fivepaisa"]["APP_NAME"],
            "APP_SOURCE": st.secrets["fivepaisa"]["APP_SOURCE"],
            "USER_ID": st.secrets["fivepaisa"]["USER_ID"],
            "PASSWORD": st.secrets["fivepaisa"]["PASSWORD"],
            "USER_KEY": st.secrets["fivepaisa"]["USER_KEY"],
            "ENCRYPTION_KEY": st.secrets["fivepaisa"]["ENCRYPTION_KEY"]
        }
        client = FivePaisaClient(cred=cred)
        logger.info("Attempting TOTP session...")
        client.get_totp_session(
            st.secrets["fivepaisa"]["CLIENT_CODE"],
            totp_code,
            st.secrets["fivepaisa"]["PIN"]
        )
        if client.get_access_token():
            logger.info("5paisa client initialized successfully")
            st.session_state.logged_in = True
            return client
        else:
            logger.error("Failed to get access token.")
            st.error("Login failed: Could not get access token.")
            return None
    except Exception as e:
        logger.error(f"Error initializing client: {str(e)}")
        st.error(f"Login failed: {str(e)}. Check credentials and TOTP.")
        return None

def prepare_trade_orders(strategy, real_data, capital):
    """Prepares trade orders for a given strategy."""
    logger.info(f"Preparing trade orders for: {strategy['Strategy']}")
    if not real_data or "option_chain" not in real_data or "atm_strike" not in real_data or "expiry" not in real_data or "straddle_price" not in real_data:
        logger.error("Incomplete real-time data.")
        st.error("Cannot prepare orders: Real-time data missing.")
        return None

    option_chain = real_data["option_chain"]
    atm_strike = real_data["atm_strike"]
    expiry_date_str = real_data["expiry"]
    straddle_price_live = real_data["straddle_price"]

    if option_chain.empty or atm_strike is None or expiry_date_str == "N/A" or straddle_price_live is None or straddle_price_live <= 0:
        logger.error("Invalid market data.")
        st.error("Cannot prepare orders: Essential market data incomplete.")
        return None

    deploy = strategy["Deploy"]
    max_loss = strategy["Max_Loss"]
    premium_per_lot = straddle_price_live * LOT_SIZE if straddle_price_live > 0 else 200 * LOT_SIZE
    lots = max(1, int(deploy / premium_per_lot) if premium_per_lot > 0 else 1)
    lots = min(lots, 10)
    orders_to_place = []
    strategy_legs = []
    expiry_timestamp = parse_5paisa_date_string(option_chain["ExpiryDate"].iloc[0])

    if strategy["Strategy"] == "Short Straddle":
        strategy_legs = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")]
    elif strategy["Strategy"] == "Short Strangle":
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        call_strike = next((s for s in strikes_sorted if s >= atm_strike + 100), None)
        put_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)
        if call_strike is None or put_strike is None:
            logger.error("Could not find strikes for Short Strangle.")
            st.error("Cannot prepare Short Strangle: Strikes not found.")
            return None
        strategy_legs = [(call_strike, "CE", "S"), (put_strike, "PE", "S")]
    elif strategy["Strategy"] == "Iron Condor":
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None)
        put_buy_strike = next((s for s in reversed(strikes_sorted) if s < put_sell_strike - 100 if put_sell_strike is not None), None)
        call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 50), None)
        call_buy_strike = next((s for s in strikes_sorted if s > call_sell_strike + 100 if call_sell_strike is not None), None)
        if None in [put_sell_strike, put_buy_strike, call_sell_strike, call_buy_strike]:
            logger.error("Could not find strikes for Iron Condor.")
            st.error("Cannot prepare Iron Condor: Strikes not found.")
            return None
        strategy_legs = [(put_buy_strike, "PE", "B"), (put_sell_strike, "PE", "S"), (call_sell_strike, "CE", "S"), (call_buy_strike, "CE", "B")]
    elif strategy
