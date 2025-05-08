import logging
import io
import re
from datetime import datetime
from py5paisa import FivePaisaClient
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Helper function to parse 5paisa date string
def parse_5paisa_date_string(date_string):
    if not isinstance(date_string, str):
        return None
    match = re.search(r'/Date\((\d+)[+-]\d+\)/', date_string)
    if match:
        return int(match.group(1))
    return None

# Helper function to format timestamp
def format_timestamp_to_date_str(timestamp_ms):
    if timestamp_ms is None:
        return "N/A"
    try:
        timestamp_s = timestamp_ms / 1000
        dt_object = datetime.fromtimestamp(timestamp_s)
        return dt_object.strftime("%Y-%m-%d")
    except Exception:
        return "N/A"

# Max Pain calculation
def calculate_max_pain(df: pd.DataFrame, nifty_spot: float):
    try:
        if df.empty or "StrikeRate" not in df.columns or "CPType" not in df.columns or "OpenInterest" not in df.columns:
            logger.warning("Option chain data incomplete for max pain")
            return None, None
        df["StrikeRate"] = pd.to_numeric(df["StrikeRate"], errors='coerce')
        df["OpenInterest"] = pd.to_numeric(df["OpenInterest"], errors='coerce').fillna(0)
        df = df.dropna(subset=["StrikeRate", "OpenInterest"]).copy()
        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
        strikes = df["StrikeRate"].unique()
        strikes.sort()
        pain = []
        for K in strikes:
            total_loss = 0
            for s in strikes:
                if s in calls:
                    total_loss += max(0, K - s) * calls.get(s, 0)
                if s in puts:
                    total_loss += max(0, s - K) * puts.get(s, 0)
            pain.append((K, total_loss))
        if not pain:
            logger.warning("No valid strikes for max pain")
            return None, None
        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0
        logger.debug(f"Max Pain: Strike={max_pain_strike}, Diff%={max_pain_diff_pct:.2f}")
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None

# Initialize 5paisa client
def initialize_5paisa_client(secrets, totp_code):
    try:
        logger.info("Initializing 5paisa client")
        cred = {
            "APP_NAME": secrets["fivepaisa"]["APP_NAME"],
            "APP_SOURCE": secrets["fivepaisa"]["APP_SOURCE"],
            "USER_ID": secrets["fivepaisa"]["USER_ID"],
            "PASSWORD": secrets["fivepaisa"]["PASSWORD"],
            "USER_KEY": secrets["fivepaisa"]["USER_KEY"],
            "ENCRYPTION_KEY": secrets["fivepaisa"]["ENCRYPTION_KEY"]
        }
        client = FivePaisaClient(cred=cred)
        client.get_totp_session(
            secrets["fivepaisa"]["CLIENT_CODE"],
            totp_code,
            secrets["fivepaisa"]["PIN"]
        )
        if client.get_access_token():
            logger.info("5paisa client initialized successfully")
            return client
        else:
            logger.error("Failed to get access token")
            return None
    except Exception as e:
        logger.error(f"Error initializing 5paisa client: {str(e)}")
        return None

# Fetch real-time market data
def fetch_real_time_market_data(client):
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available")
        return None
    logger.info("Fetching real-time market data")
    nifty_spot = None
    vix = None
    atm_strike = None
    straddle_price = 0
    pcr = 0
    max_pain_strike = None
    max_pain_diff_pct = None
    expiry_date_str = None
    df_option_chain = pd.DataFrame()
    expiry_timestamp = None
    try:
        # Fetch NIFTY 50
        nifty_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920000,
            "Symbol": "NIFTY",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        nifty_market_feed = client.fetch_market_feed(nifty_req)
        if nifty_market_feed and nifty_market_feed.get("Data"):
            nifty_data = nifty_market_feed["Data"][0]
            nifty_spot = float(nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0)))
            logger.info(f"Fetched NIFTY Spot: {nifty_spot}")

        # Fetch India VIX
        vix_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920005,
            "Symbol": "INDIAVIX",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        vix_market_feed = client.fetch_market_feed(vix_req)
        if vix_market_feed and vix_market_feed.get("Data"):
            vix_data = vix_market_feed["Data"][0]
            vix = float(vix_data.get("LTP", vix_data.get("LastRate", 0)))
            logger.info(f"Fetched VIX: {vix}")

        # Fetch NIFTY expiries
        expiries = client.get_expiry("N", "NIFTY")
        if expiries and expiries.get("Expiry"):
            first_expiry = expiries["Expiry"][0]
            expiry_date_string = first_expiry.get("ExpiryDate")
            expiry_timestamp = parse_5paisa_date_string(expiry_date_string)
            expiry_date_str = format_timestamp_to_date_str(expiry_timestamp)
            logger.info(f"Fetched expiry: {expiry_date_str}")

        # Fetch Option Chain
        if expiry_timestamp:
            option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
            if option_chain and option_chain.get("Options"):
                df_option_chain = pd.DataFrame(option_chain["Options"])
                df_option_chain["StrikeRate"] = pd.to_numeric(df_option_chain["StrikeRate"], errors='coerce')
                df_option_chain["OpenInterest"] = pd.to_numeric(df_option_chain["OpenInterest"], errors='coerce').fillna(0)
                df_option_chain["LastRate"] = pd.to_numeric(df_option_chain["LastRate"], errors='coerce')
                df_option_chain = df_option_chain.dropna(subset=["StrikeRate", "OpenInterest", "LastRate"]).copy()
                logger.info(f"Option chain fetched: {len(df_option_chain)} rows")

        # Calculate ATM, Straddle, PCR, Max Pain
        if nifty_spot and not df_option_chain.empty:
            atm_strike_iloc = (df_option_chain["StrikeRate"] - nifty_spot).abs().argmin()
            atm_strike = df_option_chain["StrikeRate"].iloc[atm_strike_iloc]
            atm_data = df_option_chain[df_option_chain["StrikeRate"] == atm_strike]
            atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
            atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
            straddle_price = atm_call + atm_put
            calls_oi_sum = df_option_chain[df_option_chain["CPType"] == "CE"]["OpenInterest"].sum()
            puts_oi_sum = df_option_chain[df_option_chain["CPType"] == "PE"]["OpenInterest"].sum()
            pcr = puts_oi_sum / calls_oi_sum if calls_oi_sum != 0 else float("inf")
            max_pain_strike, max_pain_diff_pct = calculate_max_pain(df_option_chain, nifty_spot)
            logger.info(f"Calculated: ATM={atm_strike}, Straddle={straddle_price}, PCR={pcr}")

        vix_change_pct = ((vix / (df_option_chain["IV"].iloc[-2] if "IV" in df_option_chain.columns and len(df_option_chain) >= 2 else vix)) - 1) * 100 if vix else 0

        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "vix_change_pct": vix_change_pct,
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "expiry": expiry_date_str,
            "option_chain": df_option_chain,
            "source": "5paisa API (LIVE)"
        }
    except Exception as e:
        logger.error(f"Error fetching real-time data: {str(e)}")
        return None

# Fetch portfolio data
def fetch_all_api_portfolio_data(client):
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available")
        return {}
    logger.info("Fetching portfolio data")
    portfolio_data = {}
    try:
        portfolio_data["holdings"] = client.holdings()
        portfolio_data["margin"] = client.margin()
        portfolio_data["positions"] = client.positions()
        portfolio_data["order_book"] = client.order_book()
        portfolio_data["trade_book"] = client.get_tradebook()
        portfolio_data["market_status"] = client.get_market_status()
        logger.info("Portfolio data fetched")
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
    return portfolio_data

# Fetch market depth
def fetch_market_depth_by_scrip(client, Exchange, ExchangeType, ScripCode):
    if client is None or not client.get_access_token():
        logger.warning("Client not available for market depth")
        return None
    try:
        req = [{
            "Exch": Exchange,
            "ExchType": ExchangeType,
            "ScripCode": ScripCode
        }]
        depth = client.fetch_market_depth(req)
        if depth and depth.get("Data"):
            return depth
        logger.warning(f"No market depth for ScripCode {ScripCode}")
        return None
    except Exception as e:
        logger.error(f"Error fetching market depth for ScripCode {ScripCode}: {str(e)}")
        return None

# Prepare trade orders
def prepare_trade_orders(strategy, real_data, capital):
    logger.info(f"Preparing orders for: {strategy['Strategy']}")
    if not real_data or "option_chain" not in real_data or real_data["option_chain"].empty:
        logger.error("Invalid real-time data")
        return None
    option_chain = real_data["option_chain"]
    atm_strike = real_data["atm_strike"]
    expiry_date_str = real_data["expiry"]
    straddle_price_live = real_data["straddle_price"]
    lot_size = 75  # NIFTY lot size
    deploy = strategy["Deploy"]
    premium_per_lot = straddle_price_live * lot_size if straddle_price_live > 0 else 200 * lot_size
    lots = max(1, min(10, int(deploy / premium_per_lot)))
    orders_to_place = []
    strategy_legs = []

    if strategy["Strategy"] == "Short Straddle":
        strategy_legs = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")]
    elif strategy["Strategy"] == "Short Strangle":
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        call_strike = next((s for s in strikes_sorted if s >= atm_strike + 100), None)
        put_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)
        if call_strike and put_strike:
            strategy_legs = [(call_strike, "CE", "S"), (put_strike, "PE", "S")]
        else:
            logger.error("No suitable strikes for Short Strangle")
            return None
    elif strategy["Strategy"] == "Iron Condor":
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None)
        put_buy_strike = next((s for s in reversed(strikes_sorted) if s < put_sell_strike - 100), None)
        call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 50), None)
        call_buy_strike = next((s for s in strikes_sorted if s > call_sell_strike + 100), None)
        if all([put_sell_strike, put_buy_strike, call_sell_strike, call_buy_strike]):
            strategy_legs = [
                (put_buy_strike, "PE", "B"),
                (put_sell_strike, "PE", "S"),
                (call_sell_strike, "CE", "S"),
                (call_buy_strike, "CE", "B")
            ]
        else:
            logger.error("No suitable strikes for Iron Condor")
            return None

    for leg in strategy_legs:
        strike, cp_type, buy_sell = leg
        opt_data = option_chain[
            (option_chain["StrikeRate"] == strike) &
            (option_chain["CPType"] == cp_type)
        ]
        if opt_data.empty:
            logger.error(f"No data for {cp_type} at strike {strike} for expiry {expiry_date_str}")
            return None
        scrip_code = int(opt_data["ScripCode"].iloc[0])
        latest_price = float(opt_data["LastRate"].iloc[0]) if not pd.isna(opt_data["LastRate"].iloc[0]) else 0.0
        proposed_price = 0  # Market order
        stop_loss_price = latest_price * 0.9 if buy_sell == "B" else latest_price * 1.1
        take_profit_price = latest_price * 1.1 if buy_sell == "B" else latest_price * 0.9
        orders_to_place.append({
            "Strategy": strategy["Strategy"],
            "Leg_Type": f"{buy_sell} {cp_type}",
            "Strike": strike,
            "Expiry": expiry_date_str,
            "Exchange": "N",
            "ExchangeType": "D",
            "ScripCode": scrip_code,
            "Quantity_Lots": lots,
            "Quantity_Units": lots * lot_size,
            "Proposed_Price": proposed_price,
            "Last_Price_API": latest_price,
            "Stop_Loss_Price": stop_loss_price,
            "Take_Profit_Price": take_profit_price
        })
    logger.info(f"Prepared {len(orders_to_place)} orders")
    return orders_to_place

# Execute trade orders
def execute_trade_orders(client, prepared_orders):
    logger.info(f"Executing {len(prepared_orders)} orders")
    if client is None or not client.get_access_token():
        logger.error("5paisa client not available")
        return False, {"error": "Invalid client session"}
    if not prepared_orders:
        logger.warning("No orders to execute")
        return False, {"error": "No orders provided"}
    market_status = client.get_market_status()
    if not market_status.get("MarketStatus", {}).get("IsOpen", False):
        logger.error("Market is closed")
        return False, {"error": "Market is closed"}
    all_successful = True
    responses = []
    for order in prepared_orders:
        try:
            logger.info(f"Placing order: {order}")
            if not isinstance(order["ScripCode"], int) or order["ScripCode"] <= 0:
                logger.error(f"Invalid ScripCode: {order['ScripCode']}")
                all_successful = False
                responses.append({"Order": order, "Response": {"Status": -1, "Message": "Invalid ScripCode"}})
                continue
            response = client.place_order(
                OrderType=order["Leg_Type"].split(" ")[0].upper(),
                Exchange=order["Exchange"],
                ExchangeType=order["ExchangeType"],
                ScripCode=order["ScripCode"],
                Qty=order["Quantity_Units"],
                Price=order["Proposed_Price"],
                IsIntraday=False
            )
            logger.debug(f"Response for ScripCode {order['ScripCode']}: {response}")
            responses.append({"Order": order, "Response": response})
            if response.get("Status") != 0:
                all_successful = False
                error_message = response.get("Message", "Unknown error")
                logger.error(f"Order failed for ScripCode {order['ScripCode']}: {error_message}")
            else:
                logger.info(f"Order placed for ScripCode {order['ScripCode']}")
        except Exception as e:
            all_successful = False
            logger.error(f"Error for ScripCode {order['ScripCode']}: {str(e)}")
            responses.append({"Order": order, "Response": {"Status": -1, "Message": f"Exception: {e}"}})
    return all_successful, {"responses": responses}

# Square off positions
def square_off_positions(client):
    try:
        if client is None or not client.get_access_token():
            logger.error("5paisa client not available")
            return False
        logger.info("Squaring off all positions")
        response = client.squareoff_all()
        if response.get("Status") == 0:
            logger.info("Square off successful")
            return True
        else:
            logger.error(f"Square off failed: {response.get('Message', 'Unknown error')}")
            return False
    except Exception as e:
        logger.error(f"Error squaring off: {str(e)}")
        return False
