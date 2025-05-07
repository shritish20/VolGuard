import logging
import io
import re
from datetime import datetime
from py5paisa import FivePaisaClient
import pandas as pd
import requests
import numpy as np # Added for isnan check

# Setup logging
logger = logging.getLogger(__name__)

# --- Helper function to parse the 5paisa date string format ---
def parse_5paisa_date_string(date_string):
    """
    Parses the /Date(1234567890000+0000)/ format string to a numerical timestamp (in milliseconds).
    Returns the numerical timestamp or None if parsing fails.
    """
    if not isinstance(date_string, str):
        return None
    match = re.search(r'/Date\((\d+)[+-]\d+\)/', date_string)
    if match:
        return int(match.group(1))
    return None

# --- Helper function to format timestamp to readable date string ---
def format_timestamp_to_date_str(timestamp_ms):
     """
     Converts a timestamp in milliseconds to a readableYYYY-MM-DD string.
     """
     if timestamp_ms is None:
          return "N/A"
     try:
          # Convert milliseconds to seconds
          timestamp_s = timestamp_ms / 1000
          # Convert timestamp to datetime object
          dt_object = datetime.fromtimestamp(timestamp_s)
          return dt_object.strftime("%Y-%m-%d")
     except Exception:
          return "N/A"

# --- Helper function for Max Pain Calculation ---
def calculate_max_pain(df: pd.DataFrame, nifty_spot: float):
    """
    Helper function to calculate the max pain strike.
    """
    try:
        if df.empty or "StrikeRate" not in df.columns or "CPType" not in df.columns or "OpenInterest" not in df.columns:
            logger.warning("Option chain data is incomplete or empty for max pain calculation.")
            return None, None

        try:
            df["StrikeRate"] = pd.to_numeric(df["StrikeRate"], errors='coerce')
            df["OpenInterest"] = pd.to_numeric(df["OpenInterest"], errors='coerce').fillna(0)
            df = df.dropna(subset=["StrikeRate", "OpenInterest"]).copy()
        except Exception as e:
            logger.error(f"Error converting columns for max pain: {e}")
            return None, None

        calls = df[df["CPType"] == "CE"].set_index("StrikeRate")["OpenInterest"]
        puts = df[df["CPType"] == "PE"].set_index("StrikeRate")["OpenInterest"]
        strikes = df["StrikeRate"].unique()
        strikes.sort()

        pain = []
        for K in strikes:
            total_loss = 0
            # Use .get() with a default value of 0 to handle missing strikes gracefully
            for s in strikes:
                 if s in calls:
                      total_loss += max(0, K - s) * calls.get(s, 0)
                 if s in puts:
                      total_loss += max(0, s - K) * puts.get(s, 0)
            pain.append((K, total_loss))

        if not pain:
             logger.warning("No valid strikes to calculate max pain.")
             return None, None

        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0

        logger.debug(f"Max Pain calculated: Strike={max_pain_strike}, Diff%={max_pain_diff_pct:.2f}")
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None


# 5paisa Client Initialization
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
        logger.info("Attempting TOTP session...")
        client.get_totp_session(
            secrets["fivepaisa"]["CLIENT_CODE"],
            totp_code,
            secrets["fivepaisa"]["PIN"]
        )
        if client.get_access_token():
            logger.info("5paisa client initialized and session obtained successfully")
            return client
        else:
            logger.error("Failed to get access token after TOTP session attempt.")
            return None
    except Exception as e:
        logger.error(f"Error initializing 5paisa client or getting session: {str(e)}")
        return None

# Data Fetching Functions
def fetch_real_time_market_data(client):
    """
    Fetches real-time NIFTY 50, India VIX, and Option Chain data from 5paisa API.
    """
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available or not logged in.")
        return None

    logger.info("Fetching real-time market data from 5paisa API")
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
        # 1. Fetch NIFTY 50
        nifty_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920000,
            "Symbol": "NIFTY",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        logger.debug(f"Fetching NIFTY market feed for: {nifty_req}")
        nifty_market_feed = client.fetch_market_feed(nifty_req)

        if not nifty_market_feed or not isinstance(nifty_market_feed, dict) or "Data" not in nifty_market_feed or not isinstance(nifty_market_feed["Data"], list) or not nifty_market_feed["Data"]:
            logger.error(f"Failed to fetch NIFTY 50 market feed or unexpected format. Response: {nifty_market_feed}")
        else:
             nifty_data = nifty_market_feed["Data"][0]
             nifty_spot = nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0))
             try:
                  nifty_spot = float(nifty_spot)
             except (ValueError, TypeError):
                  logger.warning("NIFTY spot price is not a valid number.")
                  nifty_spot = 0.0

             if not nifty_spot or nifty_spot == 0:
                 logger.warning("NIFTY price not found or is zero after parsing.")
             else:
                 logger.info(f"Fetched NIFTY Spot: {nifty_spot}")


        # 2. Fetch India VIX
        vix_req = [{
            "Exch": "N",
            "ExchType": "C",
            "ScripCode": 999920005,
            "Symbol": "INDIAVIX",
            "Expiry": "",
            "StrikePrice": "0",
            "OptionType": ""
        }]
        logger.debug(f"Fetching VIX market feed for: {vix_req}")
        vix_market_feed = client.fetch_market_feed(vix_req)

        if not vix_market_feed or not isinstance(vix_market_feed, dict) or "Data" not in vix_market_feed or not isinstance(vix_market_feed["Data"], list) or not vix_market_feed["Data"]:
            logger.warning(f"Failed to fetch India VIX market feed or unexpected format. Response: {vix_market_feed}")
        else:
            vix_data = vix_market_feed["Data"][0]
            vix = vix_data.get("LTP", vix_data.get("LastRate", vix_data.get("LastTradedPrice", 0)))
            try:
                 vix = float(vix)
            except (ValueError, TypeError):
                 logger.warning("VIX price is not a valid number.")
                 vix = None

            if vix is None or vix == 0:
                 logger.warning("VIX price not found or is zero after trying LTP/LastRate.")
                 vix = None
            else:
                 logger.info(f"Fetched VIX: {vix}")

        # 3. Fetch NIFTY expiries
        logger.debug("Fetching NIFTY expiries")
        expiries = client.get_expiry("N", "NIFTY")
        logger.debug(f"Expiries response type: {type(expiries)}, value: {expiries}")

        if not expiries or not isinstance(expiries, dict) or "Expiry" not in expiries or not isinstance(expiries["Expiry"], list) or not expiries["Expiry"]:
            logger.error(f"Failed to fetch NIFTY expiries or unexpected format. Response: {expiries}")
        else:
            first_expiry = expiries["Expiry"][0]
            expiry_date_string_from_api = first_expiry.get("ExpiryDate")

            if not expiry_date_string_from_api:
                 logger.error("Expiry data missing ExpiryDate in the first expiry item.")
            else:
                 expiry_timestamp = parse_5paisa_date_string(expiry_date_string_from_api)

                 if expiry_timestamp is not None:
                      expiry_date_str = format_timestamp_to_date_str(expiry_timestamp)
                      logger.info(f"Fetched first expiry: {expiry_date_str} (Timestamp: {expiry_timestamp})")
                 else:
                      logger.error(f"Could not parse timestamp from ExpiryDate string: {expiry_date_string_from_api}")

        # 4. Fetch Option Chain
        if expiry_timestamp is not None:
            logger.debug(f"Fetching Option Chain for expiry timestamp: {expiry_timestamp}")
            option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
            logger.debug(f"Option chain response type: {type(option_chain)}, value: {option_chain}")

            if not option_chain or not isinstance(option_chain, dict) or "Options" not in option_chain or not isinstance(option_chain["Options"], list) or not option_chain["Options"]:
                logger.error(f"Failed to fetch NIFTY option chain or unexpected format. Response: {option_chain}")
            else:
                 df_option_chain = pd.DataFrame(option_chain["Options"])
                 logger.debug(f"Option chain DataFrame created with shape: {df_option_chain.shape}")

                 required_cols = ["StrikeRate", "CPType", "LastRate", "OpenInterest", "ScripCode"]
                 if not all(col in df_option_chain.columns for col in required_cols):
                     missing = [col for col in required_cols if col not in df_option_chain.columns]
                     logger.warning(f"Required columns missing in option chain data DataFrame: {missing}")

                 df_option_chain["StrikeRate"] = pd.to_numeric(df_option_chain["StrikeRate"], errors='coerce')
                 df_option_chain["OpenInterest"] = pd.to_numeric(df_option_chain["OpenInterest"], errors='coerce').fillna(0)
                 df_option_chain["LastRate"] = pd.to_numeric(df_option_chain["LastRate"], errors='coerce') # Ensure LastRate is numeric

                 df_option_chain = df_option_chain.dropna(subset=["StrikeRate", "OpenInterest", "LastRate"]).copy()

                 if df_option_chain.empty:
                     logger.warning("Option chain DataFrame is empty after cleaning missing strikes/OI/LastRate.")

        else:
            logger.error("Cannot fetch Option Chain: Valid expiry timestamp was not obtained.")


        # 5. Calculate ATM, Straddle, PCR, Max Pain (only if Nifty and Option Chain are available)
        if nifty_spot is not None and nifty_spot > 0 and not df_option_chain.empty:
             logger.debug(f"Calculating ATM, Straddle, PCR, Max Pain for NIFTY spot: {nifty_spot}")
             if pd.api.types.is_numeric_dtype(df_option_chain["StrikeRate"]):
                 # Find ATM strike based on absolute difference to nifty_spot
                 atm_strike_iloc = (df_option_chain["StrikeRate"] - nifty_spot).abs().argmin()
                 atm_strike = df_option_chain["StrikeRate"].iloc[atm_strike_iloc]
                 atm_data = df_option_chain[df_option_chain["StrikeRate"] == atm_strike]

                 if 'LastRate' in atm_data.columns and pd.api.types.is_numeric_dtype(atm_data["LastRate"]):
                     atm_call_data = atm_data[atm_data["CPType"] == "CE"]
                     # Use .iloc[0] to get the value, check if empty before accessing
                     atm_call = atm_call_data["LastRate"].iloc[0] if not atm_call_data.empty and not pd.isna(atm_call_data["LastRate"].iloc[0]) else 0
                     atm_put_data = atm_data[atm_data["CPType"] == "PE"]
                     atm_put = atm_put_data["LastRate"].iloc[0] if not atm_put_data.empty and not pd.isna(atm_put_data["LastRate"].iloc[0]) else 0
                     straddle_price = (atm_call + atm_put) if atm_call is not None and atm_put is not None else 0
                 else:
                      logger.warning("LastRate column missing or not numeric for straddle calculation.")
                      straddle_price = 0.0


                 if 'OpenInterest' in df_option_chain.columns and pd.api.types.is_numeric_dtype(df_option_chain["OpenInterest"]):
                     calls_oi_sum = df_option_chain[df_option_chain["CPType"] == "CE"]["OpenInterest"].sum()
                     puts_oi_sum = df_option_chain[df_option_chain["CPType"] == "PE"]["OpenInterest"].sum()
                     pcr = puts_oi_sum / calls_oi_sum if calls_oi_sum != 0 else float("inf")
                 else:
                     logger.warning("OpenInterest column missing or not numeric for PCR calculation.")
                     pcr = 0.0

                 max_pain_strike, max_pain_diff_pct = calculate_max_pain(df_option_chain, nifty_spot)
                 logger.debug(f"Calculated ATM Strike: {atm_strike}, Straddle: {straddle_price}, PCR: {pcr}, Max Pain: {max_pain_strike}")
             else:
                  logger.warning("StrikeRate column is not numeric, cannot calculate ATM strike.")

        elif nifty_spot is None or nifty_spot <= 0:
             logger.warning("NIFTY spot price not available or is zero for calculating derivatives metrics.")
        elif df_option_chain.empty:
             logger.warning("Option chain is empty, cannot calculate derivatives metrics.")


        vix_change_pct = ((vix / (df_option_chain["IV"].iloc[-2] if "IV" in df_option_chain.columns and len(df_option_chain) >= 2 and not pd.isna(df_option_chain["IV"].iloc[-2]) else vix)) - 1) * 100 if vix is not None and vix > 0 else 0.0 # Calculate VIX change based on previous VIX or current if no history


        logger.info("Real-time market data fetching and processing function completed.")

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
            "source": "5paisa API (LIVE)" # Tag the source
        }

    except Exception as e:
        logger.error(f"An unexpected error occurred during real-time data fetch: {str(e)}", exc_info=True)
        return None


def fetch_all_api_portfolio_data(client):
    """
    Fetches comprehensive portfolio and account data from 5paisa API.
    """
    if client is None or not client.get_access_token():
        logger.warning("5paisa client not available or not logged in for portfolio data.")
        return {}

    logger.info("Fetching all API portfolio/account data")
    portfolio_data = {}

    try:
        portfolio_data["holdings"] = client.holdings()
        portfolio_data["margin"] = client.margin()
        portfolio_data["positions"] = client.positions()
        portfolio_data["order_book"] = client.order_book()
        portfolio_data["trade_book"] = client.get_tradebook()
        portfolio_data["market_status"] = client.get_market_status()
        logger.info("All API portfolio/account data fetched.")
    except Exception as e:
        logger.error(f"Error fetching some portfolio data: {str(e)}")

    return portfolio_data


def prepare_trade_orders(strategy, real_data, capital):
    """
    Prepares the list of individual orders for a given strategy based on real-time data.
    Returns a list of dictionaries representing the orders.
    """
    logger.info(f"Preparing trade orders for: {strategy['Strategy']}")
    if not real_data or "option_chain" not in real_data or "atm_strike" not in real_data or "expiry" not in real_data or "straddle_price" not in real_data:
        logger.error("Invalid or incomplete real-time data for order preparation.")
        return None

    option_chain = real_data["option_chain"]
    atm_strike = real_data["atm_strike"]
    expiry_date_str = real_data["expiry"]
    straddle_price_live = real_data["straddle_price"]

    if option_chain.empty or atm_strike is None or expiry_date_str == "N/A" or straddle_price_live is None or straddle_price_live <= 0:
         logger.error("Essential market data (Option Chain, ATM, Expiry, Straddle) is not valid for order preparation.")
         return None


    lot_size = 25 # Standard NIFTY lot size
    deploy = strategy["Deploy"]
    max_loss = strategy["Max_Loss"] # Max loss for the strategy

    # Determine lots based on deployable capital and live straddle price
    premium_per_lot = straddle_price_live * lot_size if straddle_price_live > 0 else 200 * lot_size # Fallback premium if live is zero
    lots = max(1, int(deploy / premium_per_lot) if premium_per_lot > 0 else 1) # Ensure at least 1 lot if deploy > 0

    # Ensure lots is a reasonable number, e.g., max 10 lots
    lots = min(lots, 10)


    orders_to_place = [] # List to hold prepared order dictionaries

    # Define strikes and types based on the strategy
    strategy_legs = []
    # Get timestamp from OC data for filtering if needed
    # expiry_timestamp = parse_5paisa_date_string(option_chain["ExpiryDate"].iloc[0])

    if strategy["Strategy"] == "Short Straddle":
        strategy_legs = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")]
    elif strategy["Strategy"] == "Short Strangle":
        # Need to find strikes approximately 100 points away - use option chain strikes
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        call_strike = None
        put_strike = None
        # Find OTM call strike >= atm_strike + 100
        for strike in strikes_sorted:
             if strike >= atm_strike + 100:
                  call_strike = strike
                  break
        # Find OTM put strike <= atm_strike - 100
        for strike in reversed(strikes_sorted):
             if strike <= atm_strike - 100:
                  put_strike = strike
                  break

        if call_strike is None or put_strike is None:
             logger.error(f"Could not find suitable strikes for Short Strangle around {atm_strike}.")
             return None

        strategy_legs = [(call_strike, "CE", "S"), (put_strike, "PE", "S")]

    elif strategy["Strategy"] == "Iron Condor":
        # Need 4 strikes: Buy OTM Put, Sell OTM Put, Sell OTM Call, Buy OTM Call
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        put_sell_strike = None
        put_buy_strike = None
        call_sell_strike = None
        call_buy_strike = None

        # Find strikes: Put Buy < Put Sell <= ATM < Call Sell < Call Buy
        # Start from ATM and move outwards
        put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None) # Approx 50-100 points below ATM
        put_buy_strike = next((s for s in reversed(strikes_sorted) if s < (put_sell_strike - 100 if put_sell_strike is not None else atm_strike - 150)), None) # Approx 100 points below sell put

        call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 50), None) # Approx 50-100 points above ATM
        call_buy_strike = next((s for s in strikes_sorted if s > (call_sell_strike + 100 if call_sell_strike is not None else atm_strike + 150)), None) # Approx 100 points above sell call


        if None in [put_sell_strike, put_buy_strike, call_sell_strike, call_buy_strike]:
            logger.error("Could not find suitable strikes for Iron Condor.")
            return None

        strategy_legs = [
            (put_buy_strike, "PE", "B"),
            (put_sell_strike, "PE", "S"),
            (call_sell_strike, "CE", "S"),
            (call_buy_strike, "CE", "B")
        ]

    elif strategy["Strategy"] == "Iron Fly":
        # Sell ATM Straddle, Buy OTM wings (e.g., 100 points out)
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
        put_buy_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)
        call_buy_strike = next((s for s in strikes_sorted if s >= atm_strike + 100), None)

        if put_buy_strike is None or call_buy_strike is None:
             logger.error("Could not find suitable strikes for Iron Fly wings.")
             return None

        strategy_legs = [
            (atm_strike, "PE", "S"),
            (atm_strike, "CE", "S"),
            (put_buy_strike, "PE", "B"),
            (call_buy_strike, "CE", "B")
        ]

    elif strategy["Strategy"] == "Butterfly Spread":
        # Using a Call Butterfly structure: Buy ITM, Sell 2x ATM, Buy OTM
        # Need to select strikes - simplify by using ATM and +/- 100 points if available
        strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()

        # A standard butterfly is equidistant. Let's try ATM +/- 100 points
        strike_lower_wing = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)
        strike_upper_wing = next((s for s in strikes_sorted if s >= atm_strike + 100), None)

        if strike_lower_wing is None or strike_upper_wing is None:
             logger.error("Could not find suitable strikes for Butterfly +/- 100 points.")
             return None

        # Assuming a short call butterfly (sell ATM, buy wings):
        strategy_legs = [
            (strike_lower_wing, "CE", "B"), # Buy ITM Call (or just lower wing)
            (atm_strike, "CE", "S"), # Sell ATM Call
            (atm_strike, "CE", "S"), # Sell ATM Call (2x quantity)
            (strike_upper_wing, "CE", "B") # Buy OTM Call (or just upper wing)
        ]
        # Need to adjust quantities below for the doubled leg


    elif strategy["Strategy"] == "Jade Lizard":
         # Short OTM Call, Short OTM Put, Long further OTM Put
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 100), None) # Approx 100 points OTM Call
         put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None) # Approx 100 points OTM Put
         put_buy_strike = next((s for s in reversed(strikes_sorted) if s < (put_sell_strike - 100 if put_sell_strike is not None else atm_strike - 150)), None) # Further OTM Put

         if None in [call_sell_strike, put_sell_strike, put_buy_strike]:
              logger.error("Could not find suitable strikes for Jade Lizard.")
              return None

         strategy_legs = [
             (call_sell_strike, "CE", "S"),
             (put_sell_strike, "PE", "S"),
             (put_buy_strike, "PE", "B")
         ]

    elif strategy["Strategy"] == "Calendar Spread":
         # Sell Near Month, Buy Far Month (same strike)
         # This requires fetching two different expiry option chains - NOT CURRENTLY IMPLEMENTED IN fetch_real_time_market_data
         # For simplicity in preparing orders, we will simulate this using the same expiry but note the limitation.
         # A real calendar spread needs the next expiry data.
         logger.warning("Calendar Spread requires fetching next expiry data, which is not fully supported in current fetch_real_time_market_data.")

         # Find the fetched expiry timestamp
         fetched_expiry_ts = parse_5paisa_date_string(option_chain["ExpiryDate"].iloc[0])
         if fetched_expiry_ts is None:
              logger.error("Could not get expiry timestamp from option chain data.")
              return None

         # To truly implement, you would need to call client.get_expiry again, find the *next* expiry,
         # then call client.get_option_chain for that next expiry.
         # For this example, we will just use the same expiry data as if it were the 'near' leg,
         # and mark the 'far' leg conceptually (won't have real ScripCode from the *actual* far expiry).
         # THIS IS A SIMPLIFICATION FOR DEMO PURPOSES.

         strategy_legs = [
             (atm_strike, "CE", "S"), # Sell Near Call @ ATM
             # (atm_strike, "CE", "B")  # Buy Far Call @ ATM - Requires next expiry data
         ]
         # Since we can't get the Far Leg ScripCode accurately with the current data,
         # we will just prepare the Short leg as a placeholder.
         # A full implementation needs modification of fetch_real_time_market_data or a new fetch function.


    elif strategy["Strategy"] == "Short Put Vertical Spread":
         # Sell OTM Put, Buy further OTM Put
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None) # Sell slightly OTM
         put_buy_strike = next((s for s in reversed(strikes_sorted) if s < (put_sell_strike - 100 if put_sell_strike is not None else atm_strike - 150)), None) # Buy further OTM

         if put_sell_strike is None or put_buy_strike is None:
              logger.error("Could not find suitable strikes for Short Put Vertical Spread.")
              return None

         strategy_legs = [
             (put_sell_strike, "PE", "S"),
             (put_buy_strike, "PE", "B")
         ]

    elif strategy["Strategy"] == "Short Call Vertical Spread":
         # Sell OTM Call, Buy further OTM Call
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 50), None) # Sell slightly OTM
         call_buy_strike = next((s for s in strikes_sorted if s > (call_sell_strike + 100 if call_sell_strike is not None else atm_strike + 150)), None) # Buy further OTM

         if call_sell_strike is None or call_buy_strike is None:
              logger.error("Could not find suitable strikes for Short Call Vertical Spread.")
              return None

         strategy_legs = [
             (call_sell_strike, "CE", "S"),
             (call_buy_strike, "CE", "B")
         ]

    elif strategy["Strategy"] == "Short Put":
         # Simple Short Put
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None) # Sell OTM Put

         if put_sell_strike is None:
              logger.error("Could not find suitable strike for Short Put.")
              return None

         strategy_legs = [
             (put_sell_strike, "PE", "S")
         ]

    elif strategy["Strategy"] == "Long Put":
         # Simple Long Put
         strikes_sorted = option_chain["StrikeRate"].sort_values().tolist()
         put_buy_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 50), None) # Buy OTM Put

         if put_buy_strike is None:
              logger.error("Could not find suitable strike for Long Put.")
              return None

         strategy_legs = [
             (put_buy_strike, "PE", "B")
         ]


    else:
        logger.error(f"Unsupported strategy for order preparation: {strategy['Strategy']}")
        return None

    # Prepare the actual order dictionaries
    for leg in strategy_legs:
        strike, cp_type, buy_sell = leg[:3] # Handle potential extra elements like "Near"/"Far"
        quantity_multiplier = 2 if strategy["Strategy"] == "Butterfly Spread" and (strike, cp_type, buy_sell) == (atm_strike, "CE", "S") else 1 # Handle doubled leg in Butterfly

        opt_data = option_chain[
            (option_chain["StrikeRate"] == strike) &
            (option_chain["CPType"] == cp_type)
            # Ensure this is for the fetched expiry timestamp if multiple expiries were fetched
            # (Assuming option_chain only contains data for the first expiry now)
            # (option_chain["ExpiryDate"] == fetched_expiry_ts) # Need to filter by timestamp if fetching multiple expiries
        ]

        if opt_data.empty:
            logger.error(f"No option chain data found for {cp_type} at strike {strike} for expiry {expiry_date_str}. Skipping leg.")
            continue # Skip this leg if data isn't found

        # Assuming the first match is correct if duplicates exist (shouldn't for unique strike/type/expiry)
        scrip_code = int(opt_data["ScripCode"].iloc[0])
        # Use LastRate as a proxy for current market price
        latest_price = float(opt_data["LastRate"].iloc[0]) if not pd.isna(opt_data["LastRate"].iloc[0]) else 0.0
        # Could also fetch live market depth here for Bids/Offers for better price estimation if needed

        # Determine default price for Market or Limit order display
        # For simplicity, propose a Market Order (Price=0) for now, user confirms
        # Could allow user to change to Limit Price in the UI confirmation step if implemented
        proposed_price = 0 # 0 for Market Order

        orders_to_place.append({
            "Strategy": strategy["Strategy"],
            "Leg_Type": f"{buy_sell} {cp_type}",
            "Strike": strike,
            "Expiry": expiry_date_str, # Use the formatted date string
            "Exchange": "N",
            "ExchangeType": "D", # Derivatives
            "ScripCode": scrip_code,
            "Quantity_Lots": lots * quantity_multiplier, # Show lots for clarity
            "Quantity_Units": lots * quantity_multiplier * lot_size, # Show units for API
            "Proposed_Price": proposed_price, # Proposed price (0 for Market)
            "Last_Price_API": latest_price # Show the last traded price from API
        })

    if not orders_to_place:
         logger.error("No valid order legs were prepared.")
         return None

    logger.info(f"Successfully prepared {len(orders_to_place)} order legs.")
    return orders_to_place


def execute_trade_orders(client, prepared_orders):
    """
    Executes a list of prepared orders using the 5paisa API.
    """
    logger.info(f"Attempting to execute {len(prepared_orders)} trade orders.")
    if client is None or not client.get_access_token():
        logger.error("5paisa client not available or not logged in for order execution.")
        return False

    if not prepared_orders:
        logger.warning("No prepared orders to execute.")
        return False

    all_successful = True
    responses = []

    for order in prepared_orders:
        try:
            logger.info(f"Placing order leg: {order['Leg_Type']} {order['Quantity_Units']} units of ScripCode {order['ScripCode']}")
            # Note: IsIntraday should potentially come from strategy logic or user choice
            response = client.place_order(
                OrderType=order["Leg_Type"].split(" ")[0].upper(), # 'BUY' or 'SELL'
                Exchange=order["Exchange"],
                ExchangeType=order["ExchangeType"],
                ScripCode=order["ScripCode"],
                Qty=order["Quantity_Units"], # Use quantity in units for API
                Price=order["Proposed_Price"], # Use the proposed price (likely 0 for Market)
                IsIntraday=False # Assuming positional for strategies unless specified otherwise
                # Can add validation/logic for other parameters like StopLossPrice here if needed
            )
            responses.append({"Order": order, "Response": response})
            logger.debug(f"Place order response for {order['ScripCode']}: {response}")

            if response.get("Status") != 0:
                all_successful = False
                error_message = response.get('Message', 'Unknown error')
                logger.error(f"Order failed for {order['ScripCode']}: {error_message}")
            else:
                 logger.info(f"Order placed successfully for {order['ScripCode']}. Response: {response}")

        except Exception as e:
            all_successful = False
            logger.error(f"Unexpected error during order placement for {order['ScripCode']}: {str(e)}", exc_info=True)
            responses.append({"Order": order, "Response": {"Status": -1, "Message": f"Exception: {e}"}})

    return all_successful # Indicate if the execution attempt was fully successful


def square_off_positions(client):
    """
    Calls the squareoff_all API endpoint.
    """
    try:
        logger.info("Attempting to square off all positions via API...")
        if client is None or not client.get_access_token():
            logger.error("5paisa client not available or not logged in for square off.")
            return False

        response = client.squareoff_all()
        logger.debug(f"Square off all response: {response}")

        if response.get("Status") == 0:
            logger.info("Square off all positions request sent successfully.")
            return True
        else:
            message = response.get("Message", "Unknown error")
            logger.error(f"Square off all failed: {message}")
            return False

    except Exception as e:
        logger.error(f"Error squaring off positions: {str(e)}", exc_info=True)
        return False
