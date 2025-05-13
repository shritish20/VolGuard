import logging
import pandas as pd
import numpy as np
from datetime import datetime
import upstox_client
from upstox_client.rest import ApiException
import requests
import time
import re

# Setup logging
logger = logging.getLogger(__name__)

# Upstox API Configuration
base_url = "https://api.upstox.com/v2"
instrument_key = "NSE_INDEX|Nifty 50"

# Helper function to parse Upstox date string
def parse_upstox_date_string(date_string):
    if not isinstance(date_string, str):
        return None
    try:
        return datetime.strptime(date_string, "%Y-%m-%d").date()
    except ValueError:
        return None

# Initialize Upstox client
def initialize_upstox_client(access_token):
    try:
        logger.info("Initializing Upstox client")
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        options_api = upstox_client.OptionsApi(client)
        portfolio_api = upstox_client.PortfolioApi(client)
        user_api = upstox_client.UserApi(client)
        order_api = upstox_client.OrderApi(client)
        market_quote_api = upstox_client.MarketQuoteApi(client)
        logger.info("Upstox client initialized successfully")
        return {
            "client": client,
            "options_api": options_api,
            "portfolio_api": portfolio_api,
            "user_api": user_api,
            "order_api": order_api,
            "market_quote_api": market_quote_api,
            "access_token": access_token
        }
    except Exception as e:
        logger.error(f"Error initializing Upstox client: {str(e)}")
        return None

# Fetch real-time market data
def fetch_real_time_market_data(upstox_client):
    if not upstox_client or not upstox_client.get("access_token"):
        logger.warning("Upstox client not available")
        return None
    logger.info("Fetching real-time market data")
    headers = {"Authorization": f"Bearer {upstox_client['access_token']}", "Content-Type": "application/json"}
    nifty_spot = None
    vix = None
    atm_strike = None
    straddle_price = None
    pcr = None
    max_pain_strike = None
    max_pain_diff_pct = None
    expiry_date_str = None
    df_option_chain = pd.DataFrame()
    try:
        # Fetch India VIX
        url = f"{base_url}/market-quote/quotes"
        res = requests.get(url, headers=headers, params={"instrument_key": "NSE_INDEX|India VIX"})
        vix_data = res.json().get('data', {}).get('NSE_INDEX|India VIX', {})
        vix = float(vix_data.get('last_price', 0.0))
        logger.info(f"Fetched VIX: {vix}")
        time.sleep(0.5)  # Rate limiting

        # Fetch nearest expiry
        response = upstox_client['options_api'].get_option_contracts(instrument_key=instrument_key)
        contracts = response.to_dict().get("data", [])
        expiry_dates = set()
        for contract in contracts:
            exp = contract.get("expiry")
            if isinstance(exp, str):
                exp = parse_upstox_date_string(exp)
            if exp:
                expiry_dates.add(exp)
        expiry_list = sorted(expiry_dates)
        today = datetime.now().date()
        valid_expiries = [e for e in expiry_list if e >= today]
        nearest_expiry = valid_expiries[0] if valid_expiries else None
        expiry_date_str = nearest_expiry.strftime("%Y-%m-%d") if nearest_expiry else None
        logger.info(f"Fetched expiry: {expiry_date_str}")
        time.sleep(0.5)  # Rate limiting

        # Fetch option chain
        if expiry_date_str:
            res = upstox_client['options_api'].get_put_call_option_chain(instrument_key=instrument_key, expiry_date=expiry_date_str)
            chain = res.to_dict().get('data', [])
            if chain:
                nifty_spot = chain[0].get("underlying_spot_price", 0.0)
                logger.info(f"Fetched NIFTY Spot: {nifty_spot}")
                rows = []
                ce_oi_total = 0
                pe_oi_total = 0
                for r in chain:
                    ce = r.get('call_options', {})
                    pe = r.get('put_options', {})
                    ce_md = ce.get('market_data', {})
                    pe_md = pe.get('market_data', {})
                    ce_gk = ce.get('option_greeks', {})
                    pe_gk = pe.get('option_greeks', {})
                    strike = r['strike_price']
                    ce_oi = ce_md.get("oi", 0)
                    pe_oi = pe_md.get("oi", 0)
                    ce_oi_total += ce_oi
                    pe_oi_total += pe_oi
                    rows.append({
                        "StrikeRate": strike,
                        "CPType": "CE",
                        "LastRate": ce_md.get("ltp"),
                        "IV": ce_gk.get("iv"),
                        "Delta": ce_gk.get("delta"),
                        "Theta": ce_gk.get("theta"),
                        "Vega": ce_gk.get("vega"),
                        "OpenInterest": ce_oi,
                        "Volume": ce_md.get("volume", 0),
                        "ScripCode": ce.get("instrument_key")
                    })
                    rows.append({
                        "StrikeRate": strike,
                        "CPType": "PE",
                        "LastRate": pe_md.get("ltp"),
                        "IV": pe_gk.get("iv"),
                        "Delta": pe_gk.get("delta"),
                        "Theta": pe_gk.get("theta"),
                        "Vega": pe_gk.get("vega"),
                        "OpenInterest": pe_oi,
                        "Volume": pe_md.get("volume", 0),
                        "ScripCode": pe.get("instrument_key")
                    })
                df_option_chain = pd.DataFrame(rows)
                df_option_chain["StrikeRate"] = pd.to_numeric(df_option_chain["StrikeRate"], errors='coerce')
                df_option_chain["OpenInterest"] = pd.to_numeric(df_option_chain["OpenInterest"], errors='coerce').fillna(0)
                df_option_chain["LastRate"] = pd.to_numeric(df_option_chain["LastRate"], errors='coerce')
                df_option_chain = df_option_chain.dropna(subset=["StrikeRate", "OpenInterest", "LastRate"]).copy()
                logger.info(f"Option chain fetched: {len(df_option_chain)} rows")

        # Calculate ATM, Straddle, PCR, Max Pain
        if nifty_spot and not df_option_chain.empty:
            atm_idx = (df_option_chain["StrikeRate"] - nifty_spot).abs().argmin()
            atm_strike = df_option_chain["StrikeRate"].iloc[atm_idx]
            atm_data = df_option_chain[df_option_chain["StrikeRate"] == atm_strike]
            atm_call = atm_data[atm_data["CPType"] == "CE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0
            atm_put = atm_data[atm_data["CPType"] == "PE"]["LastRate"].iloc[0] if not atm_data[atm_data["CPType"] == "PE"].empty else 0
            straddle_price = atm_call + atm_put
            pcr = pe_oi_total / ce_oi_total if ce_oi_total != 0 else float("inf")
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
            "source": "Upstox API (LIVE)"
        }
    except Exception as e:
        logger.error(f"Error fetching real-time data: {str(e)}")
        return None

# Max Pain calculation (unchanged from 5paisa)
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

# Fetch portfolio data
def fetch_all_api_portfolio_data(upstox_client):
    if not upstox_client or not upstox_client.get("access_token"):
        logger.warning("Upstox client not available")
        return {}
    logger.info("Fetching portfolio data")
    portfolio_data = {}
    try:
        portfolio_data["holdings"] = upstox_client['portfolio_api'].get_holdings(api_version="v2").to_dict()
        portfolio_data["margin"] = upstox_client['user_api'].get_user_fund_margin(api_version="v2").to_dict()
        portfolio_data["positions"] = upstox_client['portfolio_api'].get_positions(api_version="v2").to_dict()
        portfolio_data["order_book"] = upstox_client['order_api'].get_order_book(api_version="v2").to_dict()
        portfolio_data["trade_book"] = upstox_client['order_api'].get_trade_history(api_version="v2").to_dict()
        logger.info("Portfolio data fetched")
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
    time.sleep(0.5)  # Rate limiting
    return portfolio_data

# Fetch market depth
def fetch_market_depth_by_scrip(upstox_client, instrument_key):
    if not upstox_client or not upstox_client.get("access_token"):
        logger.warning("Client not available for market depth")
        return None
    try:
        headers = {"Authorization": f"Bearer {upstox_client['access_token']}", "Content-Type": "application/json"}
        url = f"{base_url}/market-quote/depth"
        res = requests.get(url, headers=headers, params={"instrument_key": instrument_key})
        depth = res.json().get('data', {}).get(instrument_key, {}).get('depth', {})
        bid_volume = sum(item.get('quantity', 0) for item in depth.get('buy', []))
        ask_volume = sum(item.get('quantity', 0) for item in depth.get('sell', []))
        ltp = res.json().get('data', {}).get(instrument_key, {}).get('last_price', 0.0)
        logger.info(f"Depth for {instrument_key}: Bid Volume={bid_volume}, Ask Volume={ask_volume}, LTP={ltp}")
        time.sleep(0.5)  # Rate limiting
        return {"Data": [{"LastTradedPrice": ltp, "BidVolume": bid_volume, "AskVolume": ask_volume}]}
    except Exception as e:
        logger.error(f"Error fetching market depth for {instrument_key}: {str(e)}")
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
    lot_size = 25  # NIFTY lot size (updated May 2025)
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
        scrip_code = opt_data["ScripCode"].iloc[0]  # Upstox instrument_key
        latest_price = float(opt_data["LastRate"].iloc[0]) if not pd.isna(opt_data["LastRate"].iloc[0]) else 0.0
        proposed_price = 0  # Market order
        stop_loss_price = latest_price * 0.9 if buy_sell == "B" else latest_price * 1.1
        take_profit_price = latest_price * 1.1 if buy_sell == "B" else latest_price * 0.9
        orders_to_place.append({
            "Strategy": strategy["Strategy"],
            "Leg_Type": f"{buy_sell} {cp_type}",
            "Strike": strike,
            "Expiry": expiry_date_str,
            "Exchange": "NSE",
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
def execute_trade_orders(upstox_client, prepared_orders):
    logger.info(f"Executing {len(prepared_orders)} orders")
    if not upstox_client or not upstox_client.get("access_token"):
        logger.error("Upstox client not available")
        return False, {"error": "Invalid client session"}
    if not prepared_orders:
        logger.warning("No orders to execute")
        return False, {"error": "No orders provided"}
    # Check market status (simplified; Upstox doesn't expose direct market status API)
    now = datetime.now()
    is_market_open = now.hour >= 9 and now.hour < 15 and now.weekday() < 5
    if not is_market_open:
        logger.error("Market is closed")
        return False, {"error": "Market is closed"}
    all_successful = True
    responses = []
    for order in prepared_orders:
        try:
            logger.info(f"Placing order: {order}")
            if not isinstance(order["ScripCode"], str) or not order["ScripCode"].startswith("NSE_FO"):
                logger.error(f"Invalid ScripCode: {order['ScripCode']}")
                all_successful = False
                responses.append({"Order": order, "Response": {"status": -1, "message": "Invalid ScripCode"}})
                continue
            order_type = "BUY" if order["Leg_Type"].startswith("B") else "SELL"
            response = upstox_client['order_api'].place_order(
                body={
                    "instrument_key": order["ScripCode"],
                    "quantity": order["Quantity_Units"],
                    "order_type": "MARKET",
                    "transaction_type": order_type,
                    "product": "I",  # Intraday
                    "price": order["Proposed_Price"],
                    "validity": "DAY"
                },
                api_version="v2"
            )
            logger.debug(f"Response for ScripCode {order['ScripCode']}: {response.to_dict()}")
            responses.append({"Order": order, "Response": response.to_dict()})
            if response.status != "success":
                all_successful = False
                error_message = response.get("message", "Unknown error")
                logger.error(f"Order failed for ScripCode {order['ScripCode']}: {error_message}")
            else:
                logger.info(f"Order placed for ScripCode {order['ScripCode']}")
        except Exception as e:
            all_successful = False
            logger.error(f"Error for ScripCode {order['ScripCode']}: {str(e)}")
            responses.append({"Order": order, "Response": {"status": -1, "message": f"Exception: {e}"}})
        time.sleep(0.5)  # Rate limiting
    return all_successful, {"responses": responses}

# Square off positions
def square_off_positions(upstox_client):
    try:
        if not upstox_client or not upstox_client.get("access_token"):
            logger.error("Upstox client not available")
            return False
        logger.info("Squaring off all positions")
        positions = upstox_client['portfolio_api'].get_positions(api_version="v2").to_dict().get("data", [])
        if not positions:
            logger.info("No open positions to square off")
            return True
        success = True
        for pos in positions:
            try:
                instrument_key = pos.get("instrument_key")
                quantity = abs(pos.get("quantity", 0))
                transaction_type = "SELL" if pos.get("quantity", 0) > 0 else "BUY"
                if quantity > 0:
                    response = upstox_client['order_api'].place_order(
                        body={
                            "instrument_key": instrument_key,
                            "quantity": quantity,
                            "order_type": "MARKET",
                            "transaction_type": transaction_type,
                            "product": "I",
                            "price": 0,
                            "validity": "DAY"
                        },
                        api_version="v2"
                    )
                    if response.status != "success":
                        logger.error(f"Failed to square off {instrument_key}: {response.message}")
                        success = False
                    else:
                        logger.info(f"Squared off {instrument_key}")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Error squaring off {pos.get('instrument_key')}: {str(e)}")
                success = False
        return success
    except Exception as e:
        logger.error(f"Error squaring off: {str(e)}")
        return False
