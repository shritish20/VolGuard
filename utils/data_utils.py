import streamlit as st
import pandas as pd
import requests
import io
import logging
from py5paisa import FivePaisaClient
from utils.date_utils import parse_5paisa_date_string, format_timestamp_to_date_str

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_real_time_market_data(client):
    """Fetches real-time NIFTY 50, India VIX, and Option Chain data from 5paisa API."""
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
        # Fetch NIFTY 50
        nifty_req = [{"Exch": "N", "ExchType": "C", "ScripCode": 999920000, "Symbol": "NIFTY", "Expiry": "", "StrikePrice": "0", "OptionType": ""}]
        nifty_market_feed = client.fetch_market_feed(nifty_req)
        if nifty_market_feed and isinstance(nifty_market_feed, dict) and "Data" in nifty_market_feed and nifty_market_feed["Data"]:
            nifty_data = nifty_market_feed["Data"][0]
            nifty_spot = nifty_data.get("LastRate", nifty_data.get("LastTradedPrice", 0))
            logger.info(f"Fetched NIFTY Spot: {nifty_spot}")
        else:
            logger.error(f"Failed to fetch NIFTY 50 market feed: {nifty_market_feed}")

        # Fetch India VIX
        vix_req = [{"Exch": "N", "ExchType": "C", "ScripCode": 999920005, "Symbol": "INDIAVIX", "Expiry": "", "StrikePrice": "0", "OptionType": ""}]
        vix_market_feed = client.fetch_market_feed(vix_req)
        if vix_market_feed and isinstance(vix_market_feed, dict) and "Data" in vix_market_feed and vix_market_feed["Data"]:
            vix_data = vix_market_feed["Data"][0]
            vix = vix_data.get("LTP", vix_data.get("LastRate", vix_data.get("LastTradedPrice", 0)))
            logger.info(f"Fetched VIX: {vix}")
        else:
            logger.warning(f"Failed to fetch India VIX market feed: {vix_market_feed}")

        # Fetch NIFTY expiries
        expiries = client.get_expiry("N", "NIFTY")
        if expiries and isinstance(expiries, dict) and "Expiry" in expiries and expiries["Expiry"]:
            first_expiry = expiries["Expiry"][0]
            expiry_date_string = first_expiry.get("ExpiryDate")
            if expiry_date_string:
                expiry_timestamp = parse_5paisa_date_string(expiry_date_string)
                if expiry_timestamp:
                    expiry_date_str = format_timestamp_to_date_str(expiry_timestamp)
                    logger.info(f"Fetched first expiry: {expiry_date_str}")
                else:
                    logger.error(f"Could not parse timestamp: {expiry_date_string}")
            else:
                logger.error("Expiry data missing ExpiryDate.")

        # Fetch Option Chain
        if expiry_timestamp:
            option_chain = client.get_option_chain("N", "NIFTY", expiry_timestamp)
            if option_chain and isinstance(option_chain, dict) and "Options" in option_chain and option_chain["Options"]:
                df_option_chain = pd.DataFrame(option_chain["Options"])
                df_option_chain["StrikeRate"] = pd.to_numeric(df_option_chain["StrikeRate"], errors='coerce')
                df_option_chain["OpenInterest"] = pd.to_numeric(df_option_chain["OpenInterest"], errors='coerce').fillna(0)
                df_option_chain = df_option_chain.dropna(subset=["StrikeRate", "OpenInterest"]).copy()
                logger.debug(f"Option chain DataFrame shape: {df_option_chain.shape}")

                # Calculate ATM, Straddle, PCR, Max Pain
                if nifty_spot and not df_option_chain.empty:
                    atm_strike = df_option_chain["StrikeRate"].iloc[(df_option_chain["StrikeRate"] - nifty_spot).abs().argmin()]
                    atm_data = df_option_chain[df_option_chain["StrikeRate"] == atm_strike]
                    if 'LastRate' in atm_data.columns:
                        atm_call_data = atm_data[atm_data["CPType"] == "CE"]
                        atm_call = atm_call_data["LastRate"].iloc[0] if not atm_call_data.empty else 0
                        atm_put_data = atm_data[atm_data["CPType"] == "PE"]
                        atm_put = atm_put_data["LastRate"].iloc[0] if not atm_put_data.empty else 0
                        straddle_price = (atm_call + atm_put) if atm_call and atm_put else 0
                    if 'OpenInterest' in df_option_chain.columns:
                        calls_oi_sum = df_option_chain[df_option_chain["CPType"] == "CE"]["OpenInterest"].sum()
                        puts_oi_sum = df_option_chain[df_option_chain["CPType"] == "PE"]["OpenInterest"].sum()
                        pcr = puts_oi_sum / calls_oi_sum if calls_oi_sum != 0 else float("inf")
                    max_pain_strike, max_pain_diff_pct = _calculate_max_pain(df_option_chain, nifty_spot)

        logger.info("Real-time market data fetching completed.")
        return {
            "nifty_spot": nifty_spot,
            "vix": vix,
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
        logger.error(f"Error fetching real-time data: {str(e)}", exc_info=True)
        st.error(f"Error fetching real-time market data: {str(e)}")
        return None

def _calculate_max_pain(df: pd.DataFrame, nifty_spot: float):
    """Calculates the max pain strike."""
    try:
        if df.empty or "StrikeRate" not in df.columns or "CPType" not in df.columns or "OpenInterest" not in df.columns:
            logger.warning("Incomplete option chain data for max pain.")
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
            logger.warning("No valid strikes for max pain.")
            return None, None

        max_pain_strike = min(pain, key=lambda x: x[1])[0]
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100 if nifty_spot != 0 else 0
        logger.debug(f"Max Pain: Strike={max_pain_strike}, Diff%={max_pain_diff_pct:.2f}")
        return max_pain_strike, max_pain_diff_pct
    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}")
        return None, None

def load_data(client):
    """Loads data from 5paisa API first, falls back to CSV if API fails."""
    df = None
    real_data = None
    data_source = "CSV (FALLBACK)"

    # Attempt API data fetch
    real_data = fetch_real_time_market_data(client)
    if real_data and real_data.get("nifty_spot") is not None and real_data.get("vix") is not None:
        logger.info("Fetched real-time data from 5paisa API.")
        data_source = real_data["source"]
        latest_date = datetime.now().date()
        live_df_row = pd.DataFrame({
            "NIFTY_Close": [real_data["nifty_spot"]],
            "VIX": [real_data["vix"]]
        }, index=[pd.to_datetime(latest_date).normalize()])

        # Load historical CSV
        try:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))

            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            historical_df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna()

            historical_df = historical_df[historical_df.index < live_df_row.index[0]]
            df = pd.concat([historical_df, live_df_row])
            df = df.groupby(df.index).last().sort_index().ffill().bfill()
            logger.debug(f"Combined historical and live data. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading CSV with live data: {str(e)}")
            df = live_df_row
            logger.warning("Proceeding with only live data point.")
    else:
        logger.warning("Failed to fetch API data. Falling back to CSV.")
        try:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"
            nifty = pd.read_csv(io.StringIO(requests.get(nifty_url).text), encoding="utf-8-sig")
            vix = pd.read_csv(io.StringIO(requests.get(vix_url).text))

            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")
            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna()

            df = df.groupby(df.index).last().sort_index().ffill().bfill()
            logger.debug(f"Loaded CSV fallback data. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Fatal error loading CSV: {str(e)}")
            st.error(f"Fatal Error: Could not load data: {str(e)}")
            return None, None, "Data Load Failed"

    if df is None or len(df) < 2:
        st.error("Insufficient data loaded for analysis.")
        return None, None, data_source

    logger.debug(f"Data loading successful. Shape: {df.shape}. Source: {data_source}")
    return df, real_data, data_source
