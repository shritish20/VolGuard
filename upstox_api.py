import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, time as datetime_time # Import time separately to avoid conflict with time.sleep
import upstox_client
from upstox_client.rest import ApiException
import requests
import time
# import re # Not used in this file, can be removed if not needed elsewhere in the module

# Setup logging for this module
logger = logging.getLogger(__name__)

# Upstox API Configuration (Base URL)
base_url = "https://api.upstox.com/v2"
# Note: Specific instrument keys (like Nifty 50 index) are used dynamically in functions.

# Helper function to parse Upstox date string
def parse_upstox_date_string(date_string):
    """
    Parses date strings from Upstox API responses (typicallyYYYY-MM-DD)
    into Python date objects. Handles errors gracefully.
    """
    if not isinstance(date_string, str):
        logger.debug(f"Input for date parsing is not a string: {date_string}")
        return None
    try:
        # Upstox expiry format is typically YYYY-MM-DD
        parsed_date = datetime.strptime(date_string, "%Y-%m-%d").date()
        return parsed_date
    except ValueError:
        logger.warning(f"Invalid date format encountered during parsing: '{date_string}'. Expected YYYY-MM-DD.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred parsing date string '{date_string}': {e}")
        return None


# Initialize Upstox client with validation
def initialize_upstox_client(access_token: str):
    """
    Initializes the Upstox API client using the SDK and validates the access token
    by fetching the user profile. Instantiates key API classes.

    Args:
        access_token (str): The access token obtained from the Upstox OAuth process.
                            WARNING: Do not hardcode this in a production app.

    Returns:
        dict or None: A dictionary containing initialized SDK client objects
                      (`options_api`, `portfolio_api`, `user_api`, `order_api`,
                       `market_quote_api`) and the `access_token` itself if successful.
                      Returns None if initialization or token validation fails.
    """
    if not access_token or not isinstance(access_token, str) or len(access_token.strip()) == 0:
        logger.error("Access token is empty or invalid. Cannot initialize client.")
        return None

    logger.info("Attempting to initialize Upstox client and validate token.")

    try:
        # --- 1. Configure and create the core API client using the SDK ---
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token
        client = upstox_client.ApiClient(configuration)
        # Note: SDK client setup successful at this point, but token validity is pending.

        # --- 2. Instantiate necessary API classes from the SDK ---
        # These classes provide the methods to call specific API endpoints.
        user_api = upstox_client.UserApi(client)
        options_api = upstox_client.OptionsApi(client)
        portfolio_api = upstox_client.PortfolioApi(client)
        order_api = upstox_client.OrderApi(client)
        market_quote_api = upstox_client.MarketQuoteApi(client) # Instance for market data endpoints

        logger.info("Upstox SDK API classes instantiated.")

        # --- 3. Validate access token by making a test API call ---
        # Fetching user profile is a common way to confirm the token is valid and active.
        logger.info("Validating token by fetching user profile...")
        # Use a small timeout for this validation call if possible, although SDK might not expose it easily.
        user_profile = user_api.get_profile(api_version="v2")

        # Check if the profile data was successfully returned in the response
        if user_profile and user_profile.data:
            user_name = user_profile.data.get('user_name', 'N/A')
            user_id = user_profile.data.get('user_id', 'N/A')
            logger.info(f"Token validated successfully for user: {user_name} (ID: {user_id}).")
            # Return the dictionary of initialized clients and the access token.
            # The access token is included because some functions below use `requests` directly.
            return {
                "client": client, # The core API client instance (might not be directly needed by app, but useful)
                "user_api": user_api,
                "options_api": options_api,
                "portfolio_api": portfolio_api,
                "order_api": order_api,
                "market_quote_api": market_quote_api, # Include if needed by other functions in this module
                "access_token": access_token # Required for `requests` calls
            }
        else:
            # If the API call was successful but returned no data or validation failed internally
            logger.error("Token validation failed: API call successful, but unable to retrieve user profile data.")
            return None # Indicate failure


    except ApiException as e:
        # Catch specific Upstox API exceptions during initialization or the profile fetch
        logger.error(f"Upstox API exception during initialization/validation: Status={e.status}, Body={e.body}")
        # Provide more specific error guidance based on common API status codes if possible
        if e.status == 401:
            logger.error("Reason: Authentication failed. The access token is likely invalid or expired.")
        return None # Indicate failure

    except requests.exceptions.RequestException as req_e:
         # Catch network-related errors during the API call (e.g., connection issues)
         logger.error(f"Network error during Upstox API initialization/validation: {req_e}")
         return None # Indicate failure

    except Exception as e:
        # Catch any other unexpected errors during the initialization process
        logger.error(f"An unexpected error occurred during Upstox client initialization: {str(e)}", exc_info=True)
        return None # Indicate failure


# Fetch real-time market data
def fetch_real_time_market_data(upstox_client: dict):
    """
    Fetches the latest real-time market data for Nifty 50 including spot price,
    VIX, nearest expiry date, full option chain, and calculates key metrics
    (PCR, Max Pain, ATM Strike, Straddle Price).
    Uses a mix of Upstox SDK and requests library as implemented in the original code.

    Args:
        upstox_client (dict): Dictionary of initialized Upstox API client objects
                             and the access token, as returned by initialize_upstox_client.
                             Requires keys 'access_token', 'options_api'.

    Returns:
        dict or None: A dictionary containing the latest real-time market data
                      and calculated metrics, or None if fetching failed.
    """
    # Ensure the client dictionary and access token are available
    if not upstox_client or not upstox_client.get("access_token") or 'options_api' not in upstox_client:
        logger.warning("Upstox client, access token, or options_api instance not available for fetching real-time market data.")
        return None

    logger.info("Starting fetch of real-time market data.")
    # Prepare headers using the access token for `requests` calls
    headers = {"Authorization": f"Bearer {upstox_client['access_token']}", "Content-Type": "application/json"}
    # Instrument key for Nifty 50 index (consistent definition)
    nifty_index_key = "NSE_INDEX|Nifty 50"

    # Initialize variables to None to indicate data is not yet fetched/calculated
    nifty_spot = None
    vix = None
    atm_strike = None
    straddle_price = None
    pcr = None
    max_pain_strike = None
    max_pain_diff_pct = None
    expiry_date_str = None
    df_option_chain = pd.DataFrame() # Initialize as empty DataFrame
    source_tag = "Upstox API (LIVE)" # Source tag for the data


    try:
        # --- 1. Fetch India VIX using the requests library ---
        # The original code used requests for this, which is fine.
        logger.info("Fetching India VIX using requests.")
        vix_url = f"{base_url}/market-quote/quotes"
        vix_params = {"instrument_key": "NSE_INDEX|India VIX"}
        vix_res = requests.get(vix_url, headers=headers, params=vix_params)
        vix_res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        vix_data = vix_res.json().get('data', {}).get('NSE_INDEX|India VIX', {})
        vix_price = vix_data.get('last_price')

        if vix_price is not None:
            vix = pd.to_numeric(vix_price, errors='coerce').mean() # Use .mean() with coerce to handle potential list/string and get single numeric
            if pd.isna(vix): vix = None # Set back to None if conversion resulted in NaN
            logger.info(f"Successfully fetched VIX: {vix}")
        else:
            logger.warning("VIX 'last_price' not found in quotes response or is invalid.")
            vix = None # Explicitly set to None if not found or invalid
        time.sleep(0.5)  # Basic Rate limiting after VIX fetch


        # --- 2. Fetch Nearest Expiry Date using Upstox SDK ---
        logger.info("Fetching option contracts to determine nearest expiry using SDK.")
        try:
            options_api = upstox_client['options_api'] # Get the OptionsApi instance from the client dict
            expiry_response = options_api.get_option_contracts(instrument_key=nifty_index_key)

            # The response body is an object, convert to dict and get the 'data' list
            contracts_data = expiry_response.to_dict().get("data", [])

            if not contracts_data:
                logger.warning(f"No option contracts returned for {nifty_index_key} from SDK.")
                # Cannot proceed without expiry date, return None
                return None

            expiry_dates_set = set()
            # Collect and parse all expiry dates from the contracts data
            for contract in contracts_data:
                exp_str = contract.get("expiry")
                exp_date_obj = parse_upstox_date_string(exp_str) # Use the helper parsing function
                if exp_date_obj: # Add to set only if parsing was successful
                    expiry_dates_set.add(exp_date_obj)

            # Sort the unique expiry dates
            expiry_list_sorted = sorted(list(expiry_dates_set))

            today = datetime.now().date()
            # Find the first expiry date that is today or in the future
            valid_expiries = [exp for exp in expiry_list_sorted if exp and exp >= today] # Ensure exp is not None

            if not valid_expiries:
                logger.error(f"No valid future expiry dates found for {nifty_index_key}.")
                # Cannot proceed without a valid future expiry, return None
                return None

            # The nearest valid expiry date is the first one in the sorted list
            nearest_expiry_date_obj = valid_expiries[0]
            expiry_date_str = nearest_expiry_date_obj.strftime("%Y-%m-%d") # Format as YYYY-MM-DD string
            logger.info(f"Successfully fetched nearest expiry date: {expiry_date_str}")
            time.sleep(0.5)  # Basic Rate limiting after expiry fetch


        except ApiException as api_e:
             logger.error(f"Upstox SDK exception fetching option contracts: Status={api_e.status}, Body={api_e.body}")
             return None # Indicate failure if this SDK call fails
        except requests.exceptions.RequestException as req_e:
             # This catch might be redundant if SDK handles requests errors internally, but kept for safety
             logger.error(f"Network error during SDK option contracts fetch: {req_e}")
             return None
        except Exception as e:
             logger.error(f"An unexpected error occurred fetching nearest expiry: {str(e)}", exc_info=True)
             return None


        # --- 3. Fetch Option Chain using Upstox SDK ---
        chain_data_raw = [] # Initialize as empty list
        if expiry_date_str: # Proceed only if expiry date was successfully determined
            try:
                logger.info(f"Fetching option chain for expiry {expiry_date_str} using SDK.")
                options_api = upstox_client['options_api'] # Get the OptionsApi instance
                chain_response = options_api.get_put_call_option_chain(instrument_key=nifty_index_key, expiry_date=expiry_date_str)

                # Convert the response object to dictionary and get the 'data' list
                chain_data_raw = chain_response.to_dict().get('data', [])

                if not chain_data_raw:
                     logger.warning(f"No option chain data returned for expiry {expiry_date_str} from SDK.")
                     # chain_data_raw remains an empty list, will be handled below


                time.sleep(0.5)  # Basic Rate limiting after option chain fetch

            except ApiException as api_e:
                 logger.error(f"Upstox SDK exception fetching option chain: Status={api_e.status}, Body={api_e.body}")
                 chain_data_raw = [] # Ensure empty list on API error
            except requests.exceptions.RequestException as req_e:
                 # This catch might be redundant if SDK handles requests errors internally, but kept for safety
                 logger.error(f"Network error during SDK option chain fetch: {req_e}")
                 chain_data_raw = []
            except Exception as e:
                logger.error(f"An unexpected error occurred fetching option chain for {expiry_date_str}: {str(e)}", exc_info=True)
                chain_data_raw = []


        # --- 4. Process Option Chain Data and Calculate Metrics ---
        # This step processes the raw chain data into a structured format (DataFrame)
        # and calculates key metrics like PCR, Max Pain, etc.
        if chain_data_raw:
            logger.info("Processing raw option chain data and calculating metrics.")
            # Nifty spot price is typically included in the first element of the chain data
            # Check if chain_data_raw is not empty before accessing index 0
            nifty_spot_from_chain = chain_data_raw[0].get("underlying_spot_price") if chain_data_raw else None

            if nifty_spot_from_chain is not None:
                nifty_spot = pd.to_numeric(nifty_spot_from_chain, errors='coerce').mean() # Convert and handle potential format issues
                if pd.isna(nifty_spot): nifty_spot = None
                logger.info(f"Successfully extracted Nifty Spot from option chain data: {nifty_spot}")
            else:
                 logger.warning("Nifty spot price not found in the first element of option chain data.")
                 # If spot is missing from the chain, attempt to fetch it separately using requests
                 try:
                     logger.info("Attempting to fetch Nifty Spot separately using requests.")
                     nifty_quote_url = f"{base_url}/market-quote/quotes"
                     nifty_quote_params = {"instrument_key": nifty_index_key}
                     nifty_quote_res = requests.get(nifty_quote_url, headers=headers, params=nifty_quote_params)
                     nifty_quote_res.raise_for_status()
                     nifty_spot_data_sep = nifty_quote_res.json().get('data', {}).get(nifty_index_key, {})
                     nifty_spot_sep = nifty_spot_data_sep.get('last_price')
                     if nifty_spot_sep is not None:
                          nifty_spot = pd.to_numeric(nifty_spot_sep, errors='coerce').mean()
                          if pd.isna(nifty_spot): nifty_spot = None
                          logger.info(f"Successfully fetched Nifty Spot separately: {nifty_spot}")
                     else:
                          logger.warning("Nifty Spot 'last_price' not found in separate quotes response either or is invalid.")
                          nifty_spot = None # Still None if separate fetch fails or is invalid
                     time.sleep(0.5)
                 except requests.exceptions.RequestException as req_e:
                      logger.warning(f"Network error fetching Nifty Spot separately: {req_e}")
                      nifty_spot = None
                 except Exception as e:
                      logger.warning(f"An unexpected error occurred fetching Nifty Spot separately: {e}")
                      nifty_spot = None


            # If Nifty spot is still None, we cannot calculate metrics depending on it
            if nifty_spot is None:
                 logger.error("Nifty Spot price could not be determined. Skipping metric calculations dependent on spot.")
                 # Proceed with potentially empty df_option_chain and None metrics dependent on spot


            # Build a list of rows for the option chain DataFrame
            rows_list = []
            ce_oi_total, pe_oi_total = 0, 0 # Initialize total OI counters

            for strike_data in chain_data_raw:
                call_options = strike_data.get('call_options', {})
                put_options = strike_data.get('put_options', {})
                call_market_data = call_options.get('market_data', {})
                put_market_data = put_options.get('market_data', {})
                call_greeks = call_options.get('option_greeks', {})
                put_greeks = put_options.get('option_greeks', {})
                strike = strike_data.get('strike_price')

                if strike is None:
                    logger.warning(f"Skipping option chain entry with missing strike_price: {strike_data}")
                    continue # Skip this entry if strike price is missing

                # Get OI values, defaulting to 0 if missing or None, ensure numeric
                ce_oi_val = pd.to_numeric(call_market_data.get("oi"), errors='coerce').fillna(0).sum() # .sum() to handle potential list formats
                pe_oi_val = pd.to_numeric(put_market_data.get("oi"), errors='coerce').fillna(0).sum() # .sum() to handle potential list formats

                # Accumulate total OI
                ce_oi_total += ce_oi_val
                pe_oi_total += pe_oi_val

                # Add data for the Call option at this strike
                rows_list.append({
                    "StrikeRate": strike,
                    "CPType": "CE",
                    "LastRate": call_market_data.get("ltp"),
                    "IV": call_greeks.get("iv"),
                    "Delta": call_greeks.get("delta"),
                    "Theta": call_greeks.get("theta"),
                    "Vega": call_greeks.get("vega"),
                    "OpenInterest": ce_oi_val, # Use numeric OI value
                    "Volume": call_market_data.get("volume", 0) or 0, # Use .get with default
                    "ScripCode": call_options.get("instrument_key"), # Upstox instrument key for the CE contract
                    # Add other relevant data if needed from the API response
                })
                # Add data for the Put option at this strike
                rows_list.append({
                    "StrikeRate": strike,
                    "CPType": "PE",
                    "LastRate": put_market_data.get("ltp"),
                    "IV": put_greeks.get("iv"),
                    "Delta": put_greeks.get("delta"),
                    "Theta": put_greeks.get("theta"),
                    "Vega": put_greeks.get("vega"),
                    "OpenInterest": pe_oi_val, # Use numeric OI value
                    "Volume": put_market_data.get("volume", 0) or 0, # Use .get with default
                    "ScripCode": put_options.get("instrument_key"), # Upstox instrument key for the PE contract
                })

            # Create the option chain DataFrame from the list of rows
            df_option_chain = pd.DataFrame(rows_list)

            # Ensure columns that should be numeric are correctly typed, coerce errors to NaN
            numeric_oc_cols = ["StrikeRate", "LastRate", "IV", "Delta", "Theta", "Vega", "OpenInterest", "Volume"]
            for col in numeric_oc_cols:
                 if col in df_option_chain.columns:
                     df_option_chain[col] = pd.to_numeric(df_option_chain[col], errors='coerce')

            # Drop rows where essential numeric data (like strike or OI) failed to parse
            df_option_chain = df_option_chain.dropna(subset=["StrikeRate", "OpenInterest"]).copy() # Work on a copy


            # Calculate derived metrics ONLY if Nifty spot and valid option chain data are available
            if nifty_spot is not None and not df_option_chain.empty:
                # Find ATM strike (strike closest to Nifty spot)
                # Ensure StrikeRate is numeric before calculating absolute difference
                if "StrikeRate" in df_option_chain.columns and pd.api.types.is_numeric_dtype(df_option_chain["StrikeRate"]):
                     # Use idxmin() on the absolute difference series to get the index of the minimum value
                     atm_idx_iloc = (df_option_chain["StrikeRate"] - nifty_spot).abs().idxmin()
                     atm_strike = df_option_chain.loc[atm_idx_iloc, "StrikeRate"] # Get the strike value using .loc
                     logger.info(f"Determined ATM strike: {atm_strike}")
                else:
                     logger.warning("StrikeRate column is missing or not numeric. Cannot determine ATM strike.")
                     atm_strike = None


                # Calculate ATM Straddle Price
                straddle_price = None # Initialize to None
                if atm_strike is not None:
                     # Find the data for the ATM strike in the DataFrame
                     atm_data_df = df_option_chain[df_option_chain["StrikeRate"] == atm_strike]
                     # Get LTP for ATM Call and Put, handle potential NaNs or missing 'LastRate' column
                     atm_call_ltp = pd.to_numeric(atm_data_df[atm_data_df["CPType"] == "CE"]["LastRate"].iloc[0], errors='coerce') if not atm_data_df[atm_data_df["CPType"] == "CE"].empty and "LastRate" in atm_data_df.columns else None
                     atm_put_ltp = pd.to_numeric(atm_data_df[atm_data_df["CPType"] == "PE"]["LastRate"].iloc[0], errors='coerce') if not atm_data_df[atm_data_df["CPType"] == "PE"].empty and "LastRate" in atm_data_df.columns else None
                     # Sum LTPs, treating None/NaN as 0 for calculation
                     straddle_price = float(atm_call_ltp if pd.notna(atm_call_ltp) else 0) + float(atm_put_ltp if pd.notna(atm_put_ltp) else 0)
                     logger.info(f"Calculated ATM Straddle Price: {straddle_price:.2f}")
                else:
                    logger.warning("ATM strike is None. Cannot calculate Straddle Price.")


                # Calculate overall PCR (Put-Call Ratio)
                # Ensure total OI is not zero before dividing
                if ce_oi_total is not None and pe_oi_total is not None: # Check if totals were successfully summed
                    # Handle division by zero. If CE OI is 0, PCR is effectively infinite if PE OI > 0, or 0 if PE OI is also 0.
                    pcr = pe_oi_total / ce_oi_total if ce_oi_total != 0 else (float('inf') if pe_oi_total > 0 else 0.0)
                    # Round PCR to 2 decimal places
                    pcr = round(pcr, 2) if pd.notna(pcr) and pcr != float('inf') else pcr # Only round if not NaN or inf
                    logger.info(f"Calculated overall PCR: {pcr:.2f}" if pd.notna(pcr) and pcr != float('inf') else f"Calculated overall PCR: {pcr}")
                else:
                    logger.warning("Total CE OI or PE OI is None. Cannot calculate overall PCR.")
                    pcr = None # Set PCR to None if totals were not valid or calculable


                # Calculate Max Pain Strike and difference percentage
                # Requires a valid option chain DataFrame and Nifty Spot
                if not df_option_chain.empty and nifty_spot is not None:
                    # Pass a copy of the DataFrame to calculate_max_pain
                    max_pain_strike, max_pain_diff_pct = calculate_max_pain(df_option_chain.copy(), nifty_spot)
                    if max_pain_strike is not None:
                         logger.info(f"Calculated Max Pain Strike: {max_pain_strike:.2f}, Diff %: {max_pain_diff_pct:.2f}%")
                    else:
                         logger.warning("Max Pain calculation failed.")
                else:
                    logger.warning("Option chain data or Nifty Spot missing for Max Pain calculation.")
                    max_pain_strike = None
                    max_pain_diff_pct = None


            else:
                 logger.warning("Raw option chain data is empty. Skipping processing and metric calculations dependent on chain data.")


        else:
             logger.warning("No raw option chain data received. Skipping processing.")


        # --- VIX Change Percentage ---
        # As discussed, calculating VIX Change Percentage accurately requires the *previous* day's VIX.
        # This single real-time API call does not provide historical VIX.
        # Therefore, VIX_Change_Pct cannot be reliably calculated *within this function*.
        # The data_processing module calculates it based on the historical VIX series + the latest real VIX.
        # We will include the key in the returned dict structure but set it to None here.
        vix_change_pct = None
        # If you needed it calculated here, you would need to pass in the previous day's VIX value.


        # --- Prepare the final result dictionary ---
        # This dictionary contains all the latest real-time data points and calculated metrics
        result = {
            "nifty_spot": nifty_spot,
            "vix": vix,
            "vix_change_pct": vix_change_pct, # Likely None here
            "atm_strike": atm_strike,
            "straddle_price": straddle_price,
            "pcr": pcr,
            "max_pain_strike": max_pain_strike,
            "max_pain_diff_pct": max_pain_diff_pct,
            "expiry": expiry_date_str, # YYYY-MM-DD string
            "option_chain": df_option_chain, # The DataFrame of the latest option chain
            "source": source_tag # Indicate data source was live API
            # Note: Market Depth and User/Portfolio Data are fetched by separate functions in this module
        }

        logger.info("Real-time market data fetch process completed.")
        return result

    except Exception as e:
        # Catch any unexpected errors during the entire process
        logger.error(f"An unexpected error occurred during real-time market data fetch: {str(e)}", exc_info=True)
        # Log traceback for better debugging
        # import traceback # Already imported at the top
        # logger.error(f"Traceback: {traceback.format_exc()}")
        return None # Indicate overall failure


# Max Pain calculation (Helper function, used within fetch_real_time_market_data)
def calculate_max_pain(df: pd.DataFrame, nifty_spot: float):
    """
    Calculates the Max Pain strike price for the given option chain data.
    Max Pain is the strike price at which the total loss for option writers
    across all strikes is minimized.

    Args:
        df (pd.DataFrame): DataFrame containing option chain data with
                           'StrikeRate', 'CPType', 'OpenInterest'.
                           Assumes 'StrikeRate' is numeric and 'OpenInterest' are numeric (or coercible).
        nifty_spot (float): The current underlying spot price.

    Returns:
        tuple: (max_pain_strike (float or None), max_pain_diff_pct (float or None))
               Returns (None, None) if calculation is not possible due to missing data or errors.
    """
    try:
        logger.debug("Calculating Max Pain.")
        # Ensure required columns are present
        if df.empty or not all(col in df.columns for col in ["StrikeRate", "CPType", "OpenInterest"]):
            logger.warning("Option chain DataFrame is incomplete or empty for max pain calculation.")
            return None, None

        # Work on a copy and ensure numeric types for critical columns, dropping rows where essential data is missing
        df_valid = df.copy()
        df_valid["StrikeRate"] = pd.to_numeric(df_valid["StrikeRate"], errors='coerce').dropna() # Drop rows where StrikeRate is not numeric
        df_valid["OpenInterest"] = pd.to_numeric(df_valid["OpenInterest"], errors='coerce').fillna(0) # Fill non-numeric OI with 0

        if df_valid.empty or nifty_spot is None or not isinstance(nifty_spot, (int, float)):
             logger.warning("Insufficient valid data rows or invalid Nifty spot for max pain calculation.")
             return None, None

        # Get unique strike prices from the valid data
        strikes = df_valid["StrikeRate"].unique()
        strikes.sort() # Sort strikes in ascending order

        pain_points = [] # List to store (strike, total_loss) tuples

        # Iterate through each unique strike price (K) to calculate total loss for option writers at that strike
        for K in strikes:
            # Loss for Call writers: sum of (K - strike) * OI for all Call options at strikes < K
            # Call writers lose if the spot price is below their strike at expiry.
            # If expiry spot is K, Call writer at strike s loses max(0, s - K) per share if they sold a call.
            # However, Max Pain calculation assumes option *exercised* if profitable for buyer at price K.
            # So, Call writer at strike s loses K-s if K > s, for calls they *sold*.
            call_writer_loss = df_valid[
                (df_valid["CPType"] == "CE") &
                (df_valid["StrikeRate"] < K) # Strikes below K become ITM if spot is K
            ].apply(
                lambda row: (K - row["StrikeRate"]) * row["OpenInterest"], axis=1 # Loss per share is K - strike if strike < K
            ).sum()


            # Loss for Put writers: sum of (strike - K) * OI for all Put options at strikes > K
            # Put writers lose if the spot price is above their strike at expiry.
            # If expiry spot is K, Put writer at strike s loses max(0, K - s) per share if they sold a put.
            # So, Put writer at strike s loses s-K if K < s, for puts they *sold*.
            put_writer_loss = df_valid[
                (df_valid["CPType"] == "PE") &
                (df_valid["StrikeRate"] > K) # Strikes above K become ITM if spot is K
            ].apply(
                 lambda row: (row["StrikeRate"] - K) * row["OpenInterest"], axis=1 # Loss per share is strike - K if strike > K
            ).sum()

            # Total theoretical loss for option writers at strike K
            total_loss = call_writer_loss + put_writer_loss
            pain_points.append((K, total_loss))

        # Find the strike price with the minimum total loss for writers
        if not pain_points:
            logger.warning("Max pain calculation resulted in no pain points.")
            return None, None

        # Use min() with a key function to find the tuple with the minimum total loss
        min_pain_point = min(pain_points, key=lambda x: x[1])
        max_pain_strike = min_pain_point[0] # The strike price

        # Calculate the percentage difference between Nifty spot and Max Pain strike
        # Ensure Nifty spot is not zero before dividing
        max_pain_diff_pct = abs(nifty_spot - max_pain_strike) / nifty_spot * 100.0 if nifty_spot != 0 else 0.0

        logger.debug(f"Max Pain calculated: Strike={max_pain_strike}, Diff%={max_pain_diff_pct:.2f}")
        return max_pain_strike, max_pain_diff_pct

    except Exception as e:
        logger.error(f"Error calculating max pain: {str(e)}", exc_info=True)
        # Log traceback for debugging unexpected errors
        # import traceback # Already imported at the top
        # logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None


# Fetch portfolio data
def fetch_all_api_portfolio_data(upstox_client: dict):
    """
    Fetches all relevant user portfolio data (holdings, margin, positions,
    order book, trade history) using the Upstox API SDK.
    Handles potential API errors for each endpoint individually.

    Args:
        upstox_client (dict): Dictionary of initialized Upstox API client objects.
                            Requires keys 'portfolio_api', 'user_api', 'order_api'.

    Returns:
        dict: A dictionary containing all fetched portfolio data, keyed by type
              (e.g., 'holdings', 'margin'). Returns an empty dict on overall error.
              Individual keys might contain empty dicts or lists if specific fetches fail,
              with logged errors.
    """
    logger.info("Starting fetch of all portfolio data.")
    # Ensure the client dictionary and necessary API instances are available
    if not upstox_client or 'portfolio_api' not in upstox_client or 'user_api' not in upstox_client or 'order_api' not in upstox_client:
        logger.error("Upstox client or required API instances not available for fetching portfolio data.")
        return {}

    # Get API instances from the client dictionary
    portfolio_api = upstox_client['portfolio_api']
    user_api = upstox_client['user_api']
    order_api = upstox_client['order_api']

    portfolio_data = {} # Dictionary to store all fetched data

    # Fetch each type of data, wrapping each call in a try-except for resilience.
    # This ensures if one fetch fails, others can still succeed.
    try:
        logger.info("Fetching holdings.")
        holdings_response = portfolio_api.get_holdings(api_version="v2")
        # Convert SDK response object to dictionary, default to empty dict on None response
        portfolio_data["holdings"] = holdings_response.to_dict() if holdings_response else {}
        logger.info("Holdings fetch completed.")
    except ApiException as e:
        logger.error(f"Upstox API exception fetching holdings: Status={e.status}, Body={e.body}")
        portfolio_data["holdings"] = {} # Ensure key exists even on API error
    except requests.exceptions.RequestException as req_e:
         logger.error(f"Network error fetching holdings: {req_e}")
         portfolio_data["holdings"] = {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching holdings: {str(e)}", exc_info=True)
        portfolio_data["holdings"] = {}

    # Introduce small delay between API calls
    time.sleep(0.5)

    try:
        logger.info("Fetching funds/margin.")
        margin_response = user_api.get_user_fund_margin(api_version="v2")
        portfolio_data["margin"] = margin_response.to_dict() if margin_response else {}
        logger.info("Funds/margin fetch completed.")
    except ApiException as e:
        logger.error(f"Upstox API exception fetching funds/margin: Status={e.status}, Body={e.body}")
        # Specific warning about access hours for funds might be relevant here
        logger.warning("Funds/margin service is typically accessible from 5:30 AM to 12:00 AM IST.")
        portfolio_data["margin"] = {}
    except requests.exceptions.RequestException as req_e:
         logger.error(f"Network error fetching funds/margin: {req_e}")
         portfolio_data["margin"] = {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching funds/margin: {str(e)}", exc_info=True)
        portfolio_data["margin"] = {}

    time.sleep(0.5)

    try:
        logger.info("Fetching positions.")
        positions_response = portfolio_api.get_positions(api_version="v2")
        portfolio_data["positions"] = positions_response.to_dict() if positions_response else {}
        logger.info("Positions fetch completed.")
    except ApiException as e:
        logger.error(f"Upstox API exception fetching positions: Status={e.status}, Body={e.body}")
        portfolio_data["positions"] = {}
    except requests.exceptions.RequestException as req_e:
         logger.error(f"Network error fetching positions: {req_e}")
         portfolio_data["positions"] = {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching positions: {str(e)}", exc_info=True)
        portfolio_data["positions"] = {}

    time.sleep(0.5)

    try:
        logger.info("Fetching order book.")
        order_book_response = order_api.get_order_book(api_version="v2")
        portfolio_data["order_book"] = order_book_response.to_dict() if order_book_response else {}
        logger.info("Order book fetch completed.")
    except ApiException as e:
        logger.error(f"Upstox API exception fetching order book: Status={e.status}, Body={e.body}")
        portfolio_data["order_book"] = {}
    except requests.exceptions.RequestException as req_e:
         logger.error(f"Network error fetching order book: {req_e}")
         portfolio_data["order_book"] = {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching order book: {str(e)}", exc_info=True)
        portfolio_data["order_book"] = {}

    time.sleep(0.5)

    try:
        logger.info("Fetching trade history.")
        trade_history_response = order_api.get_trade_history(api_version="v2")
        portfolio_data["trade_book"] = trade_history_response.to_dict() if trade_history_response else {} # Using key 'trade_book' as in streamlit_app
        logger.info("Trade history fetch completed.")
    except ApiException as e:
        logger.error(f"Upstox API exception fetching trade history: Status={e.status}, Body={e.body}")
        portfolio_data["trade_book"] = {}
    except requests.exceptions.RequestException as req_e:
         logger.error(f"Network error fetching trade history: {req_e}")
         portfolio_data["trade_book"] = {}
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching trade history: {str(e)}", exc_info=True)
        portfolio_data["trade_book"] = {}

    logger.info("All portfolio data fetch attempts completed.")
    return portfolio_data

# Fetch market depth by scrip (instrument key)
def fetch_market_depth_by_scrip(upstox_client: dict, instrument_key: str):
    """
    Fetches market depth (bid/ask quantities) and latest price for a specific
    instrument key using the Upstox API via the requests library.
    This matches the implementation style in the original code for this endpoint.

    Args:
        upstox_client (dict): Dictionary containing the access token.
                            Requires key 'access_token'.
        instrument_key (str): The specific instrument key for the option or stock
                              (e.g., "NSE_FO|46204" for a Nifty option, "NSE_EQ|INE009A01021" for Reliance).

    Returns:
        dict or None: A dictionary containing 'LastTradedPrice', 'BidVolume',
                      'AskVolume' within a 'Data' list (to match an expected format elsewhere),
                      or None on error.
                      Example successful return structure:
                      {"Data": [{"LastTradedPrice": 150.50, "BidVolume": 2500, "AskVolume": 1800}]}
    """
    # Ensure necessary information is provided
    if not upstox_client or not upstox_client.get("access_token") or not instrument_key or not isinstance(instrument_key, str):
        logger.warning("Client dict, access token, or instrument key missing/invalid for market depth fetch.")
        return None

    logger.info(f"Fetching market depth for instrument key: {instrument_key} using requests.")
    try:
        # Use access token from the client dict for authorization header
        headers = {"Authorization": f"Bearer {upstox_client.get('access_token')}", "Content-Type": "application/json"}
        url = f"{base_url}/market-quote/depth" # Market depth endpoint
        params = {"instrument_key": instrument_key} # Query parameters

        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status() # Raise HTTPError for bad responses

        # Parse the JSON response structure
        # Expected structure: { 'data': { 'INSTRUMENT_KEY': { 'depth': { 'buy': [...], 'sell': [...] }, 'last_price': ... } } }
        data_response = res.json().get('data', {})
        instrument_data = data_response.get(instrument_key, {})
        depth_details = instrument_data.get('depth', {})
        buy_depth = depth_details.get('buy', [])
        sell_depth = depth_details.get('sell', [])

        # Sum quantities from the buy and sell depth lists
        # Ensure items are dictionaries and get the 'quantity', default to 0 if missing/None
        bid_volume = sum(item.get('quantity', 0) or 0 for item in buy_depth if isinstance(item, dict))
        ask_volume = sum(item.get('quantity', 0) or 0 for item in sell_depth if isinstance(item, dict))

        # Get the last traded price
        ltp = instrument_data.get('last_price')
        if ltp is not None:
             ltp = pd.to_numeric(ltp, errors='coerce').mean() # Convert LTP to float if available
             if pd.isna(ltp): ltp = 0.0 # Default to 0.0 if conversion resulted in NaN
        else:
             ltp = 0.0 # Default to 0.0 if LTP is not provided


        logger.info(f"Market Depth for {instrument_key}: Bid Volume={bid_volume}, Ask Volume={ask_volume}, LTP={ltp}")
        time.sleep(0.5)  # Basic Rate limiting

        # Return the data in a dictionary format expected by the caller (e.g., streamlit_app's PnL calculation)
        # The 'Data' key containing a list with one item seems to be a convention used elsewhere.
        return {"Data": [{"LastTradedPrice": ltp, "BidVolume": bid_volume, "AskVolume": ask_volume}]}

    except requests.exceptions.RequestException as req_e:
         logger.error(f"Network error during market depth fetch for {instrument_key}: {req_e}")
         return None # Indicate failure

    except Exception as e:
        logger.error(f"An unexpected error occurred fetching market depth for {instrument_key}: {str(e)}", exc_info=True)
        return None # Indicate failure


# Prepare trade orders based on strategy
def prepare_trade_orders(strategy: dict, real_data: dict, capital: float):
    """
    Analyzes the recommended strategy and prepares a list of corresponding
    order dictionaries ready for execution via the Upstox API.
    Includes logic for specific option strategies (Straddle, Strangle, Condor).

    Args:
        strategy (dict): Dictionary containing the generated strategy details,
                         e.g., {"Strategy": "Short Straddle", "Deploy": 50000}.
                         Requires 'Strategy' and 'Deploy' keys.
        real_data (dict): Dictionary containing the latest real-time market data,
                          as returned by fetch_real_time_market_data.
                          Requires 'option_chain', 'atm_strike', 'expiry', 'straddle_price' keys.
        capital (float): The user's total trading capital. Used for context,
                         but position sizing is based on 'Deploy' amount here.

    Returns:
        list or None: A list of dictionaries, where each dict represents a single
                      order leg to be placed. Returns None if preparation fails
                      (e.g., missing data, invalid strategy, strikes not found).
        Example order dictionary structure:
        {
            "Strategy": "Short Straddle", "Leg_Type": "S CE", "Strike": 21000,
            "Expiry": "YYYY-MM-DD", "Exchange": "NSE", "ExchangeType": "D",
            "ScripCode": "NSE_FO|...", "Quantity_Lots": 2, "Quantity_Units": 100,
            "Transaction_Type": "SELL", "Order_Type": "MARKET", "Product": "I",
            "Proposed_Price": 0, "Last_Price_API": 150.50,
            "Stop_Loss_Price": 165.55, "Take_Profit_Price": 135.45
        }
    """
    logger.info(f"Starting order preparation for strategy: {strategy.get('Strategy', 'Unknown')}.")

    # --- 1. Validate and extract necessary data from inputs ---
    if not strategy or not isinstance(strategy, dict) or not strategy.get("Strategy"):
        logger.error("Invalid or missing strategy data provided.")
        return None
    # Ensure 'Deploy' amount is a valid number and greater than 0
    deploy_amount = pd.to_numeric(strategy.get("Deploy"), errors='coerce').mean() # Use .mean() with coerce
    if pd.isna(deploy_amount) or deploy_amount <= 0:
         logger.error(f"Invalid or zero deploy amount specified in strategy: {strategy.get('Deploy')}. Cannot prepare orders.")
         return None

    if not real_data or not isinstance(real_data, dict) or "option_chain" not in real_data or not isinstance(real_data["option_chain"], pd.DataFrame) or real_data["option_chain"].empty:
        logger.error("Invalid, incomplete, or empty real-time market data provided for order preparation.")
        return None

    # Extract key market data from real_data
    option_chain_df = real_data["option_chain"] # DataFrame from fetch_real_time_market_data
    atm_strike = pd.to_numeric(real_data.get("atm_strike"), errors='coerce').mean() # Ensure numeric
    expiry_date_str = real_data.get("expiry") # YYYY-MM-DD string
    straddle_price_live = pd.to_numeric(real_data.get("straddle_price"), errors='coerce').mean() # Ensure numeric

    # Validate essential extracted market data
    if pd.isna(atm_strike) or expiry_date_str is None or not isinstance(expiry_date_str, str) or option_chain_df.empty:
         logger.error("Missing essential real-time market data (ATM strike, expiry, or option chain) for order preparation.")
         return None

    # Use a default straddle price if the live one is invalid/missing (needed for sizing estimation)
    if pd.isna(straddle_price_live) or straddle_price_live <= 0:
         logger.warning(f"Live straddle price is invalid or zero ({straddle_price_live}). Using default 200.0 for sizing estimation.")
         straddle_price_live = 200.0 # Default value if live price is bad

    # --- 2. Define lot size and calculate number of lots ---
    # Note: Lot size is hardcoded for NIFTY 50.
    # This should ideally be fetched dynamically based on the instrument ('NSE_INDEX|Nifty 50')
    # or configured externally for use with different instruments.
    lot_size = 50 # Current NIFTY 50 lot size (as of April 2024)
    logger.info(f"Using NIFTY 50 lot size: {lot_size}")

    # Estimate premium per lot to determine how many lots the deploy amount can cover.
    # A very rough estimate is needed here. Using Straddle price / 2 as a proxy for premium per point,
    # then multiplying by lot size. This is just for sizing estimation, not the actual premium paid/received.
    estimated_premium_per_point = straddle_price_live / 2.0 # Rough estimate per point
    estimated_premium_per_lot = estimated_premium_per_point * lot_size

    lots = 1 # Start with minimum 1 lot
    if estimated_premium_per_lot > 0:
        calculated_lots = int(deploy_amount / estimated_premium_per_lot)
        # Cap the number of lots to a sensible range (e.g., min 1, max 20)
        lots = max(1, min(calculated_lots, 20))
    else:
        logger.warning(f"Estimated premium per lot is zero or negative ({estimated_premium_per_lot}). Defaulting to 1 lot.")
        lots = 1 # Default to 1 lot if premium estimate is bad

    logger.info(f"Calculated number of lots to deploy: {lots}")

    orders_to_place = [] # List to store the prepared order dictionaries
    strategy_legs = [] # List to define the strike, type (CE/PE), and action (Buy/Sell) for each leg

    # --- 3. Define strategy legs based on the strategy name ---
    # This is the core logic that translates a strategy name into specific option contracts.
    strategy_name = strategy.get("Strategy")

    if strategy_name == "Short Straddle":
        logger.info(f"Defining legs for Short Straddle at ATM strike {atm_strike}.")
        # A Short Straddle involves selling the ATM Call and selling the ATM Put of the same expiry.
        strategy_legs = [(atm_strike, "CE", "S"), (atm_strike, "PE", "S")] # (Strike, CPType, Action)

    elif strategy_name == "Short Strangle":
        logger.info(f"Defining legs for Short Strangle around ATM strike {atm_strike}.")
        # A Short Strangle involves selling an OTM Call and selling an OTM Put of the same expiry.
        # The strike selection logic here is an example; a real strategy would have more specific rules.
        # Ensure 'StrikeRate' column is numeric before sorting
        if "StrikeRate" not in option_chain_df.columns or not pd.api.types.is_numeric_dtype(option_chain_df["StrikeRate"]):
            logger.error("Option chain DataFrame missing 'StrikeRate' or it's not numeric. Cannot find strikes for Strangle.")
            return None

        strikes_sorted = option_chain_df["StrikeRate"].sort_values().tolist()

        # Find an OTM Call strike (e.g., first available strike >= ATM + 100 points)
        call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + 100), None)
        # Find an OTM Put strike (e.g., first available strike <= ATM - 100 points)
        put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - 100), None)

        if call_sell_strike is not None and put_sell_strike is not None:
            logger.info(f"Short Strangle Strikes: Call {call_sell_strike}, Put {put_sell_strike}.")
            strategy_legs = [(call_sell_strike, "CE", "S"), (put_sell_strike, "PE", "S")] # Sell OTM Call, Sell OTM Put
        else:
            logger.error(f"Could not find suitable strikes for Short Strangle around {atm_strike}. Call strike found: {call_sell_strike}, Put strike found: {put_sell_strike}.")
            return None # Indicate failure if suitable strikes are not found

    elif strategy_name == "Iron Condor":
        logger.info(f"Defining legs for Iron Condor around ATM strike {atm_strike}.")
        # An Iron Condor is a 4-leg strategy: Buy OTM Put (Hedge), Sell OTM Put, Sell OTM Call, Buy OTM Call (Hedge).
        # Strike selection logic here is an example; a real strategy would have more specific rules.
        # Ensure 'StrikeRate' column is numeric before sorting
        if "StrikeRate" not in option_chain_df.columns or not pd.api.types.is_numeric_dtype(option_chain_df["StrikeRate"]):
            logger.error("Option chain DataFrame missing 'StrikeRate' or it's not numeric. Cannot find strikes for Iron Condor.")
            return None

        strikes_sorted = option_chain_df["StrikeRate"].sort_values().tolist()

        # Define strike separation (example: 100 points between short legs, 200 points spread for wings)
        short_leg_distance = 100 # Distance from ATM for short Put/Call
        spread_width = 200 # Distance between short and long leg strikes

        # Find Short Put strike (OTM)
        put_sell_strike = next((s for s in reversed(strikes_sorted) if s <= atm_strike - short_leg_distance), None)
        # Find Long Put strike (further OTM, for hedge)
        put_buy_strike = next((s for s in reversed(strikes_sorted) if s < (put_sell_strike if put_sell_strike is not None else atm_strike) - spread_width), None)

        # Find Short Call strike (OTM)
        call_sell_strike = next((s for s in strikes_sorted if s >= atm_strike + short_leg_distance), None)
        # Find Long Call strike (further OTM, for hedge)
        call_buy_strike = next((s for s in strikes_sorted if s > (call_sell_strike if call_sell_strike is not None else atm_strike) + spread_width), None)

        # Check if all four strikes were successfully found
        if put_sell_strike is not None and put_buy_strike is not None and call_sell_strike is not None and call_buy_strike is not None:
            logger.info(f"Iron Condor Strikes: Put Buy {put_buy_strike}, Put Sell {put_sell_strike}, Call Sell {call_sell_strike}, Call Buy {call_buy_strike}.")
            strategy_legs = [
                (put_buy_strike, "PE", "B"), # Buy OTM Put (Hedge leg)
                (put_sell_strike, "PE", "S"), # Sell OTM Put (Short leg)
                (call_sell_strike, "CE", "S"), # Sell OTM Call (Short leg)
                (call_buy_strike, "CE", "B")  # Buy OTM Call (Hedge leg)
            ]
        else:
            logger.error(f"Could not find all four suitable strikes for Iron Condor around {atm_strike}.")
            logger.debug(f"Strike search results: Put Buy {put_buy_strike}, Put Sell {put_sell_strike}, Call Sell {call_sell_strike}, Call Buy {call_buy_strike}")
            return None # Indicate failure if strikes not found

    # Add definitions for other strategies here if needed...
    # elif strategy_name == "Butterfly Spread":
    #    ...
    # elif strategy_name == "Short Put Vertical Spread":
    #    ...


    else:
        # If the strategy name is not one of the recognized ones
        logger.warning(f"Strategy '{strategy_name}' is not recognized or supported for automatic order preparation.")
        return None # Indicate that order preparation failed for this strategy


    # --- 4. Create order dictionaries for each defined leg ---
    # Iterate through the list of strategy legs defined above
    for leg in strategy_legs:
        strike, cp_type, buy_sell_action = leg # Unpack the tuple
        logger.debug(f"Preparing order details for leg: {buy_sell_action} {cp_type} {strike}.")

        # Find the specific row in the option chain DataFrame that matches this leg's strike and type
        # Ensure StrikeRate is numeric before comparison
        strike_numeric = pd.to_numeric(strike, errors='coerce')
        if pd.isna(strike_numeric):
            logger.error(f"Invalid strike value '{strike}' for leg {buy_sell_action} {cp_type}. Cannot prepare order.")
            return None # Fail if strike is not numeric

        opt_data_row_df = option_chain_df[
            (option_chain_df["StrikeRate"] == strike_numeric) &
            (option_chain_df["CPType"] == cp_type)
        ]

        if opt_data_row_df.empty:
            # This should ideally not happen if strike finding logic was correct, but added for safety
            logger.error(f"Option chain data not found for required leg: {buy_sell_action} {cp_type} at strike {strike} for expiry {expiry_date_str}. Cannot prepare order.")
            # If a leg's data is missing, the whole strategy preparation fails
            return None # Indicate failure

        # Ensure we get the first row in case of duplicates (shouldn't happen if data is clean)
        opt_data_row = opt_data_row_df.iloc[0]

        # Extract instrument key (ScripCode) and latest price for this specific option contract
        # Ensure keys exist and handle potential missing values
        scrip_code = opt_data_row.get("ScripCode")
        # Get the latest price (LTP), handle potential NaN or missing key
        latest_price = pd.to_numeric(opt_data_row.get("LastRate"), errors='coerce').mean() # Use .mean() with coerce
        latest_price = float(latest_price if pd.notna(latest_price) else 0.0) # Ensure float, default to 0.0


        # Validate ScripCode - it's critical for placing the order
        if not isinstance(scrip_code, str) or not scrip_code:
             logger.error(f"Instrument key (ScripCode) is missing or invalid for leg: {buy_sell_action} {cp_type} at strike {strike}. Cannot prepare order.")
             return None # Fail if ScripCode is missing

        # Determine the transaction type ("BUY" or "SELL") based on the leg's action
        transaction_type = "BUY" if buy_sell_action == "B" else "SELL"

        # --- Define Order Parameters for the API Call ---
        # These parameters match what the Upstox place_order API expects.
        # Note: Order Type, Product, and Validity are hardcoded here for simplicity,
        # matching the likely intent of the original code.
        order_type = "MARKET" # Place order at the best available market price
        # For MARKET orders, the price parameter is typically ignored or set to 0.
        proposed_price = 0 # Set to 0 for MARKET orders

        # Product type: 'I' for Intraday, 'D' for Delivery (equity), 'M' for Margin (equity).
        # For F&O, Intraday ('I') is common, but it depends on the user's account settings
        # and whether they intend to carry forward the position (use 'D' for carryforward F&O).
        product = "I" # Hardcoded as Intraday

        # Validity: 'DAY' (Valid until end of trading day), 'IOC' (Immediate Or Cancel).
        validity = "DAY" # Hardcoded as Day validity

        # Stop Loss and Take Profit prices - these are calculated here,
        # but the current `execute_trade_orders` function places simple MARKET orders
        # and does NOT automatically add SL/TP orders.
        # Implementing SL/TP requires additional logic (e.g., placing OCO orders if supported
        # by the API, or placing separate SL/TP limit/trigger orders after the main order fills).
        # The calculated prices are stored in the order dictionary for informational purposes
        # or potential future use if advanced order types are implemented.
        # Example SL/TP logic (adjust as needed based on strategy's risk management rules):
        # If Buying, SL is below entry, TP is above entry.
        # If Selling, SL is above entry, TP is below entry.
        # Using a simple percentage-based calculation as an example:
        sl_percentage = 0.10 # 10% SL
        tp_percentage = 0.20 # 20% TP

        stop_loss_price = None # Initialize SL/TP to None
        take_profit_price = None

        if latest_price > 0: # Calculate SL/TP only if latest price is valid and non-zero
            if transaction_type == "BUY":
                stop_loss_price = latest_price * (1 - sl_percentage)
                take_profit_price = latest_price * (1 + tp_percentage)
            elif transaction_type == "SELL":
                stop_loss_price = latest_price * (1 + sl_percentage)
                take_profit_price = latest_price * (1 - tp_percentage)

            # Optional: Round SL/TP prices to appropriate decimal places for the instrument
            # This is important for some exchanges/brokers. Upstox API might handle this.
            # stop_loss_price = round(stop_loss_price, 2) if stop_loss_price is not None else None
            # take_profit_price = round(take_profit_price, 2) if take_profit_price is not None else None


        # Add the prepared order details as a dictionary to the list
        orders_to_place.append({
            "Strategy": strategy_name, # Name of the strategy
            "Leg_Type": f"{buy_sell_action} {cp_type}", # e.g., "S CE", "B PE"
            "Strike": strike, # Strike price of the option leg
            "Expiry": expiry_date_str, # Expiry date of the option
            "Exchange": "NSE", # Hardcoded Exchange (National Stock Exchange)
            "ExchangeType": "D", # Hardcoded Exchange Type (Derivatives)
            "ScripCode": scrip_code, # The unique instrument key for the API call
            "Quantity_Lots": lots, # Number of lots to trade for this leg
            "Quantity_Units": lots * lot_size, # Total quantity (lots * lot size)
            "Transaction_Type": transaction_type, # "BUY" or "SELL"
            "Order_Type": order_type, # e.g., "MARKET", "LIMIT"
            "Product": product, # e.g., "I" (Intraday), "D" (Delivery/Carryforward F&O)
            "Validity": validity, # e.g., "DAY", "IOC"
            "Proposed_Price": proposed_price, # Price for the order (0 for MARKET)
            "Last_Price_API": latest_price, # The latest price fetched for this leg (for info)
            "Stop_Loss_Price": stop_loss_price, # Calculated SL price (for info/future use)
            "Take_Profit_Price": take_profit_price # Calculated TP price (for info/future use)
            # You can add other order parameters here if required by your strategy
            # or supported by the Upstox API (e.g., disclose_quantity, trigger_price)
        })

    # Check if any orders were successfully prepared
    if not orders_to_place:
         logger.error("Order preparation failed: No orders were added to the list.")
         return None # Indicate failure if no orders were prepared

    logger.info(f"Order preparation completed. Prepared {len(orders_to_place)} orders.")
    # Return the list of prepared order dictionaries
    return orders_to_place

# Execute trade orders
def execute_trade_orders(upstox_client: dict, prepared_orders: list):
    """
    Executes a list of prepared order dictionaries by placing them with the
    Upstox API using the SDK. Checks market hours before placing orders.

    Args:
        upstox_client (dict): Dictionary of initialized Upstox API client objects.
                            Requires key 'order_api'.
        prepared_orders (list): A list of order dictionaries prepared by
                                prepare_trade_orders.

    Returns:
        tuple: (overall_success (bool), response_details (dict)).
               overall_success is True if the process ran without critical errors,
               False if market is closed, client is invalid, or any order placement failed.
               response_details is a dictionary containing a list of results for each order placement.
               Example response_details: {"responses": [{"Order": {...}, "Response": {...}}, ...]}
    """
    logger.info(f"Attempting to execute {len(prepared_orders) if prepared_orders else 0} orders.")

    # --- 1. Validate inputs ---
    if not upstox_client or 'order_api' not in upstox_client:
        logger.error("Upstox client or OrderApi instance not available for order execution.")
        # Return failure state with an error message
        return False, {"error": "Invalid client session or missing OrderApi instance."}
    if not prepared_orders or not isinstance(prepared_orders, list):
        logger.warning("No valid orders provided for execution.")
        # Return failure state as no orders could be processed
        return False, {"error": "No orders provided for execution."}

    # Get the OrderApi instance from the client dictionary
    order_api = upstox_client['order_api']

    # --- 2. Check market status before executing live orders ---
    # This is a crucial step for live trading to prevent errors outside market hours.
    # Note: This is a simple check for standard Indian equity derivatives trading hours.
    # A more robust check would account for exchange holidays or special trading sessions
    # by potentially fetching exchange status via the API if available.
    now = datetime.now()
    # Standard hours: Monday to Friday, 9:15 AM to 3:30 PM IST (15:30)
    market_open_time = datetime_time(9, 15)
    market_close_time = datetime_time(15, 30)
    is_weekday = now.weekday() < 5 # Monday (0) to Friday (4)

    if not is_weekday or not (market_open_time <= now.time() <= market_close_time):
        logger.error("Market is closed. Orders can only be executed between 9:15 AM and 3:30 PM IST on weekdays.")
        # Return failure state with a specific error message
        return False, {"error": "Market is closed. Please try placing orders during trading hours (9:15 AM - 3:30 PM IST, Mon-Fri)."}


    overall_successful = True # Flag to track if ALL orders were placed successfully
    responses_list = [] # List to store the API response details for each order placement attempt

    # --- 3. Iterate through the prepared orders and place each one via API ---
    for i, order in enumerate(prepared_orders):
        # Log details of the order being attempted
        log_message = f"Placing Order {i+1}/{len(prepared_orders)}: {order.get('Leg_Type', 'N/A')} {order.get('Strike', 'N/A')} ({order.get('Quantity_Units', 'N/A')} units) for Strategy '{order.get('Strategy', 'N/A')}'."
        logger.info(log_message)
        logger.debug(f"Order details: {order}") # Log full order details for debugging

        try:
            # --- Prepare the order body dictionary for the API call ---
            # Use .get() with default values to safely access order details
            order_body = {
                "instrument_key": order.get("ScripCode"), # The unique identifier for the contract
                "quantity": order.get("Quantity_Units"), # Total number of units (lots * lot_size)
                "order_type": order.get("Order_Type", "MARKET"), # e.g., "MARKET", "LIMIT" - default to MARKET
                "transaction_type": order.get("Transaction_Type"), # "BUY" or "SELL"
                "product": order.get("Product", "I"), # e.g., "I" (Intraday) - default to Intraday
                "price": order.get("Proposed_Price", 0), # Price for LIMIT order, typically 0 for MARKET
                "validity": order.get("Validity", "DAY"), # e.g., "DAY", "IOC" - default to DAY
                # Add other optional parameters from the prepared order if needed and supported by API
                # "trigger_price": order.get("Trigger_Price"), # For SL orders
                # "disclosed_quantity": order.get("Disclosed_Quantity")
            }

            # Validate essential fields in the order_body before sending
            if not all([order_body.get("instrument_key"), order_body.get("quantity") is not None, order_body.get("order_type"), order_body.get("transaction_type"), order_body.get("product"), order_body.get("validity")]):
                 logger.error(f"Essential data missing or invalid in order body for order {i+1}. Skipping. Body: {order_body}")
                 overall_successful = False # Mark overall execution as failed
                 # Record this specific order's failure
                 responses_list.append({"Order": order, "Response": {"status": "failed", "message": "Missing essential order parameters."}})
                 continue # Skip to the next order

            # Basic validation for ScripCode format (should contain '|')
            if not isinstance(order_body["instrument_key"], str) or '|' not in order_body["instrument_key"]:
                 logger.error(f"Invalid instrument_key format for order {i+1}: {order_body['instrument_key']}. Skipping.")
                 overall_successful = False # Mark overall execution as failed
                 # Record this specific order's failure
                 responses_list.append({"Order": order, "Response": {"status": "failed", "message": "Invalid instrument key format."}})
                 continue # Skip to the next order in the list


            # --- Make the API call to place the order ---
            api_response_object = order_api.place_order(body=order_body, api_version="v2")

            # Convert the SDK response object to a dictionary for easier handling and storage
            response_dict = api_response_object.to_dict() if api_response_object else {"status": "unknown", "message": "No response object returned from API."}

            # Store the original prepared order details along with the API response details
            responses_list.append({"Order": order, "Response": response_dict})

            # Check the status provided in the API response
            if response_dict.get("status") != "success":
                # If the API reported a failure for this order, mark the overall process as failed
                overall_successful = False
                api_error_message = response_dict.get("message", "Unknown API error message.")
                logger.error(f"Order placement failed for order {i+1} ({order.get('ScripCode')}): {api_error_message}")
            else:
                # Log success with the returned order ID if available
                order_id = response_dict.get('data', {}).get('order_id', 'N/A')
                logger.info(f"Order {i+1} placed successfully. Upstox Order ID: {order_id}")

        except ApiException as api_e:
            # Catch specific Upstox API exceptions during the place_order call
            overall_successful = False # Mark overall execution as failed
            logger.error(f"Upstox API exception placing order {i+1} ({order.get('ScripCode')}): Status={api_e.status}, Body={api_e.body}")
            # Record the failure details, including API status and body
            responses_list.append({"Order": order, "Response": {"status": "api_exception", "message": f"API Exception: Status {api_e.status}, Body: {api_e.body}"}})

        except requests.exceptions.RequestException as req_e:
             # Catch network errors (might happen if SDK uses requests internally)
             overall_successful = False
             logger.error(f"Network error placing order {i+1} ({order.get('ScripCode')}): {req_e}")
             responses_list.append({"Order": order, "Response": {"status": "network_error", "message": f"Network Error: {req_e}"}})

        except Exception as e:
            # Catch any other unexpected errors during the processing of this order
            overall_successful = False # Mark overall execution as failed
            logger.error(f"An unexpected error occurred placing order {i+1} ({order.get('ScripCode')}): {str(e)}", exc_info=True)
            # Record the failure details
            responses_list.append({"Order": order, "Response": {"status": "exception", "message": f"Exception: {str(e)}"}}.)

        # Introduce a small delay between placing each order to respect rate limits.
        # The Upstox API has rate limits per minute per endpoint.
        time.sleep(0.5) # 500 milliseconds delay


    logger.info("Order execution process completed.")
    # Return the overall success status and the list containing responses for each order.
    return overall_successful, {"responses": responses_list}

# Square off all open positions
def square_off_positions(upstox_client: dict):
    """
    Fetches all open positions from the portfolio and places market orders
    to square them off using the Upstox API SDK. Checks market hours.

    Args:
        upstox_client (dict): Dictionary of initialized Upstox API client objects.
                            Requires keys 'portfolio_api' and 'order_api'.

    Returns:
        bool: True if the process of fetching positions and attempting to place
              square-off orders completed without critical errors. False otherwise.
              Does NOT guarantee that all square-off orders were successfully filled
              by the exchange, only that they were submitted to the API without error.
    """
    logger.info("Attempting to square off all open positions.")

    # --- 1. Validate inputs ---
    if not upstox_client or 'portfolio_api' not in upstox_client or 'order_api' not in upstox_client:
        logger.error("Upstox client or required API instances not available for squaring off positions.")
        return False # Indicate failure

    # Get API instances from the client dictionary
    portfolio_api = upstox_client['portfolio_api']
    order_api = upstox_client['order_api']

    # --- 2. Check market status before attempting to square off live positions ---
    now = datetime.now()
    market_open_time = datetime_time(9, 15)
    market_close_time = datetime_time(15, 30)
    is_weekday = now.weekday() < 5 # Monday (0) to Friday (4)

    if not is_weekday or not (market_open_time <= now.time() <= market_close_time):
        logger.error("Market is closed. Cannot square off live positions.")
        # Indicate failure because square off is a trading action needing live market hours
        return False

    overall_success = True # Flag to track if the overall process ran without errors

    try:
        # --- 3. Fetch current open positions from the portfolio ---
        logger.info("Fetching current open positions.")
        positions_response = portfolio_api.get_positions(api_version="v2")

        # Convert response object to dictionary and get the list of positions
        positions_list = positions_response.to_dict().get("data", [])

        if not positions_list or not isinstance(positions_list, list):
            logger.info("No open positions found to square off.")
            # Consider this a successful completion if there were no positions to begin with
            return True

        logger.info(f"Found {len(positions_list)} open positions to attempt to square off.")

        # --- 4. Iterate through positions and place square-off orders ---
        for i, pos in enumerate(positions_list):
            if not isinstance(pos, dict):
                logger.warning(f"Skipping position data in unexpected format at index {i}: {pos}")
                # Skip malformed position entries but continue with others
                overall_success = False # Mark overall process as having encountered issues
                continue # Move to the next item in the list

            # Get essential details for squaring off this position
            instrument_key = pos.get("instrument_key")
            quantity = pos.get("quantity", 0) # Positive for a long position, negative for a short position
            product_type = pos.get("product") # Get the product type of the original position

            # Only attempt to square off positions with a non-zero quantity and valid instrument key
            if instrument_key and isinstance(instrument_key, str) and quantity != 0:
                # Determine the necessary transaction type to close the position
                # If quantity > 0 (long), need to SELL to square off.
                # If quantity < 0 (short), need to BUY to square off.
                transaction_type_to_close = "SELL" if quantity > 0 else "BUY"
                abs_quantity = abs(quantity) # Use the absolute quantity for the order

                log_msg = f"Squaring off position {i+1}/{len(positions_list)} for {instrument_key} ({quantity} units). "
                log_msg += f"Attempting to place {transaction_type_to_close} order for {abs_quantity} units."
                logger.info(log_msg)

                try:
                    # --- Prepare the order body for the square-off order ---
                    order_body = {
                        "instrument_key": instrument_key, # The contract to close
                        "quantity": abs_quantity, # Absolute quantity
                        "order_type": "MARKET", # Square off at market price
                        "transaction_type": transaction_type_to_close, # BUY or SELL to close
                        # Attempt to use the original position's product type (I, D, M)
                        # Default to 'I' (Intraday) if product type is missing in the position data or invalid.
                        "product": product_type if product_type in ["I", "D", "M"] else "I",
                        "price": 0, # Price is 0 for MARKET orders
                        "validity": "DAY" # Order valid for the day
                        # Add other optional parameters if needed
                    }
                    logger.debug(f"Square-off order body: {order_body}")

                    # --- Make the API call to place the square-off order ---
                    response = order_api.place_order(body=order_body, api_version="v2")

                    # Check the status of the order placement response
                    if response.status != "success":
                        # If placing this specific square-off order failed, mark the overall process as having issues
                        overall_success = False
                        error_message = response.message if response and response.message else "Unknown API error message."
                        logger.error(f"Failed to place square-off order for {instrument_key}: {error_message}")
                    else:
                        # Log success with the returned order ID if available
                        order_id = response.data.get('order_id', 'N/A') if response and response.data else 'N/A'
                        logger.info(f"Square-off order placed successfully for {instrument_key}. Upstox Order ID: {order_id}")

                except ApiException as api_e:
                    logger.error(f"Upstox API exception squaring off {instrument_key}: Status={api_e.status}, Body={api_e.body}")
                    overall_success = False # Mark overall process as having issues

                except requests.exceptions.RequestException as req_e:
                     # Catch network errors
                     overall_success = False
                     logger.error(f"Network error squaring off {instrument_key}: {req_e}")

                except Exception as e:
                    # Catch any other unexpected errors during the processing of this position
                    overall_success = False # Mark overall process as having issues
                    logger.error(f"An unexpected error occurred placing square-off order for {instrument_key}: {str(e)}", exc_info=True)

                # Introduce a small delay between placing each square-off order
                time.sleep(0.5) # 500 milliseconds delay

            else:
                # Log if a position was skipped because its quantity was zero or missing info
                 logger.debug(f"Skipping position {i+1}/{len(positions_list)} due to zero quantity or missing/invalid instrument key: {pos}")


    except ApiException as api_e:
        # Catch API exceptions during the initial fetch of positions
        logger.error(f"Upstox API exception fetching positions during square off: Status={api_e.status}, Body={api_e.body}")
        overall_success = False

    except requests.exceptions.RequestException as req_e:
         # Catch network errors fetching positions
         logger.error(f"Network error fetching positions during square off: {req_e}")
         overall_success = False

    except Exception as e:
        # Catch any other unexpected errors during the initial fetch of positions or loop setup
        logger.error(f"An unexpected error occurred during the square off process: {str(e)}", exc_info=True)
        overall_success = False
        # Log traceback for better debugging
        # import traceback # Already imported at the top
        # logger.error(f"Traceback: {traceback.format_exc()}")


    logger.info("Square off process completed.")
    # Return the overall success status. True means the process ran without major errors,
    # but doesn't guarantee all orders were successfully placed or filled.
    return overall_success

# The __main__ block is removed as this module's functions are intended to be imported
# and used by other parts of the application (like streamlit_app.py).
# To test these functions, you would typically call them from a separate script
# with dummy data or mocked API responses.
