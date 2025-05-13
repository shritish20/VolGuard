import logging
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Assuming upstox_api module exists and can be imported for real-time data fetching
# from upstox_api import fetch_real_time_market_data # This function is called from load_data

# Setup logging for this module
logger = logging.getLogger(__name__)

# Define feature columns used in modeling
# IMPORTANT NOTE: Due to the limitations of the provided data sources (CSV + Latest API),
# many of these features are synthetically generated for historical dates.
# Only NIFTY_Close, VIX, and Realized_Vol are derived from historical data.
# Features marked "Real (live) or Synthetic" use live API data for the latest day
# but are synthetic for historical days if no historical source is available.
FEATURE_COLS = [
    'VIX', # Real historical (from CSV) + Real live (from API)
    'ATM_IV', # Derived from VIX (historically synthetic) + Real live (from API VIX)
    'IVP', # Derived from ATM_IV (historically synthetic)
    'PCR', # Real live (from API) or Synthetic fallback (historically synthetic)
    'VIX_Change_Pct', # Derived from VIX (historically real/synthetic)
    'IV_Skew', # Synthetic (historically and for fallback)
    'Straddle_Price', # Real live (from API) or Synthetic fallback (historically synthetic)
    'Spot_MaxPain_Diff_Pct', # Real live (from API) or Synthetic fallback (historically synthetic)
    'Days_to_Expiry', # Derived from expiry date (historically approximate/synthetic)
    'Event_Flag', # Derived from Days_to_Expiry or date (historically approximate/synthetic)
    'FII_Index_Fut_Pos', # Synthetic (historically and for fallback)
    'FII_Option_Pos', # Synthetic (historically and for fallback) - Proxy for DII in original code logic
    # Added other potentially useful features calculated below:
    'Realized_Vol', # Derived from NIFTY_Close (historically real)
    'Advance_Decline_Ratio', # Synthetic
    'Capital_Pressure_Index', # Synthetic/Derived from Synthetic features
    'Gamma_Bias', # Synthetic/Derived from Synthetic features
    'NIFTY_Close', # Raw historical (from CSV) + Real live (from API)
    'Total_Capital' # Input Parameter (constant)
]

# Data Loading (API First, then CSV Fallback)
def load_data(upstox_client):
    """
    Attempts to load historical NIFTY/VIX data from CSV and the latest real-time
    NIFTY and VIX data from Upstox API if client is provided. Combines these.
    Note: This function primarily loads historical NIFTY and VIX, and the latest
    real-time NIFTY and VIX. It does NOT load historical option chain data or
    derived metrics (like historical PCR, IV Skew, etc.) for historical dates.

    Args:
        upstox_client (dict or None): Initialized Upstox client dictionary if logged in,
                                     otherwise None.

    Returns:
        tuple: (df (pd.DataFrame or None), real_data (dict or None), data_source_tag (str)).
               df is the combined historical + latest DataFrame.
               real_data is the latest real-time data dict from API.
               data_source_tag indicates where the primary data came from.
               Returns (None, None, tag) if data loading fails critically.
    """
    logger.info("Starting data loading process.")
    df = None # This will be the combined historical + live DataFrame
    real_data = None # This will hold the latest real-time data dict from API
    data_source = "CSV (FALLBACK)"  # Default source tag

    # Attempt to fetch real-time data from API ONLY if a client is provided
    if upstox_client:
        try:
            # Assuming fetch_real_time_market_data is imported from upstox_api
            # This function fetches the latest Nifty, VIX, PCR, Straddle, Option Chain, etc.
            from upstox_api import fetch_real_time_market_data
            real_data = fetch_real_time_market_data(upstox_client)

            # Check if real-time data was successfully fetched and contains essential values
            if real_data is not None and real_data.get("nifty_spot") is not None and real_data.get("vix") is not None:
                logger.info("Successfully fetched latest essential real-time data from Upstox API.")
                # Use the source tag provided by fetch_real_time_market_data if available
                data_source = real_data.get("source", "Upstox API (LIVE)")

                latest_date = datetime.now().date()
                # Create a DataFrame row for the latest live Nifty and VIX data point
                # We only add NIFTY_Close and VIX to the base DF here. Other real_data
                # metrics (PCR, Straddle, etc.) will be used in generate_features.
                live_df_row = pd.DataFrame({
                    "NIFTY_Close": [real_data["nifty_spot"]],
                    "VIX": [real_data["vix"]]
                }, index=[pd.to_datetime(latest_date).normalize()]) # Use normalized datetime as index


                # Load historical CSV data regardless of API success, so we have history
                try:
                    nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
                    vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

                    logger.info("Fetching historical NIFTY and VIX data from CSV URLs.")
                    # Fetch Nifty data
                    nifty_res = requests.get(nifty_url)
                    nifty_res.raise_for_status() # Raise an exception for bad status codes (e.g., 404)
                    # Use io.StringIO to read text content as if it were a file
                    nifty = pd.read_csv(io.StringIO(nifty_res.text), encoding="utf-8-sig")

                    # Fetch VIX data
                    vix_res = requests.get(vix_url)
                    vix_res.raise_for_status() # Raise an exception for bad status codes
                    vix = pd.read_csv(io.StringIO(vix_res.text))

                    # Standardize column names by stripping whitespace
                    nifty.columns = nifty.columns.str.strip()
                    vix.columns = vix.columns.str.strip()

                    # Convert 'Date' columns to datetime, coercing errors to NaT (Not a Time)
                    # Use the correct format string based on the CSV format ("%d-%b-%Y")
                    nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
                    vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")

                    # Set 'Date' as index and select relevant columns ('Close')
                    # Drop rows where date parsing failed (Date is NaT)
                    nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
                    vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

                    # Normalize index (remove time part) and handle potential duplicate dates by keeping the last entry for each date
                    nifty.index = nifty.index.normalize()
                    vix.index = vix.index.normalize()
                    nifty = nifty.groupby(nifty.index).last()
                    vix = vix.groupby(vix.index).last()

                    # Find common dates between Nifty and VIX historical data DataFrames
                    common_dates = nifty.index.intersection(vix.index)
                    historical_df = pd.DataFrame({
                        "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                        "VIX": vix["VIX"].loc[common_dates]
                    }, index=common_dates).dropna() # Drop rows with NaN in NIFTY_Close or VIX on common dates


                    # Exclude the live data date from historical data before combining
                    # This prevents having two entries for the same date (one from CSV, one live)
                    # Ensure live_df_row index is not empty before accessing its date
                    if not live_df_row.empty:
                        historical_df = historical_df[historical_df.index < live_df_row.index[0]]
                    else:
                        logger.warning("Live DataFrame row is empty, cannot exclude its date from historical data.")


                    # Combine historical CSV data with the latest live API data row
                    # Use ignore_index=False to keep the date index
                    df = pd.concat([historical_df, live_df_row], ignore_index=False)
                    # Handle any potential remaining duplicates after concat (shouldn't happen if filter worked)
                    df = df.groupby(df.index).last()
                    # Sort by date index to ensure chronological order
                    df = df.sort_index()
                    # Fill any remaining NaNs in NIFTY_Close or VIX - ffill then bfill.
                    # This is important if there are gaps in CSV data or between historical and live data.
                    df = df.ffill().bfill()

                    logger.debug(f"Combined historical CSV and latest live API data. Resulting DataFrame shape: {df.shape}")

                except requests.exceptions.RequestException as req_e:
                     logger.error(f"Error fetching historical CSV data (network/request): {str(req_e)}")
                     logger.warning("Proceeding with only latest live data point due to CSV fetch error.")
                     # If CSV fetch fails, df is set to live_df_row (if it was successfully created)
                     if live_df_row is not None and not live_df_row.empty:
                          df = live_df_row
                          data_source = "Upstox API (LIVE, CSV Fetch Failed)"
                     else:
                          # If live_df_row was also not created, df remains None, fallback below will handle this
                          df = None
                          data_source = "CSV (FALLBACK FAILED), API (LIVE, CSV Fetch Failed)" # Update source to reflect full failure


                except Exception as e:
                    logger.error(f"Error processing historical CSV data: {str(e)}")
                    logger.warning("Proceeding with only latest live data point due to CSV parsing/processing error.")
                    # If CSV processing fails, df is set to live_df_row (if it was successfully created)
                    if live_df_row is not None and not live_df_row.empty:
                         df = live_df_row
                         data_source = "Upstox API (LIVE, CSV Process Failed)"
                    else:
                          # If live_df_row was also not created, df remains None, fallback below will handle this
                          df = None
                          data_source = "CSV (FALLBACK FAILED), API (LIVE, CSV Process Failed)"


            else:
                # API fetch was successful but did not return valid Nifty/VIX data
                logger.warning("Fetched real-time data, but essential NIFTY_Close or VIX is missing or invalid. DF will be based on CSV only.")
                # real_data is already set to the (incomplete) fetched data, will be returned.
                # df remains None, the fallback block below will attempt to load from CSV only.
                df = None # Ensure df is None to trigger CSV fallback for the DataFrame base


        except Exception as api_e:
             logger.error(f"An unexpected error occurred during real-time API fetch attempt: {str(api_e)}")
             # Any error during the API fetch attempt means we can't get live data for the DF base from API
             real_data = None # Ensure real_data is None if API fetch had an exception
             df = None # Ensure df is None to trigger CSV fallback for the DataFrame base


    # Fallback to CSV only for DataFrame base if API fetch was not attempted or failed to get essential data
    if df is None: # This condition is met if the API block above didn't successfully create the base df
        logger.info("API fetch failed or skipped, or essential data missing from API. Falling back to CSV only for DataFrame base.")
        data_source = "CSV (FALLBACK)" # Reset source tag for CSV-only load
        try:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

            logger.info("Fetching historical NIFTY and VIX data from CSV URLs (fallback).")
            # Fetch Nifty data
            nifty_res = requests.get(nifty_url)
            nifty_res.raise_for_status()
            nifty = pd.read_csv(io.StringIO(nifty_res.text), encoding="utf-8-sig")

            # Fetch VIX data
            vix_res = requests.get(vix_url)
            vix_res.raise_for_status()
            vix = pd.read_csv(io.StringIO(vix_res.text))

            # Standardize column names
            nifty.columns = nifty.columns.str.strip()
            vix.columns = vix.columns.str.strip()

            # Convert 'Date' columns to datetime, coercing errors to NaT
            nifty["Date"] = pd.to_datetime(nifty["Date"], format="%d-%b-%Y", errors="coerce")
            vix["Date"] = pd.to_datetime(vix["Date"], format="%d-%b-%Y", errors="coerce")

            # Set 'Date' as index and select relevant columns
            nifty = nifty.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "NIFTY_Close"})
            vix = vix.dropna(subset=["Date"]).set_index("Date")[["Close"]].rename(columns={"Close": "VIX"})

            # Normalize index and handle potential duplicates
            nifty.index = nifty.index.normalize()
            vix.index = vix.index.normalize()
            nifty = nifty.groupby(nifty.index).last()
            vix = vix.groupby(vix.index).last()

            # Find common dates and create combined historical DataFrame
            common_dates = nifty.index.intersection(vix.index)
            df = pd.DataFrame({
                "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                "VIX": vix["VIX"].loc[common_dates]
            }, index=common_dates).dropna() # Drop rows with NaN after merge

            df = df.groupby(df.index).last() # Just in case any duplicates survived
            df = df.sort_index()
            df = df.ffill().bfill() # Fill NaNs

            logger.debug(f"Loaded DataFrame base from CSV fallback. Shape: {df.shape}")

        except requests.exceptions.RequestException as req_e:
             logger.critical(f"Fatal error loading data from CSV fallback (network/request): {str(req_e)}")
             # If CSV fallback also fails critically, return None for df
             return None, real_data, "Data Load Failed (CSV Req Error)" # Pass real_data if it was partially fetched

        except Exception as e:
            logger.critical(f"Fatal error loading data from CSV fallback: {str(e)}")
            # If CSV fallback also fails critically, return None for df
            return None, real_data, "Data Load Failed (CSV Error)" # Pass real_data if it was partially fetched


    # Final check to ensure we have enough data points after loading
    # Need at least 1 data point to generate features for the latest day
    if df is None or len(df) < 1:
        logger.critical(f"Insufficient data loaded for analysis. Need at least 1 data point, got {len(df) if df is not None else 0}.")
        return None, real_data, data_source # Return None for df if data is insufficient


    # Ensure the DataFrame index is datetime and sorted before returning
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()

    logger.info(f"Data loading successful. Final DataFrame base shape: {df.shape}. Source: {data_source}")
    # Return the base DataFrame (NIFTY_Close, VIX) and the latest real_data dictionary
    return df, real_data, data_source

# Feature Generation
def generate_features(df, real_data, capital):
    """
    Generates technical and market-related features from the base DataFrame (NIFTY_Close, VIX).
    Uses the latest real-time data (real_data) to populate features for the last day.
    For historical dates, features are derived from historical NIFTY/VIX where possible,
    or synthetically generated where historical source data is not available.

    Args:
        df (pd.DataFrame): The base DataFrame containing historical and latest NIFTY_Close and VIX.
                           Needs a datetime index.
        real_data (dict or None): Dictionary containing the latest real-time market data,
                                  as returned by fetch_real_time_market_data, or None.
        capital (float): The user's total trading capital.

    Returns:
        pd.DataFrame or None: The DataFrame with all generated features, or None on error.
    """
    try:
        logger.info("Generating features.")
        # Work on a copy of the DataFrame to avoid modifying the original in session state
        df = df.copy()
        # Ensure index is datetime and normalized
        df.index = pd.to_datetime(df.index).normalize()
        n_days = len(df) # Number of data points

        if n_days == 0:
            logger.warning("Cannot generate features from empty DataFrame.")
            return None
        # Provide a warning if DataFrame is short, as some features need history
        if n_days < 50: # Increased warning threshold for features requiring more history
             logger.warning(f"DataFrame has only {n_days} days. Some features (e.g., rolling metrics, IVP) may not be meaningful or stable.")


        # --- Prepare latest real-time data for overriding the last row ---
        # Safely access real_data values if available and valid, otherwise they remain None
        latest_real_pcr = real_data.get("pcr") if real_data and real_data.get("pcr") is not None and not pd.isna(real_data.get("pcr")) else None
        latest_real_straddle_price = real_data.get("straddle_price") if real_data and real_data.get("straddle_price") is not None and not pd.isna(real_data.get("straddle_price")) else None
        latest_real_max_pain_diff_pct = real_data.get("max_pain_diff_pct") if real_data and real_data.get("max_pain_diff_pct") is not None and not pd.isna(real_data.get("max_pain_diff_pct")) else None
        latest_real_vix_change_pct = real_data.get("vix_change_pct") if real_data and real_data.get("vix_change_pct") is not None and not pd.isna(real_data.get("vix_change_pct")) else None
        # ATM_IV on the last day will be the latest VIX, which is already in df if API load was successful.


        # --- Calculate Days to Expiry and Event Flag ---
        # Prioritize the fetched expiry date from real_data for calculating DTE accurately for the latest day
        fetched_expiry_date = None
        if real_data and real_data.get("expiry") and isinstance(real_data["expiry"], str):
             try:
                 # Parse the fetched expiry date string (expected format YYYY-MM-DD)
                 fetched_expiry_date = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date()
             except ValueError:
                 logger.warning(f"Could not parse fetched expiry date string: {real_data['expiry']}. Using approximate DTE for latest day.")
                 fetched_expiry_date = None # Set to None if parsing fails

        def calculate_days_to_expiry(dates, latest_date_in_df):
            """
            Calculates days to nearest Thursday expiry for historical dates
            and uses the fetched expiry date for the latest date if available.
            """
            days_to_expiry = []
            for date_index in dates:
                date_only = date_index.date() # Get just the date part

                # For the latest date in the DataFrame, use the fetched expiry if available
                if date_only == latest_date_in_df.date() and fetched_expiry_date:
                     dte = (fetched_expiry_date - date_only).days
                else:
                     # For historical dates or if fetched expiry isn't available for the latest day,
                     # use the simple Thursday approximation. This historical approximation is a
                     # SYNTHETIC representation of historical DTE based on a fixed rule.
                     # This might not align with actual historical expiries that shifted due to holidays etc.
                     # Find the days ahead to the next Thursday (weekday 3)
                     days_ahead = (3 - date_only.weekday()) % 7
                     # If today is Thursday, the next expiry is today (DTE 0)
                     if days_ahead == 0:
                         dte = 0
                     else:
                         # Otherwise, DTE is the number of days until the next Thursday
                         dte = days_ahead
                     # Note: This simple logic doesn't account for monthly/quarterly expiries or holidays shifting expiry historically.

                # Ensure DTE is not negative (can happen if expiry date is in the past due to timing)
                days_to_expiry.append(max(0, dte))
            return np.array(days_to_expiry)

        # Apply the DTE calculation to the DataFrame index
        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index, df.index[-1])

        # Event Flag: Set to 1 if DTE is within a few days of expiry (e.g., <= 3 days)
        # For the latest day, this check uses the DTE calculated based on fetched expiry (if available).
        # For historical days, it uses the DTE based on the Thursday approximation.
        expiry_threshold_days = 3 # Consider an event if DTE is 3 days or less

        df["Event_Flag"] = np.where(
            df["Days_to_Expiry"] <= expiry_threshold_days, # Check if Days_to_Expiry is less than or equal to the threshold
            1, # Set Event_Flag to 1 if condition is met
            0  # Set Event_Flag to 0 otherwise
        )
        # Ensure the Event Flag for the last day is correctly set based on the latest DTE calculation
        # This line might be redundant after the np.where, but double-checking specific to the last day is safe.
        if df["Days_to_Expiry"].iloc[-1] <= expiry_threshold_days:
             df.loc[df.index[-1], "Event_Flag"] = 1
        else:
             df.loc[df.index[-1], "Event_Flag"] = 0 # Ensure it's 0 otherwise


        # --- Calculate ATM_IV (derived from VIX) ---
        # Historically, we assume ATM_IV is approximated by VIX.
        # For the latest day, ATM_IV is the latest VIX value, which is already in df['VIX'].iloc[-1]
        # if the API load was successful and returned a valid VIX.
        df["ATM_IV"] = df["VIX"] # Initialize ATM_IV with VIX values for all dates


        # --- Calculate IVP (Implied Volatility Percentile) ---
        # IVP calculation uses the historical ATM_IV series (which is based on VIX historically).
        # It shows how the current ATM_IV compares to its recent historical range.
        # Use a rolling window to calculate the percentile rank of the current day's ATM_IV
        # relative to the values within that rolling window.
        def dynamic_ivp(series):
            """
            Calculates the percentile rank of the last value in a pandas Series
            relative to the preceding values in the same series.
            """
            # Ensure sufficient data points in the rolling window (at least 5) and the last value is valid
            # pd.Series objects passed by rolling.apply have a length of the window size (or less at the start)
            if len(series) >= 5 and series.iloc[-1] is not None and pd.notna(series.iloc[-1]):
                # Get historical values from the series *excluding* the last one
                historical_values = series.iloc[:-1].dropna()
                current_value = series.iloc[-1]

                if not historical_values.empty:
                    # Calculate percentile rank: (number of historical values <= current) / total number of historical values * 100
                    # np.sum(historical_values <= current_value) counts how many past values are less than or equal to the current
                    percentile = (np.sum(historical_values <= current_value) / len(historical_values)) * 100
                    return percentile
            # Return 50.0 (representing the median) if insufficient data in the window or the current value is invalid
            return 50.0

        # Apply the dynamic_ivp function over a rolling window (e.g., 252 trading days, roughly 1 year)
        # min_periods=5 ensures the function is only called if at least 5 data points are in the window
        # raw=False passes a pandas Series chunk to the function, which is needed for .iloc
        df["IVP"] = df["ATM_IV"].rolling(window=252, min_periods=5).apply(dynamic_ivp, raw=False)

        # The rolling window creates NaNs at the beginning (where window is less than min_periods).
        # Interpolate these NaNs linearly and then fill any remaining NaNs (at the very start if min_periods is large) with 50.0
        df["IVP"] = df["IVP"].interpolate(method='linear').fillna(50.0)


        # --- Generate/Calculate Other Features ---

        # Market Trend: Calculate the 5-day simple moving average of NIFTY percentage change.
        # This feature is calculated historically from the NIFTY_Close data.
        df["NIFTY_Change_Pct"] = df["NIFTY_Close"].pct_change() # Calculate daily percentage change
        market_trend = df["NIFTY_Change_Pct"].rolling(window=5).mean().fillna(0) # 5-day rolling average, fill initial NaNs with 0


        # PCR: Use the live PCR for the last day if available from API, otherwise use a historical synthetic value.
        # Historically, PCR requires option chain data which is not available in the CSVs.
        # Therefore, historical PCR values are SYNTHETIC in this implementation.
        logger.warning("Historical PCR is synthetically generated due to lack of historical option chain data.")
        # Create a base synthetic PCR series for all dates. Varies around a base (e.g., 1.0) with random noise
        # and a potential negative correlation with market trend (falling market -> rising PCR generally).
        # np.random.normal(0, 0.05, n_days) generates random noise with mean 0 and standard deviation 0.05
        synthetic_pcr_history = 1.0 + np.random.normal(0, 0.05, n_days) + market_trend * -5
        # Clip values to a reasonable range for PCR (e.g., 0.7 to 2.0)
        synthetic_pcr_history = np.clip(synthetic_pcr_history, 0.7, 2.0)
        df["PCR"] = synthetic_pcr_history # Initialize PCR column with historical synthetic values

        # Override the last value in the PCR column with the actual live PCR from real_data if it's available and valid.
        if latest_real_pcr is not None:
             df.loc[df.index[-1], "PCR"] = latest_real_pcr
        else:
             # If live PCR is not available, ensure the last day's PCR is based on the synthetic generation or a default
             # The synthetic value is already assigned, so no action needed unless a specific default fallback is desired here.
             pass


        # VIX Change Percentage: Calculate the percentage change in VIX from the previous day.
        # This can be calculated historically from the historical VIX data available in the DataFrame.
        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100 # Calculate percentage change, fill initial NaN with 0, multiply by 100 for percentage
        # Override the last value with the actual live VIX Change % from real_data if available.
        if latest_real_vix_change_pct is not None:
             df.loc[df.index[-1], "VIX_Change_Pct"] = latest_real_vix_change_pct
        else:
             # If live VIX change % is not available, recalculate it for the last day
             # using the historical VIX series in the DataFrame.
             # Ensure there's a previous day and the VIX values are not NaN/zero.
             if n_days > 1 and pd.notna(df['VIX'].iloc[-1]) and pd.notna(df['VIX'].iloc[-2]) and df['VIX'].iloc[-2] != 0:
                  df.loc[df.index[-1], "VIX_Change_Pct"] = ((df['VIX'].iloc[-1] / df['VIX'].iloc[-2]) - 1) * 100
             else:
                  # If calculation is not possible (e.g., only one data point), default the last day's VIX change to 0.0.
                  df.loc[df.index[-1], "VIX_Change_Pct"] = 0.0


        # Spot vs Max Pain Difference Percentage: Use live data for the last day if available, otherwise synthetic.
        # Historically, Max Pain requires historical option chain data, which is not available.
        # Therefore, historical Spot_MaxPain_Diff_Pct values are SYNTHETIC.
        logger.warning("Historical Spot_MaxPain_Diff_Pct is synthetically generated due to lack of historical option chain data.")
        # Create a base synthetic series. Varies around a base (e.g., 0.5%) with random noise.
        # Potentially increases with Days_to_Expiry (Max Pain might be further from spot with more time).
        synthetic_max_pain_diff_history = 0.5 + np.random.normal(0, 0.1, n_days) + df["Days_to_Expiry"] * 0.01
        # Clip values to a reasonable range (e.g., 0.1% to 5.0%)
        synthetic_max_pain_diff_history = np.clip(synthetic_max_pain_diff_history, 0.1, 5.0)
        df["Spot_MaxPain_Diff_Pct"] = synthetic_max_pain_diff_history # Initialize column

        # Override the last value with actual live data if available and valid.
        if latest_real_max_pain_diff_pct is not None:
            df.loc[df.index[-1], "Spot_MaxPain_Diff_Pct"] = latest_real_max_pain_diff_pct
        # Else, the synthetic value for the last day remains.


        # FII/DII Positions: SYNTHETICALLY generated cumulative random walks.
        # Actual FII/DII data is external and requires a different data source.
        logger.warning("FII/DII positions (FII_Index_Fut_Pos, FII_Option_Pos) are synthetically generated and do NOT represent actual market data.")
        # Generate synthetic daily changes with some swings
        fii_trend_daily_change = np.random.normal(0, 5000, n_days)
        fii_trend_daily_change[::10] *= -1.5 # Introduce larger swings every 10 days (example rule)
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend_daily_change).astype(int) # Cumulative sum represents total position

        dii_trend_daily_change = np.random.normal(0, 2000, n_days)
        df["FII_Option_Pos"] = np.cumsum(dii_trend_daily_change).astype(int) # Using FII_Option_Pos as a placeholder/proxy for DII in original logic


        # IV Skew: SYNTHETICALLY generated, related to VIX and DTE.
        # Requires historical option chain data for real historical values.
        logger.warning("IV_Skew is synthetically generated due to lack of historical option chain data.")
        # Create a synthetic IV Skew. Can be related to VIX level (higher VIX often means higher skew)
        # and DTE (skew shape changes closer to expiry).
        # Example formula: varies around 0 with noise, adjusted by VIX level and DTE.
        df["IV_Skew"] = np.random.normal(0, 0.5, n_days) + (df["VIX"] / 20 - 1) * 2 + (df["Days_to_Expiry"] / 15 - 1) * 0.5
        # Clip values to a reasonable range for IV Skew (e.g., -3 to 3)
        df["IV_Skew"] = np.clip(df["IV_Skew"], -3, 3)


        # Realized Volatility: Annualized standard deviation of historical log returns.
        # This is calculated from the historical NIFTY_Close data and is NOT synthetic for historical dates.
        # Calculate daily log returns: log(Price_today / Price_yesterday)
        df["Log_Returns"] = np.log(df["NIFTY_Close"].pct_change() + 1)
        # Calculate rolling standard deviation of log returns (e.g., over 5 trading days)
        # Annulize by multiplying by sqrt(252) (approx trading days in a year) and multiply by 100 for percentage.
        df["Realized_Vol"] = df["Log_Returns"].rolling(window=5, min_periods=5).std() * np.sqrt(252) * 100
        # Fill initial NaNs created by the rolling window (first few days).
        # Use VIX values as a proxy fallback, then forward fill, then backward fill.
        # Finally, fill any remaining NaNs (e.g., if entire VIX column was NaN) with a default value (e.g., 15.0).
        df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"]) # Fill with VIX where Realized_Vol is NaN
        df["Realized_Vol"] = df["Realized_Vol"].fillna(method='ffill') # Forward fill remaining NaNs
        df["Realized_Vol"] = df["Realized_Vol"].fillna(method='bfill') # Backward fill any remaining NaNs at the start
        df["Realized_Vol"] = df["Realized_Vol"].fillna(15.0) # Fill any remaining NaNs with a default value


        # Advance/Decline Ratio: SYNTHETICALLY generated, related to market trend.
        # Actual Advance/Decline data requires external market breadth data.
        logger.warning("Advance_Decline_Ratio is synthetically generated.")
        # Create a synthetic ratio. Varies around 1.0 (equal advances/declines).
        # Positively correlated with market trend (rising market -> more advances).
        df["Advance_Decline_Ratio"] = 1.0 + np.random.normal(0, 0.1, n_days) + market_trend * 5
        # Clip values to a reasonable range (e.g., 0.7 to 1.5)
        df["Advance_Decline_Ratio"] = np.clip(df["Advance_Decline_Ratio"], 0.7, 1.5)


        # Capital Pressure Index: SYNTHETIC, derived from synthetic FII/DII and PCR.
        # This is a composite index based on other (historically synthetic) features.
        logger.warning("Capital_Pressure_Index is synthetically derived from synthetic features.")
        # Ensure component columns are numeric before calculation, coercing errors and filling NaNs with defaults.
        fii_fut = pd.to_numeric(df["FII_Index_Fut_Pos"], errors='coerce').fillna(0)
        fii_opt = pd.to_numeric(df["FII_Option_Pos"], errors='coerce').fillna(0) # Using FII_Option_Pos as proxy for DII
        pcr_numeric = pd.to_numeric(df["PCR"], errors='coerce').fillna(1.0) # Default PCR to 1.0 if NaN

        # Example formula combining components (scale factors are illustrative)
        df["Capital_Pressure_Index"] = (fii_fut / 50000.0 + fii_opt / 20000.0 + pcr_numeric - 1.0) / 3.0
        # Clip values to a reasonable range
        df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -1.5, 1.5)


        # Gamma Bias: SYNTHETIC, related to IV Skew and DTE.
        # This is a composite index based on other (historically synthetic) features.
        logger.warning("Gamma_Bias is synthetically derived from synthetic features.")
        # Ensure component columns are numeric, coercing errors and filling NaNs with defaults.
        iv_skew_numeric = pd.to_numeric(df["IV_Skew"], errors='coerce').fillna(0.0)
        dte_numeric = pd.to_numeric(df["Days_to_Expiry"], errors='coerce').fillna(5) # Default DTE to 5 if NaN

        # Example formula relating skew and DTE (bias might be stronger closer to expiry)
        # np.clip(dte_numeric, 1, 30) limits DTE to a range for the formula
        df["Gamma_Bias"] = iv_skew_numeric * (30 - np.clip(dte_numeric, 1, 30)) / 30
        # Clip values to a reasonable range
        df["Gamma_Bias"] = np.clip(df["Gamma_Bias"], -2, 2)


        # PnL_Day: SYNTHETICALLY generated random daily PnL.
        # This does NOT reflect PnL from any actual trading strategy run historically.
        # It's likely used as a placeholder feature or for demonstration purposes.
        logger.warning("PnL_Day is synthetically generated and does NOT reflect actual trading performance.")
        # Ensure capital is numeric, default to 1000000 if NaN
        capital_numeric = pd.to_numeric(capital, errors='coerce').fillna(1000000)
        # Generate random daily PnL, potentially reduced on event days.
        # np.random.normal(0, capital_numeric * 0.005, n_days) generates random daily returns around 0
        # (e.g., mean 0, std dev 0.5% of capital).
        # (1 - df["Event_Flag"] * 0.2) reduces the magnitude of PnL by 20% on event days (where Event_Flag is 1).
        df["PnL_Day"] = np.random.normal(0, capital_numeric * 0.005, n_days) * (1 - df["Event_Flag"] * 0.2)


        # Straddle Price: Use the live Straddle Price for the last day if available from API, otherwise synthetic.
        # Historically, Straddle Price requires option chain data which is not available.
        # Therefore, historical Straddle_Price values are SYNTHETIC.
        logger.warning("Historical Straddle_Price is synthetically generated due to lack of historical option chain data.")
        # Create a base synthetic series for all dates. Varies around a default/estimated base price with noise.
        # Use the live straddle price as a base if available, otherwise a default (e.g., 200.0).
        base_straddle = latest_real_straddle_price if latest_real_straddle_price is not None else 200.0
        synthetic_straddle_history = base_straddle + np.random.normal(0, base_straddle * 0.1, n_days)
        # Clip values to a reasonable range around the base price.
        synthetic_straddle_history = np.clip(synthetic_straddle_history, base_straddle * 0.5, base_straddle * 1.5)
        df["Straddle_Price"] = synthetic_straddle_history # Initialize column

        # Override the last value with the actual live Straddle Price from real_data if available and valid.
        if latest_real_straddle_price is not None:
            df.loc[df.index[-1], "Straddle_Price"] = latest_real_straddle_price
        # Else, the synthetic value for the last day remains.


        # Add Total Capital as a feature.
        # This is a constant value across all dates in this DataFrame.
        # Ensure capital input is numeric, default to 1000000 if NaN
        capital_numeric = pd.to_numeric(capital, errors='coerce').fillna(1000000)
        df["Total_Capital"] = capital_numeric


        # --- Final Data Cleaning and Validation ---

        # Ensure all defined FEATURE_COLS are in the DataFrame after generation.
        # If any expected column is missing, add it with NaN values.
        for col in FEATURE_COLS:
            if col not in df.columns:
                logger.warning(f"Adding missing FEATURE_COLS column with NaNs: {col}")
                df[col] = np.nan

        # Ensure all FEATURE_COLS are numeric where expected. Coerce errors to NaN.
        # Identify columns that are expected to be numeric (excluding Event_Flag which is 0 or 1)
        numeric_feature_cols = [col for col in FEATURE_COLS if col not in ['Event_Flag', 'Total_Capital']] # Total_Capital is numeric, but added separately
        # Let's include Total_Capital in numeric check
        numeric_feature_cols = [col for col in FEATURE_COLS if col != 'Event_Flag'] # Event_Flag is 0/1 int

        for col in numeric_feature_cols:
            if col in df.columns: # Only attempt if the column exists
                # Convert to numeric, turning non-numeric values into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for any remaining NaNs in the resulting DataFrame, specifically within the FEATURE_COLS.
        initial_nan_count = df[FEATURE_COLS].isna().sum().sum()
        if initial_nan_count > 0:
            logger.warning(f"NaNs found in FEATURE_COLS after initial generation and coercion: {initial_nan_count}. Attempting final fill.")
            # Display columns with NaNs before filling for debugging
            # logger.debug(df[FEATURE_COLS].isna().sum()[df[FEATURE_COLS].isna().sum() > 0])

            # Fill remaining NaNs in FEATURE_COLS.
            # Interpolate numeric columns first to fill gaps where values can be estimated.
            for col in numeric_feature_cols:
                 if col in df.columns:
                      df[col] = df[col].interpolate(method='linear')

            # Then fill any remaining NaNs (especially at the edges where interpolation doesn't work)
            # using forward fill (ffill) then backward fill (bfill).
            df[FEATURE_COLS] = df[FEATURE_COLS].fillna(method='ffill').fillna(method='bfill')

            # Re-check for NaNs after the final fill. If still present, it's a more serious issue.
            final_nan_count = df[FEATURE_COLS].isna().sum().sum()
            if final_nan_count > 0:
                logger.error(f"FATAL ERROR: NaNs still present in FEATURE_COLS after final interpolation/fill: {final_nan_count}")
                # Display columns that still contain NaNs
                logger.error("Columns in FEATURE_COLS with persistent NaNs:")
                logger.error(df[FEATURE_COLS].isna().sum()[df[FEATURE_COLS].isna().sum() > 0])
                # Depending on how critical NaNs are for downstream models, you might return None here.
                # For this example, we will return the DataFrame but log the critical error.
                # Downstream components should ideally handle potential NaNs gracefully or check for them.
            else:
                 logger.info("All NaNs in FEATURE_COLS successfully filled.")


        logger.info(f"Features generated successfully. Final DataFrame shape: {df.shape}.")
        # Return the DataFrame containing all generated features, including the base NIFTY_Close and VIX.
        return df

    except Exception as e:
        # Catch any unexpected errors during the feature generation process
        logger.error(f"An unexpected error occurred during feature generation: {str(e)}", exc_info=True)
        return None # Indicate failure


# The __main__ block is removed as this module's functions are intended to be imported
# and used by other parts of the application (like streamlit_app.py).
# To test these functions, you would typically call them from a separate script
# with dummy data or mocked API responses.
