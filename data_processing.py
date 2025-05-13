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
    derived metrics (like historical PCR, IV Skew, etc.).
    """
    logger.info("Starting data loading process.")
    df = None # This will be the combined historical + live DataFrame
    real_data = None # This will hold the latest real-time data dict from API
    data_source = "CSV (FALLBACK)"  # Default source tag

    # Attempt to fetch real-time data from API ONLY if a client is provided
    if upstox_client:
        try:
            # Assuming fetch_real_time_market_data is imported from upstox_api
            from upstox_api import fetch_real_time_market_data
            # fetch_real_time_market_data gets latest Nifty, VIX, PCR, Straddle, etc.
            real_data = fetch_real_time_market_data(upstox_client)

            # Check if real-time data was successfully fetched and contains essential values
            if real_data and real_data.get("nifty_spot") is not None and real_data.get("vix") is not None:
                logger.info("Successfully fetched latest real-time data from Upstox API.")
                # Use the source tag provided by fetch_real_time_market_data
                data_source = real_data.get("source", "Upstox API (LIVE)")

                latest_date = datetime.now().date()
                # Create a DataFrame row for the latest live Nifty and VIX data point
                # Other real_data metrics (PCR, Straddle, etc.) will be used in generate_features
                live_df_row = pd.DataFrame({
                    "NIFTY_Close": [real_data["nifty_spot"]],
                    "VIX": [real_data["vix"]]
                }, index=[pd.to_datetime(latest_date).normalize()])

                # Load historical CSV data regardless of API success, so we have history
                try:
                    nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
                    vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

                    logger.info("Fetching historical NIFTY and VIX data from CSV URLs.")
                    # Fetch Nifty data
                    nifty_res = requests.get(nifty_url)
                    nifty_res.raise_for_status() # Raise an exception for bad status codes
                    nifty = pd.read_csv(io.StringIO(nifty_res.text), encoding="utf-8-sig")

                    # Fetch VIX data
                    vix_res = requests.get(vix_url)
                    vix_res.raise_for_status() # Raise an exception for bad status codes
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

                    # Normalize index (remove time part) and handle potential duplicate dates by keeping the last
                    nifty.index = nifty.index.normalize()
                    vix.index = vix.index.normalize()
                    nifty = nifty.groupby(nifty.index).last()
                    vix = vix.groupby(vix.index).last()

                    # Find common dates between Nifty and VIX historical data
                    common_dates = nifty.index.intersection(vix.index)
                    historical_df = pd.DataFrame({
                        "NIFTY_Close": nifty["NIFTY_Close"].loc[common_dates],
                        "VIX": vix["VIX"].loc[common_dates]
                    }, index=common_dates).dropna() # Drop rows with NaN in NIFTY_Close or VIX on common dates


                    # Exclude the live data date from historical data before combining
                    # This prevents having two entries for the same date (one from CSV, one live)
                    historical_df = historical_df[historical_df.index < live_df_row.index[0]]

                    # Combine historical CSV data with the latest live API data row
                    df = pd.concat([historical_df, live_df_row])
                    # Handle any potential remaining duplicates after concat (shouldn't happen if filter worked)
                    df = df.groupby(df.index).last()
                    # Sort by date index
                    df = df.sort_index()
                    # Fill any remaining NaNs - ffill then bfill. This is important if there are gaps in CSV data.
                    df = df.ffill().bfill()

                    logger.debug(f"Combined historical CSV and latest live API data. Shape: {df.shape}")

                except requests.exceptions.RequestException as req_e:
                     logger.error(f"Error fetching historical CSV data (network/request): {str(req_e)}")
                     logger.warning("Proceeding with only latest live data point due to CSV fetch error.")
                     df = live_df_row # Use only live data if CSV fetch fails
                     data_source = "Upstox API (LIVE, CSV Fetch Failed)"
                except Exception as e:
                    logger.error(f"Error processing historical CSV data: {str(e)}")
                    logger.warning("Proceeding with only latest live data point due to CSV parsing/processing error.")
                    df = live_df_row # Use only live data if CSV processing fails
                    data_source = "Upstox API (LIVE, CSV Process Failed)"

            else:
                # API fetch was successful but did not return valid Nifty/VIX data
                logger.warning("Fetched real-time data but NIFTY_Close or VIX is missing/invalid. real_data will be returned, but DF will be based on CSV only.")
                # real_data is already set, will be returned. df will be loaded from CSV in the fallback below.
                real_data = None # Ensure real_data is None for the subsequent check to trigger CSV fallback for df

        except Exception as api_e:
             logger.error(f"Generic error during real-time API fetch attempt: {str(api_e)}")
             # Any error during the API fetch attempt means we can't get live data for the DF base
             real_data = None # Ensure real_data is None for the subsequent check to trigger CSV fallback for df


    # Fallback to CSV only for DataFrame base if API fetch was not attempted or failed to get essential data
    if df is None: # This condition is met if the API block above didn't successfully create the base df
        logger.info("API fetch failed or skipped, or essential data missing from API. Falling back to CSV only for DataFrame base.")
        data_source = "CSV (FALLBACK)"
        try:
            nifty_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv"
            vix_url = "https://raw.githubusercontent.com/shritish20/VolGuard/main/india_vix.csv"

            logger.info("Fetching historical NIFTY and VIX data from CSV URLs (fallback).")
            nifty_res = requests.get(nifty_url)
            nifty_res.raise_for_status()
            nifty = pd.read_csv(io.StringIO(nifty_res.text), encoding="utf-8-sig")

            vix_res = requests.get(vix_url)
            vix_res.raise_for_status()
            vix = pd.read_csv(io.StringIO(vix_res.text))

            nifty.columns = nifty.columns.str.strip()
            vix.columns = vix.columns.str.strip()

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
            }, index=common_dates).dropna() # Drop rows with NaN

            df = df.groupby(df.index).last()
            df = df.sort_index()
            df = df.ffill().bfill() # Fill NaNs

            logger.debug(f"Loaded DataFrame base from CSV fallback. Shape: {df.shape}")

        except requests.exceptions.RequestException as req_e:
             logger.error(f"Fatal error loading data from CSV fallback (network/request): {str(req_e)}")
             return None, real_data, "Data Load Failed (CSV Req Error)" # Return failure, but pass real_data if it was fetched
        except Exception as e:
            logger.error(f"Fatal error loading data from CSV fallback: {str(e)}")
            return None, real_data, "Data Load Failed (CSV Error)" # Return failure, but pass real_data if it was fetched

    # Final check to ensure we have enough data points after loading
    if df is None or len(df) < 2:
        logger.error(f"Insufficient data loaded for analysis. Need at least 2 data points, got {len(df) if df is not None else 0}.")
        return None, real_data, data_source # Return failure, but pass real_data if it was fetched

    # Ensure the DataFrame index is datetime and sorted
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()

    logger.info(f"Data loading successful. Final DataFrame base shape: {df.shape}. Source: {data_source}")
    # Return the base DataFrame (NIFTY_Close, VIX) and the latest real_data
    return df, real_data, data_source

# Feature Generation
def generate_features(df, real_data, capital):
    """
    Generates technical and market-related features from the base DataFrame (NIFTY_Close, VIX).
    Uses the latest real-time data (real_data) to populate features for the last day.
    For historical dates, features are derived from historical NIFTY/VIX where possible,
    or synthetically generated where historical source data is not available.
    """
    try:
        logger.info("Generating features.")
        df = df.copy() # Work on a copy to avoid modifying the original DataFrame in session state
        df.index = pd.to_datetime(df.index).normalize() # Ensure index is datetime and normalized
        n_days = len(df)

        if n_days == 0:
            logger.warning("Cannot generate features from empty DataFrame.")
            return None
        if n_days < 50: # Increased warning threshold for features requiring more history
             logger.warning(f"DataFrame has only {n_days} days. Some features (e.g., rolling metrics, IVP) may not be meaningful or stable.")


        # --- Prepare latest real-time data for overriding the last row ---
        # Use real_data values if available and valid, otherwise use fallbacks
        latest_real_pcr = real_data.get("pcr") if real_data and real_data.get("pcr") is not None and not pd.isna(real_data.get("pcr")) else None
        latest_real_straddle_price = real_data.get("straddle_price") if real_data and real_data.get("straddle_price") is not None and not pd.isna(real_data.get("straddle_price")) else None
        latest_real_max_pain_diff_pct = real_data.get("max_pain_diff_pct") if real_data and real_data.get("max_pain_diff_pct") is not None and not pd.isna(real_data.get("max_pain_diff_pct")) else None
        latest_real_vix_change_pct = real_data.get("vix_change_pct") if real_data and real_data.get("vix_change_pct") is not None and not pd.isna(real_data.get("vix_change_pct")) else None
        # ATM_IV on the last day will be the latest VIX, which is already in df if API load was successful

        # --- Calculate Days to Expiry and Event Flag ---
        # Prioritize the fetched expiry date from real_data for calculating DTE accurately for the latest day
        fetched_expiry_date = None
        if real_data and real_data.get("expiry") and isinstance(real_data["expiry"], str):
             try:
                 fetched_expiry_date = datetime.strptime(real_data["expiry"], "%Y-%m-%d").date()
             except ValueError:
                 logger.warning(f"Could not parse fetched expiry date string: {real_data['expiry']}")
                 fetched_expiry_date = None

        def calculate_days_to_expiry(dates, latest_date_in_df):
            """Calculates days to nearest Thursday expiry. Uses fetched expiry for the latest date."""
            days_to_expiry = []
            for date in dates:
                date_only = date.date()
                # For the latest date in the DataFrame, use the fetched expiry if available
                if date_only == latest_date_in_df.date() and fetched_expiry_date:
                     dte = (fetched_expiry_date - date_only).days
                else:
                     # For historical dates or if fetched expiry isn't available, use the simple Thursday approximation
                     # This historical approximation is a SYNTHETIC representation of historical DTE.
                     days_ahead = (3 - date_only.weekday()) % 7 # Thursday is weekday 3
                     if days_ahead == 0: # If the historical date was a Thursday, the expiry is on that day
                         days_ahead = 0 # DTE is 0
                     # Note: This simple logic doesn't account for monthly/quarterly expiries or holidays shifting expiry historically.
                     dte = (date_only + timedelta(days=days_ahead) - date_only).days # Recalculate based on approx expiry date

                days_to_expiry.append(max(0, dte)) # Ensure DTE is not negative
            return np.array(days_to_expiry)

        # Apply the DTE calculation
        df["Days_to_Expiry"] = calculate_days_to_expiry(df.index, df.index[-1])

        # Event Flag: Set to 1 if today is expiry or within 3 days of expiry
        # For the latest day, check against the fetched expiry. For historical days, check against the approximate DTE.
        df["Event_Flag"] = np.where(
            (df["Days_to_Expiry"] <= 3) | # Within 3 days *before* the (approx/real) expiry
            ( (df.index.date == fetched_expiry_date) if fetched_expiry_date else False ) # OR exactly on the fetched expiry date for the last day
            , 1, 0
        )
        # Ensure the Event Flag for the last day is correctly set based on the latest DTE calculation
        if df["Days_to_Expiry"].iloc[-1] <= 3 or (fetched_expiry_date and df.index[-1].date() == fetched_expiry_date):
             df.loc[df.index[-1], "Event_Flag"] = 1
        else:
             df.loc[df.index[-1], "Event_Flag"] = 0


        # --- Calculate ATM_IV (derived from VIX) ---
        # Historically, we assume ATM_IV is close to VIX.
        # For the latest day, ATM_IV is the latest VIX, which is already in df['VIX'].iloc[-1].
        df["ATM_IV"] = df["VIX"] # Historically, ATM_IV is approximated by VIX


        # --- Calculate IVP (Implied Volatility Percentile) ---
        # IVP calculation uses the historical ATM_IV series (which is based on VIX historically).
        # Use a rolling window to calculate the percentile of the current day's ATM_IV
        # relative to the values in the window.
        def dynamic_ivp(series):
            """Calculates the percentile rank of the last value in a series."""
            # Ensure sufficient data points in the rolling window and the last value is valid
            if len(series) >= 5 and series.iloc[-1] is not None and not pd.isna(series.iloc[-1]):
                # Use values in the series *excluding* the last one as historical data
                historical_values = series.iloc[:-1].dropna()
                current_value = series.iloc[-1]
                if not historical_values.empty:
                    # Calculate percentile rank: (number of historical values <= current) / total historical values
                    percentile = (np.sum(historical_values <= current_value) / len(historical_values)) * 100
                    return percentile
            # Return 50.0 (median) if insufficient data or value is NaN
            return 50.0

        # Apply dynamic_ivp over a rolling window (e.g., 252 trading days ~ 1 year)
        # min_periods=5 ensures calculation only happens with at least 5 data points
        # Use raw=False to pass pandas Series chunks to the function
        df["IVP"] = df["ATM_IV"].rolling(252, min_periods=5).apply(dynamic_ivp, raw=False)
        # Interpolate any NaNs created by the rolling window (especially at the beginning)
        df["IVP"] = df["IVP"].interpolate(method='linear').fillna(50.0) # Fill remaining NaNs with 50.0


        # --- Generate/Calculate Other Features ---

        # Market Trend: 5-day moving average of NIFTY percentage change
        df["NIFTY_Change_Pct"] = df["NIFTY_Close"].pct_change()
        market_trend = df["NIFTY_Change_Pct"].rolling(5).mean().fillna(0) # Fill initial NaNs with 0

        # PCR: Use live PCR for the last day if available, otherwise use a historical synthetic value.
        # Historically, PCR is SYNTHETIC as we don't have historical option chain data.
        logger.warning("Historical PCR is synthetically generated due to lack of historical option chain data.")
        # Create a base synthetic PCR series for all dates except the last
        synthetic_pcr_history = 1.0 + np.random.normal(0, 0.05, n_days) + market_trend * -5
        synthetic_pcr_history = np.clip(synthetic_pcr_history, 0.7, 2.0) # Keep within a reasonable range
        df["PCR"] = synthetic_pcr_history
        # Override the last value with the actual live PCR if available
        if latest_real_pcr is not None:
             df.loc[df.index[-1], "PCR"] = latest_real_pcr


        # VIX Change Percentage: Calculate percentage change from previous day's VIX
        # This can be calculated historically from the historical VIX data.
        df["VIX_Change_Pct"] = df["VIX"].pct_change().fillna(0) * 100 # Fill initial NaN with 0
        # Override the last value with the actual live VIX Change % if available
        if latest_real_vix_change_pct is not None:
             df.loc[df.index[-1], "VIX_Change_Pct"] = latest_real_vix_change_pct
        # Handle case where VIX might be 0 or NaN previously for the latest day
        elif n_days > 1 and not pd.isna(df['VIX'].iloc[-1]) and not pd.isna(df['VIX'].iloc[-2]) and df['VIX'].iloc[-2] != 0:
             df.loc[df.index[-1], "VIX_Change_Pct"] = ((df['VIX'].iloc[-1] / df['VIX'].iloc[-2]) - 1) * 100
        else:
             # If even historical VIX change for the last day cannot be calculated, default to 0
             df.loc[df.index[-1], "VIX_Change_Pct"] = 0.0


        # Spot vs Max Pain Difference Percentage: Use live data for the last day if available, otherwise synthetic.
        # Historically, Max Pain requires historical option chain data, so this is SYNTHETIC.
        logger.warning("Historical Spot_MaxPain_Diff_Pct is synthetically generated due to lack of historical option chain data.")
        # Create a base synthetic series, potentially increasing with DTE
        synthetic_max_pain_diff_history = 0.5 + np.random.normal(0, 0.1, n_days) + df["Days_to_Expiry"] * 0.01
        synthetic_max_pain_diff_history = np.clip(synthetic_max_pain_diff_history, 0.1, 5.0) # Keep within a reasonable range
        df["Spot_MaxPain_Diff_Pct"] = synthetic_max_pain_diff_history
        # Override the last value with actual live data if available
        if latest_real_max_pain_diff_pct is not None:
            df.loc[df.index[-1], "Spot_MaxPain_Diff_Pct"] = latest_real_max_pain_diff_pct


        # FII/DII Positions: SYNTHETICALLY generated cumulative random walks.
        # These do NOT reflect actual FII/DII data. Real FII/DII data is external.
        logger.warning("FII/DII positions are synthetically generated and do NOT represent actual market data.")
        fii_trend = np.random.normal(0, 5000, n_days)
        fii_trend[::10] *= -1.5 # Introduce some swings
        df["FII_Index_Fut_Pos"] = np.cumsum(fii_trend).astype(int)
        df["FII_Option_Pos"] = np.cumsum(np.random.normal(0, 2000, n_days)).astype(int)


        # IV Skew: SYNTHETICALLY generated, related to VIX and DTE.
        # Requires historical option chain data for real historical values.
        logger.warning("IV_Skew is synthetically generated due to lack of historical option chain data.")
        df["IV_Skew"] = np.random.normal(0, 0.5, n_days) + (df["VIX"] / 20 - 1) * 2 + (df["Days_to_Expiry"] / 15 - 1) * 0.5
        df["IV_Skew"] = np.clip(df["IV_Skew"], -3, 3) # Keep within a reasonable range


        # Realized Volatility: Annualized standard deviation of historical log returns
        # This is calculated from the historical NIFTY_Close data and is NOT synthetic.
        # Already calculated above when preparing market_trend
        # Ensure the column exists and is filled
        if "Realized_Vol" not in df.columns:
             df["Realized_Vol"] = df["Log_Returns"].rolling(5, min_periods=5).std() * np.sqrt(252) * 100
             df["Realized_Vol"] = df["Realized_Vol"].fillna(df["VIX"]).fillna(method='ffill').fillna(method='bfill').fillna(15.0)


        # Advance/Decline Ratio: SYNTHETICALLY generated, related to market trend
        # Requires historical advance/decline data for real values.
        logger.warning("Advance_Decline_Ratio is synthetically generated.")
        df["Advance_Decline_Ratio"] = 1.0 + np.random.normal(0, 0.1, n_days) + market_trend * 5
        df["Advance_Decline_Ratio"] = np.clip(df["Advance_Decline_Ratio"], 0.7, 1.5) # Keep within a reasonable range


        # Capital Pressure Index: SYNTHETIC, derived from synthetic FII/DII and PCR
        logger.warning("Capital_Pressure_Index is synthetically derived.")
        # Ensure PCR, FII_Index_Fut_Pos, FII_Option_Pos are numeric before calculation
        df["PCR"] = pd.to_numeric(df["PCR"], errors='coerce').fillna(1.0)
        df["FII_Index_Fut_Pos"] = pd.to_numeric(df["FII_Index_Fut_Pos"], errors='coerce').fillna(0)
        df["FII_Option_Pos"] = pd.to_numeric(df["FII_Option_Pos"], errors='coerce').fillna(0)

        df["Capital_Pressure_Index"] = (df["FII_Index_Fut_Pos"] / 5e4 + df["FII_Option_Pos"] / 2e4 + df["PCR"] - 1) / 3
        df["Capital_Pressure_Index"] = np.clip(df["Capital_Pressure_Index"], -1.5, 1.5) # Keep within a reasonable range


        # Gamma Bias: SYNTHETIC, related to IV Skew and DTE
        logger.warning("Gamma_Bias is synthetically derived.")
        # Ensure IV_Skew and Days_to_Expiry are numeric before calculation
        df["IV_Skew"] = pd.to_numeric(df["IV_Skew"], errors='coerce').fillna(0.0)
        df["Days_to_Expiry"] = pd.to_numeric(df["Days_to_Expiry"], errors='coerce').fillna(5) # Default DTE to 5 if NaN


        df["Gamma_Bias"] = df["IV_Skew"] * (30 - np.clip(df["Days_to_Expiry"], 1, 30)) / 30
        df["Gamma_Bias"] = np.clip(df["Gamma_Bias"], -2, 2) # Keep within a reasonable range


        # PnL_Day: SYNTHETICALLY generated random daily PnL
        # This does NOT reflect PnL from any actual trading strategy run historically.
        logger.warning("PnL_Day is synthetically generated and does NOT reflect actual trading performance.")
        # Base random PnL, slightly reduced on event days
        # Ensure capital is numeric, default to 1000000 if NaN
        capital = pd.to_numeric(capital, errors='coerce').fillna(1000000)
        df["PnL_Day"] = np.random.normal(0, capital * 0.005, n_days) * (1 - df["Event_Flag"] * 0.2)


        # Straddle Price: Use live Straddle Price for the last day if available, otherwise synthetic.
        # Historically, Straddle Price requires historical option chain data, so this is SYNTHETIC.
        logger.warning("Historical Straddle_Price is synthetically generated due to lack of historical option chain data.")
        # Create a base synthetic series with noise
        synthetic_straddle_history = base_straddle_price + np.random.normal(0, base_straddle_price * 0.1, n_days)
        synthetic_straddle_history = np.clip(synthetic_straddle_history, base_straddle_price * 0.5, base_straddle_price * 1.5) # Keep within a range
        df["Straddle_Price"] = synthetic_straddle_history
        # Override the last value with actual live data if available
        if latest_real_straddle_price is not None:
            df.loc[df.index[-1], "Straddle_Price"] = latest_real_straddle_price


        # Add Total Capital as a feature (constant across time in this DF)
        df["Total_Capital"] = capital

        # --- Final Data Cleaning and Validation ---

        # Ensure all defined FEATURE_COLS are in the DataFrame and are numeric where expected.
        # If missing, add them with NaN and attempt to fill.
        for col in FEATURE_COLS:
            if col not in df.columns:
                logger.warning(f"Adding missing FEATURE_COLS column: {col}")
                df[col] = np.nan

        # Ensure all FEATURE_COLS are numeric after all calculations/synthetic generation
        numeric_features = [col for col in FEATURE_COLS if col not in ['Event_Flag', 'Total_Capital']] # Identify cols expected to be numeric
        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Coerce to numeric, turns errors into NaN

        # Fill any remaining NaNs in FEATURE_COLS using interpolation and ffill/bfill
        initial_nan_count = df[FEATURE_COLS].isna().sum().sum()
        if initial_nan_count > 0:
            logger.warning(f"NaNs found in FEATURE_COLS before final fill: {initial_nan_count}")
            # Interpolate numeric columns first
            for col in numeric_features:
                 if col in df.columns:
                      df[col] = df[col].interpolate(method='linear')

            # Fill remaining NaNs using ffill then bfill for all FEATURE_COLS
            df[FEATURE_COLS] = df[FEATURE_COLS].fillna(method='ffill').fillna(method='bfill')

            # Re-check for NaNs after filling, if still present, log error
            final_nan_count = df[FEATURE_COLS].isna().sum().sum()
            if final_nan_count > 0:
                logger.error(f"FATAL ERROR: NaNs still present in FEATURE_COLS after final interpolation/fill: {final_nan_count}")
                # Display columns with persistent NaNs for debugging
                logger.error("Columns with persistent NaNs:")
                logger.error(df[FEATURE_COLS].isna().sum()[df[FEATURE_COLS].isna().sum() > 0])
                # Depending on strictness, you might return None here.
                # For this example, we'll return the df and log the error.


        logger.info(f"Features generated successfully. Final DataFrame shape: {df.shape}")
        # Return the DataFrame with all generated features
        return df

    except Exception as e:
        logger.error(f"Error generating features: {str(e)}", exc_info=True)
        return None

# Example of how to use these functions (for testing purposes, not part of the Streamlit app flow)
if __name__ == '__main__':
    # This block only runs if you execute this script directly
    print("Running data_processing.py as a standalone script (for testing).")

    # --- Test load_data (CSV fallback) ---
    print("\n--- Testing load_data (CSV Fallback) ---")
    # Simulate loading without an Upstox client
    test_df_csv, test_real_data_csv, test_source_csv = load_data(None)

    if test_df_csv is not None:
        print(f"Successfully loaded data from {test_source_csv}. Shape: {test_df_csv.shape}")
        print("\nHead of loaded DataFrame:")
        print(test_df_csv.head().to_string())
        print("\nTail of loaded DataFrame:")
        print(test_df_csv.tail().to_string())
    else:
        print(f"Failed to load data from {test_source_csv}.")

    # --- Test generate_features ---
    if test_df_csv is not None:
        print("\n--- Testing generate_features ---")
        # Create dummy real_data for testing feature generation
        # Simulate receiving data from fetch_real_time_market_data
        dummy_real_data = {
            'nifty_spot': test_df_csv['NIFTY_Close'].iloc[-1],
            'vix': test_df_csv['VIX'].iloc[-1] + 1.5, # Simulate a VIX increase
            'pcr': 1.18, # Dummy live PCR
            'straddle_price': 265.00, # Dummy live Straddle Price
            'max_pain_diff_pct': 0.75, # Dummy live Max Pain Diff
             # Approximate expiry date - should ideally come from load_data if client was used
            'expiry': (datetime.now().date() + timedelta(days=(3 - datetime.now().date().weekday()) % 7)).strftime("%Y-%m-%d") if datetime.now().date().weekday() != 3 else (datetime.now().date()).strftime("%Y-%m-%d"),
            'vix_change_pct': ((test_df_csv['VIX'].iloc[-1] + 1.5) / test_df_csv['VIX'].iloc[-1] - 1) * 100 if test_df_csv['VIX'].iloc[-1] != 0 else 0.0,
            'source': 'Dummy Real Data'
            # Add other potential keys from fetch_real_time_market_data if needed for testing
            # 'option_chain': pd.DataFrame({...}) # Dummy option chain
        }
        test_analysis_df = generate_features(test_df_csv.copy(), dummy_real_data, 1000000) # Use capital 1M

        if test_analysis_df is not None:
            print(f"Successfully generated features. Shape: {test_analysis_df.shape}")
            print("\nHead of Analysis DataFrame (with features):")
            print(test_analysis_df.head().to_string())
            print("\nTail of Analysis DataFrame (with features):")
            print(test_analysis_df.tail().to_string())
            print("\nColumns and their data types:")
            print(test_analysis_df.info())
            print("\nNaN counts per column:")
            print(test_analysis_df.isna().sum())
            # Verify FEATURE_COLS are present
            missing_features = [col for col in FEATURE_COLS if col not in test_analysis_df.columns]
            if missing_features:
                 print(f"\nWARNING: Missing FEATURES_COLS in generated df: {missing_features}")
            else:
                 print("\nAll FEATURE_COLS are present.")

            # Check if the last row was updated with dummy real data
            print("\nChecking last row against dummy real data:")
            last_row = test_analysis_df.iloc[-1]
            print(f"Last row VIX: {last_row['VIX']:.2f} (Dummy Real VIX: {dummy_real_data['vix']:.2f})")
            print(f"Last row PCR: {last_row['PCR']:.2f} (Dummy Real PCR: {dummy_real_data['pcr']:.2f})")
            print(f"Last row Straddle Price: {last_row['Straddle_Price']:.2f} (Dummy Real Straddle Price: {dummy_real_data['straddle_price']:.2f})")
            print(f"Last row Max Pain Diff %: {last_row['Spot_MaxPain_Diff_Pct']:.2f}% (Dummy Real Max Pain Diff %: {dummy_real_data['max_pain_diff_pct']:.2f}%)")
            print(f"Last row VIX Change %: {last_row['VIX_Change_Pct']:.2f}% (Dummy Real VIX Change %: {dummy_real_data['vix_change_pct']:.2f}%)")
            print(f"Last row DTE: {last_row['Days_to_Expiry']} (Dummy Expiry: {dummy_real_data['expiry']})")


        else:
            print("Failed to generate features.")
    else:
        print("Skipping feature generation test as data loading failed.")
