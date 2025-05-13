import pandas as pd
import re
import streamlit as st # This class is designed to work within a Streamlit app
import csv
import requests # Needed for API fallback in fetch_app_data
import numpy as np # Needed for np.sum in dynamic_ivp (although IVP calc is usually in data_processing, included here as it might be in context)
# Assuming upstox_api module exists and fetch_market_depth_by_scrip is available
# from upstox_api import fetch_market_depth_by_scrip # Imported below within the function if needed

import logging
logger = logging.getLogger(__name__)

class SmartBhaiGPT:
    """
    A simple rule-based chatbot for the VolGuard Pro app.
    Matches user queries against patterns and generates responses using app data.
    It is NOT a true large language model but uses predefined rules and data context.
    """
    def __init__(self, responses_file="responses.csv"):
        """
        Initializes the SmartBhaiGPT by loading response rules from a CSV file.

        Args:
            responses_file (str): Path to the CSV file containing query patterns,
                                  needed context keys, and response templates.

        Raises:
            FileNotFoundError: If the responses file is not found.
            ValueError: If the responses file has parsing errors or missing required columns.
            RuntimeError: For other unexpected errors during loading.
        """
        logger.info(f"Initializing SmartBhai GPT from {responses_file}")
        try:
            # Read the responses CSV file into a pandas DataFrame
            # Use quoting=csv.QUOTE_ALL to handle commas within quoted text fields
            # Use utf-8 encoding for broader character support
            self.responses = pd.read_csv(
                responses_file,
                quoting=csv.QUOTE_ALL, # Important if your CSV fields contain commas
                encoding='utf-8'
            )
            logger.info(f"Loaded {len(self.responses)} response rules from {responses_file}.")

            # Basic validation: Check if required columns are present in the loaded DataFrame
            required_cols = ["query_pattern", "context_needed", "response_template"]
            if not all(col in self.responses.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in self.responses.columns]
                logger.error(f"Responses file '{responses_file}' is missing required columns: {missing_cols}")
                raise ValueError(f"Responses file '{responses_file}' is missing required columns: {missing_cols}")

            # Log preview of loaded responses for debugging
            logger.debug("Responses file preview:")
            logger.debug(self.responses.head())


        except FileNotFoundError:
            # Re-raise FileNotFoundError with a user-friendly message
            logger.critical(f"Responses file not found: {responses_file}. SmartBhai GPT cannot be initialized.")
            raise FileNotFoundError(f"Bhai, SmartBhai GPT responses.csv file nahi mila! Check kar project folder mein. ({responses_file} not found)")
        except pd.errors.ParserError as e:
            # Catch pandas specific parsing errors
            logger.critical(f"Error parsing responses file {responses_file}: {str(e)}. Check CSV format.")
            raise ValueError(f"Bhai, responses.csv ka format galat hai! Error: {str(e)}")
        except Exception as e:
            # Catch any other unexpected errors during the loading process
            logger.critical(f"An unexpected error occurred while loading responses file {responses_file}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Bhai, responses.csv load karte waqt kuch gadbad ho gayi: {str(e)}")


    def fetch_app_data(self, context_needed_keys: str):
        """
        Fetches relevant real-time data from the Streamlit app's session state
        or fallbacks to provide context for generating chatbot responses.
        Prioritizes session state data. Includes fallbacks to limited API calls
        (for Nifty/VIX) or a static CSV file if session state data is incomplete.

        Args:
            context_needed_keys (str): A comma-separated string of data keys required
                                       for the response template (e.g., "iv,vix,pnl,strategy").
                                       These keys correspond to the keys in the returned dictionary.

        Returns:
            dict: A dictionary containing the fetched data, with keys matching
                  the requested `context_needed_keys`. Values will be "N/A"
                  or a sensible default if data cannot be fetched from any source.
                  All requested keys will be present in the returned dictionary.
        """
        logger.debug(f"Fetching app data for context keys: {context_needed_keys}")
        # Initialize a dictionary to store the fetched context data.
        # Populate it with default "N/A" values for all requested keys first.
        required_keys = [key.strip() for key in context_needed_keys.split(",") if key.strip()]
        data = {key: "N/A" for key in required_keys} # Initialize with defaults


        # --- Attempt to fetch data from Streamlit session state (Primary Source) ---
        # This is the most comprehensive source as it includes processed analysis data,
        # real-time API data, and portfolio data stored by streamlit_app.
        try:
            logger.debug("Attempting to fetch data from session state.")
            session_state_data_found = False # Flag to track if we successfully got data from state

            # Get core data structures from session state safely
            analysis_df = st.session_state.get("analysis_df")
            real_time_data = st.session_state.get("real_time_market_data", {})
            portfolio_data = st.session_state.get("api_portfolio_data", {})
            generated_strategy = st.session_state.get("generated_strategy", {})
            capital = pd.to_numeric(st.session_state.get("capital"), errors='coerce').fillna(1000000) # Get capital safely

            # If analysis_df exists and is not empty, we can extract many features from its last row
            if analysis_df is not None and not analysis_df.empty:
                latest_analysis_data = analysis_df.iloc[-1]
                logger.debug("Analysis DataFrame found in session state.")
                session_state_data_found = True # We have at least analysis data

                # Extract values from the last row of the analysis DataFrame
                # Use .get() with default fallback for safety
                data["iv"] = pd.to_numeric(latest_analysis_data.get("ATM_IV"), errors='coerce').fillna("N/A")
                # Use Gamma_Bias as a proxy for Gamma/Delta if needed by response templates
                data["gamma"] = pd.to_numeric(latest_analysis_data.get("Gamma_Bias"), errors='coerce').fillna("N/A")
                data["delta"] = pd.to_numeric(latest_analysis_data.get("Gamma_Bias"), errors='coerce').fillna("N/A") # Simplified proxy
                # Theta/Vega are typically specific to individual contracts, not overall features
                # They might be available in the option_chain df within real_time_data
                # For now, default to N/A or get from real_time_data if possible below
                data["theta"] = "N/A" # Placeholder
                data["vega"] = "N/A" # Placeholder

                data["vix"] = pd.to_numeric(latest_analysis_data.get("VIX"), errors='coerce').fillna("N/A")
                data["ivp"] = pd.to_numeric(latest_analysis_data.get("IVP"), errors='coerce').fillna("N/A")
                # Determine IVP status based on the numeric IVP value
                ivp_numeric = pd.to_numeric(data["ivp"], errors='coerce')
                if not pd.isna(ivp_numeric):
                    data["ivp_status"] = "high" if ivp_numeric > 70 else ("low" if ivp_numeric < 30 else "medium")
                else:
                    data["ivp_status"] = "N/A" # Cannot determine status if IVP is not numeric

                data["dte"] = pd.to_numeric(latest_analysis_data.get("Days_to_Expiry"), errors='coerce').fillna("N/A")

                # Incorporate other synthetic/placeholder data from analysis_df
                data["fii_flow"] = pd.to_numeric(latest_analysis_data.get("FII_Index_Fut_Pos"), errors='coerce').fillna("N/A")
                data["dii_flow"] = pd.to_numeric(latest_analysis_data.get("FII_Option_Pos"), errors='coerce').fillna("N/A") # DII proxy
                # Crude sentiment proxy from synthetic Capital_Pressure_Index
                capital_pressure = pd.to_numeric(latest_analysis_data.get("Capital_Pressure_Index"), errors='coerce').fillna(0)
                data["sentiment"] = int(np.clip(capital_pressure * -10 + 50, 0, 100)) # Scale and clip to 0-100 range

                # Proxy for impact from VIX change percentage
                vix_change_pct = pd.to_numeric(latest_analysis_data.get("VIX_Change_Pct"), errors='coerce').fillna(0)
                data["impact"] = int(np.clip(abs(vix_change_pct) * 5, 0, 100)) # Scale absolute change, clip to 0-100

                # Proxy for potential loss risk from synthetic PnL_Day relative to capital
                pnl_day = pd.to_numeric(latest_analysis_data.get("PnL_Day"), errors='coerce').fillna(0.0)
                # Example calculation: 2x the magnitude of a typical day's PnL as % of capital
                data["loss_risk"] = abs(pnl_day) / capital * 100.0 * 2.0 if capital > 0 else 0.0


            # If real_time_data is available (fetched from API), it contains the latest market data
            if real_time_data and isinstance(real_time_data, dict):
                 logger.debug("Real-time data found in session state.")
                 session_state_data_found = True # We have at least real-time data

                 # Override/add metrics from real_time_data (more accurate for the latest point)
                 data["nifty_spot"] = pd.to_numeric(real_time_data.get("nifty_spot"), errors='coerce').fillna(data.get("nifty_spot", "N/A")) # Use analysis_df spot if real_data fails
                 data["vix"] = pd.to_numeric(real_time_data.get("vix"), errors='coerce').fillna(data.get("vix", "N/A")) # Use analysis_df vix if real_data fails
                 data["pcr"] = pd.to_numeric(real_time_data.get("pcr"), errors='coerce').fillna(data.get("pcr", "N/A")) # Use analysis_df pcr (synthetic) if real_data fails
                 data["straddle_price"] = pd.to_numeric(real_time_data.get("straddle_price"), errors='coerce').fillna(data.get("straddle_price", "N/A")) # Use analysis_df straddle (synthetic) if real_data fails
                 data["max_pain_diff_pct"] = pd.to_numeric(real_time_data.get("max_pain_diff_pct"), errors='coerce').fillna(data.get("max_pain_diff_pct", "N/A")) # Use analysis_df maxpain (synthetic) if real_data fails
                 data["atm_strike"] = pd.to_numeric(real_time_data.get("atm_strike"), errors='coerce').fillna(data.get("atm_strike", "N/A")) # Use derived atm_strike if real_data fails

                 # Try to get theta/vega from the ATM option in the option chain if available in real_time_data
                 option_chain_df = real_time_data.get("option_chain")
                 atm_strike_real = data.get("atm_strike") # Use the potentially updated ATM strike
                 if isinstance(option_chain_df, pd.DataFrame) and not option_chain_df.empty and atm_strike_real is not None and pd.notna(atm_strike_real):
                      atm_row_ce = option_chain_df[(option_chain_df['StrikeRate'] == atm_strike_real) & (option_chain_df['CPType'] == 'CE')].iloc[0] if not option_chain_df[(option_chain_df['StrikeRate'] == atm_strike_real) & (option_chain_df['CPType'] == 'CE')].empty else {}
                      atm_row_pe = option_chain_df[(option_chain_df['StrikeRate'] == atm_strike_real) & (option_chain_df['CPType'] == 'PE')].iloc[0] if not option_chain_df[(option_chain_df['StrikeRate'] == atm_strike_real) & (option_chain_df['CPType'] == 'PE')].empty else {}
                      # Average Theta/Vega from ATM CE and PE if available
                      ce_theta = pd.to_numeric(atm_row_ce.get("Theta"), errors='coerce').fillna(0.0)
                      pe_theta = pd.to_numeric(atm_row_pe.get("Theta"), errors='coerce').fillna(0.0)
                      ce_vega = pd.to_numeric(atm_row_ce.get("Vega"), errors='coerce').fillna(0.0)
                      pe_vega = pd.to_numeric(atm_row_pe.get("Vega"), errors='coerce').fillna(0.0)

                      # Average only if both are valid numbers
                      data["theta"] = (ce_theta + pe_theta) / 2.0 if pd.notna(ce_theta) and pd.notna(pe_theta) else (ce_theta if pd.notna(ce_theta) else (pe_theta if pd.notna(pe_theta) else "N/A"))
                      data["vega"] = (ce_vega + pe_vega) / 2.0 if pd.notna(ce_vega) and pd.notna(pe_vega) else (ce_vega if pd.notna(ce_vega) else (pe_vega if pd.notna(pe_vega) else "N/A"))


            # If api_portfolio_data is available, extract relevant metrics
            if portfolio_data and isinstance(portfolio_data, dict):
                logger.debug("Portfolio data found in session state.")
                session_state_data_found = True # We have at least portfolio data

                margin_data = portfolio_data.get("margin", {}).get("data", {})
                # Ensure margin_data is a dict before accessing keys
                if isinstance(margin_data, dict):
                     available_margin = pd.to_numeric(margin_data.get("available_margin"), errors='coerce').fillna(0.0)
                     # Calculate available margin percentage
                     data["margin"] = (available_margin / capital * 100.0) if capital > 0 else 0.0
                else:
                     logger.warning("Margin data not in expected dictionary format in portfolio_data.")
                     data["margin"] = "N/A" # Default if format is wrong


                trade_book_list = portfolio_data.get("trade_book", {}).get("data", [])
                # Ensure trade_book_list is a list
                if isinstance(trade_book_list, list):
                     data["trade_count"] = len(trade_book_list)
                     # Approximate trade frequency (e.g., trades per week) - simplified
                     # Assumes the trade_book covers a recent period (e.g., a week)
                     data["trade_frequency"] = data["trade_count"] / 7.0 if data["trade_count"] > 0 else 0.0 # Simple approximation
                else:
                     logger.warning("Trade book data not in expected list format in portfolio_data.")
                     data["trade_count"] = "N/A"
                     data["trade_frequency"] = "N/A"


                # Calculate current PnL percentage from positions
                positions_list = portfolio_data.get("positions", {}).get("data", [])
                # Ensure positions_list is a list
                if isinstance(positions_list, list):
                     # Sum current PnL (using 'unrealized_mtm' + 'realized_profit' as in portfolio summary)
                     current_pnl_total = sum(
                         pd.to_numeric(pos.get("unrealized_mtm"), errors='coerce').fillna(0.0) +
                         pd.to_numeric(pos.get("realized_profit"), errors='coerce').fillna(0.0)
                         for pos in positions_list if isinstance(pos, dict)
                     )
                     data["pnl"] = (current_pnl_total / capital * 100.0) if capital > 0 else 0.0
                else:
                     logger.warning("Positions data not in expected list format in portfolio_data.")
                     data["pnl"] = "N/A"


            # If generated_strategy is available, extract details
            if generated_strategy and isinstance(generated_strategy, dict):
                 logger.debug("Generated strategy data found in session state.")
                 session_state_data_found = True # We have at least strategy data
                 data["strategy"] = generated_strategy.get("Strategy", "No strategy generated yet")
                 data["reason"] = generated_strategy.get("Reasoning", "N/A") # Reason for the strategy


            # Add placeholder data for buzz, news, community, revenge, bias, win rate, breakeven if not derived
            # These are often synthetic or external data not explicitly in the core data fetch.
            # Ensure these keys are added if requested but not found elsewhere.
            if data.get("buzz_topic", "N/A") == "N/A": data["buzz_topic"] = "Market volatility" # Placeholder
            if data.get("news_headline", "N/A") == "N/A": data["news_headline"] = "NIFTY holds firm" # Placeholder
            if data.get("community_hedge", "N/A") == "N/A": data["community_hedge"] = 70 # Placeholder community hedging level
            if data.get("revenge_count", "N/A") == "N/A": data["revenge_count"] = pd.to_numeric(st.session_state.get("violations", 0), errors='coerce').fillna(0) # Using 'violations' as proxy
            if data.get("bias", "N/A") == "N/A": data["bias"] = "Neutral" # Placeholder trader bias
            if data.get("bias_count", "N/A") == "N/A": data["bias_count"] = 0 # Placeholder bias count
            if data.get("win_rate", "N/A") == "N/A": data["win_rate"] = 60 # Placeholder win rate
            if data.get("breakeven", "N/A") == "N/A": data["breakeven"] = "N/A" # Placeholder breakeven

            # After attempting to fetch from session state, check if we got any data.
            if session_state_data_found:
                 logger.debug("Data fetching from session state completed.")
                 # Ensure all required keys are populated, use "N/A" for those still missing
                 for key in required_keys:
                      if key not in data or data[key] is None or (isinstance(data[key], str) and data[key].strip() == ""):
                          data[key] = "N/A"
                 # Return the data dictionary
                 return data

            else:
                 # If no data was found in session state (e.g., analysis_df is None, real_time_data is empty)
                 logger.warning("No significant data found in session state. Attempting API fallback.")
                 # Proceed to the next fallback source

        except Exception as e:
            # Log any unexpected errors during the session state fetch process
            logger.warning(f"Error fetching data from session state: {str(e)}. Proceeding to API fallback.", exc_info=True)
            # data dictionary might be partially filled, but we will proceed to fallbacks


        # --- Attempt to fetch data using limited API calls (Fallback 1) ---
        # This fallback is limited as it cannot get all historical/complex metrics or full portfolio.
        # It can primarily fetch live Nifty and VIX using basic quotes endpoint via requests.
        try:
            logger.debug("Attempting API fallback for data fetching.")
            # Ensure the Upstox client with access token is available in session state
            upstox_client = st.session_state.get("client")
            if upstox_client and upstox_client.get("access_token"):
                 # Import necessary function inside the try block to avoid import errors if upstox_api is missing
                 # from upstox_api import fetch_market_depth_by_scrip # Not directly used in this specific API fallback part, fetch quotes instead
                 # import requests # Already imported at the top

                 headers_fallback = {"Authorization": f"Bearer {upstox_client.get('access_token')}", "Content-Type": "application/json"}
                 base_url_fallback = "https://api.upstox.com/v2"
                 nifty_index_key_fallback = "NSE_INDEX|Nifty 50"
                 vix_index_key_fallback = "NSE_INDEX|India VIX"

                 # Attempt to fetch Nifty Spot and India VIX using basic quotes API via requests
                 try:
                     # Combine instrument keys for a single quotes request if possible
                     quotes_url_fallback = f"{base_url_fallback}/market-quote/quotes"
                     quotes_params_fallback = {"instrument_key": f"{nifty_index_key_fallback},{vix_index_key_fallback}"}
                     quotes_res_fallback = requests.get(quotes_url_fallback, headers=headers_fallback, params=quotes_params_fallback)
                     quotes_res_fallback.raise_for_status() # Raise HTTPError for bad status codes

                     quotes_data_fallback = quotes_res_fallback.json().get('data', {})

                     # Extract Nifty Spot and VIX safely
                     nifty_spot_fallback = quotes_data_fallback.get(nifty_index_key_fallback, {}).get('last_price')
                     vix_fallback = quotes_data_fallback.get(vix_index_key_fallback, {}).get('last_price')

                     # Update data dictionary with fetched values, converting to numeric where applicable
                     data["nifty_spot"] = pd.to_numeric(nifty_spot_fallback, errors='coerce').fillna(data.get("nifty_spot", "N/A"))
                     data["vix"] = pd.to_numeric(vix_fallback, errors='coerce').fillna(data.get("vix", "N/A"))

                     logger.debug("Data fetched using limited API fallback (Nifty/VIX).")

                 except requests.exceptions.RequestException as req_e:
                      logger.warning(f"Network/request error during limited API fallback: {req_e}. Cannot fetch Nifty/VIX.")
                      # Data dictionary remains as it was (potentially "N/A" or partial from session state)
                 except Exception as e:
                      logger.warning(f"An unexpected error occurred during limited API fallback: {str(e)}. Cannot fetch Nifty/VIX.")
                      # Data dictionary remains as it was


                 # We still need to ensure all required_keys are populated.
                 # Keys not fetched by this limited API fallback (like PCR, Straddle, etc.)
                 # will remain as "N/A" or whatever was potentially fetched from session state earlier.
                 # We will fill any still missing keys with default placeholders below.

                 # Proceed to the next fallback source for any remaining missing keys.
                 # The 'data' dict is now partially filled from the API fallback attempt.
                 logger.debug("API fallback completed. Proceeding to static CSV fallback if needed.")
                 pass # Do nothing here, proceed to the static CSV fallback block below


            else:
                 logger.warning("API client not available in session state for API fallback.")
                 # If client is not available, proceed directly to static CSV fallback
                 pass # Do nothing, proceed to the static CSV fallback block below


        except Exception as e:
            # Catch any unexpected errors during the API fallback setup/attempt
            logger.warning(f"An unexpected error occurred during API fallback attempt: {str(e)}. Proceeding to static CSV fallback.", exc_info=True)
            pass # Proceed to static CSV fallback


        # --- Attempt to fetch data from static fallback_data.csv (Fallback 2) ---
        # This CSV is assumed to contain a row of default or last known values.
        try:
            logger.debug("Attempting static CSV fallback for data fetching.")
            # Assuming 'fallback_data.csv' exists and has a header row matching context keys.
            fallback_df = pd.read_csv("fallback_data.csv", encoding='utf-8')

            if not fallback_df.empty:
                # Get the last row from the fallback CSV as a dictionary
                latest_fallback_data = fallback_df.iloc[-1].to_dict()

                # Update the data dictionary with values from the static CSV fallback.
                # This will override any "N/A" values that were not fetched from session state or API fallback.
                # Iterate through required keys and fill them from the fallback CSV if they are still "N/A".
                for key in required_keys:
                    # Check if the key is still "N/A" in the data dictionary
                    if data.get(key, "N/A") == "N/A":
                        # Attempt to get the value from the static CSV fallback data
                        fallback_value = latest_fallback_data.get(key)
                        if fallback_value is not None:
                             # Convert fallback value to appropriate type or keep as string if needed
                             # For simplicity, keeping as string here, but could add type-specific conversion
                             data[key] = str(fallback_value)
                             logger.debug(f"Filled key '{key}' from static CSV fallback.")


                logger.debug("Data fetching from static CSV fallback completed.")
                # After attempting static CSV fallback, all required_keys should be populated if available in CSV.
                # We will perform the final fill with hardcoded defaults if they are STILL missing below.
                pass # Proceed to the final hardcoded fallback

            else:
                 logger.warning("Static CSV fallback file 'fallback_data.csv' is empty. Cannot use for fallback.")
                 pass # Proceed to the final hardcoded fallback

        except FileNotFoundError:
             logger.warning("Static CSV fallback file not found: fallback_data.csv. Cannot use for fallback.")
             pass # Proceed to the final hardcoded fallback
        except Exception as e:
            logger.warning(f"An unexpected error occurred fetching data from static CSV fallback: {str(e)}. Proceeding to final fallback.", exc_info=True)
            pass # Proceed to the final hardcoded fallback


        # --- Final Hardcoded Fallback: Ensure all required keys have a value ---
        # If after trying session state, API fallback, and static CSV fallback,
        # any required key is still "N/A", fill it with a hardcoded default or keep "N/A".
        # This ensures the dictionary returned always contains all requested keys
        # with some value, preventing KeyError during response formatting.
        logger.debug("Attempting final hardcoded fallback for any remaining missing keys.")
        # Define a dictionary of hardcoded default values for common keys
        hardcoded_defaults = {
            "iv": 25.0, "gamma": 0.0, "delta": 0.0, "theta": 0.0, "vega": 0.0,
            "vix": 15.0, "ivp": 50.0, "ivp_status": "medium", "margin": 100.0, "strike": 20000.0,
            "fii_flow": 0, "dii_flow": 0, "buzz_topic": "Market update", "sentiment": 50,
            "news_headline": "No news available", "impact": 0, "community_hedge": 50,
            "trade_count": 0, "trade_frequency": 0, "revenge_count": 0,
            "bias": "Neutral", "bias_count": 0, "strategy": "None",
            "win_rate": 50, "reason": "No data", "dte": 7, "pnl": 0.0,
            "loss_risk": 0.0, "breakeven": "N/A", "nifty_spot": 20000.0, "pcr": 1.0,
            "straddle_price": 200.0, "max_pain_diff_pct": 0.0, "atm_strike": 20000.0
        }

        # Iterate through the requested keys and fill them with hardcoded defaults if they are still "N/A"
        for key in required_keys:
             if data.get(key, "N/A") == "N/A":
                 # Use the hardcoded default if available, otherwise keep "N/A"
                 data[key] = hardcoded_defaults.get(key, "N/A")
                 logger.debug(f"Filled key '{key}' with hardcoded default.")

        # Log the final state of the fetched data dictionary
        logger.debug("Final fetched data dictionary for context:")
        logger.debug(data)

        # Return the final data dictionary, ensuring all requested keys have a value.
        return data


    def generate_response(self, user_query: str):
        """
        Generates a response for the user query by matching patterns loaded from CSV
        and formatting response templates using fetched app data context.

        Args:
            user_query (str): The input query string from the user.

        Returns:
            str: The generated response string. Returns a fallback message if
                 no pattern matches or an error occurs during response generation.
        """
        # Convert user query to lowercase and strip whitespace for case-insensitive matching
        user_query_lower = user_query.lower().strip()
        logger.info(f"Generating response for query: '{user_query_lower}'")

        # Ensure responses DataFrame was loaded successfully during initialization
        if self.responses is None or self.responses.empty:
             logger.error("SmartBhai GPT responses not loaded. Cannot generate response.")
             return "Bhai, SmartBhai GPT responses load nahi hue. Main abhi baat nahi kar sakta. 😟"


        # Iterate through each response rule (row) loaded from the responses DataFrame
        for index, row in self.responses.iterrows():
            # Safely get pattern, context keys needed, and response template from the row
            # Handle potential missing cells in the CSV by defaulting to empty strings
            pattern = row.get("query_pattern", "")
            context_needed_keys = row.get("context_needed", "")
            response_template = row.get("response_template", "Bhai, kuch gadbad hai response template mein. 😕") # Default template on error


            # Skip this rule if the query pattern is empty
            if not pattern.strip():
                 logger.warning(f"Skipping response rule at index {index} due to empty query_pattern.")
                 continue # Move to the next rule


            # Check if the user query matches the current pattern using regular expressions
            try:
                # re.search looks for the pattern anywhere within the string
                if re.search(pattern, user_query_lower):
                    logger.debug(f"Query matched pattern: '{pattern}' (Index: {index})")

                    # If a match is found, fetch the necessary app data context
                    # Pass the comma-separated string of keys needed
                    context = self.fetch_app_data(context_needed_keys)
                    # logger.debug(f"Fetched context for formatting: {context}") # Too verbose maybe

                    try:
                        # Format the response template using the fetched context data dictionary.
                        # Use the .format(**context) syntax which unpacks the dictionary into keyword arguments.
                        # Handle potential KeyError if a key used in the response_template's format string
                        # is not present in the 'context' dictionary (which shouldn't happen if fetch_app_data
                        # guarantees all requested keys, but included for robustness).
                        # Convert context values to strings before formatting to prevent type issues.
                        context_str = {k: str(v) for k, v in context.items()}
                        response = response_template.format(**context_str)

                        logger.info(f"Generated response for matched pattern: '{response}'")
                        return response # Return the generated response immediately after finding the first match

                    except KeyError as ke:
                        logger.error(f"KeyError formatting response for pattern '{pattern}' (Index {index}): Missing key '{ke}' in context.")
                        # Return a specific error message to the user if a required key for formatting is missing
                        return f"Bhai, data thoda off lag raha hai ya response template mein gadbad hai. Try again! Do your own research! (Error: Missing Data Key: {ke})"
                    except Exception as e:
                         # Catch any other unexpected errors during the formatting process
                         logger.error(f"An unexpected error occurred formatting response for pattern '{pattern}' (Index {index}): {str(e)}", exc_info=True)
                         return "Bhai, response generate karte waqt kuch technical issue ho gaya. Try again! Do your own research! (Error: Formatting Failed)"

            except re.error as re_e:
                 # Catch errors if the regex pattern itself is invalid
                 logger.error(f"Invalid regex pattern at index {index}: '{pattern}'. Error: {re_e}")
                 # Log the error but continue to the next pattern to avoid crashing
                 continue # Move to the next rule if the current pattern is invalid

            except Exception as e:
                # Catch any other unexpected errors that might occur during pattern matching or initial data fetching setup for this rule
                logger.error(f"An unexpected error occurred during pattern matching or data fetching setup for pattern '{pattern}' (Index {index}): {str(e)}", exc_info=True)
                # Log the error but continue to the next pattern
                continue


        # If the loop finishes without finding any matching pattern for the user query
        logger.info(f"No pattern matched query: '{user_query_lower}'.")
        # Return a default fallback response
        return "Bhai, yeh query thodi alag hai. Kya bol raha hai, thoda clearly bata? 🤔 Do your own research!"

# The __main__ block is removed because this class is designed to be run
# within the Streamlit app and directly accesses st.session_state,
# which is not available when running this script standalone.
# To test this class independently, you would typically need to mock
# the st.session_state object and potentially the upstox_api functions.
