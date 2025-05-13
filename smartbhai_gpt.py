import pandas as pd
import re
import streamlit as st # This class is designed to work within a Streamlit app
import csv
# Assuming upstox_api module exists and fetch_market_depth_by_scrip is available
# from upstox_api import fetch_market_depth_by_scrip # Imported below within the function

import logging
logger = logging.getLogger(__name__)

class SmartBhaiGPT:
    """
    A simple rule-based chatbot for the VolGuard Pro app.
    Matches user queries against patterns and generates responses using app data.
    """
    def __init__(self, responses_file="responses.csv"):
        """
        Initializes the SmartBhaiGPT by loading response rules from a CSV file.

        Args:
            responses_file (str): Path to the CSV file containing query patterns and response templates.
        """
        logger.info(f"Initializing SmartBhai GPT from {responses_file}")
        try:
            # Read the responses CSV file into a pandas DataFrame
            # Use quoting=csv.QUOTE_ALL to handle commas within quoted text fields
            # Use utf-8 encoding
            self.responses = pd.read_csv(
                responses_file,
                quoting=csv.QUOTE_ALL,
                encoding='utf-8'
            )
            logger.info(f"Loaded {len(self.responses)} response rules.")
            # Basic validation of columns
            if not all(col in self.responses.columns for col in ["query_pattern", "context_needed", "response_template"]):
                logger.error(f"Missing required columns in {responses_file}")
                raise ValueError(f"Responses file '{responses_file}' is missing required columns.")

        except FileNotFoundError:
            logger.critical(f"Responses file not found: {responses_file}")
            raise FileNotFoundError(f"Bhai, responses.csv nahi mila! Check kar project folder mein. ({responses_file} not found)")
        except pd.errors.ParserError as e:
            logger.critical(f"Error parsing responses file {responses_file}: {str(e)}")
            raise ValueError(f"Bhai, responses.csv ka format galat hai! Error: {str(e)}")
        except Exception as e:
            logger.critical(f"Unexpected error loading responses file {responses_file}: {str(e)}")
            raise RuntimeError(f"Bhai, responses.csv load karte waqt kuch gadbad ho gayi: {str(e)}")


    def fetch_app_data(self, context_needed_keys):
        """
        Fetches relevant real-time data from the Streamlit app's session state
        or fallbacks to provide context for responses.

        Args:
            context_needed_keys (str): A comma-separated string of data keys needed
                                       for the response template (e.g., "iv,vix,pnl").

        Returns:
            dict: A dictionary containing the fetched data, with keys matching
                  context_needed_keys. Defaults to "N/A" if data cannot be fetched.
        """
        logger.debug(f"Fetching app data for context: {context_needed_keys}")
        data = {} # Dictionary to store the fetched context data
        # Split the comma-separated string into a list of required keys
        required_keys = [key.strip() for key in context_needed_keys.split(",") if key.strip()]

        # --- Attempt to fetch data from Streamlit session state (Primary Source) ---
        try:
            # Access data from the main analysis DataFrame
            df = st.session_state.get("analysis_df")
            if df is not None and not df.empty:
                latest_data = df.iloc[-1]
                # Populate data dict with values from the latest analysis data
                data["iv"] = latest_data.get("ATM_IV", "N/A")
                data["gamma"] = latest_data.get("Gamma_Bias", "N/A") # Using Gamma_Bias as a proxy for Gamma
                data["delta"] = latest_data.get("Gamma_Bias", "N/A") # Using Gamma_Bias as a proxy for Delta (simplified)
                data["theta"] = "N/A" # Theta is not directly in analysis_df features, needs specific option data
                data["vega"] = "N/A" # Vega is not directly in analysis_df features, needs specific option data
                data["vix"] = latest_data.get("VIX", "N/A")
                data["ivp"] = latest_data.get("IVP", "N/A")
                # Determine IVP status based on IVP value
                data["ivp_status"] = "high" if pd.to_numeric(data["ivp"], errors='coerce') > 70 else "low" if pd.to_numeric(data["ivp"], errors='coerce') < 30 else "medium" # Handle non-numeric IVP

                # Access data from the real-time market data dictionary
                real_time_data = st.session_state.get("real_time_market_data", {})
                data["strike"] = real_time_data.get("atm_strike", "N/A")
                data["pcr"] = real_time_data.get("pcr", "N/A")
                data["dte"] = latest_data.get("Days_to_Expiry", "N/A") # Get DTE from analysis_df

                # Access data from the API portfolio data
                portfolio_data = st.session_state.get("api_portfolio_data", {})
                # Calculate available margin percentage based on capital
                capital = st.session_state.get("capital", 1000000) # Get capital from session state
                available_margin = portfolio_data.get("margin", {}).get("data", {}).get("available_margin", 0.0)
                data["margin"] = (available_margin / capital * 100) if capital > 0 else 0.0
                # Get trade count from the trade book
                data["trade_count"] = len(portfolio_data.get("trade_book", {}).get("data", []))
                # Approximate trade frequency (e.g., trades per week)
                data["trade_frequency"] = data["trade_count"] / 7 if data["trade_count"] > 0 else 0 # Simple approximation
                # Calculate current PnL percentage based on capital
                # Sum unrealized_mtm from positions
                positions = portfolio_data.get("positions", {}).get("data", [])
                current_pnl = sum(pos.get("unrealized_mtm", 0.0) for pos in positions if isinstance(pos, dict))
                data["pnl"] = (current_pnl / capital * 100) if capital > 0 else 0.0

                # Get generated strategy details
                generated_strategy = st.session_state.get("generated_strategy", {})
                data["strategy"] = generated_strategy.get("Strategy", "No strategy generated yet")
                data["reason"] = generated_strategy.get("Reasoning", "N/A") # Reason for the strategy

                # --- Incorporate other potentially synthetic/placeholder data ---
                # These might be present in the analysis_df or need synthetic generation if not
                data["fii_flow"] = latest_data.get("FII_Index_Fut_Pos", "N/A") # FII from analysis_df (synthetic historical)
                data["dii_flow"] = latest_data.get("FII_Option_Pos", "N/A") # DII proxy from analysis_df (synthetic historical)
                data["sentiment"] = latest_data.get("Capital_Pressure_Index", 0) * -10 + 50 # Crude sentiment proxy from synthetic index
                data["buzz_topic"] = "Market volatility" # Placeholder
                data["news_headline"] = "NIFTY holds firm" # Placeholder
                data["impact"] = abs(pd.to_numeric(latest_data.get("VIX_Change_Pct", 0), errors='coerce').fillna(0)) * 5 # Proxy impact from VIX change
                data["community_hedge"] = 70 # Placeholder community hedging level
                data["revenge_count"] = st.session_state.get("violations", 0) # Using 'violations' as a proxy for 'revenge_count'
                data["bias"] = "Neutral" # Placeholder trader bias
                data["bias_count"] = 0 # Placeholder bias count
                data["win_rate"] = 60 # Placeholder win rate
                data["loss_risk"] = abs(pd.to_numeric(latest_data.get("PnL_Day", 0), errors='coerce').fillna(0)) / capital * 100 * 2 # Proxy loss risk from synthetic PnL
                data["breakeven"] = "N/A" # Placeholder breakeven

                logger.debug("Data fetched from session state.")
                # Data is ready, no need for fallbacks in this path for the keys that were successfully retrieved
                # Fill in any remaining required_keys that were not found in session state with "N/A"
                for key in required_keys:
                    if key not in data:
                        data[key] = "N/A"
                return data # Return data fetched from session state


        except Exception as e:
            logger.warning(f"Error fetching data from session state: {str(e)}")
            # If session state data fetch fails, proceed to next fallback source
            pass # Do nothing, error is logged, proceed to fallback


        # --- Attempt to fetch data using limited API calls (Fallback 1) ---
        # This fallback is limited as it cannot get all historical/complex metrics
        try:
            logger.warning("Session state data fetch failed. Attempting API fallback.")
            upstox_client = st.session_state.get("client")
            if upstox_client and upstox_client.get("access_token"):
                 # Import inside the try block to avoid import errors if upstox_api is missing
                 from upstox_api import fetch_market_depth_by_scrip

                 # Attempt to fetch a few key metrics using basic API calls
                 # This requires knowing instrument keys, e.g., for an ATM option
                 # This is complex without the full option chain, so just fetch Nifty/VIX if possible
                 headers = {"Authorization": f"Bearer {upstox_client.get('access_token')}", "Content-Type": "application/json"}
                 base_url = "https://api.upstox.com/v2"

                 # Fetch Nifty Spot
                 try:
                     nifty_res = requests.get(f"{base_url}/market-quote/quotes", headers=headers, params={"instrument_key": "NSE_INDEX|Nifty 50"})
                     nifty_res.raise_for_status()
                     nifty_spot_data = nifty_res.json().get('data', {}).get('NSE_INDEX|Nifty 50', {})
                     data["nifty_spot"] = nifty_spot_data.get('last_price', "N/A")
                 except Exception:
                     data["nifty_spot"] = "N/A"

                 # Fetch India VIX
                 try:
                     vix_res = requests.get(f"{base_url}/market-quote/quotes", headers=headers, params={"instrument_key": "NSE_INDEX|India VIX"})
                     vix_res.raise_for_status()
                     vix_data = vix_res.json().get('data', {}).get('NSE_INDEX|India VIX', {})
                     data["vix"] = vix_data.get('last_price', "N/A")
                 except Exception:
                     data["vix"] = "N/A"

                 # For other required keys, we might not be able to get real values via simple API calls
                 # We will fill them with placeholders or default values below

                 logger.debug("Data fetched using API fallback.")
                 # Fill in any remaining required_keys that were not found in API fallback with "N/A"
                 for key in required_keys:
                     if key not in data:
                         data[key] = "N/A"

                 # We still need to fill in the context keys needed that weren't fetched by the limited API calls
                 # We can fill these with default/placeholder values here
                 fallback_defaults = {
                     "iv": 30.0, "gamma": 0.05, "delta": 0.4, "theta": -0.02, "vega": 0.1,
                     "ivp": 75.0, "ivp_status": "high", "margin": 85.0, "strike": "N/A",
                     "fii_flow": 0.0, "dii_flow": 0.0, "buzz_topic": "NIFTY expiry", "sentiment": 60,
                     "news_headline": "No major news", "impact": 50, "community_hedge": 70,
                     "trade_count": 0, "trade_frequency": 0, "revenge_count": 0,
                     "bias": "Neutral", "bias_count": 0, "strategy": "Iron Condor",
                     "win_rate": 60, "reason": "IV spike", "dte": 5, "pnl": 0.0,
                     "loss_risk": 15.0, "breakeven": "N/A"
                 }
                 for key in required_keys:
                      if key not in data or data[key] == "N/A": # Only fill if key is missing or value is N/A
                          data[key] = fallback_defaults.get(key, "N/A") # Use specific default or "N/A"


                 return data # Return data fetched (partially) from API and filled with defaults


            else:
                 logger.warning("API client not available for API fallback.")
                 # If client is not available, proceed to static CSV fallback
                 pass # Do nothing, warning is logged, proceed to fallback


        except Exception as e:
            logger.warning(f"Error fetching data using API fallback: {str(e)}")
            # If API fallback fails, proceed to static CSV fallback
            pass # Do nothing, error is logged, proceed to fallback


        # --- Attempt to fetch data from static fallback_data.csv (Fallback 2) ---
        try:
            logger.warning("API fallback failed. Attempting static CSV fallback.")
            fallback_df = pd.read_csv("fallback_data.csv", encoding='utf-8')
            if not fallback_df.empty:
                latest_fallback = fallback_df.iloc[-1]
                # Populate data dict with values from the latest fallback CSV data
                # Convert needed keys to strings before .get() if CSV stores them as numbers
                data = {key: str(latest_fallback.get(key, "N/A")) for key in required_keys} # Get all required keys

                logger.debug("Data fetched from static CSV fallback.")
                # Fill in any remaining required_keys that were not found in static CSV with "N/A"
                for key in required_keys:
                    if key not in data:
                         data[key] = "N/A"
                return data # Return data fetched from static CSV


            else:
                 logger.warning("Static CSV fallback file is empty.")
                 pass # Do nothing, warning is logged, proceed to final fallback


        except FileNotFoundError:
             logger.warning("Static CSV fallback file not found: fallback_data.csv")
             pass # Do nothing, warning is logged, proceed to final fallback
        except Exception as e:
            logger.warning(f"Error fetching data from static CSV fallback: {str(e)}")
            pass # Do nothing, error is logged, proceed to final fallback


        # --- Final Fallback: Return all required keys with "N/A" or default placeholders ---
        logger.error("All data fetching fallbacks failed. Returning default/N/A values.")
        final_fallback_data = {
            "iv": "N/A", "gamma": "N/A", "delta": "N/A", "theta": "N/A", "vega": "N/A",
            "vix": "N/A", "ivp": "N/A", "ivp_status": "N/A", "margin": "N/A", "strike": "N/A",
            "fii_flow": "N/A", "dii_flow": "N/A", "buzz_topic": "N/A", "sentiment": "N/A",
            "news_headline": "N/A", "impact": "N/A", "community_hedge": "N/A",
            "trade_count": "N/A", "trade_frequency": "N/A", "revenge_count": "N/A",
            "bias": "N/A", "bias_count": "N/A", "strategy": "N/A",
            "win_rate": "N/A", "reason": "N/A", "dte": "N/A", "pnl": "N/A",
            "loss_risk": "N/A", "breakeven": "N/A"
        }
        # Return a dictionary containing only the required keys from the final fallback defaults
        return {key: final_fallback_data.get(key, "N/A") for key in required_keys}


    def generate_response(self, user_query):
        """
        Generates a response for the user query by matching patterns and formatting templates.

        Args:
            user_query (str): The input query from the user.

        Returns:
            str: The generated response string.
        """
        user_query_lower = user_query.lower().strip()
        logger.info(f"Generating response for query: '{user_query_lower}'")

        # Iterate through each response rule loaded from the CSV
        for index, row in self.responses.iterrows():
            pattern = row["query_pattern"]
            context_needed_keys = row["context_needed"]
            response_template = row["response_template"]

            # Check if the user query matches the current pattern using regex
            try:
                if re.search(pattern, user_query_lower):
                    logger.debug(f"Query matched pattern: '{pattern}'")
                    # If a match is found, fetch the necessary app data context
                    context = self.fetch_app_data(context_needed_keys)
                    logger.debug(f"Fetched context: {context}")
                    try:
                        # Format the response template using the fetched context data
                        # Handle potential KeyError if a key in the template is missing in context
                        response = response_template.format(**context)
                        logger.info(f"Generated response: '{response}'")
                        return response # Return the generated response

                    except KeyError as ke:
                        logger.error(f"KeyError formatting response for pattern '{pattern}': Missing key '{ke}'")
                        return "Bhai, data thoda off lag raha hai ya response template mein gadbad hai. Try again! Do your own research! (Error: Missing Data)"
                    except Exception as e:
                         logger.error(f"Error formatting response for pattern '{pattern}': {str(e)}")
                         return "Bhai, response generate karte waqt kuch technical issue ho gaya. Try again! Do your own research! (Error: Formatting Failed)"

            except re.error as re_e:
                 logger.error(f"Regex error in pattern '{pattern}': {re_e}")
                 # Continue to next pattern if a regex is invalid
                 continue
            except Exception as e:
                logger.error(f"Unexpected error during pattern matching or data fetching for pattern '{pattern}': {str(e)}")
                # Continue to next pattern if an error occurs during processing this pattern
                continue


        # If no pattern matched the user query after checking all rules
        logger.info(f"No pattern matched query: '{user_query_lower}'")
        return "Bhai, yeh query thodi alag hai. Kya bol raha hai, thoda clearly bata? 😜 Do your own research!"

# The __main__ block is removed because this class is designed to be run
# within the Streamlit app and directly accesses st.session_state,
# which is not available when running this script standalone.
# To test, you would need to run the Streamlit app.
