# Open your file: smartbhai_gpt.py
# Replace the ENTIRE content of this file with the code below.
import os
import pandas as pd
import re
import streamlit as st # Streamlit is needed to access st.session_state
import csv
import numpy as np  # Import numpy for calculations
import logging      # Import logging

# Assuming fivepaisa_api is a separate file/module you have
from fivepaisa_api import fetch_market_depth_by_scrip # Ensure this import path is correct

# Assuming data_processing is a separate file/module you have
# from data_processing import FEATURE_COLS # You might need imports from other modules if you use them directly here

# Setup logging for this specific module
# This ensures logs from smartbhai_gpt.py are formatted correctly
logger = logging.getLogger(__name__) # Use __name__ to get a logger specific to this module


class SmartBhaiGPT:
    """
    The brain and voice of the VolGuard Pro copilot. Handles conversational
    responses and proactive risk assessments.
    """
    def __init__(self, responses_file="responses.csv"):
        """
        Initializes the SmartBhaiGPT with conversational response templates.
        Loads response patterns and templates from a CSV file.
        """
        logger.info(f"SmartBhaiGPT: Initializing, loading responses from {responses_file}")
        # Load response templates from the specified CSV file
        try:
            self.responses = pd.read_csv(
                responses_file,
                quoting=csv.QUOTE_ALL,  # Force quoting to handle commas in text, important for messages
                encoding='utf-8'        # Specify UTF-8 encoding
            )
            logger.info(f"SmartBhaiGPT: Loaded {len(self.responses)} response templates successfully.")
        except FileNotFoundError:
            logger.error(f"SmartBhaiGPT: Responses file not found at {responses_file}")
            # Re-raise the error with a user-friendly message if the file is missing
            raise FileNotFoundError(f"Bhai, responses.csv nahi mila! Check kar project folder mein: {responses_file}")
        except pd.errors.ParserError as e:
             logger.error(f"SmartBhaiGPT: Error parsing responses file {responses_file}: {e}")
             raise ValueError(f"Bhai, responses.csv ka format galat hai! Error: {str(e)}")
        except Exception as e:
             logger.error(f"SmartBhaiGPT: Unexpected error loading responses file {responses_file}: {e}", exc_info=True)
             raise RuntimeError(f"Bhai, responses.csv load karte time kuch gadbad ho gaya: {str(e)}")


    def fetch_app_data(self, context_needed):
        """
        Fetches specific data points from the application's session state
        or fallback sources to fulfill conversational query context for template formatting.

        Args:
            context_needed (str): A comma-separated string of data keys required
                                  by the response template (e.g., "iv,margin,strategy").

        Returns:
            dict: A dictionary containing the requested data keys and their values.
        """
        logger.debug(f"SmartBhaiGPT: Fetching app data for context: {context_needed}")
        data = {} # Dictionary to hold fetched data

        # Attempt to get data primarily from Streamlit's session state
        # This is the most direct way to access the live state of the app
        try:
            # Access session state directly. Handle cases where session_state or required keys might not exist.
            session_state_data = st.session_state # Get the entire session state object

            # Get data from analysis_df if available
            analysis_df = session_state_data.get("analysis_df")
            latest_data = analysis_df.iloc[-1] if analysis_df is not None and not analysis_df.empty else {} # Get latest row or empty dict

            # Get data from API portfolio data if available
            portfolio_data = session_state_data.get("api_portfolio_data", {})

            # Populate data dictionary from session state sources
            # Use .get() on latest_data, portfolio_data, and session_state_data with default values
            # Ensure data types are consistent where used in formatting templates
            data["iv"] = latest_data.get("ATM_IV", latest_data.get("VIX", 30.0)) # Prefer ATM_IV, fallback to VIX
            data["gamma"] = latest_data.get("Gamma", 0.05)
            data["delta"] = latest_data.get("Delta", 0.4)
            data["theta"] = latest_data.get("Theta", -0.02)
            data["vega"] = latest_data.get("Vega", 0.1)
            data["vix"] = latest_data.get("VIX", 25.0)
            data["ivp"] = latest_data.get("IVP", 75.0)
            # Calculate ivp_status dynamically if ivp is a number
            data["ivp_status"] = "high" if isinstance(data["ivp"], (int, float)) and data["ivp"] > 70 else ("low" if isinstance(data["ivp"], (int, float)) and data["ivp"] <= 70 else "N/A")

            # Calculate margin % - handle potential division by zero or zero capital
            total_capital = session_state_data.get("capital", 1000000)
            utilized_margin = portfolio_data.get("margin", {}).get("UtilizedMargin", 0.0)
            data["margin"] = (utilized_margin / total_capital * 100) if total_capital > 0 else 0
            data["margin_used"] = utilized_margin # Also provide absolute margin used

            data["strike"] = latest_data.get("StrikePrice", "N/A")
            data["fii_flow"] = latest_data.get("FII_Index_Fut_Pos", "N/A") # Use relevant FII/DII columns from analysis_df
            data["dii_flow"] = latest_data.get("FII_Option_Pos", "N/A") # Adjust column names as per data_processing.py

            # Placeholders/Proxies for data that might come from other modules (News, Community, Behavioral)
            data["buzz_topic"] = session_state_data.get("news_sentiment", {}).get("buzz_topic", "NIFTY expiry") # Assuming a news_sentiment state key
            data["sentiment"] = session_state_data.get("news_sentiment", {}).get("sentiment_score", "N/A") # Assuming a news_sentiment state key
            data["news_headline"] = session_state_data.get("news_sentiment", {}).get("latest_headline", "No major news") # Assuming a news_sentiment state key
            data["impact"] = session_state_data.get("forecast_metrics", {}).get("confidence_score", 50) # Using forecast confidence as a proxy for impact clarity
            data["community_hedge"] = session_state_data.get("community_data", {}).get("hedge_pct", "N/A") # Assuming a community_data state key
            data["trade_count"] = len(session_state_data.get("trades_log", [])) if session_state_data.get("trades_log") is not None else "N/A" # Assuming a trades_log state key or using journal
            data["trade_frequency"] = "N/A" # Requires time-based analysis of trade log
            # Need actual logic to count revenge trades from journal/trade log
            data["revenge_count"] = latest_data.get("Revenge_Trades", "N/A") # Assuming added to features or journal analysis
            data["bias"] = latest_data.get("Detected_Bias", "N/A") # Assuming added to features or behavioral analysis
            data["bias_count"] = latest_data.get("Bias_Count", "N/A") # Assuming added to features or behavioral analysis

            # Get data related to the generated strategy and backtest results
            generated_strategy = session_state_data.get("generated_strategy", {})
            backtest_results = session_state_data.get("backtest_results", {})
            active_strategy_details = session_state_data.get("active_strategy_details", {}) # Details of the strategy actually being traded

            data["strategy"] = generated_strategy.get("Strategy", "N/A") # Recommended strategy
            data["win_rate"] = backtest_results.get("win_rate", 0) * 100 if isinstance(backtest_results.get("win_rate"), (int, float)) else "N/A" # Overall backtest win rate %
            data["reason"] = generated_strategy.get("Reason", active_strategy_details.get("Reason", "N/A")) # Reason for recommended or active strategy

            data["dte"] = latest_data.get("Days_to_Expiry", "N/A")
            data["pnl"] = portfolio_data.get("weekly_pnl", "N/A") # Use a relevant PnL metric, e.g., weekly PnL
            data["loss_risk"] = generated_strategy.get("Max_Loss", "N/A") # Max loss of recommended strategy
            data["breakeven"] = "N/A" # Needs strategy-specific calculation

            logger.debug(f"SmartBhaiGPT: Data fetched from session state for templates: {data}")

        except Exception as e:
            logger.warning(f"SmartBhaiGPT: Error fetching data from session state for templates ({e}). Returning default N/A data.", exc_info=True)
            # If fetching from session state fails for any reason, provide default N/A data
            data = {
                "iv": "N/A", "gamma": "N/A", "delta": "N/A", "theta": "N/A", "vega": "N/A",
                "vix": "N/A", "ivp": "N/A", "ivp_status": "N/A", "margin": "N/A", "margin_used": "N/A",
                "strike": "N/A", "fii_flow": "N/A", "dii_flow": "N/A", "buzz_topic": "N/A", "sentiment": "N/A",
                "news_headline": "N/A", "impact": "N/A", "community_hedge": "N/A",
                "trade_count": "N/A", "trade_frequency": "N/A", "revenge_count": "N/A",
                "bias": "N/A", "bias_count": "N/A", "strategy": "N/A", "win_rate": "N/A",
                "reason": "N/A", "dte": "N/A", "pnl": "N/A", "loss_risk": "N/A", "breakeven": "N/A"
            }
            # Note: Fallback to API or CSV is not implemented in this version of fetch_app_data
            # as the primary goal is to use session state data for conversational context.
            # If session state is empty (e.g., before analysis run), most values will be N/A or defaults.


        # Return only the needed context requested by the query pattern
        requested_context = {}
        # Ensure context_needed is a string before splitting
        if isinstance(context_needed, str):
            for key in context_needed.split(","):
                 key = key.strip() # Remove leading/trailing whitespace
                 if key in data:
                      requested_context[key] = data[key]
                 else:
                     logger.warning(f"SmartBhaiGPT: Requested context key '{key}' not found in available data for template.")
                     requested_context[key] = "N/A" # Indicate if a requested key wasn't found
        else:
            logger.warning(f"SmartBhaiGPT: Invalid context_needed format: {context_needed}. Cannot fetch specific context for template.")
            # If context_needed is not a valid string, return an empty dictionary or log error
            requested_context = {} # Return empty dict if context_needed is invalid


        logger.debug(f"SmartBhaiGPT: Returning requested context: {requested_context}")
        return requested_context


    def generate_response(self, user_query, app_state=None):
        """
        Generates a conversational response for the user's query.
        Prioritizes dynamic responses based on app state, falls back to template matching.

        Args:
            user_query (str): The user's input query string.
            app_state (dict, optional): The application's session state (st.session_state).
                                        Defaults to None. Provides access to live data for dynamic responses.

        Returns:
            str: The generated response string from SmartBhai.
        """
        logger.info(f"SmartBhaiGPT: Generating response for query: '{user_query}'")
        user_query = user_query.lower().strip()

        # --- Dynamic Responses based on App State ---
        # Add logic here to generate responses for specific queries
        # that require understanding the current state of the app.
        # Example: "How is the market?" uses live market status from app_state.
        if app_state and ("how is the market" in user_query or "market status" in user_query):
             market_status_data = app_state.get("api_portfolio_data", {}).get("market_status", {})
             if market_status_data:
                  market_open = market_status_data.get("MarketStatus", {}).get("IsOpen", False)
                  if market_open:
                      dynamic_resp = "Bhai, market abhi khula hai aur kaam chal raha hai! 😎 Kya dekhna hai?"
                      logger.info("SmartBhaiGPT: Generated dynamic response: Market Open.")
                      return dynamic_resp + " Do your own research!" # Add disclaimer
                  else:
                      dynamic_resp = "Market band hai bhai abhi. Kal subah hi dekhenge! 😴"
                      logger.info("SmartBhaiGPT: Generated dynamic response: Market Closed.")
                      return dynamic_resp + " Do your own research!" # Add disclaimer
             else:
                  logger.warning("SmartBhaiGPT: Market status data not available in app_state for dynamic response.")
                  return "Market status data available nahi hai abhi. Do your own research!" # Fallback generic message

        # You can add more dynamic response triggers here, e.g.:
        # If "my portfolio" in user_query and app_state and "api_portfolio_data" in app_state:
        #     # Access portfolio_data and craft a dynamic summary message
        #     portfolio_summary = app_state.get("api_portfolio_data", {})
        #     # ... build message using portfolio_summary ...
        #     return "Bhai, tere portfolio ka yeh haal hai... Do your own research!"


        # --- Fallback to Template Matching ---
        # If no dynamic response matches the query, try matching it against predefined templates.
        logger.debug("SmartBhaiGPT: No dynamic response matched. Checking response templates.")
        for _, row in self.responses.iterrows():
            pattern = row["query_pattern"]
            # Use re.search to find the pattern anywhere in the user query (case-insensitive due to .lower())
            if re.search(pattern, user_query):
                logger.debug(f"SmartBhaiGPT: Matched query pattern: '{pattern}'")
                # Fetch required context using the method.
                # fetch_app_data implicitly uses st.session_state to get data when run within Streamlit.
                context = self.fetch_app_data(row["context_needed"])

                # Fill the response template string with the fetched context data.
                try:
                    response = row["response_template"].format(**context)
                    logger.info("SmartBhaiGPT: Generated response from template.")
                    # Add the disclaimer to all template-based responses
                    return response + " Do your own research!"
                except KeyError as e:
                    # Handle cases where the template expects a key that fetch_app_data couldn't provide
                    logger.error(f"SmartBhaiGPT: KeyError formatting template for pattern '{pattern}': Missing context key '{e}'. Available context: {context}")
                    return "Bhai, kuch data points missing lag rahe hain is query ke liye. Try again! Do your own research!"
                except Exception as e:
                     # Handle any other unexpected errors during template formatting
                     logger.error(f"SmartBhaiGPT: Unexpected error formatting template for pattern '{pattern}': {e}", exc_info=True)
                     return "Bhai, response generate karte time kuch gadbad ho gaya. Do your own research!"


        # --- Fallback response for queries that don't match any dynamic logic or template ---
        logger.info("SmartBhaiGPT: No pattern matched for query. Returning default fallback.")
        return "Bhai, yeh query thodi alag hai. Kya bol raha hai, thoda clearly bata? ðŸ˜œ Do your own research!"


    # --- START OF THE assess_trade_risk METHOD (Proactive Risk Officer Brain) ---
    # This method performs the risk checks and returns a structured result.
    def assess_trade_risk(self, app_state):
        """
        Assesses the risk of the currently proposed strategy based on market conditions
        and potentially user behavior. Returns a dictionary with main message and
        educational explanations if risk is detected according to the defined rules,
        otherwise returns None.

        This function acts as a key part of SmartBhai's 'risk officer' brain.
        It looks at the current situation in the app (market, strategy, portfolio, etc.)
        and decides if a warning needs to be issued BEFORE a trade is placed.

        Args:
            app_state: A dictionary containing relevant application state, typically
                       st.session_state from your Streamlit app.
                       It should contain keys like 'real_time_market_data',
                       'forecast_log', 'generated_strategy', 'analysis_df',
                       'api_portfolio_data', 'capital', 'journal_log' (potentially).

        Returns:
            dict or None: A dictionary like {"main_message": str, "explanations": list[str]}
                          if risk is detected, otherwise returns None (meaning no warning needed).
        """
        # Log that we are starting the risk assessment
        logger.info("SmartBhai: Starting trade risk assessment.")

        # Initialize the return value. Use a dictionary to hold the main message and explanations.
        # Initialize main_message as None and explanations as an empty list.
        risk_assessment_result = {"main_message": None, "explanations": []}

        # --- 1. Access Necessary Data from the Application State ---
        # SmartBhai needs to 'see' what's happening in the app. We get this from app_state.
        # We use .get() with a default (like None or {}) to avoid errors if a key is missing.
        # It's important to get all relevant data points here that your risk rules might need.
        market_data = app_state.get("real_time_market_data")
        forecast_log = app_state.get("forecast_log") # Contains the forecast results, including blended volatility
        generated_strategy = app_state.get("generated_strategy") # Details of the strategy the engine recommended
        analysis_df = app_state.get("analysis_df") # The main DataFrame with market features
        portfolio_data = app_state.get("api_portfolio_data", {}) # Portfolio data for margin/position checks
        capital = app_state.get("capital", 1000000) # User's total capital (default to 1M if not set)
        journal_df = app_state.get("journal_df") # Access journal data if loaded into state (Needs to be loaded in app.py)


        # Basic check: If essential data isn't available (e.g., analysis hasn't run), we can't assess risk meaningfully.
        if generated_strategy is None or market_data is None or forecast_log is None or analysis_df is None or analysis_df.empty:
            logger.warning("SmartBhai: Risk assessment skipped due to missing essential analysis data.")
            # Consider adding a UI element in Streamlit to inform the user to run analysis first.
            return None # No warning if we don't have the necessary data to check


        # --- 2. Extract Key Information for Risk Checks ---
        # Get specific values from the fetched data that your risk rules will use.
        current_vix = market_data.get("vix", 0) # Get the latest India VIX value (default to 0 if missing)

        # Calculate average forecasted blended volatility from the forecast log.
        # Ensure we handle cases where forecast_log might be empty or Blended_Vol is all NaNs.
        forecasted_blended_vol = 0
        if forecast_log is not None and not forecast_log.empty and 'Blended_Vol' in forecast_log.columns:
             valid_forecasts = forecast_log['Blended_Vol'].dropna() # Only consider non-NaN forecast values
             if not valid_forecasts.empty:
                  forecasted_blended_vol = valid_forecasts.mean()
             else:
                  logger.warning("SmartBhai: Blended_Vol column in forecast_log contains only NaNs.")


        # Get days to expiry from the last (latest) row of your analysis DataFrame.
        # Safely access 'Days_to_Expiry' with .get().
        days_to_expiry = analysis_df.iloc[-1].get("Days_to_Expiry", 99) # Default to a high number if missing

        strategy_name = generated_strategy.get("Strategy") # Get the name of the proposed strategy (e.g., "Short Straddle")
        strategy_deploy_pct = generated_strategy.get("Exposure", 0) # Capital exposure percentage for the strategy
        strategy_max_loss = generated_strategy.get("Max_Loss", 0) # Max loss for the strategy (absolute value)
        risk_flags = generated_strategy.get("Risk_Flags", []) # Get any risk flags already added by the strategy generator (list of strings)
        behavior_warnings = generated_strategy.get("Behavior_Warnings", []) # Get any behavioral warnings added by the strategy generator (list of strings)

        # Get portfolio-level risk metrics from portfolio data
        current_margin_used = portfolio_data.get("margin", {}).get("UtilizedMargin", 0)
        # Calculate margin usage percentage, handle potential division by zero or zero capital
        current_margin_used_pct = (current_margin_used / capital * 100) if capital > 0 else 0


        logger.debug(f"SmartBhai: Risk check inputs extracted - VIX: {current_vix:.1f}%, Forecast Vol: {forecasted_blended_vol:.1f}%, DTE: {days_to_expiry}, Strategy: '{strategy_name}', Strat Exposure: {strategy_deploy_pct:.1f}%, Margin Used: {current_margin_used_pct:.1f}%")


        # --- 3. Implement Risk Rules (This is where SmartBhai's 'training' logic resides) ---
        # These 'if' conditions are the rules that trigger SmartBhai's warnings.
        # You can add more rules here as you identify potential risks you want SmartBhai to flag.
        # For each triggered rule, add a main message (if none exists) and an explanation.

        # Rule 1: High Volatility / IV near Expiry with Sensitive Strategies (Mimics your RBI example)
        # Purpose: Warns the user if they are considering strategies vulnerable to large, fast moves
        # when market volatility is high and expiry is close.
        high_volatility_threshold = 20 # VIX or forecast above this is considered high vol
        approaching_expiry_days = 5 # Less than or equal to this many days is near expiry

        # List strategies that are particularly risky in high vol, near expiry environments
        sensitive_strategies = ["Short Straddle", "Short Strangle", "Naked Put", "Naked Call"] # Add other similar strategies if needed

        # Check the condition: (Is volatility high?) AND (Is expiry near?) AND (Is the strategy sensitive?)
        if (current_vix > high_volatility_threshold or forecasted_blended_vol > high_volatility_threshold) and days_to_expiry <= approaching_expiry_days:
            if strategy_name in sensitive_strategies:
                 # Set the main message if this is the first rule triggered
                 if risk_assessment_result["main_message"] is None:
                     risk_assessment_result["main_message"] = f"🔥 Bhai, volatility ({max(current_vix, forecasted_blended_vol):.1f}%) high hai aur expiry ({days_to_expiry} din) paas hai! {strategy_name} mein sudden big move ka risk hai."

                 # Add an explanation specific to why this is risky in this context
                 explanation = f"High volatility near expiry means option prices (especially for short options like in a {strategy_name}) can change dramatically and quickly with small market moves. This increases the chance of hitting your stop-loss or facing significant losses rapidly if the market moves against you before expiry. Consider hedging or reducing position size."
                 risk_assessment_result["explanations"].append(explanation + " Do your own research!") # Add disclaimer to explanation

                 logger.warning(f"SmartBhai Risk Rule 1 triggered: High Volatility near Expiry with {strategy_name}.") # Log the trigger


        # Rule 2: Highlight Risk Flags identified by the Strategy Generator.
        # Purpose: Ensure the user sees and acknowledges potential risks already identified by the analytical engine.
        # This rule triggers if the 'Risk_Flags' list from the generated strategy is not empty.
        if risk_flags:
             # Set a main message if this is the first rule triggered
             if risk_assessment_result["main_message"] is None:
                  risk_assessment_result["main_message"] = f"⚠️ Strategy analysis ne kuch Risk Flags identify kiye hain."

             # Add a general explanation for risk flags
             explanation = f"The strategy analysis engine identified specific potential issues ({', '.join(risk_flags)}) based on current market data and the strategy structure. Review these flags carefully as they indicate areas of concern for this particular trade."
             risk_assessment_result["explanations"].append(explanation + " Do your own research!")

             logger.warning(f"SmartBhai Risk Rule 2 triggered: Strategy Risk Flags: {risk_flags}.")


        # Rule 3: Highlight Behavioral Warnings from the Strategy Generator.
        # Purpose: Remind the user about potential behavioral biases detected, encouraging discipline.
        # This rule triggers if the 'Behavior_Warnings' list from the generated strategy is not empty.
        if behavior_warnings:
             # Set a main message if this is the first rule triggered
             if risk_assessment_result["main_message"] is None:
                  risk_assessment_result["main_message"] = f"🧠 Ek behavioral warning bhi hai."

             explanation = f"Behavioral warnings ({', '.join(behavior_warnings)}) are based on detected patterns in your trading habits (potentially from your journal). They are nudges to help you stick to your trading plan and avoid impulsive decisions that may have led to unfavorable outcomes in the past. Stay disciplined!"
             risk_assessment_result["explanations"].append(explanation + " Do your own research!")
             logger.warning(f"SmartBhai Risk Rule 3 triggered: Behavioral Warnings: {behavior_warnings}.")

        # Rule 4: High Portfolio Margin Usage Warning.
        # Purpose: Alert the user if their overall account leverage (margin usage) is high.
        high_margin_threshold_pct = 75 # Example: Warn if margin used is > 75% of total capital

        # Check if the calculated margin usage percentage exceeds the threshold.
        if current_margin_used_pct > high_margin_threshold_pct:
             # Set a main message if this is the first rule triggered
             if risk_assessment_result["main_message"] is None:
                  risk_assessment_result["main_message"] = f"💰 Bhai, tera total margin usage ({current_margin_used_pct:.1f}%) high ho raha hai."

             explanation = f"High margin usage ({current_margin_used_pct:.1f}%) indicates you are using significant leverage in your portfolio. While leverage can amplify profits, it also drastically increases your risk of margin calls and large losses if the market moves against your positions, especially during volatile periods. Review your overall portfolio exposure."
             risk_assessment_result["explanations"].append(explanation + " Do your own research!")
             logger.warning(f"SmartBhai Risk Rule 4 triggered: High Margin Usage ({current_margin_used_pct:.1f}%).")

        # Rule 5: Warning if the proposed strategy's maximum loss is a high percentage of total capital.
        # Purpose: Make the user aware of the potential downside impact of the specific trade on their overall capital.
        high_strategy_max_loss_pct_threshold = 2 # Example: Warn if max loss is > 2% of total capital

        # Check if the strategy's max loss exceeds the threshold relative to total capital.
        if capital > 0 and strategy_max_loss > (capital * high_strategy_max_loss_pct_threshold / 100):
             # Set a main message if this is the first rule triggered
             if risk_assessment_result["main_message"] is None:
                  risk_assessment_result["main_message"] = f"💣 Is strategy ka maximum loss ({strategy_max_loss:,.0f} ₹) tere total capital ka lagbhag {strategy_max_loss / capital * 100:.1f}% hai."

             explanation = f"The maximum potential loss ({strategy_max_loss:,.0f} ₹) on this single trade represents a significant portion ({strategy_max_loss / capital * 100:.1f}%) of your total trading capital. This concentration of risk in one position can lead to substantial portfolio drawdowns if the trade goes wrong. Ensure this potential loss is within your acceptable risk limits for a single position."
             risk_assessment_result["explanations"].append(explanation + " Do your own research!")
             logger.warning(f"SmartBhai Risk Rule 5 triggered: High Strategy Max Loss ({strategy_max_loss / capital * 100:.1f}%).")

        # --- Add more risk rules here based on other factors you deem important ---
        # Remember to add a main_message (if none exists) and an explanation for each new rule.
        # You can access other data from app_state here (e.g., VaR from Risk Dashboard, journal analysis).

        # Example Rule 6: Warning for low forecast confidence on the volatility prediction
        # low_confidence_threshold = 60 # Example threshold for low confidence
        # forecast_confidence = app_state.get("forecast_metrics", {}).get("confidence_score", 100) # Default to 100 if missing
        # if forecast_confidence < low_confidence_threshold:
        #      if risk_assessment_result["main_message"] is None:
        #           risk_assessment_result["main_message"] = f"📉 Volatility forecast confidence thoda low hai ({forecast_confidence:.1f}%)."
        #      explanation = f"The confidence score ({forecast_confidence:.1f}%) for the volatility forecast is lower than usual. This means the model is less certain about the future volatility prediction, increasing the uncertainty around strategies that depend heavily on volatility assumptions. Be extra cautious with volatility-sensitive trades."
        #      risk_assessment_result["explanations"].append(explanation + " Do your own research!")
        #      logger.warning(f"SmartBhai Risk Rule 6 triggered: Low Forecast Confidence ({forecast_confidence:.1f}%).")


        # Example Rule 7: Behavioral nudge based on recent losing trades (requires loading/analyzing journal_log.csv in app_state)
        # Check if journal data is available and has recent entries
        # if journal_df is not None and not journal_df.empty:
        #      # Example: Check last 3 trades PnL from journal (assuming PnL is logged or can be derived)
        #      # This requires journal_log.csv to store PnL or link to trades
        #      recent_trades = journal_df.tail(3) # Look at last 3 journal entries
        #      recent_losses = recent_trades[recent_trades['Discipline_Score'] < 5] # Simplified check based on low score (needs actual PnL check)
        #
        #      if len(recent_losses) >= 2: # If at least 2 out of last 3 entries had low discipline scores (proxy for loss/poor trade)
        #           if risk_assessment_result["main_message"] is None:
        #                risk_assessment_result["main_message"] = f"🧠 Dhyan de, bhai! Tere recent trades mein kuch issues dikh rahe hain."
        #           explanation = f"Your recent trading journal entries suggest a pattern of poor outcomes or discipline issues in the last few trades. Trading after losses can lead to impulsive 'revenge' trading. Review your journal and ensure you are trading with a clear, disciplined mindset for this trade."
        #           risk_assessment_result["explanations"].append(explanation + " Do your own research!")
        #           logger.warning(f"SmartBhai Risk Rule 7 triggered: Recent potential losses detected.")


        # --- Final Return Value ---
        # If at least one rule was triggered and set a main_message, the dictionary is returned.
        # If no rules were triggered or no main_message was set, risk_assessment_result['main_message'] will still be None,
        # and the explanations list might be empty or contain explanations without a main message (less likely with current logic).
        # We only return the dictionary if there's a main message.
        if risk_assessment_result["main_message"] is not None:
             logger.info(f"SmartBhai: Risk assessment completed. Warning generated.")
             return risk_assessment_result # Return the dictionary containing the main message and explanations
        else:
             logger.info("SmartBhai: Risk assessment completed. No warning triggered.")
             return None # Return None if no risk rules were triggered


# --- END OF THE assess_trade_risk METHOD ---


# Test the class (your existing test code - requires mocking st.session_state)
if __name__ == "__main__":
    # This test block needs st.session_state and some data mocked to test the methods
    # as they rely heavily on the app's state.

    print("Running SmartBhaiGPT module test script...")
    print("Note: Full testing requires mocking Streamlit's session state.")

    # --- Mocking st.session_state ---
    # We create a simple class that mimics st.session_state for testing purposes
    class MockSessionState:
        def __init__(self):
            # Populate this mock state with realistic dummy data for testing different scenarios
            # Configure this state to trigger some risk rules for testing assess_trade_risk
            self.real_time_market_data = {"vix": 28.0} # Mock high VIX
            self.forecast_log = pd.DataFrame({"Blended_Vol": [27.0, 26.0, 25.0], "Date": pd.to_datetime(["2025-05-10", "2025-05-11", "2025-05-12"])}) # Mock high forecast vol with Dates
            # Mock a generated strategy - e.g., a Short Straddle which is sensitive to high vol near expiry
            self.generated_strategy = {
                "Regime": "HIGH",
                "Strategy": "Short Straddle", # Mocking a sensitive strategy
                "Reason": "Event Anticipation",
                "Tags": ["Volatility", "Event"],
                "Confidence": 70,
                "Risk_Reward": 1.8,
                "Deploy": 100000,
                "Max_Loss": 50000, # Mock a high max loss relative to capital
                "Exposure": 10, # Mock exposure
                "Risk_Flags": ["High IV Risk", "Near Expiry Risk"], # Mock existing risk flags
                "Behavior_Warnings": ["Overtrading Detected"], # Mock behavioral warning
                "Discipline_Lock": False
            }
            # Mock analysis_df with at least one row and relevant columns needed by assess_trade_risk
            # Ensure it has a DatetimeIndex if used by other parts
            self.analysis_df = pd.DataFrame({
                "Days_to_Expiry": [3], # Mock near expiry
                "VIX": [28.0],
                "ATM_IV": [28.0], # Keep IV matching VIX for simplicity in mock
                "Realized_Vol": [20.0],
                "IV_Skew": [0.5],
                "PCR": [1.0],
                 "Event_Flag": [1], # Mock event flag
                 "Spot_MaxPain_Diff_Pct": [0.5],
                 "PnL_Day": [0],
                 "VIX_Change_Pct": [10],
                 "FII_Index_Fut_Pos": [0],
                 "FII_Option_Pos": [0],
                 "IVP": [80],
                 "Straddle_Price": [300],
                 "Advance_Decline_Ratio": [1],
                 "Capital_Pressure_Index": [0]
            }, index=pd.to_datetime(['2025-05-10'])) # Use a dummy recent date as DatetimeIndex

            # Mock portfolio data for margin check
            self.api_portfolio_data = {"margin": {"UtilizedMargin": 800000}} # Mock high utilized margin

            self.capital = 1000000 # Mock total capital (10 Lakhs)

            # Mock other state variables that fetch_app_data or future rules might try to access
            self.backtest_results = {"win_rate": 0.6} # Mock backtest results
            self.active_strategy_details = {"Reason": "Event Play"} # Mock active strategy details
            self.news_sentiment = {"buzz_topic": "Election impact", "sentiment_score": 40, "latest_headline": "Market volatile before polls"} # Mock news/sentiment
            self.community_data = {"hedge_pct": 70} # Mock community data
            self.trades_log = [{"strategy": "IC", "pnl": 5000}, {"strategy": "Straddle", "pnl": -15000}] # Mock trade log (simple list of dicts)
            # Mock journal data - requires loading journal_log.csv or mocking a DataFrame
            # For this mock test, let's just mock a simple journal_df if the file exists or create a dummy
            dummy_journal_path = "journal_log.csv"
            if os.path.exists(dummy_journal_path):
                 self.journal_df = pd.read_csv(dummy_journal_path)
            else:
                 # Create a dummy journal DataFrame if the file doesn't exist for testing
                 self.journal_df = pd.DataFrame({
                     "Date": pd.to_datetime(['2025-05-08', '2025-05-09']),
                     "Strategy_Reason": ["Expiry Play", "Momentum"],
                     "Override_Risk": ["No", "Yes"],
                     "Expected_Outcome": ["Profit", "Big Profit"],
                     "Lessons_Learned": ["Stick to plan", ""],
                     "Discipline_Score": [8, 4]
                 })
            self.violations = 1 # Mock existing violations
            self.journal_complete = False # Mock journal status

        # This method is needed because SmartBhaiGPT accesses state like st.session_state.get(...)
        def get(self, key, default=None):
            """Mimics st.session_state.get() behavior."""
            return getattr(self, key, default)

    # Create a mock session state instance
    mock_state = MockSessionState()

    # --- Initialize SmartBhaiGPT ---
    # Ensure you have a dummy responses.csv file in your project folder for this test
    # or handle the FileNotFoundError if you don't.
    dummy_responses_file = "responses.csv"
    if not os.path.exists(dummy_responses_file):
        print(f"Creating a dummy {dummy_responses_file} for test...")
        # Create a minimal dummy file if it doesn't exist
        with open(dummy_responses_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['query_pattern', 'context_needed', 'response_template'])
            writer.writerow(['hello', '', 'Hello bhai! Kaise ho?']) # Add a simple rule
            # Add a rule that uses context data
            writer.writerow(['what is my margin', 'margin,margin_used', 'Bhai, tera margin usage {margin:.2f}% hai, matlab {margin_used:,.0f}₹ use ho raha hai.'])


    try:
        gpt = SmartBhaiGPT(responses_file=dummy_responses_file)
        print("SmartBhaiGPT initialized for testing.")
    except Exception as e:
        print(f"Error initializing SmartBhaiGPT: {e}. Cannot run tests.")
        gpt = None # Set gpt to None if initialization fails


    # --- Run Tests ---
    if gpt:
        print("\n--- Testing assess_trade_risk method ---")
        # Pass the mock session state to assess_trade_risk
        risk_warning_result = gpt.assess_trade_risk(mock_state)

        if risk_warning_result and isinstance(risk_warning_result, dict):
            print("\n--- SmartBhai Risk Warning (TEST) ---")
            print(f"Main Message: {risk_warning_result.get('main_message')}")
            print("Explanations:")
            for i, exp in enumerate(risk_warning_result.get('explanations', [])):
                 print(f"- {exp}")
            print("------------------------------------")
        else:
            print("\nNo risk warning triggered by the current mock state.")


        print("\n--- Testing generate_response (chat) method ---")
        test_queries = [
            "What is IV?", # Should use fetch_app_data
            "Check my straddle at 21000", # Example of a query that might need specific logic
            "Should I hedge?", # Example - might need logic based on position
            "Random query that won't match templates", # Should trigger fallback
            "how is the market", # Should trigger dynamic market status check
            "What is my margin used?", # Example using mock margin data via fetch_app_data
            "Tell me about my generated strategy", # Example using mock strategy data via fetch_app_data (if template exists)
             "hello" # Test simple template
        ]
         # Pass the mock session state to generate_response as well
        for query in test_queries:
           print(f"\nQuery: {query}")
           response = gpt.generate_response(query, app_state=mock_state) # Pass mock_state
           print(f"Response: {response}")


    print("\nSmartBhaiGPT module test script finished.")

# Ensure os is imported for the test block

