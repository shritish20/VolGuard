import pandas as pd
import re
import streamlit as st
import csv
from fivepaisa_api import fetch_market_depth_by_scrip

class SmartBhaiGPT:
    def __init__(self, responses_file="responses.csv"):
        # Load response templates
        try:
            self.responses = pd.read_csv(
                responses_file,
                quoting=csv.QUOTE_ALL,  # Force quoting to handle commas in text
                encoding='utf-8'
            )
        except FileNotFoundError:
            raise FileNotFoundError("Bhai, responses.csv nahi mila! Check kar project folder mein.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Bhai, responses.csv ka format galat hai! Error: {str(e)}")
    
    def fetch_app_data(self, context_needed):
        """
        Fetch real-time data (e.g., IV, gamma) from VolGuard Pro's data pipeline.
        Primary: st.session_state.analysis_df (from generate_features).
        Fallback 1: 5paisa API via fetch_market_depth_by_scrip.
        Fallback 2: Static fallback_data.csv.
        """
        try:
            # Primary source: st.session_state.analysis_df from generate_features
            if "analysis_df" in st.session_state and st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
                df = st.session_state.analysis_df
                latest_data = df.iloc[-1]  # Get the latest row
                data = {
                    "iv": latest_data.get("IV", 30.0),  # Implied Volatility
                    "gamma": latest_data.get("Gamma", 0.05),
                    "delta": latest_data.get("Delta", 0.4),
                    "vix": latest_data.get("VIX", 25.0),  # India VIX
                    "margin": (st.session_state.get("api_portfolio_data", {}).get("margin", {}).get("UtilizedMargin", 0.0) / st.session_state.get("capital", 1000000) * 100) or 85.0
                }
            else:
                raise ValueError("Analysis DataFrame not available")
        
        except Exception as e:
            # Fallback 1: Try 5paisa API
            try:
                client = st.session_state.get("client")
                if client and client.get_access_token():
                    # Example: Fetch NIFTY options data (adjust ScripCode as needed)
                    market_data = fetch_market_depth_by_scrip(client, Exchange="N", ExchangeType="D", ScripCode=999920000)  # NIFTY index placeholder
                    ltp = market_data["Data"][0].get("LastTradedPrice", 0.0) if market_data and market_data.get("Data") else 0.0
                    # Mock option greeks (replace with actual option chain data if available)
                    data = {
                        "iv": 30.0,  # Replace with actual IV from option chain
                        "gamma": 0.05,
                        "delta": 0.4,
                        "vix": 25.0,  # Replace with actual VIX from API
                        "margin": (st.session_state.get("api_portfolio_data", {}).get("margin", {}).get("UtilizedMargin", 0.0) / st.session_state.get("capital", 1000000) * 100) or 85.0
                    }
                else:
                    raise ValueError("5paisa client not available")
            except Exception as e2:
                # Fallback 2: Load from fallback_data.csv
                try:
                    fallback_df = pd.read_csv("fallback_data.csv", encoding='utf-8')
                    latest_fallback = fallback_df.iloc[-1]
                    data = {
                        "iv": latest_fallback.get("iv", 30.0),
                        "gamma": latest_fallback.get("gamma", 0.05),
                        "delta": latest_fallback.get("delta", 0.4),
                        "vix": latest_fallback.get("vix", 25.0),
                        "margin": latest_fallback.get("margin", 85.0)
                    }
                except Exception as e3:
                    # Last resort: Hardcoded defaults
                    data = {
                        "iv": "N/A",
                        "gamma": "N/A",
                        "delta": "N/A",
                        "vix": "N/A",
                        "margin": "N/A"
                    }
                    print(f"Error fetching data: Primary failed ({e}), API failed ({e2}), CSV failed ({e3})")
        
        # Return only the needed context
        return {key: data.get(key, "N/A") for key in context_needed.split(",")}
    
    def generate_response(self, user_query):
        """
        Match user query to a response template and fill with app data.
        """
        user_query = user_query.lower().strip()
        
        # Find matching response
        for _, row in self.responses.iterrows():
            pattern = row["query_pattern"]
            if re.search(pattern, user_query):
                # Fetch required context (e.g., IV, gamma)
                context = self.fetch_app_data(row["context_needed"])
                
                # Fill response template
                try:
                    response = row["response_template"].format(**context)
                    return response
                except KeyError:
                    return "Bhai, data thoda off lag raha hai. Try again! Do your own research!"
        
        # Fallback response for unmatched queries
        return "Bhai, yeh query thodi alag hai. Kya bol raha hai, thoda clearly bata? 😜 Do your own research!"

# Test the class
if __name__ == "__main__":
    gpt = SmartBhaiGPT()
    test_queries = [
        "What is IV?",
        "Check my straddle at 21000",
        "Should I hedge?",
        "Random query"
    ]
    for query in test_queries:
        print(f"Query: {query}")
        print(f"Response: {gpt.generate_response(query)}\n")
