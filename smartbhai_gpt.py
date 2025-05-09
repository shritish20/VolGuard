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
                portfolio_data = st.session_state.get("api_portfolio_data", {})
                data = {
                    "iv": latest_data.get("IV", 30.0),  # Implied Volatility
                    "gamma": latest_data.get("Gamma", 0.05),
                    "delta": latest_data.get("Delta", 0.4),
                    "theta": latest_data.get("Theta", -0.02),
                    "vega": latest_data.get("Vega", 0.1),
                    "vix": latest_data.get("VIX", 25.0),  # India VIX
                    "ivp": latest_data.get("IVP", 75.0),  # Implied Volatility Percentile
                    "ivp_status": "high" if latest_data.get("IVP", 75.0) > 70 else "low",
                    "margin": (portfolio_data.get("margin", {}).get("UtilizedMargin", 0.0) / st.session_state.get("capital", 1000000) * 100) or 85.0,
                    "strike": latest_data.get("StrikePrice", "N/A"),
                    "fii_flow": latest_data.get("FII_Flow", 0.0),  # FII net flow in crores
                    "dii_flow": latest_data.get("DII_Flow", 0.0),  # DII net flow in crores
                    "buzz_topic": latest_data.get("Buzz_Topic", "NIFTY expiry"),  # X buzz
                    "sentiment": latest_data.get("Sentiment", 60),  # Bullish sentiment %
                    "news_headline": latest_data.get("News_Headline", "No major news"),
                    "impact": latest_data.get("Impact", 50),  # Bearish move probability
                    "community_hedge": latest_data.get("Community_Hedge", 70),  # % users hedging
                    "trade_count": latest_data.get("Trade_Count", 10),  # Total trades
                    "trade_frequency": latest_data.get("Trade_Frequency", 2),  # Trades/day
                    "revenge_count": latest_data.get("Revenge_Trades", 2),  # Revenge trades
                    "bias": latest_data.get("Bias", "FOMO"),  # Detected bias
                    "bias_count": latest_data.get("Bias_Count", 3),  # Bias occurrences
                    "strategy": latest_data.get("Strategy", "Iron Condor"),  # Last strategy
                    "win_rate": latest_data.get("Win_Rate", 60),  # Backtest win rate %
                    "reason": latest_data.get("Failure_Reason", "IV spike"),  # Trade failure
                    "dte": latest_data.get("DTE", 5),  # Days to Expiry
                    "pnl": portfolio_data.get("pnl", 2.5),  # Portfolio PnL %
                    "loss_risk": portfolio_data.get("loss_risk", 15),  # Potential loss %
                    "breakeven": latest_data.get("Breakeven", "N/A")  # Breakeven price
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
                    portfolio_data = st.session_state.get("api_portfolio_data", {})
                    data = {
                        "iv": 30.0,  # Replace with actual IV
                        "gamma": 0.05,
                        "delta": 0.4,
                        "theta": -0.02,
                        "vega": 0.1,
                        "vix": 25.0,
                        "ivp": 75.0,
                        "ivp_status": "high",
                        "margin": (portfolio_data.get("margin", {}).get("UtilizedMargin", 0.0) / st.session_state.get("capital", 1000000) * 100) or 85.0,
                        "strike": "N/A",
                        "fii_flow": 0.0,
                        "dii_flow": 0.0,
                        "buzz_topic": "NIFTY expiry",
                        "sentiment": 60,
                        "news_headline": "No major news",
                        "impact": 50,
                        "community_hedge": 70,
                        "trade_count": 10,
                        "trade_frequency": 2,
                        "revenge_count": 2,
                        "bias": "FOMO",
                        "bias_count": 3,
                        "strategy": "Iron Condor",
                        "win_rate": 60,
                        "reason": "IV spike",
                        "dte": 5,
                        "pnl": 2.5,
                        "loss_risk": 15,
                        "breakeven": "N/A"
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
                        "theta": latest_fallback.get("theta", -0.02),
                        "vega": latest_fallback.get("vega", 0.1),
                        "vix": latest_fallback.get("vix", 25.0),
                        "ivp": latest_fallback.get("ivp", 75.0),
                        "ivp_status": latest_fallback.get("ivp_status", "high"),
                        "margin": latest_fallback.get("margin", 85.0),
                        "strike": latest_fallback.get("strike", "N/A"),
                        "fii_flow": latest_fallback.get("fii_flow", 0.0),
                        "dii_flow": latest_fallback.get("dii_flow", 0.0),
                        "buzz_topic": latest_fallback.get("buzz_topic", "NIFTY expiry"),
                        "sentiment": latest_fallback.get("sentiment", 60),
                        "news_headline": latest_fallback.get("news_headline", "No major news"),
                        "impact": latest_fallback.get("impact", 50),
                        "community_hedge": latest_fallback.get("community_hedge", 70),
                        "trade_count": latest_fallback.get("trade_count", 10),
                        "trade_frequency": latest_fallback.get("trade_frequency", 2),
                        "revenge_count": latest_fallback.get("revenge_count", 2),
                        "bias": latest_fallback.get("bias", "FOMO"),
                        "bias_count": latest_fallback.get("bias_count", 3),
                        "strategy": latest_fallback.get("strategy", "Iron Condor"),
                        "win_rate": latest_fallback.get("win_rate", 60),
                        "reason": latest_fallback.get("reason", "IV spike"),
                        "dte": latest_fallback.get("dte", 5),
                        "pnl": latest_fallback.get("pnl", 2.5),
                        "loss_risk": latest_fallback.get("loss_risk", 15),
                        "breakeven": latest_fallback.get("breakeven", "N/A")
                    }
                except Exception as e3:
                    # Last resort: Hardcoded defaults
                    data = {
                        "iv": "N/A",
                        "gamma": "N/A",
                        "delta": "N/A",
                        "theta": "N/A",
                        "vega": "N/A",
                        "vix": "N/A",
                        "ivp": "N/A",
                        "ivp_status": "N/A",
                        "margin": "N/A",
                        "strike": "N/A",
                        "fii_flow": "N/A",
                        "dii_flow": "N/A",
                        "buzz_topic": "N/A",
                        "sentiment": "N/A",
                        "news_headline": "N/A",
                        "impact": "N/A",
                        "community_hedge": "N/A",
                        "trade_count": "N/A",
                        "trade_frequency": "N/A",
                        "revenge_count": "N/A",
                        "bias": "N/A",
                        "bias_count": "N/A",
                        "strategy": "N/A",
                        "win_rate": "N/A",
                        "reason": "N/A",
                        "dte": "N/A",
                        "pnl": "N/A",
                        "loss_risk": "N/A",
                        "breakeven": "N/A"
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
