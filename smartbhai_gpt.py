import pandas as pd
import re
import streamlit as st
import csv
from upstox_api import fetch_market_depth_by_scrip

class SmartBhaiGPT:
    def __init__(self, responses_file="responses.csv"):
        try:
            self.responses = pd.read_csv(
                responses_file,
                quoting=csv.QUOTE_ALL,
                encoding='utf-8'
            )
        except FileNotFoundError:
            raise FileNotFoundError("Bhai, responses.csv nahi mila! Check kar project folder mein.")
        except pd.errors.ParserError as e:
            raise ValueError(f"Bhai, responses.csv ka format galat hai! Error: {str(e)}")

    def fetch_app_data(self, context_needed):
        """
        Fetch real-time data from VolGuard Pro's data pipeline.
        Primary: st.session_state.analysis_df.
        Fallback 1: Upstox API via fetch_market_depth_by_scrip.
        Fallback 2: Static fallback_data.csv.
        """
        try:
            if "analysis_df" in st.session_state and st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
                df = st.session_state.analysis_df
                latest_data = df.iloc[-1]
                portfolio_data = st.session_state.get("api_portfolio_data", {})
                option_chain = st.session_state.real_time_market_data.get("option_chain", pd.DataFrame()) if st.session_state.real_time_market_data else pd.DataFrame()
                atm_strike = st.session_state.real_time_market_data.get("atm_strike", "N/A") if st.session_state.real_time_market_data else "N/A"
                atm_data = option_chain[option_chain["StrikeRate"] == atm_strike] if not option_chain.empty else pd.DataFrame()
                ce_iv = atm_data[atm_data["CPType"] == "CE"]["IV"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 30.0
                ce_gamma = atm_data[atm_data["CPType"] == "CE"]["Gamma"].iloc[0] if "Gamma" in atm_data.columns and not atm_data[atm_data["CPType"] == "CE"].empty else 0.05
                ce_delta = atm_data[atm_data["CPType"] == "CE"]["Delta"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0.4
                ce_theta = atm_data[atm_data["CPType"] == "CE"]["Theta"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else -0.02
                ce_vega = atm_data[atm_data["CPType"] == "CE"]["Vega"].iloc[0] if not atm_data[atm_data["CPType"] == "CE"].empty else 0.1
                data = {
                    "iv": ce_iv,
                    "gamma": ce_gamma,
                    "delta": ce_delta,
                    "theta": ce_theta,
                    "vega": ce_vega,
                    "vix": latest_data.get("VIX", 25.0),
                    "ivp": latest_data.get("IVP", 75.0),
                    "ivp_status": "high" if latest_data.get("IVP", 75.0) > 70 else "low",
                    "margin": (portfolio_data.get("margin", {}).get("available_margin", 0.0) / st.session_state.get("capital", 1000000) * 100) or 85.0,
                    "strike": atm_strike,
                    "fii_flow": latest_data.get("FII_Index_Fut_Pos", 0.0) / 1e4,  # Convert to crores
                    "dii_flow": latest_data.get("FII_Option_Pos", 0.0) / 1e4,  # Proxy for DII
                    "buzz_topic": "NIFTY expiry",
                    "sentiment": 60,
                    "news_headline": "No major news",
                    "impact": 50,
                    "community_hedge": 70,
                    "trade_count": len(portfolio_data.get("trade_book", {}).get("data", [])),
                    "trade_frequency": len(portfolio_data.get("trade_book", {}).get("data", [])) / 7 or 2,
                    "revenge_count": 2,
                    "bias": "FOMO",
                    "bias_count": 3,
                    "strategy": st.session_state.get("active_strategy_details", {}).get("Strategy", "Iron Condor"),
                    "win_rate": latest_data.get("Win_Rate", 60),
                    "reason": "IV spike",
                    "dte": latest_data.get("Days_to_Expiry", 5),
                    "pnl": sum(pos.get("unrealized_mtm", 0.0) for pos in portfolio_data.get("positions", {}).get("data", [])) / st.session_state.get("capital", 1000000) * 100 or 2.5,
                    "loss_risk": 15,
                    "breakeven": "N/A"
                }
            else:
                raise ValueError("Analysis DataFrame not available")

        except Exception as e:
            try:
                upstox_client = st.session_state.get("client")
                if upstox_client and upstox_client.get("access_token"):
                    market_data = fetch_market_depth_by_scrip(upstox_client, instrument_key="NSE_FO|46274")  # Example ATM CE
                    ltp = market_data["Data"][0].get("LastTradedPrice", 0.0) if market_data and market_data.get("Data") else 0.0
                    portfolio_data = st.session_state.get("api_portfolio_data", {})
                    data = {
                        "iv": 30.0,
                        "gamma": 0.05,
                        "delta": 0.4,
                        "theta": -0.02,
                        "vega": 0.1,
                        "vix": 25.0,
                        "ivp": 75.0,
                        "ivp_status": "high",
                        "margin": (portfolio_data.get("margin", {}).get("available_margin", 0.0) / st.session_state.get("capital", 1000000) * 100) or 85.0,
                        "strike": "N/A",
                        "fii_flow": 0.0,
                        "dii_flow": 0.0,
                        "buzz_topic": "NIFTY expiry",
                        "sentiment": 60,
                        "news_headline": "No major news",
                        "impact": 50,
                        "community_hedge": 70,
                        "trade_count": len(portfolio_data.get("trade_book", {}).get("data", [])),
                        "trade_frequency": len(portfolio_data.get("trade_book", {}).get("data", [])) / 7 or 2,
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
                    raise ValueError("Upstox client not available")
            except Exception as e2:
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

        return {key: data.get(key, "N/A") for key in context_needed.split(",")}

    def generate_response(self, user_query):
        user_query = user_query.lower().strip()
        for _, row in self.responses.iterrows():
            pattern = row["query_pattern"]
            if re.search(pattern, user_query):
                context = self.fetch_app_data(row["context_needed"])
                try:
                    response = row["response_template"].format(**context)
                    return response
                except KeyError:
                    return "Bhai, data thoda off lag raha hai. Try again! Do your own research!"
        return "Bhai, yeh query thodi alag hai. Kya bol raha hai, thoda clearly bata? 😜 Do your own research!"

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
