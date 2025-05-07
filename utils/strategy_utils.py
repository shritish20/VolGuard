import streamlit as st
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital):
    """Generates a trading strategy based on market conditions."""
    try:
        logger.info("Generating trading strategy")
        df = df.copy()
        df.index = df.index.normalize()
        if df.empty:
            logger.error("Empty DataFrame for strategy generation.")
            return None

        latest = df.iloc[-1]
        required_cols = ["ATM_IV", "Realized_Vol", "IV_Skew", "PCR", "Days_to_Expiry", "Event_Flag", "VIX", "Spot_MaxPain_Diff_Pct", "PnL_Day"]
        if not all(col in latest.index for col in required_cols):
            missing = [col for col in required_cols if col not in latest.index]
            logger.error(f"Missing columns: {missing}")
            st.error(f"Cannot generate strategy: Missing data ({', '.join(missing)})")
            return None

        avg_vol = np.mean(forecast_log["Blended_Vol"]) if forecast_log is not None and not forecast_log.empty else latest["Realized_Vol"]
        iv = latest["ATM_IV"]
        hv = latest["Realized_Vol"]
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"]
        pcr = latest["PCR"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]
        latest_vix = latest["VIX"]
        spot_max_pain_diff_pct = latest["Spot_MaxPain_Diff_Pct"]
        pnl_day = latest["PnL_Day"]

        risk_flags = []
        if latest_vix > 25:
            risk_flags.append("VIX > 25% - High Volatility Risk")
        if spot_max_pain_diff_pct > 3:
            risk_flags.append(f"Spot-Max Pain Diff > {spot_max_pain_diff_pct:.1f}% - Pinning Risk")
        if pnl_day < -0.01 * capital:
            risk_flags.append(f"Recent Loss ({pnl_day:,.0f} ₹) - Reduce size")
        if latest["VIX_Change_Pct"] > 8:
            risk_flags.append(f"VIX Spike ({latest['VIX_Change_Pct']:+.1f}%)")

        if risk_flags:
            st.session_state.violations += 1
            if st.session_state.violations >= 2 and not st.session_state.journal_complete:
                return None

        regime = "LOW" if avg_vol < 15 else "MEDIUM" if avg_vol < 20 else "HIGH"
        if event_flag == 1 or dte <= 3:
            regime = "EVENT-DRIVEN"

        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0

        if regime == "LOW":
            if iv_hv_gap > 2 and dte < 10:
                strategy = "Butterfly Spread"
                reason = "Low vol, IV > HV, near expiry favors pinning"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.5
            elif iv_skew < -1:
                strategy = "Short Put"
                reason = "Low vol, negative skew suggests put selling"
                tags = ["Directional", "Bullish", "Premium Selling"]
                risk_reward = 1.5
            else:
                strategy = "Iron Fly"
                reason = "Low volatility favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8
        elif regime == "MEDIUM":
            if iv_hv_gap > 1.5 and iv_skew > 0.5:
                strategy = "Iron Condor"
                reason = "Medium vol, IV > HV, positive skew"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.5
            elif pcr > 1.1:
                strategy = "Short Put Vertical Spread"
                reason = "Medium vol, bullish PCR"
                tags = ["Directional", "Bullish", "Defined Risk"]
                risk_reward = 1.2
            elif pcr < 0.9:
                strategy = "Short Call Vertical Spread"
                reason = "Medium vol, bearish PCR"
                tags = ["Directional", "Bearish", "Defined Risk"]
                risk_reward = 1.2
            else:
                strategy = "Short Strangle"
                reason = "Medium vol, premium capture"
                tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                risk_reward = 1.6
        elif regime == "HIGH":
            if iv_hv_gap > 5 or latest_vix > 28:
                strategy = "Jade Lizard"
                reason = "High IV spike, IV > HV"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.0
            elif iv_skew < -2:
                strategy = "Long Put"
                reason = "High vol, negative skew suggests protection"
                tags = ["Directional", "Bearish", "Protection"]
                risk_reward = 2.0
            else:
                strategy = "Iron Condor"
                reason = "High vol, wide range favors premium"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.3
        elif regime == "EVENT-DRIVEN":
            if iv > 30 and dte < 3:
                strategy = "Short Straddle"
                reason = "High IV, near expiry for premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.8
            else:
                strategy = "Calendar Spread"
                reason = "Event uncertainty favors term structure"
                tags = ["Volatility", "Event", "Calendar"]
                risk_reward = 1.5

        confidence_score_from_forecast = confidence_score if confidence_score is not None else 50
        capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.08)
        position_size_multiplier = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc_pct * position_size_multiplier
        max_loss_pct = {
            "Iron Condor": 0.02, "Butterfly Spread": 0.015, "Iron Fly": 0.02, "Short Strangle": 0.03,
            "Calendar Spread": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04,
            "Short Put Vertical Spread": 0.015, "Long Put": 0.03, "Short Put": 0.02,
            "Short Call Vertical Spread": 0.015
        }.get(strategy, 0.025)
        max_loss = deploy * max_loss_pct
        total_exposure = deploy / capital if capital > 0 else 0

        behavior_score = 8 if total_exposure < 0.10 else 6
        behavior_warnings = ["Exposure > 10%, reduce size"] if behavior_score < 8 else []

        logger.debug(f"Strategy generated: {strategy}")
        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence_score_from_forecast,
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure * 100,
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings
        }
    except Exception as e:
        st.error(f"Error generating strategy: {str(e)}")
        logger.error(f"Error generating strategy: {str(e)}")
        return None
