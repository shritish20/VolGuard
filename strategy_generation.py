import logging
import pandas as pd
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital, current_violations, journal_complete):
    """
    Generates a trading strategy based on market conditions and forecasts.
    """
    try:
        logger.info("Generating trading strategy")
        df = df.copy()
        df.index = df.index.normalize()
        if df.empty:
             logger.error("Cannot generate strategy: Input DataFrame is empty.")
             return None

        # Discipline Lock check
        if current_violations >= 2 and not journal_complete:
             logger.info("Discipline Lock active. Cannot generate strategy.")
             return {"Discipline_Lock": True} # Indicate lock is active


        latest = df.iloc[-1]
        # Ensure required columns exist in the latest row
        required_latest_cols = ["ATM_IV", "Realized_Vol", "IV_Skew", "PCR", "Days_to_Expiry", "Event_Flag", "VIX", "Spot_MaxPain_Diff_Pct", "PnL_Day", "VIX_Change_Pct"]
        if not all(col in latest.index for col in required_latest_cols):
             missing = [col for col in required_latest_cols if col not in latest.index]
             logger.error(f"Missing required columns in latest data for strategy generation: {missing}")
             return None # Indicate failure


        # Use realized_vol as a fallback if forecast_log is missing or empty
        avg_vol = np.mean(forecast_log["Blended_Vol"]) if forecast_log is not None and not forecast_log.empty and 'Blended_Vol' in forecast_log.columns and not forecast_log["Blended_Vol"].isna().all() else realized_vol

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
        vix_change_pct = latest["VIX_Change_Pct"]


        risk_flags = []
        if latest_vix > 25:
            risk_flags.append("VIX > 25% - High Volatility Risk")
        if spot_max_pain_diff_pct > 3: # Adjust threshold
            risk_flags.append(f"Spot-Max Pain Diff > {spot_max_pain_diff_pct:.1f}% - Potential Pinning Risk")
        if pnl_day < -0.01 * capital: # Daily loss > 1% of capital
            risk_flags.append(f"Recent Daily Loss ({pnl_day:,.0f} â‚¹ ) - Consider reducing size")
        if vix_change_pct > 8: # Adjust threshold
            risk_flags.append(f"High VIX Spike Detected ({vix_change_pct:+.1f}%)")


        # Regime determination based on average forecast volatility
        if avg_vol < 15:
            regime = "LOW"
        elif avg_vol < 20:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        # Add Event-Driven regime check
        if event_flag == 1 or dte <= 3: # Within 3 days of expiry or explicit event flag
             regime = "EVENT-DRIVEN"


        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0

        # Strategy selection logic - refine based on combined factors
        if regime == "LOW":
            if iv_hv_gap > 2 and dte < 10:
                strategy = "Butterfly Spread"
                reason = "Low forecast vol, IV > HV, near expiry favors pinning"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.5
            elif iv_skew < -1: # Negative skew
                 strategy = "Short Put" # Simple directional play
                 reason = "Low forecast vol, negative IV skew suggests put selling opportunity"
                 tags = ["Directional", "Bullish", "Premium Selling"]
                 risk_reward = 1.5
            else:
                strategy = "Iron Fly"
                reason = "Low forecast volatility favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8

        elif regime == "MEDIUM":
            if iv_hv_gap > 1.5 and iv_skew > 0.5:
                strategy = "Iron Condor"
                reason = "Medium forecast vol, IV > HV, positive skew favors Iron Condor"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.5
            elif pcr > 1.1:
                 strategy = "Short Put Vertical Spread" # Bullish bias
                 reason = "Medium forecast vol, bullish PCR suggests defined risk put spread"
                 tags = ["Directional", "Bullish", "Defined Risk"]
                 risk_reward = 1.2
            elif pcr < 0.9:
                 strategy = "Short Call Vertical Spread" # Bearish bias
                 reason = "Medium forecast vol, bearish PCR suggests defined risk call spread"
                 tags = ["Directional", "Bearish", "Defined Risk"]
                 risk_reward = 1.2
            else:
                strategy = "Short Strangle"
                reason = "Medium forecast vol, balanced indicators, premium capture"
                tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                risk_reward = 1.6

        elif regime == "HIGH":
            if iv_hv_gap > 5 or latest_vix > 28: # High IV spike
                strategy = "Jade Lizard"
                reason = "High IV spike, IV > HV favors Jade Lizard for defined upside"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.0
            elif iv_skew < -2: # Extreme negative skew
                 strategy = "Long Put" # Protective or bearish play
                 reason = "High forecast vol, extreme negative skew suggests downside protection"
                 tags = ["Directional", "Bearish", "Protection"]
                 risk_reward = 2.0
            else:
                 strategy = "Iron Condor" # Can still work in high vol if range is wide enough
                 reason = "High forecast vol, wide expected range favors Iron Condor premium"
                 tags = ["Neutral", "Theta", "Range Bound"]
                 risk_reward = 1.3

        elif regime == "EVENT-DRIVEN":
            if iv > 30 and dte < 3:
                strategy = "Short Straddle"
                reason = "High IV, very near expiry event â€” max premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.8
            elif dte < 7: # Near expiry event
                strategy = "Calendar Spread"
                reason = "Event-based uncertainty and near expiry favors term structure opportunity"
                tags = ["Volatility", "Event", "Calendar"]
                risk_reward = 1.5
            else: # Event further out, but still impactful
                 strategy = "Iron Condor" # Capture premium before the event
                 reason = "Event anticipation favors capturing premium with Iron Condor"
                 tags = ["Neutral", "Event", "Range Bound"]
                 risk_reward = 1.4


        # Confidence score from forecast function
        confidence_score_from_forecast = confidence_score if confidence_score is not None else 50 # Use calculated confidence, default to 50

        # Capital allocation based on regime and risk tolerance
        capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.08) # Match backtest allocation
        position_size_multiplier = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc_pct * position_size_multiplier # Scale by risk tolerance

        # Max loss calculation matching backtest logic
        max_loss_pct = {"Iron Condor": 0.02, "Butterfly Spread": 0.015, "Iron Fly": 0.02, "Short Strangle": 0.03, "Calendar Spread": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04, "Short Put Vertical Spread": 0.015, "Long Put": 0.03, "Short Put": 0.02, "Short Call Vertical Spread": 0.015}.get(strategy, 0.025)
        max_loss = deploy * max_loss_pct


        total_exposure = deploy / capital if capital > 0 else 0

        # Behavior Score
        behavior_score = 8 if total_exposure < 0.10 else 6 # Adjust threshold to 10% exposure
        behavior_warnings = ["Consider reducing position size (Exposure > 10%)"] if behavior_score < 8 else []


        logger.debug(f"Trading strategy generated: {strategy}")
        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence_score_from_forecast, # Use calculated confidence
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure * 100, # Display as percentage
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings,
            "Discipline_Lock": False # Indicate lock is NOT active
        }
    except Exception as e:
        logger.error(f"Error generating strategy: {str(e)}", exc_info=True)
        return None

