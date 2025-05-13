import logging
import pandas as pd
import numpy as np
# Assuming FEATURE_COLS is defined in data_processing or a common config file
# from data_processing import FEATURE_COLS # Import FEATURE_COLS if not defined here

# Setup logging for this module
logger = logging.getLogger(__name__)

# --- Define FEATURE_COLS if not imported from data_processing ---
# Ensure FEATURE_COLS is defined if it's not imported globally
# If you have a common config.py, import it from there.
try:
    from data_processing import FEATURE_COLS
    logger.info("FEATURE_COLS imported from data_processing.")
except ImportError:
    logger.warning("FEATURE_COLS not found in data_processing. Using a default list. Ensure consistency!")
    # Define a default list if import fails - MUST match features generated in data_processing
    FEATURE_COLS = [
        'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew',
        'Straddle_Price', 'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry',
        'Event_Flag', 'FII_Index_Fut_Pos', 'FII_Option_Pos', 'Realized_Vol',
        'Advance_Decline_Ratio', 'Capital_Pressure_Index', 'Gamma_Bias',
        'NIFTY_Close', 'Total_Capital' # Assuming these are the features generated
    ]


# === Strategy Generation Function ===

def generate_trading_strategy(analysis_df: pd.DataFrame, real_time_data: dict, forecast_metrics: dict, capital: float, risk_tolerance: str):
    """
    Generates a trading strategy recommendation based on market conditions,
    volatility forecast, and user risk tolerance.

    Args:
        analysis_df (pd.DataFrame): DataFrame containing historical and latest
                                   market data with generated features.
                                   Must have a datetime index and columns used by
                                   the strategy logic. Expected to be sorted by date.
                                   The last row contains the latest data.
        real_time_data (dict): Dictionary containing the latest real-time market data.
                               Fetched directly from the API. May contain overlapping
                               info with analysis_df's last row but is preferred
                               for latest snapshot values (e.g., ATM Strike, Option Chain).
        forecast_metrics (dict): Dictionary containing volatility forecast metrics
                                 (e.g., 'forecasted_vix', 'confidence', 'rmse', 'ivp', 'vix').
        capital (float): The user's total trading capital.
        risk_tolerance (str): User's risk tolerance level ("Conservative", "Moderate", "Aggressive").

    Returns:
        dict or None: A dictionary containing the recommended strategy details
                      ('Strategy', 'Confidence', 'Deploy', 'Reasoning'),
                      or None if strategy generation is not possible or no
                      strategy is recommended.
    """
    logger.info(f"Generating trading strategy for capital ₹{capital} and risk profile '{risk_tolerance}'.")

    # --- 1. Validate and Extract Key Metrics ---
    # Ensure essential data is available before proceeding
    if analysis_df is None or analysis_df.empty or real_time_data is None or not real_time_data or forecast_metrics is None or not forecast_metrics:
        logger.error("Missing essential data (analysis_df, real_time_data, or forecast_metrics) for strategy generation.")
        return None

    # Extract latest data and forecast metrics safely
    try:
        # Get the last row from the analysis DataFrame as a Series. Safely access columns.
        latest_analysis_data = analysis_df.iloc[-1]
        latest_analysis_data = latest_analysis_data.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0.0) # Ensure numeric and fill NaNs for safety


        # Safely extract metrics from the latest analysis data and forecast metrics
        # Use .get() with default fallbacks and pd.to_numeric for safety
        vix = pd.to_numeric(forecast_metrics.get('vix', latest_analysis_data.get('VIX')), errors='coerce').mean() if forecast_metrics.get('vix') is not None else pd.to_numeric(latest_analysis_data.get('VIX'), errors='coerce').mean() # Prioritize live VIX if available in forecast_metrics
        ivp = pd.to_numeric(forecast_metrics.get('ivp', latest_analysis_data.get('IVP')), errors='coerce').mean() if forecast_metrics.get('ivp') is not None else pd.to_numeric(latest_analysis_data.get('IVP'), errors='coerce').mean() # Prioritize IVP from forecast_metrics if available
        pcr = pd.to_numeric(real_time_data.get('pcr', latest_analysis_data.get('PCR')), errors='coerce').mean() if real_time_data.get('pcr') is not None else pd.to_numeric(latest_analysis_data.get('PCR'), errors='coerce').mean() # Prioritize live PCR
        forecasted_vix = pd.to_numeric(forecast_metrics.get('forecasted_vix'), errors='coerce').mean() if forecast_metrics.get('forecasted_vix') is not None else None # Forecasted VIX is only from forecast_metrics
        forecast_confidence = pd.to_numeric(forecast_metrics.get('confidence'), errors='coerce').mean() if forecast_metrics.get('confidence') is not None else 50.0 # Default confidence
        vix_change_pct = pd.to_numeric(latest_analysis_data.get('VIX_Change_Pct'), errors='coerce').mean()
        event_flag = int(latest_analysis_data.get('Event_Flag', 0)) # Ensure integer for flag
        dte = pd.to_numeric(latest_analysis_data.get('Days_to_Expiry'), errors='coerce').mean()

        # Calculate VIX trend (simplified: based on recent VIX change percentage direction)
        vix_trend = "Rising" if vix_change_pct > 2.0 else ("Falling" if vix_change_pct < -2.0 else "Neutral") # Thresholds (e.g., 2%) for significant change

        # Ensure key metrics are numeric and not NaN before using in logic
        vix = vix if pd.notna(vix) else 15.0
        ivp = ivp if pd.notna(ivp) else 50.0
        pcr = pcr if pd.notna(pcr) else 1.0
        forecasted_vix = forecasted_vix if pd.notna(forecasted_vix) else vix # Default forecasted VIX to current VIX if missing
        dte = dte if pd.notna(dte) else 7 # Default DTE to 7 if missing

        logger.info(f"Key Metrics: VIX={vix:.2f}, IVP={ivp:.2f}, PCR={pcr:.2f}, ForecastedVIX={forecasted_vix:.2f}, Confidence={forecast_confidence:.2f}%, VIX Trend={vix_trend}, DTE={dte:.1f}, Event={event_flag}")


    except Exception as e:
        logger.error(f"Error extracting key metrics for strategy generation: {str(e)}", exc_info=True)
        # Return None if metric extraction fails
        return None


    # --- 2. Define Strategy Parameters and Thresholds ---
    # Define thresholds for risk tolerance and market conditions
    # These thresholds are examples and should be based on trading strategy rules.
    risk_thresholds = {
        "Conservative": {"ivp_high": 80, "ivp_low": 20, "vix_high": 22, "vix_low": 14, "dte_min": 10},
        "Moderate": {"ivp_high": 70, "ivp_low": 30, "vix_high": 20, "vix_low": 15, "dte_min": 7},
        "Aggressive": {"ivp_high": 60, "ivp_low": 40, "vix_high": 18, "vix_low": 16, "dte_min": 5}
    }

    # Get thresholds based on user's risk tolerance. Default to Moderate if invalid.
    current_thresholds = risk_thresholds.get(risk_tolerance, risk_thresholds["Moderate"])
    logger.debug(f"Using thresholds for '{risk_tolerance}' profile: {current_thresholds}")


    # --- 3. Strategy Decision Logic (Rule-Based) ---
    # This is the core logic. Strategies are recommended based on combinations
    # of market conditions (VIX, IVP, PCR, trend), forecast, and risk tolerance.
    # The reasoning string explains the factors leading to the decision.

    recommended_strategy = "No Strategy" # Default if no rules are met
    confidence = forecast_confidence # Base confidence on forecast confidence
    deploy_amount = 0.0 # Default deploy amount
    reasoning = "Current market conditions do not match a predefined strategy pattern, or data is insufficient."


    # Example Rules (Expand and refine this section based on actual strategy rules):

    # Rule 1: Short Straddle (Typically when IV is High and expected to fall or stay flat)
    if (ivp > current_thresholds["ivp_high"] and # IVP is high
        vix < current_thresholds["vix_high"] and # VIX is not excessively high (avoiding extreme fear)
        (forecasted_vix < vix or abs(forecasted_vix - vix) < 1.0) and # Forecasted VIX is lower or similar to current VIX
        dte > current_thresholds["dte_min"] and # Sufficient days to expiry
        event_flag == 0): # No major event near expiry

        recommended_strategy = "Short Straddle"
        reasoning = (
            f"Conditions favor Short Straddle: IVP is high ({ivp:.2f}), VIX is manageable ({vix:.2f}), "
            f"forecast suggests VIX might decrease or stay flat ({forecasted_vix:.2f}), "
            f"and there are sufficient days to expiry ({dte:.1f})."
        )
        # Deploy logic for Short Straddle: Deploy a percentage of capital, possibly scaled by IV/Straddle Price
        # Simple approach: Deploy a fixed percentage (e.g., 5%) scaled by Risk Tolerance (Conservative=lower %, Aggressive=higher %)
        risk_scale = {"Conservative": 0.03, "Moderate": 0.05, "Aggressive": 0.07}.get(risk_tolerance, 0.05)
        deploy_amount = capital * risk_scale # Example deployment calculation

        # Further adjust confidence based on how strongly conditions match (e.g., higher IVP, stronger forecast direction)
        confidence += (ivp - current_thresholds["ivp_high"]) * 0.5 # Add points for being further above high IVP threshold
        confidence = np.clip(confidence, 0, 100) # Ensure confidence is between 0 and 100


    # Rule 2: Short Strangle (Similar to Straddle, but typically when IV is High/Medium and looking for range-bound movement)
    elif (ivp > current_thresholds["ivp_low"] and # IVP is at least medium
          vix < current_thresholds["vix_high"] and # VIX is not excessively high
          (forecasted_vix < vix or abs(forecasted_vix - vix) < 2.0) and # Forecast suggests VIX might decrease or stay relatively flat
          dte > current_thresholds["dte_min"] + 3 and # More days to expiry than Straddle (Strangle needs more time)
          event_flag == 0):

        recommended_strategy = "Short Strangle"
        reasoning = (
            f"Conditions favor Short Strangle: IVP is medium to high ({ivp:.2f}), VIX is manageable ({vix:.2f}), "
            f"forecast suggests VIX might decrease or stay relatively flat ({forecasted_vix:.2f}), "
            f"and there is ample time to expiry ({dte:.1f}). Suitable for range-bound view."
        )
        # Deploy logic for Short Strangle (similar to Straddle, maybe slightly higher due to wider range, or same)
        risk_scale = {"Conservative": 0.04, "Moderate": 0.06, "Aggressive": 0.08}.get(risk_tolerance, 0.06)
        deploy_amount = capital * risk_scale

        confidence += (ivp - current_thresholds["ivp_low"]) * 0.4 # Add points for being further above low IVP threshold
        confidence = np.clip(confidence, 0, 100)


    # Rule 3: Long Straddle / Strangle (Typically when IV is Low and expected to increase significantly)
    elif (ivp < current_thresholds["ivp_low"] and # IVP is low
          vix > current_thresholds["vix_low"] and # VIX is not extremely low (some volatility present)
          forecasted_vix > vix + 2.0 and # Forecasted VIX is significantly higher than current VIX
          dte > current_thresholds["dte_min"] and # Sufficient days to expiry
          event_flag == 0): # No immediate event (Long options lose value quickly near expiry)

        recommended_strategy = "Long Straddle / Strangle" # Grouping similar strategies
        reasoning = (
            f"Conditions favor Long Volatility (Straddle/Strangle): IVP is low ({ivp:.2f}), "
            f"VIX is not excessively low ({vix:.2f}), and the forecast strongly suggests VIX "
            f"will increase ({forecasted_vix:.2f}). Sufficient days to expiry ({dte:.1f})."
        )
        # Deploy logic for Long Volatility: Fixed small percentage of capital (risk defined by premium paid)
        risk_scale = {"Conservative": 0.01, "Moderate": 0.02, "Aggressive": 0.03}.get(risk_tolerance, 0.02)
        deploy_amount = capital * risk_scale # Deploy a small percentage based on premium cost

        confidence += (forecasted_vix - vix) * 1.0 # Add points for stronger forecast increase
        confidence = np.clip(confidence, 0, 100)


    # Rule 4: Iron Condor / Iron Fly (Defined risk strategies, often used in moderate IV environments with a range-bound view)
    elif (ivp >= current_thresholds["ivp_low"] and ivp <= current_thresholds["ivp_high"] and # IVP is in medium range
          vix >= current_thresholds["vix_low"] and vix <= current_thresholds["vix_high"] and # VIX is in medium range
          abs(forecasted_vix - vix) < 2.0 and # Forecast suggests relatively flat VIX
          dte > current_thresholds["dte_min"] + 5 and # Sufficient days to expiry for multi-leg strategy
          event_flag == 0):

        recommended_strategy = "Iron Condor" # Or "Iron Fly" depending on strike selection logic (using Condor as general example)
        reasoning = (
            f"Conditions favor Iron Condor: IVP ({ivp:.2f}) and VIX ({vix:.2f}) are in the medium range, "
            f"forecast suggests VIX will remain relatively flat ({forecasted_vix:.2f}), "
            f"and there is ample time to expiry ({dte:.1f}). Defined risk strategy."
        )
        # Deploy logic for Iron Condor: Deploy based on max loss percentage relative to capital
        # Example: Max loss is typically the width of the wings - premium received.
        # Here, approximate deploy as a percentage of capital based on risk tolerance for defined risk.
        risk_scale = {"Conservative": 0.05, "Moderate": 0.08, "Aggressive": 0.10}.get(risk_tolerance, 0.08)
        deploy_amount = capital * risk_scale

        confidence += (100 - abs(ivp - 50) - abs(vix - 17.5)) * 0.2 # Add points for IVP/VIX closer to mid-range (50/17.5)
        confidence = np.clip(confidence, 0, 100)


    # Rule 5: Event Day Strategy (e.g., Calendar Spread, or avoid trading if high risk)
    # This is highly strategy dependent. Could be specific spreads or simply advising caution.
    elif event_flag == 1: # An event is near expiry
         if dte <= 3 and risk_tolerance in ["Conservative", "Moderate"]:
             recommended_strategy = "No Strategy" # Or "Avoid Trading"
             reasoning = f"Event day approaching (DTE <= {expiry_threshold_days}). Market volatility can be unpredictable. Recommend caution/no new trades for conservative profiles."
             deploy_amount = 0.0 # No deployment

         else: # More aggressive or different event strategy (e.g., Calendar Spread across expiries)
             recommended_strategy = "Calendar Spread" # Example Event strategy
             reasoning = f"Event day approaching (DTE <= {expiry_thresholds_days}). Conditions may favor Calendar Spreads to trade volatility differences across expiries."
             # Deploy logic for Calendar Spread (depends on spread width, time decay)
             risk_scale = {"Conservative": 0.02, "Moderate": 0.04, "Aggressive": 0.06}.get(risk_tolerance, 0.04)
             deploy_amount = capital * risk_scale

         confidence = np.clip(confidence - 20, 0, 100) # Lower confidence slightly around events due to unpredictability


    # Add more strategy rules here based on other combinations of metrics and risk tolerance...
    # Example: Bear Call Spread (Bearish view, moderate IV), Bull Put Spread (Bullish view, moderate IV)
    # Example: Ratio Spreads, Butterfly Spreads, etc.

    # Ensure deploy amount is within sensible bounds (e.g., max 50% of capital for aggressive strategies)
    deploy_amount = np.clip(deploy_amount, 0, capital * 0.50) # Cap deploy amount


    # --- 4. Prepare Results Dictionary ---
    # Return the recommended strategy details
    strategy_details = {
        "Strategy": recommended_strategy,
        "Confidence": float(confidence), # Ensure float type
        "Deploy": float(deploy_amount), # Ensure float type
        "Reasoning": reasoning
    }

    logger.info(f"Strategy generated: {strategy_details}")

    # Return None if the strategy is "No Strategy" or if deploy amount is effectively zero
    if recommended_strategy == "No Strategy" or deploy_amount < capital * 0.005: # Consider very small deploy amounts as effectively no strategy
        logger.info("No specific strategy recommended or deploy amount too small.")
        return None
    else:
        return strategy_details


# The __main__ block is removed as this module's functions are intended to be imported
# and used by other parts of the application.
# To test this function, you would typically call it from a separate script
# with dummy analysis_df, real_time_data, and forecast_metrics dictionaries,
# along with dummy capital and risk tolerance values.
