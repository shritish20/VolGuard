import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import math # Import math for log and sqrt

from data_processing import FEATURE_COLS # Import FEATURE_COLS

# Setup logging
logger = logging.getLogger(__name__)

# --- Constants for Transaction Costs (Illustrative - Adjust Based on Broker) ---
# These are examples; check your broker's charges
BROKERAGE_PER_LOT = 20 # Example: ₹20 per lot per side (buy/sell)
EXCHANGE_TRANSACTION_CHARGE_PCT = 0.00053 # Example: NSE F&O percentage
SEBI_TURNOVER_FEE_PCT = 0.0001 # Example: SEBI fee percentage
STT_SELL_OPTIONS_PCT = 0.017 # Example: 0.017% on Sell side (Premium Value)
CLEARING_CHARGE_PER_LOT = 1 # Example: ₹1 per lot per side
GST_ON_TOTAL_COSTS_PCT = 0.18 # 18% GST on Brokerage + Exchange Charges + SEBI Fee + Clearing Charge
STAMP_DUTY_PCT = 0.003 # Example: 0.003% on Buy side (Premium Value)

# --- Black-Scholes Option Pricing Model ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the theoretical price of a European option using the Black-Scholes model.

    Parameters:
    S (float): Underlying asset price
    K (float): Strike price
    T (float): Time to expiry in years (e.g., 30 days / 365)
    r (float): Risk-free interest rate (annual, decimal)
    sigma (float): Volatility (annual, decimal)
    option_type (str): 'call' for a call option, 'put' for a put option.

    Returns:
    float: The theoretical option price.
    """
    # Prevent division by zero or log of zero/negative
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")

    return max(0.0, price) # Option price cannot be negative

# --- Calculate Transaction Costs ---
def calculate_transaction_costs(strategy, trade_type, num_lots, premium_per_lot):
    """
    Calculates estimated transaction costs for a trade leg.

    Parameters:
    strategy (str): The trading strategy name.
    trade_type (str): "BUY" or "SELL"
    num_lots (int): Number of lots traded.
    premium_per_lot (float): Premium paid/received per lot.

    Returns:
    float: Estimated total transaction cost for this leg.
    """
    if num_lots <= 0 or premium_per_lot < 0:
        return 0.0

    brokerage = BROKERAGE_PER_LOT * num_lots
    # Exchange charges, SEBI fee based on turnover (lots * lot_size * underlying_price_approx - or use premium?)
    # Using premium value is more accurate for options transaction costs
    turnover = premium_per_lot * num_lots * 25 # Assuming Nifty lot size 25 - make this dynamic?
    exchange_charge = turnover * EXCHANGE_TRANSACTION_CHARGE_PCT
    sebi_fee = turnover * SEBI_TURNOVER_FEE_PCT
    clearing_charge = CLEARING_CHARGE_PER_LOT * num_lots

    total_statutory_charges_base = exchange_charge + sebi_fee + clearing_charge
    gst = (brokerage + total_statutory_charges_base) * GST_ON_TOTAL_COSTS_PCT

    stt = 0.0
    stamp_duty = 0.0

    if trade_type == "SELL":
        # STT on premium value on the sell side for options
        stt = premium_per_lot * num_lots * 25 * STT_SELL_OPTIONS_PCT
    elif trade_type == "BUY":
        # Stamp duty on premium value on the buy side
        stamp_duty = premium_per_lot * num_lots * 25 * STAMP_DUTY_PCT # Stamp duty is often very small/negligible

    total_costs = brokerage + total_statutory_charges_base + gst + stt + stamp_duty

    return total_costs


def run_strategy_engine(day_data, avg_vol_forecast, portfolio_pnl, capital):
    """
    Determines the trading strategy based on market regime and indicators.
    (Logic remains similar, but deployment/max_loss are based on strategy type)
    """
    # ... (Strategy engine logic is the same as before) ...
    try:
        # Use day_data (real historical/live features) for strategy decision
        iv = day_data.get("ATM_IV", 0.0)
        hv = day_data.get("Realized_Vol", 0.0)
        iv_hv_gap = iv - hv
        iv_skew = day_data.get("IV_Skew", 0.0)
        dte = day_data.get("Days_to_Expiry", 0)
        event_flag = day_data.get("Event_Flag", 0)
        pcr = day_data.get("PCR", 1.0)
        vix_change_pct = day_data.get("VIX_Change_Pct", 0.0)

        # Drawdown limit check based on total capital
        if portfolio_pnl < -0.10 * capital: # 10% drawdown limit
            return None, None, "Portfolio drawdown limit reached", [], 0, 0, 0, [] # Added empty list for legs


        # Determine regime based on blended forecast volatility (avg_vol_forecast)
        if avg_vol_forecast is None:
             regime = "MEDIUM" # Default if forecast failed
        elif avg_vol_forecast < 15:
            regime = "LOW"
        elif avg_vol_forecast < 20:
            regime = "MEDIUM"
        else:
            regime = "HIGH"

        # Add Event-Driven regime check
        if event_flag == 1 or dte <= 3: # Within 3 days of expiry or explicit event flag
             regime = "EVENT-DRIVEN"


        strategy = "Undefined"
        reason = "N/A"
        tags = []
        risk_reward = 1.0 # Base risk-reward
        strategy_legs_definition = [] # Define the legs for the chosen strategy


        # Strategy selection logic based on regime and real-time indicators
        # Define legs with (strike_offset_from_atm, type, buy_sell, quantity_multiplier)
        # strike_offset_from_atm: 0 for ATM, +ve for OTM Calls / ITM Puts, -ve for ITM Calls / OTM Puts
        # This is a simplification; actual strikes would need to be looked up in the option chain data.
        # For backtesting simulation, we'll derive strikes relative to the day's Nifty close.
        # Let's use a simple points offset for strike selection in simulation for now.
        # A more advanced backtest would find actual tradable strikes near these offsets.

        strike_step = 50 # Nifty strike increments

        if regime == "LOW":
            if iv_hv_gap > 3 and dte < 15:
                strategy = "Butterfly Spread (Call)"
                reason = "Low vol & moderate expiry favors pinning strategies"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.5
                # Example Call Butterfly: Buy ITM (ATM - 100), Sell 2x ATM (ATM), Buy OTM (ATM + 100)
                strike_offset = 100
                strategy_legs_definition = [
                    (-strike_offset, "CE", "B", 1), # Buy ITM Call
                    (0, "CE", "S", 2),           # Sell 2x ATM Call
                    (strike_offset, "CE", "B", 1)            # Buy OTM Call
                ]

            elif iv_skew < -1:
                 strategy = "Short Put"
                 reason = "Low forecast vol, negative IV skew suggests put selling opportunity"
                 tags = ["Directional", "Bullish", "Premium Selling"]
                 risk_reward = 1.5
                 # Short OTM Put
                 strike_offset = -100 # Sell 100 points below ATM
                 strategy_legs_definition = [
                     (strike_offset, "PE", "S", 1)
                 ]

            else:
                strategy = "Iron Fly (Short Straddle + Bought Wings)"
                reason = "Low volatility environment favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8
                # Sell ATM Straddle, Buy OTM wings (e.g., +/- 100 points)
                strike_offset = 100
                strategy_legs_definition = [
                    (0, "CE", "S", 1),           # Sell ATM Call
                    (0, "PE", "S", 1),           # Sell ATM Put
                    (strike_offset, "CE", "B", 1), # Buy OTM Call wing
                    (-strike_offset, "PE", "B", 1) # Buy OTM Put wing
                ]

        elif regime == "MEDIUM":
            if iv_hv_gap > 2 and iv_skew > 0.5:
                strategy = "Iron Condor"
                reason = "Medium vol and skew favor wide-range Iron Condor"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.5
                # Sell OTM Strangle, Buy further OTM wings (e.g., Sell +/- 100, Buy +/- 200)
                sell_offset = 100
                buy_offset = 200
                strategy_legs_definition = [
                    (sell_offset, "CE", "S", 1),    # Sell OTM Call
                    (-sell_offset, "PE", "S", 1),   # Sell OTM Put
                    (buy_offset, "CE", "B", 1),     # Buy further OTM Call wing
                    (-buy_offset, "PE", "B", 1)     # Buy further OTM Put wing
                ]

            elif pcr > 1.1 and dte < 10:
                 strategy = "Short Put Vertical Spread"
                 reason = "Medium vol, bullish PCR, and short expiry"
                 tags = ["Directional", "Bullish", "Defined Risk"]
                 risk_reward = 1.2
                 # Sell OTM Put, Buy further OTM Put (e.g., Sell -100, Buy -200)
                 sell_offset = -100
                 buy_offset = -200
                 strategy_legs_definition = [
                     (sell_offset, "PE", "S", 1),
                     (buy_offset, "PE", "B", 1)
                 ]

            elif pcr < 0.9 and dte < 10:
                 strategy = "Short Call Vertical Spread"
                 reason = "Medium vol, bearish PCR, and short expiry"
                 tags = ["Directional", "Bearish", "Defined Risk"]
                 risk_reward = 1.2
                 # Sell OTM Call, Buy further OTM Call (e.g., Sell +100, Buy +200)
                 sell_offset = 100
                 buy_offset = 200
                 strategy_legs_definition = [
                     (sell_offset, "CE", "S", 1),
                     (buy_offset, "CE", "B", 1)
                 ]

            else:
                strategy = "Short Strangle"
                reason = "Balanced vol, premium-rich environment for Short Strangle"
                tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                risk_reward = 1.6
                # Sell OTM Strangle (e.g., +/- 100 points)
                strike_offset = 100
                strategy_legs_definition = [
                    (strike_offset, "CE", "S", 1),
                    (-strike_offset, "PE", "S", 1)
                ]

        elif regime == "HIGH":
            if iv_hv_gap > 5 or vix_change_pct > 5:
                strategy = "Jade Lizard"
                reason = "High IV spike/gap favors Jade Lizard for defined upside risk"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.0
                # Short OTM Call, Short OTM Put, Long further OTM Put (e.g., Short +100 CE, Short -100 PE, Long -200 PE)
                call_sell_offset = 100
                put_sell_offset = -100
                put_buy_offset = -200
                strategy_legs_definition = [
                    (call_sell_offset, "CE", "S", 1),
                    (put_sell_offset, "PE", "S", 1),
                    (put_buy_offset, "PE", "B", 1)
                ]

            elif dte < 10 and iv_hv_gap > 5:
                 strategy = "Iron Condor" # Still viable for premium capture
                 reason = "High vol and near expiry favors wide premium collection"
                 tags = ["Neutral", "Theta", "Range Bound"]
                 risk_reward = 1.3
                 # Use wider strikes in high vol (e.g., Sell +/- 150, Buy +/- 300)
                 sell_offset = 150
                 buy_offset = 300
                 strategy_legs_definition = [
                    (sell_offset, "CE", "S", 1),
                    (-sell_offset, "PE", "S", 1),
                    (buy_offset, "CE", "B", 1),
                    (-buy_offset, "PE", "B", 1)
                 ]
            elif iv_skew < -2:
                 strategy = "Long Put"
                 reason = "High vol suggests potential downside risk, protective put"
                 tags = ["Directional", "Bearish", "Protection"]
                 risk_reward = 2.0
                 # Buy OTM Put (e.g., -100 points)
                 strike_offset = -100
                 strategy_legs_definition = [
                     (strike_offset, "PE", "B", 1)
                 ]
            else:
                 strategy = "Short Strangle"
                 reason = "High volatility offers rich premium for Short Strangle (wider strikes needed)"
                 tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                 risk_reward = 1.5
                 # Use wider strikes (e.g., +/- 150 points)
                 strike_offset = 150
                 strategy_legs_definition = [
                    (strike_offset, "CE", "S", 1),
                    (-strike_offset, "PE", "S", 1)
                 ]


        elif regime == "EVENT-DRIVEN":
            if iv > 35 and dte < 3:
                strategy = "Short Straddle"
                reason = "Extreme IV + very near expiry event — max premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.8
                 # Sell ATM Straddle
                strategy_legs_definition = [
                    (0, "CE", "S", 1),
                    (0, "PE", "S", 1)
                ]
            elif dte < 7:
                strategy = "Calendar Spread (Call)" # Simplified for simulation - assumes buying longer expiry
                reason = "Event-based uncertainty and near expiry favors term structure opportunity"
                tags = ["Volatility", "Event", "Calendar"]
                risk_reward = 1.5
                # Sell Near ATM Call, Buy Far ATM Call (Far not simulated yet, so this is indicative)
                strategy_legs_definition = [
                    (0, "CE", "S", 1) # Only near leg is defined for now
                ]
            else:
                 strategy = "Iron Condor" # Capture premium before the event
                 reason = "Event anticipation favors capturing premium with Iron Condor"
                 tags = ["Neutral", "Event", "Range Bound"]
                 risk_reward = 1.4
                 # Use standard medium vol strikes for now
                 sell_offset = 100
                 buy_offset = 200
                 strategy_legs_definition = [
                    (sell_offset, "CE", "S", 1),
                    (-sell_offset, "PE", "S", 1),
                    (buy_offset, "CE", "B", 1),
                    (-buy_offset, "PE", "B", 1)
                 ]


        # Capital allocation based on regime
        capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.07)
        deploy_capital_base = capital * capital_alloc_pct # Base capital for the strategy


        # Determine the number of lots based on the most expensive leg or total premium
        # In a real scenario, you'd estimate the premium/max loss of the chosen strategy
        # For simulation, let's approximate based on Straddle price and number of legs
        approx_premium_per_lot_total = day_data.get("Straddle_Price", 200.0) * len(strategy_legs_definition) / 2 # Rough estimate
        lots = max(1, int(deploy_capital_base / (approx_premium_per_lot_total * 25)) if approx_premium_per_lot_total > 0 else 1) # Target deploying approx base capital

        # Ensure lots is a reasonable number
        lots = min(lots, 20) # Cap at 20 lots for simulation

        # Calculate actual deployed capital based on number of lots and estimated premium (rough)
        deployed = lots * approx_premium_per_lot_total * 25 # Rough deployed value for reporting

        # Max loss calculation (can be refined per strategy based on strikes)
        # For now, use a simplified max loss percentage of deployed capital
        max_loss_pct = {"Iron Condor": 0.02, "Butterfly Spread (Call)": 0.015, "Iron Fly (Short Straddle + Bought Wings)": 0.02, "Short Strangle": 0.03, "Calendar Spread (Call)": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04, "Short Put Vertical Spread": 0.015, "Long Put": 0.03, "Short Put": 0.02, "Short Call Vertical Spread": 0.015}.get(strategy, 0.025)
        max_loss = deployed * max_loss_pct # Max loss absolute value


        return regime, strategy, reason, tags, deployed, max_loss, risk_reward, strategy_legs_definition, lots # Return lots and legs definition

    except Exception as e:
        logger.error(f"Error in backtest strategy engine: {str(e)}")
        return None, None, f"Strategy engine failed: {str(e)}", [], 0, 0, 0, [], 0


def calculate_trade_pnl(strategy, day_data_start, day_data_end, strategy_legs_definition, num_lots, initial_capital, risk_free_rate_daily):
     """
     Simulates daily PnL for a given strategy trade using Black-Scholes.
     Calculates PnL based on option price changes and transaction costs.
     """
     try:
        lot_size = 25
        total_daily_pnl = 0.0
        total_transaction_costs = 0.0

        # Black-Scholes requires volatility as an annual decimal, time to expiry in years
        volatility_annual_decimal = day_data_start.get("ATM_IV", 15.0) / 100.0 # Use ATM_IV as proxy, convert to decimal
        days_to_expiry_start = day_data_start.get("Days_to_Expiry", 1)
        days_to_expiry_end = day_data_end.get("Days_to_Expiry", max(0, days_to_expiry_start - 1)) # Ensure DTE doesn't increase

        time_to_expiry_start_years = days_to_expiry_start / 365.0 if days_to_expiry_start > 0 else 0.0001 # Avoid T=0 initially
        time_to_expiry_end_years = days_to_expiry_end / 365.0 if days_to_expiry_end > 0 else 0.0001 # Avoid T=0 initially

        underlying_start = day_data_start.get("NIFTY_Close", 0.0)
        underlying_end = day_data_end.get("NIFTY_Close", underlying_start)

        # Simulate volatility change for the day (optional, adds realism)
        # Could use VIX change or a random walk around the forecast
        # For simplicity, let's use the average forecast volatility for pricing in this step
        # A more complex model might use different vols for different strikes (skew)
        volatility_start_of_day = day_data_start.get("ATM_IV", 15.0) / 100.0
        volatility_end_of_day = day_data_end.get("ATM_IV", 15.0) / 100.0 # Use end-of-day ATM_IV from features

        # Handle potential division by zero or log(0) in Black-Scholes parameters
        if underlying_start <= 0 or underlying_end <= 0 or volatility_start_of_day <= 0:
            logger.warning(f"Invalid data for option pricing on {day_data_start.name}. Underlying={underlying_start}/{underlying_end}, Vol={volatility_start_of_day}")
            return 0.0 # Return 0 PnL if pricing inputs are invalid


        for leg in strategy_legs_definition:
            strike_offset_points, option_type_short, buy_sell, quantity_multiplier = leg
            quantity_units = num_lots * lot_size * quantity_multiplier

            # Calculate the strike price for this leg based on the day's ATM price and offset
            # Find the nearest tradable strike to (underlying_start + strike_offset_points)
            # In a real backtest, you'd need historical option chain data to find actual strikes.
            # For this simulation, let's approximate by rounding to the nearest 50 or 100.
            target_strike = underlying_start + strike_offset_points
            strike_price = round(target_strike / strike_step) * strike_step # Round to nearest strike_step

            option_type_bs = 'call' if option_type_short == 'CE' else 'put'

            # Calculate option price at the start of the day
            try:
                 price_start = black_scholes(underlying_start, strike_price, time_to_expiry_start_years, risk_free_rate_daily*365, volatility_start_of_day, option_type_bs)
            except ValueError as e:
                 logger.error(f"BS Error (Start) for {strategy} {option_type_short} K={strike_price}: {e}")
                 price_start = 0.0 # Default to 0 if BS fails


            # Calculate option price at the end of the day
            # Use end-of-day underlying and potentially end-of-day volatility
            try:
                 price_end = black_scholes(underlying_end, strike_price, time_to_expiry_end_years, risk_free_rate_daily*365, volatility_end_of_day, option_type_bs)
            except ValueError as e:
                 logger.error(f"BS Error (End) for {strategy} {option_type_short} K={strike_price}: {e}")
                 price_end = 0.0 # Default to 0 if BS fails


            # Calculate PnL for this leg for the day
            # For a BUY trade, PnL = (Price_End - Price_Start) * Quantity
            # For a SELL trade, PnL = (Price_Start - Price_End) * Quantity (you profit if the price goes down)
            if buy_sell == "B":
                leg_pnl = (price_end - price_start) * quantity_units
            elif buy_sell == "S":
                leg_pnl = (price_start - price_end) * quantity_units
            else:
                leg_pnl = 0.0
                logger.warning(f"Unknown buy_sell type: {buy_sell}")


            total_daily_pnl += leg_pnl

            # Calculate transaction costs for entering and potentially exiting (assuming daily PnL closure for simplicity)
            # A more realistic backtest would track positions and apply costs on open/close events.
            # For simplicity here, let's apply costs as if the position was opened and closed daily.
            # This overestimates costs but provides a conservative estimate.
            # A better approach: apply cost when strategy is ENTERED and when EXITED (either by time or stop-loss/target)
            # Let's refactor to apply costs only on the first day a strategy is active.

            # This simplified model applies costs on the first day of the strategy only.
            # This requires tracking if this is the first day of this strategy instance.
            # For now, let's stick to applying costs per leg, perhaps scaled down.
            # A truly realistic backtest needs to manage position entry/exit explicitly.

            # Let's calculate costs per leg for opening the position
            # We need the premium at the time of opening. Use price_start as initial premium proxy.
            transaction_cost_leg = calculate_transaction_costs(strategy, buy_sell, num_lots * quantity_multiplier, price_start / lot_size) # Cost per unit

            total_transaction_costs += transaction_cost_leg


        # Subtract total transaction costs (applied once per strategy instance in a real backtest)
        # For this simplified daily PnL calculation, let's apply a fraction of the costs daily
        # or just apply full costs on day 1. Let's apply full costs on the first day for simplicity.
        # This needs logic in the main backtest loop to know the first day.

        # Let's refine the backtest loop instead to handle strategy entry/exit and apply costs there.
        # For NOW, within this simplified daily PnL, let's just return the gross PnL.
        # Transaction costs will be applied in the main backtest loop on the day the strategy starts.


        return total_daily_pnl # Return gross PnL for the day


    except Exception as e:
        logger.error(f"Error calculating trade PnL using Black-Scholes for {strategy} on {day_data_start.name}: {str(e)}", exc_info=True)
        return 0.0 # Return 0 PnL on error


# Backtesting Function
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.info(f"Starting robust backtest for {strategy_choice} from {start_date} to {end_date}")
        if df is None or df.empty:
            logger.error("Backtest failed: No data available")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        # Ensure index is datetime and unique, then slice and copy
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep='first')] # Remove duplicates keeping the first
        df = df.sort_index() # Sort by date
        df = df.loc[start_date:end_date].copy() # Slice and copy

        if len(df) < 50:
            logger.warning(f"Backtest failed: Insufficient data ({len(df)} days) in selected range.")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        required_cols = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price", "PCR", "VIX_Change_Pct", "Spot_MaxPain_Diff_Pct"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Backtest failed: Missing required columns after date slicing: {missing_cols}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()


        backtest_results = []
        portfolio_pnl = 0.0
        risk_free_rate_annual = 0.06 # Annual risk-free rate (adjust as needed)
        # risk_free_rate_daily = risk_free_rate_annual / 252 # Daily risk-free rate (used in Sharpe etc)

        current_strategy = None
        strategy_entry_date = None
        strategy_legs_active = []
        strategy_lots = 0


        # Backtest loop - iterates through days in the selected range
        for i in range(1, len(df)):
            try:
                day_data_end = df.iloc[i] # Data at the end of the current day
                day_data_start = df.iloc[i-1] # Data at the start of the current day (previous day's close)
                date = day_data_end.name

                # Use the historical realized vol from the past few days
                avg_vol_for_strategy = df["Realized_Vol"].iloc[max(0, i-5):i].mean()

                # --- Strategy Decision and Entry ---
                if current_strategy is None: # If no strategy is currently active
                    regime, strategy_name, reason, tags, deploy, max_loss, risk_reward, strategy_legs_definition, lots = run_strategy_engine(
                        day_data_start, avg_vol_for_strategy, portfolio_pnl, capital # Use start-of-day data for decision
                    )

                    # Filter strategies if a specific one is chosen
                    if strategy_name is not None and (strategy_choice == "All Strategies" or strategy_name == strategy_choice):
                         # Decide to enter the strategy
                         current_strategy = strategy_name
                         strategy_entry_date = date # Strategy starts today
                         strategy_legs_active = strategy_legs_definition # Store legs definition
                         strategy_lots = lots # Store number of lots
                         deployed_capital_for_strategy = deploy # Store deployed capital for this instance
                         max_loss_for_strategy = max_loss # Store max loss for this instance
                         risk_reward_for_strategy = risk_reward # Store risk reward for this instance


                         # Calculate and apply transaction costs on entry
                         entry_costs = 0.0
                         for leg in strategy_legs_active:
                             strike_offset, option_type_short, buy_sell, quantity_multiplier = leg
                             quantity_units = strategy_lots * lot_size * quantity_multiplier

                             # Need to get the estimated premium at entry (use start-of-day Black-Scholes price)
                             target_strike = day_data_start.get("NIFTY_Close", 0.0) + strike_offset_points
                             strike_price = round(target_strike / strike_step) * strike_step # Round to nearest strike_step
                             option_type_bs = 'call' if option_type_short == 'CE' else 'put'
                             days_to_expiry_entry = day_data_start.get("Days_to_Expiry", 1)
                             time_to_expiry_entry_years = days_to_expiry_entry / 365.0 if days_to_expiry_entry > 0 else 0.0001
                             volatility_entry = day_data_start.get("ATM_IV", 15.0) / 100.0

                             try:
                                  premium_per_unit_entry = black_scholes(day_data_start.get("NIFTY_Close", 0.0), strike_price, time_to_expiry_entry_years, risk_free_rate_annual, volatility_entry, option_type_bs)
                             except ValueError:
                                  premium_per_unit_entry = 0.0
                                  logger.warning(f"BS Error at entry cost calculation for {strategy_name} {option_type_short} K={strike_price} on {date}")


                             entry_costs += calculate_transaction_costs(strategy_name, buy_sell, strategy_lots * quantity_multiplier, premium_per_unit_entry * lot_size) # Pass premium per lot


                         portfolio_pnl -= entry_costs # Subtract entry costs from portfolio PnL


                         backtest_results.append({
                            "Date": date,
                            "Event": "ENTRY",
                            "Regime": regime,
                            "Strategy": current_strategy,
                            "PnL": -entry_costs, # Record entry costs as negative PnL on entry day
                            "Cumulative_PnL": portfolio_pnl,
                            "Capital_Deployed": deployed_capital_for_strategy,
                            "Max_Loss": max_loss_for_strategy,
                            "Risk_Reward": risk_reward_for_strategy,
                            "Notes": f"Entered {current_strategy} ({strategy_lots} lots)"
                         })
                         logger.debug(f"Entered strategy {current_strategy} on {date}. Entry Costs: {entry_costs:.2f}")


                # --- Daily PnL Calculation for Active Strategy ---
                if current_strategy is not None:
                    # Calculate PnL for the current day for the active strategy
                    daily_gross_pnl = calculate_trade_pnl(
                        current_strategy, day_data_start, day_data_end, strategy_legs_active, strategy_lots, capital, risk_free_rate_annual / 365 # Pass daily risk-free rate for BS
                    )

                    portfolio_pnl += daily_gross_pnl # Add daily PnL to portfolio


                    # Record daily PnL
                    backtest_results.append({
                        "Date": date,
                        "Event": "DAILY_PNL",
                        "Regime": None, # Regime is determined at entry
                        "Strategy": current_strategy,
                        "PnL": daily_gross_pnl,
                        "Cumulative_PnL": portfolio_pnl,
                        "Capital_Deployed": deployed_capital_for_strategy, # Report deployed capital while active
                        "Max_Loss": max_loss_for_strategy,
                        "Risk_Reward": risk_reward_for_strategy,
                        "Notes": "Daily PnL"
                    })


                    # --- Strategy Exit Conditions (Simplified) ---
                    # Exit if DTE is 0 or 1 (end of expiry cycle for the primary legs)
                    # A real backtest would have stop-loss, target, and time-based exits.
                    if day_data_end.get("Days_to_Expiry", 0) <= 1: # Exit on expiry day or day before
                         exit_costs = 0.0 # Assuming minimal exit costs at expiry

                         portfolio_pnl -= exit_costs # Subtract exit costs

                         backtest_results.append({
                             "Date": date,
                             "Event": "EXIT (Expiry)",
                             "Regime": None,
                             "Strategy": current_strategy,
                             "PnL": -exit_costs, # Record exit costs
                             "Cumulative_PnL": portfolio_pnl,
                             "Capital_Deployed": 0, # Capital is freed up
                             "Max_Loss": 0,
                             "Risk_Reward": 0,
                             "Notes": "Exited due to expiry"
                         })
                         logger.debug(f"Exited strategy {current_strategy} on {date} due to expiry.")

                         # Reset active strategy
                         current_strategy = None
                         strategy_entry_date = None
                         strategy_legs_active = []
                         strategy_lots = 0
                         deployed_capital_for_strategy = 0
                         max_loss_for_strategy = 0
                         risk_reward_for_strategy = 0


            except Exception as e:
                logger.error(f"Error in backtest loop at date {date}: {str(e)}", exc_info=True)
                # If an error occurs during trading simulation for an active strategy, exit it.
                if current_strategy is not None:
                     logger.warning(f"Force exiting strategy {current_strategy} on {date} due to error.")
                     backtest_results.append({
                         "Date": date,
                         "Event": "EXIT (Error)",
                         "Regime": None,
                         "Strategy": current_strategy,
                         "PnL": 0, # Assume 0 PnL for the exit on error day
                         "Cumulative_PnL": portfolio_pnl,
                         "Capital_Deployed": 0,
                         "Max_Loss": 0,
                         "Risk_Reward": 0,
                         "Notes": f"Exited due to error: {e}"
                     })
                     current_strategy = None
                     strategy_entry_date = None
                     strategy_legs_active = []
                     strategy_lots = 0
                     deployed_capital_for_strategy = 0
                     max_loss_for_strategy = 0
                     risk_reward_for_strategy = 0

                continue # Continue backtest even if one day fails


        backtest_df = pd.DataFrame(backtest_results)

        if backtest_df.empty:
            logger.warning(f"No trades generated for {strategy_choice} from {start_date} to {end_date}")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()


        # --- Performance Metrics Calculation ---
        # Need to calculate metrics based on DAILY returns on the *entire* capital
        # Create a daily PnL series, aligning with all dates in the backtest range
        daily_pnl_series = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].set_index('Date')['PnL']
        # Also include entry and exit costs on their respective dates
        costs_series = backtest_df[backtest_df['Event'].isin(['ENTRY', 'EXIT (Expiry)', 'EXIT (Error)'])].set_index('Date')['PnL']

        # Combine all PnL events into a single daily series
        all_daily_pnl_events = daily_pnl_series.add(costs_series, fill_value=0).sort_index()

        # Reindex to include all dates in the backtest range, filling missing days with 0 PnL
        full_date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='B') # Business days
        daily_total_pnl = all_daily_pnl_events.reindex(full_date_range, fill_value=0).sort_index()


        # Calculate cumulative PnL over the full range
        cumulative_pnl_full = daily_total_pnl.cumsum() + capital # Add initial capital

        # Calculate daily returns based on capital at start of each day
        # Capital at start of day t = Capital at end of day t-1
        capital_at_start_of_day = cumulative_pnl_full.shift(1).fillna(capital)
        daily_returns = daily_total_pnl / capital_at_start_of_day

        # Drop the first day's return if it's NaN (due to shift)
        daily_returns = daily_returns.dropna()

        total_pnl = daily_total_pnl.sum() # Sum of all daily PnL and costs


        # Calculate Max Drawdown correctly from the full cumulative PnL series
        cumulative_pnl_values = cumulative_pnl_full.values
        peak_values = np.maximum.accumulate(cumulative_pnl_values)
        drawdown_values = peak_values - cumulative_pnl_values
        max_drawdown = np.max(drawdown_values) - capital # Max drawdown relative to initial capital


        # Ensure NIFTY returns are aligned for comparison
        df_aligned_for_returns = df.reindex(daily_returns.index)
        nifty_daily_returns = df_aligned_for_returns["NIFTY_Close"].pct_change().dropna()

        # Reindex daily_returns to match nifty_daily_returns for excess return calculation
        daily_returns_aligned = daily_returns.reindex(nifty_daily_returns.index).fillna(0)

        # Calculate excess returns
        risk_free_rate_daily = risk_free_rate_annual / 252 # Daily risk-free rate
        excess_returns = daily_returns_aligned - nifty_daily_returns - risk_free_rate_daily

        # Ensure there are enough data points for standard deviation for Sharpe/Sortino
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 and len(excess_returns) > 1 else 0
        sortino_std_negative = excess_returns[excess_returns < 0].std()
        sortino_ratio = excess_returns.mean() / sortino_std_negative * np.sqrt(252) if sortino_std_negative != 0 and len(excess_returns[excess_returns < 0]) > 1 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown > 0 and capital > 0 else float('inf')


        # Win Rate (calculated based on days with positive gross PnL before costs or based on trades?)
        # Let's calculate win rate based on days where the active strategy had positive gross PnL
        days_with_active_strategy = backtest_df[backtest_df['Event'] == 'DAILY_PNL']
        win_rate = (days_with_active_strategy['PnL'] > 0).sum() / len(days_with_active_strategy) if len(days_with_active_strategy) > 0 else 0


        # Performance by Strategy and Regime (recalculate based on the PnL recorded in backtest_df)
        # Filter for DAILY_PNL events to aggregate strategy performance
        strategy_perf = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        # Win rate by strategy is trickier with daily PnL - counts days with positive PnL while that strategy was active
        strategy_win_rates = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].groupby("Strategy")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        strategy_perf = pd.merge(strategy_perf, strategy_win_rates, on="Strategy")

        regime_perf = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        # Win rate by regime - counts days with positive PnL while in that regime (during active strategy)
        regime_win_rates = backtest_df[backtest_df['Event'] == 'DAILY_PNL'].groupby("Regime")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        regime_perf = pd.merge(regime_perf, regime_win_rates, on="Regime")


        logger.debug("Robust backtest completed successfully")

        # Return the refined cumulative PnL for charting
        cumulative_pnl_chart_data = pd.DataFrame({'Cumulative_PnL': cumulative_pnl_full})


        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf, cumulative_pnl_chart_data # Return cumulative PnL for charting

    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty cumulative PnL on error

