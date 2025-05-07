import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_processing import FEATURE_COLS # Import FEATURE_COLS

# Setup logging
logger = logging.getLogger(__name__)

def run_strategy_engine(day_data, avg_vol_forecast, portfolio_pnl, capital):
    """
    Determines the trading strategy based on market regime and indicators.
    """
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
            return None, None, "Portfolio drawdown limit reached", [], 0, 0, 0

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

        # Strategy selection logic based on regime and real-time indicators
        if regime == "LOW":
            if iv_hv_gap > 3 and dte < 15: # Adjust thresholds
                strategy = "Butterfly Spread"
                reason = "Low vol & moderate expiry favors pinning strategies"
                tags = ["Neutral", "Theta", "Expiry Play"]
                risk_reward = 2.5 # Higher reward potential
            elif iv_skew < -1: # Negative skew
                 strategy = "Short Put" # Simple directional play
                 reason = "Low forecast vol, negative IV skew suggests put selling opportunity"
                 tags = ["Directional", "Bullish", "Premium Selling"]
                 risk_reward = 1.5
            else:
                strategy = "Iron Fly"
                reason = "Low volatility environment favors delta-neutral Iron Fly"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.8

        elif regime == "MEDIUM":
            if iv_hv_gap > 2 and iv_skew > 0.5: # Adjust thresholds
                strategy = "Iron Condor"
                reason = "Medium vol and skew favor wide-range Iron Condor"
                tags = ["Neutral", "Theta", "Range Bound"]
                risk_reward = 1.5
            elif pcr > 1.1 and dte < 10: # Example: Bullish bias + short expiry
                 strategy = "Short Put Vertical Spread" # Example directional strategy
                 reason = "Medium vol, bullish PCR, and short expiry"
                 tags = ["Directional", "Bullish", "Defined Risk"]
                 risk_reward = 1.2
            elif pcr < 0.9 and dte < 10:
                 strategy = "Short Call Vertical Spread" # Bearish bias
                 reason = "Medium vol, bearish PCR, and short expiry"
                 tags = ["Directional", "Bearish", "Defined Risk"]
                 risk_reward = 1.2
            else:
                strategy = "Short Strangle"
                reason = "Balanced vol, premium-rich environment for Short Strangle"
                tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                risk_reward = 1.6

        elif regime == "HIGH":
            if iv_hv_gap > 8 or vix_change_pct > 5: # High gap or spike
                strategy = "Jade Lizard"
                reason = "High IV spike/gap favors Jade Lizard for defined upside risk"
                tags = ["Skewed", "Volatility", "Defined Risk"]
                risk_reward = 1.0
            elif dte < 10 and iv_hv_gap > 5: # High vol + near expiry + high gap
                 strategy = "Iron Condor" # Still viable for premium capture
                 reason = "High vol and near expiry favors wide premium collection"
                 tags = ["Neutral", "Theta", "Range Bound"]
                 risk_reward = 1.3
            elif iv_skew < -2: # Extreme negative skew
                 strategy = "Long Put" # Example protective/bearish strategy in high vol
                 reason = "High vol suggests potential downside risk, protective put"
                 tags = ["Directional", "Bearish", "Protection"]
                 risk_reward = 2.0 # High potential if market drops
            else:
                 strategy = "Short Strangle" # Can still work in high vol if range is wide
                 reason = "High volatility offers rich premium for Short Strangle (wider strikes needed)"
                 tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                 risk_reward = 1.5


        elif regime == "EVENT-DRIVEN":
            if iv > 35 and dte < 3: # Higher IV threshold, very near expiry
                strategy = "Short Straddle"
                reason = "Extreme IV + very near expiry event Ã¢â€ â€™ max premium capture"
                tags = ["Volatility", "Event", "Neutral"]
                risk_reward = 1.8 # Higher potential reward due to premium
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


        # Capital allocation based on regime
        capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.07)
        deploy = capital * capital_alloc_pct # Simple allocation

        # Dynamic max loss based on strategy risk and capital allocation
        max_loss_pct = {"Iron Condor": 0.02, "Butterfly Spread": 0.015, "Iron Fly": 0.02, "Short Strangle": 0.03, "Calendar Spread": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04, "Short Put Vertical Spread": 0.015, "Long Put": 0.03, "Short Put": 0.02, "Short Call Vertical Spread": 0.015}.get(strategy, 0.025)
        max_loss = deploy * max_loss_pct # Max loss absolute value

        return regime, strategy, reason, tags, deploy, max_loss, risk_reward
    except Exception as e:
        logger.error(f"Error in backtest strategy engine: {str(e)}")
        return None, None, f"Strategy engine failed: {str(e)}", [], 0, 0, 0

def calculate_trade_pnl(strategy, day_data, prev_day_data, deploy, max_loss, risk_reward):
     """
     Simulates daily PnL for a given strategy trade.
     (Simplified PnL calculation for backtest)
     """
     try:
        # Simplified PnL calculation for backtest
        # A real backtest would simulate option prices based on Black-Scholes or similar,
        # tracking Greeks, volatility changes, and time decay. This is a simplification.
        premium = day_data.get("Straddle_Price", 200.0) # Use Straddle Price as a proxy for strategy premium

        lot_size = 25
        # Approximate lots based on deployable capital and proxy premium
        lots = max(1, int(deploy / (premium * lot_size)) if premium > 0 else 1)
        if lots == 0: lots = 1 # Ensure at least 1 lot if deploy > 0

        # Simulate market move impact
        nifty_close = day_data.get("NIFTY_Close", 0.0)
        prev_nifty_close = prev_day_data.get("NIFTY_Close", nifty_close)
        nifty_move_abs_pct = abs(nifty_close / prev_nifty_close - 1) * 100 if prev_nifty_close > 0 else 0

        atm_iv = day_data.get("ATM_IV", 15.0)
        implied_move_1sd = atm_iv / np.sqrt(252) # Daily 1-sigma move in %

        # PnL depends on strategy type and market move relative to volatility
        base_pnl = 0 # Default PnL if no clear outcome simulated
        strategy_sensitivity = {
            "Iron Condor": -0.5, # Loses when market moves beyond range
            "Butterfly Spread": -0.8, # Very sensitive to market pinning
            "Iron Fly": -0.6, # Sensitive to move away from strike
            "Short Strangle": -0.7, # Loses from large moves
            "Calendar Spread": 0.3, # Benefits from time decay and changing term structure
            "Jade Lizard": -0.4, # Limited upside risk, but loses on large moves
            "Short Straddle": -1.0, # Loses heavily from large moves
            "Short Put Vertical Spread": 0.8 if nifty_close > prev_nifty_close else -1.2, # Directional
            "Short Call Vertical Spread": -1.2 if nifty_close > prev_nifty_close else 0.8, # Directional
            "Short Put": 0.6 if nifty_close > prev_nifty_close else -1.0, # Directional
            "Long Put": -0.5 if nifty_close > prev_nifty_close else 1.5 # Directional
        }.get(strategy, -0.5) # Default sensitivity

        # PnL simulation based on premium and market move vs expected move
        move_ratio = nifty_move_abs_pct / implied_move_1sd if implied_move_1sd > 0 else 2.0 # Ratio of actual move to expected move

        # Base daily gain/loss as a percentage of deployed capital
        base_daily_gain_pct = 0.001 # Small gain from theta decay daily
        # Strategies selling premium benefit from decay unless move is large
        if "Short" in strategy or "Iron" in strategy or "Jade Lizard" in strategy:
            loss_factor = max(0, move_ratio - 1.0) * abs(strategy_sensitivity)
            base_daily_gain_pct -= loss_factor * 0.02 # Higher loss factor

        # Strategies buying premium or more complex
        elif "Long" in strategy or "Calendar Spread" in strategy or "Butterfly Spread" in strategy:
            gain_factor = max(0, move_ratio - 0.5) * abs(strategy_sensitivity)
            # Add directionality for Long Put
            if strategy == "Long Put":
                 gain_factor *= np.sign(prev_nifty_close - nifty_close) # Positive if market drops
            base_daily_gain_pct += gain_factor * 0.015 # Higher gain factor


        # Apply decay factor - strategies selling premium benefit from time passing
        decay_benefit_factor = 0.0005 * max(0, 15 - day_data.get("Days_to_Expiry", 0)) if "Short" in strategy or "Iron" in strategy or "Jade Lizard" in strategy else 0
        base_daily_gain_pct += decay_benefit_factor

        # Apply event shock if applicable
        if day_data.get("Event_Flag", 0) == 1:
            event_impact = np.random.uniform(-0.03, 0.03) * abs(strategy_sensitivity) # Random positive or negative shock
            base_daily_gain_pct += event_impact

        # Convert percentage PnL to absolute PnL based on deployed capital
        pnl = deploy * base_daily_gain_pct * np.random.uniform(0.8, 1.2) # Add some daily randomness

        # Ensure PnL is within max_loss boundaries
        # Winning PnL should also be capped relative to deploy or max_loss/risk_reward
        max_win = max_loss * risk_reward if risk_reward is not None else max_loss * 1.0
        pnl = max(-max_loss, min(pnl, max_win))


        # Simulate transaction costs (simplified)
        num_legs = {"Short Straddle": 2, "Short Strangle": 2, "Iron Condor": 4, "Iron Fly": 4, "Butterfly Spread": 4, "Jade Lizard": 3, "Calendar Spread": 2, "Short Put Vertical Spread": 2, "Long Put": 1, "Short Put": 1, "Short Call Vertical Spread": 2}.get(strategy, 2) # Added new strategies
        base_transaction_cost_factor = 0.002
        stt_factor = 0.0005 # Securities Transaction Tax
        transaction_cost = deploy * base_transaction_cost_factor * num_legs + deploy * stt_factor # Simplified cost calculation
        pnl -= transaction_cost

        # Add small random noise to final PnL
        pnl += np.random.normal(0, deploy * 0.001)


        return pnl

     except Exception as e:
        logger.error(f"Error calculating trade PnL for {strategy} on {day_data.name}: {str(e)}")
        return 0 # Return 0 PnL on error


# Backtesting Function
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.info(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")
        if df is None or df.empty:
            logger.error("Backtest failed: No data available")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        df = df.groupby(df.index).last().copy() # Ensure unique index and copy
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
        portfolio_pnl = 0
        risk_free_rate_daily = 0.06 / 252 # Approx daily risk-free rate assuming 252 trading days

        # Backtest loop - iterates through days in the selected range
        for i in range(1, len(df)):
            try:
                day_data = df.iloc[i]
                prev_day_data = df.iloc[i-1]
                date = day_data.name

                # In a true backtest, forecast would ideally be re-run each day using past data only.
                # For this version, we'll use the historical realized vol from the past few days
                # within the backtest window as a proxy for "current" market volatility expectation.
                avg_vol_for_strategy = df["Realized_Vol"].iloc[max(0, i-5):i].mean()

                regime, strategy, reason, tags, deploy, max_loss, risk_reward = run_strategy_engine(
                    day_data, avg_vol_for_strategy, portfolio_pnl, capital # Pass capital to engine
                )

                # Filter strategies if a specific one is chosen
                if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy_choice):
                    continue

                # Calculate PnL for the day's trade
                pnl = calculate_trade_pnl(strategy, day_data, prev_day_data, deploy, max_loss, risk_reward)

                portfolio_pnl += pnl # Accumulate portfolio PnL

                backtest_results.append({
                    "Date": date,
                    "Regime": regime,
                    "Strategy": strategy,
                    "PnL": pnl,
                    "Cumulative_PnL": portfolio_pnl, # Track cumulative PnL
                    "Capital_Deployed": deploy,
                    "Max_Loss": max_loss,
                    "Risk_Reward": risk_reward
                })
            except Exception as e:
                logger.error(f"Error in backtest loop at index {i}: {str(e)}")
                continue # Continue backtest even if one day fails

        backtest_df = pd.DataFrame(backtest_results)

        if backtest_df.empty:
            logger.warning(f"No trades generated for {strategy_choice} from {start_date} to {end_date}")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        # Final performance metrics calculation
        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0

        # Calculate Max Drawdown correctly from Cumulative PnL
        backtest_df['Cumulative_PnL'] = backtest_df['PnL'].cumsum()
        backtest_df['Peak'] = backtest_df['Cumulative_PnL'].cummax()
        backtest_df['Drawdown'] = backtest_df['Peak'] - backtest_df['Cumulative_PnL']
        max_drawdown = backtest_df['Drawdown'].max() if not backtest_df.empty else 0


        backtest_df.set_index("Date", inplace=True)

        # Calculate daily returns based on capital at start of each day (simplified)
        # Assuming capital grows by cumulative PnL
        # Reindex df to align with backtest_df index
        df_aligned_for_capital = df.reindex(backtest_df.index)
        # Create a series for capital that starts with initial capital and adds cumulative PnL
        capital_series_start_of_day = pd.Series(capital, index=backtest_df.index) + backtest_df['Cumulative_PnL'].shift(1).fillna(0)

        daily_backtest_pnl = backtest_df['PnL']
        daily_returns = daily_backtest_pnl / capital_series_start_of_day.fillna(capital) # Daily return based on previous day's capital (or initial capital)
        daily_returns = daily_returns.dropna() # Drop potential NaNs from division


        # Ensure NIFTY returns are aligned and calculated correctly
        df_returns_base = df.reindex(daily_returns.index) # Reindex original df for NIFTY returns
        nifty_daily_returns = df_returns_base["NIFTY_Close"].pct_change()
        nifty_daily_returns = nifty_daily_returns.reindex(daily_returns.index).dropna() # Align and drop any NaNs

        # Reindex daily_returns to match nifty_daily_returns for excess return calculation
        daily_returns_aligned = daily_returns.reindex(nifty_daily_returns.index).fillna(0)


        excess_returns = daily_returns_aligned - nifty_daily_returns - risk_free_rate_daily
        # Ensure there are enough data points for standard deviation
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 and len(excess_returns) > 1 else 0 # Annualized Sharpe
        sortino_std_negative = excess_returns[excess_returns < 0].std()
        sortino_ratio = excess_returns.mean() / sortino_std_negative * np.sqrt(252) if sortino_std_negative != 0 and len(excess_returns[excess_returns < 0]) > 1 else 0 # Annualized Sortino
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown > 0 and capital > 0 else float('inf') # Handle division by zero for Calmar


        # Performance by Strategy and Regime
        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_win_rates = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        strategy_perf = pd.merge(strategy_perf, strategy_win_rates, on="Strategy")

        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_win_rates = backtest_df.groupby("Regime")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        regime_perf = pd.merge(regime_perf, regime_win_rates, on="Regime")


        logger.debug("Backtest completed successfully")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}", exc_info=True)
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

