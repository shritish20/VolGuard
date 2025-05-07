import streamlit as st
import pandas as pd
import numpy as np
import logging
from config.settings import LOT_SIZE, BASE_TRANSACTION_COST_FACTOR, STT_FACTOR, RISK_FREE_RATE_DAILY

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(df, capital, strategy_choice, start_date, end_date):
    """Runs a backtest for the given strategy and date range."""
    try:
        logger.info(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")
        if df.empty:
            st.error("Backtest failed: No data available")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        df = df.groupby(df.index).last().copy()
        df = df.loc[start_date:end_date].copy()
        if len(df) < 50:
            st.error(f"Backtest failed: Insufficient data ({len(df)} days).")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        required_cols = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price", "PCR", "VIX_Change_Pct", "Spot_MaxPain_Diff_Pct"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Backtest failed: Missing columns: {missing_cols}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        backtest_results = []
        portfolio_pnl = 0

        def run_strategy_engine(day_data, avg_vol_forecast, portfolio_pnl, capital):
            try:
                iv = day_data["ATM_IV"]
                hv = day_data["Realized_Vol"]
                iv_hv_gap = iv - hv
                iv_skew = day_data["IV_Skew"]
                dte = day_data["Days_to_Expiry"]
                event_flag = day_data["Event_Flag"]
                pcr = day_data["PCR"]
                vix_change_pct = day_data["VIX_Change_Pct"]

                if portfolio_pnl < -0.10 * capital:
                    return None, None, "Drawdown limit reached", [], 0, 0, 0

                regime = "MEDIUM" if avg_vol_forecast is None else "LOW" if avg_vol_forecast < 15 else "MEDIUM" if avg_vol_forecast < 20 else "HIGH"
                strategy = "Undefined"
                reason = "N/A"
                tags = []
                risk_reward = 1.0

                if regime == "LOW":
                    if iv_hv_gap > 3 and dte < 15:
                        strategy = "Butterfly Spread"
                        reason = "Low vol & moderate expiry favors pinning"
                        tags = ["Neutral", "Theta", "Expiry Play"]
                        risk_reward = 2.5
                    else:
                        strategy = "Iron Fly"
                        reason = "Low vol favors delta-neutral Iron Fly"
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward = 1.8
                elif regime == "MEDIUM":
                    if iv_hv_gap > 2 and iv_skew > 1:
                        strategy = "Iron Condor"
                        reason = "Medium vol and skew favor Iron Condor"
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward = 1.5
                    elif pcr > 1.2 and dte < 10:
                        strategy = "Short Put Vertical Spread"
                        reason = "Medium vol, bullish PCR, short expiry"
                        tags = ["Directional", "Bullish", "Defined Risk"]
                        risk_reward = 1.2
                    else:
                        strategy = "Short Strangle"
                        reason = "Balanced vol for Short Strangle"
                        tags = ["Neutral", "Premium Selling", "Volatility Harvest"]
                        risk_reward = 1.6
                elif regime == "HIGH":
                    if iv_hv_gap > 8 or vix_change_pct > 5:
                        strategy = "Jade Lizard"
                        reason = "High IV spike favors Jade Lizard"
                        tags = ["Skewed", "Volatility", "Defined Risk"]
                        risk_reward = 1.0
                    elif dte < 10:
                        strategy = "Iron Condor"
                        reason = "High vol and near expiry favors premium"
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward = 1.3
                    else:
                        strategy = "Long Put"
                        reason = "High vol suggests downside risk"
                        tags = ["Directional", "Bearish", "Protection"]
                        risk_reward = 2.0
                elif regime == "EVENT-DRIVEN":
                    if iv > 35 and dte < 3:
                        strategy = "Short Straddle"
                        reason = "Extreme IV + near expiry"
                        tags = ["Volatility", "Event", "Neutral"]
                        risk_reward = 1.8
                    else:
                        strategy = "Calendar Spread"
                        reason = "Event uncertainty favors term structure"
                        tags = ["Volatility", "Event", "Calendar"]
                        risk_reward = 1.5

                capital_alloc_pct = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.07}.get(regime, 0.07)
                deploy = capital * capital_alloc_pct
                max_loss_pct = {
                    "Iron Condor": 0.02, "Butterfly Spread": 0.015, "Iron Fly": 0.02, "Short Strangle": 0.03,
                    "Calendar Spread": 0.015, "Jade Lizard": 0.025, "Short Straddle": 0.04,
                    "Short Put Vertical Spread": 0.015, "Long Put": 0.03
                }.get(strategy, 0.025)
                max_loss = deploy * max_loss_pct
                return regime, strategy, reason, tags, deploy, max_loss, risk_reward
            except Exception as e:
                logger.error(f"Error in strategy engine: {str(e)}")
                return None, None, f"Strategy engine failed: {str(e)}", [], 0, 0, 0

        def calculate_trade_pnl(strategy, day_data, prev_day_data, deploy, max_loss):
            try:
                premium = day_data["Straddle_Price"]
                lots = max(1, int(deploy / (premium * LOT_SIZE)) if premium > 0 else 1)
                nifty_move_abs_pct = abs(day_data["NIFTY_Close"] / prev_day_data["NIFTY_Close"] - 1) * 100 if prev_day_data["NIFTY_Close"] > 0 else 0
                implied_move_1sd = day_data["ATM_IV"] / np.sqrt(252)

                strategy_sensitivity = {
                    "Iron Condor": -0.5, "Butterfly Spread": -0.8, "Iron Fly": -0.6, "Short Strangle": -0.7,
                    "Calendar Spread": 0.3, "Jade Lizard": -0.4, "Short Straddle": -1.0,
                    "Short Put Vertical Spread": 0.8 if day_data["NIFTY_Close"] > prev_day_data["NIFTY_Close"] else -1.2,
                    "Long Put": -0.5 if day_data["NIFTY_Close"] > prev_day_data["NIFTY_Close"] else 1.5
                }.get(strategy, -0.5)

                move_ratio = nifty_move_abs_pct / implied_move_1sd if implied_move_1sd > 0 else 2.0
                base_daily_gain_pct = 0.001
                if "Short" in strategy or "Iron" in strategy or "Jade Lizard" in strategy:
                    loss_factor = max(0, move_ratio - 1.0) * abs(strategy_sensitivity)
                    base_daily_gain_pct -= loss_factor * 0.02
                elif "Long" in strategy or "Calendar Spread" in strategy:
                    gain_factor = max(0, move_ratio - 0.5) * abs(strategy_sensitivity) * np.sign(day_data["NIFTY_Close"] - prev_day_data["NIFTY_Close"])
                    base_daily_gain_pct += gain_factor * 0.015

                decay_benefit_factor = 0.0005 * max(0, 15 - day_data["Days_to_Expiry"]) if "Short" in strategy or "Iron" in strategy or "Jade Lizard" in strategy else 0
                base_daily_gain_pct += decay_benefit_factor

                if day_data["Event_Flag"] == 1:
                    event_impact = np.random.uniform(-0.03, 0.03) * abs(strategy_sensitivity)
                    base_daily_gain_pct += event_impact

                pnl = deploy * base_daily_gain_pct * np.random.uniform(0.8, 1.2)
                max_win = max_loss * risk_reward if risk_reward is not None else max_loss * 1.0
                pnl = max(-max_loss, min(pnl, max_win))

                num_legs = {"Short Straddle": 2, "Short Strangle": 2, "Iron Condor": 4, "Iron Fly": 4, "Butterfly Spread": 3,
                            "Jade Lizard": 3, "Calendar Spread": 2, "Short Put Vertical Spread": 2, "Long Put": 1}.get(strategy, 2)
                transaction_cost = deploy * BASE_TRANSACTION_COST_FACTOR * num_legs + deploy * STT_FACTOR
                pnl -= transaction_cost
                pnl += np.random.normal(0, deploy * 0.001)
                return pnl
            except Exception as e:
                logger.error(f"Error calculating PnL for {strategy}: {str(e)}")
                return 0

        blended_vols_forecast = st.session_state.forecast_log["Blended_Vol"].iloc[0] if 'forecast_log' in st.session_state and st.session_state.forecast_log is not None and not st.session_state.forecast_log.empty else None
        for i in range(1, len(df)):
            try:
                day_data = df.iloc[i]
                prev_day_data = df.iloc[i-1]
                date = day_data.name
                avg_vol_for_strategy = blended_vols_forecast if blended_vols_forecast is not None else df["Realized_Vol"].iloc[max(0, i-5):i].mean()
                regime, strategy, reason, tags, deploy, max_loss, risk_reward = run_strategy_engine(day_data, avg_vol_for_strategy, portfolio_pnl, capital)

                if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy_choice):
                    continue

                pnl = calculate_trade_pnl(strategy, day_data, prev_day_data, deploy, max_loss)
                portfolio_pnl += pnl
                backtest_results.append({
                    "Date": date,
                    "Regime": regime,
                    "Strategy": strategy,
                    "PnL": pnl,
                    "Cumulative_PnL": portfolio_pnl,
                    "Capital_Deployed": deploy,
                    "Max_Loss": max_loss,
                    "Risk_Reward": risk_reward
                })
            except Exception as e:
                logger.error(f"Error in backtest loop at index {i}: {str(e)}")
                continue

        backtest_df = pd.DataFrame(backtest_results)
        if backtest_df.empty:
            logger.warning(f"No trades generated for {strategy_choice} from {start_date} to {end_date}")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        backtest_df['Cumulative_PnL'] = backtest_df['PnL'].cumsum()
        backtest_df['Peak'] = backtest_df['Cumulative_PnL'].cummax()
        backtest_df['Draw Inlet'] = backtest_df['Peak'] - backtest_df['Cumulative_PnL']
        max_drawdown = backtest_df['Drawdown'].max() if not backtest_df.empty else 0
        backtest_df.set_index("Date", inplace=True)

        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0
        capital_series = pd.Series(capital, index=df.index).add(backtest_df['Cumulative_PnL'].reindex(df.index).fillna(method='ffill').fillna(0))
        daily_backtest_pnl = backtest_df['PnL'].reindex(capital_series.index).fillna(0)
        daily_returns = daily_backtest_pnl / capital_series.shift(1).fillna(capital)
        daily_returns = daily_returns.dropna()

        df_aligned = df.reindex(daily_returns.index)
        nifty_daily_returns = df_aligned["NIFTY_Close"].pct_change().dropna()
        daily_returns_aligned = daily_returns.reindex(nifty_daily_returns.index).fillna(0)
        excess_returns = daily_returns_aligned - nifty_daily_returns - RISK_FREE_RATE_DAILY
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0
        sortino_std_negative = excess_returns[excess_returns < 0].std()
        sortino_ratio = excess_returns.mean() / sortino_std_negative * np.sqrt(252) if sortino_std_negative != 0 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 and capital != 0 else float('inf')

        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_win_rates = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        strategy_perf = pd.merge(strategy_perf, strategy_win_rates, on="Strategy")
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_win_rates = backtest_df.groupby("Regime")["PnL"].apply(lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0).reset_index(name="Win_Rate")
        regime_perf = pd.merge(regime_perf, regime_win_rates, on="Regime")

        logger.debug("Backtest completed successfully")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()
