import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Fetch portfolio data
def fetch_portfolio_data(client, capital):
    try:
        positions = client.positions()
        if not positions:
            raise Exception("Failed to fetch positions")

        total_pnl = 0
        total_margin = 0
        total_exposure = 0
        for position in positions:
            total_pnl += position.get("ProfitLoss", 0)
            total_margin += position.get("MarginUsed", 0)
            total_exposure += position.get("Exposure", 0)

        # Calculate VaR and CVaR
        returns = pd.Series([pos.get("ProfitLoss", 0) / capital for pos in positions])
        var = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar = returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var

        return {
            "weekly_pnl": total_pnl,
            "margin_used": total_margin,
            "exposure": total_exposure / capital * 100 if capital > 0 else 0,
            "var": var * capital,
            "cvar": cvar * capital
        }
    except Exception as e:
        logger.error(f"Error fetching portfolio data: {str(e)}")
        return {"weekly_pnl": 0, "margin_used": 0, "exposure": 0, "var": 0, "cvar": 0}

# Stress test portfolio
def stress_test_portfolio(df, capital, portfolio_data):
    try:
        scenarios = {
            "Crash": {"nifty_drop": -0.2, "vol_spike": 2.0},
            "Vol Surge": {"nifty_drop": -0.05, "vol_spike": 1.5},
            "Flat High Vol": {"nifty_drop": 0.0, "vol_spike": 1.3}
        }
        stress_results = {}
        for name, params in scenarios.items():
            stressed_pnl = portfolio_data["weekly_pnl"] * (1 + params["nifty_drop"]) * params["vol_spike"]
            stress_results[name] = {
                "PnL": stressed_pnl,
                "Loss_Pct": stressed_pnl / capital * 100 if capital > 0 else 0
            }
        return stress_results
    except Exception as e:
        logger.error(f"Error in stress testing: {str(e)}")
        return {}

# Backtest function
def run_backtest(df, capital, strategy_choice, start_date, end_date):
    try:
        logger.debug(f"Starting backtest for {strategy_choice} from {start_date} to {end_date}")
        if df.empty:
            st.error("Backtest failed: No data available.")
            logger.error("Backtest failed: Empty DataFrame")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()
        
        df_backtest = df.loc[start_date:end_date].copy()
        if len(df_backtest) < 50:
            st.error(f"Backtest failed: Insufficient data ({len(df_backtest)} days). Need at least 50 days.")
            logger.error(f"Backtest failed: Insufficient data ({len(df_backtest)} days)")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()
        
        required_cols = ["NIFTY_Close", "ATM_IV", "Realized_Vol", "IV_Skew", "Days_to_Expiry", "Event_Flag", "Total_Capital", "Straddle_Price"]
        missing_cols = [col for col in required_cols if col not in df_backtest.columns]
        if missing_cols:
            st.error(f"Backtest failed: Missing columns {missing_cols}")
            logger.error(f"Backtest failed: Missing columns {missing_cols}")
            return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        backtest_results = []
        lot_size = 25
        base_transaction_cost = 0.002
        stt = 0.0005
        portfolio_pnl = 0
        risk_free_rate = 0.06 / 126
        nifty_returns = df_backtest["NIFTY_Close"].pct_change()

        def run_strategy_engine(day_data, avg_vol, portfolio_pnl):
            try:
                iv = day_data["ATM_IV"]
                hv = day_data["Realized_Vol"]
                iv_hv_gap = iv - hv
                iv_skew = day_data["IV_Skew"]
                dte = day_data["Days_to_Expiry"]
                event_flag = day_data["Event_Flag"]

                if portfolio_pnl < -0.1 * day_data["Total_Capital"]:
                    return None, None, "Portfolio drawdown limit reached", [], 0, 0, 0, 0, 0

                if event_flag == 1:
                    regime = "EVENT-DRIVEN"
                elif avg_vol < 15:
                    regime = "LOW"
                elif avg_vol < 20:
                    regime = "MEDIUM"
                else:
                    regime = "HIGH"

                strategy = "Undefined"
                reason = "N/A"
                tags = []
                risk_reward = 1.5 if iv_hv_gap > 5 else 1.0

                stop_loss_pct = 0.02 if regime == "LOW" else 0.03 if regime == "MEDIUM" else 0.04
                take_profit_pct = stop_loss_pct * 2

                if regime == "LOW":
                    if iv_hv_gap > 5 and dte < 10:
                        strategy = "Butterfly Spread"
                        reason = "Low vol & short expiry favors pinning strategies."
                        tags = ["Neutral", "Theta", "Expiry Play"]
                        risk_reward = 2.0
                    else:
                        strategy = "Iron Fly"
                        reason = "Low volatility and time decay favors delta-neutral Iron Fly."
                        tags = ["Neutral", "Theta", "Range Bound"]

                elif regime == "MEDIUM":
                    if iv_hv_gap > 3 and iv_skew > 2:
                        strategy = "Iron Condor"
                        reason = "Medium vol and skew favor wide-range Iron Condor."
                        tags = ["Neutral", "Theta", "Range Bound"]
                        risk_reward = 1.8
                    else:
                        strategy = "Short Strangle"
                        reason = "Balanced vol, premium-rich environment for Short Strangle."
                        tags = ["Neutral", "Premium Selling", "Volatility Harvest"]

                elif regime == "HIGH":
                    if iv_hv_gap > 10:
                        strategy = "Jade Lizard"
                        reason = "High IV + call skew = Jade Lizard for defined upside risk."
                        tags = ["Skewed", "Volatility", "Defined Risk"]
                        risk_reward = 1.2
                    else:
                        strategy = "Iron Condor"
                        reason = "High vol favors wide-range Iron Condor for premium collection."
                        tags = ["Neutral", "Theta", "Range Bound"]

                elif regime == "EVENT-DRIVEN":
                    if iv > 30 and dte < 5:
                        strategy = "Calendar Spread"
                        reason = "Event + near expiry + IV spike → term structure opportunity."
                        tags = ["Volatility", "Event", "Calendar"]
                        risk_reward = 1.5
                    else:
                        strategy = "Iron Fly"
                        reason = "Event-based uncertainty favors defined-risk Iron Fly."
                        tags = ["Neutral", "Theta", "Event"]

                capital = day_data["Total_Capital"]
                capital_alloc = {"LOW": 0.10, "MEDIUM": 0.08, "HIGH": 0.06, "EVENT-DRIVEN": 0.06}
                deploy = capital * capital_alloc.get(regime, 0.06)
                max_loss = deploy * stop_loss_pct
                return regime, strategy, reason, tags, deploy, max_loss, risk_reward, stop_loss_pct, take_profit_pct
            except Exception as e:
                logger.error(f"Error in strategy engine: {str(e)}")
                return None, None, f"Strategy engine failed: {str(e)}", [], 0, 0, 0, 0, 0

        def get_dynamic_slippage(strategy, iv, dte):
            base_slippage = 0.005
            iv_multiplier = min(iv / 20, 2.5)
            dte_factor = 1.5 if dte < 5 else 1.0
            strategy_multipliers = {
                "Iron Condor": 1.8,
                "Butterfly Spread": 2.2,
                "Iron Fly": 1.5,
                "Short Strangle": 1.6,
                "Calendar Spread": 1.3,
                "Jade Lizard": 1.4
            }
            return base_slippage * strategy_multipliers.get(strategy, 1.0) * iv_multiplier * dte_factor

        def apply_volatility_shock(pnl, nifty_move, iv, event_flag):
            shock_prob = 0.35 if event_flag == 1 else 0.20
            if np.random.rand() < shock_prob:
                shock_factor = nifty_move / (iv * 100) if iv != 0 else 1.0
                shock = -abs(pnl) * min(shock_factor * 1.5, 2.0)
                return shock
            return pnl

        def apply_liquidity_discount(premium):
            if np.random.rand() < 0.05:
                return premium * 0.8
            return premium

        def apply_execution_delay(premium):
            if np.random.rand() < 0.10:
                return premium * 0.9
            return premium

        for i in range(1, len(df_backtest)):
            try:
                day_data = df_backtest.iloc[i]
                prev_day = df_backtest.iloc[i-1]
                date = day_data.name
                avg_vol = df_backtest["Realized_Vol"].iloc[max(0, i-5):i].mean()

                regime, strategy, reason, tags, deploy, max_loss, risk_reward, stop_loss_pct, take_profit_pct = run_strategy_engine(day_data, avg_vol, portfolio_pnl)
                
                if strategy is None or (strategy_choice != "All Strategies" and strategy != strategy_choice):
                    continue

                extra_cost = 0.001 if "Iron" in strategy else 0
                total_cost = base_transaction_cost + extra_cost + stt
                slippage = get_dynamic_slippage(strategy, day_data["ATM_IV"], day_data["Days_to_Expiry"])
                entry_price = day_data["Straddle_Price"]
                lots = int(deploy / (entry_price * lot_size))
                lots = max(1, min(lots, 2))

                decay_factor = max(0.75, 1 - day_data["Days_to_Expiry"] / 10)
                premium = entry_price * lot_size * lots * (1 - slippage - total_cost) * decay_factor
                premium = apply_liquidity_discount(premium)
                premium = apply_execution_delay(premium)

                iv_factor = min(day_data["ATM_IV"] / avg_vol, 1.5) if avg_vol != 0 else 1.0
                breakeven_factor = 0.04 if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) else 0.06
                breakeven = entry_price * (1 + iv_factor * breakeven_factor)
                nifty_move = abs(day_data["NIFTY_Close"] - prev_day["NIFTY_Close"])
                loss = max(0, nifty_move - breakeven) * lot_size * lots

                max_strategy_loss = premium * 0.6 if strategy in ["Iron Fly", "Iron Condor"] else premium * 0.8
                loss = min(loss, max_strategy_loss)
                pnl = premium - loss

                pnl = apply_volatility_shock(pnl, nifty_move, day_data["ATM_IV"], day_data["Event_Flag"])
                if (day_data["Event_Flag"] == 1 or day_data["ATM_IV"] > 25) and np.random.rand() < 0.08:
                    gap_loss = premium * np.random.uniform(0.5, 1.0)
                    pnl -= gap_loss
                if np.random.rand() < 0.02:
                    crash_loss = premium * np.random.uniform(1.0, 1.5)
                    pnl -= crash_loss

                stop_loss = -stop_loss_pct * deploy
                take_profit = take_profit_pct * deploy
                pnl = max(stop_loss, min(pnl, take_profit))
                portfolio_pnl += pnl

                backtest_results.append({
                    "Date": date,
                    "Regime": regime,
                    "Strategy": strategy,
                    "PnL": pnl,
                    "Capital_Deployed": deploy,
                    "Max_Loss": max_loss,
                    "Risk_Reward": risk_reward
                })
            except Exception as e:
                logger.error(f"Error in backtest loop at index {i}: {str(e)}")
                continue

        backtest_df = pd.DataFrame(backtest_results)
        if len(backtest_df) == 0:
            logger.warning(f"No trades generated for {strategy_choice} from {start_date} to {end_date}")
            return backtest_df, 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

        total_pnl = backtest_df["PnL"].sum()
        win_rate = len(backtest_df[backtest_df["PnL"] > 0]) / len(backtest_df) if len(backtest_df) > 0 else 0
        max_drawdown = (backtest_df["PnL"].cumsum().cummax() - backtest_df["PnL"].cumsum()).max() if len(backtest_df) > 0 else 0

        backtest_df.set_index("Date", inplace=True)
        returns = backtest_df["PnL"] / df_backtest["Total_Capital"].reindex(backtest_df.index, method="ffill")
        nifty_returns = df_backtest["NIFTY_Close"].pct_change().reindex(backtest_df.index, method="ffill").fillna(0)
        excess_returns = returns - nifty_returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(126) if excess_returns.std() != 0 else 0
        sortino_ratio = excess_returns.mean() / excess_returns[excess_returns < 0].std() * np.sqrt(126) if len(excess_returns[excess_returns < 0]) > 0 and excess_returns[excess_returns < 0].std() != 0 else 0
        calmar_ratio = (total_pnl / capital) / (max_drawdown / capital) if max_drawdown != 0 else 0

        strategy_perf = backtest_df.groupby("Strategy")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        strategy_perf["Win_Rate"] = backtest_df.groupby("Strategy")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)
        regime_perf = backtest_df.groupby("Regime")["PnL"].agg(['sum', 'count', 'mean']).reset_index()
        regime_perf["Win_Rate"] = backtest_df.groupby("Regime")["PnL"].apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0).reset_index(drop=True)

        logger.debug("Backtest completed successfully.")
        return backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf
    except Exception as e:
        st.error(f"Error running backtest: {str(e)}")
        logger.error(f"Error running backtest: {str(e)}")
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0, pd.DataFrame(), pd.DataFrame()

# Square off all positions
def square_off_positions(client):
    if st.checkbox("Confirm: I understand this will close all open positions."):
        try:
            positions = client.positions()
            for pos in positions:
                client.place_order(
                    OrderType="B" if pos["BuySell"] == "S" else "S",
                    Exchange=pos["Exch"],
                    ExchangeType=pos["ExchType"],
                    ScripCode=pos["ScripCode"],
                    Qty=pos["Qty"],
                    Price=pos["LastRate"],
                    IsIntraday=False
                )
            st.success("✅ All positions squared off!")
            st.session_state.trading_halted = True
            st.session_state.risk_alerts.append("Trading halted post square-off.")
        except Exception as e:
            st.error(f"Square Off Failed: {str(e)}")
