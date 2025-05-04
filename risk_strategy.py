import streamlit as st
import pandas as pd
import numpy as np
from arch import arch_model
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Feature columns
feature_cols = [
    'VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price',
    'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos',
    'FII_Option_Pos'
]

# Forecast volatility
@st.cache_data
def forecast_volatility_future(df, forecast_horizon):
    try:
        df.index = pd.to_datetime(df.index)
        df_garch = df.tail(len(df))
        if len(df_garch) < 200:
            st.error(f"Insufficient data for GARCH: {len(df_garch)} days.")
            return None, None, None, None, None, None, None, None

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='B')

        df_garch['Log_Returns'] = np.log(df_garch['NIFTY_Close'] / df_garch['NIFTY_Close'].shift(1)).dropna() * 100
        garch_model = arch_model(df_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1, rescale=False)
        garch_fit = garch_model.fit(disp="off")
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
        garch_vols = np.sqrt(garch_forecast.variance.iloc[-1].values) * np.sqrt(252)
        garch_vols = np.clip(garch_vols, 5, 50)

        realized_vol = df["Realized_Vol"].dropna().iloc[-5:].mean()

        df_xgb = df.tail(len(df))
        df_xgb['Target_Vol'] = df_xgb['Realized_Vol'].shift(-1)
        df_xgb = df_xgb.dropna()

        X = df_xgb[feature_cols]
        y = df_xgb['Target_Vol']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X_scaled.iloc[:split_index], X_scaled.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.03, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        xgb_vols = []
        current_row = df_xgb[feature_cols].iloc[-1].copy()
        for i in range(forecast_horizon):
            current_row_df = pd.DataFrame([current_row], columns=feature_cols)
            current_row_scaled = scaler.transform(current_row_df)
            next_vol = model.predict(current_row_scaled)[0]
            xgb_vols.append(next_vol)

            current_row["Days_to_Expiry"] = max(1, current_row["Days_to_Expiry"] - 1)
            current_row["VIX"] *= np.random.uniform(0.98, 1.02)
            current_row["Straddle_Price"] *= np.random.uniform(0.98, 1.02)
            current_row["VIX_Change_Pct"] = (current_row["VIX"] / df_xgb["VIX"].iloc[-1] - 1) * 100
            current_row["ATM_IV"] = current_row["VIX"] * (1 + np.random.normal(0, 0.1))
            current_row["Realized_Vol"] = np.clip(next_vol * np.random.uniform(0.95, 1.05), 5, 50)
            current_row["IVP"] = current_row["IVP"] * np.random.uniform(0.99, 1.01)
            current_row["PCR"] = np.clip(current_row["PCR"] + np.random.normal(0, 0.05), 0.7, 2.0)
            current_row["Spot_MaxPain_Diff_Pct"] = np.clip(current_row["Spot_MaxPain_Diff_Pct"] * np.random.uniform(0.95, 1.05), 0.1, 1.0)
            current_row["Event_Flag"] = df_xgb["Event_Flag"].iloc[-1]
            current_row["FII_Index_Fut_Pos"] += np.random.normal(0, 1000)
            current_row["FII_Option_Pos"] += np.random.normal(0, 500)
            current_row["IV_Skew"] = np.clip(current_row["IV_Skew"] + np.random.normal(0, 0.1), -3, 3)

        xgb_vols = np.clip(xgb_vols, 5, 50)
        if df["Event_Flag"].iloc[-1] == 1:
            xgb_vols = [v * 1.1 for v in xgb_vols]

        garch_diff = np.abs(garch_vols[0] - realized_vol)
        xgb_diff = np.abs(xgb_vols[0] - realized_vol)
        garch_weight = xgb_diff / (garch_diff + xgb_diff) if (garch_diff + xgb_diff) > 0 else 0.5
        xgb_weight = 1 - garch_weight
        blended_vols = [(garch_weight * g) + (xgb_weight * x) for g, x in zip(garch_vols, xgb_vols)]
        confidence_score = min(100, max(50, 80 - abs(garch_diff - xgb_diff)))

        forecast_log = pd.DataFrame({
            "Date": future_dates,
            "GARCH_Vol": garch_vols,
            "XGBoost_Vol": xgb_vols,
            "Blended_Vol": blended_vols,
            "Confidence": [confidence_score] * forecast_horizon
        })
        logger.debug("Volatility forecast completed.")
        return forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, model.feature_importances_
    except Exception as e:
        st.error(f"Error in volatility forecasting: {str(e)}")
        logger.error(f"Error in volatility forecasting: {str(e)}")
        return None, None, None, None, None, None, None, None

# Generate trading strategy
@st.cache_data
def generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital, portfolio_data):
    try:
        latest = df.iloc[-1]
        avg_vol = np.mean(forecast_log["Blended_Vol"])
        iv = latest["ATM_IV"]
        hv = latest["Realized_Vol"]
        iv_hv_gap = iv - hv
        iv_skew = latest["IV_Skew"]
        pcr = latest["PCR"]
        dte = latest["Days_to_Expiry"]
        event_flag = latest["Event_Flag"]
        max_pain_diff = latest["Spot_MaxPain_Diff_Pct"]

        # Enhanced risk flags
        risk_flags = []
        if latest["VIX"] > 25:
            risk_flags.append("VIX > 25% - High Volatility Risk")
        if portfolio_data["exposure"] > 50:
            risk_flags.append("Exposure > 50% - High Exposure Risk")
        if portfolio_data["weekly_pnl"] < -0.1 * capital:
            risk_flags.append("Portfolio Loss > 10% - High Drawdown Risk")
        if latest["VIX_Change_Pct"] > 10:
            risk_flags.append("VIX Spike > 10% - Volatility Shock Risk")
        if portfolio_data["var"] < -0.05 * capital:
            risk_flags.append("VaR Breach: High Potential Loss")

        # Check risk limits
        if portfolio_data["exposure"] > 50 or portfolio_data["weekly_pnl"] < -0.1 * capital:
            st.session_state.trading_halted = True
            st.session_state.risk_alerts.append("You are way beyond your risk limit! Trading halted.")
            return None

        if risk_flags:
            st.session_state.violations += 1
            if st.session_state.violations >= 2 and not st.session_state.journal_complete:
                st.error("🚨 Discipline Lock: Complete Journaling to Unlock Trading")
                return None

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
        risk_reward = 1.0
        confidence = 0.5 + 0.5 * (1 - np.abs(forecast_log["GARCH_Vol"].iloc[0] - forecast_log["XGBoost_Vol"].iloc[0]) / max(forecast_log["GARCH_Vol"].iloc[0], forecast_log["XGBoost_Vol"].iloc[0]))

        # Dynamic stop-loss and take-profit
        stop_loss_pct = 0.02 if regime == "LOW" else 0.03 if regime == "MEDIUM" else 0.04
        take_profit_pct = stop_loss_pct * 2
        stop_loss = -stop_loss_pct * capital
        take_profit = take_profit_pct * capital

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
                strategy = "Calendar Spread"
                reason = "IV skew and medium vol suggest potential vol expansion."
                tags = ["Volatility Play", "Neutral", "Skew"]
                risk_reward = 1.8
            else:
                strategy = "Short Strangle"
                reason = "Balanced vol, no events, and premium-rich environment."
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

        # Kelly Criterion for position sizing
        expected_return = risk_reward / (1 + risk_reward)
        kelly_fraction = (expected_return - 0.06 / 252) / (stop_loss_pct ** 2) if stop_loss_pct != 0 else 0.1
        kelly_fraction = np.clip(kelly_fraction, 0.05, 0.2)

        capital_alloc = {"LOW": 0.35, "MEDIUM": 0.25, "HIGH": 0.15, "EVENT-DRIVEN": 0.2}
        position_size = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_tolerance]
        deploy = capital * capital_alloc.get(regime, 0.2) * position_size * kelly_fraction
        max_loss = deploy * stop_loss_pct
        total_exposure = deploy / capital

        behavior_score = 8 if deploy < 0.5 * capital else 6
        behavior_warnings = ["Consider reducing position size"] if behavior_score < 7 else []

        logger.debug("Trading strategy generated successfully.")
        return {
            "Regime": regime,
            "Strategy": strategy,
            "Reason": reason,
            "Tags": tags,
            "Confidence": confidence,
            "Risk_Reward": risk_reward,
            "Deploy": deploy,
            "Max_Loss": max_loss,
            "Exposure": total_exposure,
            "Risk_Flags": risk_flags,
            "Behavior_Score": behavior_score,
            "Behavior_Warnings": behavior_warnings,
            "Stop_Loss": stop_loss,
            "Take_Profit": take_profit
        }
    except Exception as e:
        st.error(f"Error generating strategy: {str(e)}")
        logger.error(f"Error generating strategy: {str(e)}")
        return None
