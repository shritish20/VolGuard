import streamlit as st
from app_setup import initialize_session_state, setup_client, render_sidebar, render_main_ui
from data_utils import load_data, generate_synthetic_features
from risk_strategy import forecast_volatility_future, generate_trading_strategy
from backtest_portfolio import fetch_portfolio_data, stress_test_portfolio, run_backtest, square_off_positions
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Added to fix np.mean issue

# Initialize session state
initialize_session_state()

# Setup client
client = setup_client()

# Render sidebar
capital, risk_tolerance, forecast_horizon, start_date, end_date, strategy_choice = render_sidebar(client)

# Main execution
if not st.session_state.logged_in:
    st.info("Please login to 5paisa from the sidebar to proceed.")
else:
    tabs = render_main_ui()

    # Square Off Button in Sidebar
    with st.sidebar:
        if st.session_state.logged_in and st.button("Square Off All Positions"):
            square_off_positions(client)

    # Run Analysis Button
    run_button = st.sidebar.button("Run Analysis")
    if run_button:
        with st.spinner("Running VolGuard Analysis..."):
            try:
                # Reset session state
                st.session_state.backtest_run = False
                st.session_state.backtest_results = None
                st.session_state.violations = 0
                st.session_state.journal_complete = False
                st.session_state.risk_alerts = []

                # Load data with error handling
                df, real_data = load_data()
                if df is None or real_data is None:
                    st.error("Failed to load data. Check logs for details or ensure 5paisa API credentials are correct.")
                    st.stop()

                # Generate synthetic features
                df = generate_synthetic_features(df, real_data, capital)
                if df is None:
                    st.error("Failed to generate synthetic features. Check logs for details.")
                    st.stop()

                # Fetch portfolio data
                portfolio_data = fetch_portfolio_data(client, capital)
                if portfolio_data is None:
                    st.error("Failed to fetch portfolio data. Check logs for details.")
                    st.stop()

                # Run backtest
                backtest_df, total_pnl, win_rate, max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, strategy_perf, regime_perf = run_backtest(
                    df, capital, strategy_choice, start_date, end_date
                )
                if backtest_df.empty:
                    st.warning("Backtest did not generate any trades. Try adjusting the date range or strategy.")
                else:
                    st.session_state.backtest_run = True
                    st.session_state.backtest_results = {
                        "backtest_df": backtest_df,
                        "total_pnl": total_pnl,
                        "win_rate": win_rate,
                        "max_drawdown": max_drawdown,
                        "sharpe_ratio": sharpe_ratio,
                        "sortino_ratio": sortino_ratio,
                        "calmar_ratio": calmar_ratio,
                        "strategy_perf": strategy_perf,
                        "regime_perf": regime_perf
                    }

                # Snapshot Tab
                with tabs[0]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📊 Market Snapshot")
                    last_date = df.index[-1].strftime("%d-%b-%Y")
                    last_nifty = df["NIFTY_Close"].iloc[-1]
                    prev_nifty = df["NIFTY_Close"].iloc[-2] if len(df) >= 2 else last_nifty
                    last_vix = df["VIX"].iloc[-1]
                    regime = "LOW" if last_vix < 15 else "MEDIUM" if last_vix < 20 else "HIGH"
                    regime_class = {"LOW": "regime-low", "MEDIUM": "regime-medium", "HIGH": "regime-high"}[regime]
                    st.markdown(f'<div class="gauge">{regime}</div><div style="text-align: center;">Market Regime</div>', unsafe_allow_html=True)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("NIFTY 50", f"{last_nifty:,.2f}", f"{(last_nifty - prev_nifty)/prev_nifty*100:+.2f}%")
                    with col2:
                        st.metric("India VIX", f"{last_vix:.2f}%", f"{df['VIX_Change_Pct'].iloc[-1]:+.2f}%")
                    with col3:
                        st.metric("PCR", f"{df['PCR'].iloc[-1]:.2f}")
                    with col4:
                        st.metric("Straddle Price", f"₹{df['Straddle_Price'].iloc[-1]:,.2f}")
                    st.markdown(f"**Last Updated**: {last_date} {'(LIVE)' if real_data else '(DEMO)'}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Forecast Tab
                with tabs[1]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📈 Volatility Forecast")
                    with st.spinner("Predicting market volatility..."):
                        forecast_log, garch_vols, xgb_vols, blended_vols, realized_vol, confidence_score, rmse, feature_importances = forecast_volatility_future(df, forecast_horizon)
                    if forecast_log is None:
                        st.error("Volatility forecasting failed. Check logs for details.")
                    else:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg Blended Volatility", f"{np.mean(blended_vols):.2f}%")
                        with col2:
                            st.metric("Realized Volatility", f"{realized_vol:.2f}%")
                        with col3:
                            st.metric("Model RMSE", f"{rmse:.2f}%")
                            st.markdown(f'<div class="gauge">{int(confidence_score)}%</div><div style="text-align: center;">Confidence</div>', unsafe_allow_html=True)
                        st.line_chart(pd.DataFrame({
                            "GARCH": garch_vols,
                            "XGBoost": xgb_vols,
                            "Blended": blended_vols
                        }, index=forecast_log["Date"]), color=["#e94560", "#00d4ff", "#ffcc00"])
                        st.markdown("### Feature Importance")
                        feature_importance = pd.DataFrame({
                            'Feature': ['VIX', 'ATM_IV', 'IVP', 'PCR', 'VIX_Change_Pct', 'IV_Skew', 'Straddle_Price', 'Spot_MaxPain_Diff_Pct', 'Days_to_Expiry', 'Event_Flag', 'FII_Index_Fut_Pos', 'FII_Option_Pos'],
                            'Importance': feature_importances
                        }).sort_values(by='Importance', ascending=False)
                        st.dataframe(feature_importance, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                ## Strategy Tab
    with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎯 Trading Strategies")
    strategy = generate_trading_strategy(df, forecast_log, realized_vol, risk_tolerance, confidence_score, capital, portfolio_data)
    if strategy is None:
        st.markdown('<div class="alert-banner">🚨 Discipline Lock or Risk Limit Breach: Complete Journaling or Reduce Risk to Unlock Trading</div>', unsafe_allow_html=True)
    else:
        regime_class = {
            "LOW": "regime-low",
            "MEDIUM": "regime-medium",
            "HIGH": "regime-high",
            "EVENT-DRIVEN": "regime-event"
        }.get(strategy["Regime"], "regime-low")
        st.markdown('<div class="strategy-carousel">', unsafe_allow_html=True)
        st.markdown(f"""
            <div class="strategy-card">
                <h4>{strategy["Strategy"]}</h4>
                <span class="regime-badge {regime_class}">{strategy["Regime"]}</span>
                <p><b>Reason:</b> {strategy["Reason"]}</p>
                <p><b>Confidence:</b> {strategy["Confidence"]:.2f}</p>
                <p><b>Risk-Reward:</b> {strategy["Risk_Reward"]:.2f}:1</p>
                <p><b>Capital:</b> ₹{strategy["Deploy"]:,.0f}</p>
                <p><b>Max Loss:</b> ₹{strategy["Max_Loss"]:,.0f}</p>
                <p><b>Stop Loss:</b> ₹{strategy["Stop_Loss"]:,.0f}</p>
                <p><b>Take Profit:</b> ₹{strategy["Take_Profit"]:,.0f}</p>
                <p><b>Tags:</b> {', '.join(strategy["Tags"])}</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if strategy["Risk_Flags"]:
            st.markdown(f'<div class="alert-banner">⚠️ Risk Flags: {", ".join(strategy["Risk_Flags"])}</div>', unsafe_allow_html=True)
        if st.button("Trade Now"):
            if st.checkbox("Confirm: I understand trading involves financial risk and market volatility."):
                if st.session_state.trading_halted:
                    st.error("🚨 Trading Halted: Risk Limits Breached!")
                else:
                    try:
                        # Step 1: Check if real_data is available
                        if real_data is None:
                            st.error("Cannot place trade: Market data (real_data) is not loaded. Run analysis first.")
                            st.stop()
                        if "option_chain" not in real_data or real_data["option_chain"] is None:
                            st.error("Cannot place trade: Option chain data missing. Check 5paisa API logs.")
                            st.stop()
                        if "atm_strike" not in real_data or real_data["atm_strike"] is None:
                            st.error("Cannot place trade: ATM strike price missing. Check 5paisa API logs.")
                            st.stop()

                        # Step 2: Extract data
                        capital_deployed = strategy["Deploy"]
                        lot_size = 25  # Fixed lot size for NIFTY options
                        option_chain = real_data["option_chain"]
                        atm_strike = real_data["atm_strike"]

                        # Debug: Check option chain data
                        if option_chain.empty:
                            st.error("Option chain data is empty. Unable to proceed with trade.")
                            st.stop()

                        # Step 3: Calculate strikes
                        call_sell_strike = atm_strike + 100
                        call_buy_strike = call_sell_strike + 100
                        put_sell_strike = atm_strike - 100
                        put_buy_strike = put_sell_strike - 100

                        # Step 4: Fetch prices with error handling
                        call_sell_data = option_chain[(option_chain["StrikeRate"] == call_sell_strike) & (option_chain["CPType"] == "CE")]
                        call_buy_data = option_chain[(option_chain["StrikeRate"] == call_buy_strike) & (option_chain["CPType"] == "CE")]
                        put_sell_data = option_chain[(option_chain["StrikeRate"] == put_sell_strike) & (option_chain["CPType"] == "PE")]
                        put_buy_data = option_chain[(option_chain["StrikeRate"] == put_buy_strike) & (option_chain["CPType"] == "PE")]

                        if call_sell_data.empty or call_buy_data.empty or put_sell_data.empty or put_buy_data.empty:
                            st.error("One or more strike prices not found in option chain. Check strikes or data source.")
                            st.stop()

                        call_sell_price = call_sell_data["LastRate"].iloc[0]
                        call_buy_price = call_buy_data["LastRate"].iloc[0]
                        put_sell_price = put_sell_data["LastRate"].iloc[0]
                        put_buy_price = put_buy_data["LastRate"].iloc[0]

                        # Step 5: Calculate quantity
                        avg_price_per_lot = (call_sell_price + call_buy_price + put_sell_price + put_buy_price) / 4 * lot_size
                        num_lots = int(capital_deployed / avg_price_per_lot) if avg_price_per_lot > 0 else 1
                        num_lots = max(1, min(num_lots, 10))  # Limit between 1 and 10 lots
                        total_quantity = num_lots * lot_size
                        total_cost = total_quantity * (call_sell_price + call_buy_price + put_sell_price + put_buy_price) / 4

                        # Step 6: Display trade details
                        st.markdown("### Trade Details")
                        st.write(f"**Strategy:** {strategy['Strategy']}")
                        st.write(f"**Capital Deployed:** ₹{capital_deployed:,.2f}")
                        st.write(f"**Number of Lots:** {num_lots}")
                        st.write(f"**Total Quantity:** {total_quantity} (Lot Size: {lot_size})")
                        st.write(f"**Estimated Cost:** ₹{total_cost:,.2f}")
                        st.write(f"**Call Sell Strike/Price:** {call_sell_strike}/₹{call_sell_price}")
                        st.write(f"**Call Buy Strike/Price:** {call_buy_strike}/₹{call_buy_price}")
                        st.write(f"**Put Sell Strike/Price:** {put_sell_strike}/₹{put_sell_price}")
                        st.write(f"**Put Buy Strike/Price:** {put_buy_strike}/₹{put_buy_price}")

                        # Step 7: Trade execution
                        orders = [
                            {
                                "Exch": "N", "ExchType": "C",
                                "ScripCode": call_sell_data["ScripCode"].iloc[0],
                                "BuySell": "S", "Qty": total_quantity,
                                "Price": call_sell_price,
                                "OrderType": "LIMIT"
                            },
                            {
                                "Exch": "N", "ExchType": "C",
                                "ScripCode": call_buy_data["ScripCode"].iloc[0],
                                "BuySell": "B", "Qty": total_quantity,
                                "Price": call_buy_price,
                                "OrderType": "LIMIT"
                            },
                            {
                                "Exch": "N", "ExchType": "C",
                                "ScripCode": put_sell_data["ScripCode"].iloc[0],
                                "BuySell": "S", "Qty": total_quantity,
                                "Price": put_sell_price,
                                "OrderType": "LIMIT"
                            },
                            {
                                "Exch": "N", "ExchType": "C",
                                "ScripCode": put_buy_data["ScripCode"].iloc[0],
                                "BuySell": "B", "Qty": total_quantity,
                                "Price": put_buy_price,
                                "OrderType": "LIMIT"
                            }
                        ]
                        for order in orders:
                            response = client.place_order(
                                OrderType=order["BuySell"],
                                Exchange=order["Exch"],
                                ExchangeType=order["ExchType"],
                                ScripCode=order["ScripCode"],
                                Qty=order["Qty"],
                                Price=order["Price"],
                                IsIntraday=False
                            )
                            if response.get("Status") != 0:  # Assuming 0 means success in 5paisa API
                                st.error(f"Order failed for {order['BuySell']} at strike {order['ScripCode']}: {response.get('Message', 'Unknown error')}")
                                st.stop()

                        # Step 8: Log the trade
                        trade_log = {
                            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Strategy": strategy["Strategy"],
                            "Regime": strategy["Regime"],
                            "Risk_Level": "High" if strategy["Risk_Flags"] else "Low",
                            "Outcome": "Pending",
                            "Stop_Loss": strategy["Stop_Loss"],
                            "Take_Profit": strategy["Take_Profit"],
                            "Capital_Deployed": capital_deployed,
                            "Quantity": total_quantity,
                            "Total_Cost": total_cost
                        }
                        st.session_state.trades.append(trade_log)
                        pd.DataFrame(st.session_state.trades).to_csv("trade_log.csv", index=False)
                        st.success("✅ Trade Placed Successfully!")
                    except Exception as e:
                        st.error(f"Trade Failed: {str(e)}. Check 5paisa API logs for more details.")
    st.markdown('</div>', unsafe_allow_html=True)
    
                

                # Portfolio Tab
                with tabs[3]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("💼 Portfolio Overview")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Weekly P&L", f"₹{portfolio_data['weekly_pnl']:,.2f}")
                    with col2:
                        st.metric("Margin Used", f"₹{portfolio_data['margin_used']:,.2f}")
                    with col3:
                        st.metric("Exposure", f"{portfolio_data['exposure']:.2f}%")
                    with col4:
                        st.metric("VaR (5%)", f"₹{portfolio_data['var']:,.2f}")
                    with col5:
                        st.metric("CVaR (5%)", f"₹{portfolio_data['cvar']:,.2f}")
                    st.markdown("### Stress Test Results")
                    stress_results = stress_test_portfolio(df, capital, portfolio_data)
                    for scenario, result in stress_results.items():
                        st.markdown(f"**{scenario}**: P&L ₹{result['PnL']:,.2f} ({result['Loss_Pct']:.2f}%)")
                    st.markdown("### Open Positions")
                    try:
                        positions = client.positions()
                        if isinstance(positions, dict) and "Data" in positions:
                            pos_df = pd.DataFrame(positions["Data"])
                        elif isinstance(positions, list):
                            pos_df = pd.DataFrame(positions)
                        else:
                            pos_df = pd.DataFrame()
                        if not pos_df.empty:
                            st.dataframe(pos_df[["ScripCode", "BuySell", "Qty", "LastRate", "ProfitLoss"]], use_container_width=True)
                        else:
                            st.info("No open positions.")
                    except Exception as e:
                        st.error(f"Error fetching positions: {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Journal Tab
                with tabs[4]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("📝 Trade Journal")
                    st.write("Log your trades to unlock disciplined trading.")
                    with st.form("journal_form"):
                        trade_date = st.date_input("Trade Date", value=datetime.now())
                        trade_strategy = st.selectbox("Strategy", ["Butterfly Spread", "Iron Condor", "Iron Fly", "Short Strangle", "Calendar Spread", "Jade Lizard"])
                        trade_pnl = st.number_input("P&L (₹)", value=0.0, step=100.0)
                        trade_notes = st.text_area("Notes", placeholder="What went well? What could be improved?")
                        submitted = st.form_submit_button("Log Trade")
                        if submitted:
                            journal_entry = {
                                "Date": trade_date,
                                "Strategy": trade_strategy,
                                "P&L": trade_pnl,
                                "Notes": trade_notes
                            }
                            st.session_state.trades.append(journal_entry)
                            pd.DataFrame(st.session_state.trades).to_csv("trade_log.csv", index=False)
                            st.session_state.journal_complete = True
                            st.session_state.violations = 0
                            st.success("✅ Trade logged successfully!")
                            st.rerun()  # Added to refresh the UI after form submission
                    if st.session_state.trades:
                        st.markdown("### Recent Trades")
                        trades_df = pd.DataFrame(st.session_state.trades)
                        st.dataframe(trades_df, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Backtest Tab
                with tabs[5]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("🔍 Backtest Results")
                    if st.session_state.backtest_run and st.session_state.backtest_results:
                        results = st.session_state.backtest_results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total P&L", f"₹{results['total_pnl']:,.2f}")
                        with col2:
                            st.metric("Win Rate", f"{results['win_rate']*100:.2f}%")
                        with col3:
                            st.metric("Max Drawdown", f"₹{results['max_drawdown']:,.2f}")
                        with col4:
                            st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                        st.markdown("### Strategy Performance")
                        st.dataframe(results["strategy_perf"], use_container_width=True)
                        st.markdown("### Regime Performance")
                        st.dataframe(results["regime_perf"], use_container_width=True)
                        st.markdown("### Equity Curve")
                        fig, ax = plt.subplots()
                        results["backtest_df"]["PnL"].cumsum().plot(ax=ax, color="#e94560")
                        ax.set_title("Equity Curve")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Cumulative P&L (₹)")
                        st.pyplot(fig)
                    else:
                        st.info("Run analysis to see backtest results.")
                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}. Check logs for more details.")
