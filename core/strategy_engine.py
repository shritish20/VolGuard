import streamlit as st
import pandas as pd
import numpy as np
from upstox_client import Configuration, ApiClient, OrderApiV3, PlaceOrderV3Request
from upstox_client.rest import ApiException
import json
import time
from utils.logger import setup_logger
from core.risk_manager import check_risk

logger = setup_logger()

def find_atm_strike(spot_price, strikes):
    """Find the ATM strike price."""
    try:
        return min(strikes, key=lambda x: abs(x - spot_price))
    except Exception as e:
        logger.error(f"ATM strike error: {e}")
        return spot_price

def build_strategy_legs(option_chain, spot_price, strategy_name, quantity, otm_distance=50):
    """Build strategy legs for execution."""
    try:
        quantity = int(float(quantity))
        strikes = [leg['strike_price'] for leg in option_chain]
        atm_strike = find_atm_strike(spot_price, strikes)
        legs = []

        def get_key(strike, opt_type):
            for leg in option_chain:
                if leg['strike_price'] == strike:
                    if opt_type == 'CE':
                        return leg.get('call_options', {}).get('instrument_key')
                    elif opt_type == 'PE':
                        return leg.get('put_options', {}).get('instrument_key')
            return None

        s = strategy_name.lower()
        if s == "iron_fly":
            legs = [
                {"instrument_key": get_key(atm_strike, "CE"), "strike": atm_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike, "PE"), "strike": atm_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "iron_condor":
            legs = [
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike + 2 * otm_distance, "CE"), "strike": atm_strike + 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "strike": atm_strike - 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "short_straddle":
            legs = [
                {"instrument_key": get_key(atm_strike, "CE"), "strike": atm_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike, "PE"), "strike": atm_strike, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "short_strangle":
            legs = [
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "bull_put_credit":
            legs = [
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "strike": atm_strike - 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "bear_call_credit":
            legs = [
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike + 2 * otm_distance, "CE"), "strike": atm_strike + 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        elif s == "jade_lizard":
            legs = [
                {"instrument_key": get_key(atm_strike + otm_distance, "CE"), "strike": atm_strike + otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - otm_distance, "PE"), "strike": atm_strike - otm_distance, "action": "SELL", "quantity": quantity, "order_type": "MARKET"},
                {"instrument_key": get_key(atm_strike - 2 * otm_distance, "PE"), "strike": atm_strike - 2 * otm_distance, "action": "BUY", "quantity": quantity, "order_type": "MARKET"},
            ]
        else:
            raise ValueError(f"Unsupported strategy: {strategy_name}")

        legs = [leg for leg in legs if leg["instrument_key"]]
        if not legs:
            raise ValueError("No valid legs generated due to missing instrument keys.")
        return legs
    except Exception as e:
        logger.error(f"Strategy legs error: {e}")
        return []

def place_order_for_leg(order_api, leg):
    """Place order for a single leg."""
    try:
        price_val = 0 if leg["order_type"] == "MARKET" else leg.get("price", 0)
        body = PlaceOrderV3Request(
            instrument_token=leg["instrument_key"],
            transaction_type=leg["action"],
            order_type=leg["order_type"],
            product="I",
            quantity=leg["quantity"],
            validity="DAY",
            disclosed_quantity=0,
            trigger_price=0.0,
            tag="volguard",
            is_amo=False,
            slice=False,
            price=price_val
        )
        logger.info(f"Placing order: {body.to_dict()}")
        response = order_api.place_order(body)
        return response.to_dict()
    except ApiException as e:
        error_msg = str(e)
        try:
            error_json = json.loads(e.body)
            reason = error_json.get("error", {}).get("message", "Unknown API error")
        except:
            reason = error_msg
        logger.error(f"Order failed for {leg['instrument_key']}: {reason}")
        logger.error(f"Payload used: {body.to_dict()}")
        return None, reason

def fetch_trade_pnl(order_api, order_id):
    """Fetch P&L for an order."""
    try:
        trade_details = order_api.get_order_details(order_id=order_id, api_version="v2").to_dict()
        trade_pnl = trade_details.get('data', {}).get('pnl', 0) or 0
        return trade_pnl
    except Exception as e:
        logger.error(f"Failed to fetch P&L for order {order_id}: {e}")
        return 0

def update_trade_metrics(pnl):
    """Update trade metrics in session state."""
    try:
        metrics = st.session_state.trade_metrics
        metrics['total_trades'] += 1
        metrics['total_pnl'] += pnl
        if pnl > 0:
            metrics['winning_trades'] += 1
        elif pnl < 0:
            metrics['losing_trades'] += 1
        metrics['pnl_history'].append({"timestamp": datetime.now(), "pnl": pnl})
    except Exception as e:
        logger.error(f"Trade metrics update error: {e}")

def calculate_regime_score(atm_iv, realized_vol, pcr, vix=15.0, iv_skew_slope=0):
    """Calculate market regime score."""
    try:
        score = 0
        if realized_vol > 0:
            iv_rv_ratio = atm_iv / realized_vol
            if iv_rv_ratio > 1.5:
                score += 30
            elif iv_rv_ratio > 1.2:
                score += 20
            elif iv_rv_ratio < 0.8:
                score -= 10
        ivp = 50.0
        if atm_iv > 25:
            ivp = 75
            score += 20
        elif atm_iv < 15:
            ivp = 25
            score -= 10
        if pcr > 1.5:
            score += 20
        elif pcr < 0.8:
            score -= 10
        if vix > 20:
            score += 20
        elif vix < 12:
            score -= 10
        if iv_skew_slope > 0.1:
            score += 10
        elif iv_skew_slope < -0.1:
            score -= 5
        score = max(0, min(100, score))
        if score > 80:
            regime = "High Vol Trend"
            explanation = "Market expects significant volatility. Consider hedged strategies like Iron Fly with long options."
        elif score > 60:
            regime = "Elevated Volatility"
            explanation = "Volatility is above average. Defensive strategies like Iron Condor are suitable."
        elif score > 40:
            regime = "Neutral Volatility"
            explanation = "Market is balanced. Explore strategies like Jade Lizard or Bull Put Credit."
        else:
            regime = "Low Volatility"
            explanation = "Market is calm. Aggressive strategies like Short Straddle may work, but monitor closely."
        return score, regime, explanation
    except Exception as e:
        logger.error(f"Regime score error: {e}")
        return 50, "Neutral Volatility", "Unable to classify regime due to data issues."

def monte_carlo_expiry_simulation(legs, spot_price, num_simulations=1000, days_to_expiry=5, volatility=0.2):
    """Run Monte Carlo simulation for strategy expiry outcomes."""
    try:
        results = []
        for _ in range(num_simulations):
            daily_returns = np.random.normal(loc=0, scale=volatility / np.sqrt(252), size=days_to_expiry)
            simulated_spot = spot_price * np.prod(1 + daily_returns)
            total_pnl = 0
            for leg in legs:
                strike = leg['strike']
                qty = int(leg.get('quantity', 0)) * 75
                opt_type = 'CE' if 'CE' in leg['instrument_key'] else 'PE'
                action = leg['action']
                intrinsic = max(0, simulated_spot - strike) if opt_type == "CE" else max(0, strike - simulated_spot)
                payoff = -intrinsic if action == "SELL" else intrinsic
                total_pnl += payoff * qty
            results.append(total_pnl)
        return results
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        return []

def execute_strategy(access_token, option_chain, spot_price, strategy_name, quantity, df, total_capital, risk_settings):
    """Execute a trading strategy."""
    try:
        configuration = Configuration()
        configuration.access_token = access_token
        client = ApiClient(configuration)
        order_api = OrderApiV3(client)
        quantity = int(float(quantity))
        legs = build_strategy_legs(option_chain, spot_price, strategy_name, quantity)
        if not legs:
            logger.error(f"No valid legs generated for {strategy_name}")
            return None, 0, 0, 0, []

        st.write("**Strategy Legs:**")
        for leg in legs:
            st.write(f"- {leg['action']} {leg['instrument_key']} (Strike: {leg.get('strike', 'N/A')}, Qty: {leg['quantity']})")

        max_loss = 0
        entry_price = 0
        for leg in legs:
            try:
                strike = leg.get('strike', 0)
                qty = leg['quantity']
                opt_type = 'CE' if 'CALL' in leg['instrument_key'].upper() else 'PE'
                row = df[df['Strike'] == strike]
                if not row.empty:
                    ltp = float(row[f'{opt_type}_LTP'].iloc[0])
                    if leg['action'] == 'SELL':
                        max_loss += ltp * qty
                        entry_price += ltp * qty
                    else:
                        max_loss -= ltp * qty
                        entry_price -= ltp * qty
                else:
                    st.warning(f"No data found for strike {strike} ({opt_type}).")
                    logger.warning(f"No data for strike {strike} ({opt_type})")
            except Exception as e:
                logger.error(f"Error calculating leg metrics for {leg['instrument_key']}: {e}")
                return None, 0, 0, 0, []

        max_loss = abs(max_loss)
        capital_to_deploy = max_loss * 1.5
        risk_status, risk_message = check_risk(capital_to_deploy, max_loss, 0, st.session_state.atm_iv, st.session_state.realized_vol, total_capital, risk_settings)
        if risk_status == "red":
            logger.error(f"Risk check failed: {risk_message}")
            return None, 0, 0, 0, []
        elif risk_status == "yellow":
            st.warning(risk_message)

        st.write("\n**Placing Orders...**")
        order_results = []
        total_pnl = 0
        for leg in legs:
            result, error_reason = place_order_for_leg(order_api, leg)
            if result:
                order_results.append(result)
                order_id = result.get('data', {}).get('order_id')
                if order_id:
                    time.sleep(2)
                    pnl = fetch_trade_pnl(order_api, order_id)
                    total_pnl += pnl
                st.success(f"Order placed: {leg['action']} {leg['instrument_key']} (Qty: {leg['quantity']})")
                logger.info(f"Order placed: {leg['action']} {leg['instrument_key']} qty={leg['quantity']}")
            else:
                st.error(f"Order failed for {leg['instrument_key']}.\n\n**Reason:** {error_reason}")
                return None, 0, 0, 0, []

        st.session_state.deployed_capital += capital_to_deploy
        st.session_state.daily_pnl += total_pnl
        update_trade_metrics(total_pnl)
        regime_score, _, _ = calculate_regime_score(st.session_state.atm_iv, st.session_state.realized_vol, df['Strike_PCR'].mean())
        st.session_state.trade_log.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "strategy": strategy_name.replace('_', ' '),
            "capital": capital_to_deploy,
            "pnl": total_pnl,
            "quantity": quantity * 75,
            "regime_score": regime_score,
            "entry_price": entry_price,
            "max_loss": max_loss
        })

        logger.info(f"Strategy executed: {strategy_name}, P&L: {total_pnl}, Capital: {capital_to_deploy}")
        return order_results, total_pnl, entry_price, max_loss, legs
    except Exception as e:
        logger.error(f"Strategy execution error: {e}")
        return None, 0, 0, 0, []
