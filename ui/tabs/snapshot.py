import streamlit as st
from core.data_fetcher import run_volguard
from core.risk_manager import check_risk
from ui.components import render_metric_card, render_highlight_card, render_risk_status
from utils.logger import setup_logger

logger = setup_logger()

def render_snapshot_tab():
    """Render the Snapshot tab."""
    st.header("Market Snapshot")
    access_token = st.text_input("Enter Upstox Access Token", type="password", help="Enter your Upstox access token to fetch live market data.")
    st.session_state.access_token = access_token

    if st.button("Run VolGuard"):
        if not access_token:
            st.error("Please enter a valid Upstox access token.")
        else:
            with st.spinner("Fetching options data..."):
                result, df, iv_skew_fig, atm_strike, atm_iv = run_volguard(access_token)
                if result:
                    st.session_state.volguard_data = result
                    st.session_state.atm_iv = atm_iv
                    st.success("Data fetched successfully!")
                    risk_status, risk_message = check_risk(
                        0, 0, 0, st.session_state.atm_iv, st.session_state.realized_vol,
                        st.session_state.risk_settings['total_capital'], st.session_state.risk_settings
                    )
                    st.session_state.risk_status = risk_status
                    render_risk_status(risk_status, risk_message)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("Market Snapshot")
                        render_metric_card("Timestamp", result['timestamp'], "schedule")
                        render_metric_card("Nifty Spot", f"{result['nifty_spot']:.2f}", "trending_up")
                        render_highlight_card("ATM IV", f"{atm_iv:.2f}%", "percent")
                        render_metric_card("Expiry", result['expiry'], "event")
                        render_metric_card("ATM Strike", f"{result['atm_strike']:.2f}", "attach_money")
                        render_metric_card("Straddle Price", f"{result['straddle_price']:.2f}", "monetization_on")
                        render_metric_card("PCR", f"{result['pcr']:.2f}", "balance")
                        render_metric_card("Max Pain", f"{result['max_pain']:.2f}", "warning")
                        render_metric_card("CE Depth", f"Bid: {result['ce_depth'].get('bid_volume', 0)}, Ask: {result['ce_depth'].get('ask_volume', 0)}", "shopping_cart")
                        render_metric_card("PE Depth", f"Bid: {result['pe_depth'].get('bid_volume', 0)}, Ask: {result['pe_depth'].get('ask_volume', 0)}", "shopping_cart")
                    with col2:
                        if iv_skew_fig:
                            st.subheader("IV Skew Plot")
                            st.plotly_chart(iv_skew_fig, use_container_width=True)

                    with st.expander("Key Strikes (ATM ± 6)"):
                        atm_idx = df[df['Strike'] == atm_strike].index[0]
                        key_strikes = df.iloc[max(0, atm_idx-6):atm_idx+7][[
                            'Strike', 'CE_LTP', 'CE_IV', 'CE_Delta', 'CE_Theta', 'CE_Vega', 'CE_OI',
                            'CE_OI_Change', 'CE_OI_Change_Pct', 'CE_Volume', 'PE_LTP', 'PE_IV', 'PE_Delta',
                            'PE_Theta', 'PE_Vega', 'PE_OI', 'PE_OI_Change', 'PE_OI_Change_Pct', 'PE_Volume',
                            'Strike_PCR', 'OI_Skew', 'IV_Skew_Slope'
                        ]]
                        key_strikes['CE_OI_Change'] = key_strikes['CE_OI_Change'].apply(
                            lambda x: f"{x:.1f}*" if x > 500000 else f"{x:.1f}"
                        )
                        key_strikes['PE_OI_Change'] = key_strikes['PE_OI_Change'].apply(
                            lambda x: f"{x:.1f}*" if x > 500000 else f"{x:.1f}"
                        )
                        for idx, row in key_strikes.iterrows():
                            st.markdown(f"""
                                <div class='metric-card'>
                                    <h4><i class='material-icons'>attach_money</i> Strike: {row['Strike']:.2f}</h4>
                                    <p>CE LTP: {row['CE_LTP']:.2f} | CE IV: {row['CE_IV']:.2f}% | CE OI: {row['CE_OI']:.0f} | OI Change: {row['CE_OI_Change']}</p>
                                    <p>PE LTP: {row['PE_LTP']:.2f} | PE IV: {row['PE_IV']:.2f}% | PE OI: {row['PE_OI']:.0f} | OI Change: {row['PE_OI_Change']}</p>
                                    <p>Strike PCR: {row['Strike_PCR']:.2f} | OI Skew: {row['OI_Skew']:.2f} | IV Skew Slope: {row['IV_Skew_Slope']:.2f}</p>
                                </div>
                            """, unsafe_allow_html=True)
