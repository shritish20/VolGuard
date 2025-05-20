import streamlit as st
from config.settings import set_page_config, apply_custom_css
from session.state_manager import initialize_session_state
from ui.components import render_top_bar
from ui.tabs.snapshot import render_snapshot_tab
from ui.tabs.forecast import render_forecast_tab
from ui.tabs.prediction import render_prediction_tab
from ui.tabs.strategies import render_strategies_tab
from ui.tabs.dashboard import render_dashboard_tab
from ui.tabs.journal import render_journal_tab

def main():
    """Main function to render the VolGuard Pro application."""
    try:
        # Initial setup
        set_page_config()
        apply_custom_css()
        initialize_session_state()
        
        # Debug message to confirm app loads
        st.markdown("VolGuard Pro app loaded successfully.")
        
        # Render top bar and sidebar
        render_top_bar()
        
        # Sidebar navigation
        tabs = {
            "Snapshot": render_snapshot_tab,
            "Forecast": render_forecast_tab,
            "Prediction": render_prediction_tab,
            "Strategies": render_strategies_tab,
            "Dashboard": render_dashboard_tab,
            "Journal": render_journal_tab
        }
        
        selected_tab = st.sidebar.selectbox("Navigate", list(tabs.keys()))
        
        # Render selected tab with error handling
        try:
            tabs[selected_tab]()
        except Exception as e:
            st.error(f"Error rendering {selected_tab} tab: {str(e)}")
            st.write("Please check the logs or refresh the app.")
            
    except Exception as e:
        st.error(f"Critical error in app setup: {str(e)}")
        st.write("The application failed to initialize. Please check the logs.")

if __name__ == "__main__":
    main()
