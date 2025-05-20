import streamlit as st
import pandas as pd
from datetime import datetime
from ui.components import render_metric_card
from utils.logger import setup_logger

logger = setup_logger()

def render_journal_tab():
    """Render the Journal tab."""
    st.header("Trading Journal")
    
    # Journal Entry Form
    with st.form("journal_form"):
        notes = st.text_area("Trade Notes", help="Enter your thoughts, rationale, or observations for the trade.")
        sentiment = st.selectbox("Market Sentiment", ["Bullish", "Bearish", "Neutral"])
        submitted = st.form_submit_button("Add Journal Entry")
        
        if submitted:
            if notes.strip():
                st.session_state.journal_entries.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "notes": notes,
                    "sentiment": sentiment
                })
                st.success("Journal entry added!")
            else:
                st.error("Please enter some notes before submitting.")

    # Display Journal Entries
    if st.session_state.journal_entries:
        st.subheader("Journal Entries")
        journal_df = pd.DataFrame(st.session_state.journal_entries)
        for idx, row in journal_df.iterrows():
            render_metric_card(
                f"Entry at {row['timestamp']}",
                f"Sentiment: {row['sentiment']}\nNotes: {row['notes']}",
                "note"
            )
    else:
        st.info("No journal entries yet. Add one above!")
