import streamlit as st

def inject_custom_css():
    """Inject shared custom CSS for the farmer portal for premium presentation."""
    st.markdown("""
        <style>
            .css-18e3th9 { padding-top: 2rem; }
            .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; background-color: #10b981; color: white; border: none; padding: 12px; }
            .stButton>button:hover { background-color: #059669; }
        </style>
    """, unsafe_allow_html=True)
