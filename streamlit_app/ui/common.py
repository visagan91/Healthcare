import json
import streamlit as st

def set_page():
    st.set_page_config(page_title="Healthcare AI Workbench", layout="wide")

def analyst_toggle():
    if "analyst_mode" not in st.session_state:
        st.session_state.analyst_mode = False
    st.session_state.analyst_mode = st.toggle("Analyst mode", value=st.session_state.analyst_mode)
    return st.session_state.analyst_mode

def show_advanced(resp: dict):
    with st.expander("Advanced (raw response + download)"):
        st.json(resp)
        st.download_button(
            "Download JSON",
            data=json.dumps(resp, indent=2),
            file_name="result.json",
            mime="application/json",
        )