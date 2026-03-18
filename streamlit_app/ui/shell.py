import os
import requests
import streamlit as st

from api_client import BASE_URL, list_models


def _safe_page_link(path: str, label: str, icon: str):
    """
    Streamlit only allows linking to:
      - the entrypoint file (Home.py)
      - files inside streamlit_app/pages/
    And it crashes if the file doesn't exist.
    So we check existence first.
    """
    streamlit_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    abs_path = os.path.join(streamlit_app_dir, path)
    if os.path.exists(abs_path):
        st.page_link(path, label=label, icon=icon)


def app_shell(title: str, subtitle: str | None = None):
    st.set_page_config(page_title="Healthcare AI Workbench", layout="wide")

    # Sidebar
    with st.sidebar:
        st.markdown("### 🩺 Healthcare AI Workbench")

        if "analyst_mode" not in st.session_state:
            st.session_state.analyst_mode = False
        analyst_mode = st.toggle("Analyst mode", value=st.session_state.analyst_mode)
        st.session_state.analyst_mode = analyst_mode

        st.caption(f"Backend: {BASE_URL}")

        ok = True
        models = []
        try:
            models = list_models().get("models", [])
            st.success("Backend connected")
        except requests.exceptions.RequestException:
            ok = False
            st.error("Backend not reachable")
            st.code("uvicorn src.main:app --reload --port 8000")

        st.divider()
        st.caption("Pages")

        # ✅ Use REAL filenames you have in streamlit_app/pages/
        _safe_page_link("Home.py", "Home", "🏠")
        _safe_page_link("pages/Risk.py", "Risk", "⚠️")
        _safe_page_link("pages/Timeseries.py", "Time-series", "📉")

        # Add others only if they exist (won't crash if missing)
        _safe_page_link("pages/Regression.py", "LOS", "📈")
        _safe_page_link("pages/Clustering.py", "Clustering", "🧠")
        _safe_page_link("pages/Pattern Association.py", "Rules", "🔗")
        _safe_page_link("pages/Sentiment.py", "Sentiment", "😊")
        _safe_page_link("pages/Chatbot.py", "Chatbot", "💬")
        _safe_page_link("pages/Xray.py", "Imaging", "🩻")

    # Main header
    st.title(title)
    if subtitle:
        st.caption(subtitle)

    return ok, models, analyst_mode