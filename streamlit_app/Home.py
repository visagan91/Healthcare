import os
import sys

# Ensure imports work when Streamlit runs from different working dirs
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st

from ui.shell import app_shell

ok, models, analyst_mode = app_shell(
    "🏠 Healthcare AI Workbench",
    "All capabilities in one place (demo + analyst mode)",
)
if not ok:
    st.stop()

# ---- Summary metrics ----
tasks = sorted({m.get("task", "") for m in models if m.get("task")})
kinds = sorted({m.get("kind", "") for m in models if m.get("kind")})

c1, c2, c3 = st.columns(3)
c1.metric("Models", len(models))
c2.metric("Capabilities", len(tasks))
c3.metric("Kinds", len(kinds))

st.divider()

# ---- Home = Dashboard ----
st.subheader("Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Risk")
    st.caption("Classify disease risk category")
    st.page_link("pages/Risk.py", label="Open Risk", icon="⚠️")

    st.markdown("### Length of Stay")
    st.caption("Predict hospitalization duration")
    st.page_link("pages/Regression.py", label="Open LOS", icon="📈")

    st.markdown("### Clustering")
    st.caption("Discover patient subgroups")
    st.page_link("pages/Clustering.py", label="Open Clustering", icon="🧠")

with col2:
    st.markdown("### Pattern Association")
    st.caption("Mine common co-occurrence patterns")
    st.page_link("pages/Pattern Association.py", label="Open Rules", icon="🔗")

    st.markdown("### Sentiment")
    st.caption("Analyze patient feedback")
    st.page_link("pages/Sentiment.py", label="Open Sentiment", icon="😊")

    st.markdown("### 💬 Chatbot (RAG)")
    st.caption("Ask questions + see evidence")
    st.page_link("pages/Chatbot.py", label="Open Chatbot", icon="💬")

with col3:
    st.markdown("### 🩻 Imaging")
    st.caption("Chest X-ray multilabel CNN")
    st.page_link("pages/Xray.py", label="Open Imaging", icon="🩻")

    st.markdown("### Time-series")
    st.caption("RNN/LSTM demo")
    st.page_link("pages/Timeseries.py", label="Open Time-series", icon="📉")

st.divider()

# ---- Analyst mode: model catalog + details ----
if analyst_mode:
    st.subheader("Model Catalog (Analyst mode)")
    st.dataframe(models, use_container_width=True)

    with st.expander("Tasks"):
        st.write(tasks)

    with st.expander("Kinds"):
        st.write(kinds)