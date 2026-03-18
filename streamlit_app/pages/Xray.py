import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import streamlit as st

from ui.shell import app_shell
from ui.common import set_page, analyst_toggle, show_advanced

ok, models, analyst_mode_from_shell = app_shell(
    "🩻 Chest X-ray Imaging",
    "Multilabel CNN-based finding detection",
)
if not ok:
    st.stop()

set_page()
st.title("🩻 Chest X-ray Imaging (CNN)")
st.caption("Upload a chest X-ray image to detect likely findings.")

analyst_mode = analyst_toggle()

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

run_clicked = st.button("Run X-ray model")

if run_clicked:
    if uploaded_file is None:
        st.warning("Please upload an image.")
        st.stop()

    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/jpeg",
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/xray/predict",
            files=files,
            timeout=120,
        )
        response.raise_for_status()
        resp = response.json()

        findings = resp.get("output", [])
        meta = resp.get("meta", {})

        st.subheader("Detected Findings")
        if findings:
            for finding in findings:
                st.write(f"- {finding}")
        else:
            st.info("No findings crossed the configured threshold.")

        if "threshold" in meta:
            st.caption(f"Decision threshold: {meta['threshold']}")

        top_scores = meta.get("top_scores", [])
        if top_scores:
            st.subheader("Top Scores")
            for item in top_scores:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    label, score = item
                    st.write(f"**{label}**: {score}")

        probabilities = meta.get("probabilities", {})
        if probabilities:
            with st.expander("All class probabilities"):
                st.json(probabilities)

        if analyst_mode:
            show_advanced(resp)

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")