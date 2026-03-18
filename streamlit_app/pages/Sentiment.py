import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from ui.shell import app_shell
from api_client import predict



ok, models, analyst_mode = app_shell("Sentiment (Patient Feedback)")
if not ok:
    st.stop()

MODEL_ID = "sentiment_tfidf_logreg_v1"

st.caption("The classifier predicts one of these three sentiment classes.")

text = st.text_area("Feedback text", value="The doctor was polite", height=100)

# Always request proba for analyst-first debugging
return_proba = st.checkbox("Return probabilities", value=True)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    resp = predict(MODEL_ID, {"text": text.strip()}, return_proba=return_proba)

    pred = resp.get("output")
    meta = resp.get("meta") or {}

    st.subheader("Prediction")
    st.write(pred)

    # If proba exists, show it nicely
    proba = meta.get("proba")
    label_map = meta.get("label_map") or {}

    if isinstance(proba, list) and len(proba) > 0:
        st.subheader("Probabilities")
        # Try to map indices to class names
        rows = []
        for i, p in enumerate(proba):
            rows.append({"class": label_map.get(str(i), str(i)), "proba": float(p)})
        rows = sorted(rows, key=lambda r: r["proba"], reverse=True)
        st.table(rows)

    if analyst_mode:
        st.divider()
        st.subheader("Raw response (Analyst mode)")
        st.json(resp)

st.divider()
