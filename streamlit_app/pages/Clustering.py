import streamlit as st
import pandas as pd

from api_client import predict
from ui.shell import app_shell
from ui.common import set_page, analyst_toggle, show_advanced

ok, models, analyst_mode = app_shell("Patient grouping", "Groups patients into clinically similar subpopulations based on vitals and laboratory measurements using KMeans segmentation.")
if not ok:
    st.stop()

set_page()
st.title("Patient Clustering")
analyst_mode = analyst_toggle()


model_id = "kmeans_v1"

st.caption(
    "Enter whatever you know. You can leave any field blank — the system will estimate missing values for clustering."
)

def opt_float(label: str, key: str, placeholder: str = "", help: str | None = None):
    """Optional float input: blank -> None"""
    s = st.text_input(label, value="", placeholder=placeholder, key=key, help=help)
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        st.error(f"'{label}' must be a number (or leave blank). You entered: {s}")
        return None

def opt_int(label: str, key: str, placeholder: str = "", help: str | None = None):
    """Optional int input: blank -> None"""
    s = st.text_input(label, value="", placeholder=placeholder, key=key, help=help)
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        st.error(f"'{label}' must be an integer (or leave blank). You entered: {s}")
        return None

# -------------------------
# Inputs (all optional)
# -------------------------
st.subheader("Basics")
col1, col2, col3 = st.columns(3)
with col1:
    triage_level = opt_int("Triage level (1–5)", "triage_level", placeholder="e.g., 3")
with col2:
    age = opt_int("Age", "age", placeholder="e.g., 30")
with col3:
    bmi = opt_float("BMI", "bmi", placeholder="e.g., 24.5")

st.subheader("Vitals")
c1, c2, c3 = st.columns(3)
with c1:
    systolic_bp = opt_float("Systolic BP", "systolic_bp", placeholder="e.g., 120")
    respiratory_rate = opt_float("Respiratory rate", "respiratory_rate", placeholder="e.g., 16")
with c2:
    diastolic_bp = opt_float("Diastolic BP", "diastolic_bp", placeholder="e.g., 80")
    temperature_c = opt_float("Temperature (°C)", "temperature_c", placeholder="e.g., 36.8")
with c3:
    heart_rate = opt_float("Heart rate", "heart_rate", placeholder="e.g., 78")
    spo2 = opt_float("SpO₂ (%)", "spo2", placeholder="e.g., 98")

st.subheader("Labs")
l1, l2, l3 = st.columns(3)
with l1:
    hb_g_dl = opt_float("Hb (g/dL)", "hb_g_dl", placeholder="e.g., 13.5")
    sodium_mmol_l = opt_float("Sodium (mmol/L)", "sodium_mmol_l", placeholder="e.g., 140")
with l2:
    wbc_10e9_l = opt_float("WBC (10^9/L)", "wbc_10e9_l", placeholder="e.g., 7.0")
    potassium_mmol_l = opt_float("Potassium (mmol/L)", "potassium_mmol_l", placeholder="e.g., 4.2")
with l3:
    platelets_10e9_l = opt_float("Platelets (10^9/L)", "platelets_10e9_l", placeholder="e.g., 250")
    creatinine_mg_dl = opt_float("Creatinine (mg/dL)", "creatinine_mg_dl", placeholder="e.g., 0.9")

crp_mg_l = opt_float("CRP (mg/L)", "crp_mg_l", placeholder="e.g., 5")

# Build features dict with ONLY provided fields
features = {
    "triage_level": triage_level,
    "age": age,
    "bmi": bmi,
    "systolic_bp": systolic_bp,
    "diastolic_bp": diastolic_bp,
    "heart_rate": heart_rate,
    "respiratory_rate": respiratory_rate,
    "temperature_c": temperature_c,
    "spo2": spo2,
    "hb_g_dl": hb_g_dl,
    "wbc_10e9_l": wbc_10e9_l,
    "platelets_10e9_l": platelets_10e9_l,
    "sodium_mmol_l": sodium_mmol_l,
    "potassium_mmol_l": potassium_mmol_l,
    "creatinine_mg_dl": creatinine_mg_dl,
    "crp_mg_l": crp_mg_l,
}
features = {k: v for k, v in features.items() if v is not None}

st.divider()

if st.button("Assign cluster"):
    resp = predict(model_id, {"features": features})
    out = resp.get("output")
    meta = resp.get("meta") or {}

    st.subheader("Result")
    st.write("**Cluster ID:**", out)

    # Helpful: show what was missing and imputed (if backend provides it)
    if "imputed_columns" in meta and meta["imputed_columns"]:
        st.info("Imputed missing inputs: " + ", ".join(meta["imputed_columns"]))

    if meta.get("cluster_label"):
        st.write("**Cluster label:**", meta["cluster_label"])

    if meta.get("cluster_profile"):
        st.subheader("Cluster profile (typical values)")
        st.dataframe(pd.DataFrame([meta["cluster_profile"]]), use_container_width=True)

    if analyst_mode:
        show_advanced(resp)