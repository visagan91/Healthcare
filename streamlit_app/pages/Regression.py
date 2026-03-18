import streamlit as st

from api_client import predict
from ui.shell import app_shell
from ui.common import show_advanced

ok, models, analyst_mode = app_shell(
    "Regression",
    "Predict the patient’s expected hospital Length of Stay (LOS) in days using clinical, vital, and laboratory inputs."
)
if not ok:
    st.stop()

st.title("Length of Stay (LOS) Prediction")

st.caption("Fill what you know. You can leave any field blank — missing values will be estimated for prediction.")

MODEL_OPTIONS = {
    "Linear Regression (los_linreg_v1)": "los_linreg_v1",
    "Random Forest (los_rf_v1)": "los_rf_v1",
}
choice = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[choice]

def opt_float(label: str, key: str, placeholder: str = ""):
    s = st.text_input(label, value="", placeholder=placeholder, key=key)
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        st.warning(f"Ignoring '{label}': please enter a number or leave blank.")
        return None

def opt_int(label: str, key: str, placeholder: str = ""):
    s = st.text_input(label, value="", placeholder=placeholder, key=key)
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        st.warning(f"Ignoring '{label}': please enter an integer or leave blank.")
        return None

st.subheader("Basics")
c1, c2, c3 = st.columns(3)
with c1:
    triage_level = opt_int("Triage level (1–5)", "los_triage", "e.g., 3")
with c2:
    age = opt_int("Age", "los_age", "e.g., 35")
with c3:
    bmi = opt_float("BMI", "los_bmi", "e.g., 26.4")

st.subheader("Vitals")
v1, v2, v3 = st.columns(3)
with v1:
    systolic_bp = opt_float("Systolic BP", "los_sys", "e.g., 120")
    respiratory_rate = opt_float("Respiratory rate", "los_rr", "e.g., 16")
with v2:
    diastolic_bp = opt_float("Diastolic BP", "los_dia", "e.g., 80")
    temperature_c = opt_float("Temperature (°C)", "los_temp", "e.g., 36.8")
with v3:
    heart_rate = opt_float("Heart rate", "los_hr", "e.g., 78")
    spo2 = opt_float("SpO₂ (%)", "los_spo2", "e.g., 98")

st.subheader("Labs")
l1, l2, l3 = st.columns(3)
with l1:
    hb_g_dl = opt_float("Hb (g/dL)", "los_hb", "e.g., 13.5")
    sodium_mmol_l = opt_float("Sodium (mmol/L)", "los_na", "e.g., 140")
    bun_mg_dl = opt_float("BUN (mg/dL)", "los_bun", "e.g., 14")
    glucose_mg_dl = opt_float("Glucose (mg/dL)", "los_glu", "e.g., 95")
    ldl_mg_dl = opt_float("LDL (mg/dL)", "los_ldl", "e.g., 110")
with l2:
    wbc_10e9_l = opt_float("WBC (10^9/L)", "los_wbc", "e.g., 7.0")
    potassium_mmol_l = opt_float("Potassium (mmol/L)", "los_k", "e.g., 4.2")
    creatinine_mg_dl = opt_float("Creatinine (mg/dL)", "los_cr", "e.g., 0.9")
    egfr_ml_min = opt_float("eGFR (mL/min)", "los_egfr", "e.g., 90")
    hba1c_pct = opt_float("HbA1c (%)", "los_a1c", "e.g., 5.4")
with l3:
    platelets_10e9_l = opt_float("Platelets (10^9/L)", "los_plts", "e.g., 250")
    crp_mg_l = opt_float("CRP (mg/L)", "los_crp", "e.g., 5")
    troponin_ng_ml = opt_float("Troponin (ng/mL)", "los_trop", "e.g., 0.01")
    lactate_mmol_l = opt_float("Lactate (mmol/L)", "los_lac", "e.g., 1.2")

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
    "bun_mg_dl": bun_mg_dl,
    "creatinine_mg_dl": creatinine_mg_dl,
    "egfr_ml_min": egfr_ml_min,
    "glucose_mg_dl": glucose_mg_dl,
    "hba1c_pct": hba1c_pct,
    "ldl_mg_dl": ldl_mg_dl,
    "crp_mg_l": crp_mg_l,
    "troponin_ng_ml": troponin_ng_ml,
    "lactate_mmol_l": lactate_mmol_l,
}
features = {k: v for k, v in features.items() if v is not None}

st.divider()

if st.button("Predict LOS"):
    resp = predict(model_id, {"features": features})
    out = resp.get("output")
    meta = resp.get("meta") or {}

    st.subheader("Predicted LOS")
    try:
        st.metric("Length of Stay (days)", f"{float(out):.2f}")
    except Exception:
        st.write(out)

    if meta.get("imputed_columns"):
        st.info("Imputed missing inputs: " + ", ".join(meta["imputed_columns"]))

    if analyst_mode:
        show_advanced(resp)