import streamlit as st
from api_client import predict
from ui.shell import app_shell

ok, models, analyst_mode = app_shell("Risk Prediction", "Predict patient disease risk level using demographic, vital, laboratory, and clinical indicators.")
if not ok:
    st.stop()

st.title(" Disease Risk Prediction")

st.caption(
    "Predict patient disease risk level using demographic, vital, laboratory, "
    "and clinical indicators. Leave fields blank if unavailable."
)

# ------------------------------------------------
# Model Selection
# ------------------------------------------------
MODEL_OPTIONS = {
    "Logistic Regression (risk_logreg_v1)": "risk_logreg_v1",
    "Random Forest (risk_rf_v1)": "risk_rf_v1",
}

selected_model_name = st.selectbox(
    "Select Risk Model",
    list(MODEL_OPTIONS.keys())
)

model_id = MODEL_OPTIONS[selected_model_name]


# ------------------------------------------------
# Helper Input Functions
# ------------------------------------------------
def opt_float(label, key):
    val = st.text_input(label, key=key)
    if val.strip() == "":
        return None
    try:
        return float(val)
    except:
        st.warning(f"Ignoring invalid input for {label}")
        return None


def opt_int(label, key):
    val = st.text_input(label, key=key)
    if val.strip() == "":
        return None
    try:
        return int(float(val))
    except:
        st.warning(f"Ignoring invalid input for {label}")
        return None


# ------------------------------------------------
# Inputs
# ------------------------------------------------

st.subheader("Basics")
triage_level = opt_int("Triage Level", "risk_triage")
age = opt_int("Age", "risk_age")

st.subheader("Anthropometrics")
height_cm = opt_float("Height (cm)", "risk_height")
weight_kg = opt_float("Weight (kg)", "risk_weight")
bmi = opt_float("BMI", "risk_bmi")

st.subheader("Vitals")
systolic_bp = opt_float("Systolic BP", "risk_sys")
diastolic_bp = opt_float("Diastolic BP", "risk_dia")
heart_rate = opt_float("Heart Rate", "risk_hr")
respiratory_rate = opt_float("Respiratory Rate", "risk_rr")
temperature_c = opt_float("Temperature (°C)", "risk_temp")
spo2 = opt_float("SpO₂", "risk_spo2")

st.subheader("Laboratory Values")
hb_g_dl = opt_float("Hemoglobin", "risk_hb")
wbc_10e9_l = opt_float("WBC", "risk_wbc")
platelets_10e9_l = opt_float("Platelets", "risk_platelets")
sodium_mmol_l = opt_float("Sodium", "risk_na")
potassium_mmol_l = opt_float("Potassium", "risk_k")
bun_mg_dl = opt_float("BUN", "risk_bun")
creatinine_mg_dl = opt_float("Creatinine", "risk_cr")
egfr_ml_min = opt_float("eGFR", "risk_egfr")
glucose_mg_dl = opt_float("Glucose", "risk_glucose")
hba1c_pct = opt_float("HbA1c", "risk_hba1c")
ldl_mg_dl = opt_float("LDL", "risk_ldl")
crp_mg_l = opt_float("CRP", "risk_crp")
troponin_ng_ml = opt_float("Troponin", "risk_trop")
lactate_mmol_l = opt_float("Lactate", "risk_lactate")

st.subheader("Clinical Outcomes (Optional)")
patient_satisfaction_score = opt_float("Patient Satisfaction Score", "risk_ps")
los_days = opt_float("Length of Stay (days)", "risk_los")
total_cost_inr = opt_float("Total Cost (INR)", "risk_cost")
icu_admission = opt_int("ICU Admission (0/1)", "risk_icu")
ventilation = opt_int("Ventilation (0/1)", "risk_vent")
readmit_30d = opt_int("Readmit within 30 days (0/1)", "risk_readmit")
mortality_in_h = opt_int("In-hospital Mortality (0/1)", "risk_mort")


# ------------------------------------------------
# Build Feature Dict (Remove None)
# ------------------------------------------------
features = {
    "triage_level": triage_level,
    "age": age,
    "height_cm": height_cm,
    "weight_kg": weight_kg,
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
    "patient_satisfaction_score": patient_satisfaction_score,
    "los_days": los_days,
    "total_cost_inr": total_cost_inr,
    "icu_admission": icu_admission,
    "ventilation": ventilation,
    "readmit_30d": readmit_30d,
    "mortality_in_h": mortality_in_h,
}

features = {k: v for k, v in features.items() if v is not None}


# ------------------------------------------------
# Predict
# ------------------------------------------------
if st.button("Predict Risk"):

    resp = predict(model_id, {"features": features})
    output = resp.get("output")
    meta = resp.get("meta", {})

    st.subheader("Prediction Result")
    st.success(f"Predicted Risk Category: {output}")

    if "proba" in meta:
        st.write("Probability:", meta["proba"])

    if "imputed_columns" in meta:
        st.info("Imputed missing fields: " + ", ".join(meta["imputed_columns"]))