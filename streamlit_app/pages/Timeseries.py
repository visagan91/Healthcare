import streamlit as st
import numpy as np

# Adjust this import to match YOUR project structure
# If api_client.py is in the same folder as streamlit_app/, this usually works:
from api_client import predict

FEATURE_ORDER = ["temp_c", "spo2", "heart_rate", "respiratory_rate"]

# Adult reference ranges (simple demo-friendly thresholds)
SAFE_RANGES = {
    "temp_c": (36.1, 37.5),            # °C
    "spo2": (95.0, 100.0),             # %
    "heart_rate": (60.0, 100.0),       # bpm
    "respiratory_rate": (12.0, 20.0),  # breaths/min
}

def _fmt_range(lo, hi, unit=""):
    u = f" {unit}".strip()
    return f"{lo:g}–{hi:g}{u}"

def check_vitals_alerts(sequence):
    """
    sequence: list[list[float]] each row: [temp_c, spo2, heart_rate, respiratory_rate]
    returns: list[str] alerts
    """
    alerts = []
    for idx, row in enumerate(sequence, start=1):
        temp_c, spo2, hr, rr = row

        lo, hi = SAFE_RANGES["temp_c"]
        if temp_c < lo or temp_c > hi:
            alerts.append(f"Timestep {idx}: Temperature {temp_c:.2f}°C (expected {_fmt_range(lo, hi, '°C')})")

        lo, hi = SAFE_RANGES["spo2"]
        if spo2 < lo or spo2 > hi:
            alerts.append(f"Timestep {idx}: SpO₂ {spo2:.0f}% (expected {_fmt_range(lo, hi, '%')})")

        lo, hi = SAFE_RANGES["heart_rate"]
        if hr < lo or hr > hi:
            alerts.append(f"Timestep {idx}: Heart Rate {hr:.0f} bpm (expected {_fmt_range(lo, hi, 'bpm')})")

        lo, hi = SAFE_RANGES["respiratory_rate"]
        if rr < lo or rr > hi:
            alerts.append(f"Timestep {idx}: Respiratory Rate {rr:.0f}/min (expected {_fmt_range(lo, hi, '/min')})")

    return alerts

def build_display_table(sequence):
    """
    Returns (data, flagged_mask) for rendering.
    flagged_mask is same shape, True where value is outside safe range.
    """
    arr = np.array(sequence, dtype=float)
    mask = np.zeros_like(arr, dtype=bool)

    # Column order matches FEATURE_ORDER
    for j, col in enumerate(FEATURE_ORDER):
        lo, hi = SAFE_RANGES[col]
        mask[:, j] = (arr[:, j] < lo) | (arr[:, j] > hi)

    return arr, mask

st.set_page_config(page_title="Time-series (RNN / LSTM)", layout="centered")

st.title("📉 Time-series (RNN / LSTM)")
st.caption(
    "Enter vitals over time (each timestep = Temp °C, SpO₂ %, Heart Rate bpm, Respiratory Rate /min). "
    "We will create the model input sequence automatically."
)

# -------------------------
# Model selector
# -------------------------
model_id = st.selectbox(
    "Choose model",
    ["los_rnn_v1", "los_lstm_v1"],
    index=0,
    help="RNN and LSTM are sequence models. Both expect a sequence of timesteps with 4 numeric features.",
)

with st.expander("Reference ranges used for warnings (adult, demo thresholds)", expanded=False):
    st.write(
        {
            "Temperature (temp_c)": _fmt_range(*SAFE_RANGES["temp_c"], "°C"),
            "SpO₂ (spo2)": _fmt_range(*SAFE_RANGES["spo2"], "%"),
            "Heart Rate (heart_rate)": _fmt_range(*SAFE_RANGES["heart_rate"], "bpm"),
            "Respiratory Rate (respiratory_rate)": _fmt_range(*SAFE_RANGES["respiratory_rate"], "/min"),
        }
    )

# Keep timesteps in session
if "ts_rows" not in st.session_state:
    st.session_state.ts_rows = []

colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.subheader("Add a timestep")
    temp_c = st.number_input("Temperature (°C) — temp_c", value=37.0, step=0.1, format="%.2f")
    spo2 = st.number_input("SpO₂ (%) — spo2", value=95.0, step=1.0, format="%.0f")
    heart_rate = st.number_input("Heart Rate (bpm) — heart_rate", value=85.0, step=1.0, format="%.0f")
    respiratory_rate = st.number_input("Respiratory Rate (/min) — respiratory_rate", value=18.0, step=1.0, format="%.0f")

    add_btn = st.button("➕ Add timestep", use_container_width=True)
    if add_btn:
        st.session_state.ts_rows.append([float(temp_c), float(spo2), float(heart_rate), float(respiratory_rate)])
        st.success(f"Added timestep #{len(st.session_state.ts_rows)}")

with colB:
    st.subheader("Your sequence")

    if len(st.session_state.ts_rows) == 0:
        st.info("No timesteps added yet. Add at least 1 timestep.")
    else:
        arr, mask = build_display_table(st.session_state.ts_rows)

        st.write(f"Timesteps: **{len(st.session_state.ts_rows)}**")
        st.dataframe(
            arr,
            use_container_width=True,
            column_config={
                0: st.column_config.NumberColumn("temp_c (°C)"),
                1: st.column_config.NumberColumn("spo2 (%)"),
                2: st.column_config.NumberColumn("heart_rate (bpm)"),
                3: st.column_config.NumberColumn("respiratory_rate (/min)"),
            },
        )

        # Quick warning summary in the right panel
        alerts_preview = check_vitals_alerts(st.session_state.ts_rows)
        if alerts_preview:
            st.warning(f"⚠ {len(alerts_preview)} alert(s) detected in current sequence.")
            # Show only first few to keep UI clean
            for a in alerts_preview[:5]:
                st.write("-", a)
            if len(alerts_preview) > 5:
                st.write(f"... and {len(alerts_preview) - 5} more.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ Remove last", use_container_width=True, disabled=len(st.session_state.ts_rows) == 0):
            st.session_state.ts_rows.pop()
            st.rerun()
    with col2:
        if st.button("🗑️ Clear all", use_container_width=True, disabled=len(st.session_state.ts_rows) == 0):
            st.session_state.ts_rows = []
            st.rerun()

st.divider()

# -------------------------
# Prediction
# -------------------------
st.subheader("Run prediction")

if st.button("📌 Predict", type="primary", use_container_width=True):
    if len(st.session_state.ts_rows) == 0:
        st.error("Please add at least 1 timestep before predicting.")
    else:
        sequence = st.session_state.ts_rows  # shape [T,4]

        # Show warnings (does not block prediction)
        alerts = check_vitals_alerts(sequence)
        if alerts:
            st.warning("⚠ Vitals outside reference ranges (prediction will still run):")
            for a in alerts:
                st.write("-", a)

        try:
            resp = predict(model_id, {"sequence": sequence})

            # ---- User-friendly output ----
            st.success("Prediction complete")

            # Expecting resp like: {"model_id":..., "output":..., "meta":...}
            if isinstance(resp, dict):
                pred_val = resp.get("output", resp)
                meta = resp.get("meta", {}) if isinstance(resp.get("meta", {}), dict) else {}
            else:
                pred_val = resp
                meta = {}

            # Show key info nicely
            st.write("**Model:**", model_id)
            st.write("**Timesteps sent:**", len(sequence))
            st.write("**Features per timestep:**", 4)
            st.write("**Feature order:**", ", ".join(FEATURE_ORDER))

            # If output is LOS days, show as days with rounding
            if isinstance(pred_val, (int, float)):
                st.metric("Predicted Length of Stay (LOS)", f"{float(pred_val):.2f} days")
            else:
                st.write("### Prediction")
                st.write(pred_val)

            # Optional: show technical meta in an expander
            with st.expander("Technical details (optional)", expanded=False):
                if meta:
                    st.json(meta)
                else:
                    st.write("No meta returned.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")