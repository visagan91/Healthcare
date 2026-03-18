import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from api_client import predict
from ui.shell import app_shell
from ui.common import set_page, analyst_toggle, show_advanced

ok, models, analyst_mode = app_shell("Pattern Association", "Discovers and displays statistically significant medical co-occurrence patterns to support clinical insight, comorbidity analysis, and operational planning.")
if not ok:
    st.stop()

set_page()
st.title("Medical Association Rules")
analyst_mode = analyst_toggle()

model_id = "assoc_rules_v1"

st.caption(
    "Tip: this demo rule set uses synthetic *tokens* like `MED_aspirin`, `COND_diabetes`, `DX_R50.9` "
    "(so plain terms like `cough` / `paracetamol` may not exist in the vocab yet)."
)

col_a, col_b = st.columns([3, 1])
with col_a:
    items_text = st.text_input(
        "Items (comma-separated)",
        "diabetes, aspirin, metformin",
        help="You can enter full tokens (e.g., MED_aspirin) or shorthand (e.g., aspirin, diabetes, R50.9).",
    )
with col_b:
    show_vocab = st.button("Show available items")

if show_vocab:
    resp = predict(model_id, {"items": [], "return_vocab": True})
    meta = resp.get("meta") or {}
    vocab = meta.get("vocab") or []
    st.subheader(f"Available items ({len(vocab)})")
    st.dataframe(pd.DataFrame({"item": vocab}), use_container_width=True, hide_index=True)

st.divider()

if st.button("Find associations"):
    items = [x.strip() for x in items_text.split(",") if x.strip()]
    resp = predict(model_id, {"items": items})
    out = resp.get("output")
    meta = resp.get("meta") or {}

    # Helpful diagnostics
    unknown = meta.get("unknown_items") or []
    mapped = meta.get("mapped_items") or []
    if mapped:
        st.info("Matched inputs: " + ", ".join(mapped))
    if unknown:
        st.warning("Not found in vocab: " + ", ".join(map(str, unknown)))

    st.subheader("Matched rules")
    if isinstance(out, list) and out:
        df = pd.json_normalize(out) if isinstance(out[0], dict) else pd.DataFrame({"rule": out})
        # reorder key fields first when available
        preferred = [c for c in ["antecedents_list", "consequents_list", "support", "confidence", "lift"] if c in df.columns]
        rest = [c for c in df.columns if c not in preferred]
        df = df[preferred + rest]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.write(out)

    if analyst_mode:
        show_advanced(resp)