"""
app.py
------
Optional Streamlit dashboard for the NIDS model.

Run:
    streamlit run app.py

Lets you enter a connection's features (with sensible defaults), runs the
trained model, and shows the predicted class, confidence, class probabilities,
and the top SHAP features driving the decision.
"""

import os
import streamlit as st

from utils import FEATURE_NAMES, CATEGORICAL_COLS, DROP_COL, CLASS_NAMES
from predict import predict_connection, BEST_MODEL_PATH

st.set_page_config(page_title="AI Network Intrusion Detection", page_icon="🛡️",
                   layout="wide")

st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.caption("NSL-KDD · classifies a connection as Normal, DoS, Probe, R2L, or U2R")

if not os.path.exists(BEST_MODEL_PATH):
    st.error(f"Model not found at {BEST_MODEL_PATH}. "
             "Run `python train_model.py` first.")
    st.stop()

# Default example connection (a benign HTTP request).
DEFAULTS = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
    "src_bytes": 215, "dst_bytes": 45076, "logged_in": 1, "count": 1,
    "srv_count": 1, "same_srv_rate": 1.0, "dst_host_count": 9,
    "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0,
}

PROTOCOLS = ["tcp", "udp", "icmp"]
COMMON_SERVICES = ["http", "private", "domain_u", "smtp", "ftp_data", "eco_i",
                   "other", "telnet", "finger", "ftp", "ssh"]
COMMON_FLAGS = ["SF", "S0", "REJ", "RSTR", "RSTO", "SH", "S1", "S2", "S3", "OTH"]

st.subheader("Connection features")
cols = st.columns(3)
features = {}

# Categorical inputs.
features["protocol_type"] = cols[0].selectbox("protocol_type", PROTOCOLS,
                                              index=0)
features["service"] = cols[1].selectbox("service", COMMON_SERVICES, index=0)
features["flag"] = cols[2].selectbox("flag", COMMON_FLAGS, index=0)

# Numeric inputs for the rest (default 0, or DEFAULTS where given).
numeric_feats = [f for f in FEATURE_NAMES
                 if f not in CATEGORICAL_COLS and f != DROP_COL]
with st.expander("Numeric features (defaults are fine for a quick test)",
                 expanded=False):
    ncols = st.columns(4)
    for i, feat in enumerate(numeric_feats):
        features[feat] = ncols[i % 4].number_input(
            feat, value=float(DEFAULTS.get(feat, 0)), format="%.4f")

if st.button("🔍 Analyze connection", type="primary"):
    res = predict_connection(features, explain=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        label = res["prediction"]
        if label == "Normal":
            st.success(f"### ✅ {label}")
        else:
            st.error(f"### 🚨 {label} attack")
        st.metric("Confidence", f"{res['confidence']*100:.2f}%")
    with c2:
        st.write("**Class probabilities**")
        st.bar_chart({k: res["probabilities"][k] for k in CLASS_NAMES})

    if res["top_features"]:
        st.write("**Top features driving this prediction (SHAP)**")
        st.table([
            {"feature": t["feature"], "effect": t["direction"],
             "shap_value": round(t["shap_value"], 4),
             "value": round(t["value"], 4)}
            for t in res["top_features"]
        ])
