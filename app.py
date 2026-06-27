"""
app.py
------
Streamlit dashboard for the NIDS model with three modes (sidebar selector):

  1. Single Prediction   – manual feature entry (original behaviour, unchanged).
  2. Real-time Monitoring – auto-refresh every 5s, reads new lines appended to
                            data/stream.csv and classifies each.
  3. Model Monitoring     – review logged predictions, label ground truth, and
                            view accuracy / precision / recall / F1 / confusion
                            matrix once labels exist.

Every prediction (any mode) is logged via logger.log_prediction, and attack
predictions above a confidence threshold trigger a Slack alert.

Run:
    streamlit run app.py
"""

import os
import csv
import sys
import subprocess
import pandas as pd
import streamlit as st

from utils import FEATURE_NAMES, CATEGORICAL_COLS, DROP_COL, CLASS_NAMES
from predict import predict_connection, BEST_MODEL_PATH
from logger import log_prediction, send_alert, LOG_PATH, LOG_DIR

# Optional auto-refresh component (real-time mode). Degrade gracefully if absent.
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

STREAM_PATH = os.path.join("data", "stream.csv")
CAPTURE_ERR_LOG = os.path.join("logs", "capture.err")

# Alert tuning — kept here so thresholds are easy to change.
ALERT_CONFIDENCE_THRESHOLD = 0.90
ALERT_ON_CLASSES = {"DoS", "Probe", "R2L", "U2R"}  # i.e. anything but Normal


# ---------------------------------------------------------------------------
# Live-capture subprocess control (drives live_capture.py from the UI)
# ---------------------------------------------------------------------------
def list_interfaces():
    """Return [(internal_name, friendly_label)] of capture interfaces, or []."""
    try:  # Windows: friendly names + descriptions
        from scapy.arch.windows import get_windows_if_list
        out = []
        for i in get_windows_if_list():
            name = i.get("name", "")
            desc = i.get("description") or name
            if name:
                out.append((name, f"{desc}  [{name}]"))
        if out:
            return out
    except Exception:
        pass
    try:  # cross-platform fallback
        from scapy.all import get_if_list
        return [(n, n) for n in get_if_list()]
    except Exception:
        return []


def capture_running():
    p = st.session_state.get("capture_proc")
    return p is not None and p.poll() is None


def start_capture(iface, bpf):
    """Launch live_capture.py as a subprocess (inherits this process's privileges)."""
    os.makedirs("logs", exist_ok=True)
    err = open(CAPTURE_ERR_LOG, "w", encoding="utf-8")
    cmd = [sys.executable, "live_capture.py", "--filter", bpf or "ip"]
    if iface:
        cmd += ["--iface", iface]
    st.session_state.capture_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=err)
    st.session_state.capture_attempted = True


def stop_capture():
    p = st.session_state.get("capture_proc")
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=3)
        except Exception:
            p.kill()
    st.session_state.capture_proc = None


def render_capture_controls():
    """Interface picker + start/stop, shown atop the Real-time page."""
    with st.expander("🎛️ Live capture control (scapy)", expanded=True):
        ifaces = list_interfaces()
        if not ifaces:
            st.info(
                "scapy not available. Install it (`pip install scapy`, plus "
                "**Npcap** on Windows) to capture from here — or run "
                "`python live_capture.py` in a separate terminal.")
            return

        labels = [lbl for _, lbl in ifaces]
        sel = st.selectbox("Network interface", labels,
                           index=0, key="capture_iface_sel")
        bpf = st.text_input("BPF filter", value="ip", key="capture_bpf",
                            help="e.g. 'ip', 'tcp', 'tcp port 80'")

        running = capture_running()
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("▶ Start", disabled=running, use_container_width=True):
            iface_name = ifaces[labels.index(sel)][0]
            start_capture(iface_name, bpf)
            st.rerun()
        if c2.button("⏹ Stop", disabled=not running, use_container_width=True):
            stop_capture()
            st.rerun()
        c3.markdown("**🟢 Capturing**" if running else "**⚪ Stopped**")

        st.caption("⚠️ Capture needs admin privileges — launch Streamlit from an "
                   "Administrator terminal (Windows) or with sudo (Linux/macOS), "
                   "else it fails with a permission/Npcap error below.")

        # If the process died, surface its stderr (Npcap missing, perms, etc.).
        if (not running and st.session_state.get("capture_attempted")
                and os.path.exists(CAPTURE_ERR_LOG)):
            err_txt = open(CAPTURE_ERR_LOG, encoding="utf-8").read().strip()
            if err_txt:
                st.error("Capture process stopped. Last error output:")
                st.code(err_txt[-1200:])

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


# ===========================================================================
# Shared helpers
# ===========================================================================
def maybe_alert(features, res):
    """Send a Slack alert if the prediction is a high-confidence attack.

    Threshold/classes are module-level constants so they're easy to tweak.
    """
    if (res["prediction"] in ALERT_ON_CLASSES
            and res["confidence"] >= ALERT_CONFIDENCE_THRESHOLD):
        top = res.get("top_features", [])[:3]
        feats = ", ".join(f"{t['feature']} ({t['direction']})" for t in top) \
            or "n/a"
        msg = (f"🚨 NIDS ALERT: {res['prediction']} attack detected "
               f"(confidence {res['confidence']*100:.1f}%). "
               f"Top features: {feats}")
        send_alert(msg)
        return msg
    return None


def run_and_record(features, explain=True):
    """Predict, log, and (conditionally) alert. Returns (result, alert_msg)."""
    res = predict_connection(features, explain=explain)
    log_prediction(features, res)
    alert_msg = maybe_alert(features, res)
    return res, alert_msg


def render_result(res, alert_msg=None):
    """Render a prediction result block (shared by Single + Real-time modes)."""
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

    if alert_msg:
        st.warning(alert_msg)

    if res["top_features"]:
        st.write("**Top features driving this prediction (SHAP)**")
        st.table([
            {"feature": t["feature"], "effect": t["direction"],
             "shap_value": round(t["shap_value"], 4),
             "value": round(t["value"], 4)}
            for t in res["top_features"]
        ])


def parse_stream_line(line):
    """Parse one CSV stream line (41 raw values) into a feature dict."""
    vals = [v.strip() for v in line.split(",")]
    # Allow either exactly 41 values, or 41 + trailing label/extra (ignored).
    if len(vals) < len(FEATURE_NAMES):
        raise ValueError(
            f"Stream line has {len(vals)} values; expected >= {len(FEATURE_NAMES)}.")
    return dict(zip(FEATURE_NAMES, vals[:len(FEATURE_NAMES)]))


# ===========================================================================
# Sidebar: mode selector
# ===========================================================================
mode = st.sidebar.radio(
    "Mode",
    ["Single Prediction", "Real-time Monitoring", "Model Monitoring"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Logs → `{LOG_PATH}`\nStream → `{STREAM_PATH}`\n"
    f"Slack alerts: {'ON' if os.environ.get('SLACK_WEBHOOK_URL') else 'OFF (set SLACK_WEBHOOK_URL)'}"
)


# ===========================================================================
# MODE 1 — Single Prediction (original behaviour, now logs + alerts)
# ===========================================================================
def page_single():
    st.subheader("Connection features")
    cols = st.columns(3)
    features = {}

    features["protocol_type"] = cols[0].selectbox("protocol_type", PROTOCOLS, index=0)
    features["service"] = cols[1].selectbox("service", COMMON_SERVICES, index=0)
    features["flag"] = cols[2].selectbox("flag", COMMON_FLAGS, index=0)

    numeric_feats = [f for f in FEATURE_NAMES
                     if f not in CATEGORICAL_COLS and f != DROP_COL]
    with st.expander("Numeric features (defaults are fine for a quick test)",
                     expanded=False):
        ncols = st.columns(4)
        for i, feat in enumerate(numeric_feats):
            features[feat] = ncols[i % 4].number_input(
                feat, value=float(DEFAULTS.get(feat, 0)), format="%.4f")

    if st.button("🔍 Analyze connection", type="primary"):
        res, alert_msg = run_and_record(features, explain=True)
        render_result(res, alert_msg)


# ===========================================================================
# MODE 2 — Real-time Monitoring
# ===========================================================================
def page_realtime():
    st.subheader("📡 Real-time Monitoring")
    st.caption("Auto-refreshes every 5s and classifies new lines appended to "
               f"`{STREAM_PATH}` (by live capture or the replay simulator).")

    # In-app live packet capture control.
    render_capture_controls()

    if HAS_AUTOREFRESH:
        st_autorefresh(interval=5000, key="rt_refresh")
    else:
        st.info("Install `streamlit-autorefresh` for true auto-refresh. "
                "Using a manual refresh button for now.")
        st.button("🔄 Refresh now")

    # Session pointer: how many data lines we've already processed.
    if "stream_pos" not in st.session_state:
        st.session_state.stream_pos = 0
    if "rt_last_result" not in st.session_state:
        st.session_state.rt_last_result = None

    if not os.path.exists(STREAM_PATH):
        st.warning(f"⏳ Waiting for `{STREAM_PATH}` … "
                   "append CSV lines (41 NSL-KDD features) to start streaming.")
        return

    # Read all data lines (skip a header row if present).
    with open(STREAM_PATH, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    if raw and raw[0].lower().startswith("duration"):
        raw = raw[1:]  # drop header

    total = len(raw)
    pos = st.session_state.stream_pos

    # Reset pointer if the file shrank/rotated.
    if pos > total:
        pos = 0

    new_lines = raw[pos:]
    if new_lines:
        # Process new lines; show the most recent one in full detail.
        processed = 0
        last_res = last_alert = None
        last_features = None
        for line in new_lines:
            try:
                feats = parse_stream_line(line)
                last_res, last_alert = run_and_record(feats, explain=True)
                last_features = feats
                processed += 1
            except Exception as e:
                st.error(f"Skipped malformed line: {e}")
        st.session_state.stream_pos = total
        if last_res is not None:
            st.session_state.rt_last_result = (last_res, last_alert)
            st.success(f"Processed {processed} new connection(s).")

    # Show the latest result (persisted across refreshes).
    col_a, col_b = st.columns([1, 3])
    col_a.metric("Lines processed", st.session_state.stream_pos)
    col_b.metric("Lines in stream file", total)

    if st.session_state.rt_last_result is not None:
        st.markdown("#### Latest connection")
        res, alert_msg = st.session_state.rt_last_result
        render_result(res, alert_msg)
    else:
        st.info("No connections processed yet — waiting for new lines.")


# ===========================================================================
# MODE 3 — Model Monitoring (ground-truth labeling + metrics)
# ===========================================================================
def page_monitoring():
    st.subheader("📊 Model Monitoring")

    if not os.path.exists(LOG_PATH):
        st.info("No predictions logged yet. Make some predictions first "
                "(Single or Real-time mode).")
        return

    df = pd.read_csv(LOG_PATH, dtype=str).fillna("")
    if df.empty:
        st.info("Prediction log is empty.")
        return

    st.write(f"**{len(df)} logged prediction(s).**")

    # --- Labeling UI -------------------------------------------------------
    st.markdown("#### Label ground truth")
    st.caption("Set the actual class for past predictions, then click Save. "
               "Leave blank to keep unlabeled.")

    # Show a compact editable view: timestamp, prediction, confidence, true_label.
    view_cols = ["timestamp", "prediction", "confidence", "true_label"]
    editable = df[view_cols].copy()

    edited = st.data_editor(
        editable,
        key="label_editor",
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("timestamp", disabled=True),
            "prediction": st.column_config.TextColumn("predicted", disabled=True),
            "confidence": st.column_config.TextColumn("confidence", disabled=True),
            "true_label": st.column_config.SelectboxColumn(
                "true_label (actual)", options=[""] + CLASS_NAMES),
        },
    )

    if st.button("💾 Save labels"):
        # Write edited true_label back into the full dataframe and persist.
        df["true_label"] = edited["true_label"].values
        df.to_csv(LOG_PATH, index=False)
        st.success("Labels saved.")
        st.rerun()

    # --- Metrics on labeled rows ------------------------------------------
    st.markdown("#### Performance on labeled data")
    labeled = df[df["true_label"].isin(CLASS_NAMES)]
    if labeled.empty:
        st.info("No labeled data yet. Start labeling predictions.")
        return

    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report)
    y_true = labeled["true_label"].tolist()
    y_pred = labeled["prediction"].tolist()

    acc = accuracy_score(y_true, y_pred)
    st.metric("Accuracy (labeled subset)", f"{acc*100:.2f}%",
              help=f"Based on {len(labeled)} labeled prediction(s).")

    rep = classification_report(y_true, y_pred, labels=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T.round(3)
    st.write("**Per-class precision / recall / F1**")
    st.dataframe(rep_df, use_container_width=True)

    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES],
                         columns=[f"pred_{c}" for c in CLASS_NAMES])
    st.write("**Confusion matrix**")
    st.dataframe(cm_df, use_container_width=True)


# ===========================================================================
# Router
# ===========================================================================
os.makedirs(LOG_DIR, exist_ok=True)

if mode == "Single Prediction":
    page_single()
elif mode == "Real-time Monitoring":
    page_realtime()
else:
    page_monitoring()
