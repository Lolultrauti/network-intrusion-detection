# AI-Powered Network Intrusion Detection System

A machine-learning Network Intrusion Detection System (NIDS) built on the
**NSL-KDD** dataset. It classifies each network connection as one of five
classes — **Normal, DoS, Probe, R2L, U2R** — and explains individual predictions
with SHAP.

## Features

- Clean preprocessing pipeline (one-hot encoding with robust unseen-category
  handling + standard scaling), artifacts persisted for reuse.
- Class-imbalance handling via **SMOTE** oversampling + **balanced class
  weights**.
- Three models trained and compared: **RandomForest**, **XGBoost**, and a
  **PyTorch neural network (MLP)**; the best one (by macro-F1 on the test set)
  is saved. The neural net (`nn_model.py`) is wrapped in a scikit-learn-style
  interface (`fit`/`predict`/`predict_proba`) so it plugs into the same
  comparison and prediction code.
- Full evaluation: accuracy, confusion matrix, per-class precision/recall/F1,
  macro-F1, weighted-F1.
- Explainability: feature-importance plots + **SHAP** global summary and
  per-sample waterfall plots for a false positive and a false negative.
- `predict.py` for single-connection predictions with top-feature explanations.
- Optional **Streamlit** dashboard (`app.py`).

## Dataset

NSL-KDD — an improved version of the KDD Cup 99 dataset.
Source: <https://www.unb.ca/cic/datasets/nsl.html>

The data files are comma-separated with no header (41 features + label +
difficulty level). The 5-class grouping of detailed attack labels is defined in
`utils.py` (`ATTACK_MAP`).

## Project structure

```
data/                  # place KDDTrain+.txt and KDDTest+.txt here
artifacts/             # generated: encoders, scaler, feature names, models
plots/                 # generated: feature importance + SHAP plots
utils.py               # column names, label mapping, loaders
data_preprocessing.py  # load/clean/encode/scale; saves artifacts
nn_model.py            # PyTorch MLP (deep learning) with sklearn-style API
train_model.py         # train RF + XGB + NN, evaluate, save best, SHAP plots
predict.py             # single-connection prediction + explanation (CLI)
logger.py              # prediction logging + Slack alerting
live_capture.py        # REAL live traffic: scapy sniffer -> stream.csv
stream_simulator.py    # replay recorded NSL-KDD rows -> stream.csv (demo)
app.py                 # Streamlit dashboard (single / real-time / monitoring)
requirements.txt
README.md
```

## How to run

### 1. Place the dataset
The dataset is **not** committed to this repo (kept out via `.gitignore`).
Download NSL-KDD from <https://www.unb.ca/cic/datasets/nsl.html> (or Kaggle:
`nsl-kdd`) and place the files in a `data/` folder:
```
data/KDDTrain+.txt
data/KDDTest+.txt
```
> Note: a pre-trained model is already committed under `artifacts/`, so you can
> run the dashboard and `predict.py` **without** the dataset. You only need the
> data to re-run `train_model.py`.

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Preprocess + train
Preprocessing runs automatically inside training, but you can run it standalone:
```bash
python data_preprocessing.py   # optional; generates artifacts/
python train_model.py          # trains, evaluates, saves best_model.pkl + plots
```

### 4. Predict a single connection
```bash
python predict.py --sample '{"duration":0,"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":215,"dst_bytes":45076,"logged_in":1,"count":1,"srv_count":1,"same_srv_rate":1.0,"dst_host_count":9,"dst_host_srv_count":9,"dst_host_same_srv_rate":1.0}' --explain
```
Or from a JSON file / raw CSV line:
```bash
python predict.py --file sample.json --explain
python predict.py --csv "0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,9,9,1,0,0.11,0,0,0,0"
```

CLI flags:
- `--explain` — include top-5 SHAP feature contributions (off by default).
- `--json` — print the result as formatted JSON instead of a table.

Any feature key you omit defaults to `0`, so a partial dict is fine for quick
tests.

### 5. (Optional) Train the neural network too
The PyTorch MLP is optional and needs an extra dependency:
```bash
pip install -r requirements-dev.txt
# then re-run training; it will compare RF, XGBoost and the NN
python train_model.py
```

## Dashboard

Run locally:
```bash
streamlit run app.py
```

### Real-time monitoring
Open the app in **Real-time Monitoring** mode. It auto-refreshes every 5s,
classifies each new line appended to `data/stream.csv`, logs to
`logs/predictions.csv`, and (if `SLACK_WEBHOOK_URL` is set) alerts on
high-confidence attacks. Two ways to feed it:

**A) Real live traffic — `live_capture.py` (scapy packet sniffer)**
Sniffs your actual network interface, assembles connections, computes the
traffic features, and appends them live. Two ways to launch it:

*From the dashboard:* in Real-time Monitoring mode, use the **🎛️ Live capture
control** panel — pick an interface, set a BPF filter, and Start/Stop. (Launch
Streamlit from an Admin terminal so the capture subprocess has privileges.)

*From a terminal:*
```bash
pip install scapy            # + install Npcap (https://npcap.com) on Windows
# run in an Administrator terminal (Windows) or with sudo (Linux/macOS):
python live_capture.py --iface "Wi-Fi"
```
> **Honest limitation:** only the ~20 *traffic* features (bytes, durations,
> per-host/per-service counts and error rates) are derivable from packet
> headers. The ~21 *content* features (`hot`, `num_failed_logins`, `logged_in`,
> `su_attempted`, ...) need payload/host inspection and are **zero-filled**.
> Practical effect: DoS and Probe attacks stay detectable; R2L/U2R (content-
> based) mostly read as Normal. This is a fundamental limit of a KDD'99-era
> model on live packets, not a bug.

**B) Replay recorded data — `stream_simulator.py`**
No capture privileges needed. Replays rows from the **recorded** NSL-KDD test
file (not live traffic) on a timer — useful for demos and for triggering alerts:
```bash
python stream_simulator.py --interval 2            # 1 line every 2s
python stream_simulator.py --only-attacks          # attacks only (fires alerts)
python stream_simulator.py --reset                 # clear stream (header only)
```

### Deploy to Streamlit Community Cloud (free, public URL)
1. Push this repo to GitHub (already done if you cloned from there).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → pick this repo, branch `main`, main file `app.py`.
4. Deploy. Streamlit installs `requirements.txt` and serves the dashboard at
   `https://<your-app>.streamlit.app`.

The committed `artifacts/best_model.pkl` (+ encoder/scaler) means the app works
immediately on the cloud — no training or dataset needed at runtime.

## Design notes

- **Unseen categories** (a `service`/`flag` value present in test but not train,
  or in a live prediction) are handled by `OneHotEncoder(handle_unknown='ignore')`
  — they map to an all-zero block instead of raising. This is the simplest robust
  approach and needs no manual `unknown` bucket.
- The `num_outbound_cmds` column is always 0 in NSL-KDD and is dropped.
- All random seeds are fixed to **42** for reproducibility.
- Models are wrapped in an `imblearn` Pipeline (`SMOTE -> classifier`) so SMOTE
  only ever sees training data.
```
```
