"""
logger.py
---------
Prediction logging + Slack alerting for the NIDS dashboard.

- log_prediction(features_dict, result_dict): append one row to
  logs/predictions.csv (timestamp, prediction, confidence, true_label, then all
  41 feature values). The header is written once; the logs/ dir is auto-created.
- send_alert(message): POST a message to a Slack incoming webhook. The URL comes
  from the SLACK_WEBHOOK_URL environment variable (falls back to a constant).
  Network/JSON errors are swallowed and logged, never raised.

Uses only the standard library for the webhook (urllib) so no extra dependency
is required.
"""

import os
import csv
import json
import logging
import urllib.request
from datetime import datetime

from utils import FEATURE_NAMES

logger = logging.getLogger("nids.logger")

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "predictions.csv")

# Column layout of the log file.
LOG_FIELDS = ["timestamp", "prediction", "confidence", "true_label"] + FEATURE_NAMES

# Slack webhook: prefer env var; optional hard-coded fallback (leave empty).
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def _ensure_log_file():
    """Create logs/ and write the CSV header if the file doesn't exist yet."""
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_FIELDS)


def log_prediction(features_dict, result_dict):
    """Append a single prediction (+ its features) to logs/predictions.csv.

    Missing feature keys default to empty. 'true_label' is written empty and can
    be filled later in the Model Monitoring page.
    """
    try:
        _ensure_log_file()
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prediction": result_dict.get("prediction", ""),
            "confidence": round(float(result_dict.get("confidence", 0.0)), 6),
            "true_label": "",
        }
        for feat in FEATURE_NAMES:
            row[feat] = features_dict.get(feat, "")
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
    except Exception as e:  # logging must never crash the app
        logger.warning("log_prediction failed: %s", e)


def send_alert(message):
    """Send `message` to the configured Slack webhook. Returns True on success.

    Fails gracefully (returns False, logs a warning) if no URL is set or the
    request errors.
    """
    url = SLACK_WEBHOOK_URL
    if not url:
        logger.info("No SLACK_WEBHOOK_URL set -> alert skipped: %s", message)
        return False
    try:
        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            ok = 200 <= resp.status < 300
        if not ok:
            logger.warning("Slack alert non-2xx status: %s", resp.status)
        return ok
    except Exception as e:
        logger.warning("send_alert failed: %s", e)
        return False
