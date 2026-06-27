"""
stream_simulator.py
-------------------
Replay NSL-KDD connections into data/stream.csv on a timer, so the Streamlit
dashboard's Real-time Monitoring mode has live traffic to classify.

It samples rows from a source file (default data/KDDTest+.txt), strips them to
the 41 features, and appends one line every N seconds. Mix of normal + attack
rows so you see the dashboard flip between states (and fire Slack alerts).

Usage
-----
# default: 1 line every 2s from data/KDDTest+.txt, runs forever
python stream_simulator.py

# faster, fixed count, attacks only
python stream_simulator.py --interval 1 --count 50 --only-attacks

# reset the stream file first (keep header)
python stream_simulator.py --reset

Run this in a SECOND terminal while `streamlit run app.py` is open in Real-time
mode.
"""

import os
import csv
import time
import random
import argparse

from utils import FEATURE_NAMES, map_label

SRC_DEFAULT = os.path.join("data", "KDDTest+.txt")
STREAM_PATH = os.path.join("data", "stream.csv")
RANDOM_STATE = 42


def load_rows(src_path):
    """Load source rows as (feature_csv_str, broad_label) tuples."""
    rows = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(FEATURE_NAMES) + 1:
                continue
            feats = parts[:len(FEATURE_NAMES)]
            label = map_label(parts[len(FEATURE_NAMES)])  # 42nd col = attack_type
            rows.append((",".join(feats), label))
    return rows


def ensure_stream_header():
    """Create data/stream.csv with the feature header if missing."""
    os.makedirs(os.path.dirname(STREAM_PATH), exist_ok=True)
    if not os.path.exists(STREAM_PATH):
        with open(STREAM_PATH, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(FEATURE_NAMES) + "\n")


def reset_stream():
    """Truncate the stream file back to just the header."""
    os.makedirs(os.path.dirname(STREAM_PATH), exist_ok=True)
    with open(STREAM_PATH, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(FEATURE_NAMES) + "\n")
    print(f"Reset {STREAM_PATH} (header only).")


def main():
    ap = argparse.ArgumentParser(description="Replay NSL-KDD rows into a stream file")
    ap.add_argument("--src", default=SRC_DEFAULT, help="Source dataset file")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="Seconds between appended lines (default 2)")
    ap.add_argument("--count", type=int, default=0,
                    help="How many lines to emit (0 = run forever)")
    ap.add_argument("--only-attacks", action="store_true",
                    help="Emit only attack connections (skip Normal)")
    ap.add_argument("--reset", action="store_true",
                    help="Reset the stream file to header-only and exit")
    args = ap.parse_args()

    if args.reset:
        reset_stream()
        return

    if not os.path.exists(args.src):
        raise SystemExit(
            f"Source not found: {args.src}\n"
            "Download NSL-KDD (see README) and place it under data/.")

    random.seed(RANDOM_STATE)
    rows = load_rows(args.src)
    if args.only_attacks:
        rows = [r for r in rows if r[1] != "Normal"]
    if not rows:
        raise SystemExit("No rows to stream after filtering.")

    ensure_stream_header()
    print(f"Streaming from {args.src} -> {STREAM_PATH} "
          f"every {args.interval}s. Ctrl+C to stop.")

    emitted = 0
    try:
        while True:
            feats, label = random.choice(rows)
            with open(STREAM_PATH, "a", newline="", encoding="utf-8") as f:
                f.write(feats + "\n")
            emitted += 1
            print(f"[{emitted:4d}] appended ({label})")
            if args.count and emitted >= args.count:
                print(f"Done: emitted {emitted} lines.")
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nStopped after {emitted} lines.")


if __name__ == "__main__":
    main()
