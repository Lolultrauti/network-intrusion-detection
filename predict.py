"""
predict.py
----------
Single-connection prediction with the trained NIDS model + optional SHAP
explanation of which features drove the decision.

Usage
-----
python predict.py --sample '{"duration":0,"protocol_type":"tcp","service":"http","flag":"SF","src_bytes":215,"dst_bytes":45076, ...}'
python predict.py --file sample.json
python predict.py --csv "0,tcp,http,SF,215,45076,0,0,..."   # 41 raw values

Unseen categories in service/flag/protocol_type are handled by the saved
OneHotEncoder(handle_unknown='ignore') -> all-zero block instead of an error.
"""

import os
import json
import argparse
import numpy as np
import joblib

from utils import logger, FEATURE_NAMES, DROP_COL, CLASS_NAMES, shap_row_for_class
from data_preprocessing import transform_single, ARTIFACT_DIR

BEST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.pkl")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Cached globals to avoid reloading model / rebuilding explainer per call.
_MODEL = None
_EXPLAINER = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(BEST_MODEL_PATH)
    return _MODEL


def _get_explainer(model):
    global _EXPLAINER
    if _EXPLAINER is None and HAS_SHAP:
        _EXPLAINER = shap.TreeExplainer(model)
    return _EXPLAINER


def predict_connection(features_dict, explain=False):
    """Predict the class of a single network connection.

    Parameters
    ----------
    features_dict : dict
        Keys = the 41 raw feature names (dropped 'num_outbound_cmds' ignored if
        present). Missing numeric keys default to 0.
    explain : bool
        If True and SHAP available, include the top-5 features driving the
        prediction.

    Returns
    -------
    dict: prediction, confidence, probabilities, top_features.
    """
    model = _get_model()
    X, feature_names = transform_single(features_dict)

    proba = model.predict_proba(X)[0]
    pred_idx = int(model.predict(X)[0])
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(np.max(proba))

    # proba columns are ordered by model.classes_ (0..4 == CLASS_NAMES order).
    probabilities = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}

    result = {
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_features": [],
    }

    if explain:
        result["top_features"] = _explain(model, X, feature_names, pred_idx)

    return result


def _explain(model, X, feature_names, pred_idx, top_n=5):
    """Return top_n feature contributions for the predicted class."""
    if not HAS_SHAP:
        logger.warning("SHAP not installed -> no explanation.")
        return []
    try:
        explainer = _get_explainer(model)
        sv = explainer.shap_values(X)
        sv_row = shap_row_for_class(sv, 0, pred_idx, len(feature_names))

        order = np.argsort(np.abs(sv_row))[::-1][:top_n]
        top = []
        for i in order:
            top.append({
                "feature": feature_names[i],
                "shap_value": float(sv_row[i]),
                "direction": "increases" if sv_row[i] > 0 else "decreases",
                "value": float(X[0, i]),
            })
        return top
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)
        return []


def _csv_to_dict(csv_line):
    """Parse a 41-value CSV line into a feature dict (FEATURE_NAMES order)."""
    vals = [v.strip() for v in csv_line.split(",")]
    if len(vals) != len(FEATURE_NAMES):
        raise ValueError(
            f"CSV line has {len(vals)} values; expected {len(FEATURE_NAMES)}.")
    return dict(zip(FEATURE_NAMES, vals))


def _print_result(res):
    print("\n" + "=" * 50)
    print(f"  PREDICTION : {res['prediction']}")
    print(f"  CONFIDENCE : {res['confidence']:.4f}")
    print("=" * 50)
    print("  Class probabilities:")
    for cls, p in sorted(res["probabilities"].items(), key=lambda x: -x[1]):
        print(f"    {cls:<8} {p:.4f}")
    if res["top_features"]:
        print("\n  Top features driving this prediction:")
        for t in res["top_features"]:
            print(f"    {t['feature']:<28} {t['direction']:<9} "
                  f"(shap={t['shap_value']:+.4f}, value={t['value']:.3f})")
    print()


def main():
    ap = argparse.ArgumentParser(description="NIDS single-connection prediction")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", help="Inline JSON string of features")
    g.add_argument("--file", help="Path to a JSON file of features")
    g.add_argument("--csv", help="A single 41-value comma-separated line")
    ap.add_argument("--explain", action="store_true",
                    help="Include SHAP explanation of the prediction")
    ap.add_argument("--json", action="store_true",
                    help="Print result as formatted JSON instead of a table")
    args = ap.parse_args()

    if args.sample:
        features = json.loads(args.sample)
    elif args.file:
        with open(args.file) as f:
            features = json.load(f)
    else:
        features = _csv_to_dict(args.csv)

    features.pop(DROP_COL, None)  # ignore dropped column if supplied

    res = predict_connection(features, explain=args.explain)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        _print_result(res)


if __name__ == "__main__":
    main()
