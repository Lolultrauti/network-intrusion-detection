"""
train_model.py
--------------
Train RandomForest + XGBoost on preprocessed NSL-KDD data, evaluate, save best.

Pipeline:
- preprocess() -> X_train, y_train(str), X_test, y_test(str), feature_names
- Labels integer-encoded in CLASS_NAMES order (index i == CLASS_NAMES[i]).
- SMOTE(random_state=42) oversamples the TRAINING data only.
- RandomForest(n_estimators=200, max_depth=15, class_weight='balanced').
- XGBoost(multi:softprob, num_class=5, max_depth=6, lr=0.1, n_estimators=200)
  with per-sample class weights from compute_class_weight('balanced').
- Metrics: accuracy, confusion matrix, per-class P/R/F1, macro-F1, weighted-F1.
- Best model = higher macro-F1; saved as artifacts/best_model.pkl via joblib.
  Only the bare classifier is saved (predict.py applies transform_single first).
- Plots: rf_feature_importance.png, best_model_feature_importance.png,
  shap_summary.png.

Run:
    python train_model.py
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend for script use
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

from utils import (
    logger, ensure_dir, CLASS_NAMES, encode_labels, shap_row_for_class,
)
from data_preprocessing import preprocess, ARTIFACT_DIR

# Optional XGBoost.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed -> training RandomForest only.")

# Optional SHAP.
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("shap not installed -> skipping SHAP summary plot.")

# Optional PyTorch neural network.
try:
    from nn_model import TorchMLPClassifier, HAS_TORCH
except ImportError:
    HAS_TORCH = False
    logger.warning("nn_model/torch unavailable -> skipping neural network.")

RANDOM_STATE = 42
PLOTS_DIR = "plots"
BEST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.pkl")


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    """Print metrics and return macro-F1."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro = f1_score(y_test, y_pred, average="macro")
    weighted = f1_score(y_test, y_pred, average="weighted")

    logger.info("=" * 60)
    logger.info("MODEL: %s", name)
    logger.info("Accuracy        : %.4f", acc)
    logger.info("Macro F1        : %.4f", macro)
    logger.info("Weighted F1     : %.4f", weighted)
    logger.info("Confusion matrix (rows=true, cols=pred):\n%s",
                confusion_matrix(y_test, y_pred))
    logger.info("Classification report:\n%s",
                classification_report(y_test, y_pred, labels=list(range(len(CLASS_NAMES))),
                                      target_names=CLASS_NAMES, zero_division=0))
    return macro


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_importance_bar(importances, feature_names, title, out_path, top=15):
    idx = np.argsort(importances)[::-1][:top]
    plt.figure(figsize=(9, 6))
    plt.barh([feature_names[i] for i in idx][::-1],
             np.asarray(importances)[idx][::-1], color="steelblue")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info("Saved %s", out_path)


def shap_summary(clf, X_test, feature_names, sample_size=100):
    if not HAS_SHAP:
        logger.warning("Skipping SHAP summary (not installed).")
        return
    try:
        rng = np.random.RandomState(RANDOM_STATE)
        n = min(sample_size, X_test.shape[0])
        X_sub = X_test[rng.choice(X_test.shape[0], n, replace=False)]
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_sub)
        plt.figure()
        shap.summary_plot(shap_values, X_sub, feature_names=feature_names,
                          plot_type="bar", class_names=CLASS_NAMES, show=False)
        out = os.path.join(PLOTS_DIR, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", out)
    except Exception as e:
        logger.warning("SHAP summary failed: %s", e)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    ensure_dir(ARTIFACT_DIR)

    # Preprocess (also saves encoders/scaler/feature_names).
    X_train, y_train_str, X_test, y_test_str, feature_names = preprocess()

    # Integer labels in CLASS_NAMES order (index i == CLASS_NAMES[i]).
    y_train = encode_labels(y_train_str)
    y_test = encode_labels(y_test_str)

    # SMOTE oversampling on TRAINING data only.
    logger.info("Applying SMOTE to training data...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    logger.info("After SMOTE: %s -> %s rows", X_train.shape[0], X_res.shape[0])

    results = {}  # name -> (model, macro_f1)

    # --- RandomForest --------------------------------------------------------
    logger.info("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_res, y_res)
    rf_macro = evaluate("RandomForest", rf, X_test, y_test)
    results["RandomForest"] = (rf, rf_macro)
    plot_importance_bar(rf.feature_importances_, feature_names,
                        "RandomForest — Top 15 Feature Importances",
                        os.path.join(PLOTS_DIR, "rf_feature_importance.png"))

    # --- XGBoost -------------------------------------------------------------
    if HAS_XGB:
        logger.info("Training XGBoost...")
        # Per-sample weights from balanced class weights (computed on resampled y).
        classes = np.unique(y_res)
        cw = compute_class_weight("balanced", classes=classes, y=y_res)
        cw_map = {c: w for c, w in zip(classes, cw)}
        sample_weight = np.array([cw_map[y] for y in y_res])

        xgb = XGBClassifier(
            objective="multi:softprob", num_class=len(CLASS_NAMES),
            max_depth=6, learning_rate=0.1, n_estimators=200,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric="mlogloss", tree_method="hist")
        xgb.fit(X_res, y_res, sample_weight=sample_weight)
        xgb_macro = evaluate("XGBoost", xgb, X_test, y_test)
        results["XGBoost"] = (xgb, xgb_macro)

    # --- Neural Network (PyTorch MLP) ---------------------------------------
    if HAS_TORCH:
        logger.info("Training Neural Network (PyTorch MLP)...")
        nn_clf = TorchMLPClassifier(hidden=(128, 64), dropout=0.3,
                                    epochs=30, lr=1e-3, batch_size=512,
                                    random_state=RANDOM_STATE)
        nn_clf.fit(X_res, y_res)
        nn_macro = evaluate("NeuralNetwork", nn_clf, X_test, y_test)
        results["NeuralNetwork"] = (nn_clf, nn_macro)

    # --- Select best by macro-F1 --------------------------------------------
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_macro = results[best_name]
    logger.info("BEST MODEL: %s (macro-F1=%.4f)", best_name, best_macro)

    # Best-model feature importance plot (tree models only).
    if hasattr(best_model, "feature_importances_"):
        plot_importance_bar(
            best_model.feature_importances_, feature_names,
            f"{best_name} (best) — Top 15 Feature Importances",
            os.path.join(PLOTS_DIR, "best_model_feature_importance.png"))
    else:
        logger.info("%s has no feature_importances_ -> skipping that plot.",
                    best_name)

    # Save bare classifier via joblib (predict.py applies transform_single first).
    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.info("Saved best model -> %s", BEST_MODEL_PATH)

    # SHAP summary on best model (TreeExplainer -> tree models only; the helper
    # already catches failures, e.g. for the neural network).
    shap_summary(best_model, X_test, feature_names)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
