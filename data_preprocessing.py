"""
data_preprocessing.py
---------------------
Load NSL-KDD train/test files, clean, encode, and scale them.

Design choices (documented):
- Categorical columns (protocol_type, service, flag) are encoded with a single
  OneHotEncoder(handle_unknown='ignore'). This is the SIMPLER, more robust
  approach for unseen categories in the test set (or in single predictions):
  any category not seen at fit time becomes an all-zero vector instead of
  crashing. No manual 'unknown' bucket needed.
- Numerical columns are standardized with StandardScaler (fit on train).
- The fitted OneHotEncoder, StandardScaler, and final ordered feature-name list
  are persisted as pickle files for reuse by train_model.py and predict.py.

Run directly to generate artifacts:
    python data_preprocessing.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    logger, load_dataset, ensure_dir,
    FEATURE_NAMES, CATEGORICAL_COLS, DROP_COL,
)

# Artifact paths.
ARTIFACT_DIR = "artifacts"
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "feature_names.pkl")

RANDOM_STATE = 42


def _numerical_cols():
    """Numerical feature columns = all features minus categorical and dropped."""
    return [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS and c != DROP_COL]


def preprocess(train_path="data/KDDTrain+.txt", test_path="data/KDDTest+.txt",
               save_artifacts=True):
    """Full preprocessing pipeline.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        Encoded + scaled feature matrices and string label arrays.
    feature_names : list[str]
        Ordered output feature names matching the columns of X_*.
    """
    ensure_dir(ARTIFACT_DIR)

    # --- Load ----------------------------------------------------------------
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    # --- Drop noisy always-zero column --------------------------------------
    for df in (train_df, test_df):
        if DROP_COL in df.columns:
            df.drop(columns=[DROP_COL], inplace=True)
    logger.info("Dropped column '%s'", DROP_COL)

    num_cols = _numerical_cols()
    cat_cols = CATEGORICAL_COLS

    # --- Split features / labels --------------------------------------------
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    X_train_df = train_df.drop(columns=['label'])
    X_test_df = test_df.drop(columns=['label'])

    # --- One-hot encode categoricals (fit on train) -------------------------
    # handle_unknown='ignore' => unseen test categories -> all-zero row.
    # sparse_output kw name differs across sklearn versions; handle both.
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:  # older sklearn
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    ohe.fit(X_train_df[cat_cols])
    train_cat = ohe.transform(X_train_df[cat_cols])
    test_cat = ohe.transform(X_test_df[cat_cols])
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    logger.info("One-hot encoded %d categorical cols -> %d columns",
                len(cat_cols), len(cat_feature_names))

    # --- Scale numericals (fit on train) ------------------------------------
    scaler = StandardScaler()
    scaler.fit(X_train_df[num_cols])
    train_num = scaler.transform(X_train_df[num_cols])
    test_num = scaler.transform(X_test_df[num_cols])
    logger.info("Scaled %d numerical columns", len(num_cols))

    # --- Combine: numerical block first, then one-hot block -----------------
    feature_names = list(num_cols) + cat_feature_names
    X_train = np.hstack([train_num, train_cat]).astype(np.float32)
    X_test = np.hstack([test_num, test_cat]).astype(np.float32)

    logger.info("Final shapes -> X_train %s, X_test %s",
                X_train.shape, X_test.shape)

    # --- Persist artifacts ---------------------------------------------------
    if save_artifacts:
        with open(ENCODER_PATH, "wb") as f:
            pickle.dump({"ohe": ohe, "num_cols": num_cols, "cat_cols": cat_cols}, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        with open(FEATURES_PATH, "wb") as f:
            pickle.dump(feature_names, f)
        logger.info("Saved artifacts to '%s/'", ARTIFACT_DIR)

    return X_train, y_train, X_test, y_test, feature_names


def transform_single(features_dict):
    """Transform one connection (dict of 41 raw features) using saved artifacts.

    Accepts all 41 original feature names; the dropped column is ignored if
    present. Returns a (1, n_features) float32 array aligned to feature_names.
    """
    with open(ENCODER_PATH, "rb") as f:
        enc = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)

    ohe = enc["ohe"]
    num_cols = enc["num_cols"]
    cat_cols = enc["cat_cols"]

    # Build a one-row DataFrame; fill missing numeric features with 0.
    row = {c: features_dict.get(c, 0) for c in (num_cols + cat_cols)}
    df = pd.DataFrame([row])

    # Cast numerics; non-parsable -> 0.
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    num_part = scaler.transform(df[num_cols])
    cat_part = ohe.transform(df[cat_cols])  # unseen categories -> all zeros
    X = np.hstack([num_part, cat_part]).astype(np.float32)
    return X, feature_names


if __name__ == "__main__":
    Xtr, ytr, Xte, yte, names = preprocess()
    logger.info("Preprocessing complete. %d output features.", len(names))
    # Quick class distribution sanity check.
    uniq, cnt = np.unique(ytr, return_counts=True)
    logger.info("Train class distribution: %s", dict(zip(uniq, cnt.tolist())))
