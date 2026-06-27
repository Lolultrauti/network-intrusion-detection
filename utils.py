"""
utils.py
--------
Helper utilities shared across the NIDS project:
- Column names for the NSL-KDD dataset.
- Mapping of detailed attack labels -> 5 broad classes.
- Dataset loading helper.
- Small plotting / logging helpers.
"""

import os
import logging
import pandas as pd

# ----------------------------------------------------------------------------
# Logging configuration (shared default logger)
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nids")


# ----------------------------------------------------------------------------
# Column names: 41 features + label + difficulty.
# The NSL-KDD .txt files actually contain 43 columns:
#   41 features, 1 label, 1 "difficulty level" integer.
# We name all 43 and drop the difficulty column after loading.
# ----------------------------------------------------------------------------
FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
]

# Full column list as stored in the raw files (features + attack_type + difficulty).
COLUMN_NAMES = FEATURE_NAMES + ['attack_type', 'difficulty']

# Categorical feature columns.
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Column that is always 0 in NSL-KDD -> dropped to reduce noise.
DROP_COL = 'num_outbound_cmds'

# Ordered list of the 5 broad classes (stable order for reports / plots).
CLASS_NAMES = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


# ----------------------------------------------------------------------------
# Detailed attack label -> broad class mapping.
# Anything not listed (and not 'normal') is treated as the closest known
# group; unknown attacks default to 'DoS' is NOT done — instead we keep an
# explicit map and treat truly-unseen labels as 'Normal' fallback only if
# absolutely necessary (logged as a warning).
# ----------------------------------------------------------------------------
ATTACK_MAP = {
    # Normal
    'normal': 'Normal',

    # DoS
    'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS', 'worm': 'DoS',

    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',

    # R2L
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L',
    'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L',
    'sendmail': 'R2L', 'named': 'R2L', 'snmpgetattack': 'R2L', 'snmpguess': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L',

    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'httptunnel': 'U2R',
}


def map_label(raw_label: str) -> str:
    """Map a detailed NSL-KDD label to one of the 5 broad classes.

    Unknown labels fall back to 'Normal' with a warning (rare; keeps pipeline
    robust against dataset variants).
    """
    key = str(raw_label).strip().lower()
    if key in ATTACK_MAP:
        return ATTACK_MAP[key]
    logger.warning("Unknown attack label '%s' -> defaulting to 'Normal'", raw_label)
    return 'Normal'


def encode_labels(label_array):
    """Encode string labels -> integers using CLASS_NAMES order (NOT alphabetical).

    Index i always corresponds to CLASS_NAMES[i], so a model's predict() output
    can be mapped straight back via CLASS_NAMES[idx].
    """
    import numpy as _np
    idx = {name: i for i, name in enumerate(CLASS_NAMES)}
    return _np.array([idx[str(l)] for l in label_array], dtype=int)


def load_dataset(path: str) -> pd.DataFrame:
    """Load an NSL-KDD .txt file (comma-separated, no header).

    Assigns the 43 column names, drops the 'difficulty' column, and maps the
    label to the 5 broad classes. Returns a DataFrame with 41 features + label.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Place the NSL-KDD files in the 'data/' folder "
            "(KDDTrain+.txt, KDDTest+.txt)."
        )

    logger.info("Loading dataset: %s", path)
    df = pd.read_csv(path, header=None)
    # Assign column names: 41 features + attack_type + difficulty.
    df.columns = COLUMN_NAMES

    # Map detailed attack types -> 5 broad classes in a new 'label' column.
    df['label'] = df['attack_type'].apply(map_label)

    # Drop attack_type and difficulty, keeping only the 41 features + label.
    df = df.drop(columns=['attack_type', 'difficulty'])

    logger.info("Loaded %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def shap_row_for_class(shap_values, sample_idx, class_idx, n_features):
    """Extract a 1-D SHAP vector (length n_features) for one sample & one class.

    Robust to the different shapes returned across SHAP / model versions:
    - list of arrays (one per class):        shap_values[class][sample]
    - 3-D ndarray (samples, features, class): shap_values[sample, :, class]
    - 2-D ndarray (samples, features):        shap_values[sample]  (binary/reg)
    """
    import numpy as _np
    if isinstance(shap_values, list):
        arr = _np.asarray(shap_values[class_idx])[sample_idx]
    else:
        arr = _np.asarray(shap_values)
        if arr.ndim == 3:                 # (samples, features, classes)
            arr = arr[sample_idx, :, class_idx]
        elif arr.ndim == 2:               # (samples, features)
            arr = arr[sample_idx]
    return _np.asarray(arr).reshape(-1)[:n_features]
