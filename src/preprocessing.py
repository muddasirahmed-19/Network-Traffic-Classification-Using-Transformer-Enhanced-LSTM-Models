"""
preprocessing.py
-----------------
Cleans raw CICFlowMeter-style traffic CSVs (CIC-Darknet2020 or
ISCXTor2016) and converts them into model-ready NumPy arrays for the
LSTM/Transformer pipeline.

Why this file exists (in plain words):
Network traffic CSVs come straight out of a flow-feature extractor and
are messy in predictable ways: duplicate rows, missing values, "inf"
values (division by zero when a flow has 0 duration, etc.), columns that
don't actually help the model (raw IP addresses, timestamps, flow IDs),
and categorical labels that need to become numbers before a neural
network can use them. This script does all of that cleaning once, then
saves clean train/val/test arrays so the model-training scripts don't
have to repeat this work every run.

Run directly:
    python src/preprocessing.py --dataset darknet2020
    python src/preprocessing.py --dataset iscxtor2016
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Columns that look like features but don't actually carry generalizable
# traffic-behavior information (identifiers / leak the answer / are
# constant). We drop these from both datasets if present.
COLUMNS_TO_DROP = [
    "Flow ID",
    "Src IP",
    "Source IP",
    "Dst IP",
    "Destination IP",
    "Timestamp",
    "Src Port",
    "Source Port",   # high-cardinality identifier-like column; protocol
    "Dst Port",      # info is kept via the 'Protocol' column instead
    "Destination Port",
    "Unnamed: 0",
]

# The raw label column name differs slightly between dataset releases.
# We try these candidates in order and use whichever is present.
LABEL_COLUMN_CANDIDATES = ["Label", "label", "class", "Class"]

# Sequence shape fed into the LSTM/Transformer: each flow's feature
# vector is reshaped into (TIMESTEPS, FEATURES_PER_STEP). We use a
# (1, num_features) "sequence of length 1" shape by default — this keeps
# every original feature visible to the LSTM at a single timestep, which
# is the same shape strategy used by the base paper's embedding+CNN+LSTM
# pipeline (their CNN's job was just to compress 1 timestep of features;
# our LSTM/Transformer can operate directly on the feature vector).
TIMESTEPS = 1


def find_label_column(df: pd.DataFrame) -> str:
    """Return the name of whichever label column exists in this dataframe."""
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No recognizable label column found. Looked for: {LABEL_COLUMN_CANDIDATES}. "
        f"Available columns: {list(df.columns)[:20]}..."
    )


def map_to_binary_tor_label(raw_label: pd.Series) -> pd.Series:
    """
    Collapse the dataset's raw label values down to binary Tor / Non-Tor.

    CIC-Darknet2020's `Label` column contains values like:
        'Tor', 'NonTor', 'VPN', 'NonVPN'
    ISCXTor2016's label column contains values like:
        'TOR', 'nonTOR'  (or similar capitalization)

    We treat anything containing the substring "tor" (case-insensitive)
    as the positive class, EXCEPT values that are explicitly "NonTor"/
    "nonTOR" style negations. VPN/NonVPN rows (only present in
    CIC-Darknet2020) are excluded entirely, since this project's
    classification target is Tor vs Non-Tor specifically.
    """
    raw_label = raw_label.astype(str).str.strip()
    lower = raw_label.str.lower()

    is_tor = lower.str.fullmatch("tor")
    is_nontor = lower.isin(["nontor", "non-tor", "non_tor"])

    # Build a Series of labels; rows that are neither Tor nor NonTor
    # (i.e. plain VPN/NonVPN rows in CIC-Darknet2020) get marked for
    # removal by the caller.
    result = pd.Series(index=raw_label.index, dtype="object")
    result[is_tor] = "Tor"
    result[is_nontor] = "NonTor"
    return result


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements CEP Step 2 (Clean the data):
      - remove duplicate records
      - handle missing values
      - remove infinite / invalid values
      - drop unhelpful identifier columns
    """
    print(f"[clean] Starting shape: {df.shape}")

    # Some CICFlowMeter/ISCXFlowMeter exports have leading/trailing
    # whitespace in column names (a well-known quirk of these tools'
    # CSV output, e.g. ' Source Port' with a leading space). We strip
    # this FIRST, before trying to drop identifier columns below --
    # otherwise a column like ' Source Port' won't match the exact
    # string 'Source Port' in COLUMNS_TO_DROP and will slip through
    # uncleaned.
    df.columns = [c.strip() for c in df.columns]

    # Drop identifier-style columns if present (errors='ignore' since not
    # every dataset has every column).
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns], errors="ignore")

    # Replace inf/-inf with NaN so they get handled by the missing-value
    # step below, instead of silently poisoning the model with huge
    # numbers (this happens when e.g. Flow Bytes/s divides by a duration
    # of 0).
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop exact duplicate rows.
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean] Removed {before - len(df)} duplicate rows")

    # Drop rows with any missing values. (We could impute instead, but
    # for flow-statistics data, rows with missing core stats are usually
    # malformed/truncated flows and are safer to drop than to guess.)
    before = len(df)
    df = df.dropna()
    print(f"[clean] Removed {before - len(df)} rows with missing/invalid values")

    print(f"[clean] Final shape: {df.shape}")
    return df


def select_numeric_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Implements CEP Step 3 (Select important features).

    Keeps every remaining numeric column except the label column(s).
    This is intentionally broad rather than hand-picking a tiny feature
    subset: CICFlowMeter's ~80 flow statistics (duration, packet length
    stats, IAT stats, byte/sec rates, flag counts, etc.) are exactly the
    kind of high-dimensional, partially-redundant feature set that a
    Transformer's attention mechanism is good at learning to weight —
    that's the whole point of using attention instead of hand-picked
    features.
    """
    # Drop any secondary label-like columns (e.g. CIC-Darknet2020's
    # 'Label.1' application-subtype column) so they don't leak into X.
    secondary_label_cols = [c for c in df.columns if c.lower() in ("label.1", "label_1", "subtype")]
    drop_cols = [label_col] + secondary_label_cols

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    feature_df = feature_df.select_dtypes(include=[np.number])

    # Drop zero-variance columns (a column that's the same value for
    # every row carries no information and can destabilize scaling).
    nunique = feature_df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print(f"[features] Dropping {len(constant_cols)} constant columns: {constant_cols}")
        feature_df = feature_df.drop(columns=constant_cols)

    print(f"[features] Selected {feature_df.shape[1]} numeric features")
    return feature_df


def load_traffic_file(file_path: str) -> pd.DataFrame:
    """
    Load a dataset file regardless of whether it's CSV or Parquet.

    Some Kaggle uploaders (e.g. the 'dhoogla' CIC-Darknet2020 mirror)
    distribute their cleaned version of this dataset as a .parquet file
    instead of .csv -- Parquet is a compressed binary format, not plain
    text, so it needs a different pandas function to read it
    (pd.read_parquet instead of pd.read_csv). This function picks the
    right one automatically based on the file's extension, so the rest
    of the pipeline doesn't need to care which format you downloaded.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        print(f"[load] Detected Parquet file, reading with pd.read_parquet()")
        return pd.read_parquet(file_path)
    elif ext in (".csv", ".txt"):
        print(f"[load] Detected CSV file, reading with pd.read_csv()")
        return pd.read_csv(file_path, low_memory=False)
    else:
        raise ValueError(
            f"Unrecognized file extension '{ext}' for '{file_path}'. "
            f"Expected .csv or .parquet."
        )


def preprocess(csv_path: str, output_dir: str, dataset_name: str) -> None:
    """Full pipeline: load -> clean -> label -> encode -> scale -> split -> save."""
    print(f"\n===== Preprocessing {dataset_name} =====")
    df = load_traffic_file(csv_path)

    label_col = find_label_column(df)
    print(f"[label] Using '{label_col}' as the raw label column")

    df["__binary_label__"] = map_to_binary_tor_label(df[label_col])
    before = len(df)
    df = df.dropna(subset=["__binary_label__"])
    print(f"[label] Dropped {before - len(df)} rows that were neither Tor nor Non-Tor "
          f"(e.g. VPN/NonVPN rows not relevant to this binary task)")

    df = clean_dataframe(df)

    X_df = select_numeric_features(df, label_col="__binary_label__")
    # Re-fetch the label column after clean_dataframe() may have changed
    # the row index via dropna()/drop_duplicates().
    y_raw = df.loc[X_df.index, "__binary_label__"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)  # Tor -> 1, NonTor -> 0 (alphabetical)
    print(f"[label] Class mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
    print(f"[label] Class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    X = X_df.values.astype(np.float32)

    # Train / validation / test split: 70% train, 15% val, 15% test.
    # stratify=y keeps the Tor/Non-Tor ratio consistent across splits,
    # which matters because Tor traffic is typically the minority class.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Normalize features using statistics from the TRAINING set only
    # (fitting the scaler on val/test data would leak information from
    # data the model is supposed to never have seen during training).
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Reshape into (samples, timesteps, features) for the LSTM/Transformer.
    num_features = X_train.shape[1]

    def to_sequence(arr):
        return arr.reshape(arr.shape[0], TIMESTEPS, num_features)

    X_train = to_sequence(X_train)
    X_val = to_sequence(X_val)
    X_test = to_sequence(X_test)

    print(f"[shapes] X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    # Save everything the training script needs.
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f"{dataset_name}_processed.npz"),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=np.array(X_df.columns.tolist()),
    )
    joblib.dump(scaler, os.path.join(output_dir, f"{dataset_name}_scaler.joblib"))
    joblib.dump(encoder, os.path.join(output_dir, f"{dataset_name}_label_encoder.joblib"))

    print(f"[done] Saved processed arrays + scaler + encoder to '{output_dir}/'")


DATASET_BASE_NAMES = {
    "darknet2020": "darknet2020",
    "iscxtor2016": "iscxtor2016",
}
SUPPORTED_EXTENSIONS = [".csv", ".parquet"]


def find_dataset_file(raw_dir: str, base_name: str) -> str:
    """
    Look for '<base_name>.csv' or '<base_name>.parquet' inside raw_dir
    (in that order) and return whichever one exists. This means you can
    save the file you downloaded as either format, without renaming it
    to force a specific extension.
    """
    for ext in SUPPORTED_EXTENSIONS:
        candidate = os.path.join(raw_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find '{base_name}.csv' or '{base_name}.parquet' inside '{raw_dir}'. "
        f"Download the dataset (see dataset/dataset_link.txt) and place it there "
        f"as either '{base_name}.csv' or '{base_name}.parquet'."
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess network traffic datasets.")
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_BASE_NAMES.keys()),
        help="Which dataset to preprocess.",
    )
    parser.add_argument("--raw_dir", default="../dataset/raw", help="Folder containing the raw file.")
    parser.add_argument("--output_dir", default="../dataset/processed", help="Folder to save processed arrays.")
    args = parser.parse_args()

    csv_path = find_dataset_file(args.raw_dir, DATASET_BASE_NAMES[args.dataset])
    print(f"[main] Found dataset file: {csv_path}")

    preprocess(csv_path, args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
