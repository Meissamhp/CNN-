import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = "data/processed/unified_dataset.csv"


def add_time_features(df):
    df = df.copy()

    if "arrival_sec" in df.columns:
        df["hour"] = (df["arrival_sec"] // 3600) % 24
        df["minute"] = (df["arrival_sec"] % 3600) // 60

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60.0)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60.0)

    return df


def encode_columns(df):
    df = df.copy()

    cat_cols = []
    for col in ["route_id", "route_short_name", "trip_id", "stop_id", "scenario"]:
        if col in df.columns:
            cat_cols.append(col)

    for col in cat_cols:
        df[col] = df[col].astype("category").cat.codes

    return df


def build_sequences(features, targets, seq_len=12):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)


def build_sequence_meta(df, seq_len=12):
    meta_rows = []
    for i in range(len(df) - seq_len):
        meta_rows.append(df.iloc[i + seq_len].copy())
    return pd.DataFrame(meta_rows).reset_index(drop=True)


def get_unified_data(seq_len=12, test_size=0.2, val_size=0.1, random_state=42, return_test_meta=False):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if "travel_time" not in df.columns:
        raise ValueError("'travel_time' column not found in dataset.")

    original_df = df.copy()

    df = add_time_features(df)
    df = encode_columns(df)

    drop_cols = []
    if "arrival_time" in df.columns:
        drop_cols.append("arrival_time")

    numeric_df = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).copy()

    feature_cols = [c for c in numeric_df.columns if c != "travel_time"]
    target_col = "travel_time"

    X_raw = numeric_df[feature_cols].values
    y_raw = numeric_df[target_col].values.reshape(-1, 1)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw).reshape(-1)

    X_seq, y_seq = build_sequences(X_scaled, y_scaled, seq_len=seq_len)
    meta_seq = build_sequence_meta(original_df, seq_len=seq_len)

    n_total = len(X_seq)
    n_test = int(n_total * test_size)
    n_trainval = n_total - n_test
    n_val = int(n_trainval * (val_size / (1 - test_size)))
    n_train = n_trainval - n_val

    X_train = X_seq[:n_train]
    y_train = y_seq[:n_train]

    X_val = X_seq[n_train:n_train + n_val]
    y_val = y_seq[n_train:n_train + n_val]

    X_test = X_seq[n_train + n_val:]
    y_test = y_seq[n_train + n_val:]

    test_meta = meta_seq.iloc[n_train + n_val:].reset_index(drop=True)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "seq_len": seq_len,
        "num_features": len(feature_cols),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "raw_shape": df.shape,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }

    if return_test_meta:
        return X_train, y_train, X_val, y_val, X_test, y_test, meta, test_meta

    return X_train, y_train, X_val, y_val, X_test, y_test, meta


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, meta, test_meta = get_unified_data(return_test_meta=True)

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("Number of features:", meta["num_features"])
    print("Feature columns:", meta["feature_cols"])
    print("Test meta shape:", test_meta.shape)
    print(test_meta.head())