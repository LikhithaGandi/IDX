import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = "ClosePrice"


def _mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Median Absolute Percentage Error.
    Mirrors the notebook's formula but avoids division-by-zero explosions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _drop_leakage_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Notebook conditionally drops this column when present.
    leakage_cols = [c for c in ["log1p_ClosePrice"] if c in df.columns]
    if leakage_cols:
        return df.drop(columns=leakage_cols)
    return df


def _infer_scale_and_categorical_cols(
    X: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str]]:
    # Numeric columns (pandas' "number" dtype includes ints/floats).
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Notebook logic: a numeric column is "binary" if all non-null values are 0/1.
    binary_cols = []
    for c in num_cols:
        non_null = X[c].dropna()
        if len(non_null) == 0:
            continue
        if non_null.isin([0, 1]).all():
            binary_cols.append(c)

    scale_cols = [c for c in num_cols if c not in binary_cols]
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return scale_cols, categorical_cols, binary_cols


def _one_hot_and_align(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    categorical_cols_train: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Mirrors notebook:
    # - train/test one-hot using categorical columns detected per split
    # - align to train columns with join='left' fill_value=0
    X_train_enc = pd.get_dummies(
        X_train,
        columns=categorical_cols_train,
        drop_first=True,
    )

    categorical_cols_other = X_other.select_dtypes(include=["object"]).columns.tolist()
    X_other_enc = pd.get_dummies(
        X_other,
        columns=categorical_cols_other,
        drop_first=True,
    )

    X_train_enc, X_other_enc = X_train_enc.align(
        X_other_enc,
        join="left",
        axis=1,
        fill_value=0,
    )

    feature_columns = X_train_enc.columns.tolist()
    return X_train_enc, X_other_enc, feature_columns


def train_xgb_and_build_artifacts(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    *,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
    xgb_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Builds an XGBoost regression model + the preprocessing artifacts needed
    to reproduce the notebook's feature engineering at inference time.
    """
    if target_col not in train_df.columns:
        raise ValueError(f"Training data must include target column '{target_col}'.")

    train_df = _drop_leakage_cols(train_df.copy())
    if test_df is not None:
        test_df = _drop_leakage_cols(test_df.copy())

    if test_df is None:
        # Split into train/test (equivalent role to notebook's predefined split).
        X_full = train_df.drop(columns=[target_col])
        y_full = train_df[target_col]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full,
            y_full,
            test_size=test_size,
            random_state=random_state,
        )
        df_for_scaling = X_tr.copy()
    else:
        y_tr = train_df[target_col]
        y_te = test_df[target_col]
        X_tr = train_df.drop(columns=[target_col])
        X_te = test_df.drop(columns=[target_col])
        df_for_scaling = X_tr.copy()

    scale_cols, categorical_cols_train, _binary_cols = _infer_scale_and_categorical_cols(df_for_scaling)

    scaler = StandardScaler(copy=False)
    X_tr_scaled = X_tr.copy()
    if scale_cols:
        # Match notebook behavior: transform to float32.
        X_tr_scaled[scale_cols] = scaler.fit_transform(X_tr_scaled[scale_cols].astype("float32"))

    X_te_scaled = X_te.copy()
    if scale_cols:
        X_te_scaled[scale_cols] = scaler.transform(X_te_scaled[scale_cols].astype("float32"))

    X_tr_enc, X_te_enc, feature_columns = _one_hot_and_align(
        X_tr_scaled,
        X_te_scaled,
        categorical_cols_train=categorical_cols_train,
    )

    # Ensure numeric dtype for xgboost.
    X_tr_enc = X_tr_enc.astype("float32")
    X_te_enc = X_te_enc.astype("float32")

    # Notebook's best model (XGBoost) hyperparameters.
    default_params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=random_state,
    )
    if xgb_params:
        default_params.update(xgb_params)

    import xgboost as xgb  # local import to keep base import light

    model = xgb.XGBRegressor(**default_params)
    model.fit(X_tr_enc, y_tr)

    y_pred = model.predict(X_te_enc)
    metrics = {
        "r2": float(r2_score(y_te, y_pred)),
        "mdape": _mdape(y_te, y_pred),
    }

    artifacts = {
        "model": model,
        "scaler": scaler,
        "scale_cols": scale_cols,
        "categorical_cols_train": categorical_cols_train,
        "feature_columns": feature_columns,
        "target_col": target_col,
        "xgb_params": default_params,
    }
    return artifacts, metrics


def save_artifacts(artifacts: Dict[str, Any], out_path: str = "model.pkl") -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    joblib.dump(artifacts, out_path)


def load_artifacts(model_path: str = "model.pkl") -> Dict[str, Any]:
    return joblib.load(model_path)


def transform_for_inference(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    *,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Transforms a raw input dataframe into the exact feature matrix layout
    expected by the trained model.
    """
    df = df.copy()
    df = _drop_leakage_cols(df)

    # Drop target if user uploaded a dataset that includes it.
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    scale_cols: List[str] = artifacts["scale_cols"]
    categorical_cols_train: List[str] = artifacts["categorical_cols_train"]
    feature_columns: List[str] = artifacts["feature_columns"]

    # Ensure required columns exist.
    for c in scale_cols:
        if c not in df.columns:
            df[c] = 0.0
    for c in categorical_cols_train:
        if c not in df.columns:
            df[c] = "Unknown"

    scaler: StandardScaler = artifacts["scaler"]
    if scale_cols:
        df[scale_cols] = scaler.transform(df[scale_cols].astype("float32"))

    # Apply the same one-hot encoding logic.
    # Notebook one-hot encodes based on detected object columns in the split.
    # For robustness at inference time, we:
    # - force known training categorical columns to object dtype
    # - also encode any additional object-typed columns present in the uploaded data
    for c in categorical_cols_train:
        if c in df.columns:
            df[c] = df[c].astype("object")

    categorical_cols_infer = sorted(set(categorical_cols_train) | set(df.select_dtypes(include=["object"]).columns.tolist()))
    X_enc = pd.get_dummies(df, columns=categorical_cols_infer, drop_first=True)

    # Align to training feature columns (fill missing dummies with 0).
    X_enc = X_enc.reindex(columns=feature_columns, fill_value=0)
    return X_enc.astype("float32")


def predict_xgb(
    df_features: pd.DataFrame,
    artifacts: Dict[str, Any],
    *,
    target_col: str = TARGET_COL,
) -> np.ndarray:
    X = transform_for_inference(df_features, artifacts, target_col=target_col)
    model = artifacts["model"]
    return model.predict(X)

