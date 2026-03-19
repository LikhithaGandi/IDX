import os

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any

from model_utils import TARGET_COL, load_artifacts, predict_xgb, save_artifacts, train_xgb_and_build_artifacts


APP_TITLE = "California ClosePrice Predictor (XGBoost)"
MODEL_PATH = "model.pkl"


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Please upload a .csv or .parquet file.")


@st.cache_resource
def _try_load_model(model_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(model_path):
        return None
    return load_artifacts(model_path)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Uses the same core preprocessing steps as your modeling notebook, and predicts `ClosePrice` with XGBoost.")

    artifacts = _try_load_model(MODEL_PATH)

    mode = st.sidebar.radio(
        "Mode",
        ["Predict (needs model.pkl)", "Train (creates model.pkl in this folder)"],
        index=0 if artifacts is not None else 1,
    )

    if mode == "Predict (needs model.pkl)":
        if artifacts is None:
            st.error("No `model.pkl` found in the current folder. Switch to Training mode to create it.")
            st.stop()

        uploaded = st.file_uploader("Upload a file with feature columns (.csv or .parquet)", type=["csv", "parquet"])
        if not uploaded:
            st.info(f"Expected target column name: `{TARGET_COL}` (if present, it will be ignored for prediction).")
            return

        df = _read_uploaded_file(uploaded)

        with st.spinner("Running predictions..."):
            # If the uploaded file includes ClosePrice, we ignore it automatically in transform_for_inference.
            preds = predict_xgb(df, artifacts, target_col=TARGET_COL)

        out = pd.DataFrame({"prediction_ClosePrice": preds})
        st.subheader("Predictions")
        st.dataframe(out, use_container_width=True)
        st.write(
            {
                "pred_min": float(np.min(preds)),
                "pred_max": float(np.max(preds)),
                "pred_mean": float(np.mean(preds)),
            }
        )
        return

    # Training mode
    st.subheader("Train an XGBoost model from your data")
    st.write("Upload a training dataset that includes `ClosePrice`. You can optionally provide a separate test dataset.")

    train_file = st.file_uploader("Training file (.csv or .parquet)", type=["csv", "parquet"], key="train_file")
    test_file = st.file_uploader("Optional test file (.csv or .parquet)", type=["csv", "parquet"], key="test_file")

    test_size = st.sidebar.slider("Auto-split test size (if no test file)", min_value=0.05, max_value=0.4, value=0.2, step=0.05)

    if not train_file:
        st.stop()

    train_df = _read_uploaded_file(train_file)
    test_df = _read_uploaded_file(test_file) if test_file else None

    if st.button("Train and save model.pkl", type="primary"):
        with st.spinner("Training model (this can take a while)..."):
            artifacts_new, metrics = train_xgb_and_build_artifacts(
                train_df,
                test_df=test_df,
                target_col=TARGET_COL,
                test_size=float(test_size),
            )
            save_artifacts(artifacts_new, MODEL_PATH)

        st.success(f"Saved `{MODEL_PATH}`")
        st.subheader("Validation metrics (R2 / MdAPE)")
        st.json(metrics)

        st.divider()
        st.info("Switch back to Predict mode to run inference with the newly created model.pkl.")


if __name__ == "__main__":
    main()

