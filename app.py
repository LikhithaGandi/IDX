import os

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any

from model_utils import TARGET_COL, load_artifacts, predict_xgb


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
    st.caption("🏡 Enter property details and get an estimated `ClosePrice` using XGBoost. (Same preprocessing as your notebook.)")

    artifacts = _try_load_model(MODEL_PATH)

    if artifacts is None:
        st.error("❌ `model.pkl` not found in this folder. Put the trained `model.pkl` next to `app.py` and reload.")
        st.stop()

    feature_columns = artifacts.get("feature_columns", [])
    if not feature_columns:
        st.error("`model.pkl` doesn't contain `feature_columns`. Re-train to generate compatible artifacts.")
        st.stop()

    st.subheader("🔎 Property details")
    st.caption("Fill these in and we’ll build the exact feature vector your model expects.")

    # Flooring selection drives multiple binary features: HasCarpet/HasVinyl/...
    flooring_map = {
        "Carpet": "HasCarpet",
        "Vinyl": "HasVinyl",
        "Stone": "HasStone",
        "Bamboo": "HasBamboo",
        "Concrete": "HasConcrete",
        "Brick": "HasBrick",
        "Laminate": "HasLaminate",
        "Tile": "HasTile",
        "Wood": "HasWood",
        "Unknown/Other": "HasUnknownFlooring",
    }

    # log1p_* features are computed from base inputs.
    has_log_living_area = "log1p_LivingArea" in feature_columns
    has_log_lot_size = "log1p_LotSizeSquareFeet" in feature_columns
    has_log_hoa = "log1p_Monthly_HOA" in feature_columns
    has_log_days = "log1p_DaysOnMarket" in feature_columns

    # Booleans in your current model (0/1 flags)
    view_flag = "ViewYN" in feature_columns
    waterfront_flag = "WaterfrontYN" in feature_columns
    basement_flag = "BasementYN" in feature_columns
    poolprivate_flag = "PoolPrivateYN" in feature_columns
    attached_garage_flag = "AttachedGarageYN" in feature_columns
    fireplace_flag = "FireplaceYN" in feature_columns
    new_construction_flag = "NewConstructionYN" in feature_columns

    with st.form("predict_form"):
        st.markdown("### 🧭 Core")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            view_yn = st.checkbox("🎥 Has Video Tour", value=False, disabled=not view_flag)
            waterfront_yn = st.checkbox("🌊 Waterfront property", value=False, disabled=not waterfront_flag)
            basement_yn = st.checkbox("🏠 Has Basement", value=False, disabled=not basement_flag)
            poolprivate_yn = st.checkbox("🏊 Private pool", value=False, disabled=not poolprivate_flag)
        with c2:
            latitude = st.number_input("📍 Latitude", value=0.0, format="%.6f")
            longitude = st.number_input("📍 Longitude", value=0.0, format="%.6f")
            living_area = st.number_input("📐 Living Area (sq ft)", value=0.0, min_value=0.0)
            days_on_market = st.number_input("⏳ Days on Market", value=0.0, min_value=0.0)
        with c3:
            attached_garage_yn = st.checkbox("🚗 Attached garage", value=False, disabled=not attached_garage_flag)
            parking_total = st.number_input("🅿️ Parking total", value=0.0, min_value=0.0, step=1.0)
            year_built = st.number_input("🏗️ Year built", value=2000.0, min_value=1800.0, max_value=2100.0, step=1.0)
            bathrooms_total = st.number_input("🛁 Bathrooms", value=0.0, min_value=0.0, step=1.0)

        st.markdown("### 🛏️ Interior / Garage / Lot")
        c4, c5 = st.columns([1, 1], gap="medium")
        with c4:
            bedrooms_total = st.number_input("🛏️ Bedrooms", value=0.0, min_value=0.0, step=1.0)
            fireplace_yn = st.checkbox("🔥 Fireplace", value=False, disabled=not fireplace_flag)
            stories = st.number_input("🏢 Stories", value=1.0, min_value=0.0, step=1.0)
            main_level_bedrooms = st.number_input("🧩 Main level bedrooms", value=0.0, min_value=0.0, step=1.0)
            new_construction_yn = st.checkbox("✨ New construction", value=False, disabled=not new_construction_flag)
            garage_spaces = st.number_input("🚙 Garage spaces", value=0.0, min_value=0.0, step=1.0)
        with c5:
            lot_size_sqft = st.number_input("🌳 Lot size (sq ft)", value=0.0, min_value=0.0, step=1.0)
            monthly_hoa = st.number_input("💸 Monthly HOA ($)", value=0.0, min_value=0.0, step=1.0)
            st.caption("Pick the primary flooring type 🧱")
            flooring = st.selectbox("Primary flooring", options=list(flooring_map.keys()), index=len(flooring_map) - 1)

        st.markdown("### 🧠 Advanced (encoded/derived)")
        district_avg_price = st.number_input("📊 District average price", value=0.0)
        postal_code_encoded = st.number_input("🏷️ Postal code (encoded)", value=0.0)
        dist_nearest_restaurant_mi = st.number_input("🍽️ Distance to nearest restaurant (mi)", value=0.0)

        submitted = st.form_submit_button("Predict ClosePrice", type="primary")

    if submitted:
        # Build a single-row feature dataframe with all model-required columns.
        feature_row: Dict[str, Any] = {c: 0.0 for c in feature_columns}

        if view_flag:
            feature_row["ViewYN"] = 1.0 if view_yn else 0.0
        if waterfront_flag:
            feature_row["WaterfrontYN"] = 1.0 if waterfront_yn else 0.0
        if basement_flag:
            feature_row["BasementYN"] = 1.0 if basement_yn else 0.0
        if poolprivate_flag:
            feature_row["PoolPrivateYN"] = 1.0 if poolprivate_yn else 0.0
        if attached_garage_flag:
            feature_row["AttachedGarageYN"] = 1.0 if attached_garage_yn else 0.0
        if fireplace_flag:
            feature_row["FireplaceYN"] = 1.0 if fireplace_yn else 0.0
        if new_construction_flag:
            feature_row["NewConstructionYN"] = 1.0 if new_construction_yn else 0.0

        feature_row["Latitude"] = float(latitude) if "Latitude" in feature_row else feature_row["Latitude"]
        feature_row["Longitude"] = float(longitude) if "Longitude" in feature_row else feature_row["Longitude"]
        feature_row["LivingArea"] = float(living_area) if "LivingArea" in feature_row else feature_row["LivingArea"]
        feature_row["DaysOnMarket"] = float(days_on_market) if "DaysOnMarket" in feature_row else feature_row["DaysOnMarket"]
        feature_row["ParkingTotal"] = float(parking_total) if "ParkingTotal" in feature_row else feature_row["ParkingTotal"]
        feature_row["YearBuilt"] = float(year_built) if "YearBuilt" in feature_row else feature_row["YearBuilt"]
        feature_row["BathroomsTotalInteger"] = float(bathrooms_total) if "BathroomsTotalInteger" in feature_row else feature_row["BathroomsTotalInteger"]
        feature_row["BedroomsTotal"] = float(bedrooms_total) if "BedroomsTotal" in feature_row else feature_row["BedroomsTotal"]
        feature_row["Stories"] = float(stories) if "Stories" in feature_row else feature_row["Stories"]
        feature_row["MainLevelBedrooms"] = float(main_level_bedrooms) if "MainLevelBedrooms" in feature_row else feature_row["MainLevelBedrooms"]
        feature_row["GarageSpaces"] = float(garage_spaces) if "GarageSpaces" in feature_row else feature_row["GarageSpaces"]
        feature_row["LotSizeSquareFeet"] = float(lot_size_sqft) if "LotSizeSquareFeet" in feature_row else feature_row["LotSizeSquareFeet"]
        feature_row["Monthly_HOA"] = float(monthly_hoa) if "Monthly_HOA" in feature_row else feature_row["Monthly_HOA"]

        # Flooring flags (HasCarpet/HasVinyl/...)
        selected_floor = flooring_map.get(flooring)
        if selected_floor:
            for col in list(feature_row.keys()):
                if col.startswith("Has"):
                    feature_row[col] = 0.0
            if selected_floor in feature_row:
                feature_row[selected_floor] = 1.0

        # log1p derived features
        if has_log_living_area:
            feature_row["log1p_LivingArea"] = float(np.log1p(max(0.0, living_area)))
        if has_log_lot_size:
            feature_row["log1p_LotSizeSquareFeet"] = float(np.log1p(max(0.0, lot_size_sqft)))
        if has_log_hoa:
            feature_row["log1p_Monthly_HOA"] = float(np.log1p(max(0.0, monthly_hoa)))
        if has_log_days:
            feature_row["log1p_DaysOnMarket"] = float(np.log1p(max(0.0, days_on_market)))

        if "District_Avg_Price" in feature_row:
            feature_row["District_Avg_Price"] = float(district_avg_price)
        if "Postal_Code_Encoded" in feature_row:
            feature_row["Postal_Code_Encoded"] = float(postal_code_encoded)
        if "DistNearestRestaurantMi" in feature_row:
            feature_row["DistNearestRestaurantMi"] = float(dist_nearest_restaurant_mi)

        df_features = pd.DataFrame([feature_row])

        with st.spinner("Running predictions..."):
            preds = predict_xgb(df_features, artifacts, target_col=TARGET_COL)

        st.subheader("💰 Prediction")
        st.metric("Estimated ClosePrice", f"${float(preds[0]):,.2f}")

    # Optional advanced path: users can upload a file that already has the correct feature columns.
    st.divider()
    with st.expander("📤 Upload prepared features (optional)"):
        st.caption("For power users: if your file already has the exact feature columns your model expects, you can upload it here.")
        uploaded = st.file_uploader("Upload .csv or .parquet", type=["csv", "parquet"])
        if uploaded is not None:
            df = _read_uploaded_file(uploaded)
            with st.spinner("Running predictions..."):
                preds = predict_xgb(df, artifacts, target_col=TARGET_COL)
            out = pd.DataFrame({"prediction_ClosePrice": preds})
            st.dataframe(out, use_container_width=True)
            st.write(
                {
                    "pred_min": float(np.min(preds)),
                    "pred_max": float(np.max(preds)),
                    "pred_mean": float(np.mean(preds)),
                }
            )
        return


if __name__ == "__main__":
    main()

