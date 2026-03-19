# Streamlit XGBoost Predictor (ClosePrice)

This app uses the preprocessing logic from your notebook (scale non-binary numeric columns, one-hot encode categorical `object` columns, then align dummy columns) and predicts `ClosePrice` with the best notebook model: **XGBoost**.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Create `model.pkl` (option A: via training script)

The training file must be a `.csv` or `.parquet` that includes the target column `ClosePrice`.

```bash
python train_model.py --train "path/to/train.csv" --test "path/to/test.csv" --out model.pkl
```

If you don't have a separate test file, omit `--test`:

```bash
python train_model.py --train "path/to/train.csv" --out model.pkl --test-size 0.2
```

## 3) Run the Streamlit app

```bash
streamlit run app.py
```

## 4) Deploy (Streamlit Community Cloud-style)

1. Create a GitHub repo that contains:
   - `app.py`
   - `requirements.txt`
   - `model.pkl`
   - `model_utils.py`
   - `train_model.py` (optional)
2. Go to Streamlit Cloud, create/deploy an app, and set:
   - **Main file path**: `app.py`

## Notes

- If `model.pkl` is not present, the app tells you to switch to **Train** mode.
- For predictions, upload a `.csv` or `.parquet` that contains the feature columns. If it also includes `ClosePrice`, the app ignores it automatically.

