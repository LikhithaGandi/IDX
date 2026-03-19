import argparse
import os

import pandas as pd

from model_utils import TARGET_COL, save_artifacts, train_xgb_and_build_artifacts


def _read_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost and save model.pkl artifacts.")
    parser.add_argument("--train", required=True, help="Path to training CSV/parquet containing ClosePrice.")
    parser.add_argument("--test", default=None, help="Optional path to test CSV/parquet containing ClosePrice.")
    parser.add_argument("--target", default=TARGET_COL, help="Target column name (default: ClosePrice).")
    parser.add_argument("--out", default="model.pkl", help="Output path (default: model.pkl).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Used only if --test is not provided.")
    args = parser.parse_args()

    train_df = _read_df(args.train)
    test_df = _read_df(args.test) if args.test else None

    artifacts, metrics = train_xgb_and_build_artifacts(
        train_df,
        test_df=test_df,
        target_col=args.target,
        test_size=args.test_size,
    )

    save_artifacts(artifacts, args.out)

    print("Saved:", args.out)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()

