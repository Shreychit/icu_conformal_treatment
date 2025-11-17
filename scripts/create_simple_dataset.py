from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.data_utils import load_parquet, ensure_dir


def create_simple_dataset() -> None:
    cfg = load_project_config()
    cohort_path = cfg["mimic"]["cohort_table"]
    df_pl = load_parquet(cohort_path)
    df = df_pl.to_pandas()

    if "hospital_expire_flag" not in df.columns:
        raise ValueError("hospital_expire_flag not found in cohort table")

    df = df.dropna(subset=["hospital_expire_flag"])

    feature_cols = [
        "anchor_age",
        "gender",
        "first_careunit",
        "admission_type",
        "admission_location",
        "insurance",
        "marital_status",
        "race",
    ]
    existing_features = [c for c in feature_cols if c in df.columns]
    df = df[existing_features + ["hospital_expire_flag"]].copy()

    df = df.dropna(subset=existing_features)

    y = df["hospital_expire_flag"].astype(int)
    X = df.drop(columns=["hospital_expire_flag"])

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    out_dir = Path(cfg["data"]["processed_dir"])
    ensure_dir(str(out_dir))

    train_df = X_train.copy()
    train_df["hospital_expire_flag"] = y_train.values
    val_df = X_val.copy()
    val_df["hospital_expire_flag"] = y_val.values
    test_df = X_test.copy()
    test_df["hospital_expire_flag"] = y_test.values

    train_path = out_dir / "simple_train.parquet"
    val_path = out_dir / "simple_val.parquet"
    test_path = out_dir / "simple_test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Train shape: {train_df.shape}, saved to {train_path}")
    print(f"Val shape:   {val_df.shape}, saved to {val_path}")
    print(f"Test shape:  {test_df.shape}, saved to {test_path}")


if __name__ == "__main__":
    create_simple_dataset()
