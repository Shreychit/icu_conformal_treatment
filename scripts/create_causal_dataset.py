from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.data_utils import load_parquet, ensure_dir


def create_causal_dataset() -> None:
    cfg = load_project_config()
    cohort_path = cfg["mimic"]["cohort_table"]
    df_pl = load_parquet(cohort_path)
    df = df_pl.to_pandas()

    if "hospital_expire_flag" not in df.columns:
        raise ValueError("hospital_expire_flag not found in cohort table")
    if "admission_type" not in df.columns:
        raise ValueError("admission_type not found in cohort table")

    df = df.dropna(subset=["hospital_expire_flag", "admission_type"])

    emergent_types = ["EW EMER.", "DIRECT EMER.", "URGENT"]
    df["treatment"] = df["admission_type"].isin(emergent_types).astype(int)

    feature_cols = [
        "anchor_age",
        "gender",
        "first_careunit",
        "admission_location",
        "insurance",
        "marital_status",
        "race",
    ]
    existing_features = [c for c in feature_cols if c in df.columns]
    if not existing_features:
        raise ValueError("No feature columns found for causal dataset")

    df = df[existing_features + ["treatment", "hospital_expire_flag"]].copy()
    df = df.dropna(subset=existing_features)

    y = df["hospital_expire_flag"].astype(int)
    t = df["treatment"].astype(int)
    X = df.drop(columns=["hospital_expire_flag", "treatment"])

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_temp, y_train, y_temp, t_train, t_temp = train_test_split(
        X, y, t, test_size=0.4, random_state=42, stratify=y
    )
    X_calib, X_test, y_calib, y_test, t_calib, t_test = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    out_dir = Path(cfg["data"]["processed_dir"])
    ensure_dir(str(out_dir))

    train_df = X_train.copy()
    train_df["treatment"] = t_train.values
    train_df["hospital_expire_flag"] = y_train.values

    calib_df = X_calib.copy()
    calib_df["treatment"] = t_calib.values
    calib_df["hospital_expire_flag"] = y_calib.values

    test_df = X_test.copy()
    test_df["treatment"] = t_test.values
    test_df["hospital_expire_flag"] = y_test.values

    train_path = out_dir / "causal_train.parquet"
    calib_path = out_dir / "causal_calib.parquet"
    test_path = out_dir / "causal_test.parquet"

    train_df.to_parquet(train_path, index=False)
    calib_df.to_parquet(calib_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Train shape: {train_df.shape}, saved to {train_path}")
    print(f"Calib shape: {calib_df.shape}, saved to {calib_path}")
    print(f"Test shape:  {test_df.shape}, saved to {test_path}")


if __name__ == "__main__":
    create_causal_dataset()
