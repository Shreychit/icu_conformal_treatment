from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.data_utils import ensure_dir


def main() -> None:
    cfg = load_project_config()
    mimic_cfg = cfg["mimic_full"]

    cohort_path = Path(mimic_cfg["cohort_table"])
    out_dir = Path(cfg["data"]["processed_dir"])
    ensure_dir(str(out_dir))

    print(f"Loading full cohort from {cohort_path}")
    cohort = pd.read_parquet(cohort_path)
    print(f"Cohort shape: {cohort.shape}")

    required_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "gender",
        "anchor_age",
        "first_careunit",
        "admission_type",
        "admission_location",
        "insurance",
        "marital_status",
        "race",
        "los_hours",
        "hospital_expire_flag",
    ]
    missing = [c for c in required_cols if c not in cohort.columns]
    if missing:
        raise ValueError(f"Missing required columns in cohort: {missing}")

    y = cohort["hospital_expire_flag"].astype(int)

    adm_type_upper = cohort["admission_type"].fillna("UNKNOWN").str.upper()
    treatment = (
        adm_type_upper.str.contains("EMER") | (adm_type_upper == "URGENT")
    ).astype(int)

    print("Treatment value counts (0=non-emergent, 1=emergent/urgent):")
    print(treatment.value_counts())

    feature_cols = [
        "gender",
        "anchor_age",
        "first_careunit",
        "admission_type",
        "admission_location",
        "insurance",
        "marital_status",
        "race",
        "los_hours",
    ]
    X_raw = cohort[feature_cols].copy()

    cat_cols = [
        "gender",
        "first_careunit",
        "admission_type",
        "admission_location",
        "insurance",
        "marital_status",
        "race",
    ]
    num_cols = ["anchor_age", "los_hours"]

    for c in cat_cols:
        X_raw[c] = X_raw[c].fillna("UNKNOWN")

    X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=False)

    df_all = X.copy()
    df_all["treatment"] = treatment.to_numpy()
    df_all["y"] = y.to_numpy()

    print(f"Final feature matrix shape (including treatment,y): {df_all.shape}")

    train_val, test = train_test_split(
        df_all,
        test_size=0.2,
        random_state=42,
        stratify=df_all["y"],
    )

    train, calib = train_test_split(
        train_val,
        test_size=0.25,
        random_state=43,
        stratify=train_val["y"],
    )

    print(f"Train shape: {train.shape}")
    print(f"Calib shape: {calib.shape}")
    print(f"Test shape: {test.shape}")

    train_path = out_dir / "mimic_v3_causal_train.parquet"
    calib_path = out_dir / "mimic_v3_causal_calib.parquet"
    test_path = out_dir / "mimic_v3_causal_test.parquet"

    train.to_parquet(train_path, index=False)
    calib.to_parquet(calib_path, index=False)
    test.to_parquet(test_path, index=False)

    print(f"Saved train to {train_path}")
    print(f"Saved calib to {calib_path}")
    print(f"Saved test to {test_path}")

    for name, df in [("train", train), ("calib", calib), ("test", test)]:
        print(f"\n[{name}] y value counts:")
        print(df["y"].value_counts())
        print(f"[{name}] treatment value counts:")
        print(df["treatment"].value_counts())


if __name__ == "__main__":
    main()
