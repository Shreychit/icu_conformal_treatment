from pathlib import Path

import pandas as pd

from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.data_utils import ensure_dir


def main() -> None:
    cfg = load_project_config()
    mimic_cfg = cfg["mimic_full"]

    root_dir = Path(mimic_cfg["root_dir"])
    out_path = Path(mimic_cfg["cohort_table"])
    ensure_dir(str(out_path.parent))

    hosp_adm_path = root_dir / "hosp" / "admissions.csv"
    hosp_pat_path = root_dir / "hosp" / "patients.csv"
    icu_stay_path = root_dir / "icu" / "icustays.csv"

    print(f"Loading admissions from {hosp_adm_path}")
    admissions = pd.read_csv(hosp_adm_path)

    print(f"Loading patients from {hosp_pat_path}")
    patients = pd.read_csv(hosp_pat_path)

    print(f"Loading ICU stays from {icu_stay_path}")
    icustays = pd.read_csv(icu_stay_path)

    icustays_sorted = icustays.sort_values(["subject_id", "intime"])
    first_icu = icustays_sorted.groupby("subject_id", as_index=False).first()

    cohort = first_icu.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="left",
        suffixes=("", "_adm"),
    )

    cohort = cohort.merge(
        patients,
        on="subject_id",
        how="left",
        suffixes=("", "_pat"),
    )

    if "anchor_age" in cohort.columns:
        cohort = cohort[cohort["anchor_age"] >= 18].copy()
    else:
        raise ValueError("anchor_age column not found in patients table")

    if "los" in cohort.columns and "los_hours" not in cohort.columns:
        cohort["los_hours"] = cohort["los"] * 24.0

    for col in ["hospital_expire_flag"]:
        if col not in cohort.columns:
            raise ValueError(f"Required column {col} not found in admissions table")

    cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "gender",
        "anchor_age",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
        "los_hours",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "marital_status",
        "race",
        "hospital_expire_flag",
    ]

    missing = [c for c in cols if c not in cohort.columns]
    if missing:
        raise ValueError(f"Missing expected columns in joined cohort: {missing}")

    cohort = cohort[cols].copy()

    cohort = cohort.dropna(subset=["hospital_expire_flag"])

    cohort.to_parquet(out_path, index=False)
    print(f"Saved full MIMIC-IV v3.0 cohort to {out_path} with shape {cohort.shape}")


if __name__ == "__main__":
    main()
