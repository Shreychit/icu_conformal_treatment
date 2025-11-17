from pathlib import Path

import polars as pl

from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.data_utils import ensure_dir, save_parquet


def build_cohort() -> None:
    cfg = load_project_config()
    root_dir = Path(cfg["mimic"]["root_dir"])
    hosp_dir = root_dir / "hosp"
    icu_dir = root_dir / "icu"

    admissions_path = hosp_dir / "admissions.csv"
    patients_path = hosp_dir / "patients.csv"
    icustays_path = icu_dir / "icustays.csv"

    admissions = pl.read_csv(admissions_path)
    patients = pl.read_csv(patients_path)
    icustays = pl.read_csv(icustays_path)

    cohort = (
        icustays.join(
            admissions,
            on=["subject_id", "hadm_id"],
            how="left",
        )
        .join(
            patients,
            on="subject_id",
            how="left",
        )
    )

    if "anchor_age" in cohort.columns:
        cohort = cohort.filter(pl.col("anchor_age") >= 18)

    if "los" in cohort.columns:
        cohort = cohort.with_columns(
            (pl.col("los") * 24.0).alias("los_hours")
        )

    columns_to_keep = [
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
    existing_columns = [c for c in columns_to_keep if c in cohort.columns]
    cohort = cohort.select(existing_columns)

    out_path = Path(cfg["mimic"]["cohort_table"])
    ensure_dir(str(out_path.parent))
    save_parquet(cohort, str(out_path))

    print(f"Saved cohort to {out_path} with shape {cohort.shape}")


if __name__ == "__main__":
    build_cohort()
