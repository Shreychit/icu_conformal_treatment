from pathlib import Path

import pandas as pd

from icu_conformal_treatment.config import load_project_config


def row_closest_alpha(df: pd.DataFrame, target_alpha: float) -> pd.Series:
    idx = (df["alpha"] - target_alpha).abs().idxmin()
    return df.loc[idx]


def main() -> None:
    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])

    logreg_path = processed_dir / "mimic_v3_conformal_policy_alpha_sweep.parquet"
    xgb_path = processed_dir / "mimic_v3_conformal_policy_alpha_sweep_xgboost.parquet"

    logreg_df = pd.read_parquet(logreg_path)
    xgb_df = pd.read_parquet(xgb_path)

    print("===== MIMIC v3 logreg T-learner conformal policy =====")
    print(logreg_df)

    print("\n-- Logreg row closest to alpha=0.2 --")
    print(row_closest_alpha(logreg_df, 0.2))

    print("\n===== MIMIC v3 XGBoost T-learner conformal policy =====")
    print(xgb_df)

    print("\n-- XGBoost row closest to alpha=0.2 --")
    print(row_closest_alpha(xgb_df, 0.2))


if __name__ == "__main__":
    main()
