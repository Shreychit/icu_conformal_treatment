from pathlib import Path

import pandas as pd

from icu_conformal_treatment.conformal import (
    tlearner_conformal_potential_outcomes_xgb,
    dominance_policy_metrics,
)
from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.data_utils import ensure_dir


def load_splits(processed_dir: Path):
    train = pd.read_parquet(processed_dir / "mimic_v3_causal_train.parquet")
    calib = pd.read_parquet(processed_dir / "mimic_v3_causal_calib.parquet")
    test = pd.read_parquet(processed_dir / "mimic_v3_causal_test.parquet")
    return train, calib, test


def split_X_ty(df: pd.DataFrame):
    X = df.drop(columns=["treatment", "y"])
    t = df["treatment"]
    y = df["y"]
    return X, t, y


def run_mimic_v3_conformal_policy_alpha_sweep_xgboost(
    alphas=None,
    score_type: str = "residual",
) -> pd.DataFrame:
    if alphas is None:
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    ensure_dir(str(processed_dir))

    train_df, calib_df, test_df = load_splits(processed_dir)

    X_train, t_train, y_train = split_X_ty(train_df)
    X_calib, t_calib, y_calib = split_X_ty(calib_df)
    X_test, t_test, y_test = split_X_ty(test_df)

    rows = []

    for alpha in alphas:
        print(f"\nRunning alpha={alpha}")
        res = tlearner_conformal_potential_outcomes_xgb(
            X_train=X_train,
            t_train=t_train,
            y_train=y_train,
            X_calib=X_calib,
            t_calib=t_calib,
            y_calib=y_calib,
            X_test=X_test,
            t_test=t_test,
            alpha=alpha,
            score_type=score_type,
        )

        metrics = dominance_policy_metrics(
            L0=res["L0"],
            U0=res["U0"],
            L1=res["L1"],
            U1=res["U1"],
            y_prob_test_t0=res["y_prob_test_t0"],
            y_prob_test_t1=res["y_prob_test_t1"],
            t_test=t_test,
            y_test=y_test,
        )

        row = {
            "alpha": alpha,
            "score_type": score_type,
            "q0": res["q0"],
            "q1": res["q1"],
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = processed_dir / "mimic_v3_conformal_policy_alpha_sweep_xgboost.parquet"
    df.to_parquet(out_path, index=False)
    print("\nFull XGBoost alpha sweep:")
    print(df)
    print(f"Saved MIMIC v3 XGBoost conformal policy alpha sweep to {out_path}")
    return df


if __name__ == "__main__":
    run_mimic_v3_conformal_policy_alpha_sweep_xgboost()
