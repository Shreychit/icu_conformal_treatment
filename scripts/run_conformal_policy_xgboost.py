from pathlib import Path

import pandas as pd

from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.conformal import (
    tlearner_conformal_potential_outcomes_xgb,
    dominance_policy_metrics,
)


def load_split(cfg, name: str):
    processed_dir = Path(cfg["data"]["processed_dir"])
    path = processed_dir / f"causal_{name}.parquet"
    df = pd.read_parquet(path)
    y = df["hospital_expire_flag"].astype(int)
    t = df["treatment"].astype(int)
    X = df.drop(columns=["hospital_expire_flag", "treatment"])
    return X, t, y


def main(alpha: float = 0.1) -> None:
    cfg = load_project_config()
    X_train, t_train, y_train = load_split(cfg, "train")
    X_calib, t_calib, y_calib = load_split(cfg, "calib")
    X_test, t_test, y_test = load_split(cfg, "test")

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
        score_type="residual",
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

    print(f"alpha={alpha}")
    print("model_type=xgboost_tlearner")
    print(f"q0={res['q0']:.4f}, q1={res['q1']:.4f}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main(alpha=0.1)
