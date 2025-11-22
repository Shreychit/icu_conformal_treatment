from pathlib import Path

import pandas as pd

from icu_conformal_treatment.conformal import (
    tlearner_conformal_potential_outcomes,
    dominance_policy_metrics,
)
from icu_conformal_treatment.config import load_project_config


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


def run_subgroup_analysis(alpha: float = 0.2, score_type: str = "residual"):
    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])

    train_df, calib_df, test_df = load_splits(processed_dir)
    X_train, t_train, y_train = split_X_ty(train_df)
    X_calib, t_calib, y_calib = split_X_ty(calib_df)
    X_test, t_test, y_test = split_X_ty(test_df)

    res = tlearner_conformal_potential_outcomes(
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

    global_metrics = dominance_policy_metrics(
        L0=res["L0"],
        U0=res["U0"],
        L1=res["L1"],
        U1=res["U1"],
        y_prob_test_t0=res["y_prob_test_t0"],
        y_prob_test_t1=res["y_prob_test_t1"],
        t_test=t_test,
        y_test=y_test,
    )

    print("=== Global metrics (logreg, alpha={:.2f}) ===".format(alpha))
    for k, v in global_metrics.items():
        print(f"{k}: {v}")

    age = X_test["anchor_age"]
    age_bins = [(0, 40), (40, 60), (60, 80), (80, 200)]
    age_rows = []

    for lo, hi in age_bins:
        mask = (age >= lo) & (age < hi)
        n = int(mask.sum())
        if n == 0:
            continue

        m = dominance_policy_metrics(
            L0=res["L0"][mask.to_numpy()],
            U0=res["U0"][mask.to_numpy()],
            L1=res["L1"][mask.to_numpy()],
            U1=res["U1"][mask.to_numpy()],
            y_prob_test_t0=res["y_prob_test_t0"][mask.to_numpy()],
            y_prob_test_t1=res["y_prob_test_t1"][mask.to_numpy()],
            t_test=t_test[mask],
            y_test=y_test[mask],
        )
        row = {
            "group_type": "age_bin",
            "group_label": f"[{lo},{hi})",
            "n": n,
        }
        row.update(m)
        age_rows.append(row)

    age_df = pd.DataFrame(age_rows)
    print("\n=== Age-bin subgroup metrics ===")
    print(age_df)

    gender_cols = [c for c in X_test.columns if c.startswith("gender_")]
    gender_rows = []

    for col in gender_cols:
        mask = X_test[col] == 1
        n = int(mask.sum())
        if n == 0:
            continue

        m = dominance_policy_metrics(
            L0=res["L0"][mask.to_numpy()],
            U0=res["U0"][mask.to_numpy()],
            L1=res["L1"][mask.to_numpy()],
            U1=res["U1"][mask.to_numpy()],
            y_prob_test_t0=res["y_prob_test_t0"][mask.to_numpy()],
            y_prob_test_t1=res["y_prob_test_t1"][mask.to_numpy()],
            t_test=t_test[mask],
            y_test=y_test[mask],
        )
        row = {
            "group_type": "gender_dummy",
            "group_label": col,
            "n": n,
        }
        row.update(m)
        gender_rows.append(row)

    gender_df = pd.DataFrame(gender_rows)
    print("\n=== Gender subgroup metrics (by one-hot dummy) ===")
    print(gender_df)


if __name__ == "__main__":
    run_subgroup_analysis()
