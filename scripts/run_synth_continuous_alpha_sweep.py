import numpy as np
import pandas as pd

from icu_conformal_treatment.conformal import (
    tlearner_conformal_potential_outcomes_regression,
)
from icu_conformal_treatment.simulations import (
    generate_synthetic_continuous_outcome_data,
)
from icu_conformal_treatment.data_utils import ensure_dir


def run_synth_continuous_alpha_sweep(
    alphas=None,
    score_type: str = "residual",
) -> pd.DataFrame:
    if alphas is None:
        alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]

    data = generate_synthetic_continuous_outcome_data(
        n_train=2000,
        n_calib=1000,
        n_test=1000,
        d=5,
        seed=777,
        noise_std=0.05,
    )

    train_df = data["train"]
    calib_df = data["calib"]
    test_df = data["test"]

    X_train = train_df[[c for c in train_df.columns if c.startswith("x")]]
    t_train = train_df["treatment"].astype(int)
    y_train = train_df["y"].astype(float)

    X_calib = calib_df[[c for c in calib_df.columns if c.startswith("x")]]
    t_calib = calib_df["treatment"].astype(int)
    y_calib = calib_df["y"].astype(float)

    X_test = test_df[[c for c in test_df.columns if c.startswith("x")]]
    t_test = test_df["treatment"].astype(int)
    y_test = test_df["y"].astype(float)
    r0_true = test_df["r0_true"].to_numpy()
    r1_true = test_df["r1_true"].to_numpy()

    rows = []

    for alpha in alphas:
        res = tlearner_conformal_potential_outcomes_regression(
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

        L0 = res["L0"]
        U0 = res["U0"]
        L1 = res["L1"]
        U1 = res["U1"]
        y_pred_t0 = res["y_prob_test_t0"]
        y_pred_t1 = res["y_prob_test_t1"]

        y_test_arr = y_test.to_numpy()

        widths0 = U0 - L0
        widths1 = U1 - L1
        mean_width_t0 = float(widths0.mean())
        mean_width_t1 = float(widths1.mean())

        factual_L = np.where(t_test.to_numpy() == 0, L0, L1)
        factual_U = np.where(t_test.to_numpy() == 0, U0, U1)
        coverage_factual = ((y_test_arr >= factual_L) & (y_test_arr <= factual_U)).mean()

        choose_t1 = U1 < L0
        choose_t0 = U0 < L1
        abstain = ~(choose_t0 | choose_t1)

        n = len(y_test_arr)
        n_t1 = int(choose_t1.sum())
        n_t0 = int(choose_t0.sum())
        n_abstain = int(abstain.sum())

        risk_prob_best = np.minimum(y_pred_t0, y_pred_t1)
        decided_mask = ~abstain
        risk_prob_decision = np.zeros(n, dtype=float)
        risk_prob_decision[choose_t0] = y_pred_t0[choose_t0]
        risk_prob_decision[choose_t1] = y_pred_t1[choose_t1]
        if decided_mask.sum() > 0:
            pseudo_regret = risk_prob_decision[decided_mask] - risk_prob_best[decided_mask]
            mean_pseudo_regret = float(pseudo_regret.mean())
            p90_pseudo_regret = float(np.percentile(pseudo_regret, 90))
            max_pseudo_regret = float(pseudo_regret.max())
        else:
            mean_pseudo_regret = float("nan")
            p90_pseudo_regret = float("nan")
            max_pseudo_regret = float("nan")

        r_best_true = np.minimum(r0_true, r1_true)
        decision_t = np.full(n, -1, dtype=int)
        decision_t[choose_t0] = 0
        decision_t[choose_t1] = 1

        regret_true = np.zeros(n, dtype=float)
        idx_decided0 = decision_t == 0
        idx_decided1 = decision_t == 1
        if idx_decided0.any():
            regret_true[idx_decided0] = r0_true[idx_decided0] - r_best_true[idx_decided0]
        if idx_decided1.any():
            regret_true[idx_decided1] = r1_true[idx_decided1] - r_best_true[idx_decided1]

        if (idx_decided0 | idx_decided1).any():
            decided_mask_true = idx_decided0 | idx_decided1
            mean_true_regret = float(regret_true[decided_mask_true].mean())
            p90_true_regret = float(np.percentile(regret_true[decided_mask_true], 90))
            max_true_regret = float(regret_true[decided_mask_true].max())
        else:
            mean_true_regret = float("nan")
            p90_true_regret = float("nan")
            max_true_regret = float("nan")

        rows.append(
            {
                "alpha": alpha,
                "score_type": score_type,
                "q0": res["q0"],
                "q1": res["q1"],
                "mean_width_t0": mean_width_t0,
                "mean_width_t1": mean_width_t1,
                "coverage_factual": coverage_factual,
                "n_test": n,
                "n_decide_t0": n_t0,
                "n_decide_t1": n_t1,
                "n_abstain": n_abstain,
                "mean_pseudo_regret_decided": mean_pseudo_regret,
                "p90_pseudo_regret_decided": p90_pseudo_regret,
                "max_pseudo_regret_decided": max_pseudo_regret,
                "mean_true_regret_decided": mean_true_regret,
                "p90_true_regret_decided": p90_true_regret,
                "max_true_regret_decided": max_true_regret,
            }
        )

    df = pd.DataFrame(rows)
    ensure_dir("data/processed")
    out_path = "data/processed/synth_continuous_alpha_sweep.parquet"
    df.to_parquet(out_path, index=False)
    print(df)
    print(f"Saved continuous synthetic alpha sweep to {out_path}")
    return df


if __name__ == "__main__":
    run_synth_continuous_alpha_sweep()
