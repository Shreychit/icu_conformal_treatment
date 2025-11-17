from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from icu_conformal_treatment.config import load_project_config


def load_split(cfg, name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    processed_dir = Path(cfg["data"]["processed_dir"])
    path = processed_dir / f"causal_{name}.parquet"
    df = pd.read_parquet(path)
    y = df["hospital_expire_flag"].astype(int)
    t = df["treatment"].astype(int)
    X = df.drop(columns=["hospital_expire_flag", "treatment"])
    return X, t, y


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    if n == 0:
        raise ValueError("No scores provided for conformal quantile")
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = max(0, min(k, n - 1))
    return float(scores_sorted[k])


def sweep_alphas(alphas: list[float]) -> None:
    cfg = load_project_config()
    X_train, t_train, y_train = load_split(cfg, "train")
    X_calib, t_calib, y_calib = load_split(cfg, "calib")
    X_test, t_test, y_test = load_split(cfg, "test")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    mask_t0_train = t_train == 0
    mask_t1_train = t_train == 1
    if mask_t0_train.sum() == 0 or mask_t1_train.sum() == 0:
        raise ValueError("One of the treatment groups is empty in training data")

    clf_t0 = LogisticRegression(max_iter=1000)
    clf_t1 = LogisticRegression(max_iter=1000)
    clf_t0.fit(X_train_scaled[mask_t0_train], y_train[mask_t0_train])
    clf_t1.fit(X_train_scaled[mask_t1_train], y_train[mask_t1_train])

    y_prob_calib_t0 = np.zeros(len(y_calib), dtype=float)
    y_prob_calib_t1 = np.zeros(len(y_calib), dtype=float)
    mask_t0_calib = t_calib == 0
    mask_t1_calib = t_calib == 1
    if mask_t0_calib.any():
        y_prob_calib_t0[mask_t0_calib] = clf_t0.predict_proba(X_calib_scaled[mask_t0_calib])[:, 1]
    if mask_t1_calib.any():
        y_prob_calib_t1[mask_t1_calib] = clf_t1.predict_proba(X_calib_scaled[mask_t1_calib])[:, 1]

    scores_t0 = np.abs(y_calib[mask_t0_calib].to_numpy() - y_prob_calib_t0[mask_t0_calib])
    scores_t1 = np.abs(y_calib[mask_t1_calib].to_numpy() - y_prob_calib_t1[mask_t1_calib])

    y_test_arr = y_test.to_numpy()
    y_prob_test_t0 = clf_t0.predict_proba(X_test_scaled)[:, 1]
    y_prob_test_t1 = clf_t1.predict_proba(X_test_scaled)[:, 1]
    risk_best = np.minimum(y_prob_test_t0, y_prob_test_t1)

    rows = []

    for alpha in alphas:
        q0 = conformal_quantile(scores_t0, alpha)
        q1 = conformal_quantile(scores_t1, alpha)

        L0 = np.clip(y_prob_test_t0 - q0, 0.0, 1.0)
        U0 = np.clip(y_prob_test_t0 + q0, 0.0, 1.0)
        L1 = np.clip(y_prob_test_t1 - q1, 0.0, 1.0)
        U1 = np.clip(y_prob_test_t1 + q1, 0.0, 1.0)

        widths0 = U0 - L0
        widths1 = U1 - L1

        factual_L = np.where(t_test == 0, L0, L1)
        factual_U = np.where(t_test == 0, U0, U1)
        coverage_factual = ((y_test_arr >= factual_L) & (y_test_arr <= factual_U)).mean()

        y_prob_factual = np.where(t_test == 0, y_prob_test_t0, y_prob_test_t1)
        auc_factual = roc_auc_score(y_test_arr, y_prob_factual)

        choose_t1 = U1 < L0
        choose_t0 = U0 < L1
        abstain = ~(choose_t0 | choose_t1)

        n = len(y_test_arr)
        n_t1 = int(choose_t1.sum())
        n_t0 = int(choose_t0.sum())
        n_abstain = int(abstain.sum())

        decided_mask = ~abstain
        risk_decision = np.zeros(n, dtype=float)
        risk_decision[choose_t0] = y_prob_test_t0[choose_t0]
        risk_decision[choose_t1] = y_prob_test_t1[choose_t1]

        if decided_mask.sum() > 0:
            regret_hat = risk_decision[decided_mask] - risk_best[decided_mask]
            mean_regret = float(regret_hat.mean())
            p90_regret = float(np.percentile(regret_hat, 90))
            max_regret = float(regret_hat.max())
        else:
            mean_regret = float("nan")
            p90_regret = float("nan")
            max_regret = float("nan")

        rows.append(
            {
                "alpha": alpha,
                "q0": q0,
                "q1": q1,
                "mean_width_t0": widths0.mean(),
                "mean_width_t1": widths1.mean(),
                "coverage_factual": coverage_factual,
                "auc_factual": auc_factual,
                "n_test": n,
                "n_decide_t0": n_t0,
                "n_decide_t1": n_t1,
                "n_abstain": n_abstain,
                "mean_regret_decided": mean_regret,
                "p90_regret_decided": p90_regret,
                "max_regret_decided": max_regret,
            }
        )

    df_out = pd.DataFrame(rows)
    out_path = Path(cfg["data"]["processed_dir"]) / "conformal_policy_alpha_sweep.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"Saved alpha sweep results to {out_path}")
    print(df_out)


if __name__ == "__main__":
    sweep_alphas([0.05, 0.1, 0.2, 0.3, 0.4])
