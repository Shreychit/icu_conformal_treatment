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


def run_conformal_policy(alpha: float = 0.1) -> None:
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

    q0 = conformal_quantile(scores_t0, alpha)
    q1 = conformal_quantile(scores_t1, alpha)

    print(f"alpha={alpha}")
    print(f"Train group counts: {np.bincount(t_train)}")
    print(f"Calib group counts: {np.bincount(t_calib)}")
    print(f"Conformal quantiles q0={q0:.4f} (T=0), q1={q1:.4f} (T=1)")

    y_prob_test_t0 = clf_t0.predict_proba(X_test_scaled)[:, 1]
    y_prob_test_t1 = clf_t1.predict_proba(X_test_scaled)[:, 1]

    L0 = np.clip(y_prob_test_t0 - q0, 0.0, 1.0)
    U0 = np.clip(y_prob_test_t0 + q0, 0.0, 1.0)
    L1 = np.clip(y_prob_test_t1 - q1, 0.0, 1.0)
    U1 = np.clip(y_prob_test_t1 + q1, 0.0, 1.0)

    widths0 = U0 - L0
    widths1 = U1 - L1

    print(f"Mean interval width T=0: {widths0.mean():.4f}")
    print(f"Mean interval width T=1: {widths1.mean():.4f}")

    y_test_arr = y_test.to_numpy()
    factual_L = np.where(t_test == 0, L0, L1)
    factual_U = np.where(t_test == 0, U0, U1)
    coverage_factual = ((y_test_arr >= factual_L) & (y_test_arr <= factual_U)).mean()
    print(f"Test factual coverage: {coverage_factual:.4f}, n={len(y_test_arr)}")

    y_prob_factual = np.where(t_test == 0, y_prob_test_t0, y_prob_test_t1)
    auc_factual = roc_auc_score(y_test_arr, y_prob_factual)
    print(f"Test AUROC (factual, T-learner): {auc_factual:.4f}")

    choose_t1 = U1 < L0
    choose_t0 = U0 < L1
    abstain = ~(choose_t0 | choose_t1)

    n = len(y_test_arr)
    n_t1 = int(choose_t1.sum())
    n_t0 = int(choose_t0.sum())
    n_abstain = int(abstain.sum())

    print(f"Decisions: T=1 for {n_t1} patients, T=0 for {n_t0} patients, abstain for {n_abstain} patients (total {n})")

    risk_decision = np.zeros(n, dtype=float)
    decided_mask = ~abstain
    risk_decision[choose_t0] = y_prob_test_t0[choose_t0]
    risk_decision[choose_t1] = y_prob_test_t1[choose_t1]

    risk_best = np.minimum(y_prob_test_t0, y_prob_test_t1)
    regret_hat = risk_decision[decided_mask] - risk_best[decided_mask]

    if decided_mask.sum() > 0:
        mean_regret = float(regret_hat.mean())
        p90_regret = float(np.percentile(regret_hat, 90))
        max_regret = float(regret_hat.max())
        print(f"Pseudo-regret among decided patients (risk difference): mean={mean_regret:.4f}, p90={p90_regret:.4f}, max={max_regret:.4f}")
    else:
        print("No patients received a treatment decision (all abstained).")

    risk_naive = np.where(y_prob_test_t1 < y_prob_test_t0, y_prob_test_t1, y_prob_test_t0)
    chosen_naive = np.where(y_prob_test_t1 < y_prob_test_t0, 1, 0)
    regret_naive = risk_naive - risk_best
    print(f"Naive oracle-following mean regret (should be ~0): {regret_naive.mean():.4f}")
    print(f"Naive choice distribution: T=0 for {(chosen_naive == 0).sum()}, T=1 for {(chosen_naive == 1).sum()}")


if __name__ == "__main__":
    run_conformal_policy(alpha=0.1)
