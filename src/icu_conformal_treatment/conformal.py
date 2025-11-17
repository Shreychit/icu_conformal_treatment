from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    if n == 0:
        raise ValueError("No scores provided for conformal quantile")
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = max(0, min(k, n - 1))
    return float(scores_sorted[k])


def fit_tlearner_logreg(
    X_train: pd.DataFrame,
    t_train: pd.Series,
    y_train: pd.Series,
) -> Tuple[StandardScaler, LogisticRegression, LogisticRegression]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    mask_t0 = t_train == 0
    mask_t1 = t_train == 1

    if mask_t0.sum() == 0 or mask_t1.sum() == 0:
        raise ValueError("One of the treatment groups is empty in training data")

    clf_t0 = LogisticRegression(max_iter=1000)
    clf_t1 = LogisticRegression(max_iter=1000)

    clf_t0.fit(X_train_scaled[mask_t0], y_train[mask_t0])
    clf_t1.fit(X_train_scaled[mask_t1], y_train[mask_t1])

    return scaler, clf_t0, clf_t1


def fit_tlearner_xgboost(
    X_train: pd.DataFrame,
    t_train: pd.Series,
    y_train: pd.Series,
) -> Tuple[StandardScaler, Any, Any]:
    from xgboost import XGBClassifier

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    mask_t0 = t_train == 0
    mask_t1 = t_train == 1

    if mask_t0.sum() == 0 or mask_t1.sum() == 0:
        raise ValueError("One of the treatment groups is empty in training data")

    clf_t0 = XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )
    clf_t1 = XGBClassifier(
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )

    clf_t0.fit(X_train_scaled[mask_t0], y_train[mask_t0])
    clf_t1.fit(X_train_scaled[mask_t1], y_train[mask_t1])

    return scaler, clf_t0, clf_t1


def compute_nonconformity_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    score_type: str = "residual",
) -> np.ndarray:
    if score_type == "residual":
        return np.abs(y_true - y_prob)
    if score_type == "nll":
        eps = 1e-6
        p = np.clip(y_prob, eps, 1.0 - eps)
        return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    raise ValueError(f"Unknown score_type: {score_type}")


def compute_conformal_scores(
    scaler: StandardScaler,
    clf_t0,
    clf_t1,
    X_calib: pd.DataFrame,
    t_calib: pd.Series,
    y_calib: pd.Series,
    score_type: str = "residual",
) -> Tuple[np.ndarray, np.ndarray]:
    X_calib_scaled = scaler.transform(X_calib)

    y_prob_calib_t0 = np.zeros(len(y_calib), dtype=float)
    y_prob_calib_t1 = np.zeros(len(y_calib), dtype=float)

    mask_t0_calib = t_calib == 0
    mask_t1_calib = t_calib == 1

    if mask_t0_calib.any():
        y_prob_calib_t0[mask_t0_calib] = clf_t0.predict_proba(X_calib_scaled[mask_t0_calib])[:, 1]
    if mask_t1_calib.any():
        y_prob_calib_t1[mask_t1_calib] = clf_t1.predict_proba(X_calib_scaled[mask_t1_calib])[:, 1]

    scores_t0 = compute_nonconformity_scores(
        y_true=y_calib[mask_t0_calib].to_numpy(),
        y_prob=y_prob_calib_t0[mask_t0_calib],
        score_type=score_type,
    )
    scores_t1 = compute_nonconformity_scores(
        y_true=y_calib[mask_t1_calib].to_numpy(),
        y_prob=y_prob_calib_t1[mask_t1_calib],
        score_type=score_type,
    )

    return scores_t0, scores_t1


def tlearner_conformal_potential_outcomes(
    X_train: pd.DataFrame,
    t_train: pd.Series,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    t_calib: pd.Series,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    t_test: pd.Series,
    alpha: float = 0.1,
    score_type: str = "residual",
) -> Dict[str, Any]:
    scaler, clf_t0, clf_t1 = fit_tlearner_logreg(X_train, t_train, y_train)

    scores_t0, scores_t1 = compute_conformal_scores(
        scaler=scaler,
        clf_t0=clf_t0,
        clf_t1=clf_t1,
        X_calib=X_calib,
        t_calib=t_calib,
        y_calib=y_calib,
        score_type=score_type,
    )

    q0 = conformal_quantile(scores_t0, alpha)
    q1 = conformal_quantile(scores_t1, alpha)

    X_test_scaled = scaler.transform(X_test)
    y_prob_test_t0 = clf_t0.predict_proba(X_test_scaled)[:, 1]
    y_prob_test_t1 = clf_t1.predict_proba(X_test_scaled)[:, 1]

    if score_type == "residual":
        L0 = np.clip(y_prob_test_t0 - q0, 0.0, 1.0)
        U0 = np.clip(y_prob_test_t0 + q0, 0.0, 1.0)
        L1 = np.clip(y_prob_test_t1 - q1, 0.0, 1.0)
        U1 = np.clip(y_prob_test_t1 + q1, 0.0, 1.0)
    elif score_type == "nll":
        L0 = np.zeros_like(y_prob_test_t0)
        U0 = np.ones_like(y_prob_test_t0)
        L1 = np.zeros_like(y_prob_test_t1)
        U1 = np.ones_like(y_prob_test_t1)
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    return {
        "scaler": scaler,
        "clf_t0": clf_t0,
        "clf_t1": clf_t1,
        "q0": q0,
        "q1": q1,
        "y_prob_test_t0": y_prob_test_t0,
        "y_prob_test_t1": y_prob_test_t1,
        "L0": L0,
        "U0": U0,
        "L1": L1,
        "U1": U1,
    }


def tlearner_conformal_potential_outcomes_xgb(
    X_train: pd.DataFrame,
    t_train: pd.Series,
    y_train: pd.Series,
    X_calib: pd.DataFrame,
    t_calib: pd.Series,
    y_calib: pd.Series,
    X_test: pd.DataFrame,
    t_test: pd.Series,
    alpha: float = 0.1,
    score_type: str = "residual",
) -> Dict[str, Any]:
    scaler, clf_t0, clf_t1 = fit_tlearner_xgboost(X_train, t_train, y_train)

    scores_t0, scores_t1 = compute_conformal_scores(
        scaler=scaler,
        clf_t0=clf_t0,
        clf_t1=clf_t1,
        X_calib=X_calib,
        t_calib=t_calib,
        y_calib=y_calib,
        score_type=score_type,
    )

    q0 = conformal_quantile(scores_t0, alpha)
    q1 = conformal_quantile(scores_t1, alpha)

    X_test_scaled = scaler.transform(X_test)
    y_prob_test_t0 = clf_t0.predict_proba(X_test_scaled)[:, 1]
    y_prob_test_t1 = clf_t1.predict_proba(X_test_scaled)[:, 1]

    if score_type == "residual":
        L0 = np.clip(y_prob_test_t0 - q0, 0.0, 1.0)
        U0 = np.clip(y_prob_test_t0 + q0, 0.0, 1.0)
        L1 = np.clip(y_prob_test_t1 - q1, 0.0, 1.0)
        U1 = np.clip(y_prob_test_t1 + q1, 0.0, 1.0)
    elif score_type == "nll":
        L0 = np.zeros_like(y_prob_test_t0)
        U0 = np.ones_like(y_prob_test_t0)
        L1 = np.zeros_like(y_prob_test_t1)
        U1 = np.ones_like(y_prob_test_t1)
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    return {
        "scaler": scaler,
        "clf_t0": clf_t0,
        "clf_t1": clf_t1,
        "q0": q0,
        "q1": q1,
        "y_prob_test_t0": y_prob_test_t0,
        "y_prob_test_t1": y_prob_test_t1,
        "L0": L0,
        "U0": U0,
        "L1": L1,
        "U1": U1,
    }


def dominance_policy_metrics(
    L0: np.ndarray,
    U0: np.ndarray,
    L1: np.ndarray,
    U1: np.ndarray,
    y_prob_test_t0: np.ndarray,
    y_prob_test_t1: np.ndarray,
    t_test: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    y_test_arr = y_test.to_numpy()
    t_test_arr = t_test.to_numpy()

    widths0 = U0 - L0
    widths1 = U1 - L1

    factual_L = np.where(t_test_arr == 0, L0, L1)
    factual_U = np.where(t_test_arr == 0, U0, U1)
    coverage_factual = ((y_test_arr >= factual_L) & (y_test_arr <= factual_U)).mean()

    y_prob_factual = np.where(t_test_arr == 0, y_prob_test_t0, y_prob_test_t1)
    auc_factual = roc_auc_score(y_test_arr, y_prob_factual)

    choose_t1 = U1 < L0
    choose_t0 = U0 < L1
    abstain = ~(choose_t0 | choose_t1)

    n = len(y_test_arr)
    n_t1 = int(choose_t1.sum())
    n_t0 = int(choose_t0.sum())
    n_abstain = int(abstain.sum())

    risk_best = np.minimum(y_prob_test_t0, y_prob_test_t1)

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

    return {
        "mean_width_t0": float(widths0.mean()),
        "mean_width_t1": float(widths1.mean()),
        "coverage_factual": float(coverage_factual),
        "auc_factual": float(auc_factual),
        "n_test": int(n),
        "n_decide_t0": n_t0,
        "n_decide_t1": n_t1,
        "n_abstain": n_abstain,
        "mean_regret_decided": mean_regret,
        "p90_regret_decided": p90_regret,
        "max_regret_decided": max_regret,
    }
