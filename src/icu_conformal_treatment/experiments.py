from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.conformal import (
    tlearner_conformal_potential_outcomes,
    dominance_policy_metrics,
)


def load_simple_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    path = processed_dir / f"simple_{name}.parquet"
    df = pd.read_parquet(path)
    y = df["hospital_expire_flag"].astype(int)
    X = df.drop(columns=["hospital_expire_flag"])
    return X, y


def run_simple_logreg_baseline() -> Dict[str, Any]:
    X_train, y_train = load_simple_split("train")
    X_val, y_val = load_simple_split("val")
    X_test, y_test = load_simple_split("test")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    metrics = {}
    for split_name, X_split, y_split in [
        ("val", X_val_scaled, y_val),
        ("test", X_test_scaled, y_test),
    ]:
        y_prob = clf.predict_proba(X_split)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_split, y_prob)
        acc = accuracy_score(y_split, y_pred)
        metrics[f"{split_name}_auc"] = float(auc)
        metrics[f"{split_name}_acc"] = float(acc)
        metrics[f"{split_name}_n"] = int(len(y_split))

    return metrics


def load_causal_split(name: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    path = processed_dir / f"causal_{name}.parquet"
    df = pd.read_parquet(path)
    y = df["hospital_expire_flag"].astype(int)
    t = df["treatment"].astype(int)
    X = df.drop(columns=["hospital_expire_flag", "treatment"])
    return X, t, y


def run_causal_tlearner_baseline() -> Dict[str, Any]:
    X_train, t_train, y_train = load_causal_split("train")
    X_calib, t_calib, y_calib = load_causal_split("calib")
    X_test, t_test, y_test = load_causal_split("test")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    mask_t0 = t_train == 0
    mask_t1 = t_train == 1
    if mask_t0.sum() == 0 or mask_t1.sum() == 0:
        raise ValueError("One of the treatment groups is empty in training data")

    clf_t0 = LogisticRegression(max_iter=1000)
    clf_t1 = LogisticRegression(max_iter=1000)
    clf_t0.fit(X_train_scaled[mask_t0], y_train[mask_t0])
    clf_t1.fit(X_train_scaled[mask_t1], y_train[mask_t1])

    metrics: Dict[str, Any] = {}

    for split_name, X_split_scaled, t_split, y_split in [
        ("calib", X_calib_scaled, t_calib, y_calib),
        ("test", X_test_scaled, t_test, y_test),
    ]:
        y_prob = np.zeros(len(y_split), dtype=float)
        mask0 = t_split == 0
        mask1 = t_split == 1
        if mask0.any():
            y_prob[mask0] = clf_t0.predict_proba(X_split_scaled[mask0])[:, 1]
        if mask1.any():
            y_prob[mask1] = clf_t1.predict_proba(X_split_scaled[mask1])[:, 1]
        auc = roc_auc_score(y_split, y_prob)
        metrics[f"{split_name}_auc"] = float(auc)
        metrics[f"{split_name}_n"] = int(len(y_split))

    metrics["train_group_counts"] = np.bincount(t_train).tolist()
    return metrics


def run_conformal_policy_alpha_sweep(alphas: List[float]) -> pd.DataFrame:
    X_train, t_train, y_train = load_causal_split("train")
    X_calib, t_calib, y_calib = load_causal_split("calib")
    X_test, t_test, y_test = load_causal_split("test")

    rows = []
    for alpha in alphas:
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
        row = {"alpha": alpha, "q0": res["q0"], "q1": res["q1"]}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    cfg = load_project_config()
    out_path = Path(cfg["data"]["processed_dir"]) / "conformal_policy_alpha_sweep.parquet"
    df.to_parquet(out_path, index=False)
    return df
