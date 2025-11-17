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


def run_causal_tlearner_logreg() -> None:
    cfg = load_project_config()
    X_train, t_train, y_train = load_split(cfg, "train")
    X_calib, t_calib, y_calib = load_split(cfg, "calib")
    X_test, t_test, y_test = load_split(cfg, "test")

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
        print(f"{split_name.upper()} AUROC (T-learner): {auc:.4f}, n={len(y_split)}")

    print("Train group counts:", np.bincount(t_train))


if __name__ == "__main__":
    run_causal_tlearner_logreg()
