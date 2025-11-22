from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from icu_conformal_treatment.conformal import fit_tlearner_logreg
from icu_conformal_treatment.config import load_project_config


def load_splits(processed_dir: Path):
    train = pd.read_parquet(processed_dir / "mimic_v3_causal_train.parquet")
    calib = pd.read_parquet(processed_dir / "mimic_v3_causal_calib.parquet")
    test = pd.read_parquet(processed_dir / "mimic_v3_causal_test.parquet")
    return train, calib, test


def split_X_ty(df: pd.DataFrame):
    X = df.drop(columns=["treatment", "y"])
    t = df["treatment"].astype(int)
    y = df["y"].astype(int)
    return X, t, y


def evaluate_tlearner_logreg():
    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])

    train_df, calib_df, test_df = load_splits(processed_dir)

    X_train, t_train, y_train = split_X_ty(train_df)
    X_calib, t_calib, y_calib = split_X_ty(calib_df)
    X_test, t_test, y_test = split_X_ty(test_df)

    scaler, clf_t0, clf_t1 = fit_tlearner_logreg(X_train, t_train, y_train)

    def predict_factual(X, t, clf0, clf1):
        Xs = scaler.transform(X)
        p0 = clf0.predict_proba(Xs)[:, 1]
        p1 = clf1.predict_proba(Xs)[:, 1]
        t_arr = t.to_numpy()
        return np.where(t_arr == 0, p0, p1)

    p_calib = predict_factual(X_calib, t_calib, clf_t0, clf_t1)
    p_test = predict_factual(X_test, t_test, clf_t0, clf_t1)

    y_calib_arr = y_calib.to_numpy()
    y_test_arr = y_test.to_numpy()

    calib_auc = roc_auc_score(y_calib_arr, p_calib)
    test_auc = roc_auc_score(y_test_arr, p_test)

    calib_brier = brier_score_loss(y_calib_arr, p_calib)
    test_brier = brier_score_loss(y_test_arr, p_test)

    print("MIMIC v3 logistic T-learner baseline:")
    print(f"Calib AUROC: {calib_auc:.4f}, Brier: {calib_brier:.4f}, n={len(y_calib_arr)}")
    print(f"Test  AUROC: {test_auc:.4f}, Brier: {test_brier:.4f}, n={len(y_test_arr)}")


if __name__ == "__main__":
    evaluate_tlearner_logreg()
