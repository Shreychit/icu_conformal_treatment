from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from icu_conformal_treatment.config import load_project_config


def load_split(cfg, name: str) -> tuple[pd.DataFrame, pd.Series]:
    processed_dir = Path(cfg["data"]["processed_dir"])
    path = processed_dir / f"simple_{name}.parquet"
    df = pd.read_parquet(path)
    y = df["hospital_expire_flag"].astype(int)
    X = df.drop(columns=["hospital_expire_flag"])
    return X, y


def run_baseline_logreg() -> None:
    cfg = load_project_config()
    X_train, y_train = load_split(cfg, "train")
    X_val, y_val = load_split(cfg, "val")
    X_test, y_test = load_split(cfg, "test")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    for split_name, X_split, y_split in [
        ("val", X_val_scaled, y_val),
        ("test", X_test_scaled, y_test),
    ]:
        y_prob = clf.predict_proba(X_split)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_split, y_prob)
        acc = accuracy_score(y_split, y_pred)
        print(f"{split_name.upper()} AUROC: {auc:.4f}, ACC: {acc:.4f}, n={len(y_split)}")


if __name__ == "__main__":
    run_baseline_logreg()
