from typing import Dict

import numpy as np
import pandas as pd


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate_synthetic_binary_outcome_data(
    n_train: int = 2000,
    n_calib: int = 1000,
    n_test: int = 1000,
    d: int = 5,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    beta_prop = np.linspace(-0.5, 0.5, d)
    beta0 = np.linspace(0.3, -0.3, d)
    beta1 = np.linspace(-0.2, 0.4, d)
    delta = 0.5

    def sample_split(n: int) -> pd.DataFrame:
        X = rng.normal(size=(n, d))
        logits_prop = X @ beta_prop
        e = _sigmoid(logits_prop)
        T = rng.binomial(1, e, size=n)

        r0 = _sigmoid(X @ beta0)
        r1 = _sigmoid(X @ beta1 - delta)

        Y = np.zeros(n, dtype=int)
        idx0 = T == 0
        idx1 = T == 1
        if idx0.any():
            Y[idx0] = rng.binomial(1, r0[idx0])
        if idx1.any():
            Y[idx1] = rng.binomial(1, r1[idx1])

        cols = {f"x{j+1}": X[:, j] for j in range(d)}
        cols["treatment"] = T
        cols["y"] = Y
        cols["r0_true"] = r0
        cols["r1_true"] = r1
        return pd.DataFrame(cols)

    train_df = sample_split(n_train)
    calib_df = sample_split(n_calib)
    test_df = sample_split(n_test)

    return {
        "train": train_df,
        "calib": calib_df,
        "test": test_df,
    }


def generate_synthetic_continuous_outcome_data(
    n_train: int = 2000,
    n_calib: int = 1000,
    n_test: int = 1000,
    d: int = 5,
    seed: int = 0,
    noise_std: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    beta_prop = np.linspace(-0.5, 0.5, d)
    beta0 = np.linspace(0.3, -0.3, d)
    beta1 = np.linspace(-0.2, 0.4, d)
    delta = 0.5

    def sample_split(n: int) -> pd.DataFrame:
        X = rng.normal(size=(n, d))
        logits_prop = X @ beta_prop
        e = _sigmoid(logits_prop)
        T = rng.binomial(1, e, size=n)

        r0 = _sigmoid(X @ beta0)
        r1 = _sigmoid(X @ beta1 - delta)

        Y = np.zeros(n, dtype=float)
        idx0 = T == 0
        idx1 = T == 1
        if idx0.any():
            Y[idx0] = r0[idx0] + rng.normal(0.0, noise_std, size=idx0.sum())
        if idx1.any():
            Y[idx1] = r1[idx1] + rng.normal(0.0, noise_std, size=idx1.sum())

        Y = np.clip(Y, 0.0, 1.0)

        cols = {f"x{j+1}": X[:, j] for j in range(d)}
        cols["treatment"] = T
        cols["y"] = Y
        cols["r0_true"] = r0
        cols["r1_true"] = r1
        return pd.DataFrame(cols)

    train_df = sample_split(n_train)
    calib_df = sample_split(n_calib)
    test_df = sample_split(n_test)

    return {
        "train": train_df,
        "calib": calib_df,
        "test": test_df,
    }
