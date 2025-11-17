from pathlib import Path

import pandas as pd


def load_parquet_if_exists(path: Path):
    if path.exists():
        return pd.read_parquet(path)
    print(f"[WARN] Missing file: {path}")
    return None


def summarize_alpha_table(name: str, df: pd.DataFrame, alpha_target: float = 0.1) -> None:
    print(f"\n===== {name} =====")
    print(df)

    idx = (df["alpha"] - alpha_target).abs().idxmin()
    row = df.loc[idx]

    print(f"\n-- Row closest to alpha={alpha_target} --")
    print(row.to_string())


def main():
    base = Path("data/processed")

    files = {
        "MIMIC demo (logreg T-learner conformal policy alpha sweep)": base / "conformal_policy_alpha_sweep.parquet",
        "Synthetic binary (logreg T-learner)": base / "synth_conformal_alpha_sweep.parquet",
        "Synthetic binary (XGBoost T-learner)": base / "synth_conformal_alpha_sweep_xgboost.parquet",
        "Synthetic continuous (regression T-learner)": base / "synth_continuous_alpha_sweep.parquet",
    }

    for name, path in files.items():
        df = load_parquet_if_exists(path)
        if df is None:
            continue
        summarize_alpha_table(name, df, alpha_target=0.1)


if __name__ == "__main__":
    main()
