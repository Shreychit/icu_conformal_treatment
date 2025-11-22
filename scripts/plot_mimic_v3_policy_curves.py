from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from icu_conformal_treatment.config import load_project_config


def load_sweep(name: str) -> pd.DataFrame:
    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    path = processed_dir / name
    return pd.read_parquet(path)


def add_decision_fraction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decision_frac"] = (df["n_decide_t0"] + df["n_decide_t1"]) / df["n_test"]
    return df


def plot_curves(df_logreg: pd.DataFrame, df_xgb: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_logreg = add_decision_fraction(df_logreg)
    df_xgb = add_decision_fraction(df_xgb)

    plt.figure()
    plt.plot(df_logreg["alpha"], df_logreg["coverage_factual"], marker="o", label="Logistic T-learner")
    plt.plot(df_xgb["alpha"], df_xgb["coverage_factual"], marker="o", label="XGBoost T-learner")
    plt.xlabel("alpha")
    plt.ylabel("Factual coverage")
    plt.title("MIMIC-IV v3: coverage vs alpha (dominance policy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mimic_v3_coverage_vs_alpha.png")
    plt.close()

    plt.figure()
    plt.plot(df_logreg["alpha"], df_logreg["decision_frac"], marker="o", label="Logistic T-learner")
    plt.plot(df_xgb["alpha"], df_xgb["decision_frac"], marker="o", label="XGBoost T-learner")
    plt.xlabel("alpha")
    plt.ylabel("Decision fraction")
    plt.title("MIMIC-IV v3: decision rate vs alpha (dominance policy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mimic_v3_decision_frac_vs_alpha.png")
    plt.close()


def main() -> None:
    df_logreg = load_sweep("mimic_v3_conformal_policy_alpha_sweep.parquet")
    df_xgb = load_sweep("mimic_v3_conformal_policy_alpha_sweep_xgboost.parquet")

    cfg = load_project_config()
    project_root = Path(cfg["project_root"])
    out_dir = project_root / "figures"

    plot_curves(df_logreg, df_xgb, out_dir)

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
