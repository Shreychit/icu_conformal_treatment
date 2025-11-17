from pathlib import Path

from icu_conformal_treatment.config import load_project_config
from icu_conformal_treatment.datasets import create_simple_dataset, create_causal_dataset
from icu_conformal_treatment.experiments import (
    run_simple_logreg_baseline,
    run_causal_tlearner_baseline,
    run_conformal_policy_alpha_sweep,
)


def ensure_datasets() -> None:
    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    simple_train = processed_dir / "simple_train.parquet"
    causal_train = processed_dir / "causal_train.parquet"

    if not simple_train.exists():
        print("Simple dataset not found. Creating...")
        create_simple_dataset()
    else:
        print("Simple dataset already exists.")

    if not causal_train.exists():
        print("Causal dataset not found. Creating...")
        create_causal_dataset()
    else:
        print("Causal dataset already exists.")


def main() -> None:
    ensure_datasets()

    print("\nRunning simple logistic regression baseline...")
    simple_metrics = run_simple_logreg_baseline()
    for k, v in simple_metrics.items():
        print(f"{k}: {v}")

    print("\nRunning causal T-learner logistic baseline...")
    causal_metrics = run_causal_tlearner_baseline()
    for k, v in causal_metrics.items():
        print(f"{k}: {v}")

    print("\nRunning conformal policy alpha sweep...")
    df_sweep = run_conformal_policy_alpha_sweep([0.05, 0.1, 0.2, 0.3, 0.4])
    print(df_sweep)


if __name__ == "__main__":
    main()
