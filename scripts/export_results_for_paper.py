from pathlib import Path

import pandas as pd

from icu_conformal_treatment.config import load_project_config


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def main() -> None:
    cfg = load_project_config()
    processed_dir = Path(cfg["data"]["processed_dir"])
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) MIMIC v3: alpha sweeps (logreg + XGB)
    logreg_m3 = pd.read_parquet(processed_dir / "mimic_v3_conformal_policy_alpha_sweep.parquet")
    xgb_m3 = pd.read_parquet(processed_dir / "mimic_v3_conformal_policy_alpha_sweep_xgboost.parquet")
    save(logreg_m3, results_dir / "mimic_v3_alpha_sweep_logreg.csv")
    save(xgb_m3, results_dir / "mimic_v3_alpha_sweep_xgboost.csv")

    # 2) Subgroup tables: age, gender, careunit
    #   Recompute quickly via the same logic as the scripts, but store as CSV.
    train = pd.read_parquet(processed_dir / "mimic_v3_causal_train.parquet")
    calib = pd.read_parquet(processed_dir / "mimic_v3_causal_calib.parquet")
    test = pd.read_parquet(processed_dir / "mimic_v3_causal_test.parquet")

    from icu_conformal_treatment.conformal import (
        tlearner_conformal_potential_outcomes,
        dominance_policy_metrics,
    )

    def split_X_ty(df: pd.DataFrame):
        X = df.drop(columns=["treatment", "y"])
        t = df["treatment"]
        y = df["y"]
        return X, t, y

    X_train, t_train, y_train = split_X_ty(train)
    X_calib, t_calib, y_calib = split_X_ty(calib)
    X_test, t_test, y_test = split_X_ty(test)

    alpha = 0.2
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
        score_type="residual",
    )

    global_metrics = dominance_policy_metrics(
        L0=res["L0"],
        U0=res["U0"],
        L1=res["L1"],
        U1=res["U1"],
        y_prob_test_t0=res["y_prob_test_t0"],
        y_prob_test_t1=res["y_prob_test_t1"],
        t_test=t_test,
        y_test=y_test,
    )
    global_df = pd.DataFrame([{"group_type": "global", "group_label": "all", **global_metrics}])
    save(global_df, results_dir / "mimic_v3_subgroup_global_alpha0.2.csv")

    age = X_test["anchor_age"]
    age_bins = [(0, 40), (40, 60), (60, 80), (80, 200)]
    age_rows = []
    for lo, hi in age_bins:
        mask = (age >= lo) & (age < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        m = dominance_policy_metrics(
            L0=res["L0"][mask.to_numpy()],
            U0=res["U0"][mask.to_numpy()],
            L1=res["L1"][mask.to_numpy()],
            U1=res["U1"][mask.to_numpy()],
            y_prob_test_t0=res["y_prob_test_t0"][mask.to_numpy()],
            y_prob_test_t1=res["y_prob_test_t1"][mask.to_numpy()],
            t_test=t_test[mask],
            y_test=y_test[mask],
        )
        age_rows.append({"group_type": "age_bin", "group_label": f"[{lo},{hi})", "n": n, **m})
    age_df = pd.DataFrame(age_rows)
    save(age_df, results_dir / "mimic_v3_subgroup_age_alpha0.2.csv")

    gender_cols = [c for c in X_test.columns if c.startswith("gender_")]
    g_rows = []
    for col in gender_cols:
        mask = X_test[col] == 1
        n = int(mask.sum())
        if n == 0:
            continue
        m = dominance_policy_metrics(
            L0=res["L0"][mask.to_numpy()],
            U0=res["U0"][mask.to_numpy()],
            L1=res["L1"][mask.to_numpy()],
            U1=res["U1"][mask.to_numpy()],
            y_prob_test_t0=res["y_prob_test_t0"][mask.to_numpy()],
            y_prob_test_t1=res["y_prob_test_t1"][mask.to_numpy()],
            t_test=t_test[mask],
            y_test=y_test[mask],
        )
        g_rows.append({"group_type": "gender_dummy", "group_label": col, "n": n, **m})
    gender_df = pd.DataFrame(g_rows)
    save(gender_df, results_dir / "mimic_v3_subgroup_gender_alpha0.2.csv")

    care_cols = [c for c in X_test.columns if c.startswith("first_careunit_")]
    cu_rows = []
    for col in sorted(care_cols):
        mask = X_test[col] == 1
        n = int(mask.sum())
        if n == 0:
            continue
        m = dominance_policy_metrics(
            L0=res["L0"][mask.to_numpy()],
            U0=res["U0"][mask.to_numpy()],
            L1=res["L1"][mask.to_numpy()],
            U1=res["U1"][mask.to_numpy()],
            y_prob_test_t0=res["y_prob_test_t0"][mask.to_numpy()],
            y_prob_test_t1=res["y_prob_test_t1"][mask.to_numpy()],
            t_test=t_test[mask],
            y_test=y_test[mask],
        )
        cu_rows.append({"group_type": "careunit", "group_label": col, "n": n, **m})
    care_df = pd.DataFrame(cu_rows)
    save(care_df, results_dir / "mimic_v3_subgroup_careunit_alpha0.2.csv")

    # 3) Synthetic sweeps
    synth_bin_logreg = pd.read_parquet(processed_dir / "synth_conformal_alpha_sweep.parquet")
    synth_bin_xgb = pd.read_parquet(processed_dir / "synth_conformal_alpha_sweep_xgboost.parquet")
    synth_cont = pd.read_parquet(processed_dir / "synth_continuous_alpha_sweep.parquet")
    save(synth_bin_logreg, results_dir / "synth_binary_alpha_sweep_logreg.csv")
    save(synth_bin_xgb, results_dir / "synth_binary_alpha_sweep_xgboost.csv")
    save(synth_cont, results_dir / "synth_continuous_alpha_sweep_logreg.csv")

    # 4) Minimal markdown summary for quick copy into a draft
    md_path = results_dir / "results_summary.md"
    with md_path.open("w") as f:
        f.write("# ICU conformal treatment results summary\n\n")
        f.write("## MIMIC-IV v3, logistic T-learner, alpha=0.2\n\n")
        f.write(global_df.to_markdown(index=False))
        f.write("\n\n### Age subgroups (alpha=0.2)\n\n")
        f.write(age_df.to_markdown(index=False))
        f.write("\n\n### Gender subgroups (alpha=0.2)\n\n")
        f.write(gender_df.to_markdown(index=False))
        f.write("\n\n### Careunit subgroups (alpha=0.2)\n\n")
        f.write(care_df.to_markdown(index=False))
    print(f"Saved markdown summary to {md_path}")


if __name__ == "__main__":
    main()
