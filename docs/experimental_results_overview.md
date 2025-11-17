# Experimental Results Overview (Current State, Demo + Synthetic)

This document captures the current experimental picture before moving to full
MIMIC-IV v3.0. It is meant to feed directly into the later paper "Results"
section.

## 1. MIMIC Demo (Binary Mortality, Logistic T-Learner)

Setup: small demo cohort from MIMIC-IV demo, binary treatment indicator
(constructed as in the code), binary outcome (hospital mortality), and a T-learner
with logistic regression per arm. Conformal intervals are built on label-level
residuals.

Key alpha sweep (see `conformal_policy_alpha_sweep.parquet`):

- Test size: n_test = 26
- At alpha = 0.10:
  - q0 ≈ 0.052, q1 ≈ 0.830
  - mean_width_t0 ≈ 0.094, mean_width_t1 ≈ 0.906
  - coverage_factual ≈ 0.846
  - auc_factual ≈ 0.609
  - decision counts:
    - n_decide_t0 = 1
    - n_decide_t1 = 2
    - n_abstain = 23
  - regret among decided patients:
    - mean_regret_decided = 0.0
    - p90_regret_decided = 0.0
    - max_regret_decided = 0.0

Interpretation:

- On this tiny test set, the dominance-based conformal policy makes very few
  recommendations and abstains on most patients, but whenever it acts its
  empirical regret (relative to the better arm’s predicted risk) is zero.
- This supports the qualitative "safe but abstaining" behavior of the policy,
  but the demo sample size is too small to be scientifically convincing.

## 2. Synthetic Binary World (Label-Level Conformal, Bernoulli Y)

Two variants:
- Logistic T-learner
- XGBoost T-learner

Both share the same synthetic DGP with known true risks r0_true(x), r1_true(x),
but outcomes are Bernoulli draws Y(t) ~ Bern(r_t(x)), and conformal intervals
are built at the label level.

Logistic T-learner (see `synth_conformal_alpha_sweep.parquet`):

- At alpha = 0.10:
  - q0 ≈ 0.625, q1 ≈ 0.641
  - mean_width_t0 ≈ 0.987
  - mean_width_t1 ≈ 0.963
  - coverage_factual ≈ 0.92
  - decision counts:
    - n_decide_t0 = 0
    - n_decide_t1 = 0
    - n_abstain = 1000
  - regret metrics: all NaN because there are no decided patients.

XGBoost T-learner (see `synth_conformal_alpha_sweep_xgboost.parquet`):

- At alpha = 0.10:
  - q0 ≈ 0.736, q1 ≈ 0.689
  - mean_width_t0 ≈ 0.980
  - mean_width_t1 ≈ 0.970
  - coverage_factual ≈ 0.899
  - decision counts:
    - n_decide_t0 = 0
    - n_decide_t1 = 0
    - n_abstain = 1000
  - regret metrics: again NaN due to no decisions.

Interpretation:

- With Bernoulli outcomes and label-level residuals, the conformal quantiles
  are large, forcing intervals to be close to [0, 1].
- As a consequence, the dominance rule almost never sees one interval strictly
  inside the other, leading to a policy that abstains on all patients.
- This is a *deliberate* negative result: it demonstrates that label-level
  conformal prediction for binary outcomes is too conservative for regret-bounded
  treatment decisions, independent of the base learner.

This supports the motivation to move towards risk-level or continuous-outcome
conformalization.

## 3. Synthetic Continuous World (Regression T-Learner, Risk-Level Surrogate)

Here the same causal structure is used, but Y is continuous in [0, 1] and
constructed as:
- Y(t) = r_t(x) + small Gaussian noise, clipped to [0, 1].

We fit a T-learner with linear regression per arm and build conformal intervals
on the prediction residuals. This approximates risk-level conformalization.

Alpha sweep (see `synth_continuous_alpha_sweep.parquet`):

At alpha = 0.10:

- Quantiles:
  - q0 ≈ 0.083
  - q1 ≈ 0.087
- Interval widths:
  - mean_width_t0 ≈ 0.166
  - mean_width_t1 ≈ 0.174
- Coverage:
  - coverage_factual ≈ 0.907
- Decision counts:
  - n_test = 1000
  - n_decide_t0 = 94
  - n_decide_t1 = 400
  - n_abstain = 506
- Regret based on model predictions:
  - mean_pseudo_regret_decided = 0.0
  - p90_pseudo_regret_decided = 0.0
  - max_pseudo_regret_decided = 0.0
- True regret using known r0_true, r1_true:
  - mean_true_regret_decided = 0.0
  - p90_true_regret_decided = 0.0
  - max_true_regret_decided = 0.0

Interpretation:

- At alpha = 0.10, the conformal policy:
  - Achieves close to nominal coverage for the continuous outcome.
  - Makes non-trivial decisions for roughly half the population.
  - Achieves zero true regret on all decided patients in this synthetic DGP,
    i.e., it never chooses the worse arm when it decides.
- This is the clean "sanity check" result that demonstrates the regret-bounded
  nature of the dominance-based conformal policy under ideal modeling and
  well-specified assumptions.

## 4. Current Narrative

Putting the three settings together:

1. MIMIC demo: real data, small, noisy; method behaves conservatively but shows
   zero empirical regret on the few decisions.
2. Synthetic Bernoulli world: highlights a fundamental limitation of
   label-level conformal prediction for binary outcomes: intervals covering
   [0, 1] almost always, zero decisions.
3. Synthetic continuous-risk world: shows the intended behavior of the method
   with non-trivial decisions and provably (by construction) zero regret on
   decided cases.

These results set up the need for:
- A risk-level or continuous proxy for mortality risk on real ICU data.
- A large-scale MIMIC-IV v3.0 experiment where we approximate the continuous
  risk with flexible models and apply the same dominance-based conformal policy.
