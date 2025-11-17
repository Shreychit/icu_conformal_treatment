# Paper Outline: Regret-Bounded Individualized ICU Treatment via Conformal Counterfactuals

## 1. Introduction

- Motivation:
  - ICU treatment decisions are high-stakes; wrong treatment can increase mortality.
  - Clinicians want **reliable treatment recommendations with explicit uncertainty**.
  - Most ML models output point risks without finite-sample guarantees or explicit
    regret control.
- Contribution (high-level):
  - Propose an individualized treatment rule that:
    - Produces **per-patient treatment intervals** for risk under each arm.
    - Makes a recommendation only when one treatment **dominates** the other under
      worst-case risk.
    - Comes with a finite-sample regret guarantee (on decided patients) under
      standard causal assumptions.
  - Empirical evaluation on:
    - Synthetic data with known ground-truth risks.
    - MIMIC-IV ICU cohorts (once full data is wired in).

## 2. Related Work (bullets to be expanded)

- Individualized treatment rules / policy learning.
- Conformal prediction for potential outcomes and treatment effects.
- Safe ML / abstention / selective prediction.
- Clinical ML with uncertainty in ICU settings.

## 3. Problem Setup

- Data: (X, T, Y), binary treatment, binary outcome (mortality) in ICU.
- Potential outcomes Y(0), Y(1); causal assumptions (unconfoundedness, overlap).
- Define oracle risks r_0(x), r_1(x) and regret R(x) = r_a(x) - min{r_0(x), r_1(x)}.
- Goal: construct π(x) ∈ {0, 1, abstain} with finite-sample regret guarantees on
  the decided set.

## 4. Method

### 4.1 Base outcome models (T-learner)

- Separate models μ̂_0, μ̂_1 for each treatment arm (currently logistic regression).
- Discussion of extensibility to richer models (XGBoost, neural nets).

### 4.2 Conformal potential-outcome intervals

- Split data into train, calibration, test.
- For each arm t:
  - Fit μ̂_t on train.
  - Compute nonconformity scores on calibration:
    - Residual or NLL-based scores.
  - Compute conformal quantiles q_t(α) and intervals [L_t(x), U_t(x)].

### 4.3 Dominance-based decision rule

- Risk-interpretation of Y; lower is better.
- Rule:
  - If U_1(x) < L_0(x): choose T=1.
  - Else if U_0(x) < L_1(x): choose T=0.
  - Else abstain.
- Discussion:
  - Conservative; focuses on cases where one arm strictly dominates in worst-case.
  - Connection to regret: if both Y(0), Y(1) lie inside their intervals, regret
    is bounded by interval widths.

### 4.4 Regret analysis sketch

- Summarize the theorem outline (see regret_theorem_outline.md).
- Highlight key ingredients:
  - Conformal coverage on Y(t) or risk-level quantities.
  - Joint coverage on both arms.
  - Bounding regret on the decided set.

## 5. Experiments

### 5.1 Synthetic experiments (ground truth known)

- Data-generating process:
  - X ~ N(0, I), treatment propensities, and separate risk functions r_0(x), r_1(x).
- Evaluate:
  - Coverage of intervals.
  - Decision vs abstention rate.
  - **True regret** using r_0(x), r_1(x).
- Ablations:
  - Vary α.
  - Vary strength of treatment effect and overlap.

### 5.2 MIMIC-IV ICU demo (current state)

- Cohort definition on demo data:
  - Basic ICU inclusion criteria.
- Task:
  - Mortality prediction with emergent vs non-emergent admission as a proxy treatment.
- Metrics:
  - Baseline predictors (logistic regression).
  - Causal T-learner performance.
  - Conformal coverage, decision rate, pseudo-regret.

### 5.3 Full MIMIC-IV v3.0 study (future work)

- Replace proxy treatment with clinically meaningful intervention (e.g., steroids).
- Larger, realistic cohort; multiple sensitivity analyses.

## 6. Discussion and Limitations

- Limitations:
  - Binary-outcome conformal on Y(t) can yield conservative intervals and high abstention.
  - Dependence on unconfoundedness and overlap; impact of unmeasured confounding.
  - Model misspecification in μ̂_t.
- Extensions:
  - Risk-level conformal (bounds on r_t(x) instead of Y(t)).
  - Multi-arm treatments and time-varying decisions.
  - Incorporating cost-sensitive regret and asymmetric risk preferences.

## 7. Conclusion

- Summary of method and guarantees.
- Emphasis on per-patient regret control and abstention.
- Outlook toward deployment and more realistic ICU treatment choices.
