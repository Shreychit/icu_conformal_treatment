# Method Sketch: Regret-Bounded Individualized ICU Treatment Decisions

## 1. Setup and Notation

- Observations: (X_i, T_i, Y_i) for i = 1,...,n
  - X_i ∈ R^d: covariates (ICU cohort features)
  - T_i ∈ {0,1}: binary treatment (e.g., emergent vs non-emergent)
  - Y_i ∈ {0,1}: outcome (e.g., in-hospital mortality)
- Potential outcomes:
  - Y_i(0), Y_i(1) are the counterfactual outcomes under each treatment.
  - Observed outcome: Y_i = Y_i(T_i).

Assumptions (classical):
- Unconfoundedness: (Y(0), Y(1)) ⫫ T | X.
- Overlap: 0 < P(T=1 | X=x) < 1 for all relevant x.
- I.I.D. sampling.

Goal:
- For a new patient with covariates X = x, choose a treatment a(x) ∈ {0,1, abstain}
  with a finite-sample bound on regret:
  R(x) = Y(a(x)) - min{Y(0), Y(1)}.

## 2. Base Models (T-learner)

We fit separate outcome models for each treatment arm:

- μ̂_0(x) ≈ E[Y | X=x, T=0]
- μ̂_1(x) ≈ E[Y | X=x, T=1]

Implementation (current prototype):
- LogisticRegression on standardized X, trained separately on T=0 and T=1 subsets.
- This gives predicted risks p̂_t(x) ∈ [0,1] for t ∈ {0,1}.

These are *not* yet causal guarantees, but serve as the base scores.

## 3. Conformal Prediction for Potential Outcomes

We split the data into:
- Train: fit μ̂_0, μ̂_1
- Calibration: estimate nonconformity scores and quantiles
- Test: evaluate coverage, decision rules, and pseudo-regret

For each arm t ∈ {0,1}:

1. Fit μ̂_t on training data with T=t.
2. On calibration data with T=t, compute residual scores:
   s_i^(t) = |Y_i - μ̂_t(X_i)|.
3. Let q_t be the (1 - alpha) conformal quantile of {s_i^(t)}.

For a new x (test data), we define risk intervals:

- L_t(x) = clip( μ̂_t(x) - q_t, 0, 1 )
- U_t(x) = clip( μ̂_t(x) + q_t, 0, 1 )

These intervals target finite-sample marginal coverage for Y(t):
P( Y(t) ∈ [L_t(X), U_t(X)] ) ≥ 1 - alpha (under exchangeability).

## 4. Worst-Case Risk Decision Rule

We treat prediction as *risk* (probability of bad outcome, e.g., death).

For each x, we compute:

- [L_0(x), U_0(x)], [L_1(x), U_1(x)]

Decision rule (prototype):

- If U_1(x) < L_0(x): choose a(x) = 1
- Else if U_0(x) < L_1(x): choose a(x) = 0
- Else: a(x) = abstain

Interpretation:
- We only act when the worst-case risk under one treatment is strictly better than
  the best-case risk under the alternative.
- Otherwise, we abstain (defer to clinician / baseline policy).

This is a conservative “dominance under intervals” rule.

## 5. Pseudo-Regret Evaluation (Model-Based)

On test data, we define model-based risks:

- r̂_0(x) = μ̂_0(x)
- r̂_1(x) = μ̂_1(x)
- r̂_best(x) = min{r̂_0(x), r̂_1(x)}

For patients where the policy decides (not abstain), with decision a(x) ∈ {0,1}, we define:

- r̂_decision(x) = r̂_{a(x)}(x)
- pseudo-regret: r̂_decision(x) - r̂_best(x)

Current experiments:
- Mean, 90th percentile, and max pseudo-regret among decided patients.
- Decision rate (fraction of patients for whom we choose T=0, T=1) vs abstention rate.

These are **not yet formal regret theorems**, but serve as empirical diagnostics.

## 6. Alpha Sweep

We vary alpha ∈ [0.05, 0.4] and record:

- Conformal quantiles q_0, q_1
- Mean interval widths per arm
- Factual coverage on test
- Decision counts for T=0, T=1, abstain
- Pseudo-regret summaries for decided patients

On the demo dataset:
- Higher alpha → narrower intervals → lower coverage and higher decision rate.
- Conservative rule yields zero pseudo-regret on decided points under the model.

## 7. Next Steps (Toward Theory)

1. Replace simple residual scores with more principled nonconformity (e.g., negative log-likelihood).
2. Formalize a regret bound theorem relating:
   - Conformal coverage guarantees on Y(0), Y(1)
   - The dominance-based decision rule
   - An upper bound on E[R(X)] or P(R(X) > ε)
3. Introduce explicit abstention thresholds and potentially asymmetric risk preferences.
4. Transition from this toy demo to full MIMIC-IV v3.0 ICU cohort and a clinically meaningful treatment (e.g., steroid use in sepsis).

