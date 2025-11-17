# Paper Outline: Conformal Counterfactual Policies for ICU Treatment

Working title (to refine later):

> Regret-Bounded ICU Treatment Recommendations via Conformal Counterfactual Policies

## 1. Introduction

- Clinical motivation:
  - ICU treatment decisions (e.g., steroids in sepsis, vasopressor strategies)
    involve high-stakes trade-offs under uncertainty.
  - Black-box risk scores are increasingly used but rarely communicate their
    uncertainty or worst-case consequences for alternative treatments.
- Methodological gap:
  - Standard causal inference methods estimate average treatment effects or
    conditional treatment effects but do not provide **individual-level**,
    distribution-free guarantees on regret if their recommendations are followed.
- Our contribution (high-level):
  - Propose a dominance-based conformal counterfactual policy that:
    - Predicts risk under multiple treatments.
    - Constructs prediction intervals for each potential outcome.
    - Acts only when one treatment’s interval is uniformly better than another’s.
    - Abstains otherwise.
  - Provide finite-sample regret bounds under standard causal assumptions.
  - Demonstrate the behavior in synthetic worlds and on ICU data derived from
    MIMIC-IV, using mortality as the primary endpoint.

## 2. Problem Setup and Notation

- Observational data:
  - Patients indexed by i with covariates X_i, binary treatment T_i ∈ {0, 1},
    and outcome Y_i (mortality or continuous risk proxy).
- Potential outcomes:
  - Y_i(0), Y_i(1); risk functions r_t(x) = E[Y(t) | X = x].
- Goal:
  - Construct a policy π(x) ∈ {0, 1, abstain} that trades off:
    - low expected regret compared to the optimal treatment at x
    - versus coverage / abstention.

## 3. Method

3.1 Base T-learner models

- Separate models μ̂_0, μ̂_1 for each treatment arm.
  - In practice: logistic regression, XGBoost, or regression models (for risk
    proxies) trained on ICU features.
  - Discuss identifiability assumptions (unconfoundedness, positivity) and the
    role of rich covariate sets.

3.2 Conformal prediction for potential outcomes

- Construction of nonconformity scores and conformal quantiles:
  - Residual-based scores for either binary labels or continuous outcomes.
  - Alpha parameter α controlling error probability and interval width.
- For each x:
  - Construct [L_t(x), U_t(x)] for t ∈ {0, 1} using calibration data.

3.3 Dominance-based policy

- Decision rule:
  - If U_1(x) < L_0(x), recommend T = 1.
  - If U_0(x) < L_1(x), recommend T = 0.
  - Otherwise, abstain.
- Interpretation:
  - When we act, we have high-confidence that the chosen treatment dominates
    the alternative in terms of worst-case risk within the conformal intervals.
- Regret theorem (sketch):
  - Show that, under correct specification and standard assumptions, the
    probability that the policy chooses a treatment with regret exceeding a
    threshold is bounded by α (plus modeling error terms).

## 4. Synthetic Experiments

4.1 Binary outcome world (Bernoulli mortality)

- Construction:
  - Features X, treatment assignment via logistic propensity, true risks r0_true(x),
    r1_true(x), Bernoulli outcomes Y(t) ~ Bern(r_t(x)).
- Findings:
  - Label-level conformal intervals are extremely wide (~[0, 1]).
  - Dominance rule rarely fires; empirical decision rate is essentially zero across
    α ∈ [0.01, 0.5].
- Message:
  - Demonstrates intrinsic conservatism of label-level conformal prediction for
    binary outcomes in this setting.
  - Motivates moving to risk-level or continuous surrogates.

4.2 Continuous-risk world

- Construction:
  - Same DGP but define Y(t) = r_t(x) + small Gaussian noise, clipped to [0, 1].
- Models:
  - Regression T-learner with linear models per arm.
- Results (at alpha ≈ 0.10, representative):
  - Coverage around 0.90.
  - Non-trivial decisions on roughly half the population.
  - Empirically zero regret on decided patients when evaluated against the
    known true risks r0_true(x), r1_true(x).
- Message:
  - Illustrates the intended regret-bounded behavior when the method is applied
    to continuous risk estimates rather than Bernoulli labels.

## 5. ICU Case Study (MIMIC-IV v3.0)

- Cohort design:
  - Adult ICU patients, first ICU stay, LOS ≥ 24 hours, etc. (as per
    `mimic_v3_experiment_design.md`).
- Treatment of interest:
  - Example: early systemic corticosteroids in suspected sepsis/shock.
- Outcome:
  - In-hospital mortality (primary), with possible continuous-risk proxies
    derived from predictive models.
- Covariates:
  - Demographics, admission context, baseline severity, organ support,
    comorbidities, and treatment-related confounders.
- Evaluation:
  - Baseline risk models (AUROC, calibration).
  - Conformal potential-outcome intervals.
  - Dominance-based policy decision rates and empirical performance across
    clinically relevant subgroups.

## 6. Discussion

- Pros:
  - Individual-level, distribution-free guarantees on worst-case treatment
    regret (up to modeling assumptions).
  - Explicit abstention region where data are insufficient to recommend.
- Cons and limitations:
  - Potential conservatism, particularly for binary endpoints when using
    label-level conformalization.
  - Dependence on unconfoundedness and representativeness of observational data.
- Future work:
  - Better risk surrogates for binary mortality (e.g., survival models).
  - Extensions to multi-arm treatments and time-varying policies.
  - Integration with clinician-facing interfaces that surface intervals and
    abstentions in an interpretable way.

