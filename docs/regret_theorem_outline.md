# Regret-Bounded Individualized Treatment: Theorem Outline (Draft)

This document sketches the theoretical target for the project: a finite-sample
regret bound for an individualized treatment rule built from conformal
potential-outcome intervals.

## 1. Setup

Data:
- Observations (X_i, T_i, Y_i) for i = 1, ..., n
- X_i ∈ ℝ^d: covariates
- T_i ∈ {0, 1}: binary treatment
- Y_i ∈ [0, 1] (in practice, {0, 1}, e.g., mortality outcome)

Potential outcomes:
- Y_i(0), Y_i(1) ∈ [0, 1]
- Observed: Y_i = Y_i(T_i)

Assumptions:
1. Unconfoundedness: (Y(0), Y(1)) ⫫ T | X
2. Overlap: ∃ ε > 0 s.t. ε ≤ P(T=1 | X=x) ≤ 1-ε for all x in support
3. Exchangeability / i.i.d. of samples

Define oracle risk:
- r_t(x) = E[Y(t) | X = x] for t ∈ {0, 1}
- Oracle best risk: r_*(x) = min{r_0(x), r_1(x)}

We want a policy π(x) ∈ {0, 1, abstain} with **regret**:
R(x) = r_{π(x)}(x) - r_*(x)
(where by convention r_abstain(x) is not directly compared; abstention means
the model does not override a baseline/clinician policy).

Our goal: control P(R(X) > ε) or E[R(X)] in finite samples.

## 2. Conformal Potential-Outcome Intervals

We fit outcome models m̂_t using only training data with T=t and get predicted
risks p̂_t(x) ≈ r_t(x). For calibration data (X_i, T_i, Y_i):

- For t ∈ {0,1}, define residual scores on calibration points with T_i = t:
  s_i^(t) = |Y_i - p̂_t(X_i)|.

Let q_t(α) be the (1 - α)-quantile of {s_i^(t)}. For a new X:

- Define intervals:
  L_t(X) = clip(p̂_t(X) - q_t(α), 0, 1)
  U_t(X) = clip(p̂_t(X) + q_t(α), 0, 1)

Under standard conformal assumptions:
P( Y(t) ∈ [L_t(X), U_t(X)] ) ≥ 1 - α (marginally over X and data split).

## 3. Dominance-Based Decision Rule

Risk interpretation: Y(t) is a loss (e.g. mortality). Lower is better.

Define policy π_α(X):

- If U_1(X) < L_0(X): π_α(X) = 1
- Else if U_0(X) < L_1(X): π_α(X) = 0
- Else: π_α(X) = abstain

In words: choose treatment whose **worst-case risk** (upper bound) is strictly
below the **best-case risk** (lower bound) of the alternative. Otherwise abstain.

This is a conservative dominance rule.

## 4. Candidate Regret Statement (High-Level)

We want statements of the form (informal):

> For a fresh test point X (and associated potential outcomes Y(0), Y(1)),
> with probability at least 1 - δ over the training/calibration data and X,
> the regret of π_α satisfies:
>
> R(X) ≤ g(α, n_train, n_calib) on all non-abstaining points,
> and P(X is non-abstaining) ≥ h(α, n_train, n_calib).

Where:
- g should shrink as calibration size grows and models get better;
- h captures the trade-off between **coverage** (α) and **decision rate**.

More concretely, for decided points:

- On the event that both Y(0) and Y(1) lie inside their conformal intervals:
  - If π_α(X) = 0, then:
      U_0(X) < L_1(X) ≤ Y(1) ⇒ Y(0) ≤ Y(1) - margin
    and similarly for π_α(X) = 1.
  - Hence, regret R(X) = Y(π_α(X)) - min{Y(0), Y(1)} is bounded by the maximal
    deviation between interval bounds and true risks.
- The main technical work is to convert conformal coverage on Y(t) into
  a bound on the probability that **both** Y(0), Y(1) are inside their intervals
  when we decide, and then bound R(X) on that event.

Target structure for a theorem (sketch):

**Theorem (Sketch).**
Under assumptions (unconfoundedness, overlap, exchangeability, regularity of
base models), for any α ∈ (0,1),

1. (Joint coverage on decided set): with high probability over the data split,
   P( {Y(0), Y(1)} ⊆ [L_0(X), U_0(X)] × [L_1(X), U_1(X)] ∣ X ∈ D_α ) ≥ 1 - C·α,
   where D_α = {x : π_α(x) ≠ abstain} and C is a constant depending on how we
   couple the two conformal procedures.

2. (Regret bound on decided set): on the event that both Y(0), Y(1) lie inside
   their intervals, R(X) can be bounded explicitly in terms of interval widths.
   E.g., for decided points,
   R(X) ≤ max{U_0(X) - L_0(X), U_1(X) - L_1(X)}.

3. (Trade-off): As α increases, intervals shrink, so the bound on R(X) tightens,
   but the joint coverage probability decreases, and the decision rate increases.
   The optimal α balances low regret and non-trivial decision rate.

This is the direction of the formal work: make each of these bullets precise and
prove them under clear assumptions (e.g., independent splits, mild regularity
on the base estimators, etc.).

## 5. Next Theoretical Tasks

1. Make the joint coverage part rigorous:
   - Option A: union bound over treatments (gives factor 2α).
   - Option B: more refined coupling if we reuse the same split.

2. Formalize the regret bound:
   - Express R(X) in terms of Y(t), r_t(X), and the conformal intervals.
   - Derive an upper bound involving interval widths and modeling error.

3. Connect empirical pseudo-regret (model-based) with the theoretical R(X):
   - When r̂_t ≈ r_t, pseudo-regret approximates regret.
   - Provide a decomposition: R(X) ≤ (interval-width term) + (model-bias term).

4. Extend the setup to:
   - Abstention with explicit baseline policy risk.
   - Dynamic or multi-time-step decisions (MDPs).
