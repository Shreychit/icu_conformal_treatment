# Synthetic Experiments: Design Notes

## 1. Observation from current synthetic setup

Setup:
- Binary outcome Y ∈ {0,1}.
- Logistic T-learner matches the data-generating process reasonably well.
- Conformal is applied directly to Y(t) via residuals |Y - p̂_t(X)| with alpha = 0.1.
- Intervals for Y(t) on test set are very wide:
  - Mean widths ≈ 0.98 for both T=0 and T=1.
  - Factual coverage ≈ 0.91.
  - Dominance-based rule abstains on essentially all patients.

This is expected:

- For a Bernoulli outcome, even a perfectly calibrated model cannot predict Y exactly.
- To guarantee high-probability coverage of Y itself, predictive intervals become
  almost [0,1] in the limit.
- With intervals that wide, the dominance condition (U_1 < L_0 or U_0 < L_1) is
  almost never satisfied.

Conclusion:
- Conformal prediction for the **label** Y(t) is intrinsically conservative in the
  binary case and tends to yield trivial intervals (close to [0,1]) for clinically
  realistic coverage levels.

## 2. Implication for our method

Our true target is regret in terms of **risk** r_t(x) = E[Y(t) | X=x], not in terms
of the instantaneous Bernoulli outcome.

Current status:
- We are using conformal guarantees on Y(t), then mapping those intervals into
  a dominance-based decision rule.
- This is conservative and can be trivial on synthetic data (no decisions).

Future direction:
- Move from conformal intervals on Y(t) to intervals (or calibrated bounds) on r_t(x).
  Potential approaches:
  - Risk-level conformal (e.g., regression-style conformal on empirical risk).
  - Cross-fitting or jackknife+ constructions that target r_t(x) directly.
  - Combine conformal with calibrated probability models (Platt / isotonic) and
    then reason about uncertainty in probabilities rather than labels.

For now:
- We keep the current label-level conformal module as the simplest baseline.
- Synthetic experiments are explicitly documented as showing that this baseline
  can be overly conservative in the binary setting, motivating a more refined
  risk-level treatment in the next iteration.
