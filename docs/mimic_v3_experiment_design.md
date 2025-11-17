# MIMIC-IV v3.0 ICU Experiment Design (Draft)

This document fixes the *clinical* side of the main real-world experiment.
Once full MIMIC-IV v3.0 access is available, the code will implement this design.

The goal: evaluate a **regret-bounded ICU treatment policy** for a clinically
meaningful binary treatment, using our conformal counterfactual framework.

---

## 1. Cohort Definition (ICU Population)

Base idea: adult ICU patients admitted with sepsis-like conditions, where a
specific therapy is variably used in practice.

### 1.1 Inclusion Criteria

- Age ≥ 18 at ICU admission.
- First ICU stay for a given hospital admission.
- ICU length-of-stay (LOS) ≥ 24 hours
  - Avoid ultra-short stays / immediate transfers.
- Sufficient covariate information available (labs, vitals, comorbidities).

### 1.2 Exclusion Criteria

- Organ donation / comfort-measures-only cases at admission (if identifiable).
- Missing key lab or vital-sign features (beyond tolerable thresholds).
- Outliers with ICU LOS > 60 days (data quality edge cases).

---

## 2. Treatment Definition (T)

We need a **binary treatment** that:

1. Has clinical debate / heterogeneity (not universally given).
2. Is plausibly causally analyzable from observational EHR data.
3. Has clear timing relative to ICU admission.

Candidate (to be finalized when digging into MIMIC v3.0):

### Option A – Early systemic steroids in suspected sepsis/shock

- T = 1:
  - Patient received systemic corticosteroids (e.g., hydrocortisone, methylpred)
    within the first 24 hours of ICU admission.
- T = 0:
  - Patient did not receive systemic steroids in the first 24 hours.

Justification:
- Steroid use in sepsis/septic shock is common but not universal.
- There is existing literature with mixed evidence → realistic, non-trivial target.
- Timing-based definition (first 24h) makes treatment assignable relative to covariates.

Alternative options (if steroids are messy in practice):

- Early vasopressor strategy (e.g., norepinephrine within X hours).
- Early vs late invasive mechanical ventilation in patients with respiratory failure.
- Early renal replacement therapy in acute kidney injury.

For now, the **default plan** for the project is: **steroids in early sepsis**.

---

## 3. Outcome Definition (Y)

Primary outcome:

- In-hospital mortality (binary):
  - Y = 1 if patient dies before hospital discharge.
  - Y = 0 otherwise.

Secondary / sensitivity outcomes (for robustness once primary is working):

- 28-day mortality (if can be approximated from MIMIC timestamps).
- ICU mortality.
- ICU length-of-stay (treated as continuous; could be a secondary conformal target).

For the main conformal regret analysis, we stick to **binary mortality Y ∈ {0,1}**.

---

## 4. Covariates (X) and Confounding Story

We must include sufficient covariates to make unconfoundedness plausible.

### 4.1 Demographics

- Age (continuous, maybe spline-transformed).
- Sex.
- Race / ethnicity (with careful handling, primarily as confounder).

### 4.2 Admission / Context

- Admission type (elective vs emergency/urgent).
- Admission location (ED, floor, transfer from another hospital).
- First ICU care unit (medical, surgical, cardiac, neuro, etc.).
- Calendar year / era (to capture practice changes).

### 4.3 Baseline Severity

At or close to ICU admission time (e.g., first 24 hours):

- Vital signs:
  - Heart rate, MAP, SBP/DBP, respiratory rate, temperature, SpO2.
- Laboratory values:
  - Lactate, creatinine, BUN, bilirubin, platelets, WBC, sodium, potassium.
- Organ support:
  - Use of mechanical ventilation at admission.
  - Use of vasopressors at admission.
- Global severity scores (if available):
  - SAPS II, SOFA components, etc.

### 4.4 Comorbidities

- Chronic conditions derived from diagnosis codes (e.g., Elixhauser or Charlson):
  - CHF, COPD, diabetes, chronic kidney disease, malignancy, etc.

### 4.5 Treatment-Related Confounders

For steroids-in-sepsis design, covariates that correlate with both:
- the *decision* to give steroids and
- the risk of death:

- Shock indicators: lactate, MAP, vasopressor dose.
- Respiratory failure severity.
- Renal/liver dysfunction at baseline.

These should be in X wherever possible.

---

## 5. Data Splits and Construction

We reuse the split logic already implemented for the demo, but applied to the
full v3.0 cohort.

### 5.1 Patient-Level Splits

- Split on **unique patients** to avoid leakage:
  - Train ~ 60%
  - Calibration ~ 20%
  - Test ~ 20%

### 5.2 T-learner Causal Dataset

For each patient/ICU stay:

- X: engineered covariate vector as above.
- T: steroid treatment indicator (within 24h ICU admission).
- Y: in-hospital mortality.

Store:

- `causal_train.parquet`
- `causal_calib.parquet`
- `causal_test.parquet`

Same structure as the demo causal dataset (X columns, `treatment`, `hospital_expire_flag`).

---

## 6. Evaluation Plan on MIMIC v3.0

We will replicate the **exact pipeline** we already implemented:

1. **Baseline predictive model** (non-causal):
   - Logistic regression or XGBoost predicting Y from X alone (no T).
   - Metrics: AUROC, AUPRC, calibration curves.

2. **T-learner outcome models**:
   - Separate models μ̂_0, μ̂_1 for each arm.
   - Metrics:
     - AUROC on factual outcomes per arm.
     - Group-wise calibration.

3. **Conformal potential-outcome intervals**:
   - Label-level conformal on Y(t) as baseline (exactly what we have now).
   - Coverage on test for factual Y.
   - Interval width distribution.

4. **Dominance-based policy**:
   - Compute [L_0, U_0], [L_1, U_1] on test.
   - Report:
     - Decision counts (T=0, T=1, abstain).
     - Factual mortality for decided vs abstained patients.
     - Pseudo-regret summarised using μ̂_t risk as oracle.

5. **Sensitivity Analyses** (once basic pipeline works):
   - Vary α (coverage level) to trace decision rate vs regret.
   - Subgroup analyses (e.g., high lactate, older patients, specific ICU types).
   - Alternative treatments if steroids prove too noisy.

---

## 7. Connection to Theory

The MIMIC v3.0 experiment instantiates the theoretical objects:

- X: high-dimensional covariates summarizing patient state.
- T: real ICU treatment (steroids).
- Y: mortality.
- μ̂_t: data-driven approximations to r_t(x).
- Conformal intervals: finite-sample approximate coverage for Y(t) (baseline), later to be extended toward risk r_t(x).
- Policy π_α: dominance-based treatment recommendation with abstention.

The main **scientific story**:

> On a large, real ICU cohort, we show that a conformal counterfactual policy can
> provide individually tailored treatment recommendations with explicit
> worst-case risk intervals and abstention, and empirically small regret on
> patients where it chooses to act.

