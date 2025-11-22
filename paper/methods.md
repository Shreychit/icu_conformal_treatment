# 3. Methods

## 3.1 Problem setup

We consider a supervised treatment decision problem with features \(X \in \mathcal{X} \subset \mathbb{R}^d\), a binary treatment \(T \in \{0,1\}\), and an outcome \(Y\). In the main clinical application, \(Y \in \{0,1\}\) denotes in-hospital mortality, and \(T\) encodes a coarse notion of admission urgency (0 = non-emergent, 1 = emergent/urgent). For each patient we observe a single factual outcome
\[
Y = Y_T,
\]
but not the counterfactual \(Y_{1-T}\).

We work in the standard potential-outcomes framework and define the **potential risks**
\[
r_t(x) = \mathbb{E}[Y_t \mid X = x], \quad t \in \{0,1\},
\]
which represent the conditional risk of the outcome under each treatment arm. The object of interest is the individualised treatment effect
\[
\tau(x) = r_1(x) - r_0(x),
\]
but our goal is not just to estimate \(\tau(x)\): we wish to construct a **treatment policy** that can (i) recommend one of the two arms when the evidence is strong, and (ii) abstain when uncertainty is high, while providing distribution-free guarantees on prediction error.

Formally, a (possibly abstaining) policy is a mapping
\[
\pi : \mathcal{X} \to \{0,1,\bot\},
\]
where \(\pi(x) = t \in \{0,1\}\) indicates a recommendation of treatment \(t\), and \(\pi(x) = \bot\) indicates abstention.

## 3.2 T-learner for potential outcomes

We estimate the two potential-outcome functions \(r_0, r_1\) using a **T-learner**. Given a dataset of \(n\) i.i.d. samples \((X_i, T_i, Y_i)\), we form two treatment-specific datasets
\[
\mathcal{D}_t = \{(X_i, Y_i) : T_i = t\}, \quad t \in \{0,1\},
\]
and fit separate supervised learning models to each:

- For the main clinical experiments on MIMIC-IV v3, we use:
  - **Logistic regression** with \(\ell_2\) regularisation as a simple, interpretable baseline.
  - **XGBoost** gradient-boosted trees as a more flexible non-linear baseline.

- For synthetic experiments, we use logistic or linear regression depending on whether \(Y\) is binary or continuous.

Given a new covariate vector \(x\), the T-learner produces two risk estimates
\[
\hat{r}_0(x) \approx r_0(x), \qquad \hat{r}_1(x) \approx r_1(x).
\]

In all experiments, we construct a feature vector \(X\) from:

- Demographics: age (continuous), gender (one-hot encoded).
- Administrative variables: admission type, admission location, discharge location, insurance, marital status, race (all one-hot encoded).
- ICU context: first careunit and last careunit (one-hot encoded).
- Simple temporal descriptors: ICU length of stay and derived summary statistics.

For MIMIC-IV v3 we split the cohort into disjoint **train**, **calibration**, and **test** sets at the patient level. The resulting sizes are approximately 60%/20%/20%; in the final cohort used here this corresponds to 39,219 train, 13,073 calibration, and 13,074 test ICU stays.

## 3.3 Conformal risk intervals per treatment arm

To obtain uncertainty-quantified predictions for each treatment arm, we build **marginally valid prediction intervals** for the potential risks \(r_t(x)\) using conformal prediction.

### 3.3.1 Nonconformity scores

For each arm \(t \in \{0,1\}\), we restrict to the calibration subset with \(T_i = t\),
\[
\mathcal{C}_t = \{(X_i, Y_i) : T_i = t \text{ and } i \in \mathcal{I}_\text{calib}\}.
\]
We then define a **residual-based nonconformity score**
\[
S_t(X_i, Y_i) = \bigl|Y_i - \hat{r}_t(X_i)\bigr|.
\]

For binary outcomes, \(Y_i \in \{0,1\}\) and \(\hat{r}_t(X_i)\) denotes the predicted risk \(\mathbb{P}(Y=1 \mid X_i, T=t)\). For continuous-outcome synthetic experiments, \(Y_i \in \mathbb{R}\) and \(\hat{r}_t(X_i)\) is a real-valued regression prediction.

### 3.3.2 Calibration and interval construction

Fix a target miscoverage level \(\alpha \in (0,1)\). For each arm \(t\), we compute the empirical \((1-\alpha)\)-quantile of the scores
\[
q_t(\alpha) = \text{Quantile}_{1-\alpha} \{ S_t(X_i, Y_i) : (X_i, Y_i) \in \mathcal{C}_t \}.
\]
The conformal prediction interval for the arm-\(t\) risk at a new point \(x\) is then
\[
[L_t(x), U_t(x)] = \left[ \hat{r}_t(x) - q_t(\alpha),\; \hat{r}_t(x) + q_t(\alpha) \right],
\]
clipped to the feasible range of \(Y\). For binary outcomes we enforce
\[
L_t(x) \leftarrow \max\{0, L_t(x)\}, \qquad U_t(x) \leftarrow \min\{1, U_t(x)\},
\]
so that the interval lies in \([0,1]\). For continuous synthetic experiments we apply symmetric clipping as appropriate to the simulated outcome range.

Under standard exchangeability assumptions, this construction yields **marginal coverage guarantees**:
\[
\mathbb{P}\bigl( r_t(X) \in [L_t(X), U_t(X)] \bigr) \geq 1 - \alpha, \quad \text{for } t \in \{0,1\},
\]
where the probability is over the joint distribution of training, calibration, and test samples.

In practice, we instantiate this separately for each arm using the T-learner’s fitted models and the held-out calibration split.

## 3.4 Dominance-based treatment policy

Given per-arm risk intervals \([L_0(x), U_0(x)]\) and \([L_1(x), U_1(x)]\), we define a **dominance-based conformal treatment policy** \(\pi_\alpha\) that:

- recommends treatment 1 when arm 1 is uniformly safer than arm 0;
- recommends treatment 0 when arm 0 is uniformly safer than arm 1;
- abstains otherwise.

Concretely,
\[
\pi_\alpha(x) =
\begin{cases}
1, & \text{if } U_1(x) < L_0(x), \\
0, & \text{if } U_0(x) < L_1(x), \\
\bot, & \text{otherwise}.
\end{cases}
\]

Intuitively, the policy acts only when one arm’s **worst-case risk** is provably below the other arm’s **best-case risk**, according to the conformal intervals. When the intervals overlap, the evidence is insufficient to claim dominance, and the policy abstains.

### 3.4.1 Evaluation metrics

We evaluate the learned models and policies using several complementary metrics.

**Predictive performance.** For the underlying risk models, we report:

- Area under the ROC curve (AUROC) on the held-out test set.
- Brier score (mean squared error of predicted probabilities).

These characterise discrimination and calibration of the base T-learner.

**Conformal coverage and interval width.** For the conformal intervals, we track:

- **Factual coverage**, defined as the fraction of test points for which the observed outcome’s arm-specific risk lies inside the interval,
  \[
  \text{coverage}_\text{factual} = \frac{1}{n_\text{test}} \sum_{i \in \text{test}} \mathbf{1}\bigl( \hat{r}_{T_i}(X_i) \in [L_{T_i}(X_i), U_{T_i}(X_i)] \bigr).
  \]
  In synthetic experiments where ground-truth risks \(r_t(x)\) are available, we also measure coverage with respect to \(r_t(x)\) directly.

- **Mean interval width** for each arm,
  \[
  \text{width}_t = \frac{1}{n_\text{test}} \sum_{i \in \text{test}} \bigl( U_t(X_i) - L_t(X_i) \bigr).
  \]

**Decision fraction and abstention rate.** For the dominance policy \(\pi_\alpha\), we report:

- Total decision fraction,
  \[
  \text{decision\_frac} = \frac{1}{n_\text{test}} \sum_{i \in \text{test}} \mathbf{1}\bigl( \pi_\alpha(X_i) \in \{0,1\} \bigr),
  \]
  and the complementary abstention fraction.
- The split of decisions between the two arms:
  \[
  n_{\text{decide},t} = \sum_{i \in \text{test}} \mathbf{1}\bigl( \pi_\alpha(X_i) = t \bigr), \quad t \in \{0,1\}.
  \]

These quantities describe how frequently the policy acts at a given \(\alpha\).

**Regret.** To quantify the quality of decisions when the policy does act, we define a notion of regret.

- In **synthetic experiments**, where the true potential risks \(r_0(x), r_1(x)\) are known, the **true regret** on a decided point with \(\pi_\alpha(x) = t\) is
  \[
  \text{regret}_\text{true}(x) = r_t(x) - \min\{r_0(x), r_1(x)\} \ge 0.
  \]
  We summarise this over the decided subset via its mean, 90th percentile, and maximum.

- In **MIMIC-IV v3**, ground-truth potential outcomes are not observable, so we use a **pseudo-regret** based on the T-learner’s risk estimates. For a decided point with \(\pi_\alpha(x) = t\),
  \[
  \text{regret}_\text{pseudo}(x) = \hat{r}_t(x) - \min\{\hat{r}_0(x), \hat{r}_1(x)\}.
  \]
  This quantity is not a causal estimand, but serves as a consistency check: a well-behaved policy should rarely choose an arm with larger estimated risk than an available alternative.

We report these regret statistics only on the decided subset, i.e. among test points with \(\pi_\alpha(X_i) \in \{0,1\}\).

## 3.5 Experimental protocols

All experiments follow a consistent protocol:

1. **Data splitting.** We split the data into disjoint train, calibration, and test subsets at the patient level to avoid information leakage across stays belonging to the same individual. For MIMIC-IV v3 this yields 39,219 train, 13,073 calibration, and 13,074 test ICU stays.

2. **Model fitting.** We fit T-learner models (logistic regression and XGBoost) on the train split only. Hyperparameters are chosen via simple heuristics or default settings; our focus is on policy behaviour rather than marginal gains in AUROC.

3. **Conformal calibration.** Using the calibration split, we compute residual-based scores \(S_t(X_i, Y_i)\) separately for each arm and obtain empirical quantiles \(q_t(\alpha)\) for a grid of \(\alpha\) values.

4. **Policy evaluation.** On the held-out test split, we construct intervals \([L_t(X_i), U_t(X_i)]\), apply the dominance policy \(\pi_\alpha\), and compute coverage, interval widths, decision and abstention rates, and regret metrics globally and within subgroups (age bands, gender, and careunit indicators).

This design ensures that both conformal calibration and policy evaluation are carried out on data unseen by the T-learner during fitting, and that all reported results reflect out-of-sample performance.
