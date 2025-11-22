# 4. Results

## 4.1 Synthetic calibration experiments

We first evaluated the proposed dominance-based conformal treatment policy on synthetic data where the true potential outcomes are known. In a binary-outcome setup, we generated paired potential risks \((r_0(x), r_1(x))\) and simulated factual outcomes under a fixed treatment assignment. A logistic T-learner was used to estimate the two potential outcome models, and residual-based nonconformity scores were calibrated separately for each arm.

Across miscoverage levels \(\alpha \in \{0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50\}\), the resulting marginal prediction intervals achieved empirical factual coverage close to the target \(1-\alpha\) and exhibited the expected decrease in mean interval width as \(\alpha\) increased. When we instantiated the dominance rule—treat with arm 1 if the upper bound for arm 1 lies below the lower bound for arm 0, and conversely for arm 0—the policy abstained whenever the intervals overlapped.

On this synthetic benchmark, the **true regret** (computed using the known \((r_0, r_1)\)) among patients for whom the policy made a decision was essentially zero across all \(\alpha\): the policy either chose the truly better arm or abstained. An analogous continuous-outcome experiment, using squared-error residuals and a regression T-learner, showed the same pattern: good coverage, shrinking intervals with larger \(\alpha\), and vanishing regret on the decided subset. These simulations validate that, under well-specified models, the dominance-based conformal policy behaves as intended.

## 4.2 ICU mortality under emergent vs non-emergent admission (MIMIC-IV v3)

We then applied the method to the MIMIC-IV v3 clinical database, focusing on adult ICU stays. For each ICU admission we constructed a feature vector including demographics (age, gender), administrative information (admission type and location, insurance, marital status, race), first ICU careunit, and simple length-of-stay statistics, and defined a binary outcome indicating in-hospital mortality. The “treatment” variable was a coarse indicator of admission urgency, equal to one for emergent or urgent admissions and zero for non-emergent types.

A logistic T-learner trained on 65k ICU stays achieved an AUROC of approximately **0.76** and a Brier score of **0.089** on the held-out test cohort of 13,074 stays. Despite its simplicity, this baseline provides reasonably calibrated risk estimates. We then calibrated conformal intervals for each treatment arm separately using a held-out calibration set, and applied the dominance policy on the test set.

Figure 1 (left) shows factual coverage as a function of the miscoverage level \(\alpha\). For both the logistic and XGBoost T-learners, the coverage curve closely follows the \(1-\alpha\) target, indicating that the conformal construction itself is robust to the choice of base model. The more interesting behaviour appears in the **decision rate** (Figure 1, right). For the logistic T-learner, the decision fraction is essentially zero for very small \(\alpha\), but rises sharply around \(\alpha = 0.20\): at \(\alpha = 0.20\) the policy issues a treatment recommendation for roughly **73%** of patients (2,829 assigned to non-emergent, 6,698 to emergent) and abstains on the remaining **27%** (3,547 patients). At this operating point, factual coverage remains high at **0.80**, and the empirical regret on the decided subset—measured as the absolute difference in estimated risk between the chosen arm and the best arm—is **0** (mean, 90th percentile, and maximum).

In contrast, an XGBoost T-learner trained on the same features achieved a higher AUROC of approximately **0.78**, but produced a far more conservative dominance policy. For \(\alpha \leq 0.10\), the XGBoost-based policy abstains on **all** test patients. Even at \(\alpha = 0.20\), it makes recommendations for only **23 out of 13,074** patients, abstaining on the rest, despite maintaining similar coverage. This illustrates a key point: **improving discrimination (AUROC) does not necessarily increase the number of confident treatment recommendations under a dominance rule**; more flexible models can produce wide, overlapping intervals that rarely dominate each other.

## 4.3 Subgroup analysis

To understand how the policy behaves across clinically relevant subgroups, we examined performance for the logistic T-learner at \(\alpha = 0.20\) along age, gender, and careunit dimensions.

### 4.3.1 Age bands

We split the test cohort into four age bands \([0, 40), [40, 60), [60, 80), [80, 200)\). In the youngest group, coverage was highest at **0.92**, but the policy decided for only about **41%** of patients, abstaining on the majority. As age increased, coverage decreased and the policy became more aggressive: coverage dropped to **0.88** and **0.79** in the [40, 60) and [60, 80) bands, with decision rates around **66–78%**. In the oldest group ([80, 200)), coverage fell further to **0.64**, while the policy issued recommendations for roughly **89%** of patients. Across all age bins, the model-based regret on the decided subset remained zero. This pattern suggests that older patients have more clearly separated risk profiles under the model, leading to more frequent dominance, but also that coverage guarantees are weaker in this high-risk group.

### 4.3.2 Gender

When stratifying by gender, behaviour was much more homogeneous. For female patients, coverage was **0.79** and the decision rate was **73%**, compared to **0.80** coverage and **72%** decision rate for male patients; AUROC was modestly higher in males (0.77 vs 0.74). In both groups, the regret among decided patients was zero. These findings indicate that, at least for this feature set and outcome, the dominance-based policy treats genders similarly in terms of how often it acts and how often it abstains.

### 4.3.3 ICU careunit

The largest heterogeneity appears across first ICU careunits. For example, in the Cardiac Vascular Intensive Care Unit, coverage was extremely high (**0.96**) but the policy abstained on a large fraction of patients, leading to a conservative operating mode. In Neuro Intermediate and Neuro Stepdown units, coverage was even higher (**≈0.98**) yet nearly all patients fell into the abstain region: intervals were tight but overlapping, so the dominance rule almost never triggered. In contrast, the Neuro Surgical Intensive Care Unit exhibited low coverage (**0.32**) and very few abstentions, indicating an aggressive, poorly calibrated regime. Mixed medical and surgical ICUs (MICU, combined medical–surgical ICU, and trauma SICU) fell in between, with moderate coverage (0.67–0.81) and high decision rates.

This heterogeneity suggests that **single global hyperparameters may not be appropriate across all care environments**, and that careunit-specific tuning or at least careunit-stratified reporting is important when deploying such policies. Overall, the subgroup analysis shows that the proposed dominance-based conformal policy can provide calibrated recommendations and principled abstentions globally, while exposing where its behaviour changes systematically across patient populations and clinical contexts.
