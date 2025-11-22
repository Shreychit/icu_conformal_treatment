# Title

Dominance-Based Conformal Treatment Policies for ICU Admission Decisions

# Abstract

Decisions about admission urgency and escalation of care in the intensive care unit (ICU) are high-stakes and inherently uncertain. Standard risk models estimate the probability of adverse outcomes, but they do not indicate when recommendations are reliable or when the model should abstain. We propose a dominance-based conformal treatment policy that combines individualised risk estimation with finite-sample uncertainty quantification and an explicit abstain option.

Given features \(X\), a binary treatment \(T \in \{0,1\}\) (emergent vs non-emergent admission), and a binary outcome \(Y\) (in-hospital mortality), we first estimate arm-specific risks \(r_0(x), r_1(x)\) using a T-learner. We then construct conformal prediction intervals for each arm’s risk by calibrating residual-based nonconformity scores on a held-out set. A simple dominance rule recommends treatment 1 when the worst-case risk for arm 1 lies below the best-case risk for arm 0, recommends treatment 0 in the symmetric case, and abstains when intervals overlap. This yields a model-agnostic policy that acts only when it can provide distribution-free guarantees on relative risk.

On synthetic data with known potential outcomes, the proposed method achieves near-nominal coverage and essentially zero true regret among patients for whom it acts. On the MIMIC-IV v3 ICU cohort (65k stays), a logistic T-learner attains an AUROC of 0.76 and, at a miscoverage level \(\alpha = 0.20\), the dominance policy issues recommendations for approximately 73% of test patients while maintaining empirical coverage around 0.80 and zero model-based regret on the decided subset. In contrast, an XGBoost T-learner with higher AUROC (0.78) induces a far more conservative policy that abstains on almost all patients for practical values of \(\alpha\). Subgroup analyses along age, gender, and ICU careunit reveal systematic variation in coverage and decision rates, underscoring the importance of reporting how selective policies behave across clinically meaningful strata.

## Contributions

- We introduce a dominance-based conformal treatment policy that wraps any potential-outcome model and provides finite-sample guarantees on arm-specific risk, together with an explicit abstain option.

- We demonstrate, on synthetic data with known potential outcomes, that the policy achieves near-nominal coverage and essentially zero true regret among decided patients.

- We apply the method to the MIMIC-IV v3 ICU cohort, showing that a simple logistic T-learner can support confident recommendations for a large fraction of patients while maintaining high empirical coverage and zero model-based regret on decisions.

- We compare logistic and XGBoost T-learners and show that higher discrimination does not necessarily yield a more actionable dominance-based policy, and we characterise how coverage and decision behaviour vary across age, gender, and careunit subgroups.
