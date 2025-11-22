# 1. Introduction

Decisions about when and how aggressively to treat critically ill patients are among the most consequential in medicine. In the intensive care unit (ICU), clinicians routinely face choices such as whether to admit a patient urgently versus non-emergently, escalate or de-escalate care, or transfer between units. These decisions occur under substantial uncertainty, yet the costs of mistakes are asymmetric: overly aggressive treatment may expose patients to unnecessary risk and resource use, while overly conservative decisions may delay life-saving interventions.

Over the last decade, machine learning has been widely explored for ICU risk prediction, triage, and decision support. Many models achieve high discrimination for outcomes such as mortality or prolonged length of stay, and are sometimes used to rank or prioritise patients. However, standard predictive models produce **point estimates** or single risk scores. They do not indicate when the model is uncertain, nor do they provide guarantees about the quality of recommendations. In high-stakes settings, a model that is unable to say “I do not know” can be more dangerous than helpful.

At the same time, there is growing interest in **individualised treatment effect** estimation and policy learning from observational data. T-learners, uplift models, and related methods can estimate how risk might change under different treatment choices. Yet such methods typically output a single recommended arm (for example, “treat” vs “do not treat”) without calibrated uncertainty. Clinicians are left without a principled way to know when a recommendation is reliable, and when the model should instead defer to human judgment.

In this work, we combine ideas from treatment effect estimation, conformal prediction, and selective classification to construct an ICU treatment policy that **acts only when it can provide distribution-free guarantees on risk** and otherwise abstains. Concretely, we consider a binary treatment \(T \in \{0,1\}\) (in our application, a coarse notion of emergent vs non-emergent admission) and a binary outcome \(Y\) (in-hospital mortality). We estimate potential-outcome risks \(r_0(x), r_1(x)\) using a T-learner, then build **conformal prediction intervals** for each arm’s risk. A dominance-based policy recommends the arm whose worst-case risk is provably lower than the other’s best-case risk; if the conformal intervals overlap, the policy abstains.

This construction yields a treatment policy with several desirable properties:

- It is **model-agnostic**: any base learner capable of predicting risk can be wrapped into the conformal procedure.
- It provides **finite-sample coverage guarantees** for the potential-outcome risks, under standard exchangeability assumptions.
- It produces a **selective policy** that knows when to abstain; the abstention rate and decision coverage can be tuned by a single parameter \(\alpha\).
- It exposes how policy behaviour varies across patient subgroups (e.g., age, gender, ICU careunit), which is critical for fairness and deployment.

We evaluate the proposed approach on both synthetic data, where the true potential outcomes are known, and on the MIMIC-IV v3 ICU cohort, focusing on mortality under emergent versus non-emergent admission. Our empirical study leads to several observations:

1. On synthetic benchmarks, the dominance-based conformal policy achieves near-nominal coverage and essentially zero regret among patients for whom it acts, validating the method in a controlled setting.

2. On MIMIC-IV v3, a simple logistic T-learner coupled with the dominance policy can make confident recommendations for a large fraction of ICU stays while maintaining high empirical coverage and low estimated regret.

3. A more flexible XGBoost T-learner attains higher AUROC but, under the same dominance rule, is dramatically more conservative: it abstains on almost all patients for practical values of the miscoverage parameter \(\alpha\). This illustrates that improving discrimination does not automatically translate into more actionable decisions under calibrated uncertainty.

4. Subgroup analyses along age, gender, and careunit reveal systematic heterogeneity in coverage and decision rates, underscoring the importance of reporting not only global metrics but also how abstaining policies behave across clinically meaningful strata.

Overall, our results suggest that conformal prediction can provide a practical, distribution-free layer of uncertainty quantification on top of standard ICU risk models, enabling treatment policies that act cautiously and transparently in the presence of uncertainty. Rather than replacing clinical judgment, such policies can highlight when the data strongly support one course of action and when the model itself recommends deferring the decision.
