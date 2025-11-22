# ICU conformal treatment results summary

## MIMIC-IV v3, logistic T-learner, alpha=0.2

| group_type   | group_label   |   mean_width_t0 |   mean_width_t1 |   coverage_factual |   auc_factual |   n_test |   n_decide_t0 |   n_decide_t1 |   n_abstain |   mean_regret_decided |   p90_regret_decided |   max_regret_decided |
|:-------------|:--------------|----------------:|----------------:|-------------------:|--------------:|---------:|--------------:|--------------:|------------:|----------------------:|---------------------:|---------------------:|
| global       | all           |        0.202585 |        0.373898 |           0.795778 |      0.756563 |    13074 |          2829 |          6698 |        3547 |                     0 |                    0 |                    0 |

### Age subgroups (alpha=0.2)

| group_type   | group_label   |    n |   mean_width_t0 |   mean_width_t1 |   coverage_factual |   auc_factual |   n_test |   n_decide_t0 |   n_decide_t1 |   n_abstain |   mean_regret_decided |   p90_regret_decided |   max_regret_decided |
|:-------------|:--------------|-----:|----------------:|----------------:|-------------------:|--------------:|---------:|--------------:|--------------:|------------:|----------------------:|---------------------:|---------------------:|
| age_bin      | [0,40)        | 1367 |        0.20385  |        0.316908 |           0.917337 |      0.755386 |     1367 |           160 |           403 |         804 |                     0 |                    0 |                    0 |
| age_bin      | [40,60)       | 3475 |        0.199375 |        0.35599  |           0.876259 |      0.759713 |     3475 |           636 |          1652 |        1187 |                     0 |                    0 |                    0 |
| age_bin      | [60,80)       | 5789 |        0.200414 |        0.383575 |           0.786492 |      0.749246 |     5789 |          1543 |          2955 |        1291 |                     0 |                    0 |                    0 |
| age_bin      | [80,200)      | 2443 |        0.211586 |        0.408327 |           0.635284 |      0.677497 |     2443 |           490 |          1688 |         265 |                     0 |                    0 |                    0 |

### Gender subgroups (alpha=0.2)

| group_type   | group_label   |    n |   mean_width_t0 |   mean_width_t1 |   coverage_factual |   auc_factual |   n_test |   n_decide_t0 |   n_decide_t1 |   n_abstain |   mean_regret_decided |   p90_regret_decided |   max_regret_decided |
|:-------------|:--------------|-----:|----------------:|----------------:|-------------------:|--------------:|---------:|--------------:|--------------:|------------:|----------------------:|---------------------:|---------------------:|
| gender_dummy | gender_F      | 5661 |        0.202856 |        0.375354 |           0.789083 |      0.740775 |     5661 |          1200 |          2961 |        1500 |                     0 |                    0 |                    0 |
| gender_dummy | gender_M      | 7413 |        0.202377 |        0.372785 |           0.80089  |      0.768954 |     7413 |          1629 |          3737 |        2047 |                     0 |                    0 |                    0 |

### Careunit subgroups (alpha=0.2)

| group_type   | group_label                                                     |    n |   mean_width_t0 |   mean_width_t1 |   coverage_factual |   auc_factual |   n_test |   n_decide_t0 |   n_decide_t1 |   n_abstain |   mean_regret_decided |   p90_regret_decided |   max_regret_decided |
|:-------------|:----------------------------------------------------------------|-----:|----------------:|----------------:|-------------------:|--------------:|---------:|--------------:|--------------:|------------:|----------------------:|---------------------:|---------------------:|
| careunit     | first_careunit_Cardiac Vascular Intensive Care Unit (CVICU)     | 2326 |        0.169524 |        0.380941 |           0.963457 |      0.759382 |     2326 |           862 |           131 |        1333 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Coronary Care Unit (CCU)                         | 1515 |        0.218767 |        0.37166  |           0.784818 |      0.689178 |     1515 |           257 |          1033 |         225 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Intensive Care Unit (ICU)                        |    2 |        0.117241 |        0.462407 |           0        |      1        |        2 |             2 |             0 |           0 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Medical Intensive Care Unit (MICU)               | 2442 |        0.221504 |        0.384548 |           0.667486 |      0.664089 |     2442 |           315 |          1915 |         212 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Medical/Surgical Intensive Care Unit (MICU/SICU) | 2029 |        0.217152 |        0.380979 |           0.722523 |      0.697389 |     2029 |           421 |          1433 |         175 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Medicine                                         |    4 |        0.122201 |        0.40769  |           0.25     |    nan        |        4 |             2 |             0 |           2 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Neuro Intermediate                               |  862 |        0.164333 |        0.320044 |           0.980278 |      0.733623 |      862 |             9 |            89 |         764 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Neuro Stepdown                                   |  232 |        0.150689 |        0.272368 |           0.982759 |      0.803728 |      232 |             0 |             6 |         226 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Neuro Surgical Intensive Care Unit (Neuro SICU)  |  262 |        0.209456 |        0.42125  |           0.320611 |      0.711675 |      262 |            84 |           157 |          21 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Neurology                                        |    1 |        0.209034 |        0.462407 |           0        |    nan        |        1 |             0 |             1 |           0 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_PACU                                             |   10 |        0.226586 |        0.424647 |           0.6      |      0.666667 |       10 |             0 |            10 |           0 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Surgery/Vascular/Intermediate                    |   19 |        0.197386 |        0.399843 |           0.578947 |      0.566667 |       19 |             5 |             1 |          13 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Surgical Intensive Care Unit (SICU)              | 1803 |        0.207425 |        0.380887 |           0.793677 |      0.731898 |     1803 |           460 |          1022 |         321 |                     0 |                    0 |                    0 |
| careunit     | first_careunit_Trauma SICU (TSICU)                              | 1567 |        0.209894 |        0.367643 |           0.811742 |      0.737239 |     1567 |           412 |           900 |         255 |                     0 |                    0 |                    0 |