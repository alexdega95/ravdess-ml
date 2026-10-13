# Binary SVM analysis

This preserves the original positive/negative emotion task. It uses the coherent
197-feature cleaned pair because the original SVM test file is absent.

## Task and data

Calm/happy map to 1; angry/disgust/fearful/sad map to 0. Neutral/surprised are
excluded. Training has 1,526 recordings; the actor-held-out test set has 528.
Both use the same 197 numeric features, with preprocessing fitted on training.

## Search and selection

Three LinearSVC C values, three gamma=auto kernel explorations, and all 28
combinations of the original C/kernel SVC grid were evaluated in five training-
actor folds. The majority baseline was included. Selection used mean accuracy.

The selected model is a linear-kernel SVC with C=0.001 and default gamma=scale.
Gamma does not affect the linear kernel. The saved fold scores are selection
statistics rather than an unbiased nested-CV estimate.

## Fixed test results

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority (negative) | 0.636364 | 0.388889 |
| Selected linear SVC | 0.679924 | 0.630563 |

Selected model ROC AUC: 0.711341, computed from decision scores for class 1.

## Training-only candidate scores

| Candidate | Mean accuracy | Mean macro F1 |
| --- | ---: | ---: |
| svc_grid_05 | 0.673210 | 0.550071 |
| linear_svc_C_0.001 | 0.664166 | 0.614330 |
| svc_grid_09 | 0.663596 | 0.601433 |
| svc_auto_rbf | 0.661726 | 0.576713 |
| svc_grid_19 | 0.661726 | 0.576713 |
| svc_grid_13 | 0.656592 | 0.607605 |
| svc_grid_27 | 0.653315 | 0.611943 |
| linear_svc_C_0.1 | 0.652444 | 0.608357 |
| svc_grid_23 | 0.650428 | 0.607298 |
| svc_grid_18 | 0.650096 | 0.523645 |
| svc_grid_16 | 0.647692 | 0.433207 |
| svc_auto_poly | 0.647234 | 0.518816 |
| linear_svc_C_1.0 | 0.646962 | 0.606211 |
| svc_auto_linear | 0.642559 | 0.599654 |
| svc_grid_17 | 0.642559 | 0.599654 |
| svc_grid_14 | 0.641292 | 0.408609 |
| svc_grid_21 | 0.639509 | 0.595193 |
| svc_grid_20 | 0.637872 | 0.579649 |
| svc_grid_10 | 0.637290 | 0.390716 |
| majority | 0.636719 | 0.389018 |
| svc_grid_01 | 0.636719 | 0.389018 |
| svc_grid_02 | 0.636719 | 0.389018 |
| svc_grid_03 | 0.636719 | 0.389018 |
| svc_grid_04 | 0.636719 | 0.389018 |
| svc_grid_06 | 0.636719 | 0.389018 |
| svc_grid_07 | 0.636719 | 0.389018 |
| svc_grid_08 | 0.636719 | 0.389018 |
| svc_grid_11 | 0.636719 | 0.389018 |
| svc_grid_12 | 0.636719 | 0.389018 |
| svc_grid_15 | 0.636719 | 0.389018 |
| svc_grid_25 | 0.634957 | 0.593373 |
| svc_grid_22 | 0.634261 | 0.582526 |
| svc_grid_26 | 0.628680 | 0.586163 |
| svc_grid_28 | 0.599014 | 0.545763 |
| svc_grid_24 | 0.598724 | 0.551919 |

## Warnings and provenance

9 warnings were recorded. See the full report for the candidate and fold.
LinearSVC convergence warnings are retained; its original 1,000-iteration limit
was not increased. The selected SVC is a separate estimator from LinearSVC.

Prediction CSVs contain filename, actor, original emotion, binary truth, and
predictions. Full settings, candidate parameters, class mapping, scores, warnings,
and input/source hashes are in [svm_results.json](svm_results.json). Local
`runs/svm-001/` also contains fitted pipelines and full fold row indices.
Those indices address retained task rows after the documented emotion exclusions,
in source order, rather than the unfiltered CSV row positions.

The old notebook shows roughly 68–72% accuracy across its saved SVM runs. Its
different/missing input pair and evaluation procedure prevent a controlled
numerical comparison. Binary accuracy must not be compared directly with the
eight-emotion boosting scores.
