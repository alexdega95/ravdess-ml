# Eight-emotion boosting analysis

Both workflows use the same cleaned tables: 1,809 training recordings from
actors 01–18, 624 test recordings from actors 19–24, and 197 numeric features.
This is a different task from the two-class SVM analysis.

## Protocol

Each workflow samples ten configurations from its original notebook search space
using seed 42, then evaluates those configurations and the majority reference in
five training-actor folds. Scalers are fitted separately within each fold.
Selection uses mean validation accuracy. Only the selected candidate and the
majority reference are evaluated on the fixed test partition. These are
selection scores rather than unbiased nested-CV estimates. The test partition
has been examined in earlier project work.

## Selected settings

Gradient boosting: 100 stages, learning rate 0.1, depth 3, minimum split size 10,
classifier seed 0. XGBoost: 100 stages, learning rate 0.1, gamma 0.1, approximate
tree method, depth 6, L2 penalty 1, multi:softmax objective, classifier seed 42.
Both retain the original search spaces. Randomized sampling and validation
grouping are explicit new choices.

## Fixed test results

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority class | 0.153846 | 0.033333 |
| Gradient boosting | 0.459936 | 0.452043 |
| XGBoost | 0.448718 | 0.433518 |

## Gradient boosting candidate scores

| Candidate | Mean accuracy | Mean macro F1 |
| --- | ---: | ---: | ---: |
| gradient_boosting_05 | 0.408947 | 0.381168 |
| gradient_boosting_01 | 0.406569 | 0.384043 |
| gradient_boosting_09 | 0.403416 | 0.382059 |
| gradient_boosting_07 | 0.402899 | 0.377091 |
| gradient_boosting_08 | 0.399711 | 0.380707 |
| gradient_boosting_06 | 0.397510 | 0.372679 |
| gradient_boosting_03 | 0.395483 | 0.364938 |
| gradient_boosting_02 | 0.392030 | 0.374374 |
| gradient_boosting_10 | 0.345286 | 0.316591 |
| gradient_boosting_04 | 0.343728 | 0.317349 |
| majority | 0.154129 | 0.033386 |

XGBoost candidate scores and provenance are explained in
[its separate report](xgboost_analysis.md). Neither workflow recorded training
warnings. Saved models were reloaded and reproduced every stored test prediction.

Full settings, scores, fold actor membership, feature importances, warnings and
source/data hashes are in [gradient_boosting_results.json](gradient_boosting_results.json)
and [xgboost_results.json](xgboost_results.json). Local run folders also contain
full fold row indices, predictions and fitted pipelines.

## Historical comparison and remaining work

The original notebook reports roughly 44% gradient-boosting accuracy and
44.71% XGBoost accuracy. This rebuild reports 45.99% and 44.87%, respectively.
These are observed scores from different validation/sampling settings and
software versions, not proof of improvement or numerical equivalence.

LightGBM and CatBoost remain pending because their dependencies are unavailable.
Bagging and legacy AdaBoost are also retained as pending experiments; the older
AdaBoost default algorithm is not silently replaced by the current one.
