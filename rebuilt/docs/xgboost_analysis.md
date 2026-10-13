# Eight-emotion XGBoost analysis

This uses the same 1,809 cleaned training recordings, 624 held-out test recordings,
and 197 numeric features as the rebuilt tabular comparison. All eight emotions
are retained. Binary SVM scores are a different task and not directly comparable.

## Search protocol

Ten distinct configurations are sampled from the original learning-rate, gamma
and exact/approximate tree-method space, with sampling seed 42. Each has 100
stages, depth 6, L2 penalty 1, multi:softmax objective and classifier seed 42.
Scaling and label encoding are fitted within each of five training-actor folds.
Selection uses mean validation accuracy; those scores are selection statistics,
not nested-CV estimates. This workflow evaluates test predictions only after
selection; the partition has been examined earlier in the wider project.

## Selected model

Learning rate 0.1, gamma 0.1, approximate tree method. The saved pipeline contains
its scaler and label encoder and predicts original emotion names. CPU execution
uses one thread per fit; validation uses two worker processes.

## Fixed test scores

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority class | 0.153846 | 0.033333 |
| Selected XGBoost | 0.448718 | 0.433518 |

## Training-only candidate scores

| Candidate | Mean accuracy | Mean macro F1 |
| --- | ---: | ---: | ---: |
| xgboost_02 | 0.419517 | 0.391485 |
| xgboost_01 | 0.414310 | 0.383918 |
| xgboost_06 | 0.413378 | 0.383935 |
| xgboost_10 | 0.410978 | 0.383247 |
| xgboost_03 | 0.409813 | 0.383526 |
| xgboost_05 | 0.407000 | 0.383899 |
| xgboost_04 | 0.385921 | 0.360139 |
| xgboost_08 | 0.380688 | 0.356391 |
| xgboost_09 | 0.375491 | 0.352499 |
| xgboost_07 | 0.367856 | 0.340495 |
| majority | 0.154129 | 0.033386 |

No warnings were recorded. Full parameters, fold actor membership, scores, named
feature importances, source/data hashes and environment are in
[xgboost_results.json](xgboost_results.json). Local `runs/xgboost-001/` also
contains row-level fold indices, predictions and fitted pipelines.

The original saved XGBoost accuracy was 0.44711538461538464; this run reports
0.44871794871794873. Their similar values do not establish equivalence: sampling,
validation grouping, selected settings and software versions may differ.
