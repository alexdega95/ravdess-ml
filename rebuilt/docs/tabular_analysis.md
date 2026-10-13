# Cleaned-feature analysis results

This is a new analysis using the historical ensemble feature tables. It is not
a reproduction of the original dimensionality notebook. See `experiment_map.md`.

## Training-only method comparison

Five actor-disjoint folds; methods are ranked by mean validation macro F1.

| Method | Mean accuracy | Mean macro F1 | Macro F1 standard deviation |
| --- | ---: | ---: | ---: |
| random_forest | 0.411427 | 0.372248 | 0.019001 |
| decision_tree | 0.275554 | 0.265002 | 0.033411 |
| select_from_model | 0.277202 | 0.264877 | 0.031233 |
| select_k_best | 0.249202 | 0.226269 | 0.027104 |
| rfe | 0.209130 | 0.184074 | 0.015439 |
| pca | 0.178548 | 0.166366 | 0.025952 |
| majority | 0.154129 | 0.033386 | 0.000304 |

The random forest was selected before test evaluation. All methods use the same
folds. Standard deviation across five folds is not a confidence interval.

## Fixed test partition

| Method | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| majority | 0.153846 | 0.033333 |
| random_forest | 0.421474 | 0.394252 |

The selected method and majority baseline were evaluated on actors 19–24. Other
candidates were not evaluated on this test table during the comparison.
The test partition has been examined previously in the wider project.

## Data and provenance

Training: 1,809 recordings, actors 01–18. Testing: 624 recordings, actors 19–24.
Both use the same 197 numeric features in the historical ensemble column order.
No additional rows or features were removed. The cleaning implementation remains
unavailable, so its original methodological choices cannot be verified.

Full scores, fold actor membership, settings, input/source hashes, warnings and
named feature importances are in [tabular_results.json](tabular_results.json).
Local `runs/tabular-001/` also contains row-level fold indices, predictions, and
loadable models. All folds completed. SelectKBest recorded ten warnings across
the five folds: `zc_q75_w4` is constant, and its ANOVA calculation encounters an
undefined division. The input feature remains unchanged; warnings are preserved
in the report. The selected random forest's final fitting recorded no warnings.

The saved original ensemble random forest reported 42.31% accuracy; this seeded
cleaned-feature run reports 42.15%. These close values do not establish numerical
equivalence; the original estimator seed and software environment differ.
