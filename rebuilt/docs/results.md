# Results and experiment status

This page consolidates existing reports. No models were trained to generate it. Original notebook outputs, completed rebuilt experiments, and incomplete runs remain distinct.

## Completed rebuilt experiments

All classification scores below are fixed-test accuracy and macro F1, not validation scores. Actors 01–18 supply training data; actors 19–24 supply test data. Candidate selection uses training actors only, where applicable. The test set was already examined in the academic project; these are not fresh independent benchmarks. Standalone artifact-verification reports are linked in the machine-readable index where available; a null verification reference means no standalone verification report is included. The recorded checks were performed when the runs completed, not rerun for this documentation update. [Project guide and methodology](../../README.md).

### Eight-class emotion classification

| Experiment | Inputs | Final fit / test rows | Test accuracy | Macro F1 | Report |
| --- | --- | ---: | ---: | ---: | --- |
| Raw-feature random forest | 425 numeric | 1828 / 624 | 44.23% | 0.4155 | [Saved report](baseline_results.json) |
| Cleaned-feature random forest | 197 numeric | 1809 / 624 | 42.15% | 0.3943 | [Saved report](tabular_results.json) |
| Gradient boosting | 197 numeric | 1809 / 624 | 45.99% | 0.4520 | [Saved report](gradient_boosting_results.json) |
| XGBoost | 197 numeric | 1809 / 624 | 44.87% | 0.4335 | [Saved report](xgboost_results.json) |
| Dense network: dropout variant | 202 numeric/categorical | 1395 / 624 | 51.28% | 0.5027 | [Saved report](neural_results.json) |
| Dense network: parameter search | 202 numeric/categorical | 1809 / 624 | 51.44% | 0.5064 | [Saved report](neural_search_results.json) |
| Waveform Euclidean 1-NN | raw [45000:100000] | 1828 / 624 | 12.82% | 0.0920 | [Saved report](timeseries_emotion_results.json) |

The dropout variant uses fourteen fitting actors (1,395 rows) and four validation actors (414 rows). The searched network refits on all eighteen training actors (1,809 rows). Different features and fitting protocols prevent treating this table as a controlled leaderboard. The raw-feature majority reference scores 15.38% accuracy and 0.0333 macro F1.

### Binary emotion task

| Experiment | Inputs | Final fit / test rows | Test accuracy | Macro F1 | Report |
| --- | --- | ---: | ---: | ---: | --- |
| Binary linear SVM | 197 numeric | 1526 / 528 | 67.99% | 0.6306 | [Saved report](svm_results.json) |

Calm/happy map to positive; angry/disgust/fearful/sad map to negative. Neutral and surprised are excluded. ROC AUC: 0.7113. The original SVM test file is missing, so the rebuild explicitly uses the validated cleaned pair. Its accuracy is not comparable to eight-class emotion accuracy.

### Vocal channel: speech versus song

| Experiment | Inputs | Final fit / test rows | Test accuracy | Macro F1 | Report |
| --- | --- | ---: | ---: | ---: | --- |
| Euclidean 1-NN | compressed [5000:13000] | 1828 / 624 | 58.17% | 0.3782 | [Saved report](timeseries_vocal_results.json) |
| MiniROCKET | compressed [5000:13000] | 1828 / 624 | 98.56% | 0.9852 | [Saved report](timeseries_vocal_full_results.json) |

Speech/song classification is a separate two-class problem; 98.56% here does not mean 98.56% emotion accuracy.

### Regression: predicting the supplied `kur` feature

| Selected model | Final fit / test rows | MSE | RMSE | MAE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lasso: alpha=0.1 | 1809 / 624 | 9.68607 | 3.11225 | 1.78256 | 0.75165 |

Regression includes emotion among the predictors and is not an emotion classifier. [Saved report](regression_results.json), [fixed-setting historical diagnostics](regression_comparison.json).

## Original notebook results

The following accuracies are copied from saved root-notebook outputs, at their displayed precision. They are historical observations, not independently reproduced full runs.

| Method | Emotion accuracy | Vocal-channel accuracy |
| --- | ---: | ---: |
| SAX + DTW | 23% | 76% |
| SAX + Sakoe–Chiba DTW | 24% | 78% |
| SAX + Itakura DTW | 25% | 80% |
| ROCKET | 33% | 92% |
| MiniROCKET | 55% | 99% |
| Euclidean 1-NN | 17% | 58% |

Copied vocal notebooks report ROCKET 93% and MiniROCKET 98%, versus 92% and 99% in the root notebook. Both versions are preserved. [Original reports, macro F1 and notebook/cell references](legacy_timeseries_results.json).

Read-only reloads of the four original neural checkpoints also reproduced their saved accuracies to displayed precision. These checks validate existing artifacts, not retraining:

| Original checkpoint | Reloaded test accuracy |
| --- | ---: |
| `best_model_DROPOUT.h5` | 50.0000% |
| `best_model_L1.h5` | 44.8718% |
| `best_model_L2.h5` | 47.4359% |
| `best_model_NOREG.h5` | 39.1026% |

[Checkpoint evidence](neural_legacy_comparison.json), [neural methodology](../../README.md#methodology-and-reproducibility).

## Cancelled and incomplete runs

| Run | Status | Retained evidence |
| --- | --- | --- |
| Full emotion ROCKET/MiniROCKET comparison | Cancelled | Frozen sources and partial ROCKET feature cache |
| Emotion SAX/DTW | Cancelled | Verified SAX preprocessing and partial predictions |
| Vocal-channel SAX/DTW | Cancelled | Verified SAX preprocessing and partial predictions |

Partial predictions and feature caches remain in the separate academic workspace, outside this public checkout, and are not complete scores. Completed vocal MiniROCKET results remain preserved. Do not launch or resume comparable long runs without an explicit request.

## Reproducibility limits and remaining work

- Raw WAV import reproduced both original NPY hashes across all 2,452 recordings. The compression candidate differs by up to 2.95e-7; analysis retains the original compressed arrays.
- Current emotion Euclidean accuracy is 12.82%, versus the saved rounded 17%. Independent original-backend checks match the rebuilt predictions; the historical discrepancy is unresolved.
- Fixed-setting Lasso, Ridge and gradient-boosting regression diagnostics match saved notebook MSE to displayed precision. The original enormous linear-regression error is not reproduced; its cause is unresolved.
- Feature extraction and the full cleaning/outlier-removal code are unavailable. External dataset links do not make a standalone public data distribution.
- Added actor validation, missing original seeds and changed library runtimes are explicit rebuild choices. Numerical proximity does not establish equivalence or statistical improvement.
- Remaining ensembles, exploratory embeddings, shapelets, clustering, motifs and explanations are documented academic analyses whose full rebuild remains incomplete.

See the [project guide](../../README.md) and [machine-readable index](results_summary.json).
