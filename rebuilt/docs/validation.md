# Validation status

## Completed

- Inspected original notebooks, Python modules, archives, data schemas, model
  files, results, and external links before implementation.
- Parsed all original notebooks and checked the original Python files' syntax.
- Recorded 82 original files and links before adding the rebuild. Original file
  contents and symlink destinations were verified unchanged afterward.
- Ran the data audit on the original supplied feature tables: 1,828 training rows,
  624 test rows, and 425 numeric features in matching order.
- Confirmed all eight emotions, finite numeric values, unique filenames, and no
  train/test overlap of actors or filenames. Actor membership matches 01–18
  versus 19–24.
- Passed 15 dependency-free data tests on Python 3.14.7. They check row/feature
  order and unchanged inputs, valid partitions, missing/nonfinite data, invalid
  labels/actors, duplicate filenames/headers, ragged/empty files, schema mismatch,
  overlap, unexpected partitions, and missing classes.
- Passed the training integration test, including artifact generation, saved
  model loading, repeatability, and protection against overwriting existing runs.
  Total: 16 tests passed, no skips in the scientific environment.
- Ran two complete real-data baselines: metrics and `predictions.csv` contents
  matched exactly. Both runs recorded zero training warnings.
- Built and installed the editable package using cached setuptools 82.0.1, then
  ran its installed `ravdess audit` command successfully from `/tmp` with explicit
  absolute input/config paths.

The full audit is saved in `data_audit.json`, including source CSV hashes and
feature order. It reports data properties, not model performance.

## New baseline results

| Model | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Majority class | 0.15384615384615385 | 0.03333333333333333 |
| Random forest | 0.4423076923076923 | 0.4155011969983793 |

Full reports, predictions, and models are in local `runs/baseline-001/` and
`runs/baseline-002/`. The compact committed report is `baseline_results.json`.
It contains input/source hashes, settings, environment, all per-class metrics,
and the prediction hash used in the repeatability comparison.

## Environment and reproduction

Validated runtime: Python 3.14.7, NumPy 2.4.2, SciPy 1.17.1, scikit-learn 1.8.0,
joblib 1.5.3, threadpoolctl 3.6.0, Linux x86_64. Exact runtime versions are pinned
in `requirements.lock.txt`; package artifact hashes/build tooling are not locked.

An isolated `.venv` was created, but network installation failed after PyPI DNS
lookup errors. A read-only existing scientific environment was subsequently
found at `/mnt/hdd/Uni/Progettini/ML tests/.venv/` and used for full validation,
with bytecode writes disabled. No dependencies were installed into that project.
The new `.venv` has the editable package for audit usage, but does not contain the
scientific dependencies. To train through it, install the pinned requirements
when network/package access is available.

To reproduce on a machine with package access, from `rebuilt/`:

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
ravdess train --data-dir ../ravdess_dm2_2223 --config configs/baseline.toml --output runs/new-baseline-001
ravdess train --data-dir ../ravdess_dm2_2223 --config configs/baseline.toml --output runs/new-baseline-002
```

The local validation commands used the existing environment's interpreter with
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src` rather than modifying its installation.
The first-milestone system Python run passed 15 data tests and skipped its training
test. After the current additions, system Python runs 36 tests: 22 data tests pass
and 14 modeling tests skip explicitly because scientific dependencies are absent.

## Limits

Only the recorded environment has been tested. Repeatability across versions,
hardware, and platforms is not established. No hyperparameter tuning, statistical
uncertainty estimate, or raw-audio feature reproduction was performed.

Historical notebooks have not been executed in this environment. Their software
versions are unknown, and some contain execution defects. This new baseline has
different inputs/features from several historical experiments. There is no claim
of historical metric equivalence.

## Second milestone validation

- Verified the cleaned loader against the original ensemble notebook's CSV
  drop-column logic: all numeric values, feature order, row order and labels
  match exactly. Actor IDs decoded from filenames match the existing partition.
  No claim is made about recreating the upstream cleaning process.
- Re-ran the original supplied-feature baseline after extracting shared reporting:
  predictions and all metrics matched the saved first-milestone run exactly.
  The new run is `runs/baseline-after-reporting-refactor/`; old artifacts remain.
- Ran five training-actor folds for each of seven configured methods on the full
  cleaned dataset. Selection used mean validation macro F1. The selected random
  forest then scored 0.421474358974359 accuracy and 0.3942518378764347 macro F1
  on the fixed test partition. SelectKBest recorded ten warnings across the five
  folds because `zc_q75_w4` is constant and its ANOVA calculation encounters an
  undefined division. All folds completed; the column was retained. The selected
  forest's final fitting recorded no warnings.
- Added cleaned-export and actor-parsing checks, disjoint/repeatable CV coverage,
  fold-only scaler fitting, correct RFE labels, candidate validation, full-run
  serialization, and transformed feature-name checks.
- Saved `cleaned_data_audit.json`, `tabular_results.json`, and a readable
  `tabular_analysis.md`. Full local artifacts include fold row indices and fitted
  pipelines as well as predictions.

All 28 tests passed in the scientific environment. Tests and preservation
verification are rerun after meaningful implementation changes.
These additions change the new comparison's evaluation procedure explicitly;
the historical notebooks and results are unchanged.

## Third milestone validation

- Added binary task tests for exclusion rules, 0/1 mapping, preserved row/feature
  alignment, source labels, and unchanged original tables.
- Verified the same mapping directly on the real cleaned tables: 1,526 training
  rows (972 negative, 554 positive), 528 test rows (336 negative, 192 positive).
- Tested all original SVM grid entries and exploration variants, fixed gradient
  search settings, repeatable sampling, invalid task/trial rejection, saved binary
  models/predictions, XGBoost encoding/cloning/serialization, and valid JSON for
  estimator parameters containing a NaN sentinel.
- Extracted shared actor validation and final reporting without changing the
  previous real-data comparison: every candidate's fold scores, fold membership,
  selected method, final metrics and prediction CSV matched `tabular-001`.
  The comparison run is `runs/tabular-after-validation-refactor/`.
- Added bounded process workers for searches. A fixture comparison verified that
  serial and parallel execution have identical scores, warnings, fold membership,
  candidate parameters and selected method. Saved result order is restored after
  worker completion; warnings are collected within each worker process.
- All 36 tests passed in the available scientific environment, including the
  optional XGBoost integration test. Runtime versions remain as previously
  recorded, with XGBoost 3.2.0 added in `requirements-xgboost.lock.txt`.

Full original-space searches and final artifact checks are recorded below.
The initially serial searches were interrupted before output creation
and restarted with bounded parallel workers; parameter spaces and trial counts
were retained.

- Completed every SVM candidate/fold: 35 candidates including the majority
  reference, or 175 fold fits. Selected linear SVC C=0.001; test accuracy
  0.6799242424242424, macro F1 0.6305628583849132, ROC AUC 0.7113405257936509.
  Nine convergence warnings from LinearSVC candidates were captured; none of
  those candidates was selected.
- Completed all ten XGBoost candidates plus the majority reference across five
  folds: 55 fold fits. Selected learning rate 0.1, gamma 0.1, tree_method=approx;
  test accuracy 0.44871794871794873, macro F1 0.4335177521117635. No warnings
  were recorded.
- Reloaded both workflows' saved pipelines and verified every stored test
  prediction. Checked recorded source and configuration hashes against the
  current files. Full compact results are `svm_results.json` and
  `xgboost_results.json`; the binary CSV retains original emotion labels.
- Completed ten gradient-boosting candidates plus the majority reference across
  five folds: 55 fold fits. Selected learning rate 0.1, max depth 3, minimum split
  size 10, preserving 100 stages and classifier seed 0. Test accuracy
  0.4599358974358974, macro F1 0.4520429520249779. No warnings were recorded.
- Verified every gradient-boosting stored prediction after loading the saved
  pipeline, checked source/config hashes, confirmed five scores per candidate,
  and checked that no test actor appears in selection folds.
- Verified the editable package builds with its optional dependency declaration
  and that the installed binary-task audit works from `/tmp` with explicit paths.
- Rechecked all 82 original files and links against their initial snapshot:
  unchanged. SVM/boosting compact reports and readable analyses are saved under
  `docs/`; full local artifacts remain in their separate run directories.

## Regression and neural milestone

- Extended the validated loader to retain explicitly configured categorical
  predictors and move `kur` out of predictors into the continuous target. The
  classification defaults remain numeric-only. Regression keeps emotion as a
  predictor; neural classification excludes it and retains `kur` as an input.
- Independent notebook-style encoders/scaler matched the actual regression
  train and test matrices exactly: 1,809/624 rows and 202 predictors. Fixed
  original Lasso/Ridge/boosting settings match saved MSE values to displayed
  precision; linear regression differs materially. No forest saved score exists.
- Completed 47 regression candidates across five actor folds, or 235 fits.
  Selected Lasso alpha=0.1, mean validation MSE 8.95109377; test MSE 9.68607083,
  RMSE 3.11224530, MAE 1.78256373 and R² 0.75164544. All ten captured convergence
  warnings came from alpha=0.001/0.0001 candidates; none from the selected model.
- Reloaded both saved regression models and verified every one of the 624 stored
  predictions, current source/config hashes, all candidate fold scores, and
  absence of test actors from validation folds.
- The neural audit confirms 202 inputs: 197 numeric and five categorical columns.
  Backend-independent tests cover predictor/target separation, actor partitions,
  original epochs/regularizers/dropout settings and deterministic search sampling.
  The backend integration test checks architectures, baseline checkpoint reload
  and early-stopping checkpoint creation when TensorFlow is available.
- Scientific environment: 48 tests ran, 47 passed, one TensorFlow integration
  test skipped. System Python: 22 passed, 26 skipped for missing dependencies.
  Core package editable installation succeeded offline; installed CLI audits
  work from outside the repository.
- Neural execution fails clearly before output creation when TensorFlow is
  absent. No metrics/checkpoints were created. Installation in a separate cached
  Python 3.13 environment failed due to unavailable PyPI DNS access. Full network
  runs and numerical comparisons remain pending; no replacement model was used.
- All 82 legacy files and links were verified unchanged. Earlier run reports
  retain the source hashes of their own execution rather than being rewritten
  to claim they ran the current source.

## Completed TensorFlow validation and neural experiments

- The user installed the neural dependencies from a normal terminal, resolving
  the earlier sandbox network restriction. TensorFlow CPU 2.21.0 and Keras 3.15.1
  run in Python 3.13.14, with NumPy 2.5.4 and scikit-learn 1.9.1. The full
  installed package snapshot is `requirements-neural.lock.txt`; earlier core
  environment snapshots are preserved.
- All 49 tests ran in the neural environment: 48 passed, one optional XGBoost
  integration test skipped because that package is absent from this environment.
  Both TensorFlow tests passed. They check all five architectures, checkpoint
  round trips, early-stopping checkpoint handling, and a deterministic training
  fixture with identical history, weights and predictions on two repeated fits.
- Completed all five original variants with unchanged limits/settings. Actual
  epochs: baseline 200, early-stopping-only 1, L1 186, L2 147, dropout 169.
  Validation holds out actors 1, 2, 6 and 9 (414 rows) from the 1,809 training
  rows; preprocessing and models fit on the remaining 1,395 rows.
- Validation selected dropout: accuracy 0.56038647. The reloaded selected
  checkpoint scored 0.5128205128 test accuracy and 0.5027121515 macro F1. Every
  one of its 624 saved test predictions, majority predictions and final metrics
  matched. All five saved networks reproduce their respective validation scores.
- Completed fifteen sampled search candidates across five actor folds (75 fits),
  then refit the selected candidate on all training rows. Selected three 50-unit
  ReLU layers, L2=0.01, Adam learning rate 0.001, 50 epochs, batch size 32. Mean
  validation macro F1 was 0.4309481993; final test accuracy 0.5144230769 and macro
  F1 0.5063590885. All 624 predictions and final metrics matched after reload.
- Verified independent notebook-style LabelEncoder/StandardScaler calculations
  exactly match saved preprocessing in both runs, and checked fold indices/actors,
  final fitting actors, finite normalized probabilities, widths, input hashes and
  current source/config hashes. No test actor enters either selection procedure.
- Read-only reloads of all four legacy HDF5 checkpoints using original full-train
  preprocessing reproduce their saved accuracies: no-regularizer 0.39102564,
  L1 0.44871795, L2 0.47435897, dropout 0.5. No legacy weights were retrained or
  rewritten. Diagnostic values/model hashes are in `neural_legacy_comparison.json`.
- The variant report captured 330 Keras/NumPy deprecation warnings; search
  captured eight. TensorFlow logged ignored dataset-attribute and retracing
  notices. Full runtime logs and exact invocations/thread settings are retained
  with each local run; compact reports summarize repeated warnings with counts.
- Added repeatable `verify_neural_run.py` and `compare_neural_checkpoints.py`
  scripts, measured results, environment snapshot and documentation. Original
  neural implementation/configurations were not altered to improve results.
  Stochastic historical training equivalence and statistical improvement are
  not claimed. Full-run retraining repeatability was not tested.

## Waveform preparation and initial time-series analyses

- Rebuilt decoding for the observed 48 kHz 16-bit mono/stereo WAV files, with
  float32 channel averaging, actor/filename ordering and original metadata.
  Preserved actors 01–18/19–24, training-derived maximum 304,304 samples, NaN
  padding, and clipping of one test recording. No resampling or imputation.
- All 2,452 raw rows matched; generated train/test raw NPY SHA-256 hashes are
  identical to the supplied files. Metadata dictionaries and row order match.
  The full preparation report retains every WAV hash and all output hashes.
- A tenfold decimation candidate preserves compressed shapes and NaN positions,
  but maximum value differences are 2.95229256e-7 (train) and 8.26548785e-8 (test),
  and float32 equality fails. It remains a separately named candidate; modeling
  uses supplied arrays. No claim of original compression-method equivalence.
- Rebuilt emotion `[45000:100000]` and compressed vocal `[5000:13000]` classifiers
  with original per-recording min/max scaling and pyts's one-neighbor Euclidean
  sklearn backend/defaults. Both selected Euclidean over majority by five-fold
  training-actor macro F1. Test emotion accuracy 0.12820513, macro F1 0.09196175;
  vocal accuracy 0.58173077, macro F1 0.37820843. No warnings were captured.
- Emotion does not reproduce the notebook's saved rounded 0.17 accuracy; the
  cause remains unresolved. Vocal matches its saved rounded 0.58. Independent
  record-by-record scaling and the current original-default backend reproduce
  all current predictions; historical numerical equivalence is not inferred.
- Reloaded both bundles and verified every one of their 624 test predictions,
  metrics, label/filename ordering, actor isolation and source hashes. Separate
  reports/scripts make those checks repeatable.
- All 60 tests ran in the available neural scientific environment: 56 passed,
  four skipped (optional XGBoost plus tslearn, pyts and sktime integration tests).
  Tests include stereo decoding, training-only padding length, interval/metadata
  validation, stateless scaling, original neighbor count, correct vocal target
  headers, artifact reload and no-overwrite behavior. Existing tabular/regression
  and TensorFlow tests still pass after shared artifact/feature-report changes.
- ROCKET/MiniROCKET adapters/configurations are present with full original kernel
  settings; their real-backend serialization fixture and full runs remain pending.
  Installation failed due to the session's network restriction, so no convolution
  scores are claimed. Instructions install them in a separate environment.
- Added `audio_preparation.py`, `time_series.py`, CLI series commands, four configs,
  optional dependencies, preparation/verification scripts and measured reports.
  All 82 original files and links were verified unchanged.

## Installed time-series backends and long comparisons

- Dedicated runtime: Python 3.13.14, sktime 0.40.1, tslearn 0.9.0, pyts 0.14.0,
  NumPy 2.3.5, Numba 0.68.0 and scikit-learn 1.7.2. Exact packages are recorded in
  `requirements-timeseries.lock.txt`; the neural environment is unchanged.
- All 64 tests ran: 61 passed, three optional skips (two TensorFlow tests and
  XGBoost). All tslearn/pyts/sktime integration tests passed, including real small
  ROCKET/MiniROCKET serialization, scaler parity and original-wrapper predictions.
- Literal real-data recreation using the original tslearn/pyts APIs matches all
  earlier rebuilt baseline predictions (zero differences in both tasks), including
  emotion accuracy 12.82%. The saved rounded 17% discrepancy remains unresolved.
- Rechecked the compression candidate across all 2,452 WAV files in the new
  runtime. Values are exactly identical to the previous candidate; original-array
  maximum differences remain 2.95229256e-7 / 8.26548785e-8. No source substitution.
- Added behavior-preserving execution reuse for ROCKET: seeded kernels are
  generated from fixed shape only, without validation observations. Tests compare
  kernels, features, fold scores, parameters and selection against ordinary fold
  pipelines exactly. MiniROCKET's fitting remains inside each actor fold.
- Added checksummed batch feature persistence; tests cover direct/chunk parity,
  completed-cache reuse, corruption detection and resume after interruption.
- Tested the frozen-source long-job launcher end to end on a small fixture:
  comparison, artifact reload, actor/preprocessing/input/config/source checks,
  result export and completion status all succeeded. Fixture metrics were kept
  under `/tmp`, not presented as real dataset results.
- Full original-count jobs used `timeseries-{emotion,vocal}-full-003-features`.
  The vocal comparison completed; the emotion comparison was cancelled at the
  user's request. Status/progress/logs are retained. No window/kernel reduction
  was introduced, and no full rebuilt emotion score is reported.
- Earlier repeated-transform attempts were interrupted before complete results,
  and subsequent checkpointed jobs resumed their retained features. No legacy
  experiments or results were removed or overwritten.

### Completed full vocal-channel comparison

- All four candidates completed all five actor folds (twenty candidate/fold fits).
  Mean macro F1: majority 0.371114, Euclidean 0.375832, ROCKET 0.920738,
  MiniROCKET 0.975574. No held-out test data entered selection.
- Selected MiniROCKET, refitted on all training actors. Test accuracy
  0.9855769231, macro F1 0.9852492731; every one of 624 stored predictions and
  metrics reproduced after reload. Independent per-recording scaling, actor
  isolation, input/config/frozen-code hashes passed automatic verification.
- The requested 10,000 kernels yield 9,996 features under standard MiniROCKET
  rounding. Full results, verification and generated readable analysis are in
  `timeseries_vocal_full_{results.json,validation.json,analysis.md}`.
- The unfinished emotion comparison and both full SAX/DTW evaluations are
  cancelled. Partial checkpoints remain preserved and are not final results.
  Report original saved notebook results from `legacy_timeseries_results.md`;
  extraction retains the displayed precision, notebook hashes, cell references,
  and differing copied vocal ROCKET/MiniROCKET reports. No comparable large run
  should be launched or resumed without an explicit user request.

### Portfolio documentation and consolidated results

- Added `results.md` and `results_summary.json`, generated directly from saved
  reports by `scripts/summarize_results.py`. The generator uses only the standard
  library and does not load models, read datasets or train experiments.
- Checked all 11 completed experiment entries against their source metrics,
  including task, test-row counts and the dropout variant's smaller fitting set.
  Retained all 27 historical time-series reports, original neural checkpoint
  diagnostics and three explicitly cancelled runs. Partial outputs have no final
  metrics in the summary. `summarize_results.py --check` passes.
- Reworked the root and package READMEs with task-separated findings, setup,
  explicit input requirements, package layout and current scope. The previous
  detailed workflow commands remain in `docs/execution.md`, with stale running
  statuses corrected. All changed documentation links resolve locally, and the
  root README's highlighted metrics match their saved reports.
- Ran the complete fixture-based suite in the time-series environment: 67 tests
  in 46.213 seconds, 64 passed and three optional neural/XGBoost tests skipped.
  No full experimental fits were started or resumed.
- The legacy preservation check confirmed all 82 original files and links are
  unchanged. No original notebook, scientific input, saved model or result was
  modified during this documentation work.
