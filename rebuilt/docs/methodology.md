# Experiment boundary and provenance

## Historical project

All pre-existing files and links remain in place. `legacy_manifest.json` records
their SHA-256 hashes or symlink destinations before this rebuild. It excludes
environment-owned hidden directories and bytecode caches. Symlink targets are
not copied or hashed by that manifest; the baseline reports hash the actual
feature CSV inputs separately.

No notebooks were executed, rewritten, cleared, renamed, or removed. Existing
models and results were not overwritten. Archived variants include different
experiments, so archives are retained.

## Supplied data

The original feature tables have 1,828 training rows, 624 test rows, and 434
columns. Nine metadata fields are excluded explicitly; 425 numeric columns
remain in source order. The supplied description identifies global and four
window statistics of waveforms, differencing, zero crossings, MFCCs, spectral
centroids, and STFT chromagrams.

Audio import in the historical project uses actors 01–18 for training and 19–24
for testing. The new loader verifies this membership directly in the supplied
feature tables and rejects overlapping actors or filenames.

The supplied feature tables are already derived data. Their extraction code,
the full cleaning/outlier-removal procedure, and generation of compressed audio
arrays are not present. Rebuilding these steps will require documented new
choices or recovered implementation. They are not claimed complete here.

## New baseline protocol

- Task: eight-class emotion classification; retain neutral and surprised.
- Input: original supplied features, not `Train_NOoutliers` or `Train_ripulito`.
- Features: every non-metadata numeric column, including `frame_count`.
- Processing: float64 conversion only; no scaling, imputation, feature selection,
  categorical encoding, row filtering, or outlier removal.
- Missing/infinite values: reject; do not infer a replacement strategy.
- Evaluation: fixed supplied actor holdout; no resplitting or tuning.
- Reference classifier: most-frequent training class.
- Random forest: 100 trees, log-loss criterion, minimum leaf size 10, minimum
  split size 5, seed 42, one job; other defaults are saved in each report.
- Metrics: accuracy, macro F1, per-class precision/recall/F1/support, and confusion
  matrix with an explicit alphabetically sorted class order. Undefined metric
  values are reported as zero.

The forest's criterion and minimum leaf/split settings come from the historical
explainability notebooks. Seed 42, use of all original numeric features, and this
experiment's protocol are explicit new choices. Its scores must not be presented
as reproductions or improvements over historical experiments using other data
and splits. No test-driven hyperparameter search is performed.

Repeated runs on this test partition verify implementation repeatability. Future
model selection should use training actors only; testing repeated candidate
models on the holdout would compromise its role as final evaluation.

## Work remaining

1. Recover feature extraction and cleaning provenance where possible.
2. Implement audio-to-feature generation with documented parameters and tests.
3. Extend the existing actor-based validation to remaining modeling workflows.
4. Rebuild each legacy experiment separately, including its task and feature set.
5. Evaluate explanations using a documented training-data reference distribution.
6. Publish comparable results with complete provenance and their limitations.

Legacy concerns—independent test scaling, RFE fitted to predictions, random
resplitting of the supplied holdout, reversed metric arguments, and exploratory
execution errors—remain documented concerns, not silent changes to old notebooks.

API reference for future pipeline-based preprocessing:
[scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
The first forest baseline has no learned preprocessing stage.

## Second milestone: cleaned tabular comparison

`configs/tabular.toml` selects the historical ensemble's cleaned artifacts. Their
numeric features, values, labels and row order were compared with that notebook's
loading/drop-column logic and matched exactly. Actor membership is decoded from
the seven-part RAVDESS filenames because cleaned tables lack an actor column.
Missing metadata fields are allowed only through the explicit cleaned-data mode.

The comparison rebuilds the decision tree and PCA/RFE/SelectKBest/SelectFromModel
analysis. Five grouped folds use only training actors; each fold clones and fits
its own preprocessing and estimator. The highest mean validation macro F1 selects
the final method; ties keep the first configured method. The test table is used
only after this selection to report final predictions for that method and a
majority baseline. Fold standard deviation describes variability, not a confidence
interval. There is no hyperparameter search.

See `experiment_map.md` for preserved settings and explicit corrections. Old
results remain available; numerical reproduction of their unseeded runs is not
claimed. Baseline runs saved in the first milestone retain their original reports
and source hashes even as the rebuilt implementation evolves.

## Third milestone: SVM and boosting

SVM follows the notebook's binary emotion definition exactly: calm/happy are
positive (1), angry/disgust/fearful/sad negative (0), and neutral/surprised are
excluded. Retained numeric values and their row order are unchanged. Original
emotion labels are saved alongside binary truth for traceability. The missing
`Test_ripulito` prevents exact reconstruction of the SVM notebook's inputs; the
validated ensemble cleaned pair supplies the current consistent feature schema.

The three LinearSVC C settings and three fixed-kernel SVC explorations are
retained, alongside all 28 combinations in the original SVC grid. Scaling lives
inside each pipeline. LinearSVC uses explicit dual=True, 1,000 iterations and
tolerance 1e-4; convergence warnings are recorded without raising those limits.
SVC exploration uses gamma=auto, while the grid keeps gamma=scale. Test ROC AUC
uses the selected model's decision scores for class 1.

Gradient boosting and XGBoost retain their original search spaces and ten-trial
randomized-search size. Samples are generated with scikit-learn ParameterSampler
and seed 42; candidates are evaluated on the same five actor folds. Selection
uses mean accuracy as in the original searches, with macro F1 also recorded.
The gradient classifier seed remains 0; XGBoost's remains 42. These seeds have
distinct roles from the fold and sampling seed. No test data enters selection.

XGBoost uses a cloned estimator and a train-fitted LabelEncoder within its saved
pipeline, restoring original emotion names at prediction time. The removed
`use_label_encoder=False` option is not passed. Model parameters containing a
nonfinite sentinel, such as XGBoost's missing=nan, are represented as strings in
JSON metadata; fitted models retain the actual estimator parameter values.

Search validation scores are selection statistics on reused folds, not nested-CV
generalization estimates. The test partition has already been examined in the
wider project. Different software defaults and the actor-fold procedure preclude
claims of historical numerical equivalence. LightGBM, CatBoost, bagging, and
legacy AdaBoost remain separate pending experiments.

API references: [LinearSVC](https://scikit-learn.org/1.8/modules/generated/sklearn.svm.LinearSVC.html),
[GradientBoostingClassifier](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html),
[XGBoost](https://xgboost.readthedocs.io/en/release_3.2.0/python/python_api.html),
[AdaBoost compatibility](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.AdaBoostClassifier.html).

## Fourth milestone: regression and neural workflow

Regression retains `kur` as a continuous target, the original metadata predictors,
model families, alpha values and forest parameter space. Numeric/categorical
feature order matches notebook-style preprocessing. Mean MSE across actor folds
selects the final pipeline; only that pipeline and the mean reference receive a
final held-out score. Added RMSE/MAE are computed from actual predictions. Lasso
iteration limits are unchanged and convergence warnings are saved.

Independent fixed-setting diagnostics match the saved Lasso, Ridge and boosting
MSE values to their displayed precision. The original linear regression's huge
error is not reproduced by current defaults. Its cause is unresolved; no solver
or feature changes were introduced to manufacture a match. See
`regression_analysis.md` and `regression_comparison.json`.

The neural implementations retain the five original variants and search spaces,
including the original one-epoch early-stopping experiment and duplicate search
entry. Actor validation, preprocessing fitted within validation partitions,
explicit seed 42 and deterministic operations are documented new choices. Direct
Keras/native checkpoint execution passed integration, checkpoint and preprocessing
checks. A read-only reload of all four legacy HDF5 networks matches the notebook's
saved accuracies. The new dropout variant scores 51.28% test accuracy after
actor-validation selection; it is a new experiment. The installed TensorFlow
runtime and its full dependency snapshot are separate from the older core
runtime. The full parameter search completed 75 fold fits and scores 51.44% test
accuracy after mean macro-F1 selection and all-training refit. See `neural_analysis.md`.

## Fifth milestone: waveform import and initial time-series analysis

The original WAV loader, metadata ordering, actor partition, training-derived
NaN padding and test clipping are rebuilt. All 2,452 rows and labels were compared;
both generated raw NPY file hashes equal the supplied originals. A compressed
candidate uses q=10, float32 input and SciPy's default eighth-order zero-phase
IIR decimation. Its maximum measured differences are 2.95e-7/8.27e-8, so it is
not claimed equivalent and is not used to replace supplied analysis inputs.

The two classification windows, per-recording min/max normalization and original
one-neighbor Euclidean backend/defaults are retained. Actor-fold selection and
readable target labels are explicit rebuild choices. Emotion test accuracy
12.82% differs from the saved rounded 17%; vocal-channel accuracy 58.17% matches
the saved rounded 58%. Independent notebook-style scaling/backend calculations
and reloaded artifacts reproduce all current predictions, but the historical
emotion discrepancy is unresolved. No test-based adjustments were introduced.

ROCKET/MiniROCKET adapters and configurations retain kernel/alpha settings,
with explicit new seed 42 and public 3D array conversion. Their optional backend
is installed and integration-validated. Reusing shape-only seeded ROCKET features
preserves tested outputs; data-dependent MiniROCKET fitting remains fold-local.
The full vocal comparison completed with frozen code and passed artifact/hash
verification. The unfinished full emotion comparison and both full SAX/DTW
evaluations were cancelled at the user's request; partial checkpoints have no
final scores. SAX preprocessing matches the original notebook calls for all
training/test rows in both tasks. Comparable large jobs must not be restarted or
launched without an explicit request. See [results and status](results.md) and
`timeseries_analysis.md` for the preprocessing inventory, explicit methodological
choices, results and verification.
