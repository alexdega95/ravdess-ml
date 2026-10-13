# Experiment execution reference

This reference preserves the detailed commands for each implemented workflow.
Run commands from `rebuilt/` unless explicitly stated otherwise. Begin with the
[quick start](../README.md) and [results summary](results.md).

Large reruns were cancelled at the user’s request. The examples below document
available workflows; they are not a recommendation to start long searches or
resume cancelled runs. The current documentation work reads saved reports only.

The rebuild validates supplied feature tables, runs eight-class emotion baselines,
and compares feature-reduction methods on the historical cleaned-feature data.
Actors 01–18 belong to training; actors 19–24 belong to testing.

The first baseline is a new experiment. It uses the original supplied feature tables, all 425
numeric features including `frame_count`, no outlier removal, and no learned
preprocessing. It does not reproduce the cleaned datasets or historical scores.

## Setup

From this `rebuilt/` directory, use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`pyproject.toml` declares NumPy, scikit-learn, and joblib with version ranges.
For the runtime versions tested on Python 3.14.7 / Linux x86_64:

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
```

The snapshot pins the five runtime dependencies, including transitive ones. It
does not lock wheel hashes or build tooling; other Python versions/platforms are
not yet verified. Version ranges in `pyproject.toml` describe supported intent,
not a guarantee that every combination has been tested.

The local machine used for initial verification has `uv` and no system pip. Its
equivalent setup uses `uv venv .venv` followed by
`uv pip install --python .venv/bin/python -e .`.
The core audit and tabular workflows require no TensorFlow, notebook, audio, or explainability dependencies.

## Input data

Provide a directory containing these original supplied files:

```text
RavdessAudioOnlyFeatures_TRAIN.csv
RavdessAudioOnlyFeatures_TEST.csv
```

In this workspace they are accessible through `../ravdess_dm2_2223/`. That link
points outside the project and is not a portable data distribution. On another
machine, supply an explicit path to your copy of the original feature tables.
Downloading the original RAVDESS audio alone does not produce these CSVs; their
feature extraction implementation is not available in this project.

## Validate and run

```bash
ravdess audit --data-dir ../ravdess_dm2_2223 --config configs/baseline.toml
ravdess train --data-dir ../ravdess_dm2_2223 --config configs/baseline.toml --output runs/baseline-001
python -m unittest discover -s tests -v
```

Without installing the package, use the following commands. Audits still require the core scientific dependencies; tests for unavailable optional packages are skipped:

```bash
PYTHONPATH=src python -m ravdess_project audit --data-dir ../ravdess_dm2_2223 --config configs/baseline.toml
PYTHONPATH=src python -m unittest discover -s tests -v
```

The audit checks matching feature names/order, finite numeric values, unique
filenames, all eight classes, configured actor membership, and no train/test
actor or filename overlap. It does not change input files. Missing or malformed
data cause an error rather than an undocumented repair.

Each training run must have a new output directory. It produces:

- `report.json`: settings, input and source hashes, versions, warnings, accuracy,
  macro F1, per-class reports, and confusion matrices.
- `predictions.csv`: test filenames, actors, true labels, and model predictions.
- `models.joblib`: fitted classifiers, feature order, and class order.

Training artifacts and saved-model loading have been verified. Two runs on the
supplied CSVs produced identical predictions and metrics in the same environment.

Configuration and input paths are explicit, so installed commands work from any
directory. JSON reports record resolved local paths; data hashes identify file
contents independently of those paths. Runtime and platform differences may
still matter: a fixed seed alone is not a guarantee across software versions.

## Layout and next milestones

`src/ravdess_project/` contains reusable code; `configs/` holds experiment
settings; `tests/` checks data contracts and experiment behavior; `docs/` records
methodology and validation. Local run artifacts stay under `runs/`.

## Cleaned-feature analysis

The second workflow uses the exact cleaned artifacts from the original ensemble
notebook: `Train_NOoutliers` and `Test_ripulito (1)`, available in `../data/`.
It retains 1,809 training recordings, 624 test recordings, and 197 numeric
features. Actor IDs are recovered from the RAVDESS filenames; exported CSV
indices and the training-only outlier flag are excluded. The loader preserves
all rows and numeric values. It does not recreate the missing cleaning procedure.

```bash
ravdess audit --data-dir ../data --config configs/tabular.toml
ravdess compare --data-dir ../data --config configs/tabular.toml --output runs/new-tabular-001
```

`compare` uses five actor-disjoint folds within training to compare a majority
baseline, random forest, decision tree, PCA, RFE, SelectKBest, and model-based
selection. It selects by mean validation macro F1, then evaluates that method and
the majority baseline on the fixed test partition. The candidates are not ranked
by test scores. Each fold fits its own scaler/selector. Original reduction sizes
and the decision tree's minimum leaf size are explicit in `configs/tabular.toml`.

Run reports include fold membership, every candidate's validation scores,
selection settings, warnings, feature names/importances for the final model,
and held-out predictions. The saved pipelines contain their fitted preprocessing.
See [the experiment map](experiment_map.md) for what carries over from each
original notebook and which corrections are deliberate.

The completed comparison selected the random forest using training-fold scores.
It scored **42.15% accuracy** and **0.3943 macro F1** on the fixed test partition.
See [the analysis report](tabular_analysis.md) for every candidate's validation
scores and the saved provenance.

## SVM and boosting searches

These workflows reuse the same cleaned tables and five training-actor folds.
Every candidate fits its own scaler within each fold. Search spaces are copied
from the academic notebooks; a fixed sampling seed makes randomized searches
repeatable. Only the selected candidate and majority baseline are evaluated on
the test partition. Selection uses mean accuracy, matching the original searches.

```bash
ravdess search --data-dir ../data --config configs/svm.toml --output runs/new-svm-001
ravdess search --data-dir ../data --config configs/gradient_boosting.toml --output runs/new-gradient-001
```

SVM retains the original binary task: exclude neutral/surprised, map calm/happy
to 1 and angry/disgust/fearful/sad to 0. This leaves 1,526 training and 528 test
recordings. The original SVM test file is missing, so this explicitly uses the
coherent 197-feature cleaned pair rather than pretending to reconstruct that
notebook's different input. Prediction CSVs retain the original emotion alongside
the binary `positive` target. Its three LinearSVC settings, three `gamma='auto'`
kernel explorations and complete 28-entry SVC grid are all recorded. The grid
retains default `gamma='scale'`, as in the original grid search.

Gradient boosting retains 100 stages, classifier seed 0, and ten sampled choices
from the original learning-rate/depth/minimum-split space. It remains an
eight-emotion task; binary SVM accuracy must not be compared with its accuracy.

XGBoost is an optional dependency:

```bash
python -m pip install -e '.[xgboost]'
ravdess search --data-dir ../data --config configs/xgboost.toml --output runs/new-xgboost-001
```

XGBoost retains `multi:softmax`, depth 6, L2 penalty 1, classifier seed 42, and the
original learning-rate/gamma/exact-or-approximate tree space. The label encoder
lives inside the fitted model, so saved pipelines predict original emotion
names. CPU execution uses one thread. No optional library is imported for audits
or the core SVM/gradient workflow.

Training-CV search scores are used for selection, not claimed to be unbiased
nested-CV estimates. Warnings are saved, including convergence warnings, rather
than silently changing iteration limits. LightGBM/CatBoost require unavailable
dependencies; bagging and legacy AdaBoost remain pending. The older AdaBoost
default algorithm is absent from current scikit-learn, so it is not silently
replaced with another algorithm.

`search.parallel_jobs` controls process workers: two for SVM/XGBoost, three for
gradient boosting. Set it to one for serial execution. Completed fits may be
reported out of order; saved fold results and tie-breaking retain configuration
order. Serial/parallel scores were compared on a test fixture and matched.
No optional dependency is silently substituted when it is missing.

`requirements-xgboost.lock.txt` records the tested optional runtime versions;
install it instead of the base snapshot when reproducing XGBoost. The included
Linux NCCL package is a dependency of the installed XGBoost wheel; these
experiments still use CPU execution.

The completed binary SVM search selected a linear-kernel SVC with C=0.001 and
scored 67.99% test accuracy, 0.6306 macro F1, and 0.7113 ROC AUC. See the
[SVM analysis](svm_analysis.md) and its full saved report.

The eight-emotion searches scored 45.99% accuracy / 0.4520 macro F1 for gradient
boosting and 44.87% accuracy / 0.4335 macro F1 for XGBoost. See the
[boosting analysis](boosting_analysis.md) for protocol, selected settings,
candidate scores and remaining experiments. These scores do not establish an
improvement over the historical runs.

## Regression

The regression workflow predicts `kur`, retaining the original six categorical
predictors (including emotion), 196 numeric predictors, five model families and
search spaces. Encoding/scaling fit within actor folds. Run from `rebuilt/`:

```bash
ravdess audit --data-dir ../data --config configs/regression.toml
ravdess regress --data-dir ../data --config configs/regression.toml --output runs/new-regression-001
python scripts/compare_regression.py --data-dir ../data
```

The 47-candidate search selected Lasso alpha=0.1: test MSE 9.6861, RMSE 3.1122,
MAE 1.7826 and R² 0.7516. Three fixed-setting diagnostics match the notebook's
saved errors to displayed precision; linear regression differs substantially.
See [regression analysis](regression_analysis.md) for that exception and the
explicit changes to validation and previously absent seeds.

## Neural experiments

The five original dense-network variants and 15-trial parameter search have
separate configurations. They use 202 predictors and eight emotion classes.
The five variants have completed: validation selected dropout, scoring 51.28%
test accuracy and 0.5027 macro F1. The completed 15-candidate search scores 51.44%
test accuracy and 0.5064 macro F1 after actor-fold selection. These are
new experiments rather than exact reproductions. Core audits and non-neural
workflows do not require TensorFlow.

Use a separate Python 3.11–3.13 environment when package downloads are available:

```bash
uv venv --python 3.13 .venv-neural
uv pip install --python .venv-neural/bin/python -r requirements-neural.lock.txt
uv pip install --python .venv-neural/bin/python --no-deps -e .
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
.venv-neural/bin/ravdess neural --data-dir ../data --config configs/neural.toml --output runs/new-neural-001
.venv-neural/bin/ravdess neural --data-dir ../data --config configs/neural_search.toml --output runs/new-neural-search-001
.venv-neural/bin/python -m unittest discover -s tests -v
.venv-neural/bin/python scripts/verify_neural_run.py --run-dir runs/new-neural-001 --data-dir ../data
```

Variant selection uses held-out training actors; parameter search uses five actor
folds. The test partition is used after selection. Checkpoints use native `.keras`
files, with the fitted preprocessing/class encoding in `preprocessing.joblib`.
`ravdess_project.neural.predict_saved(run_directory, feature_rows)` reloads the
selected model. Long runs retain partial checkpoints on failure; `report.json`
is written only on successful completion. Use a fresh output directory on retry.
See [neural analysis](neural_analysis.md) for preserved settings, explicit
runtime changes, measured results and completed validation.

## Time-series preprocessing and classification

The raw WAV-to-array import has been rebuilt and verified across all 2,452
recordings. Both generated raw NPY files have the same SHA-256 hashes as the
originals. A q=10 compression candidate differs slightly (up to 2.95e-7), so the
classification workflows continue using the supplied compressed arrays.

The two Euclidean 1-neighbor analyses completed: emotion accuracy 12.82% / macro
F1 0.0920; vocal-channel accuracy 58.17% / macro F1 0.3782. The original saved
emotion accuracy was rounded to 17%, so that result is not reproduced. The
vocal-channel score matches the saved rounded 58%. Weak results are retained.
See [time-series analysis](timeseries_analysis.md) for the full preprocessing
inventory, comparisons and limitations.

Run from `rebuilt/`. The data directory for waveform analysis is the project root,
which contains the NPY/metadata links, rather than the tabular `../data/` folder.
The audio preparation command writes approximately 6.3 GB into a **new** output
directory and refuses to overwrite an existing run:

```bash
python scripts/prepare_audio.py --audio-dir ../ravdess_ts --reference-dir .. --output runs/new-audio-001
ravdess series-audit --data-dir .. --config configs/timeseries_emotion_euclidean.toml
ravdess series --data-dir .. --config configs/timeseries_emotion_euclidean.toml --output runs/new-series-emotion-001
ravdess series --data-dir .. --config configs/timeseries_vocal_euclidean.toml --output runs/new-series-vocal-001
python scripts/verify_timeseries_run.py --data-dir .. --run-dir runs/new-series-emotion-001
```

These commands require the installed core scientific dependencies. The complete
ROCKET/MiniROCKET comparison configurations preserve the original 10,000 requested
kernels and ridge alpha grid. Their optional packages are installed and all time-series integration tests pass.
The full vocal comparison completed: MiniROCKET test accuracy 98.56%, macro F1
0.9852, with all saved predictions verified. The unfinished emotion job was cancelled at the user’s request.
These long runs use resumable, frozen-source jobs. Install in a separate environment from a terminal with network access,
from the **project root**:

```bash
uv venv --python 3.13 rebuilt/.venv-timeseries
uv pip install --python rebuilt/.venv-timeseries/bin/python -e './rebuilt[timeseries]'
rebuilt/.venv-timeseries/bin/python -m unittest discover -s rebuilt/tests -v
rebuilt/.venv-timeseries/bin/python rebuilt/scripts/run_timeseries_job.py --project-root . --data-dir . --config rebuilt/configs/timeseries_emotion.toml --run-dir rebuilt/runs/new-series-emotion-full-001 --export-dir rebuilt/runs/new-emotion-results
rebuilt/.venv-timeseries/bin/python rebuilt/scripts/run_timeseries_job.py --project-root . --data-dir . --config rebuilt/configs/timeseries_vocal.toml --run-dir rebuilt/runs/new-series-vocal-full-001 --export-dir rebuilt/runs/new-vocal-results
```

**Large reruns are cancelled at the user's request.** Report the
[original saved time-series results](legacy_timeseries_results.md) instead.
The commands here document execution; do not launch or resume similarly costly
experiments without an explicit request. Partial runs and completed rebuilt
results are preserved separately.

The launcher freezes project source/configuration, resumes checksummed ROCKET
feature batches and automatically verifies/export results after comparison.
`job_status.json`, `state.json` and `comparison.log` live in the run's sibling
`-features` directory. Final outputs are not overwritten. If generation is
interrupted before final output creation, the launcher supports resuming with
unchanged source/configuration. The cancelled jobs must not be restarted without
an explicit user request. Keep the dedicated environment unchanged during
training. Raw-waveform ROCKET may take several hours at the original settings.

`requirements-timeseries.lock.txt` records the tested runtime. Install its snapshot
instead of resolving new dependencies when reproducing these runs, then install
the project editable with `--no-deps`.

No compressed candidate or imputation is automatically substituted for original
inputs. [SAX/DTW](sax_dtw_analysis.md) has a separate checkpointed runner
that preserves all three notebook distances and their original preprocessing.
Shapelets, clustering and motifs remain distinct pending experiments.

Remaining work includes extraction/cleaning provenance, pending ensemble
variants, time-series analyses, clustering,
explanations, and MDS/Isomap/t-SNE exploration.

## Initial results

| Model | Test accuracy | Macro F1 |
| --- | ---: | ---: |
| Majority class | 0.153846 | 0.033333 |
| Random forest | 0.442308 | 0.415501 |

These are new baseline results on 624 held-out recordings. They are not a
comparison with legacy scores. Full per-class metrics and provenance are saved
in [baseline_results.json](baseline_results.json).

See [methodology](methodology.md) and [validation](validation.md).

To check that original files still match their pre-rebuild contents, run:

```bash
python scripts/verify_legacy.py --project-root ..
```
