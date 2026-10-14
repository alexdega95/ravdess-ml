# RAVDESS-ML

Machine learning analysis of acted speech and song: eight-class emotion
recognition, speech-versus-song classification, and regression of the supplied
`kur` feature. This public repository presents the subsequent refactoring and
rebuild of an academic project, with reusable Python code, explicit experiment
settings and documented results.

[Results](#selected-results) · [Code guide](#code-guide) ·
[Setup](#setup-and-checks) · [Methodology](#methodology-and-reproducibility) ·
[Public scope](#public-repository-scope)

## Selected results

These are completed rebuilt experiments on the original actor holdout.
Their saved predictions and artifacts were checked when the runs completed.

| Task | Experiment | Test accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Eight emotions | Cleaned-feature random forest | 42.15% | 0.3943 |
| Eight emotions | Gradient boosting | 45.99% | 0.4520 |
| Eight emotions | Dense network, actor-fold parameter search | 51.44% | 0.5064 |
| Binary emotion task | Linear SVM | 67.99% | 0.6306 |
| Speech versus song | MiniROCKET | 98.56% | 0.9852 |

Regression selected Lasso with alpha=0.1: test MSE **9.6861**, RMSE **3.1122**,
MAE **1.7826**, and R² **0.7516**.

Speech/song and binary emotion classification are different tasks from
recognizing eight emotions. Feature sets and fitting protocols also differ,
so these scores are not a controlled leaderboard. No statistical improvement
over the academic results is claimed.

The [detailed results table](rebuilt/docs/results.md) separates completed rebuilt
experiments, original saved notebook scores and cancelled runs. Each experiment
links to its JSON report with parameters, fold scores, metrics and provenance.

## Code guide

The project demonstrates preprocessing pipelines, feature reduction, model
selection, classical ML, dense neural networks, waveform representations,
actor-based evaluation and saved-model verification.

| Area | Start here | What it covers |
| --- | --- | --- |
| Data and preprocessing | [data.py](rebuilt/src/ravdess_project/data.py), [preprocessing.py](rebuilt/src/ravdess_project/preprocessing.py) | Schema checks, feature/label alignment and train-fitted transforms |
| Validation and reporting | [validation.py](rebuilt/src/ravdess_project/validation.py), [reporting.py](rebuilt/src/ravdess_project/reporting.py) | Actor-disjoint folds, metrics, warnings and reproducibility records |
| Classical classification | [tabular.py](rebuilt/src/ravdess_project/tabular.py), [model_search.py](rebuilt/src/ravdess_project/model_search.py) | Forests, PCA/RFE/feature selection, SVM and boosting searches |
| Regression | [regression.py](rebuilt/src/ravdess_project/regression.py) | Linear, regularized and ensemble models for `kur` |
| Neural networks | [neural.py](rebuilt/src/ravdess_project/neural.py) | Dense-network variants, parameter search and checkpoint inference |
| Waveforms | [audio_preparation.py](rebuilt/src/ravdess_project/audio_preparation.py), [time_series.py](rebuilt/src/ravdess_project/time_series.py), [sax_dtw.py](rebuilt/src/ravdess_project/sax_dtw.py) | Audio import, Euclidean neighbors, ROCKET/MiniROCKET and SAX/DTW |
| Experiment settings | [configs/](rebuilt/configs) | Task definitions, original parameters and explicit seeds |
| Quality checks | [tests/](rebuilt/tests), [scripts/](rebuilt/scripts) | Small fixtures, artifact checks and results-summary consistency |

```text
rebuilt/
  src/ravdess_project/   ML package and command-line interface
  configs/              Experiment configurations
  tests/                Fixture-based checks
  scripts/              Execution, verification and reporting utilities
  docs/                 Detailed results table and JSON evidence
  requirements*.txt     Tested dependency snapshots
  pyproject.toml        Package definition
.github/workflows/      Core CI configuration
```

## Setup and checks

From the repository root, use Python 3.11 or newer:

```bash
python -m venv rebuilt/.venv
source rebuilt/.venv/bin/activate
python -m pip install -e ./rebuilt
python -m unittest discover -s rebuilt/tests -v
python rebuilt/scripts/summarize_results.py --check
python -m ravdess_project --help
```

With `uv`, create the environment using `uv venv rebuilt/.venv`, then install
with `uv pip install --python rebuilt/.venv/bin/python -e ./rebuilt`.
Core dependencies are NumPy, scikit-learn and joblib. Optional `xgboost`, `neural`
and `timeseries` extras have separate tested dependency snapshots in `rebuilt/`.
The core snapshot records Python 3.14.7; neural and time-series snapshots record
Python 3.13.14 on Linux x86_64. Version pins do not lock wheel hashes or build
tooling, and other runtime/platform combinations are not guaranteed.

Tests use small fixtures and skip optional integrations when their dependencies
are absent. The summary check reads saved reports without datasets or training.
The recorded local time-series suite passed 64 tests with three optional skips.

To audit your own copy of the supplied feature tables:

```bash
ravdess audit --data-dir /path/to/feature-tables --config rebuilt/configs/baseline.toml
```

That directory must contain `RavdessAudioOnlyFeatures_TRAIN.csv` and
`RavdessAudioOnlyFeatures_TEST.csv`. Cleaned-feature configurations instead
require `Train_NOoutliers` and `Test_ripulito (1)`. Waveform configurations require
the original train/test NPY arrays and Y metadata CSVs; raw import also needs
WAV recordings. All inputs must be supplied separately.

The CLI supports `train`, `compare`, `search`, `regress`, `neural` and `series`;
use the matching configuration and a fresh `--output` directory. Commands that
train models are separate from the lightweight checks above. Full searches can
be expensive; the cancelled long experiments are not part of the quick start.

## Methodology and reproducibility

- Actors 01–18 supply training data and actors 19–24 supply the fixed test set.
  Rebuilt model selection uses training actors, with preprocessing fitted within
  folds where applicable. Per-waveform min/max scaling operates independently
  on each recording.
- The rebuild preserves documented tasks, features and original model settings.
  Actor-based validation, previously absent seeds and current library runtimes
  are explicit changes. RFE uses true labels rather than the prior predictions
  used in the original notebook. The binary SVM uses the coherent cleaned pair
  because its original test file is missing.
- The five neural variants retain their original settings, including the
  one-epoch early-stopping-only variant. Dropout selection uses 1,395 fitting
  rows and 414 validation rows; the searched network refits on all 1,809 training
  rows. Regression includes emotion as a predictor and does not classify it.
- Raw WAV import reproduced both original NPY hashes across 2,452 recordings.
  A compression candidate differs by up to 2.95e-7, so analyses retain the original
  compressed inputs. Tabular extraction and complete cleaning code are unavailable.
- Rebuilt emotion Euclidean accuracy is 12.82%, versus the saved rounded 17%.
  The discrepancy remains unresolved. Fixed-setting regression diagnostics match
  three saved MSE values, but the historical linear-regression error is not
  reproduced. Numerical proximity does not prove historical equivalence.
- The holdout was already examined in the academic project. Recorded validation
  scores are selection statistics, not unbiased nested-CV estimates. Fixed seeds
  and dependency snapshots do not guarantee identical results across runtimes.

The full emotion ROCKET comparison and both full SAX/DTW evaluations were
cancelled because of runtime. Their partial checkpoints remain outside this
checkout and have no final scores. Original notebook scores are reported as
historical observations. Shapelets, clustering, motifs, remaining ensembles,
exploratory embeddings and explanations are not fully rebuilt.

## Public repository scope

The public [repository](https://github.com/alexdega95/ravdess-ml) contains source,
configurations, tests and small results reports. Datasets, original notebooks,
third-party academic libraries, model binaries, environments, run directories
and archives remain in the separate preserved academic workspace and are
excluded from Git. Historical notebook paths and the legacy preservation utility
refer to that workspace, not files available in this public checkout.

Publication copies shorten machine-specific input paths to filenames while
retaining metrics, settings and recorded data/model hashes. Repeating experiments
requires the external inputs and matching dependencies; installing this package
alone does not recreate the missing upstream preprocessing or original artifacts.

The [CI workflow](.github/workflows/ci.yml) is configured for core fixture tests,
result-summary checks, a CLI check and a 5 MB tracked-file limit on Python 3.14.
Optional backends and full experimental searches are excluded from core CI.
Successful GitHub Actions runs could not be verified during this documentation
update; consult the [Actions page](https://github.com/alexdega95/ravdess-ml/actions)
for current status.
