# Reproducible analysis package

This directory contains the independent Python rebuild of the academic RAVDESS
project. Original notebooks and binary artifacts are preserved separately and
are not distributed in this Git checkout.
Start with the [project overview](../README.md) and
[results summary](docs/results.md); no installation is needed to read them.

## Install the core package

Run from the **repository root** with Python 3.11 or newer:

```bash
python -m venv rebuilt/.venv
source rebuilt/.venv/bin/activate
python -m pip install -e ./rebuilt
```

With `uv`:

```bash
uv venv rebuilt/.venv
uv pip install --python rebuilt/.venv/bin/python -e ./rebuilt
```

Core dependencies are NumPy, scikit-learn and joblib, with transitive runtime
packages. Audits and tabular analyses do not require TensorFlow or time-series
backends. Optional integrations have separate tested environments.

| Dependency snapshot | Recorded runtime | Workflows |
| --- | --- | --- |
| `requirements.lock.txt` | Python 3.14.7, Linux x86_64 | Core tabular/regression |
| `requirements-xgboost.lock.txt` | Python 3.14.7, Linux x86_64 | XGBoost |
| `requirements-neural.lock.txt` | Python 3.13.14, Linux x86_64 | TensorFlow/Keras |
| `requirements-timeseries.lock.txt` | Python 3.13.14, Linux x86_64 | sktime/tslearn/pyts |

To use a snapshot, install it in the matching separate environment, then install
the package with `--no-deps`. For the recorded core runtime:

```bash
python -m pip install -r rebuilt/requirements.lock.txt
python -m pip install --no-deps -e ./rebuilt
```

Snapshots pin package versions; they do not lock wheel hashes or build tooling.
Other runtimes/platforms have not been verified. Optional setup commands are in
the [execution reference](docs/execution.md).

## Supply the data

| Workflow | Required input files | Config |
| --- | --- | --- |
| Raw-feature audit/baseline | `RavdessAudioOnlyFeatures_TRAIN.csv`, `RavdessAudioOnlyFeatures_TEST.csv` | `configs/baseline.toml` |
| Cleaned-feature audit/comparison | `Train_NOoutliers`, `Test_ripulito (1)` | `configs/tabular.toml` |
| Waveform audit | Original train/test NPY arrays and Y metadata CSVs | `configs/timeseries_emotion_euclidean.toml` |

In the original academic workspace, feature tables were linked through
`ravdess_dm2_2223/`; cleaned tables were in `data/`; waveform links were at the
project root. Those links and data are excluded from this Git checkout.
Provide an explicit directory containing your input files.
Downloading WAVs alone does not reconstruct the unavailable tabular
extraction/cleaning steps.

## Lightweight checks

From the repository root, after installing the core package:

```bash
ravdess audit --data-dir /path/to/feature-tables --config rebuilt/configs/baseline.toml
python -m unittest discover -s rebuilt/tests -v
python rebuilt/scripts/summarize_results.py --check
```

`audit` checks schemas, finite values, row/filename uniqueness, labels and actor
separation. Tests use small fixtures; optional tests are skipped when their
dependencies are unavailable. The summary check reads saved reports without
reading datasets or loading models. The legacy preservation utility applies
only to the full academic workspace; see [publication scope](../PUBLISHING.md).

The latest complete suite in the time-series environment passed 64 tests with
three optional neural/XGBoost tests skipped. Earlier dedicated backend checks
are recorded in [validation](docs/validation.md).

## Results and execution

[Results and status](docs/results.md) distinguish completed rebuilt experiments,
original saved notebook scores and cancelled runs. Its machine-readable index
is [results_summary.json](docs/results_summary.json). Regenerate only these
documentation summaries with:

```bash
python rebuilt/scripts/summarize_results.py
```

The [execution reference](docs/execution.md) retains detailed workflow commands.
Commands that train models require a fresh output directory; original artifacts
are never overwritten. Reports contain settings, hashes, versions, warnings and
metrics; local runs hold predictions and models. `runs/` and virtual environments
are gitignored.

Large experiments are cancelled at the user's request. Do not launch comparable
searches or resume cancelled emotion ROCKET/SAX-DTW jobs without an explicit
request. Completed results and partial checkpoints are preserved separately.

## Package layout

`src/ravdess_project/` contains shared loading, preprocessing, actor-aware
validation and reporting, plus tabular, regression, neural and waveform modules.
`configs/` holds settings, `tests/` holds checks, `scripts/` provides reporting
and verification, and `docs/` holds evidence and methodology.

See the [experiment map](docs/experiment_map.md) for retained settings and explicit
evaluation changes, and [methodology](docs/methodology.md) for unresolved inputs
and reproduction limits. The rebuild does not claim reproduction of every
academic experiment.
