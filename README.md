# RAVDESS-ML: audio emotion and time-series analysis

An academic machine-learning study of acted speech and song, rebuilt as a Python
package with explicit experiment settings, actor-based validation, saved results,
and reproducibility checks.

The project studies eight-class emotion recognition using audio features and
waveforms, speech-versus-song classification, and regression of the supplied
`kur` feature. This Git checkout presents the rebuilt implementation and its
results. Original notebooks, datasets and model binaries remain preserved in
the separate academic workspace.

**[Results and experiment status](rebuilt/docs/results.md)** ·
[Setup and usage](rebuilt/README.md) ·
[Methodology](rebuilt/docs/methodology.md) ·
[Original time-series scores](rebuilt/docs/legacy_timeseries_results.md)

## ML skills demonstrated

The implemented workflows cover feature auditing, preprocessing pipelines,
dimensionality reduction, classical classifiers, boosting, regularized
regression, dense neural networks and waveform-based classification. The main
engineering focus is separating model selection from test evaluation, fitting
preprocessing within actor folds, preserving model/class encodings for inference,
and checking saved predictions against reproducible result reports.

## Main findings

These are completed rebuilt experiments on the original actor holdout.
Their saved predictions and artifacts were checked when the runs completed.

| Task | Experiment | Test accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Eight emotions | Cleaned-feature gradient boosting | 45.99% | 0.4520 |
| Eight emotions | Dense network, actor-fold parameter search | 51.44% | 0.5064 |
| Speech versus song | MiniROCKET | 98.56% | 0.9852 |

Speech/song accuracy concerns a separate two-class task. Different feature sets
and training protocols also prevent interpreting the emotion results as a
controlled leaderboard. The [complete summary](rebuilt/docs/results.md) includes
baselines, binary SVM, regression, historical outputs and unresolved differences.
No statistical improvement over the original work is claimed.

## Engineering approach

- Python modules share loading, preprocessing, validation and artifact reporting.
- TOML files separate experiment settings from implementation.
- Training actors 01–18 and test actors 19–24 remain separate. Model selection
  uses training actors, with fold-specific preprocessing where applicable.
- Run artifacts record input/configuration/source hashes, software versions, warnings,
  predictions and metrics.
- Tests cover data contracts, actor isolation, pipelines, model reloading and
  checkpoint recovery; optional backends have separate tested environments.
- A legacy manifest records preservation checks from the original workspace.
- GitHub Actions checks core fixture tests, imports and saved-result consistency.

Evaluation changes such as actor-based validation and previously absent seeds
are documented in the [experiment map](rebuilt/docs/experiment_map.md).
Original saved scores and new rebuilt scores are labelled separately.

## Quick start

From the repository root, install with Python 3.11 or newer:

```bash
python -m venv rebuilt/.venv
source rebuilt/.venv/bin/activate
python -m pip install -e ./rebuilt
```

You need copies of `RavdessAudioOnlyFeatures_TRAIN.csv` and
`RavdessAudioOnlyFeatures_TEST.csv`. Set their directory explicitly;
this workspace's external links are not portable.

```bash
ravdess audit --data-dir /path/to/feature-tables --config rebuilt/configs/baseline.toml
python -m unittest discover -s rebuilt/tests -v
python rebuilt/scripts/summarize_results.py --check
```

These commands audit data, run fixture-based tests and check documentation;
they do not run full experimental searches. Viewing saved results requires no
installation or dataset. [Detailed setup](rebuilt/README.md) covers `uv`, tested
dependency snapshots and the cleaned-feature inputs.

## Repository layout

```text
rebuilt/
  src/ravdess_project/   Reusable analysis package and CLI
  configs/              Explicit experiment configurations
  tests/                Data, pipeline and artifact checks
  scripts/              Verification and reporting tools
  docs/                 Results, methodology and validation evidence
  requirements*.txt     Tested dependency snapshots
  runs/                 Local artifacts, created on execution (gitignored)
.github/workflows/       Core tests and result-summary checks
PUBLISHING.md            Publication scope and GitHub push instructions
```

Datasets, notebook duplicates, model weights, virtual environments and archives
are excluded from this checkout. Provide data explicitly to run experiments;
fixture tests and saved-result checks do not need it. Raw WAV import was
reproduced exactly, but tabular extraction and complete cleaning code remain
unavailable. Reproduction limits are visible in the
[results summary](rebuilt/docs/results.md). [Publishing notes](PUBLISHING.md)
explain historical references and which artifacts remain outside this repository.

## Current scope

Tabular, SVM, boosting, regression, neural and selected waveform analyses have
completed rebuilt results. Full emotion ROCKET and both SAX/DTW evaluations were
cancelled because of runtime; partial outputs are preserved outside this checkout
and have no final scores. Original notebook scores are documented for reporting.

Remaining work includes extraction/cleaning provenance and further organization
of original ensembles, exploratory embeddings, shapelets, clustering, motifs
and explanations. Comparable long experiments will not be launched or resumed
without an explicit request.
