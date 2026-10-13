# Original SAX/DTW analysis

This rebuild extracts `dtw.ipynb` and `dtw_vocal.ipynb` into a separate module.
Original notebooks and input arrays are preserved. It does not select a winner
using the held-out test set or add cross-validation to the notebook experiment.
All three original distances are reported independently.

| Task | Input | Interval | SAX segments | Alphabet |
| --- | --- | --- | ---: | ---: |
| Emotion | Original raw waveform | `[45000:130000]` | 1,000 | 50 |
| Vocal channel | Original compressed waveform | `[5000:13000]` | 1,000 | 50 |

Each recording is scaled to `[0,1]` using its own minimum and maximum, then
transformed with tslearn `SymbolicAggregateApproximation`. Its original
`scale=False` default is explicit. Gaussian breakpoints are preserved; adding
z-score normalization would change the analysis. The classifier treats SAX
symbols as numeric sequences, exactly as the notebooks do, rather than using
SAX's own symbolic distance.

The three pyts classifiers are `KNeighborsClassifier(metric='dtw')`,
`metric='dtw_sakoechiba'`, and `metric='dtw_itakura'`. All other defaults are
preserved, including one neighbor and the default constraint settings.
Training uses actors 1–18 and testing actors 19–24.

Preparation independently executes the notebook's tslearn scaler and SAX calls
on every training and test recording and checks exact equality of all symbols.
It records input, metadata, configuration, source, environment and artifact
hashes in `preparation.json`. Fixtures additionally compare all three distances
against direct notebook calls and verify saved models after reloading.

DTW compares each of 624 queries with all 1,828 training recordings. Its dynamic
programming matrices operate on 1,000 symbols, so full evaluation can take many
hours. Prediction batches change execution only: every query still searches all
training recordings in their original order. `status.json` atomically saves
completed predictions with a checksum. Restart the same command to resume;
changed execution code, packages, inputs or artifacts are rejected. No shorter
window, smaller alphabet, reduced training set or alternative distance is
substituted to speed up the scientific experiment.

## Current validation and runs

All SAX symbols match the notebook preprocessing exactly on the real data for
both tasks: 1,828 training and 624 test recordings each. Recorded evidence is
in [vocal preprocessing](sax_dtw_vocal_preparation.json) and
[emotion preprocessing](sax_dtw_emotion_preparation.json).
The full DTW evaluations in `runs/sax-dtw-vocal-001` and
`runs/sax-dtw-emotion-001` were cancelled at the user's request because of their
duration. Partial prediction checkpoints are preserved, but no full rebuilt
classification score is reported. Use the
[original saved results](legacy_timeseries_results.md) for reporting. Do not
restart these jobs or launch comparable large runs without an explicit request.

The complete test suite passed 64 tests with three optional neural/XGBoost tests
skipped in the time-series environment. New tests cover direct notebook parity
for all three distances, reloaded models, single-record prediction, interrupted
and resumed evaluation, and corruption/configuration rejection. All 82 original
files and links passed the legacy preservation check.

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/ravdess-numba-cache \
  rebuilt/.venv-timeseries/bin/python rebuilt/scripts/run_sax_dtw.py \
  --data-dir . --config rebuilt/configs/sax_dtw_vocal.toml \
  --output rebuilt/runs/new-sax-dtw-vocal-001 --batch-size 8
```

Use `sax_dtw_emotion.toml` and a separate output directory for emotion. Add
`--prepare-only` to validate and save preprocessing without starting prediction.
After preparation, omit that option with the same directory to start evaluation.

Saved notebook accuracies are rounded: emotion 0.23/0.24/0.25 and vocal channel
0.76/0.78/0.80 for classic/Sakoe–Chiba/Itakura respectively. These are historical
observations, not new results. Exact historical prediction equivalence cannot
be established from these rounded reports alone. A completed run writes
`report.json`, `predictions.csv` and `models.joblib`; a preparation report or
partially populated checkpoint is not a completed classification result.

Implementation: `src/ravdess_project/sax_dtw.py`, `scripts/run_sax_dtw.py`,
`configs/sax_dtw_{emotion,vocal}.toml`, and `tests/test_sax_dtw.py`.
