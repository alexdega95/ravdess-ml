# Time-series preprocessing and rebuilt analysis

This milestone rebuilds audio preparation and the Euclidean classification
experiments from `time_series_classification.ipynb` and
`time_series_classification_vocal.ipynb`. Original notebooks, data and models are
unchanged. ROCKET/MiniROCKET backends are now installed and integration-validated. Full
vocal-channel comparison has completed and passed verification; the emotion
comparison was cancelled at the user's request because of its duration. Use
[original saved results](legacy_timeseries_results.md) for historical reporting.
Do not restart large experiments without an explicit request.

## What preprocessing actually does

1. Enumerate WAV files and decode the seven filename fields into labels. Sort
   by actor and filename, keeping waveform rows and labels together.
2. Decode original 48 kHz, signed 16-bit PCM into float32 amplitudes. Some files
   have two channels: average their decoded channels into one mono waveform.
   No new resampling, silence trimming or amplitude normalization is introduced.
3. Assign actors 01–18 to training and 19–24 to testing. There are 1,828 training
   and 624 test recordings. Actor 18 lacks song recordings in the supplied data;
   no missing recordings are invented.
4. Obtain the padding length from training only: 304,304 samples. Pad shorter
   recordings with trailing NaNs and clip longer test recordings to this length,
   matching the original `pad_none(..., clip=True)` operation. One test recording
   is clipped. Save float64 arrays, matching the original Awkward-to-NumPy export.
5. Treat compression separately. Its generation code is absent. A candidate uses
   SciPy decimation by ten on float32 finite waveforms, eighth-order IIR filtering
   and zero-phase filtering, then NaN padding to 30,431 samples. Its finite/padding
   positions match supplied compressed arrays, but its values are not bit-identical.
6. Crop each experiment's stated sample interval, then scale each recording
   independently into `[0, 1]`. Reject NaNs inside the chosen interval instead of
   silently imputing them. Both selected classification intervals are finite.

The import notebook's full audio-loading behavior was recoverable. The rebuilt
PCM decoder deliberately supports the formats actually observed and rejects
other formats rather than guessing conversions. All WAV contents have recorded
hashes; generated metadata rows match the supplied labels exactly.

The notebooks contain moving averages and detrending in exploratory plots.
Those operations do **not** feed these classifiers, so they were not added to
the modeling pipeline. Imports of `decimate` alone do not establish how the
historical compressed arrays were generated.

## Validation of regenerated arrays

All 2,452 decoded waveforms, row positions, labels, NaN padding and clipping were
compared against the supplied artifacts. The new **raw train and test NPY files
have exactly the same SHA-256 hashes as the originals**. This verifies the whole
raw import, not just a few samples.

The compression candidate has maximum absolute differences of 2.95229256e-7
(training) and 8.26548785e-8 (test). It fails exact float32 equality. The original
implementation/runtime remains unknown, so compression equivalence is not
claimed. Filtering or runtime differences are possible explanations, not proven
causes. Both classification runs use the **supplied** raw/compressed arrays.
The candidate never silently replaces historical compressed data.

[Preparation results](audio_preparation_results.json) record the measured
comparisons and output hashes. The full local `runs/audio-preparation-001/`
contains the six regenerated data files, per-recording WAV hashes and full report.
It occupies roughly 6.3 GB, separately from the original arrays.

## Keep experiment inputs distinct

Intervals below use NumPy's exclusive upper bound. Duplicate notebook copies
remain untouched; their settings/results must not be combined indiscriminately.

| Original analysis | Representation | Interval | Target or scope |
| --- | --- | --- | --- |
| Emotion classification | Raw | `[45000:100000]`, 55,000 samples | Eight emotions |
| Vocal-channel classification | Compressed | `[5000:13000]`, 8,000 samples | Speech/song |
| Emotion DTW | Raw | `[45000:130000]` | Eight emotions; SAX 1,000 segments, alphabet 50 |
| Vocal-channel DTW | Compressed | `[5000:13000]` | Speech/song; SAX 1,000 segments, alphabet 50 |
| Shapelet notebooks | Compressed | `[5000:13000]` | Separate emotion and vocal-channel tasks |
| `ts_clustering.ipynb` | Compressed | `[4800:13000]` | All recordings and speech-only exploration |
| `ts_clustering_compresso.ipynb` | Compressed | `[4000:13000]` | Different window and representation sizes |
| Matrix profile | Compressed | `[4000:13000]` | Motif/discord exploration |

DTW, shapelets, clustering and motif analysis remain separate pending experiments.
They must retain their own targets, intervals, representations and parameters.
The loader/scaler can be reused, but this milestone does not invent their metrics.

## Completed Euclidean runs

The original pyts Euclidean classifier delegates to sklearn with one neighbor,
uniform weights and the stated Euclidean metric. The rebuild uses that same
backend directly; it does not accidentally adopt sklearn's five-neighbor default.
Label names remain readable in saved predictions.

Per-recording min/max scaling is stateless. Calling it independently on test rows
is correct here; it does not fit a shared scaling range from test data. This is
different from fitting a global StandardScaler on test data. The scaling step
lives in each fitted pipeline so inference consistently applies it.

A new five-fold actor comparison selects between the fixed Euclidean method and
majority reference by mean macro F1. No test data enters selection. The selected
method then fits all training actors and receives a final test evaluation.

| Task | Model | Test accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Emotion | Majority | 15.38% | 0.033333 |
| Emotion | Euclidean 1-NN | 12.82% | 0.091962 |
| Vocal channel | Majority | 57.69% | 0.365854 |
| Vocal channel | Euclidean 1-NN | 58.17% | 0.378208 |

Euclidean has higher macro F1 than the reference in training folds, so it is
selected for both tasks. Emotion accuracy is lower than the reference; this
negative result is retained. The original saved emotion accuracy is rounded to
17%, which the new 12.82% does not reproduce. The original runtime/execution state
was not recovered, so the discrepancy's cause remains unresolved. No settings
were changed to make the scores resemble the historical output. Vocal-channel
accuracy matches the original rounded 58%, but that does not prove identical
historical predictions.

Both saved pipelines were reloaded and every one of their 624 predictions and
metrics reproduced. Independent record-by-record normalization matched both
train/test transforms exactly; an independent notebook-style run using the
original Euclidean backend/defaults produced exactly the same current predictions.
Thus the software reconstruction is internally verified even though one saved
historical score differs. Direct tslearn/pyts integration tests now pass. A literal recreation using those
libraries on all real recordings reproduces both rebuilt baseline predictions
exactly (zero different test predictions), including the 12.82% emotion result.
Thus installing the original APIs does not resolve the saved 17% discrepancy.
[Direct backend comparison](timeseries_original_backend_check.json) records this check.

Results: [emotion](timeseries_emotion_results.json),
[vocal channel](timeseries_vocal_results.json). Verification:
[emotion](timeseries_emotion_validation.json),
[vocal channel](timeseries_vocal_validation.json). Each full local run retains
its model bundle, predictions, actor-fold indices, input/metadata hashes, source
hashes and runtime log. Earlier run reports retain the source hashes of their
own execution as the implementation evolves.

## ROCKET and MiniROCKET

The full configurations retain 10,000 requested kernels, ROCKET's default internal
normalization, MiniROCKET's 32 maximum dilations per kernel, and
`RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`. The transform uses sktime's public
3D representation instead of a private nested-DataFrame converter. New seed 42
makes previously unseeded transforms explicit. Five actor folds select methods;
the ridge classifier retains its original internal alpha-selection defaults.
Changing that inner procedure requires a separate methodological decision.

The installed runtime is sktime 0.40.1, tslearn 0.9.0, pyts 0.14.0, NumPy 2.3.5,
Numba 0.68.0 and scikit-learn 1.7.2; exact dependencies are in
`requirements-timeseries.lock.txt`. This environment is separate from TensorFlow.
All 64 tests ran: 61 passed, with only two TensorFlow tests and optional XGBoost
skipped. No time-series integration tests are skipped. Small 84-kernel fixtures
validate serialization and execution parity; full runs retain 10,000 kernels.

### Execution optimization without changing methods

The installed ROCKET implementation generates seeded kernels from channel count
and interval length only. It does not learn from waveform values or labels.
The rebuild generates them from a one-record, zero-valued shape placeholder,
then transforms the training recordings once. The ridge classifiers still fit
only the actor-fold fitting rows. No validation observations are used to fit
kernels or classifiers. MiniROCKET's data-dependent biases remain fitted in each
actor fold; its feature fitting is never shared across folds.

Integration tests compare cached versus ordinary pipelines: kernels, feature
values, every fold metric, candidate parameters, tie order and selected method
are identical. Feature generation in batches also matches direct execution.
Checksummed batch checkpoints support interruption/resume and detect corruption.
The cache rejects mismatched inputs, software, source or transform parameters.

Long jobs freeze source/configuration under their feature directory, so future
workspace edits do not change a running experiment. The tested job runner then
verifies the selected model, every test prediction, metrics, actor isolation,
preprocessing and input/config/source hashes before exporting results. A complete
small fixture tested that entire launch/verification/export chain; fixture scores
are not reported as RAVDESS results.

Job directories: `runs/timeseries-emotion-full-003-features/` and
`runs/timeseries-vocal-full-003-features/`. The vocal job is complete; the
emotion job is cancelled. Their `job_status.json` records the
stage and completion/failure; `state.json` records how many of the 1,828 training
recordings have cached ROCKET features. That count measures feature generation,
not overall completion: actor CV, final fitting and verification follow it.
At the observed rate, raw-waveform ROCKET can take several hours. Original counts
and windows are preserved instead of substituting a smaller experiment.

After successful completion, the runner writes
`timeseries_emotion_full_results.json` / `timeseries_vocal_full_results.json`,
matching `_validation.json` files, and separate `_analysis.md` pages in `docs/`.
The completed vocal comparison selects MiniROCKET: mean actor-CV macro F1
0.975574, test accuracy **98.56%** and test macro F1 **0.985249**. All twenty
candidate/fold fits completed; all 624 test predictions and metrics were verified
after reload, along with actor isolation and input/config/frozen-source hashes.
Its requested 10,000 kernels yield 9,996 MiniROCKET features through the library's
standard rounding; this is not a manually reduced full-data model. The original
vocal MiniROCKET test accuracy is rounded to 99%, so the result is nearby but
historical numerical equivalence is not claimed.

[Completed vocal analysis](timeseries_vocal_full_analysis.md),
[full comparison results](timeseries_vocal_full_results.json),
[verification](timeseries_vocal_full_validation.json).

No full rebuilt emotion-comparison metrics exist: that job was cancelled.
Automatic verification used frozen execution sources for the completed vocal
job. Earlier execution attempts and the final unfinished emotion run have
retained checkpoints; they must not be restarted without an explicit request.

The compression candidate was checked again across all 2,452 recordings using
the new NumPy/runtime. It is exactly identical to the prior candidate and has the
same differences from the originals. This rules out the tested runtime change
as the explanation, but does not identify the missing historical implementation.
[Runtime comparison](compression_runtime_comparison.json) records the measured
maxima and the affected recordings.

## Changes that should be considered separately

- Establish the original compression implementation/runtime before claiming its
  exact reproduction, or approve the measured candidate as a new artifact.
- Change clipping, padding, cropping or silence handling only as a new experiment.
  Those choices affect the waveform content and classification task.
- Improve the weak Euclidean baseline with other distances or representations as
  separately evaluated methods, rather than rewriting its historical settings.
- Consider actor-grouped inner alpha tuning after validating the original ridge
  configuration; changing it now would mix refactoring with methodology changes.

References: [tslearn per-series scaling](https://tslearn.readthedocs.io/en/stable/gen_modules/preprocessing/tslearn.preprocessing.TimeSeriesScalerMinMax.html),
[pyts Euclidean backend](https://pyts.readthedocs.io/en/stable/_modules/pyts/classification/knn.html),
[ROCKET example](https://www.sktime.org/en/latest/examples/rocket.html).

## Implementation files

- `src/ravdess_project/audio_preparation.py`: filename decoding, PCM loading,
  actor ordering, padding/clipping, separate compression candidate and comparison.
- `src/ravdess_project/time_series.py`: validated waveform loader, row-wise scaler,
  fixed Euclidean classifier and optional convolution adapters/actor comparison.
- `src/ravdess_project/cli.py`: `series-audit` and `series` commands.
- `src/ravdess_project/reporting.py`: prediction headers use the configured target,
  preserving existing emotion/regression output while supporting vocal channel.
- `src/ravdess_project/tabular.py`: feature reporting handles convolution outputs
  without incorrectly treating them as selected original waveform samples.
- `configs/timeseries_{emotion,vocal}.toml` and their `_euclidean` variants:
  full original-model comparisons and separately runnable core-only experiments.
- `scripts/prepare_audio.py`, `scripts/verify_timeseries_run.py`,
  `tests/test_time_series.py`, optional dependencies in `pyproject.toml`, and
  updated README/methodology/validation/results documents complete this milestone.

The long-job launcher is `scripts/run_timeseries_job.py`. Its source snapshot,
configuration, logs, status and automatic verification make the remaining work
repeatable without requiring an interactive session to remain open.
