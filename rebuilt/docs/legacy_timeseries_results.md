# Original time-series results

These are saved outputs from the original notebooks, rounded to their displayed precision. They are not new rebuilt results. All reports use 624 test recordings.

| Task | Method | Accuracy | Macro F1 | Original notebook (zero-based cell) |
| --- | --- | ---: | ---: | --- |
| emotion | SAX + DTW | 23% | 0.21 | `dtw.ipynb` (5) |
| emotion | SAX + Sakoe–Chiba DTW | 24% | 0.22 | `dtw.ipynb` (6) |
| emotion | SAX + Itakura DTW | 25% | 0.25 | `dtw.ipynb` (7) |
| vocal_channel | SAX + DTW | 76% | 0.75 | `dtw_vocal.ipynb` (5) |
| vocal_channel | SAX + Sakoe–Chiba DTW | 78% | 0.78 | `dtw_vocal.ipynb` (6) |
| vocal_channel | SAX + Itakura DTW | 80% | 0.79 | `dtw_vocal.ipynb` (7) |
| emotion | ROCKET | 33% | 0.33 | `time_series_classification.ipynb` (17) |
| emotion | MiniROCKET | 55% | 0.54 | `time_series_classification.ipynb` (19) |
| emotion | Euclidean 1-NN | 17% | 0.11 | `time_series_classification.ipynb` (20) |
| vocal_channel | ROCKET | 92% | 0.92 | `time_series_classification_vocal.ipynb` (17) |
| vocal_channel | MiniROCKET | 99% | 0.99 | `time_series_classification_vocal.ipynb` (19) |
| vocal_channel | Euclidean 1-NN | 58% | 0.38 | `time_series_classification_vocal.ipynb` (20) |

The copied vocal-channel notebook under `TIME SERIES/classification/vocal channel/` reports ROCKET accuracy/macro F1 of 0.93/0.93 and MiniROCKET of 0.98/0.98, versus 0.92/0.92 and 0.99/0.99 in the root notebook. Both versions are retained; neither is presented as the uniquely authoritative result. Other duplicated classification reports agree.

[Source reports, notebook hashes and cell references](legacy_timeseries_results.json) preserve all versions, including `dtw copy.ipynb`.

Full rebuilt SAX/DTW runs were cancelled at the user’s request. Their partial prediction checkpoints are retained and are not reported as complete scores. The unfinished raw emotion ROCKET run is also cancelled. Completed rebuilt results remain separate and are preserved. Do not restart these jobs or launch experiments of comparable duration unless the user explicitly requests them.
