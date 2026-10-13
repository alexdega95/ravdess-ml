# Neural experiments

The rebuilt workflow retains the eight emotion classes and the original 202
predictors: five categorical metadata fields and 197 numeric features. Emotion
and filename are excluded from the inputs. All original rows and actor partitions
are retained. Label encoding and feature scaling are fitted on training rows.

`configs/neural.toml` describes the five variants. Hidden layers retain 128 and
64 ReLU units; the output retains eight softmax units, sparse categorical
cross-entropy, and Adam with learning rate 0.001.

| Variant | Maximum epochs | Original distinguishing settings |
| --- | ---: | --- |
| Baseline | 200 | No regularizer, dropout, or early stopping |
| Early stopping | 1 | Monitor validation accuracy, patience 100 |
| L1 | 500 | Strength 0.01 on both hidden layers; early stopping |
| L2 | 500 | Strength 0.01 on both hidden layers; early stopping |
| Dropout | 500 | Rate 0.1 after each hidden layer; batch size 100; early stopping |

Other variants retain batch size 32. The one-epoch early-stopping variant reflects
SciKeras's default: the original notebook did not specify epochs. Extending it
would be a methodological change. Callback instances are fresh for each model.
Regularized and early-stopping variants reload their best validation-accuracy
checkpoint, matching the notebook's subsequent checkpoint-based comparisons;
the baseline uses its final epoch model.

The new validation partition holds out approximately 20% of training actors
(four of eighteen), replacing Keras's final 20% of rows. The scaler fits only on
the remaining fitting actors. Variants are compared by validation accuracy,
and only the selected variant and majority reference are evaluated on the test
partition. The selected variant retains its fitting-actor training set; the report
records those actors. It is not silently retrained with a different stopping rule.

`configs/neural_search.toml` retains the original 15 sampled candidates, five-fold
macro-F1 selection, hidden-layer lists, activations, optimizer choices, learning
rates and epoch values. The duplicate `[10,10,10]` option remains because removing
it would change sampling probabilities. The rebuilt search fits preprocessing
inside actor folds and then refits the winner on all training actors for its
configured epoch count. Candidate/fold fits within the search run serially. The separate five-variant
workflow and search ran concurrently with one compute thread per backend.

## Explicit changes and limits

Direct TensorFlow/Keras calls replace the SciKeras wrapper; architecture, loss,
optimizer and training settings are explicit. Seed 42 and deterministic operations
are new choices because original network seeds were absent. Current native
`.keras` checkpoints replace new HDF5 output files; legacy HDF5 models are untouched.
These runtime changes prevent a claim of historical numerical equivalence.
Original saved neural scores remain historical results.

## Completed variant analysis

The run uses 1,395 fitting rows from fourteen training actors. Actors 1, 2, 6
and 9 supply the 414 validation rows. Test actors remain 19–24, with 624 rows.
The table below reports validation scores, not five separate test evaluations.

| Variant | Actual epochs | Validation accuracy |
| --- | ---: | ---: |
| Baseline | 200 | 51.21% |
| Early stopping | 1 | 40.10% |
| L1 | 186 | 52.90% |
| L2 | 147 | 55.56% |
| Dropout | 169 | 56.04% |

Dropout is selected by validation accuracy. Its best validation checkpoint scores
**51.28% test accuracy and 0.502712 macro F1**. The majority reference scores
15.38% accuracy and 0.033333 macro F1. Early stopping ended the regularized runs
before their maximum 500 epochs; no configured limit was shortened.

The original notebook's saved dropout accuracy is 50.00%, and its tuned network
accuracy is 49.04%. The rebuilt dropout score is nearby, supporting broad numerical
coherence for that variant, but different fitting/validation actors, explicit seeds,
software and training outcomes prevent a strict reproduction claim. Other original
variants' test scores are not directly comparable to the new validation scores.
No statistical improvement is claimed.

A separate **read-only reload of the original HDF5 checkpoints** used the
notebook's full-training preprocessing. It reproduced every saved checkpoint's
accuracy to the displayed precision:

| Original checkpoint | Notebook accuracy | Reloaded accuracy |
| --- | ---: | ---: |
| Early stopping, no regularizer | 39.1026% | 39.1026% |
| L1 | 44.8718% | 44.8718% |
| L2 | 47.4359% | 47.4359% |
| Dropout | 50.0000% | 50.0000% |

This verifies the old model artifacts and input/class encoding in the current
runtime; it does not establish that new stochastic training reproduces the old
weights. These diagnostic test evaluations never enter rebuilt model selection.
Values and original model hashes are in
[legacy checkpoint comparison](neural_legacy_comparison.json). Repeat with
`scripts/compare_neural_checkpoints.py --project-root ..` from `rebuilt/`.

[Compact results](neural_results.json) include histories, specifications, input
hashes, validation metrics and test metrics. Full local artifacts are in
`runs/neural-001/`: five native checkpoints, `preprocessing.joblib`, prediction CSV,
report, execution settings and runtime log. Compact reports compress repeated Python warning entries
and retain their total count; original full reports are unchanged.

## Runtime and verification

TensorFlow CPU 2.21.0, Keras 3.15.1, NumPy 2.5.4 and scikit-learn 1.9.1 are
installed in the separate Python 3.13.14 environment. Exact installed packages
are recorded in `requirements-neural.lock.txt`; this runtime differs from the
older core/XGBoost snapshots. Use Python 3.11–3.13 for TensorFlow in this project.
The initial network restriction was resolved by installing from the user's normal
terminal; it does not prevent execution of installed packages.

The full suite ran 49 tests: 48 passed, including both TensorFlow integration
tests; the optional XGBoost test was skipped because XGBoost is absent from this
separate environment. The earlier scientific environment still provides XGBoost.
No neural test was skipped.

Verification reloaded all five networks and reproduced their validation metrics,
then checked all 624 selected-network and majority test predictions. It checked
finite, normalized selected-network probabilities, recorded actor membership,
input/config/source hashes, model widths, and saved preprocessing against an
independent column-wise LabelEncoder/StandardScaler implementation. A seeded
two-epoch fixture repeated with identical history, weights and predictions;
full-run retraining equivalence is not claimed. Machine-checkable evidence is
in [artifact validation](neural_artifact_validation.json).

To repeat artifact verification from `rebuilt/`, run:

```bash
.venv-neural/bin/python scripts/verify_neural_run.py --run-dir runs/neural-001 --data-dir ../data
```

The variant report captured 330 repeated Keras/NumPy deprecation warnings.
TensorFlow also logged an ignored dataset attribute and function retracing notices;
complete stdout/stderr is retained in `runtime.log`. These messages were not
silenced by changing model algorithms or iteration limits. Execution completed
successfully and saved outputs passed the checks above.

## Parameter search

All fifteen sampled candidates completed all five actor folds: **75 validation
fits**, followed by one selected-model refit. Mean macro F1 selected `search_05`:
three 50-unit ReLU hidden layers, L2 strength 0.01, Adam learning rate 0.001,
50 epochs and batch size 32. Its mean validation macro F1 is 0.4309482; the five
fold scores are 0.49659, 0.43939, 0.37689, 0.37974 and 0.46213. These reused-fold
selection scores are not nested-CV estimates.

After refitting on all 1,809 training rows, the selected model scores **51.44%
test accuracy and 0.506359 macro F1**. The original tuned network's saved accuracy
was 49.04%, using different sampled settings and validation. The new result is
nearby, but no improvement or historical equivalence is claimed. The variant and
search scores also use different final fitting sets (fourteen versus eighteen
actors), so their small difference is not a controlled model comparison.

[Compact search results](neural_search_results.json) retain all candidate settings,
75 fold scores, winning history and final metrics. Full local artifacts are in
`runs/neural-search-001/`. The selected checkpoint reproduces all 624 recorded
test predictions and final metrics after reload; preprocessing, actor isolation,
finite normalized probabilities, source/config hashes and final-fitting actors
were verified. [Search artifact validation](neural_search_artifact_validation.json)
records those checks. Eight Keras/NumPy deprecation warnings are retained in the
full report; TensorFlow stderr messages are preserved in the runtime log.

Both runs record their exact invocation and thread settings in `execution.json`.
Their original reports are unchanged. All 82 original project files and links
were verified unchanged after the checkpoint reloads.

API references: [TensorFlow CPU package](https://pypi.org/project/tensorflow-cpu/),
[Keras training defaults](https://keras.io/api/models/model_training_apis/),
[SciKeras classifier defaults](https://adriangb.com/scikeras/stable/generated/scikeras.wrappers.KerasClassifier.html),
[EarlyStopping](https://keras.io/api/callbacks/early_stopping/).
