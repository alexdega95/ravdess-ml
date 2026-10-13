"""Notebook SAX/DTW comparisons, with resumable held-out prediction batches."""

import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .data import sha256
from .reporting import classification_metrics, parameters, provenance
from .time_series import SeriesMinMax


METRICS = ("dtw", "dtw_sakoechiba", "dtw_itakura")


class SAXFeatures(TransformerMixin, BaseEstimator):
    """Keep tslearn's original Gaussian breakpoints and scale=False default."""

    def __init__(self, n_segments=1000, alphabet_size_avg=50):
        self.n_segments = n_segments
        self.alphabet_size_avg = alphabet_size_avg

    def fit(self, X, y=None):
        from tslearn.piecewise import SymbolicAggregateApproximation

        values = np.asarray(X)
        if values.ndim != 2 or not 1 <= self.n_segments <= values.shape[1]:
            raise ValueError("SAX segments must fit inside a two-dimensional series interval.")
        self.transformer_ = SymbolicAggregateApproximation(
            n_segments=self.n_segments, alphabet_size_avg=self.alphabet_size_avg, scale=False
        ).fit(values[:, :, None])
        return self

    def transform(self, X):
        check_is_fitted(self, "transformer_")
        # Remove only the channel axis: a single-record batch must remain 2D.
        return self.transformer_.transform(np.asarray(X)[:, :, None])[:, :, 0]


def preprocessing(config):
    settings = config["sax_dtw"]
    return Pipeline([("scale", SeriesMinMax()), ("sax", SAXFeatures(
        settings["n_segments"], settings["alphabet_size_avg"]))])


def classifiers(config):
    from pyts.classification import KNeighborsClassifier

    metrics = config["sax_dtw"]["metrics"]
    if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(METRICS):
        raise ValueError("SAX/DTW metrics must be unique original DTW names.")
    # Preserve pyts defaults, including one neighbor and default constraint widths.
    return {metric: KNeighborsClassifier(metric=metric) for metric in metrics}


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _checkpoint(path, state):
    payload = {key: value for key, value in state.items() if key != "sha256"}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    _json(path, payload)


def prepare(train, test, config, config_path, output):
    """Save symbols and independently check the complete notebook preprocessing."""
    from tslearn.piecewise import SymbolicAggregateApproximation
    from tslearn.preprocessing import TimeSeriesScalerMinMax

    if output.exists():
        raise ValueError(f"Output already exists: {output}.")
    classifiers(config)  # Validate before writing anything.
    transform = preprocessing(config).fit(train.values)
    symbols = {"train": transform.transform(train.values), "test": transform.transform(test.values)}
    # Execute the notebook API independently on every actual input row.
    original = SymbolicAggregateApproximation(
        n_segments=config["sax_dtw"]["n_segments"],
        alphabet_size_avg=config["sax_dtw"]["alphabet_size_avg"])
    scaler = TimeSeriesScalerMinMax()
    for split, table in (("train", train), ("test", test)):
        scaled = scaler.fit_transform(table.values)
        expected = original.fit_transform(scaled) if split == "train" else original.transform(scaled)
        np.testing.assert_array_equal(symbols[split], expected[:, :, 0])
    report = provenance("notebook_sax_dtw_preparation_v1", train, test, config, config_path)
    report["environment"]["packages"].update({name: version(name) for name in ("tslearn", "pyts", "numba")})
    report["preprocessing"] = parameters(transform)
    report["validation"] = {"all_notebook_symbols_exact": True,
                            "training_rows": len(train.labels), "test_rows": len(test.labels)}
    report["evaluation"] = {"protocol": "Fit each original metric on actors 1–18; evaluate actors 19–24.",
                            "model_selection": "None; report all configured metrics, as in the notebooks."}
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(transform, output / "preprocessor.joblib")
    for split, values in symbols.items():
        np.save(output / f"sax_{split}.npy", values, allow_pickle=False)
    report["artifact_sha256"] = {name: sha256(output / name) for name in
                                 ("preprocessor.joblib", "sax_train.npy", "sax_test.npy")}
    _json(output / "preparation.json", report)
    return report


def evaluate(train, test, config, config_path, output, batch_size=8, progress=None):
    """Resume only verified predictions; every query uses all training recordings."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Prediction batch size must be positive.")
    models = classifiers(config)
    report = json.loads((output / "preparation.json").read_text())
    if report["config"] != config or report["config_sha256"] != sha256(config_path):
        raise ValueError("Prepared configuration differs.")
    for split, table in (("train", train), ("test", test)):
        if table.summary() != report["inputs"][split]:
            raise ValueError("Prepared inputs or metadata differ.")
    # Only execution modules affect resuming this job; unrelated rebuilds may continue.
    for name in ("sax_dtw.py", "time_series.py", "data.py", "reporting.py"):
        if sha256(Path(__file__).parent / name) != report["source_sha256"][name]:
            raise ValueError(f"Prepared execution source differs: {name}.")
    for name, digest in report["artifact_sha256"].items():
        if sha256(output / name) != digest:
            raise ValueError(f"Prepared artifact checksum differs: {name}.")
    for name, expected in report["environment"]["packages"].items():
        if version(name) != expected:
            raise ValueError(f"Prepared package version differs: {name}.")
    train_symbols = np.load(output / "sax_train.npy", allow_pickle=False)
    test_symbols = np.load(output / "sax_test.npy", allow_pickle=False)
    status_path = output / "status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {
        "state": "running", "predictions": {name: [] for name in models}}
    if status_path.exists():
        digest = state.pop("sha256", None)
        if digest != hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest():
            raise ValueError("Prediction checkpoint checksum differs.")
    if set(state["predictions"]) != set(models):
        raise ValueError("Checkpoint metrics differ.")
    for predicted in state["predictions"].values():
        if len(predicted) > len(test.labels) or set(predicted) - set(train.labels):
            raise ValueError("Invalid prediction checkpoint.")
    if state["state"] == "complete":
        return json.loads((output / "report.json").read_text())
    _checkpoint(status_path, state)
    for name, model in models.items():
        model.fit(train_symbols, train.labels)
        predicted = state["predictions"][name]
        for start in range(len(predicted), len(test.labels), batch_size):
            stop = min(start + batch_size, len(test.labels))
            predicted.extend(model.predict(test_symbols[start:stop]).tolist())
            state.update(state="running", metric=name)
            _checkpoint(status_path, state)
            if progress:
                progress(f"{name}: saved {stop}/{len(test.labels)} held-out predictions")
    report["metrics"] = {name: classification_metrics(test.labels, predicted, report["class_order"])
                         for name, predicted in state["predictions"].items()}
    report["classifier_parameters"] = {name: parameters(model) for name, model in models.items()}
    transform = joblib.load(output / "preprocessor.joblib")
    joblib.dump({"models": {name: Pipeline([("preprocess", transform), ("classifier", model)])
                           for name, model in models.items()}, "target_name": test.target_name},
                output / "models.joblib")
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "actor", test.target_name, *models])
        for i, (filename, actor, label) in enumerate(zip(test.filenames, test.actors, test.labels)):
            writer.writerow([filename, actor, label, *(state["predictions"][name][i] for name in models)])
    _json(output / "report.json", report)
    state.update(state="complete")
    _checkpoint(status_path, state)
    return report
