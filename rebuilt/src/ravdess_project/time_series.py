"""Original waveform intervals and row-wise scaling, with actor-aware comparison."""

import csv
import json
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .audio_preparation import decode_filename
from .data import sha256
from .reporting import parameters
from .tabular import finalize_selection
from .validation import select_on_training


@dataclass
class SeriesTable:
    path: Path
    values: np.ndarray
    labels: list[str]
    filenames: list[str]
    actors: list[int]
    digest: str
    metadata_path: Path
    metadata_digest: str
    target_name: str
    interval: tuple[int, int]
    representation: str
    source_labels: None = None
    task_kind: str = "classification"
    categorical_features: tuple = ()

    @property
    def feature_names(self):
        return tuple(f"sample_{i}" for i in range(*self.interval))

    def summary(self):
        return {"path": str(self.path.resolve()), "sha256": self.digest,
                "metadata_path": str(self.metadata_path.resolve()), "metadata_sha256": self.metadata_digest,
                "rows": len(self.labels), "timesteps": self.values.shape[1],
                "target": self.target_name, "representation": self.representation,
                "interval": list(self.interval), "actors": sorted(set(self.actors)),
                "class_counts": dict(sorted(Counter(self.labels).items())),
                "nonfinite_in_selected_interval": 0}


def load_series(data_dir, config):
    settings = config["time_series"]
    representation = settings["representation"]
    if representation not in {"raw", "compressed"}:
        raise ValueError("time_series.representation must be raw or compressed.")
    target = config["target"]
    if target not in {"emotion", "vocal_channel"}:
        raise ValueError("Time-series target must be emotion or vocal_channel.")
    start, stop = settings["start"], settings["stop"]
    if not isinstance(start, int) or not isinstance(stop, int) or not 0 <= start < stop:
        raise ValueError("A nonempty, nonnegative sample interval is required.")
    result = []
    for split in ("train", "test"):
        suffix = "_compresso" if representation == "compressed" else ""
        path = data_dir / f"RavdessAudioOnlyNumpy__X_{split}{suffix}.npy"
        metadata_path = data_dir / f"RavdessAudioOnlyNumpy__Y_{split}.csv"
        before = path.stat()
        metadata_digest = sha256(metadata_path)
        with metadata_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if not rows or len({row["filename"] for row in rows}) != len(rows):
            raise ValueError("Metadata requires nonempty unique recording names.")
        for row in rows:
            if row != decode_filename(row["filename"]):
                raise ValueError(f"Filename and metadata disagree: {row['filename']}")
        if rows != sorted(rows, key=lambda row: (int(row["actor"]), row["filename"])):
            raise ValueError("Metadata must retain original actor/filename order.")
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        if source.ndim != 2 or source.shape[0] != len(rows) or stop > source.shape[1] or source.dtype.kind != "f":
            raise ValueError("Numeric waveform shape must match metadata and contain the complete interval.")
        values = np.array(source[:, start:stop], dtype="float64", order="C", copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Selected interval contains padding/nonfinite samples; no implicit imputation.")
        actors = [int(row["actor"]) for row in rows]
        if set(actors) != set(config[f"{split}_actors"]):
            raise ValueError(f"{split}: actors differ from the configured partition.")
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino) or sha256(metadata_path) != metadata_digest:
            raise ValueError("Time-series inputs changed while reading.")
        result.append(SeriesTable(path, values, [row[target] for row in rows],
                                  [row["filename"] for row in rows], actors, digest,
                                  metadata_path, metadata_digest, target, (start, stop), representation))
    train, test = result
    if set(train.actors) & set(test.actors) or set(train.filenames) & set(test.filenames):
        raise ValueError("Train/test recordings or actors overlap.")
    if set(train.labels) != set(test.labels):
        raise ValueError("Train/test task classes differ.")
    return train, test


class SeriesMinMax(TransformerMixin, BaseEstimator):
    """The finite-interval equivalent of tslearn's default per-recording scaler."""

    def _array(self, X):
        values = np.asarray(X, dtype="float64")
        if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
            raise ValueError("Expected nonempty finite univariate time series.")
        return values

    def fit(self, X, y=None):
        self.n_features_in_ = self._array(X).shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        values = self._array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Time-series lengths differ from fitted interval.")
        low, high = values.min(axis=1, keepdims=True), values.max(axis=1, keepdims=True)
        span = high - low
        span[span == 0] = 1
        return (values - low) / span


class RocketFeatures(TransformerMixin, BaseEstimator):
    """Adapt sktime's public 3D panel representation to a sklearn pipeline."""

    def __init__(self, kind="minirocket", num_kernels=10000, random_state=42, n_jobs=1):
        self.kind = kind
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        from sktime.transformations.panel.rocket import MiniRocket, Rocket

        if self.kind == "rocket":
            self.transformer_ = Rocket(num_kernels=self.num_kernels, normalise=True,
                                       random_state=self.random_state, n_jobs=self.n_jobs)
        elif self.kind == "minirocket":
            self.transformer_ = MiniRocket(num_kernels=self.num_kernels, max_dilations_per_kernel=32,
                                           random_state=self.random_state, n_jobs=self.n_jobs)
        else:
            raise ValueError("Unknown convolution transform.")
        self.transformer_.fit(np.asarray(X)[:, None, :])
        return self

    def transform(self, X):
        check_is_fitted(self, "transformer_")
        return np.asarray(self.transformer_.transform(np.asarray(X)[:, None, :]))


def series_models(config):
    settings = config["time_series"]
    names = settings["methods"]
    if not names or len(names) != len(set(names)) or set(names) - {"majority", "euclidean", "rocket", "minirocket"}:
        raise ValueError("Time-series methods must be unique supported names.")
    result = {}
    for name in names:
        if name == "majority":
            result[name] = DummyClassifier(strategy="most_frequent")
            continue
        if name == "euclidean":
            # pyts delegates its default n_neighbors=1 Euclidean classifier to sklearn.
            model = KNeighborsClassifier(n_neighbors=1, weights="uniform", algorithm="auto",
                                          leaf_size=30, p=2, metric="euclidean", n_jobs=1)
            steps = [("scale", SeriesMinMax())]
        else:
            model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            steps = [("scale", SeriesMinMax()), ("reduce", RocketFeatures(
                name, settings["num_kernels"], config["seed"], settings.get("transform_jobs", 1)))]
        result[name] = Pipeline([*steps, ("classifier", model)])
    return result




def rocket_matrix(train, transform, cache_dir, batch_size, progress=None):
    """Write deterministic per-record features in resumable, verified chunks."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("rocket_batch_size must be positive.")
    key = {"input_sha256": train.digest, "metadata_sha256": train.metadata_digest,
           "interval": list(train.interval), "rows": len(train.labels),
           "parameters": parameters(transform),
           "versions": {name: version(name) for name in ("numpy", "sktime", "numba")},
           "source_sha256": sha256(Path(__file__))}
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path, matrix_path = cache_dir / "state.json", cache_dir / "rocket.npy"
    shape = (len(train.labels), 2 * transform.num_kernels)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["key"] != key:
            raise ValueError("ROCKET cache inputs/software differ; choose a new output directory.")
        matrix = np.load(matrix_path, mmap_mode="r+", allow_pickle=False)
        if matrix.shape != shape or matrix.dtype != np.dtype("float64"):
            raise ValueError("ROCKET cache shape/dtype differs.")
        expected_start = 0
        for block in state["blocks"]:
            if block["start"] != expected_start or not block["start"] < block["stop"] <= shape[0]:
                raise ValueError("Invalid cached block ordering.")
            expected_start = block["stop"]
            import hashlib
            digest = hashlib.sha256(matrix[block["start"]:block["stop"]].tobytes()).hexdigest()
            if digest != block["sha256"]:
                raise ValueError("ROCKET cached block checksum differs.")
    else:
        if matrix_path.exists():
            raise ValueError("Incomplete cache initialization; choose a new output directory.")
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype="float64", shape=shape)
        matrix[:] = np.nan
        matrix.flush()
        state = {"key": key, "completed_rows": 0, "blocks": []}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
    if state["blocks"] and state["blocks"][-1]["stop"] != state["completed_rows"]:
        raise ValueError("Cached progress does not match block coverage.")
    completed = state["completed_rows"]
    if not 0 <= completed <= len(train.labels):
        raise ValueError("Invalid ROCKET cache progress.")
    scaler = SeriesMinMax().fit(np.zeros((1, train.values.shape[1])))
    for start in range(completed, len(train.labels), batch_size):
        import hashlib
        stop = min(start + batch_size, len(train.labels))
        block = transform.transform(scaler.transform(train.values[start:stop]))
        if block.shape != (stop - start, shape[1]) or not np.isfinite(block).all():
            raise ValueError("ROCKET produced invalid features.")
        matrix[start:stop] = block
        matrix.flush()
        digest = hashlib.sha256(matrix[start:stop].tobytes()).hexdigest()
        state["blocks"].append({"start": start, "stop": stop, "sha256": digest})
        state["completed_rows"] = stop
        temporary = cache_dir / "state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(state_path)
        if progress:
            progress(f"Saved ROCKET features: {stop}/{len(train.labels)} training records")
    return matrix

def compare_series(train, config, progress=None, feature_cache=None):
    """Cache only shape-derived ROCKET features; MiniROCKET stays fold-fitted.

    ROCKET's seeded kernels depend only on channel count and fixed interval length.
    Generate them from a shape-only placeholder, never validation observations.
    Each classifier still fits only its actor-fold training rows.
    """
    candidates = series_models(config)
    settings = config["time_series"]
    if "rocket" not in candidates or not settings.get("reuse_rocket_features", False):
        return select_on_training(train, candidates, settings["cv_folds"], config["seed"], progress=progress)
    if progress:
        progress("Generating shape-only ROCKET kernels and training-record features once for actor CV")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler = SeriesMinMax().fit(np.zeros((1, train.values.shape[1])))
        transform = RocketFeatures("rocket", settings["num_kernels"], config["seed"], settings.get("transform_jobs", 1))
        transform.fit(np.zeros((1, train.values.shape[1])))
        if feature_cache is None:
            features = transform.transform(scaler.transform(train.values))
        else:
            features = rocket_matrix(train, transform, feature_cache,
                                     settings.get("rocket_batch_size", 32), progress)
    recorded = [{"stage": "rocket_feature_generation", "method": "rocket",
                 "category": item.category.__name__, "message": str(item.message)} for item in caught]
    if progress:
        progress(f"Generated ROCKET feature matrix: {features.shape[0]} rows, {features.shape[1]} features")
    rocket_validation = select_on_training(
        replace(train, values=features), {"rocket": candidates["rocket"].named_steps["classifier"]},
        settings["cv_folds"], config["seed"], progress=progress,
    )
    del features
    remaining = {name: model for name, model in candidates.items() if name != "rocket"}
    validation = select_on_training(train, remaining, settings["cv_folds"], config["seed"], progress=progress) if remaining else rocket_validation
    if remaining and validation["folds"] != rocket_validation["folds"]:
        raise ValueError("Cached and uncached methods used different actor folds.")
    scores = {**validation["scores"], **rocket_validation["scores"]}
    validation["scores"] = {name: scores[name] for name in candidates}
    validation["selected_method"] = max(validation["scores"], key=lambda name: scores[name]["mean_f1_macro"])
    validation["candidate_parameters"] = {name: parameters(model) for name, model in candidates.items()}
    validation["warnings"] = recorded + (validation["warnings"] + rocket_validation["warnings"] if remaining else rocket_validation["warnings"])
    validation["execution_optimization"] = {
        "name": "reuse shape-only seeded ROCKET features across actor folds",
        "placeholder_shape": [1, train.values.shape[1]],
        "feature_matrix_shape": [len(train.labels), 2 * settings["num_kernels"]],
        "data_dependent_minirocket_cached": False,
        "classifiers_fit_on_actor_fold_rows_only": True,
    }
    return validation


def run_series(train, test, config, config_path, output, progress=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}.")
    candidates = series_models(config)
    if set(candidates) & {"rocket", "minirocket"}:
        from sktime.transformations.panel.rocket import Rocket, MiniRocket
    validation = compare_series(train, config, progress, output.with_name(output.name + "-features"))
    return finalize_selection(train, test, config, config_path, output, validation,
                               candidates, "waveform_actor_comparison_v1", progress)
