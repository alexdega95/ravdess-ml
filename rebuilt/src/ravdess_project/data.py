"""Read and validate supplied feature tables without scientific dependencies."""

import csv
import hashlib
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

EMOTIONS = frozenset(
    {"neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"}
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FeatureTable:
    path: Path
    feature_names: tuple[str, ...]
    values: list[list[float | str]]
    labels: list[str | int | float]
    filenames: list[str]
    actors: list[int]
    digest: str
    target_name: str = "emotion"
    source_labels: list[str] | None = None
    source_row_count: int | None = None

    categorical_features: tuple[str, ...] = ()
    task_kind: str = "classification"

    def summary(self) -> dict:
        result = {
            "path": str(self.path.resolve()),
            "sha256": self.digest,
            "rows": len(self.labels),
            "numeric_features": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "actors": sorted(set(self.actors)),
            "class_counts": dict(sorted(Counter(self.labels).items())),
        }
        if self.categorical_features:
            result["numeric_features"] -= len(self.categorical_features)
            result["categorical_features"] = list(self.categorical_features)
        if self.task_kind == "regression":
            result.pop("class_counts")
            result["target_statistics"] = {
                "min": min(self.labels), "max": max(self.labels),
                "mean": statistics.mean(self.labels), "std": statistics.pstdev(self.labels),
            }
        if self.source_labels is not None:
            result.update({
                "target": self.target_name,
                "source_rows": self.source_row_count,
                "retained_emotion_counts": dict(sorted(Counter(self.source_labels).items())),
            })
        return result


def positive_negative_task(table: FeatureTable) -> FeatureTable:
    """Preserve the SVM notebook's exclusions and 0/1 emotion mapping."""
    if any(label not in EMOTIONS for label in table.labels):
        raise ValueError("Binary emotion conversion requires original emotion labels.")
    keep = [i for i, label in enumerate(table.labels) if label not in {"neutral", "surprised"}]
    original = [table.labels[i] for i in keep]
    labels = [int(label in {"happy", "calm"}) for label in original]
    if set(labels) != {0, 1}:
        raise ValueError("Binary emotion task requires both positive and negative classes.")
    actors = [table.actors[i] for i in keep]
    if set(actors) != set(table.actors):
        raise ValueError("Binary exclusions removed an entire actor from the partition.")
    return FeatureTable(
        table.path, table.feature_names, [table.values[i] for i in keep], labels,
        [table.filenames[i] for i in keep], actors, table.digest,
        target_name="positive", source_labels=original, source_row_count=len(table.labels),
    )


def read_features(
    path: Path, metadata_columns: list[str], target: str, *, cleaned: bool = False, categorical_columns: tuple[str, ...] = ()
) -> FeatureTable:
    """Retain CSV feature/row order; reject malformed or nonfinite values.

    No row filtering, imputation, scaling, encoding, or feature selection occurs.
    """
    if target != "emotion":
        raise ValueError("This first experiment supports target='emotion' only.")
    required = {target, "actor", "filename"}
    if not required.issubset(metadata_columns):
        raise ValueError("metadata_columns must exclude emotion, actor, and filename.")
    values, labels, filenames, actors = [], [], [], []
    seen = set()
    before = sha256(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if (
            not columns
            or any(not name for name in columns if not cleaned)
            or len(set(columns)) != len(columns)
            or (cleaned and "" in columns and columns[0] != "")
        ):
            raise ValueError(f"{path}: headers must be nonempty and unique.")
        optional = {"modality", "actor", "outlier_detection", ""} if cleaned else set()
        missing = set(metadata_columns) - set(columns) - optional
        if missing:
            raise ValueError(f"{path}: missing metadata columns: {sorted(missing)}")
        excluded = set(metadata_columns) | ({"", "outlier_detection"} if cleaned else set())
        if set(categorical_columns) - set(columns):
            raise ValueError("Configured categorical columns are missing from the CSV.")
        if {"filename", "actor"} & set(categorical_columns):
            raise ValueError("Identifiers cannot be predictors.")
        features = tuple(name for name in columns if name not in excluded or name in categorical_columns)
        if not features:
            raise ValueError(f"{path}: no numeric features remain.")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: incorrect number of CSV fields.")
            label, filename = row[target], row["filename"]
            if label not in EMOTIONS:
                raise ValueError(f"{path}:{line}: unknown emotion {label!r}.")
            if not filename or filename in seen:
                raise ValueError(f"{path}:{line}: empty or duplicate filename {filename!r}.")
            try:
                if cleaned:
                    match = re.fullmatch(r"(?:\d{2}-){6}(\d{2})\.wav", filename)
                    if match is None:
                        raise ValueError("expected a RAVDESS filename")
                    actor = int(match.group(1))
                    if "actor" in row and int(row["actor"]) != actor:
                        raise ValueError("actor column disagrees with filename")
                else:
                    actor = int(row["actor"])
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: invalid actor: {exc}.") from exc
            if not 1 <= actor <= 24:
                raise ValueError(f"{path}:{line}: actor must be between 1 and 24.")
            numbers = []
            for name in features:
                if name in categorical_columns:
                    if not row[name]:
                        raise ValueError(f"{path}:{line}: empty category {name!r}.")
                    numbers.append(row[name])
                    continue
                try:
                    number = float(row[name])
                except ValueError as exc:
                    raise ValueError(f"{path}:{line}: feature {name!r} is not numeric.") from exc
                if not math.isfinite(number):
                    raise ValueError(f"{path}:{line}: feature {name!r} is missing or nonfinite.")
                numbers.append(number)
            values.append(numbers)
            labels.append(label)
            filenames.append(filename)
            actors.append(actor)
            seen.add(filename)
    if not labels:
        raise ValueError(f"{path}: no data rows.")
    if sha256(path) != before:
        raise ValueError(f"{path}: input changed while it was being read.")
    return FeatureTable(path, features, values, labels, filenames, actors, before,
                        categorical_features=tuple(name for name in features if name in categorical_columns))


def validate_split(train: FeatureTable, test: FeatureTable, config: dict) -> None:
    if train.feature_names != test.feature_names:
        raise ValueError("Train/test feature names and their order must match exactly.")
    if set(train.filenames) & set(test.filenames):
        raise ValueError("Train and test share filenames.")
    if set(train.actors) & set(test.actors):
        raise ValueError("Train and test share actors.")
    for name, table in (("train", train), ("test", test)):
        if set(table.actors) != set(config[f"{name}_actors"]):
            raise ValueError(f"{name}: actor set does not match the configured partition.")
        if set(table.labels) != EMOTIONS:
            raise ValueError(f"{name}: expected all eight emotion classes.")


def load_dataset(data_dir: Path, config: dict) -> tuple[FeatureTable, FeatureTable]:
    variant = config.get("dataset", {}).get("variant", "supplied")
    if variant not in {"supplied", "cleaned"}:
        raise ValueError("dataset.variant must be 'supplied' or 'cleaned'.")
    cleaned = variant == "cleaned"
    train_file = "Train_NOoutliers" if cleaned else "RavdessAudioOnlyFeatures_TRAIN.csv"
    test_file = "Test_ripulito (1)" if cleaned else "RavdessAudioOnlyFeatures_TEST.csv"
    train = read_features(
        data_dir / train_file, config["metadata_columns"], "emotion", cleaned=cleaned,
        categorical_columns=tuple(config.get("categorical_columns", []))
    )
    test = read_features(
        data_dir / test_file, config["metadata_columns"], "emotion", cleaned=cleaned,
        categorical_columns=tuple(config.get("categorical_columns", []))
    )
    validate_split(train, test, config)
    task = config.get("task", "emotion")
    if task == "positive_negative":
        train, test = positive_negative_task(train), positive_negative_task(test)
    elif task == "kur_regression":
        if config["target"] != "kur":
            raise ValueError("kur_regression requires target='kur'.")
        train, test = regression_target(train), regression_target(test)
    elif task != "emotion":
        raise ValueError("Unsupported task.")
    elif config["target"] != "emotion":
        raise ValueError("Emotion classification requires target='emotion'.")
    return train, test


def regression_target(table: FeatureTable) -> FeatureTable:
    """Move kur from predictors to the continuous target; retain original emotions."""
    index = table.feature_names.index("kur")
    labels = [float(row[index]) for row in table.values]
    return FeatureTable(
        table.path, tuple(name for name in table.feature_names if name != "kur"),
        [row[:index] + row[index + 1:] for row in table.values], labels,
        table.filenames, table.actors, table.digest, target_name="kur",
        source_labels=list(table.labels), source_row_count=len(table.labels),
        categorical_features=table.categorical_features, task_kind="regression",
    )
