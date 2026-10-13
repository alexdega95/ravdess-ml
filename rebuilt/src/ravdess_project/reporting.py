"""Shared metrics, provenance, and artifact writing for rebuilt experiments."""

import csv
import json
import math
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .data import sha256


def classification_metrics(truth, predicted, classes):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "classification_report": classification_report(
            truth, predicted, labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=classes).tolist(),
    }


def parameters(model):
    """Describe estimator parameters without pickling them into a JSON report."""
    def serializable(value):
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return repr(value)

    return {key: serializable(value) for key, value in model.get_params().items()}


def provenance(experiment, train, test, config, config_path):
    result = {
        "experiment": experiment,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_equivalence": "Not claimed; this is a new experiment.",
        "config": config,
        "config_sha256": sha256(config_path),
        "source_sha256": {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")
            },
        },
        "inputs": {"train": train.summary(), "test": test.summary()},
        "class_order": sorted(set(train.labels)) if train.task_kind == "classification" else None,
    }
    if train.task_kind == "regression":
        result["task"] = {"name": "kur_regression", "target": "kur"}
    elif train.source_labels is not None:
        result["task"] = {
            "name": "positive_negative", "target": "positive",
            "excluded_emotions": ["neutral", "surprised"],
            "class_mapping": {"0": "negative", "1": "positive"},
            "positive_emotions": ["calm", "happy"],
            "negative_emotions": ["angry", "disgust", "fearful", "sad"],
        }
    if config.get("search", {}).get("algorithm") == "xgboost":
        result["environment"]["packages"]["xgboost"] = version("xgboost")
        try:
            result["environment"]["packages"]["nvidia-nccl-cu12"] = version("nvidia-nccl-cu12")
        except PackageNotFoundError:
            pass
    return result


def write_artifacts(output, report, models, predictions, train, test):
    import joblib

    # Serialize before creating a directory, and never overwrite an existing run.
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(encoded)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        target_columns = ["emotion", test.target_name] if test.source_labels is not None else [test.target_name]
        writer.writerow(["filename", "actor", *target_columns, *models])
        for i, (filename, actor, label) in enumerate(zip(test.filenames, test.actors, test.labels)):
            truth = [test.source_labels[i], label] if test.source_labels is not None else [label]
            writer.writerow([filename, actor, *truth, *(predictions[name][i] for name in models)])
    joblib.dump(
        {"models": models, "feature_names": train.feature_names, "class_order": report["class_order"],
         "target_name": train.target_name},
        output / "models.joblib",
    )


def regression_metrics(truth, predicted):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = float(mean_squared_error(truth, predicted))
    return {"mse": mse, "rmse": math.sqrt(mse),
            "mae": float(mean_absolute_error(truth, predicted)),
            "r2": float(r2_score(truth, predicted, force_finite=False))}
