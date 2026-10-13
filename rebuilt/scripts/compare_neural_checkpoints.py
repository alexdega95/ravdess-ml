"""Reload original neural checkpoints using notebook-style full-training preprocessing.

Read-only diagnostics: no training, checkpoint updates, or rebuilt model selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ravdess_project.data import load_dataset, sha256
from ravdess_project.neural import tensorflow_backend
from ravdess_project.preprocessing import legacy_preprocessing
from ravdess_project.reporting import classification_metrics


def compare(project_root):
    import tomllib

    config_path = Path(__file__).resolve().parents[1] / "configs/neural.toml"
    config = tomllib.loads(config_path.read_text())
    train, test = load_dataset(project_root / "data", config)
    preprocessing = legacy_preprocessing(train).fit(np.asarray(train.values, dtype=object))
    X = preprocessing.transform(np.asarray(test.values, dtype=object))
    labels = LabelEncoder().fit(train.labels)
    tf = tensorflow_backend()
    results = {}
    paths = sorted((project_root / "models").glob("best_model_*.h5"))
    if not paths:
        raise ValueError("Original neural checkpoints were not found under models/.")
    for path in paths:
        try:
            # Original optimizer state is unnecessary for evaluating saved weights.
            model = tf.keras.models.load_model(path, compile=False)
            probabilities = model.predict(X, verbose=0)
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite checkpoint predictions.")
            predicted = labels.inverse_transform(np.argmax(probabilities, axis=1))
            metrics = classification_metrics(test.labels, predicted, labels.classes_)
            results[path.name] = {
                "sha256": sha256(path), "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"], "predictions_finite": True,
            }
        except (ValueError, TypeError, OSError) as exc:
            results[path.name] = {"load_error": f"{type(exc).__name__}: {exc}"}
    return {
        "check": "Read-only reload of legacy HDF5 checkpoints with full-training notebook-style preprocessing, current TensorFlow; not a retraining experiment.",
        "models": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.project_root), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
