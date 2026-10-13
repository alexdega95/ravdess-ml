"""Verify a completed neural run's saved predictions, preprocessing and provenance."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ravdess_project.data import load_dataset, sha256
from ravdess_project.neural import predict_saved, tensorflow_backend
from ravdess_project.reporting import classification_metrics


def verify(output, data_dir):
    report = json.loads((output / "report.json").read_text())
    train, test = load_dataset(data_dir, report["config"])
    bundle = joblib.load(output / "preprocessing.joblib")
    selected = report["validation"]["selected_method"]
    if bundle["selected_method"] != selected or bundle["feature_names"] != train.feature_names:
        raise ValueError("Saved model metadata differs from the report/dataset.")
    for name, table in (("train", train), ("test", test)):
        if table.digest != report["inputs"][name]["sha256"]:
            raise ValueError(f"{name} input changed since this run.")
    with (output / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(test.labels):
        raise ValueError("Prediction count does not match the test partition.")
    for index, row in enumerate(rows):
        if (row["filename"], int(row["actor"]), row["emotion"]) != (
            test.filenames[index], test.actors[index], test.labels[index]
        ):
            raise ValueError("Prediction CSV row order or truth changed.")
    predicted = predict_saved(output, test.values)
    np.testing.assert_array_equal(predicted, [row[selected] for row in rows])
    majority = bundle["majority"].predict(np.asarray(test.values, dtype=object))
    np.testing.assert_array_equal(majority, [row["majority"] for row in rows])
    measured = classification_metrics(test.labels, predicted, report["class_order"])
    if measured != report["metrics"][selected]:
        raise ValueError("Reloaded predictions do not reproduce recorded metrics.")
    mode = report["config"]["neural"]["mode"]
    folds = report["validation"]["folds"]
    for fold in folds:
        fit, validation = fold["train_indices"], fold["validation_indices"]
        if set(fit) & set(validation):
            raise ValueError("Fitting and validation rows overlap.")
        if set(fit) | set(validation) != set(range(len(train.labels))):
            raise ValueError("A fold does not partition all training rows.")
        fit_actors = set(train.actors[index] for index in fit)
        validation_actors = set(train.actors[index] for index in validation)
        if fit_actors & validation_actors or (fit_actors | validation_actors) & set(test.actors):
            raise ValueError("Actor overlap in validation or test partitions.")
        if fit_actors != set(fold["train_actors"]) or validation_actors != set(fold["validation_actors"]):
            raise ValueError("Fold actor descriptions disagree with row indices.")
    fit = folds[0]["train_indices"] if mode == "variants" else list(range(len(train.labels)))
    if set(report["final_training_actors"]) != {train.actors[index] for index in fit}:
        raise ValueError("Final fitting actors differ from the intended partition.")
    # Independently reproduce the notebook's column-wise LabelEncoder + scaler.
    X = np.asarray(train.values, dtype=object)
    encoded = np.empty(X.shape, dtype=float)
    count = len(train.categorical_features)
    for index in range(count):
        encoder = LabelEncoder().fit(X[fit, index])
        encoded[:, index] = encoder.transform(X[:, index])
    encoded[:, count:] = X[:, count:].astype(float)
    expected = StandardScaler().fit(encoded[fit]).transform(encoded)
    np.testing.assert_array_equal(bundle["preprocessing"].transform(X), expected)
    tf = tensorflow_backend()
    model = tf.keras.models.load_model(output / f"{selected}.keras")
    probabilities = model.predict(bundle["preprocessing"].transform(np.asarray(test.values, dtype=object)), verbose=0)
    if not np.isfinite(probabilities).all():
        raise ValueError("Saved network produces nonfinite probabilities.")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    spec = report["model_specs"][selected]
    dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    if [layer.units for layer in dense] != [*spec["hidden_units"], len(report["class_order"])]:
        raise ValueError("Saved hidden/output widths differ from the selected specification.")
    for path in output.glob("*.keras"):
        candidate = tf.keras.models.load_model(path)
        validation = folds[0]["validation_indices"] if mode == "variants" else None
        if validation is not None:
            values = candidate.predict(bundle["preprocessing"].transform(X[validation]), verbose=0)
            if not np.isfinite(values).all():
                raise ValueError(f"{path.name}: nonfinite validation probabilities.")
            truth = np.asarray(train.labels)[validation]
            labels = bundle["labels"].inverse_transform(np.argmax(values, axis=1))
            if classification_metrics(truth, labels, report["class_order"]) != report["validation"]["scores"][path.stem]:
                raise ValueError(f"{path.name}: saved validation metrics differ.")
    source_dir = Path(__file__).resolve().parents[1] / "src/ravdess_project"
    source_matches = all(sha256(source_dir / name) == digest
                         for name, digest in report["source_sha256"].items())
    config_name = "neural.toml" if mode == "variants" else "neural_search.toml"
    config_matches = sha256(source_dir.parents[1] / "configs" / config_name) == report["config_sha256"]
    return {"run": output.name, "selected_method": selected, "predictions_verified": len(predicted),
            "preprocessing_matches_notebook_exactly": True, "actor_isolation_verified": True,
            "probabilities_finite": True, "current_source_matches": source_matches,
            "current_config_matches": config_matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.data_dir), indent=2))


if __name__ == "__main__":
    main()
