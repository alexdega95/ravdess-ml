"""Check saved waveform pipelines, original row-wise scaling, and actor isolation."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from ravdess_project.data import sha256
from ravdess_project.reporting import classification_metrics
from ravdess_project.time_series import load_series


def verify(output, data_dir):
    report = json.loads((output / "report.json").read_text())
    train, test = load_series(data_dir, report["config"])
    bundle = joblib.load(output / "models.joblib")
    with (output / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(test.labels):
        raise ValueError("Prediction count does not match test rows.")
    for index, row in enumerate(rows):
        if (row["filename"], int(row["actor"]), row[test.target_name]) != (
                test.filenames[index], test.actors[index], test.labels[index]):
            raise ValueError("Prediction CSV labels or order differ from input metadata.")
    for name, model in bundle["models"].items():
        predicted = model.predict(test.values)
        np.testing.assert_array_equal(predicted, [row[name] for row in rows])
        if classification_metrics(test.labels, predicted, report["class_order"]) != report["metrics"][name]:
            raise ValueError("Reloaded predictions do not reproduce report metrics.")
    for fold in report["validation"]["folds"]:
        fit, validation = fold["train_indices"], fold["validation_indices"]
        if set(fit) & set(validation) or set(fit) | set(validation) != set(range(len(train.labels))):
            raise ValueError("Invalid training/validation indices.")
        fit_actors = {train.actors[i] for i in fit}
        validation_actors = {train.actors[i] for i in validation}
        if fit_actors & validation_actors or (fit_actors | validation_actors) & set(test.actors):
            raise ValueError("Actor overlap in selection.")
    selected = report["validation"]["selected_method"]
    scaled_match, reference_match = None, None
    if selected != "majority":
        pipeline = bundle["models"][selected]
        def notebook_scale(X):
            # Independent record-by-record implementation of the original scaler.
            result = np.empty_like(X)
            for i, row in enumerate(X):
                low, high = np.nanmin(row), np.nanmax(row)
                result[i] = (row - low) / (high - low if high != low else 1)
            return result
        expected_train, expected_test = notebook_scale(train.values), notebook_scale(test.values)
        np.testing.assert_array_equal(pipeline.named_steps["scale"].transform(train.values), expected_train)
        np.testing.assert_array_equal(pipeline.named_steps["scale"].transform(test.values), expected_test)
        scaled_match = True
        if selected == "euclidean":
            # Same sklearn backend/defaults used by the original pyts wrapper.
            reference = KNeighborsClassifier(n_neighbors=1, metric="euclidean", n_jobs=1)
            reference.fit(expected_train, train.labels)
            np.testing.assert_array_equal(reference.predict(expected_test), [row[selected] for row in rows])
            reference_match = True
    source_dir = Path(__file__).resolve().parents[1] / "src/ravdess_project"
    source_matches = all(sha256(source_dir / name) == digest for name, digest in report["source_sha256"].items())
    return {"run": output.name, "predictions_verified": len(rows), "target": test.target_name,
            "selected_method": selected, "actor_isolation_verified": True,
            "notebook_style_scaling_exact": scaled_match,
            "independent_euclidean_predictions_exact": reference_match,
            "current_source_matches": source_matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.data_dir), indent=2))


if __name__ == "__main__":
    main()
