"""Fixed baseline experiment on the original actor-held-out feature tables."""

import warnings
from pathlib import Path

from .data import FeatureTable
from .reporting import classification_metrics, parameters, provenance, write_artifacts


def run_baselines(
    train: FeatureTable, test: FeatureTable, config: dict, config_path: Path, output: Path
) -> dict:
    # Keep the audit command usable without NumPy/scikit-learn installed.
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier

    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new run directory.")
    X_train = np.asarray(train.values, dtype=np.float64)
    X_test = np.asarray(test.values, dtype=np.float64)
    classes = sorted(set(train.labels))
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "random_forest": RandomForestClassifier(
            random_state=config["seed"], **config["random_forest"]
        ),
    }
    metrics, predictions, recorded_warnings = {}, {}, []
    for name, model in models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, train.labels)
            predicted = model.predict(X_test)
        recorded_warnings.extend(
            {"model": name, "category": w.category.__name__, "message": str(w.message)}
            for w in caught
        )
        predictions[name] = predicted
        metrics[name] = classification_metrics(test.labels, predicted, classes)
    variant = config.get("dataset", {}).get("variant", "supplied")
    report = provenance(f"{variant}_features_actor_holdout_v1", train, test, config, config_path)
    report.update({
        "model_parameters": {name: parameters(model) for name, model in models.items()},
        "warnings": recorded_warnings, "metrics": metrics,
    })
    write_artifacts(output, report, models, predictions, train, test)
    return report
