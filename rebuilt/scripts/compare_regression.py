"""Check notebook-style regression preprocessing and fixed historical settings.

Print diagnostics only. These test scores never enter rebuilt model selection.
Run with the rebuilt package installed, or set PYTHONPATH=rebuilt/src from root.
"""

import argparse
import csv
import json
import tomllib
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ravdess_project.data import load_dataset
from ravdess_project.preprocessing import legacy_preprocessing
from ravdess_project.reporting import regression_metrics


def read_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names = [name for name in reader.fieldnames
                 if name not in {"", "outlier_detection", "filename", "kur"}]
        return names, list(reader)


def compare(data_dir):
    config_path = Path(__file__).resolve().parents[1] / "configs/regression.toml"
    config = tomllib.loads(config_path.read_text())
    train, test = load_dataset(data_dir, config)
    names, rows = read_rows(data_dir / "Train_NOoutliers")
    test_names, test_rows = read_rows(data_dir / "Test_ripulito (1)")
    if names != list(train.feature_names) or names != test_names:
        raise ValueError("Notebook and rebuilt feature orders differ.")
    manual_train = np.empty((len(rows), len(names)))
    manual_test = np.empty((len(test_rows), len(names)))
    for index, name in enumerate(names):
        if name in config["categorical_columns"]:
            encoder = LabelEncoder().fit([row[name] for row in rows])
            manual_train[:, index] = encoder.transform([row[name] for row in rows])
            manual_test[:, index] = encoder.transform([row[name] for row in test_rows])
        else:
            manual_train[:, index] = [float(row[name]) for row in rows]
            manual_test[:, index] = [float(row[name]) for row in test_rows]
    np.testing.assert_array_equal(train.labels, [float(row["kur"]) for row in rows])
    np.testing.assert_array_equal(test.labels, [float(row["kur"]) for row in test_rows])
    scaler = StandardScaler()
    manual_train = scaler.fit_transform(manual_train)
    manual_test = scaler.transform(manual_test)
    rebuilt = legacy_preprocessing(train).fit(np.asarray(train.values, dtype=object))
    np.testing.assert_array_equal(manual_train, rebuilt.transform(np.asarray(train.values, dtype=object)))
    np.testing.assert_array_equal(manual_test, rebuilt.transform(np.asarray(test.values, dtype=object)))
    result = {
        "check": "Independent recreation of notebook encoding/scaling with current libraries; not a historical rerun.",
        "train_rows": len(rows), "test_rows": len(test_rows), "predictors": len(names),
        "categorical_predictors": config["categorical_columns"],
        "train_transform_exact": True, "test_transform_exact": True,
        "fixed_historical_settings_metrics": {},
    }
    models = {
        "linear": LinearRegression(), "lasso_alpha_0.01": Lasso(alpha=.01),
        "ridge_alpha_0.99": Ridge(alpha=.99),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=0),
    }
    for name, model in models.items():
        model.fit(manual_train, train.labels)
        result["fixed_historical_settings_metrics"][name] = regression_metrics(
            test.labels, model.predict(manual_test)
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.data_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
