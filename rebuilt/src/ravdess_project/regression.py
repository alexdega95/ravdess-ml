"""Predict the original kur target using the original five regression families."""

import warnings

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline

from .preprocessing import legacy_preprocessing
from .reporting import parameters, provenance, regression_metrics, write_artifacts
from .validation import select_on_training


def build_regressors(train, config):
    settings = config["regression"]
    if train.task_kind != "regression" or "kur" in train.feature_names:
        raise ValueError("Regression requires a continuous kur target excluded from predictors.")
    estimators = {"mean": DummyRegressor(strategy="mean"), "linear": LinearRegression()}
    for family, constructor in (("lasso", Lasso), ("ridge", Ridge)):
        for index, alpha in enumerate(settings["alphas"]):
            estimators[f"{family}_{index + 1:02d}"] = constructor(alpha=alpha)
    sampled = list(ParameterSampler(
        {"criterion": ["squared_error"],
         "min_samples_split": settings["forest_min_samples_split"],
         "min_samples_leaf": settings["forest_min_samples_leaf"]},
        n_iter=settings["forest_trials"], random_state=config["seed"],
    ))
    for index, values in enumerate(sampled):
        estimators[f"random_forest_{index + 1:02d}"] = RandomForestRegressor(
            n_estimators=100, random_state=config["seed"], n_jobs=1, **values,
        )
    estimators["gradient_boosting"] = GradientBoostingRegressor(n_estimators=100, random_state=0)
    return {
        name: Pipeline([("preprocess", legacy_preprocessing(train)), ("regressor", estimator)])
        for name, estimator in estimators.items()
    }


def run_regression(train, test, config, config_path, output, progress=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}.")
    candidates = build_regressors(train, config)
    settings = config["regression"]
    validation = select_on_training(
        train, candidates, settings["cv_folds"], config["seed"], metric="mse",
        jobs=settings.get("parallel_jobs", 1), progress=progress,
    )
    selected = validation["selected_method"]
    models = {name: candidates[name] for name in dict.fromkeys(["mean", selected])}
    X_train, X_test = np.asarray(train.values, dtype=object), np.asarray(test.values, dtype=object)
    predictions, metrics = {}, {}
    recorded = validation.pop("warnings")
    for name, model in models.items():
        if progress:
            progress(f"Fitting {name} on all training actors; evaluating the fixed test partition")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, train.labels)
            predictions[name] = model.predict(X_test)
        recorded.extend({"stage": "final", "method": name, "category": item.category.__name__,
                         "message": str(item.message)} for item in caught)
        metrics[name] = regression_metrics(test.labels, predictions[name])
    report = provenance("kur_regression_actor_selection_v1", train, test, config, config_path)
    report.update({"validation": validation, "metrics": metrics, "warnings": recorded,
                   "model_parameters": {name: parameters(model) for name, model in models.items()}})
    write_artifacts(output, report, models, predictions, train, test)
    return report
