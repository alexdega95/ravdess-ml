"""Actor-disjoint validation shared by fixed comparisons and parameter searches."""

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from .reporting import classification_metrics, regression_metrics, parameters


def actor_folds(train, count, seed):
    if not isinstance(count, int) or count < 2 or count > len(set(train.actors)):
        raise ValueError("cv_folds must be between 2 and the number of training actors.")
    splitter = GroupKFold(n_splits=count, shuffle=True, random_state=seed)
    groups, labels = np.asarray(train.actors), np.asarray(train.labels)
    classes = set(train.labels)
    folds = list(splitter.split(train.values, labels, groups=groups))
    descriptions = []
    for fit, validation in folds:
        if train.task_kind == "classification" and (set(labels[fit]) != classes or set(labels[validation]) != classes):
            raise ValueError("Every training/validation fold must contain all task classes.")
        descriptions.append({
            "train_indices": fit.tolist(), "validation_indices": validation.tolist(),
            "train_actors": sorted(set(groups[fit].tolist())),
            "validation_actors": sorted(set(groups[validation].tolist())),
        })
    return folds, descriptions


def score_fold(name, index, prototype, X, y, fit, validation, classes):
    model = clone(prototype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X[fit], y[fit])
        predicted = model.predict(X[validation])
    recorded = [
        {"stage": "validation", "method": name, "fold": index + 1,
         "category": warning.category.__name__, "message": str(warning.message)}
        for warning in caught
    ]
    if classes is None:
        return name, index, regression_metrics(y[validation], predicted), recorded
    metrics = classification_metrics(y[validation], predicted, classes)
    return name, index, {key: metrics[key] for key in ("accuracy", "f1_macro")}, recorded


def select_on_training(train, candidates, count, seed, metric="f1_macro", progress=None, jobs=1):
    """Fit every candidate on identical training folds; no test data accepted."""
    allowed = {"mse"} if train.task_kind == "regression" else {"accuracy", "f1_macro"}
    if metric not in allowed:
        raise ValueError(f"selection_metric must be one of {sorted(allowed)}.")
    if not candidates:
        raise ValueError("At least one candidate is required.")
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError("parallel_jobs must be a positive integer.")
    X, y = np.asarray(train.values, dtype=object if train.categorical_features else float), np.asarray(train.labels)
    classes = sorted(set(train.labels)) if train.task_kind == "classification" else None
    folds, descriptions = actor_folds(train, count, seed)
    tasks = (
        (name, index, prototype, X, y, fit, validation, classes)
        for name, prototype in candidates.items()
        for index, (fit, validation) in enumerate(folds)
    )
    if jobs == 1:
        completed = (score_fold(*task) for task in tasks)
    else:
        # Separate processes keep warnings isolated; indexed slots restore stable fold order.
        completed = Parallel(n_jobs=jobs, backend="loky", return_as="generator_unordered")(
            delayed(score_fold)(*task) for task in tasks
        )
    results = {name: [None] * len(folds) for name in candidates}
    scores, recorded_warnings = {}, []
    for name, index, metrics, recorded in completed:
        results[name][index] = metrics
        recorded_warnings.extend(recorded)
        if progress:
            progress(f"Validated {name}: fold {index + 1}/{len(folds)}")
    order = {name: index for index, name in enumerate(candidates)}
    recorded_warnings.sort(key=lambda warning: (order[warning["method"]], warning["fold"]))
    for name, fold_scores in results.items():
        if train.task_kind == "regression":
            scores[name] = {"fold_scores": fold_scores,
                            **{f"mean_{key}": float(np.mean([score[key] for score in fold_scores]))
                               for key in ("mse", "rmse", "mae", "r2")},
                            "std_mse": float(np.std([score["mse"] for score in fold_scores]))}
            continue
        scores[name] = {
            "fold_scores": fold_scores,
            "mean_accuracy": float(np.mean([score["accuracy"] for score in fold_scores])),
            "mean_f1_macro": float(np.mean([score["f1_macro"] for score in fold_scores])),
            "std_f1_macro": float(np.std([score["f1_macro"] for score in fold_scores])),
        }
    choose = min if train.task_kind == "regression" else max
    selected = choose(scores, key=lambda name: scores[name][f"mean_{metric}"])
    return {
        "selection_metric": f"mean_{metric}", "selected_method": selected,
        "tie_break": "first method in configuration order",
        "folds": descriptions, "scores": scores, "warnings": recorded_warnings,
        "candidate_parameters": {name: parameters(model) for name, model in candidates.items()},
        "parallel_jobs": jobs,
    }
