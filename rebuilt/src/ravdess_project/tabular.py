"""Compare legacy feature-reduction methods using training actors only."""

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .reporting import classification_metrics, parameters, provenance, write_artifacts
from .validation import actor_folds, select_on_training


def build_models(config):
    settings, seed = config["comparison"], config["seed"]
    methods = settings["methods"]
    known = {"majority", "random_forest", "decision_tree", "pca", "rfe", "select_k_best", "select_from_model"}
    if not methods or len(set(methods)) != len(methods) or set(methods) - known:
        raise ValueError("comparison.methods must contain unique supported method names.")
    result = {}
    for name in methods:
        if name == "majority":
            result[name] = DummyClassifier(strategy="most_frequent")
        elif name == "random_forest":
            result[name] = RandomForestClassifier(random_state=seed, **config["random_forest"])
        else:
            classifier = DecisionTreeClassifier(
                min_samples_leaf=settings["decision_tree_min_samples_leaf"], random_state=seed
            )
            steps = [("scale", StandardScaler())]
            if name == "pca":
                reduction = PCA(n_components=settings["pca_components"], random_state=seed)
            elif name == "rfe":
                reduction = RFE(
                    DecisionTreeClassifier(random_state=seed),
                    n_features_to_select=settings["rfe_features"], step=settings["rfe_step"],
                )
            elif name == "select_k_best":
                reduction = SelectKBest(k=settings["k_best_features"])
            elif name == "select_from_model":
                reduction = SelectFromModel(DecisionTreeClassifier(random_state=seed))
            else:
                reduction = None
            if reduction is not None:
                steps.append(("reduce", reduction))
            result[name] = Pipeline([*steps, ("classifier", classifier)])
    return result


def compare_on_training(train, config, progress=None):
    """Select a method without accepting test features or test labels."""
    return select_on_training(
        train, build_models(config), config["comparison"]["cv_folds"], config["seed"],
        progress=progress,
    )


def feature_details(model, names):
    """Use the fitted transform's column order when naming model importances."""
    details, active = {}, list(names)
    if isinstance(model, Pipeline):
        reduction = model.named_steps.get("reduce")
        if isinstance(reduction, PCA):
            active = [f"PC{i + 1}" for i in range(reduction.n_components_)]
            details["explained_variance_ratio"] = reduction.explained_variance_ratio_.tolist()
            details["component_loadings"] = {
                component: dict(zip(names, weights.tolist()))
                for component, weights in zip(active, reduction.components_)
            }
        elif reduction is not None and hasattr(reduction, "get_support"):
            active = np.asarray(names)[reduction.get_support()].tolist()
        elif reduction is not None:
            active = [f"transformed_{i}" for i in range(model.named_steps["classifier"].coef_.shape[-1])]
        classifier = model.named_steps["classifier"]
    else:
        classifier = model
    details["active_features"] = active
    if hasattr(classifier, "feature_importances_"):
        details["feature_importances"] = sorted(
            ({"feature": name, "importance": float(value)}
             for name, value in zip(active, classifier.feature_importances_)),
            key=lambda item: item["importance"], reverse=True,
        )
    if hasattr(classifier, "support_"):
        details["support_vector_count"] = len(classifier.support_)
        details["support_vectors_per_class"] = classifier.n_support_.tolist()
    return details


def run_comparison(train, test, config, config_path, output, progress=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new run directory.")
    validation = compare_on_training(train, config, progress)
    return finalize_selection(
        train, test, config, config_path, output, validation, build_models(config),
        "tabular_training_actor_comparison_v1", progress,
    )


def finalize_selection(train, test, config, config_path, output, validation, candidates, experiment, progress=None):
    selected = validation["selected_method"]
    models = {"majority": DummyClassifier(strategy="most_frequent")}
    if selected != "majority":
        models[selected] = candidates[selected]
    X_train, X_test = np.asarray(train.values), np.asarray(test.values)
    predictions, metrics, details = {}, {}, {}
    recorded_warnings = list(validation.pop("warnings"))
    for name, model in models.items():
        if progress:
            progress(f"Fitting {name} on all training actors; evaluating the fixed test partition")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, train.labels)
            predictions[name] = model.predict(X_test)
        recorded_warnings.extend(
            {"stage": "final", "method": name, "category": warning.category.__name__,
             "message": str(warning.message)} for warning in caught
        )
        metrics[name] = classification_metrics(test.labels, predictions[name], sorted(set(train.labels)))
        if set(train.labels) == {0, 1} and hasattr(model, "decision_function"):
            from sklearn.metrics import roc_auc_score

            metrics[name]["roc_auc"] = float(roc_auc_score(test.labels, model.decision_function(X_test)))
        details[name] = feature_details(model, train.feature_names)
    report = provenance(experiment, train, test, config, config_path)
    report.update({
        "validation": validation, "metrics": metrics, "warnings": recorded_warnings,
        "model_parameters": {name: parameters(model) for name, model in models.items()},
        "feature_analysis": details,
    })
    write_artifacts(output, report, models, predictions, train, test)
    return report
