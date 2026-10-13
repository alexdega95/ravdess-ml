"""Explicit SVM and boosting searches over the original parameter spaces."""

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_is_fitted

from .tabular import finalize_selection
from .validation import select_on_training


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Keep XGBoost's integer label encoding inside each fitted estimator."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.encoder_.transform(y))
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.encoder_.inverse_transform(self.estimator_.predict(X).astype(int))

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_importances_


def scaled(model):
    return Pipeline([("scale", StandardScaler()), ("classifier", model)])


def build_search_candidates(config):
    settings, seed = config["search"], config["seed"]
    algorithm = settings["algorithm"]
    space = settings["space"]
    if not space or any(not isinstance(values, list) or not values for values in space.values()):
        raise ValueError("Every search.space parameter must have a nonempty list of values.")
    candidates = {"majority": DummyClassifier(strategy="most_frequent")}
    if algorithm == "svm":
        if config.get("task") != "positive_negative":
            raise ValueError("The SVM workflow requires task='positive_negative'.")
        for C in config["exploration"]["linear_c"]:
            candidates[f"linear_svc_C_{C}"] = scaled(LinearSVC(
                C=C, random_state=seed, dual=True, max_iter=1000, tol=1e-4,
            ))
        for kernel in config["exploration"]["kernels"]:
            candidates[f"svc_auto_{kernel}"] = scaled(SVC(kernel=kernel, gamma="auto", random_state=seed))
        sampled = list(ParameterGrid(space))
        for i, params in enumerate(sampled, start=1):
            candidates[f"svc_grid_{i:02d}"] = scaled(SVC(random_state=seed, **params))
    elif algorithm in {"gradient_boosting", "xgboost"}:
        if config.get("task", "emotion") != "emotion":
            raise ValueError("Boosting workflows require the eight-emotion task.")
        count = settings["n_iter"]
        if not isinstance(count, int) or count < 1 or count > len(ParameterGrid(space)):
            raise ValueError("n_iter must be between 1 and the size of the search space.")
        sampled = list(ParameterSampler(space, n_iter=count, random_state=seed))
        for i, params in enumerate(sampled, start=1):
            combined = {**config["classifier"], **params}
            if algorithm == "gradient_boosting":
                classifier = GradientBoostingClassifier(**combined)
            else:
                from xgboost import XGBClassifier

                classifier = LabelEncodedClassifier(XGBClassifier(**combined))
            candidates[f"{algorithm}_{i:02d}"] = scaled(classifier)
    else:
        raise ValueError("search.algorithm must be svm, gradient_boosting, or xgboost.")
    return candidates


def run_search(train, test, config, config_path, output, progress=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new run directory.")
    candidates = build_search_candidates(config)
    settings = config["search"]
    validation = select_on_training(
        train, candidates, settings["cv_folds"], config["seed"],
        metric=settings["selection_metric"], progress=progress,
        jobs=settings.get("parallel_jobs", 1),
    )
    validation["search_space"] = settings["space"]
    validation["search_algorithm"] = settings["algorithm"]
    validation["score_interpretation"] = (
        "Training-fold scores used for model/parameter selection; not nested-CV estimates."
    )
    return finalize_selection(
        train, test, config, config_path, output, validation, candidates,
        f"{settings['algorithm']}_training_actor_search_v1", progress,
    )
