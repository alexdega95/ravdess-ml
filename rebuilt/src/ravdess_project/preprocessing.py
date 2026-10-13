"""Fold-fitted categorical encoding and scaling used in the original notebooks."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def legacy_preprocessing(table):
    """OrdinalEncoder uses the same sorted category mapping as LabelEncoder.

    Metadata predictors precede numeric features in the supplied cleaned tables.
    Reject other layouts rather than silently changing feature order.
    """
    count = len(table.categorical_features)
    if table.feature_names[:count] != table.categorical_features:
        raise ValueError("Categorical predictors must precede numeric predictors in CSV order.")
    steps = []
    if count:
        steps.append(("encode", ColumnTransformer([
            ("categories", OrdinalEncoder(handle_unknown="error"), list(range(count))),
            ("numeric", "passthrough", list(range(count, len(table.feature_names)))),
        ])))
    return Pipeline([*steps, ("scale", StandardScaler())])
