"""Original dense-network variants and search; TensorFlow is an optional backend."""

import csv
import json
import warnings
from importlib.metadata import version

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit, ParameterSampler
from sklearn.preprocessing import LabelEncoder

from .preprocessing import legacy_preprocessing
from .reporting import classification_metrics, provenance
from .validation import actor_folds


def tensorflow_backend():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Neural experiments require TensorFlow; use Python 3.11–3.13 and install rebuilt[neural].") from exc
    return tf


def variant_specs(config):
    settings = config["neural"]
    names = settings["variants"]
    known = {"baseline", "early_stopping", "l1", "l2", "dropout"}
    if not names or len(set(names)) != len(names) or set(names) - known:
        raise ValueError("neural.variants must contain unique supported names.")
    return {
        name: {"hidden_units": settings["hidden_units"], "activation": settings["activation"],
               "optimizer": settings["optimizer"], "learning_rate": settings["learning_rate"],
               "regularization": name if name in {"l1", "l2"} else None,
               "regularizer_strength": settings["regularizer_strength"],
               "dropout_rate": settings["dropout_rate"] if name == "dropout" else 0.0,
               "epochs": (settings["baseline_epochs"] if name == "baseline" else
                          settings["early_stopping_epochs"] if name == "early_stopping" else
                          settings["regularized_epochs"]),
               "batch_size": settings["dropout_batch_size"] if name == "dropout" else settings["batch_size"],
               "early_stopping": name != "baseline"}
        for name in names
    }


def search_specs(config):
    settings, search = config["neural"], config["neural"]["search"]
    sampled = ParameterSampler({
        "hidden_units": search["hidden_layers"], "learning_rate": search["learning_rates"],
        "activation": search["activations"], "optimizer": search["optimizers"],
        "epochs": search["epochs"],
    }, n_iter=search["trials"], random_state=config["seed"])
    return {
        f"search_{index + 1:02d}": {**values, "regularization": "l2",
            "regularizer_strength": settings["regularizer_strength"], "dropout_rate": 0.0,
            "batch_size": settings["batch_size"], "early_stopping": False}
        for index, values in enumerate(sampled)
    }


def neural_partition(train, config):
    fraction = config["neural"]["validation_actor_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_actor_fraction must be between 0 and 1.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=config["seed"])
    fit, validation = next(splitter.split(train.values, train.labels, groups=train.actors))
    labels = np.asarray(train.labels)
    if set(labels[fit]) != set(train.labels) or set(labels[validation]) != set(train.labels):
        raise ValueError("Neural fitting/validation partitions must contain all eight classes.")
    return fit, validation


def build_network(tf, spec, input_features, classes):
    layers = [tf.keras.layers.Input(shape=(input_features,))]
    regularizer = None
    if spec["regularization"]:
        constructor = getattr(tf.keras.regularizers, spec["regularization"])
        regularizer = constructor(spec["regularizer_strength"])
    for width in spec["hidden_units"]:
        layers.append(tf.keras.layers.Dense(width, activation=spec["activation"], kernel_regularizer=regularizer))
        if spec["dropout_rate"]:
            layers.append(tf.keras.layers.Dropout(spec["dropout_rate"]))
    layers.append(tf.keras.layers.Dense(classes, activation="softmax"))
    model = tf.keras.Sequential(layers)
    constructor = {"adam": tf.keras.optimizers.Adam, "sgd": tf.keras.optimizers.SGD}[spec["optimizer"]]
    model.compile(optimizer=constructor(learning_rate=spec["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_network(tf, spec, X, y, config, classes, validation_data=None, checkpoint=None):
    # No original seeds were specified; this explicit seed belongs to the new experiment.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])
    model = build_network(tf, spec, X.shape[1], classes)
    callbacks = []
    if spec["early_stopping"]:
        if validation_data is None or checkpoint is None:
            raise ValueError("Early stopping requires actor-held-out validation and a checkpoint path.")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=config["neural"]["patience"],
                                             restore_best_weights=False),
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_accuracy", mode="max",
                                               save_best_only=True),
        ]
    history = model.fit(X, y, epochs=spec["epochs"], batch_size=spec["batch_size"],
                        validation_data=validation_data, callbacks=callbacks, shuffle=True, verbose=0)
    if spec["early_stopping"]:
        model = tf.keras.models.load_model(checkpoint)
    elif checkpoint is not None:
        model.save(checkpoint)
    return model, {key: [float(value) for value in values] for key, values in history.history.items()}


def run_neural(train, test, config, config_path, output, progress=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}.")
    if train.task_kind != "classification" or train.target_name != "emotion" or len(set(train.labels)) != 8:
        raise ValueError("The original neural experiments require eight-class emotion prediction.")
    mode = config["neural"]["mode"]
    if mode not in {"variants", "search"}:
        raise ValueError("neural.mode must be variants or search.")
    metric = config["neural"]["selection_metric"]
    if metric != ("accuracy" if mode == "variants" else "f1_macro"):
        raise ValueError("Variants select accuracy; neural search selects macro F1.")
    specs = variant_specs(config) if mode == "variants" else search_specs(config)
    tf = tensorflow_backend()  # Fail before creating a run directory if the backend is unavailable.
    tf.config.experimental.enable_op_determinism()
    X, y = np.asarray(train.values, dtype=object), np.asarray(train.labels)
    labels = LabelEncoder().fit(y)
    encoded = labels.transform(y)
    classes = len(labels.classes_)
    scores, histories, recorded = {}, {}, []
    report = provenance(f"neural_actor_{mode}_v1", train, test, config, config_path)
    report["environment"]["packages"].update({"tensorflow-cpu": version("tensorflow-cpu"), "keras": version("keras")})
    # Long neural runs retain partial checkpoints on failure, but report.json is written only after completion.
    output.mkdir(parents=True, exist_ok=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mode == "variants":
            fit, validation = neural_partition(train, config)
            preprocessing = legacy_preprocessing(train).fit(X[fit])
            X_fit, X_val = preprocessing.transform(X[fit]), preprocessing.transform(X[validation])
            folds = [{"train_indices": fit.tolist(), "validation_indices": validation.tolist(),
                      "train_actors": sorted(set(np.asarray(train.actors)[fit].tolist())),
                      "validation_actors": sorted(set(np.asarray(train.actors)[validation].tolist()))}]
            for name, spec in specs.items():
                if progress:
                    progress(f"Training neural variant {name} ({spec['epochs']} epochs maximum)")
                model, histories[name] = fit_network(tf, spec, X_fit, encoded[fit], config, classes,
                    validation_data=(X_val, encoded[validation]), checkpoint=output / f"{name}.keras")
                predicted = labels.inverse_transform(np.argmax(model.predict(X_val, verbose=0), axis=1))
                scores[name] = classification_metrics(y[validation], predicted, labels.classes_)
            selected = max(scores, key=lambda name: scores[name][metric])
            model = tf.keras.models.load_model(output / f"{selected}.keras")
            final_fit = fit
        else:
            split, folds = actor_folds(train, config["neural"]["search"]["cv_folds"], config["seed"])
            for name, spec in specs.items():
                fold_scores = []
                for index, (fit, validation) in enumerate(split):
                    if progress:
                        progress(f"Training {name}: fold {index + 1}/{len(split)}")
                    preprocessing = legacy_preprocessing(train).fit(X[fit])
                    model, _ = fit_network(tf, spec, preprocessing.transform(X[fit]), encoded[fit], config, classes)
                    predicted = labels.inverse_transform(np.argmax(model.predict(preprocessing.transform(X[validation]), verbose=0), axis=1))
                    fold_scores.append(classification_metrics(y[validation], predicted, labels.classes_)[metric])
                scores[name] = {"fold_scores": fold_scores, f"mean_{metric}": float(np.mean(fold_scores))}
            selected = max(scores, key=lambda name: scores[name][f"mean_{metric}"])
            preprocessing = legacy_preprocessing(train).fit(X)
            model, histories[selected] = fit_network(tf, specs[selected], preprocessing.transform(X), encoded,
                config, classes, checkpoint=output / f"{selected}.keras")
            final_fit = np.arange(len(y))
        predicted = labels.inverse_transform(np.argmax(model.predict(preprocessing.transform(np.asarray(test.values, dtype=object)), verbose=0), axis=1))
        majority = DummyClassifier(strategy="most_frequent").fit(X[final_fit], y[final_fit])
        predictions = {"majority": majority.predict(np.asarray(test.values, dtype=object)), selected: predicted}
    recorded.extend({"category": item.category.__name__, "message": str(item.message)} for item in caught)
    report.update({"validation": {"selected_method": selected, "selection_metric": metric,
                                  "tie_break": "first candidate in configuration order",
                                  "folds": folds, "scores": scores},
                   "model_specs": specs, "histories": histories, "warnings": recorded,
                   "final_training_actors": sorted(set(np.asarray(train.actors)[final_fit].tolist())),
                   "metrics": {name: classification_metrics(test.labels, values, labels.classes_)
                               for name, values in predictions.items()}})
    joblib.dump({"preprocessing": preprocessing, "labels": labels, "majority": majority,
                 "feature_names": train.feature_names, "selected_method": selected}, output / "preprocessing.joblib")
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "actor", "emotion", *predictions])
        for index, (filename, actor, truth) in enumerate(zip(test.filenames, test.actors, test.labels)):
            writer.writerow([filename, actor, truth, *(values[index] for values in predictions.values())])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def predict_saved(output, values):
    """Reload the selected network with its exact feature preprocessing and class order."""
    bundle = joblib.load(output / "preprocessing.joblib")
    tf = tensorflow_backend()
    model = tf.keras.models.load_model(output / f"{bundle['selected_method']}.keras")
    probabilities = model.predict(bundle["preprocessing"].transform(np.asarray(values, dtype=object)), verbose=0)
    return bundle["labels"].inverse_transform(np.argmax(probabilities, axis=1))
