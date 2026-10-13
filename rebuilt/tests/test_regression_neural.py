"""Checks for target isolation, original preprocessing, and saved regression predictions."""

import csv
import importlib.util
import json
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ravdess_project.data import EMOTIONS, FeatureTable, load_dataset

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Training dependencies are not installed")
class RegressionNeuralTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = tomllib.loads((CONFIGS / "regression.toml").read_text())
        self.config.update(train_actors=[1, 2, 3, 4, 5, 6], test_actors=[19, 20])
        headers = ["", "vocal_channel", "emotion", "emotional_intensity", "statement", "repetition",
                   "sex", "filename", "kur", "pitch"]
        for filename, actors in (("Train_NOoutliers", self.config["train_actors"]),
                                 ("Test_ripulito (1)", self.config["test_actors"])):
            with (self.root / filename).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(headers)
                for actor in actors:
                    for i, emotion in enumerate(sorted(EMOTIONS)):
                        writer.writerow([i, "speech", emotion, "normal", "one", "first", "male" if actor % 2 else "female",
                                         f"03-01-{i+1:02d}-01-01-01-{actor:02d}.wav", actor * 2 + i * .3, actor + i])
        self.train, self.test = load_dataset(self.root, self.config)

    def test_kur_is_target_and_emotion_remains_predictor(self):
        self.assertEqual(self.train.target_name, "kur")
        self.assertEqual(self.train.task_kind, "regression")
        self.assertNotIn("kur", self.train.feature_names)
        self.assertIn("emotion", self.train.feature_names)
        self.assertEqual(self.train.labels[0], 2)
        self.assertEqual(set(self.train.source_labels), EMOTIONS)
        self.assertNotIn("class_counts", self.train.summary())
        self.assertEqual(self.train.summary()["numeric_features"], 1)

    def test_preprocessing_matches_original_label_encoders_and_scaler(self):
        import numpy as np
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from ravdess_project.preprocessing import legacy_preprocessing

        X = np.asarray(self.train.values, dtype=object)
        manual = np.empty(X.shape, dtype=float)
        for index in range(6):
            manual[:, index] = LabelEncoder().fit_transform(X[:, index])
        manual[:, 6:] = X[:, 6:].astype(float)
        expected = StandardScaler().fit_transform(manual)
        actual = legacy_preprocessing(self.train).fit_transform(X)
        np.testing.assert_array_equal(actual, expected)

    def test_unseen_categories_fail_instead_of_changing_mapping(self):
        import numpy as np
        from ravdess_project.preprocessing import legacy_preprocessing

        preprocessing = legacy_preprocessing(self.train).fit(np.asarray(self.train.values, dtype=object))
        unexpected = np.asarray(self.test.values, dtype=object)
        unexpected[0, 0] = "unknown"
        with self.assertRaisesRegex(ValueError, "unknown categories"):
            preprocessing.transform(unexpected)

    def test_regression_folds_do_not_treat_continuous_values_as_classes(self):
        from ravdess_project.validation import actor_folds

        folds, descriptions = actor_folds(self.train, 3, 42)
        self.assertEqual(len(folds), 3)
        for description in descriptions:
            self.assertFalse(set(description["train_actors"]) & set(description["validation_actors"]))

    def test_preserved_regression_spaces_and_explicit_seeds(self):
        from ravdess_project.regression import build_regressors

        candidates = build_regressors(self.train, self.config)
        self.assertEqual(len(candidates), 47)
        self.assertEqual([model.named_steps["regressor"].alpha for name, model in candidates.items() if name.startswith("lasso")],
                         [0.5, 0.99, 0.4, 0.1, 0.01, 0.001, 0.0001])
        self.assertEqual(candidates["gradient_boosting"].named_steps["regressor"].random_state, 0)
        forest = [model.named_steps["regressor"] for name, model in candidates.items() if name.startswith("random_forest")]
        self.assertEqual(len(forest), 30)
        self.assertTrue(all(model.n_estimators == 100 and model.random_state == 42 for model in forest))
        repeated = build_regressors(self.train, self.config)
        self.assertEqual([model.get_params() for model in forest],
                         [model.named_steps["regressor"].get_params() for name, model in repeated.items() if name.startswith("random_forest")])

    def test_selection_minimizes_mse_and_scaler_fits_only_training_fold(self):
        import numpy as np
        from sklearn.dummy import DummyRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from ravdess_project.validation import actor_folds, select_on_training

        recorded = []
        original = StandardScaler.fit
        numeric = replace(self.train, values=[[value] for value in self.train.labels],
                          feature_names=("x",), categorical_features=())
        candidates = {"bad": Pipeline([("scale", StandardScaler()), ("regressor", DummyRegressor(strategy="constant", constant=1000))]),
                      "mean": DummyRegressor(strategy="mean")}
        def recording(scaler, X, y=None, **kwargs):
            recorded.append(X.copy())
            return original(scaler, X, y, **kwargs)
        with patch.object(StandardScaler, "fit", recording):
            result = select_on_training(numeric, candidates, 3, 42, metric="mse")
        self.assertEqual(result["selected_method"], "mean")
        folds, _ = actor_folds(numeric, 3, 42)
        for X, (fit, _) in zip(recorded, folds):
            np.testing.assert_array_equal(X, np.asarray(numeric.values)[fit])

    def test_regression_artifacts_reload_and_refuse_overwrite(self):
        import joblib
        import numpy as np
        from ravdess_project.regression import run_regression

        config = dict(self.config, regression={**self.config["regression"], "alphas": [.1], "forest_trials": 1,
                                               "cv_folds": 3, "parallel_jobs": 1})
        config_path = CONFIGS / "regression.toml"
        output = self.root / "run"
        report = run_regression(self.train, self.test, config, config_path, output)
        bundle = joblib.load(output / "models.joblib")
        with (output / "predictions.csv").open() as stream:
            saved = list(csv.DictReader(stream))
        self.assertIsNone(report["class_order"])
        self.assertEqual(report["task"]["name"], "kur_regression")
        for name, model in bundle["models"].items():
            np.testing.assert_allclose(model.predict(np.asarray(self.test.values, dtype=object)),
                                       [float(row[name]) for row in saved], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_regression(self.train, self.test, config, config_path, output)
        json.loads((output / "report.json").read_text())

    def test_neural_predictors_exclude_emotion_but_include_kur(self):
        config = tomllib.loads((CONFIGS / "neural.toml").read_text())
        config.update(train_actors=self.config["train_actors"], test_actors=self.config["test_actors"])
        train, _ = load_dataset(self.root, config)
        self.assertNotIn("emotion", train.feature_names)
        self.assertIn("kur", train.feature_names)
        self.assertEqual(len(train.categorical_features), 5)
        self.assertEqual(set(train.labels), EMOTIONS)

    def test_neural_actor_partition_is_disjoint_repeatable(self):
        from ravdess_project.neural import neural_partition

        config = tomllib.loads((CONFIGS / "neural.toml").read_text())
        train = replace(self.train, labels=self.train.source_labels, task_kind="classification")
        fit, validation = neural_partition(train, config)
        repeated, _ = neural_partition(train, config)
        self.assertEqual(fit.tolist(), repeated.tolist())
        self.assertFalse(set(self.train.actors[i] for i in fit) & set(self.train.actors[i] for i in validation))

    def test_neural_specs_preserve_epochs_and_regularization(self):
        from ravdess_project.neural import search_specs, variant_specs

        config = tomllib.loads((CONFIGS / "neural.toml").read_text())
        specs = variant_specs(config)
        self.assertEqual(specs["baseline"]["epochs"], 200)
        self.assertEqual(specs["early_stopping"]["epochs"], 1)
        self.assertEqual(specs["l1"]["regularizer_strength"], .01)
        self.assertEqual(specs["l2"]["epochs"], 500)
        self.assertEqual(specs["dropout"]["dropout_rate"], .1)
        self.assertEqual(specs["dropout"]["batch_size"], 100)
        self.assertEqual(config["neural"]["search"]["hidden_layers"].count([10, 10, 10]), 2)
        self.assertEqual(len(search_specs(config)), 15)
        self.assertEqual(search_specs(config), search_specs(config))

    def test_missing_tensorflow_creates_no_result_directory(self):
        from ravdess_project.neural import run_neural

        config = tomllib.loads((CONFIGS / "neural.toml").read_text())
        train = replace(self.train, labels=self.train.source_labels, task_kind="classification", target_name="emotion")
        output = self.root / "neural-run"
        with patch("ravdess_project.neural.tensorflow_backend", side_effect=ImportError("unavailable")):
            with self.assertRaises(ImportError):
                run_neural(train, train, config, CONFIGS / "neural.toml", output)
        self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow backend is not installed")
class TensorFlowIntegrationTests(unittest.TestCase):
    def test_original_architectures_and_small_checkpoint_round_trip(self):
        import numpy as np
        import tensorflow as tf
        from ravdess_project.neural import build_network, fit_network, variant_specs

        config = tomllib.loads((CONFIGS / "neural.toml").read_text())
        specs = variant_specs(config)
        for name, spec in specs.items():
            model = build_network(tf, spec, 202, 8)
            dense = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
            self.assertEqual([layer.units for layer in dense], [128, 64, 8])
            self.assertEqual(dense[-1].activation.__name__, "softmax")
            dropout = [layer.rate for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)]
            self.assertEqual(dropout, [.1, .1] if name == "dropout" else [])
            self.assertEqual(dense[0].kernel_regularizer is not None, name in {"l1", "l2"})
        spec = {**specs["baseline"], "epochs": 1}
        X = np.random.default_rng(42).normal(size=(16, 202)).astype("float32")
        y = np.arange(16) % 8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model, history = fit_network(tf, spec, X, y, config, 8, checkpoint=path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_allclose(model.predict(X, verbose=0), restored.predict(X, verbose=0), atol=0, rtol=0)
            self.assertEqual(len(history["loss"]), 1)
            early_spec = {**specs["early_stopping"], "epochs": 1}
            early, _ = fit_network(tf, early_spec, X, y, config, 8,
                                   validation_data=(X, y), checkpoint=Path(directory) / "best.keras")
            self.assertTrue((Path(directory) / "best.keras").is_file())
            self.assertEqual(early.output_shape[-1], 8)

    def test_seeded_training_repeats_exactly(self):
        import numpy as np
        import tensorflow as tf
        from ravdess_project.neural import fit_network, variant_specs

        tf.config.experimental.enable_op_determinism()
        config = tomllib.loads((CONFIGS / "neural.toml").read_text())
        spec = {**variant_specs(config)["baseline"], "epochs": 2}
        X = np.random.default_rng(42).normal(size=(24, 202)).astype("float32")
        y = np.arange(24) % 8
        first, first_history = fit_network(tf, spec, X, y, config, 8)
        weights = first.get_weights()
        probabilities = first.predict(X, verbose=0)
        second, second_history = fit_network(tf, spec, X, y, config, 8)
        self.assertEqual(first_history, second_history)
        for expected, actual in zip(weights, second.get_weights()):
            np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(probabilities, second.predict(X, verbose=0))
