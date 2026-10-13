import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ravdess_project.data import read_features
from test_data import METADATA, write_table


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Training dependencies are not installed")
class TabularTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        write_table(self.root / "train.csv", [1, 2, 3, 4, 5, 6])
        write_table(self.root / "test.csv", [19, 20])
        self.train = read_features(self.root / "train.csv", METADATA, "emotion")
        self.test = read_features(self.root / "test.csv", METADATA, "emotion")
        self.config = {
            "seed": 42,
            "random_forest": {"n_estimators": 5, "n_jobs": 1},
            "comparison": {
                "cv_folds": 3,
                "methods": ["majority", "random_forest", "decision_tree", "pca", "rfe",
                            "select_k_best", "select_from_model"],
                "pca_components": 1, "rfe_features": 1, "rfe_step": 1,
                "k_best_features": 1, "decision_tree_min_samples_leaf": 1,
            },
        }

    def test_actor_folds_are_disjoint_repeatable_and_cover_training_once(self):
        from ravdess_project.tabular import actor_folds

        folds, descriptions = actor_folds(self.train, 3, 42)
        _, repeated = actor_folds(self.train, 3, 42)
        self.assertEqual(descriptions, repeated)
        all_validation = []
        for (fit, validation), description in zip(folds, descriptions):
            self.assertFalse(set(description["train_actors"]) & set(description["validation_actors"]))
            self.assertFalse(set(fit) & set(validation))
            all_validation.extend(validation.tolist())
        self.assertEqual(sorted(all_validation), list(range(len(self.train.labels))))

    def test_invalid_fold_counts_are_rejected(self):
        from ravdess_project.tabular import actor_folds

        for count in (1, 7, "3"):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "cv_folds"):
                actor_folds(self.train, count, 42)

    def test_scaler_fits_only_on_each_folds_training_rows(self):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from ravdess_project.tabular import actor_folds, compare_on_training

        self.config["comparison"]["methods"] = ["pca"]
        recorded = []
        original_fit = StandardScaler.fit

        def record_fit(scaler, X, y=None, **kwargs):
            recorded.append(X.copy())
            return original_fit(scaler, X, y, **kwargs)

        with patch.object(StandardScaler, "fit", record_fit):
            compare_on_training(self.train, self.config)
        folds, _ = actor_folds(self.train, 3, 42)
        self.assertEqual(len(recorded), 3)
        for actual, (fit, _) in zip(recorded, folds):
            np.testing.assert_array_equal(actual, np.asarray(self.train.values)[fit])

    def test_rfe_receives_true_training_labels_in_each_fold(self):
        import numpy as np
        from sklearn.feature_selection import RFE
        from ravdess_project.tabular import actor_folds, compare_on_training

        self.config["comparison"]["methods"] = ["rfe"]
        recorded, original_fit = [], RFE.fit

        def record_fit(selector, X, y, **kwargs):
            recorded.append(y.copy())
            return original_fit(selector, X, y, **kwargs)

        with patch.object(RFE, "fit", record_fit):
            compare_on_training(self.train, self.config)
        folds, _ = actor_folds(self.train, 3, 42)
        self.assertEqual(len(recorded), 3)
        for actual, (fit, _) in zip(recorded, folds):
            np.testing.assert_array_equal(actual, np.asarray(self.train.labels)[fit])

    def test_invalid_method_lists_are_rejected(self):
        from ravdess_project.tabular import build_models

        for methods in ([], ["pca", "pca"], ["unknown"]):
            self.config["comparison"]["methods"] = methods
            with self.subTest(methods=methods), self.assertRaisesRegex(ValueError, "methods"):
                build_models(self.config)

    def test_feature_names_follow_the_fitted_selector_and_pca_components(self):
        import numpy as np
        from ravdess_project.tabular import build_models, feature_details

        self.config["comparison"]["methods"] = ["pca", "rfe", "select_k_best", "select_from_model"]
        X = np.asarray(self.train.values).copy()
        # Avoid zero within-class variance in this tiny synthetic ANOVA fixture.
        X[:, 1] += np.arange(len(X)) * 0.001
        for name, model in build_models(self.config).items():
            with self.subTest(name=name):
                model.fit(X, self.train.labels)
                details = feature_details(model, self.train.feature_names)
                if name == "pca":
                    self.assertEqual(details["active_features"], ["PC1"])
                    self.assertEqual(set(details["component_loadings"]["PC1"]), set(self.train.feature_names))
                else:
                    expected = np.asarray(self.train.feature_names)[model.named_steps["reduce"].get_support()].tolist()
                    self.assertEqual(details["active_features"], expected)
                self.assertEqual(
                    {item["feature"] for item in details["feature_importances"]},
                    set(details["active_features"]),
                )

    def test_comparison_selection_artifacts_and_feature_names(self):
        import joblib
        import numpy as np
        from ravdess_project.tabular import run_comparison

        config_path = self.root / "config.toml"
        config_path.write_text("seed = 42\n")
        output = self.root / "run"
        report = run_comparison(self.train, self.test, self.config, config_path, output)
        validation = report["validation"]
        scores = validation["scores"]
        self.assertEqual(set(scores), set(self.config["comparison"]["methods"]))
        selected = max(scores, key=lambda name: scores[name]["mean_f1_macro"])
        self.assertEqual(validation["selected_method"], selected)
        self.assertEqual(set(report["metrics"]), {"majority", selected})
        for name in report["feature_analysis"]:
            details = report["feature_analysis"][name]
            for importance in details.get("feature_importances", []):
                self.assertIn(importance["feature"], details["active_features"])
        artifact = joblib.load(output / "models.joblib")
        prediction = artifact["models"][selected].predict(self.test.values)
        self.assertEqual(len(prediction), 16)
        self.assertEqual(json.loads((output / "report.json").read_text())["validation"], validation)
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_comparison(self.train, self.test, self.config, config_path, output)
