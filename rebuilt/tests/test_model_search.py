import csv
import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from ravdess_project.data import positive_negative_task, read_features
from test_data import METADATA, write_table

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Training dependencies are not installed")
class SearchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        write_table(self.root / "train.csv", [1, 2, 3, 4, 5, 6])
        write_table(self.root / "test.csv", [19, 20])
        self.train = read_features(self.root / "train.csv", METADATA, "emotion")
        self.test = read_features(self.root / "test.csv", METADATA, "emotion")

    def test_svm_preserves_exploration_and_full_grid(self):
        from ravdess_project.model_search import build_search_candidates

        config = tomllib.loads((ROOT / "configs/svm.toml").read_text())
        candidates = build_search_candidates(config)
        self.assertEqual(len(candidates), 35)  # majority + 3 linear + 3 kernels + 28 grid entries
        grid = [model.named_steps["classifier"] for name, model in candidates.items() if name.startswith("svc_grid")]
        self.assertEqual({model.C for model in grid}, {0.0001, 0.001, 0.01, 0.1, 1, 10, 100})
        self.assertEqual({model.kernel for model in grid}, {"linear", "poly", "rbf", "sigmoid"})
        self.assertTrue(all(model.gamma == "scale" for model in grid))
        self.assertEqual(candidates["svc_auto_rbf"].named_steps["classifier"].gamma, "auto")
        self.assertEqual(candidates["linear_svc_C_1.0"].named_steps["classifier"].random_state, 42)

    def test_gradient_sampling_is_repeatable_and_matches_original_space(self):
        from ravdess_project.model_search import build_search_candidates

        config = tomllib.loads((ROOT / "configs/gradient_boosting.toml").read_text())
        first, second = build_search_candidates(config), build_search_candidates(config)
        self.assertEqual(len(first), 11)
        for name in first:
            if name == "majority":
                continue
            model = first[name].named_steps["classifier"]
            for field in ("learning_rate", "max_depth", "min_samples_split"):
                self.assertEqual(getattr(model, field), getattr(second[name].named_steps["classifier"], field))
                self.assertIn(getattr(model, field), config["search"]["space"][field])
            self.assertEqual(model.n_estimators, 100)
            self.assertEqual(model.random_state, 0)

    def test_binary_run_preserves_emotions_predictions_and_model_loading(self):
        import joblib
        import numpy as np
        from ravdess_project.model_search import run_search

        config = tomllib.loads((ROOT / "configs/svm.toml").read_text())
        config["search"]["cv_folds"] = 3
        config["search"]["space"] = {"C": [0.1], "kernel": ["rbf"]}
        config["exploration"] = {"linear_c": [0.1], "kernels": ["rbf"]}
        train, test = positive_negative_task(self.train), positive_negative_task(self.test)
        path = self.root / "settings.toml"
        path.write_text("task = 'positive_negative'\n")
        output = self.root / "run"
        report = run_search(train, test, config, path, output)
        self.assertEqual(report["class_order"], [0, 1])
        self.assertEqual(report["task"]["positive_emotions"], ["calm", "happy"])
        self.assertEqual(report["validation"]["selection_metric"], "mean_accuracy")
        saved = joblib.load(output / "models.joblib")
        with (output / "predictions.csv").open() as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        self.assertIn("emotion", reader.fieldnames)
        self.assertIn("positive", reader.fieldnames)
        self.assertEqual([row["emotion"] for row in rows], test.source_labels)
        self.assertEqual([int(row["positive"]) for row in rows], test.labels)
        for name, model in saved["models"].items():
            np.testing.assert_array_equal(model.predict(test.values), [int(row[name]) for row in rows])
        for name, metrics in report["metrics"].items():
            if name != "majority":
                self.assertTrue(0 <= metrics["roc_auc"] <= 1)
        before = (output / "report.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_search(train, test, config, path, output)
        self.assertEqual((output / "report.json").read_bytes(), before)

    def test_search_rejects_wrong_task_and_invalid_trial_count(self):
        from ravdess_project.model_search import build_search_candidates

        config = tomllib.loads((ROOT / "configs/svm.toml").read_text())
        config["task"] = "emotion"
        with self.assertRaisesRegex(ValueError, "positive_negative"):
            build_search_candidates(config)
        config = tomllib.loads((ROOT / "configs/gradient_boosting.toml").read_text())
        config["search"]["n_iter"] = 100
        with self.assertRaisesRegex(ValueError, "n_iter"):
            build_search_candidates(config)

    def test_parallel_validation_matches_serial_scores_and_fold_order(self):
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from ravdess_project.validation import select_on_training

        train = positive_negative_task(self.train)
        candidates = {
            "majority": DummyClassifier(strategy="most_frequent"),
            "svm": Pipeline([("scale", StandardScaler()), ("classifier", SVC(C=0.1))]),
        }
        serial = select_on_training(train, candidates, 3, 42, metric="accuracy", jobs=1)
        parallel = select_on_training(train, candidates, 3, 42, metric="accuracy", jobs=2)
        for key in ("scores", "folds", "selected_method", "warnings", "candidate_parameters"):
            self.assertEqual(serial[key], parallel[key])

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "Optional XGBoost dependency is not installed")
    def test_xgboost_label_encoding_clone_and_round_trip(self):
        import joblib
        import numpy as np
        from importlib.metadata import version
        from sklearn.base import clone
        from xgboost import XGBClassifier
        from ravdess_project.model_search import LabelEncodedClassifier, build_search_candidates, run_search

        config = tomllib.loads((ROOT / "configs/xgboost.toml").read_text())
        candidates = build_search_candidates(config)
        self.assertEqual(len(candidates), 11)
        model = clone(LabelEncodedClassifier(XGBClassifier(
            n_estimators=3, max_depth=2, n_jobs=1, objective="multi:softmax", random_state=42,
        )))
        model.fit(self.train.values, self.train.labels)
        self.assertEqual(model.classes_.tolist(), sorted(set(self.train.labels)))
        predicted = model.predict(self.test.values)
        self.assertTrue(set(predicted).issubset(set(self.train.labels)))
        joblib.dump(model, self.root / "model.joblib")
        np.testing.assert_array_equal(joblib.load(self.root / "model.joblib").predict(self.test.values), predicted)
        self.assertEqual(model.feature_importances_.shape, (2,))
        config["classifier"]["n_estimators"] = 3
        config["search"]["n_iter"] = 1
        config["search"]["cv_folds"] = 3
        path = self.root / "xgboost.toml"
        path.write_text("task = 'emotion'\n")
        report = run_search(self.train, self.test, config, path, self.root / "xgboost-run")
        saved_report = json.loads((self.root / "xgboost-run/report.json").read_text())
        self.assertEqual(saved_report["validation"], report["validation"])
        self.assertEqual(report["environment"]["packages"]["xgboost"], version("xgboost"))
