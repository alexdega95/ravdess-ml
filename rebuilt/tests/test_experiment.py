import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from ravdess_project.data import read_features
from ravdess_project.experiment import run_baselines
from test_data import METADATA, write_table


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Training dependencies are not installed")
class ExperimentTests(unittest.TestCase):
    def test_artifacts_repeatability_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, test_path = root / "train.csv", root / "test.csv"
            write_table(train_path, [1, 2])
            write_table(test_path, [19])
            train = read_features(train_path, METADATA, "emotion")
            test = read_features(test_path, METADATA, "emotion")
            config = {
                "seed": 42,
                "random_forest": {"n_estimators": 5, "min_samples_leaf": 1, "n_jobs": 1},
            }
            config_path = root / "config.toml"
            config_path.write_text("seed = 42\n")
            first = run_baselines(train, test, config, config_path, root / "first")
            second = run_baselines(train, test, config, config_path, root / "second")
            self.assertEqual(first["metrics"], second["metrics"])
            self.assertEqual(
                (root / "first/predictions.csv").read_bytes(),
                (root / "second/predictions.csv").read_bytes(),
            )
            for name in ("report.json", "predictions.csv", "models.joblib"):
                self.assertTrue((root / "first" / name).is_file())
            saved = json.loads((root / "first/report.json").read_text())
            self.assertEqual(saved["inputs"]["train"]["sha256"], train.digest)
            self.assertEqual(saved["model_parameters"]["random_forest"]["random_state"], 42)
            import joblib
            import numpy as np

            restored = joblib.load(root / "first/models.joblib")
            self.assertEqual(restored["feature_names"], train.feature_names)
            np.testing.assert_array_equal(
                restored["models"]["majority"].predict(test.values), ["angry"] * 8
            )
            before = (root / "first/report.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_baselines(train, test, config, config_path, root / "first")
            self.assertEqual((root / "first/report.json").read_bytes(), before)
