import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "sklearn", "tslearn", "pyts"))


@unittest.skipUnless(AVAILABLE, "Optional time-series dependencies are not installed")
class SAXDTWTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from ravdess_project.data import sha256
        from ravdess_project.time_series import SeriesTable

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = {"target": "vocal_channel", "sax_dtw": {
            "n_segments": 4, "alphabet_size_avg": 5,
            "metrics": ["dtw", "dtw_sakoechiba", "dtw_itakura"]}}
        self.config_path = self.root / "config.toml"
        self.config_path.write_text("# fixture config")
        self.tables = []
        rng = np.random.default_rng(42)
        for split, actors in (("train", [1, 2, 3, 4]), ("test", [19, 20])):
            values = rng.normal(size=(len(actors), 16))
            path = self.root / f"{split}.npy"
            np.save(path, values)
            metadata = self.root / f"{split}.csv"
            metadata.write_text("fixture")
            self.tables.append(SeriesTable(path, values, ["song", "speech"] * (len(actors) // 2),
                [f"{split}-{actor}.wav" for actor in actors], actors, sha256(path), metadata,
                sha256(metadata), "vocal_channel", (0, 16), "raw"))

    def test_symbols_and_all_distances_match_notebook_and_reload(self):
        import joblib
        import numpy as np
        from pyts.classification import KNeighborsClassifier
        from tslearn.preprocessing import TimeSeriesScalerMinMax
        from tslearn.piecewise import SymbolicAggregateApproximation
        from ravdess_project.sax_dtw import prepare, evaluate

        train, test = self.tables
        output = self.root / "run"
        prep = prepare(train, test, self.config, self.config_path, output)
        self.assertTrue(prep["validation"]["all_notebook_symbols_exact"])
        report = evaluate(train, test, self.config, self.config_path, output, batch_size=1)
        transform = SymbolicAggregateApproximation(n_segments=4, alphabet_size_avg=5)
        reference_train = transform.fit_transform(TimeSeriesScalerMinMax().fit_transform(train.values))[:, :, 0]
        reference_test = transform.transform(TimeSeriesScalerMinMax().fit_transform(test.values))[:, :, 0]
        saved = joblib.load(output / "models.joblib")["models"]
        for name in self.config["sax_dtw"]["metrics"]:
            expected = KNeighborsClassifier(metric=name).fit(reference_train, train.labels).predict(reference_test)
            np.testing.assert_array_equal(saved[name].predict(test.values), expected)
            self.assertEqual(saved[name].predict(test.values[:1]).shape, (1,))
        self.assertEqual(report["evaluation"]["model_selection"], "None; report all configured metrics, as in the notebooks.")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(train, test, self.config, self.config_path, output)

    def test_interrupt_resume_and_corrupted_predictions_rejected(self):
        from ravdess_project.sax_dtw import prepare, evaluate

        train, test = self.tables
        output = self.root / "run"
        prepare(train, test, self.config, self.config_path, output)
        def interrupt(message):
            raise RuntimeError("Interrupted after saved batch")
        with self.assertRaisesRegex(RuntimeError, "Interrupted"):
            evaluate(train, test, self.config, self.config_path, output, batch_size=1, progress=interrupt)
        state = json.loads((output / "status.json").read_text())
        self.assertEqual(len(state["predictions"]["dtw"]), 1)
        resumed = evaluate(train, test, self.config, self.config_path, output, batch_size=1)
        direct_output = self.root / "direct"
        prepare(train, test, self.config, self.config_path, direct_output)
        direct = evaluate(train, test, self.config, self.config_path, direct_output, batch_size=2)
        self.assertEqual(resumed["metrics"], direct["metrics"])
        state = json.loads((output / "status.json").read_text())
        state["predictions"]["dtw"][0] = "speech" if state["predictions"]["dtw"][0] == "song" else "song"
        (output / "status.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "checksum"):
            evaluate(train, test, self.config, self.config_path, output)

    def test_modified_symbols_and_config_are_rejected(self):
        import numpy as np
        from ravdess_project.sax_dtw import prepare, evaluate

        train, test = self.tables
        output = self.root / "run"
        prepare(train, test, self.config, self.config_path, output)
        values = np.load(output / "sax_train.npy")
        values[0, 0] += 1
        np.save(output / "sax_train.npy", values)
        with self.assertRaisesRegex(ValueError, "artifact checksum"):
            evaluate(train, test, self.config, self.config_path, output)
        self.config_path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "configuration"):
            evaluate(train, test, self.config, self.config_path, output)
