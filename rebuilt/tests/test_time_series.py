import csv
import importlib.util
import json
import tempfile
import unittest
import wave
from pathlib import Path

SCIENCE = bool(importlib.util.find_spec("sklearn"))


@unittest.skipUnless(SCIENCE, "Scientific dependencies are not installed")
class TimeSeriesTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from ravdess_project.audio_preparation import decode_filename

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {"target": "vocal_channel", "seed": 42, "train_actors": [1, 2, 3, 4],
                       "test_actors": [19, 20], "time_series": {"representation": "raw", "start": 2,
                       "stop": 7, "cv_folds": 2, "methods": ["majority", "euclidean"]}}
        for split in ("train", "test"):
            rows, values = [], []
            for actor in self.config[f"{split}_actors"]:
                for channel in (1, 2):
                    filename = f"03-{channel:02d}-01-01-01-01-{actor:02d}.wav"
                    rows.append(decode_filename(filename))
                    values.append([0, 0, channel, channel + 1, channel + actor, channel + 2, channel + 3, float("nan")])
            np.save(self.root / f"RavdessAudioOnlyNumpy__X_{split}.npy", values)
            with (self.root / f"RavdessAudioOnlyNumpy__Y_{split}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def test_loader_retains_interval_and_filename_alignment(self):
        import numpy as np
        from ravdess_project.time_series import load_series

        train, test = load_series(self.root, self.config)
        self.assertEqual(train.values.shape, (8, 5))
        self.assertEqual(set(train.labels), {"song", "speech"})
        self.assertEqual(train.summary()["interval"], [2, 7])
        self.assertFalse(set(train.actors) & set(test.actors))
        source = np.load(self.root / "RavdessAudioOnlyNumpy__X_train.npy")
        np.testing.assert_array_equal(train.values, source[:, 2:7])

    def test_padding_in_interval_is_rejected(self):
        from ravdess_project.time_series import load_series

        self.config["time_series"]["stop"] = 8
        with self.assertRaisesRegex(ValueError, "padding/nonfinite"):
            load_series(self.root, self.config)

    def test_metadata_disagreement_is_rejected(self):
        from ravdess_project.time_series import load_series

        path = self.root / "RavdessAudioOnlyNumpy__Y_train.csv"
        path.write_text(path.read_text().replace("speech", "song", 1))
        with self.assertRaisesRegex(ValueError, "disagree"):
            load_series(self.root, self.config)

    def test_scale_is_per_recording_and_constant_series_is_zero(self):
        import numpy as np
        from ravdess_project.time_series import SeriesMinMax

        scale = SeriesMinMax().fit([[1, 2, 3]])
        np.testing.assert_array_equal(scale.transform([[10, 20, 30], [9, 9, 9]]), [[0, .5, 1], [0, 0, 0]])
        original = np.array([[1, 2, 3]], dtype=float)
        scale.transform(original)
        np.testing.assert_array_equal(original, [[1, 2, 3]])

    def test_nearest_neighbor_keeps_original_pyts_default_one(self):
        from ravdess_project.time_series import series_models

        model = series_models(self.config)["euclidean"].named_steps["classifier"]
        self.assertEqual(model.n_neighbors, 1)
        self.assertEqual(model.metric, "euclidean")
        self.assertEqual(model.weights, "uniform")

    def test_vocal_artifacts_reload_and_have_correct_target_header(self):
        import numpy as np
        import joblib
        from ravdess_project.time_series import load_series, run_series

        train, test = load_series(self.root, self.config)
        config_path = self.root / "config.toml"
        config_path.write_text("# fixture")
        output = self.root / "run"
        report = run_series(train, test, self.config, config_path, output)
        with (output / "predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertIn("vocal_channel", rows[0])
        self.assertNotIn("emotion", rows[0])
        models = joblib.load(output / "models.joblib")["models"]
        for name, model in models.items():
            np.testing.assert_array_equal(model.predict(test.values), [row[name] for row in rows])
        self.assertEqual(report["inputs"]["train"]["target"], "vocal_channel")
        with self.assertRaisesRegex(ValueError, "already exists"):
            run_series(train, test, self.config, config_path, output)

    def test_pcm_float32_decoding_and_stereo_mean(self):
        import numpy as np
        from ravdess_project.audio_preparation import load_pcm

        pcm = np.array([[100, 300], [-32768, 32767]], dtype="<i2")
        path = self.root / "stereo.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(48000)
            stream.writeframes(pcm.tobytes())
        expected = (pcm.astype("float32") / np.float32(32768)).mean(axis=1, dtype="float32")
        np.testing.assert_array_equal(load_pcm(path), expected)

    def test_audio_import_uses_training_maximum_and_preserves_actor_split(self):
        import numpy as np
        from ravdess_project.audio_preparation import build_audio

        audio = self.root / "audio"
        audio.mkdir()
        for actor in range(1, 25):
            length = 80 if actor < 19 else 100
            path = audio / f"03-01-01-01-01-01-{actor:02d}.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(48000)
                stream.writeframes(np.arange(length, dtype="<i2").tobytes())
        output = self.root / "prepared"
        report = build_audio(audio, output)
        self.assertEqual(report["pad_length_from_training"], 80)
        self.assertEqual(report["partitions"]["test"]["clipped_recordings"], 6)
        self.assertEqual(np.load(output / "RavdessAudioOnlyNumpy__X_train.npy").shape, (18, 80))
        self.assertEqual(np.load(output / "RavdessAudioOnlyNumpy__X_test.npy").shape, (6, 80))
        self.assertTrue((output / "preparation.json").is_file())

    @unittest.skipUnless(importlib.util.find_spec("tslearn"), "Optional tslearn is not installed")
    def test_scaling_matches_tslearn_on_train_and_unseen_series(self):
        import numpy as np
        from tslearn.preprocessing import TimeSeriesScalerMinMax
        from ravdess_project.time_series import SeriesMinMax

        X = np.array([[1., 4., 9.], [0., 0., 0.]])
        test = np.array([[-100., 2., 200.]])
        actual = SeriesMinMax().fit(X)
        expected = TimeSeriesScalerMinMax().fit(X)
        np.testing.assert_array_equal(actual.transform(X), expected.transform(X).reshape(X.shape))
        np.testing.assert_array_equal(actual.transform(test), expected.transform(test).reshape(test.shape))

    @unittest.skipUnless(importlib.util.find_spec("pyts"), "Optional pyts is not installed")
    def test_euclidean_predictions_match_original_pyts_wrapper(self):
        import numpy as np
        from pyts.classification import KNeighborsClassifier as OriginalKNN
        from ravdess_project.time_series import series_models, load_series

        train, test = load_series(self.root, self.config)
        pipeline = series_models(self.config)["euclidean"].fit(train.values, train.labels)
        scale = pipeline.named_steps["scale"]
        expected = OriginalKNN(metric="euclidean").fit(scale.transform(train.values), train.labels)
        np.testing.assert_array_equal(expected.predict(scale.transform(test.values)), pipeline.predict(test.values))

    @unittest.skipUnless(importlib.util.find_spec("sktime"), "Optional sktime is not installed")
    def test_rocket_adapters_fit_and_save_real_small_models(self):
        import joblib
        import numpy as np
        from ravdess_project.time_series import series_models

        config = {"seed": 42, "time_series": {"methods": ["rocket", "minirocket"],
                   "num_kernels": 84, "transform_jobs": 1}}
        X = np.random.default_rng(42).normal(size=(16, 32))
        labels = ["a", "b"] * 8
        for name, model in series_models(config).items():
            model.fit(X, labels)
            path = self.root / f"{name}.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(model.predict(X), joblib.load(path).predict(X))

    @unittest.skipUnless(importlib.util.find_spec("sktime"), "Optional sktime is not installed")
    def test_shape_only_rocket_cache_preserves_all_fold_scores_and_parameters(self):
        import numpy as np
        from dataclasses import replace
        from ravdess_project.time_series import load_series, compare_series

        train, _ = load_series(self.root, self.config)
        # Enough length for both transforms; actor labels/groups/order are retained.
        train = replace(train, values=np.random.default_rng(42).normal(size=(8, 32)), interval=(0, 32))
        config = {**self.config, "time_series": {**self.config["time_series"],
                  "methods": ["majority", "euclidean", "rocket", "minirocket"],
                  "num_kernels": 84, "transform_jobs": 1}}
        naive = compare_series(train, config)
        config["time_series"]["reuse_rocket_features"] = True
        cached = compare_series(train, config)
        for field in ("scores", "folds", "selected_method", "candidate_parameters", "selection_metric", "tie_break"):
            self.assertEqual(naive[field], cached[field], field)
        self.assertFalse(cached["execution_optimization"]["data_dependent_minirocket_cached"])

    @unittest.skipUnless(importlib.util.find_spec("sktime"), "Optional sktime is not installed")
    def test_shape_only_rocket_kernels_and_features_match_fitting_each_real_fold(self):
        import numpy as np
        from ravdess_project.time_series import RocketFeatures, SeriesMinMax

        X = np.random.default_rng(42).normal(size=(12, 64))
        scaled = SeriesMinMax().fit_transform(X)
        shape_only = RocketFeatures("rocket", 84, 42, 1).fit(np.zeros((1, 64)))
        all_features = shape_only.transform(scaled)
        for indices in (np.arange(0, 8), np.arange(4, 12)):
            ordinary = RocketFeatures("rocket", 84, 42, 1).fit(scaled[indices])
            for left, right in zip(shape_only.transformer_.kernels, ordinary.transformer_.kernels):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(all_features[indices], ordinary.transform(scaled[indices]))

    @unittest.skipUnless(importlib.util.find_spec("sktime"), "Optional sktime is not installed")
    def test_chunked_rocket_cache_matches_direct_and_reloads_verified_blocks(self):
        import numpy as np
        from dataclasses import replace
        from unittest.mock import patch
        from ravdess_project.time_series import load_series, compare_series, RocketFeatures

        train, _ = load_series(self.root, self.config)
        train = replace(train, values=np.random.default_rng(42).normal(size=(8, 32)), interval=(0, 32))
        config = {**self.config, "time_series": {**self.config["time_series"],
                  "methods": ["rocket"], "num_kernels": 84, "transform_jobs": 1,
                  "reuse_rocket_features": True, "rocket_batch_size": 3}}
        direct = compare_series(train, config)
        cache = self.root / "feature-cache"
        chunked = compare_series(train, config, feature_cache=cache)
        self.assertEqual(direct["scores"], chunked["scores"])
        with patch.object(RocketFeatures, "transform", side_effect=AssertionError("Recomputed completed features")):
            reloaded = compare_series(train, config, feature_cache=cache)
        self.assertEqual(direct["scores"], reloaded["scores"])
        state = json.loads((cache / "state.json").read_text())
        self.assertEqual(state["completed_rows"], 8)
        self.assertEqual(len(state["blocks"]), 3)
        matrix = np.load(cache / "rocket.npy", mmap_mode="r+")
        matrix[0, 0] += 1
        matrix.flush()
        with self.assertRaisesRegex(ValueError, "checksum"):
            compare_series(train, config, feature_cache=cache)

    @unittest.skipUnless(importlib.util.find_spec("sktime"), "Optional sktime is not installed")
    def test_rocket_cache_resumes_after_a_completed_batch(self):
        import numpy as np
        from dataclasses import replace
        from ravdess_project.time_series import load_series, compare_series

        train, _ = load_series(self.root, self.config)
        train = replace(train, values=np.random.default_rng(42).normal(size=(8, 32)), interval=(0, 32))
        config = {**self.config, "time_series": {**self.config["time_series"],
                  "methods": ["rocket"], "num_kernels": 84, "transform_jobs": 1,
                  "reuse_rocket_features": True, "rocket_batch_size": 3}}
        cache = self.root / "interrupted-features"
        def interrupt(message):
            if message.startswith("Saved ROCKET features"):
                raise RuntimeError("Simulated interruption")
        with self.assertRaisesRegex(RuntimeError, "Simulated interruption"):
            compare_series(train, config, progress=interrupt, feature_cache=cache)
        self.assertEqual(json.loads((cache / "state.json").read_text())["completed_rows"], 3)
        resumed = compare_series(train, config, feature_cache=cache)
        direct = compare_series(train, config)
        self.assertEqual(direct["scores"], resumed["scores"])
        self.assertEqual(json.loads((cache / "state.json").read_text())["completed_rows"], 8)
