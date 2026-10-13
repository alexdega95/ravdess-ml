import csv
import tempfile
import unittest
from pathlib import Path

from ravdess_project.data import EMOTIONS, load_dataset, positive_negative_task, read_features, validate_split

METADATA = ["emotion", "actor", "filename"]


def write_table(path, actors, *, features=("feature_b", "feature_a")):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*METADATA, *features])
        for actor in actors:
            for i, emotion in enumerate(sorted(EMOTIONS)):
                writer.writerow([emotion, actor, f"{actor}-{emotion}.wav", i + actor / 10, i * 2])


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.train_path, self.test_path = self.root / "train.csv", self.root / "test.csv"
        write_table(self.train_path, [1, 2])
        write_table(self.test_path, [19])
        self.config = {"train_actors": [1, 2], "test_actors": [19]}

    def read(self, path):
        return read_features(path, METADATA, "emotion")

    def change_first_row(self, column, value):
        with self.train_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        rows[0][column] = value
        with self.train_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_preserves_feature_and_row_order_without_changing_input(self):
        before = self.train_path.read_bytes()
        table = self.read(self.train_path)
        self.assertEqual(table.feature_names, ("feature_b", "feature_a"))
        self.assertEqual(table.values[0], [0.1, 0.0])
        self.assertEqual(table.filenames[0], "1-angry.wav")
        self.assertEqual(table.labels, sorted(EMOTIONS) * 2)
        self.assertEqual(table.summary()["rows"], 16)
        self.assertEqual(self.train_path.read_bytes(), before)

    def test_accepts_disjoint_configured_actor_split(self):
        validate_split(self.read(self.train_path), self.read(self.test_path), self.config)

    def test_rejects_nonfinite_and_missing_numeric_data(self):
        for value in ("nan", "inf", "-inf", "", "not-a-number"):
            with self.subTest(value=value):
                self.change_first_row("feature_b", value)
                with self.assertRaisesRegex(ValueError, "feature_b"):
                    self.read(self.train_path)

    def test_rejects_duplicate_filenames(self):
        self.change_first_row("filename", "1-calm.wav")
        with self.assertRaisesRegex(ValueError, "duplicate filename"):
            self.read(self.train_path)

    def test_rejects_unknown_emotion(self):
        self.change_first_row("emotion", "unknown")
        with self.assertRaisesRegex(ValueError, "unknown emotion"):
            self.read(self.train_path)

    def test_rejects_invalid_actor(self):
        for value in ("0", "25", "one"):
            with self.subTest(value=value):
                self.change_first_row("actor", value)
                with self.assertRaisesRegex(ValueError, "actor"):
                    self.read(self.train_path)

    def test_rejects_missing_metadata(self):
        self.train_path.write_text("emotion,filename,feature\nangry,a.wav,1\n")
        with self.assertRaisesRegex(ValueError, "missing metadata"):
            self.read(self.train_path)

    def test_rejects_duplicate_headers(self):
        self.train_path.write_text("emotion,actor,filename,feature,feature\nangry,1,a.wav,1,2\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            self.read(self.train_path)

    def test_rejects_ragged_csv(self):
        for row in ("angry,1,a.wav\n", "angry,1,a.wav,1,extra\n"):
            with self.subTest(row=row):
                self.train_path.write_text("emotion,actor,filename,feature\n" + row)
                with self.assertRaisesRegex(ValueError, "number of CSV fields"):
                    self.read(self.train_path)

    def test_rejects_empty_dataset(self):
        self.train_path.write_text("emotion,actor,filename,feature\n")
        with self.assertRaisesRegex(ValueError, "no data rows"):
            self.read(self.train_path)

    def test_rejects_feature_order_mismatch(self):
        write_table(self.test_path, [19], features=("feature_a", "feature_b"))
        with self.assertRaisesRegex(ValueError, "order"):
            validate_split(self.read(self.train_path), self.read(self.test_path), self.config)

    def test_rejects_actor_leakage(self):
        train, test = self.read(self.train_path), self.read(self.test_path)
        test.actors[0] = 1
        with self.assertRaisesRegex(ValueError, "share actors"):
            validate_split(train, test, self.config)

    def test_rejects_filename_leakage(self):
        train, test = self.read(self.train_path), self.read(self.test_path)
        test.filenames[0] = train.filenames[0]
        with self.assertRaisesRegex(ValueError, "share filenames"):
            validate_split(train, test, self.config)

    def test_rejects_unexpected_partition(self):
        self.config["test_actors"] = [20]
        with self.assertRaisesRegex(ValueError, "configured partition"):
            validate_split(self.read(self.train_path), self.read(self.test_path), self.config)

    def test_rejects_missing_class(self):
        train, test = self.read(self.train_path), self.read(self.test_path)
        test.labels = [label for label in test.labels if label != "angry"]
        with self.assertRaisesRegex(ValueError, "eight emotion classes"):
            validate_split(train, test, self.config)

    def test_cleaned_export_recovers_actor_and_excludes_index_and_flag(self):
        self.train_path.write_text(
            ',emotion,filename,feature,outlier_detection\n'
            '0,angry,03-01-05-01-01-01-02.wav,3.5,inlier\n'
        )
        before = self.train_path.read_bytes()
        table = read_features(self.train_path, METADATA, "emotion", cleaned=True)
        self.assertEqual(table.actors, [2])
        self.assertEqual(table.feature_names, ("feature",))
        self.assertEqual(table.values, [[3.5]])
        self.assertEqual(self.train_path.read_bytes(), before)

    def test_cleaned_export_rejects_invalid_filename(self):
        self.train_path.write_text('emotion,filename,feature\nangry,other.wav,1\n')
        with self.assertRaisesRegex(ValueError, "invalid actor"):
            read_features(self.train_path, METADATA, "emotion", cleaned=True)

    def test_cleaned_export_rejects_actor_disagreement(self):
        self.train_path.write_text(
            'emotion,actor,filename,feature\nangry,1,03-01-05-01-01-01-02.wav,1\n'
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            read_features(self.train_path, METADATA, "emotion", cleaned=True)

    def test_cleaned_export_rejects_index_in_wrong_position(self):
        self.train_path.write_text(
            'emotion,,filename,feature\nangry,0,03-01-05-01-01-01-02.wav,1\n'
        )
        with self.assertRaisesRegex(ValueError, "headers"):
            read_features(self.train_path, METADATA, "emotion", cleaned=True)

    def test_unknown_dataset_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset.variant"):
            load_dataset(self.root, {"dataset": {"variant": "unknown"}})

    def test_binary_task_preserves_mapping_alignment_and_original_data(self):
        table = self.read(self.train_path)
        original_labels = list(table.labels)
        binary = positive_negative_task(table)
        keep = [i for i, label in enumerate(table.labels) if label not in {"neutral", "surprised"}]
        self.assertEqual(binary.values, [table.values[i] for i in keep])
        self.assertEqual(binary.filenames, [table.filenames[i] for i in keep])
        self.assertEqual(binary.actors, [table.actors[i] for i in keep])
        self.assertEqual(binary.source_labels, [table.labels[i] for i in keep])
        self.assertEqual(binary.labels, [int(table.labels[i] in {"happy", "calm"}) for i in keep])
        self.assertEqual(binary.source_row_count, 16)
        self.assertEqual(binary.target_name, "positive")
        self.assertEqual(binary.digest, table.digest)
        self.assertEqual(table.labels, original_labels)

    def test_binary_task_rejects_invalid_labels_or_missing_class(self):
        table = self.read(self.train_path)
        table.labels[0] = "unknown"
        with self.assertRaisesRegex(ValueError, "original emotion labels"):
            positive_negative_task(table)
        table.labels = ["happy"] * len(table.labels)
        with self.assertRaisesRegex(ValueError, "both positive and negative"):
            positive_negative_task(table)


if __name__ == "__main__":
    unittest.main()
