"""Recover the observed PCM audio import without changing legacy artifacts."""

import csv
import json
import math
import wave
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from .data import sha256

FIELDS = ("modality", "vocal_channel", "emotion", "emotional_intensity", "statement", "repetition", "actor")
MAPPINGS = (
    {"01": "full-AV", "02": "video-only", "03": "audio-only"},
    {"01": "speech", "02": "song"},
    dict(zip((f"{i:02d}" for i in range(1, 9)),
             ("neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"))),
    {"01": "normal", "02": "strong"},
    {"01": "Kids are talking by the door", "02": "Dogs are sitting by the door"},
    {"01": "1st", "02": "2nd"},
    {f"{i:02d}": f"{i:02d}" for i in range(1, 25)},
)


def decode_filename(filename):
    path = Path(filename)
    codes = path.stem.split("-")
    if path.name != filename or path.suffix != ".wav" or len(codes) != 7:
        raise ValueError(f"Invalid RAVDESS filename: {filename}")
    try:
        result = {field: mapping[code] for field, mapping, code in zip(FIELDS, MAPPINGS, codes)}
    except KeyError as exc:
        raise ValueError(f"Unknown RAVDESS filename code: {filename}") from exc
    if result["modality"] != "audio-only":
        raise ValueError("This workflow requires audio-only recordings.")
    result.update(sex="F" if int(result["actor"]) % 2 == 0 else "M", filename=filename)
    return result


def inspect_pcm(path):
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() not in {1, 2} or (stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) != (2, 48000, "NONE"):
            raise ValueError(f"{path}: expected observed mono/stereo 16-bit PCM at 48 kHz; no implicit conversion.")
        return stream.getnframes()


def load_pcm(path):
    length = inspect_pcm(path)
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        content = stream.readframes(length)
    if len(content) != length * channels * 2:
        raise ValueError(f"{path}: truncated PCM payload.")
    # These fractions are exactly representable in float32, as in librosa.load(sr=None).
    values = np.frombuffer(content, dtype="<i2").astype(np.float32) / np.float32(32768)
    return values.reshape(-1, channels).mean(axis=1, dtype=np.float32)


def build_audio(audio_dir, output, reference_dir=None, progress=None):
    """Actor/filename order, train-maximum NaN padding, and validated q=10 candidate.

    Compression's implementation is absent from the notebooks. Validation against
    supplied arrays is therefore recorded, not assumed. Analysis uses supplied
    compressed arrays unless an explicitly chosen rebuilt directory is provided.
    """
    if output.exists():
        raise ValueError(f"Output already exists: {output}.")
    paths = sorted(audio_dir.glob("*.wav"), key=lambda path: (int(decode_filename(path.name)["actor"]), path.name))
    if not paths:
        raise ValueError("No WAV recordings found.")
    metadata = [decode_filename(path.name) for path in paths]
    lengths = [inspect_pcm(path) for path in paths]
    train_indices = [i for i, row in enumerate(metadata) if int(row["actor"]) < 19]
    test_indices = [i for i, row in enumerate(metadata) if int(row["actor"]) >= 19]
    if {int(metadata[i]["actor"]) for i in train_indices} != set(range(1, 19)) or {int(metadata[i]["actor"]) for i in test_indices} != set(range(19, 25)):
        raise ValueError("Audio files do not contain the original complete actor partition.")
    maximum = max(lengths[i] for i in train_indices)
    report = {"sample_rate": 48000, "pcm": "mono/stereo signed 16-bit; float32 channel mean", "raw_dtype": "float64",
              "padding": "trailing NaN", "pad_length_from_training": maximum,
              "compression_candidate": {"function": "scipy.signal.decimate", "q": 10,
                  "input_dtype": "float32", "ftype": "iir", "n": 8, "zero_phase": True,
                  "order": "decimate each finite recording before padding"},
              "recordings": [], "partitions": {}}
    output.mkdir(parents=True, exist_ok=False)
    for split, indices in (("train", train_indices), ("test", test_indices)):
        filename = f"RavdessAudioOnlyNumpy__X_{split}"
        raw = np.lib.format.open_memmap(output / f"{filename}.npy", mode="w+", dtype="float64", shape=(len(indices), maximum))
        compressed = np.lib.format.open_memmap(output / f"{filename}_compresso.npy", mode="w+", dtype="float64", shape=(len(indices), math.ceil(maximum / 10)))
        raw[:] = np.nan
        compressed[:] = np.nan
        reference, reference_compressed = None, None
        part = {"rows": len(indices), "actors": sorted({int(metadata[i]["actor"]) for i in indices}),
                "clipped_recordings": 0, "raw_matches_reference": None,
                "compression_max_absolute_difference": None, "compression_float32_matches": None}
        if reference_dir is not None:
            with (reference_dir / f"RavdessAudioOnlyNumpy__Y_{split}.csv").open(newline="") as stream:
                supplied_metadata = list(csv.DictReader(stream))
            if supplied_metadata != [metadata[i] for i in indices]:
                raise ValueError(f"{split}: reconstructed metadata/order differs from reference.")
            reference = np.load(reference_dir / f"{filename}.npy", mmap_mode="r", allow_pickle=False)
            reference_compressed = np.load(reference_dir / f"{filename}_compresso.npy", mmap_mode="r", allow_pickle=False)
            if reference.shape != raw.shape or reference_compressed.shape != compressed.shape:
                raise ValueError(f"{split}: reconstructed shape differs from reference.")
            part.update(raw_matches_reference=True, compression_max_absolute_difference=0.0,
                        compression_float32_matches=True)
        for row, index in enumerate(indices):
            path = paths[index]
            before = sha256(path)
            signal = load_pcm(path)
            if sha256(path) != before:
                raise ValueError(f"{path}: input changed during decoding.")
            raw[row, :min(len(signal), maximum)] = signal[:maximum]
            reduced = decimate(signal, 10, n=8, ftype="iir", zero_phase=True)
            compressed[row, :min(len(reduced), compressed.shape[1])] = reduced[:compressed.shape[1]]
            part["clipped_recordings"] += int(len(signal) > maximum)
            report["recordings"].append({"filename": path.name, "actor": int(metadata[index]["actor"]),
                                         "samples": len(signal), "sha256": before})
            if reference is not None:
                if not np.array_equal(raw[row], reference[row], equal_nan=True):
                    part["raw_matches_reference"] = False
                mask = np.isfinite(reference_compressed[row])
                if not np.array_equal(np.isfinite(compressed[row]), mask):
                    raise ValueError(f"{path}: compressed padding positions differ from reference.")
                difference = float(np.max(np.abs(compressed[row, mask] - reference_compressed[row, mask])))
                part["compression_max_absolute_difference"] = max(part["compression_max_absolute_difference"], difference)
                part["compression_float32_matches"] &= bool(np.array_equal(compressed[row, mask].astype("float32"), reference_compressed[row, mask].astype("float32")))
            if progress and (row + 1) % 100 == 0:
                progress(f"Decoded and checked {split}: {row + 1}/{len(indices)}")
        raw.flush()
        compressed.flush()
        del raw, compressed
        with (output / f"RavdessAudioOnlyNumpy__Y_{split}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*FIELDS, "sex", "filename"])
            writer.writeheader()
            writer.writerows(metadata[i] for i in indices)
        report["partitions"][split] = part
    report["output_sha256"] = {path.name: sha256(path) for path in sorted(output.glob("Ravdess*"))}
    (output / "preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
