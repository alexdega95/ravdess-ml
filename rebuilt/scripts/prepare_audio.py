"""Rebuild original waveform arrays and verify them against the supplied artifacts."""

import argparse
import json
from pathlib import Path

from ravdess_project.audio_preparation import build_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    args = parser.parse_args()
    report = build_audio(args.audio_dir, args.output, args.reference_dir, progress=print)
    print(json.dumps(report["partitions"], indent=2))


if __name__ == "__main__":
    main()
