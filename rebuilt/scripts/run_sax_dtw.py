"""Prepare original SAX symbols or resume all three notebook DTW evaluations."""

import argparse
import json
import tomllib
from pathlib import Path

from ravdess_project.sax_dtw import evaluate, prepare
from ravdess_project.time_series import load_series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    train, test = load_series(args.data_dir, config)
    if not args.output.exists():
        prepare(train, test, config, args.config, args.output)
    elif args.prepare_only:
        raise ValueError("Prepared output already exists; use a fresh directory.")
    if not args.prepare_only:
        result = evaluate(train, test, config, args.config, args.output, args.batch_size, progress=print)
        print(json.dumps(result["metrics"], indent=2))
    else:
        print("Saved SAX symbols; all train/test symbols exactly match notebook preprocessing.")


if __name__ == "__main__":
    main()
