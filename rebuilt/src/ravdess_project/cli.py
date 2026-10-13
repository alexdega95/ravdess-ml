"""Commands for data auditing, comparisons, and explicit parameter searches."""

import argparse
import json
import sys
import tomllib
from pathlib import Path

from .data import load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "train", "compare", "search", "regress", "neural", "series", "series-audit"))
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing the configured feature tables")
    parser.add_argument("--config", type=Path, required=True, help="Experiment TOML file")
    parser.add_argument("--output", type=Path, help="New directory for experiment artifacts; must not already exist")
    args = parser.parse_args(argv)
    if args.command not in {"audit", "series-audit"} and args.output is None:
        parser.error(f"{args.command} requires --output")
    if args.command in {"audit", "series-audit"} and args.output is not None:
        parser.error("audit prints to stdout; --output is only used with experiment commands")
    try:
        config = tomllib.loads(args.config.read_text())
        if args.command in {"series", "series-audit"}:
            from .time_series import load_series

            train, test = load_series(args.data_dir, config)
        else:
            train, test = load_dataset(args.data_dir, config)
        if args.command in {"audit", "series-audit"}:
            result = {"train": train.summary(), "test": test.summary(), "actor_overlap": []}
        else:
            from .experiment import run_baselines

            if (args.command == "regress") != (train.task_kind == "regression"):
                raise ValueError("Use regress with a kur_regression config; classifiers require classification targets.")
            if train.categorical_features and args.command not in {"regress", "neural"}:
                raise ValueError("Categorical predictor configs require regress or neural.")
            if args.command == "series":
                from .time_series import run_series

                report = run_series(
                    train, test, config, args.config, args.output,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            elif args.command == "regress":
                from .regression import run_regression

                report = run_regression(
                    train, test, config, args.config, args.output,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            elif args.command == "neural":
                from .neural import run_neural

                report = run_neural(
                    train, test, config, args.config, args.output,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            elif args.command == "search":
                from .model_search import run_search

                report = run_search(
                    train, test, config, args.config, args.output,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            elif args.command == "compare":
                from .tabular import run_comparison

                report = run_comparison(
                    train, test, config, args.config, args.output,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            else:
                report = run_baselines(train, test, config, args.config, args.output)
            result = {
                "output": str(args.output.resolve()),
                "metrics": {
                    name: {metric: values[metric] for metric in (("mse", "rmse", "mae", "r2") if train.task_kind == "regression" else ("accuracy", "f1_macro"))}
                    for name, values in report["metrics"].items()
                },
            }
            if "validation" in report:
                result["selected_method"] = report["validation"]["selected_method"]
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"Missing training dependency: {exc}. Install the project dependencies.", file=sys.stderr)
        return 1
    return 0
