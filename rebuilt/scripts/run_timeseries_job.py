"""Run a long comparison, verify it, and export results using frozen project code.

Feature checkpoints survive interruption. Existing completed runs are never
rewritten. A job status file and logs live alongside the feature cache.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()
    project, data, config, output, exports = [path.resolve() for path in (
        args.project_root, args.data_dir, args.config, args.run_dir, args.export_dir)]
    if output.exists():
        raise ValueError("Completed/partial output already exists; choose a fresh run directory.")
    workspace = output.with_name(output.name + "-features")
    snapshot = workspace / "execution"
    source = snapshot / "src/ravdess_project"
    source.mkdir(parents=True, exist_ok=True)
    for path in sorted((project / "rebuilt/src/ravdess_project").glob("*.py")):
        destination = source / path.name
        if destination.exists() and digest(destination) != digest(path):
            raise ValueError("Frozen source differs from current source; use a new run directory.")
        if not destination.exists():
            shutil.copyfile(path, destination)
    verifier = snapshot / "scripts/verify_timeseries_run.py"
    verifier.parent.mkdir(exist_ok=True)
    if not verifier.exists():
        shutil.copyfile(project / "rebuilt/scripts/verify_timeseries_run.py", verifier)
    frozen_config = snapshot / "configs" / config.name
    frozen_config.parent.mkdir(exist_ok=True)
    if frozen_config.exists() and digest(frozen_config) != digest(config):
        raise ValueError("Frozen config differs from requested config.")
    if not frozen_config.exists():
        shutil.copyfile(config, frozen_config)
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(snapshot / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       NUMBA_CACHE_DIR="/tmp/ravdess-numba-cache")
    status_path = workspace / "job_status.json"
    status = {"state": "running", "stage": "comparison", "run": str(output),
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {path.name: digest(path) for path in sorted(source.glob("*.py"))}}
    def save_status():
        temporary = workspace / "job_status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)
    save_status()
    command = [sys.executable, "-m", "ravdess_project", "series", "--data-dir", str(data),
               "--config", str(frozen_config), "--output", str(output)]
    try:
        with (workspace / "comparison.log").open("a") as stream:
            subprocess.run(command, env=environment, cwd=project, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        status["stage"] = "verification"
        save_status()
        with (workspace / "verification.log").open("w") as stream:
            verified = subprocess.run([sys.executable, str(verifier), "--data-dir", str(data),
                                       "--run-dir", str(output)], env=environment, cwd=project,
                                      stdout=subprocess.PIPE, stderr=stream, text=True, check=True)
        validation = json.loads(verified.stdout)
        required = ("actor_isolation_verified", "current_source_matches")
        if not all(validation[key] for key in required):
            raise ValueError("Automatic artifact verification did not pass all required checks.")
        if validation["selected_method"] != "majority" and not validation["notebook_style_scaling_exact"]:
            raise ValueError("Saved waveform preprocessing did not match notebook-style scaling.")
        report = json.loads((output / "report.json").read_text())
        if digest(frozen_config) != report["config_sha256"]:
            raise ValueError("Config hash differs from the completed run.")
        for table in report["inputs"].values():
            if digest(Path(table["path"])) != table["sha256"] or digest(Path(table["metadata_path"])) != table["metadata_sha256"]:
                raise ValueError("Inputs differ from the completed run.")
        for fold in report["validation"]["folds"]:
            fold["train_rows"] = len(fold.pop("train_indices"))
            fold["validation_rows"] = len(fold.pop("validation_indices"))
        for table in report["inputs"].values():
            table["path"] = Path(table["path"]).name
            table["metadata_path"] = Path(table["metadata_path"]).name
        for details in report["feature_analysis"].values():
            details["active_feature_count"] = len(details.pop("active_features"))
        exports.mkdir(parents=True, exist_ok=True)
        task = "emotion" if report["config"]["target"] == "emotion" else "vocal"
        prefix = f"timeseries_{task}_full"
        for path in (exports / f"{prefix}_results.json", exports / f"{prefix}_validation.json"):
            if path.exists():
                raise ValueError("Export already exists; no results will be overwritten.")
        (output / "validation.json").write_text(verified.stdout)
        (exports / f"{prefix}_results.json").write_text(json.dumps(report, indent=2) + "\n")
        (exports / f"{prefix}_validation.json").write_text(verified.stdout)
        selected = report["validation"]["selected_method"]
        summary = {"selected_method": selected, "test_accuracy": report["metrics"][selected]["accuracy"],
                   "test_f1_macro": report["metrics"][selected]["f1_macro"],
                   "predictions_verified": validation["predictions_verified"]}
        table = ["| Method | Mean actor-CV macro F1 |", "| --- | ---: |"]
        table.extend(f"| {name} | {score['mean_f1_macro']:.6f} |" for name, score in report["validation"]["scores"].items())
        text = (f"# {report['config']['target']} waveform comparison\n\n"
                f"All configured methods completed their actor folds. Validation selected **{selected}**.\n\n"
                f"Test accuracy: **{100 * summary['test_accuracy']:.2f}%**; macro F1: **{summary['test_f1_macro']:.6f}**.\n\n"
                + "\n".join(table) + "\n\n"
                + f"All {summary['predictions_verified']} saved test predictions and metrics were reproduced after reload. "
                + "Input/config/source hashes and actor isolation passed verification. "
                + "These are new seeded actor-validation experiments, not claims of historical equivalence.\n\n"
                + f"[Results]({prefix}_results.json), [verification]({prefix}_validation.json).\n")
        analysis_path = exports / f"{prefix}_analysis.md"
        if analysis_path.exists():
            raise ValueError("Analysis export already exists.")
        analysis_path.write_text(text)
        status.update(state="complete", stage="complete", results=summary,
                      completed_at_utc=datetime.now(timezone.utc).isoformat())
        shutil.copyfile(workspace / "comparison.log", output / "runtime.log")
        (output / "execution.json").write_text(json.dumps({"command": command,
            "source_snapshot": str(snapshot), "environment_variables": {name: environment[name] for name in (
                "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_CACHE_DIR")}}, indent=2) + "\n")
        save_status()
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}")
        save_status()
        raise


if __name__ == "__main__":
    main()
