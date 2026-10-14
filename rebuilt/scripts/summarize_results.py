"""Summarize saved reports without loading models, reading datasets, or training."""

import argparse
import hashlib
import json
from pathlib import Path


# Labels explain the experiment; all numeric results come from saved reports.
EXPERIMENTS = (
    ("baseline_results.json", "Raw-feature random forest", "emotion", "425 numeric", "random_forest", None),
    ("tabular_results.json", "Cleaned-feature random forest", "emotion", "197 numeric", None, None),
    ("gradient_boosting_results.json", "Gradient boosting", "emotion", "197 numeric", None, None),
    ("xgboost_results.json", "XGBoost", "emotion", "197 numeric", None, None),
    ("neural_results.json", "Dense network: dropout variant", "emotion", "202 numeric/categorical", None, "neural_artifact_validation.json"),
    ("neural_search_results.json", "Dense network: parameter search", "emotion", "202 numeric/categorical", None, "neural_search_artifact_validation.json"),
    ("timeseries_emotion_results.json", "Waveform Euclidean 1-NN", "emotion", "raw [45000:100000]", None, "timeseries_emotion_validation.json"),
    ("svm_results.json", "Binary linear SVM", "positive_negative", "197 numeric", None, None),
    ("timeseries_vocal_results.json", "Euclidean 1-NN", "vocal_channel", "compressed [5000:13000]", None, "timeseries_vocal_validation.json"),
    ("timeseries_vocal_full_results.json", "MiniROCKET", "vocal_channel", "compressed [5000:13000]", None, "timeseries_vocal_full_validation.json"),
    ("regression_results.json", "Lasso: alpha=0.1", "kur_regression", "202 numeric/categorical", None, None),
)


def summarize(docs):
    sources = {}
    def read(name):
        path = docs / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    rows = []
    for filename, label, task, inputs, method, evidence in EXPERIMENTS:
        report = read(filename)
        selected = method or report["validation"]["selected_method"]
        metrics = report["metrics"][selected]
        fit_rows = report["inputs"]["train"]["rows"]
        if filename == "neural_results.json":
            fit_rows = report["validation"]["folds"][0]["train_rows"]
        if evidence is not None and not (docs / evidence).is_file():
            raise ValueError(f"Missing validation evidence: {evidence}")
        rows.append({"label": label, "task": task, "inputs": inputs,
                     "training_table_rows": report["inputs"]["train"]["rows"],
                     "final_fit_rows": fit_rows, "test_rows": report["inputs"]["test"]["rows"],
                     "method": selected,
                     "selection_metric": report.get("validation", {}).get("selection_metric", "Fixed settings; no selection"),
                     "metrics": {key: metrics[key] for key in
                                 ("accuracy", "f1_macro", "roc_auc", "mse", "rmse", "mae", "r2") if key in metrics},
                     "report": filename, "validation_evidence": evidence})
    historical = read("legacy_timeseries_results.json")
    checkpoints = read("neural_legacy_comparison.json")
    result = {"scope": "Index of saved evidence; no experiments executed to produce this summary.",
              "completed_rebuilt": rows,
              "historical_notebook_outputs": historical,
              "historical_neural_checkpoint_reload": checkpoints,
              "cancelled_incomplete": ["sax-dtw-emotion-001", "sax-dtw-vocal-001", "timeseries-emotion-full-003"],
              "large_run_policy": "Do not launch or resume comparably long experiments without an explicit user request.",
              "source_report_sha256": sources}

    lines = ["# Results and experiment status", "",
             "This page consolidates existing reports. No models were trained to generate it. "
             "Original notebook outputs, completed rebuilt experiments, and incomplete runs remain distinct.", "",
             "## Completed rebuilt experiments", "",
             "All classification scores below are fixed-test accuracy and macro F1, not validation scores. "
             "Actors 01–18 supply training data; actors 19–24 supply test data. "
             "Candidate selection uses training actors only, where applicable. "
             "The test set was already examined in the academic project; these are not fresh independent benchmarks. "
             "Standalone artifact-verification reports are linked in the machine-readable index where available; "
             "a null verification reference means no standalone verification report is included. "
             "The recorded checks were performed when the runs completed, not rerun for this documentation update. "
             "[Project guide and methodology](../../README.md).", ""]
    for task, heading in (("emotion", "Eight-class emotion classification"),
                          ("positive_negative", "Binary emotion task"),
                          ("vocal_channel", "Vocal channel: speech versus song")):
        lines += [f"### {heading}", "",
                  "| Experiment | Inputs | Final fit / test rows | Test accuracy | Macro F1 | Report |",
                  "| --- | --- | ---: | ---: | ---: | --- |"]
        for row in rows:
            if row["task"] != task:
                continue
            m = row["metrics"]
            lines.append(f"| {row['label']} | {row['inputs']} | {row['final_fit_rows']} / {row['test_rows']} "
                         f"| {100*m['accuracy']:.2f}% | {m['f1_macro']:.4f} | [Saved report]({row['report']}) |")
        lines.append("")
        if task == "emotion":
            lines += ["The dropout variant uses fourteen fitting actors (1,395 rows) and four validation actors "
                      "(414 rows). The searched network refits on all eighteen training actors (1,809 rows). "
                      "Different features and fitting protocols prevent treating this table as a controlled leaderboard. "
                      "The raw-feature majority reference scores 15.38% accuracy and 0.0333 macro F1.", ""]
        elif task == "positive_negative":
            auc = next(row["metrics"]["roc_auc"] for row in rows if row["task"] == task)
            lines += [f"Calm/happy map to positive; angry/disgust/fearful/sad map to negative. "
                      f"Neutral and surprised are excluded. ROC AUC: {auc:.4f}. "
                      "The original SVM test file is missing, so the rebuild explicitly uses the validated cleaned pair. "
                      "Its accuracy is not comparable to eight-class emotion accuracy.", ""]
        else:
            lines += ["Speech/song classification is a separate two-class problem; "
                      "98.56% here does not mean 98.56% emotion accuracy.", ""]
    row = next(row for row in rows if row["task"] == "kur_regression")
    m = row["metrics"]
    lines += ["### Regression: predicting the supplied `kur` feature", "",
              "| Selected model | Final fit / test rows | MSE | RMSE | MAE | R² |",
              "| --- | ---: | ---: | ---: | ---: | ---: |",
              f"| {row['label']} | {row['final_fit_rows']} / {row['test_rows']} "
              f"| {m['mse']:.5f} | {m['rmse']:.5f} | {m['mae']:.5f} | {m['r2']:.5f} |", "",
              "Regression includes emotion among the predictors and is not an emotion classifier. "
              "[Saved report](regression_results.json), [fixed-setting historical diagnostics](regression_comparison.json).", "",
              "## Original notebook results", "",
              "The following accuracies are copied from saved root-notebook outputs, at their displayed precision. "
              "They are historical observations, not independently reproduced full runs.", "",
              "| Method | Emotion accuracy | Vocal-channel accuracy |",
              "| --- | ---: | ---: |"]
    primary = [r for r in historical["results"] if "/" not in r["notebook"] and r["notebook"] != "dtw copy.ipynb"]
    for method in dict.fromkeys(r["method"] for r in primary):
        values = {r["task"]: r["accuracy_displayed"] for r in primary if r["method"] == method}
        lines.append(f"| {method} | {100*float(values['emotion']):.0f}% | {100*float(values['vocal_channel']):.0f}% |")
    lines += ["", "Copied vocal notebooks report ROCKET 93% and MiniROCKET 98%, versus 92% and 99% "
              "in the root notebook. Both versions are preserved. "
              "[Original reports, macro F1 and notebook/cell references](legacy_timeseries_results.json).", "",
              "Read-only reloads of the four original neural checkpoints also reproduced their saved accuracies "
              "to displayed precision. These checks validate existing artifacts, not retraining:", "",
              "| Original checkpoint | Reloaded test accuracy |",
              "| --- | ---: |"]
    for name, metrics in checkpoints["models"].items():
        lines.append(f"| `{name}` | {100*metrics['accuracy']:.4f}% |")
    lines += ["", "[Checkpoint evidence](neural_legacy_comparison.json), "
              "[neural methodology](../../README.md#methodology-and-reproducibility).", "",
              "## Cancelled and incomplete runs", "",
              "| Run | Status | Retained evidence |",
              "| --- | --- | --- |",
              "| Full emotion ROCKET/MiniROCKET comparison | Cancelled | Frozen sources and partial ROCKET feature cache |",
              "| Emotion SAX/DTW | Cancelled | Verified SAX preprocessing and partial predictions |",
              "| Vocal-channel SAX/DTW | Cancelled | Verified SAX preprocessing and partial predictions |", "",
              "Partial predictions and feature caches remain in the separate academic workspace, outside this public checkout, "
              "and are not complete scores. Completed vocal MiniROCKET results "
              "remain preserved. Do not launch or resume comparable long runs without an explicit request.", "",
              "## Reproducibility limits and remaining work", "",
              "- Raw WAV import reproduced both original NPY hashes across all 2,452 recordings. "
              "The compression candidate differs by up to 2.95e-7; analysis retains the original compressed arrays.",
              "- Current emotion Euclidean accuracy is 12.82%, versus the saved rounded 17%. "
              "Independent original-backend checks match the rebuilt predictions; the historical discrepancy is unresolved.",
              "- Fixed-setting Lasso, Ridge and gradient-boosting regression diagnostics match saved notebook MSE "
              "to displayed precision. The original enormous linear-regression error is not reproduced; its cause is unresolved.",
              "- Feature extraction and the full cleaning/outlier-removal code are unavailable. "
              "External dataset links do not make a standalone public data distribution.",
              "- Added actor validation, missing original seeds and changed library runtimes are explicit rebuild choices. "
              "Numerical proximity does not establish equivalence or statistical improvement.",
              "- Remaining ensembles, exploratory embeddings, shapelets, clustering, motifs and explanations "
              "are documented academic analyses whose full rebuild remains incomplete.", "",
              "See the [project guide](../../README.md) and [machine-readable index](results_summary.json).", ""]
    return json.dumps(result, indent=2, ensure_ascii=False) + "\n", "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs")
    parser.add_argument("--check", action="store_true", help="Check summaries against reports without writing")
    args = parser.parse_args()
    outputs = dict(zip(("results_summary.json", "results.md"), summarize(args.docs_dir)))
    for name, content in outputs.items():
        path = args.docs_dir / name
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Summary differs from saved reports: {name}")
        else:
            path.write_text(content)
    print("Result summaries match saved reports." if args.check else "Wrote result summaries from saved reports; no experiments executed.")


if __name__ == "__main__":
    main()
