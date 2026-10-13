# Publishing this portfolio

This checkout contains the rebuilt ML package, explicit experiment settings,
fixture tests, and small documentation/results reports. Datasets, original
notebooks, third-party academic libraries, model binaries, local environments,
run directories and archives remain in the preserved academic workspace and
are excluded from this Git repository. The CI workflow also rejects tracked
files larger than 5 MB.

Original notebook paths in historical reports identify the source academic
artifacts; they are not files bundled with this checkout. The legacy manifest
and its verification script describe the full academic workspace. Do not run
that preservation check against this smaller publication checkout.

Machine-specific absolute input paths in publication copies of reports are
shortened to filenames. Metrics, experimental settings and recorded data/model
hashes are retained. The original reports are untouched; the consolidated
summary is regenerated against these publication copies.

## Check locally

After installing the core package:

```bash
python -m unittest discover -s rebuilt/tests -v
python rebuilt/scripts/summarize_results.py --check
python -m ravdess_project --help
```

GitHub Actions performs these core checks on Python 3.14. Optional neural,
XGBoost and time-series tests require their separate dependencies and are skipped
in the core-only CI environment. CI uses fixtures, never full datasets or long
experimental searches. The workflow has not yet run on GitHub.

## Send to GitHub

Create an empty GitHub repository, then run from this checkout:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

No remote is configured and nothing has been uploaded. `.gitignore` excludes
data, model weights, environments and archives. Keep those outside Git; small
JSON result reports belong in `rebuilt/docs/`. No repository license is assigned
here; choose one that you have the rights to apply before distributing under it.

Comparable long experiments remain cancelled unless explicitly requested.
