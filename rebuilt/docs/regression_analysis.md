# Regression: predicting `kur`

The original regression predicts the supplied `kur` feature, not emotion. The
rebuilt experiment retains all 1,809 training and 624 test rows, six categorical
predictors (including emotion), and 196 numeric predictors. `kur` and recording
identifiers are excluded from the inputs. The meaning and generation of `kur`
remain dependent on the supplied academic feature tables; they are not inferred.

Each pipeline fits sorted categorical encodings and StandardScaler on fitting
rows only. Independent notebook-style LabelEncoder/scaler calculations matched
both transformed train and test matrices exactly with the current libraries.
The check can be repeated with `scripts/compare_regression.py --data-dir ../data`.

## Selection and results

All five original families are represented: linear regression, Lasso, Ridge,
random forest, and gradient boosting. The seven original alpha values and the
30-trial forest search are retained; forest split/leaf options are unchanged.
Gradient boosting retains 100 stages and random_state=0. Lasso retains its default
1,000-iteration limit. The added mean predictor supplies a reference.

There are 47 candidates and five actor-disjoint folds: 235 validation fits.
Selection minimizes mean training-validation MSE, with configuration order
breaking ties. The selected Lasso uses alpha=0.1; its mean validation MSE is
8.95109. Only the selected pipeline and mean reference are subsequently evaluated
on the fixed test actors 19–24.

| Test model | MSE | RMSE | MAE | R² |
| --- | ---: | ---: | ---: | ---: |
| Mean reference | 39.14882 | 6.25690 | 4.68509 | -0.00379 |
| Selected Lasso, alpha=0.1 | 9.68607 | 3.11225 | 1.78256 | 0.75165 |

Ten Lasso convergence warnings were captured from other alpha candidates;
none came from the selected pipeline. Warnings and every candidate's fold
scores appear in the full local report and [compact results](regression_results.json).
Saved pipelines were reloaded and their predictions compared with the stored CSV.

## Comparison with the notebook

A separate diagnostic used the original saved Lasso/Ridge settings and fixed
boosting settings, fitted on all training rows. It does not influence selection.

| Fixed setting | Notebook saved MSE | Current-library diagnostic MSE |
| --- | ---: | ---: |
| Lasso, alpha=0.01 | 9.045267 | 9.0452670296 |
| Ridge, alpha=0.99 | 9.534614 | 9.5346138005 |
| Gradient boosting | 9.399994 | 9.3999943312 |
| Linear regression | approximately 2.48 × 10¹⁹ | 9.6825139587 |

Three saved results match to their displayed precision. The linear-regression
result differs radically. The original runtime was not recovered; solver/version
behavior and ill-conditioned predictors need further investigation before
attributing a cause. No features were removed or solvers changed to force a match.
The forest cell has no saved score, so no historical forest comparison is claimed.
[Diagnostic details](regression_comparison.json) retain the measured values.

Actor-based folds replace the notebook's sample-based folds. Random seed 42 is
added for previously unseeded forest fitting and random parameter sampling.
Consequently the selected alpha differs from the original alpha=0.01. These are
explicit evaluation/reproducibility changes in the rebuilt analysis, not evidence
that the new experiment improves the historical project.
