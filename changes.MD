# Presentation Changes: Previous PPT vs Updated PPT

The updated presentation changes the project narrative from an early prototype to the latest full-scale implementation in the repository.

## Major Changes

| Area | Previous PPT | Updated PPT |
|---|---|---|
| Project title | Agent-based heart-disease modeling | Longitudinal disease progression using deep learning and gradient boosting |
| Dataset | Demo MIMIC-IV / MIMIC-III Demo + eICU | Full MIMIC-IV 2.1 cardiac progression cohort |
| Dataset size | Small/demo dataset | 35,256 samples: 24,678 train, 5,289 validation, and 5,289 test |
| Input structure | 30 longitudinal features | Six visits with 59 features per visit |
| Models | LSTM, GRU, and Transformer | BiLSTM-attention, BiGRU, Transformer, HistGB, LightGBM, XGBoost, and ensembles |
| Agent layer | Presented as an implemented component | Removed because it is not implemented in the latest repository |
| Methodology | Generic clinical-data pipeline | Actual preprocessing, sequence construction, temporal feature engineering, calibration, and evaluation workflow |
| Temporal modeling | Mainly direct sequence learning | Adds trends, slopes, variability, velocity, acceleration, range, IQR, and other temporal descriptors |
| Previous result | GRU F1 of approximately 0.736 on a smaller dataset | Replaced with a fair comparison on identical full-data arrays |
| Best reproduced baseline | Previously presented as GRU | Transformer: F1 0.6319 and ROC-AUC 0.9168 |
| Best final model | GRU/Transformer | Calibrated GBM soft-vote ensemble |
| Best final results | Earlier experimental metrics | F1 0.7334, ROC-AUC 0.9462, precision 0.7226, and recall 0.7444 |
| Explainability | Attention and SHAP proposed as future work | Temporal feature-importance results included |
| Deployment | Prototype/future decision-support system | CPU-friendly inference and a 3.3 MB deployable model artifact |
| Visualizations | Old small-dataset tables and figures | Current comparison, feature importance, confusion matrix, and ROC curves |
| References | General web and research references | MIMIC-IV, Transformer, XGBoost, LightGBM, and repository references |

## Slide-by-Slide Changes

### Slide 1 — Title

- Changed the project title to **Longitudinal Temporal Disease Progression: Heart Disease Modeling with Deep Learning & Gradient Boosting**.
- Removed the unsupported **Agent-Based** label.

### Slide 2 — Contents

- Updated the section names and page ranges.
- Changed **Expected Outcome** to **Project Outcomes & Results**.

### Slide 3 — Introduction

- Reframed cardiovascular deterioration as a longitudinal trajectory rather than a single clinical snapshot.
- Added the binary target: worsening versus stable/improving progression.
- Added the current input representation of six visits and 59 features.
- Added the two implemented modeling tracks:
  - Sequence models
  - Engineered temporal features with gradient boosting
- Removed the legacy agent-based illustrations.

### Slide 4 — Problem Statement

- Added the 16.9% positive-class imbalance.
- Explained the baseline precision and false-positive problem.
- Added calibration, interpretability, and CPU deployment requirements.
- Defined the updated research question: whether explicit temporal descriptors can outperform heavier sequence models.

### Slide 5 — Literature Survey

- Retained the existing first literature-survey table.

### Slide 6 — Literature Survey

- Replaced the unrelated fluid-dynamics entry with the MIMIC-IV dataset publication.
- Added MIMIC-IV advantages and limitations relevant to this project.

### Slide 7 — Methodology

- Rebuilt the methodology using the actual repository workflow:
  1. Select the MIMIC-IV 2.1 cardiac cohort.
  2. Impute, scale, sanitize, and split the data.
  3. Construct six-visit sequences containing 59 features.
  4. Train sequence models and temporal GBM models.
  5. Calibrate probabilities and tune the threshold using validation data.
- Added the final output and evaluation metrics.

### Slide 8 — Project Outcomes

- Changed the slide from expected outcomes to completed outcomes.
- Added:
  - 35,256 total samples
  - Best F1 of 0.7334
  - Best ROC-AUC of 0.9462
  - 3.3 MB deployable model artifact
- Added reproducibility, CPU inference, calibration, and interpretability outcomes.

### Slide 9 — Dataset and Results Overview

- Replaced the demo-data description with the full MIMIC-IV 2.1 cohort:
  - Train: 24,678 samples, including 4,178 worsening cases
  - Validation: 5,289 samples, including 896 worsening cases
  - Test: 5,289 samples, including 896 worsening cases
  - Positive-class rate: approximately 16.9%
- Added the baseline-versus-improved model comparison chart.

### Slide 10 — Temporal Feature Engineering

- Added engineered temporal descriptors including:
  - First, last, mean, and median
  - Minimum, maximum, range, and IQR
  - Delta, slope, and half-split trend
  - Velocity and acceleration
  - Variability and total variation
- Added the HistGB, LightGBM, and XGBoost ensemble workflow.
- Added the temporal feature-importance visualization.

### Slide 11 — Best Model

- Replaced the previous small-dataset result with the calibrated GBM soft-vote ensemble.
- Added the validation-tuned threshold of 0.335.
- Added the held-out test metrics:
  - F1: 0.7334
  - ROC-AUC: 0.9462
  - Precision: 0.7226
  - Recall: 0.7444
- Added the current confusion matrix for the 5,289-sample test set.

### Slide 12 — Key Findings

- Added the fairly reproduced Transformer baseline:
  - F1: 0.6319
  - ROC-AUC: 0.9168
- Added the F1 improvement of 0.1015, or approximately 16%.
- Explained that precision increased from 0.5174 to 0.7226 while recall remained 0.7444.
- Added the latest ROC-curve comparison.
- Added the most important clinical signals, including SpO₂, blood-pressure and heart-rate variability, bicarbonate, and WBC.

### Slide 13 — References

- Replaced general web references with references directly related to the implementation:
  - MIMIC-IV dataset and PhysioNet release
  - Transformer architecture
  - XGBoost
  - LightGBM
  - Project repository
- Recorded repository commit `505e4e5`, accessed on 12 August 2026.

### Slide 14 — Closing Slide

- Corrected the overlapping **Thank You!** typography.

## Short Presentation Explanation

> Our previous presentation represented the initial prototype, which used a small demonstration dataset, 30 features, and only sequence models such as LSTM, GRU, and Transformer. The current implementation has progressed to the complete MIMIC-IV 2.1 cardiac cohort with 35,256 samples, where every sample contains six visits and 59 clinical features.
>
> We found that the patient sequence was relatively short, so we explicitly engineered temporal properties such as trends, variability, velocity, and acceleration. These features were used with HistGradientBoosting, LightGBM, and XGBoost. After probability calibration and validation-based threshold tuning, the soft-voting ensemble achieved an F1 score of 0.7334 and a ROC-AUC of 0.9462. This improved F1 by approximately 16% over the fairly reproduced Transformer baseline.
>
> We also removed the previously proposed agent layer because it has not yet been implemented. Therefore, the updated presentation accurately distinguishes completed work from possible future extensions.

## Important Clarification

The old GRU F1 score of approximately 0.736 and the new GBM F1 score of 0.7334 must not be compared directly. The previous result came from an earlier, smaller-data setup. The updated result comes from the full MIMIC-IV 2.1 cohort and a fair held-out comparison in which all models use the same arrays and split.
