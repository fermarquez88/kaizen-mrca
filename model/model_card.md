# Model card: Kaizen-MRCA v1 (TRIPOD+AI)

**Model:** self-report estimate of the risk of a low cognitive screening score, used to prioritize referral to a memory unit.
**Artifact:** `model/kaizen_mrca_v1.joblib`. **Version:** v1. **Maintainer and contact:** Fernando Márquez, MD (fmarquez.mum@gmail.com).
**Reference environment:** Python 3.9.6, scikit-learn 1.6.1, numpy 2.0.2, scipy 1.13.1, pandas 2.3.3, joblib 1.5.3, `random_state=42`.

scikit-learn artifacts are version sensitive: loading or training with another version can break parity. Reproduce with `requirements.txt`; the gate is `src/parity_test.py`.

## 1. Intended and unintended use

- **Intended:** first-level, self-reported screening to prioritize and order referrals to the Memory and Brain Health Unit, Hospital Dr. Guillermo Rawson, San Juan, Argentina. Guiding metric: negative predictive value and sensitivity (rule-out value).
- **Not intended:** diagnosis of cognitive impairment or dementia; replacement of clinical assessment; self-diagnosis; denial of care. Local recalibration is required before use in any other population.

## 2. Population and outcome

- **Development:** Neuromentia community screening program (led by Dr. Martín Bruno, Universidad Católica de Cuyo), San Juan, 2023 wave (n = 631). **Temporal validation:** 2024 wave (n = 146). Adults aged 40 or older with self-report data and an ACE-III score.
- **Outcome:** ACE-III total score below the education-adjusted cut-off used in practice: below 85 with 12 or more years of education, below 68 with fewer than 12. Interpreted as risk, not diagnosis.

## 3. Predictors (21, available in both waves)

Only self-reported variables present in both waves (`09_FE_SAP_v2.2_manifest.json`, `model_features_wave_invariant`): age, age squared, sex, years of education, CQC total, ADLQ percentage, GDS total, CHSC physical, nutrition, social, sleep and mental subscales, cognitive reserve, reserve by age, cardiometabolic burden, obesity, hearing loss, smoking, living alone, sanitation deficit, visual deficit.

Variables available only in 2023 (exposome, multimorbidity, housing, occupational complexity) are used only in the descriptive inequality analysis, never as predictors.

## 4. Architecture (frozen)

`ColumnTransformer(IterativeImputer + StandardScaler)` then `ElasticNet(alpha = 0.3, l1_ratio = 0.8)` on the raw ACE-III score; risk = education cut-off minus estimated ACE-III; `IsotonicRegression`; frozen threshold 0.216 (sensitivity at least 0.80 out of fold).

- Fixed hyperparameters confirmed as winners of nested cross-validation on 2023 (coefficients identical to `GridSearchCV.best_estimator_`).
- Isotonic calibrator and threshold derived from the nested out-of-fold predictions of 2023.
- Preprocessing within folds (no leakage). Trained on 2023 only, not pooled.

## 5. Metrics

| Set | AUC | NPV | Threshold | Note |
|---|---|---|---|---|
| Development 2023 (nested out-of-fold) | 0.802 | 0.827 | 0.216 | discrimination and rule-out |
| Temporal validation 2024 (single run) | 0.626 | 0.722 | 0.216 | more educated cohort |
| Validation 2024, recalibrated | 0.626 | | | calibration recoverable (slope to 1) |

The drop in 2024 reflects a change in population composition (a more educated cohort with higher reserve), not missing variables. Local recalibration is mandatory.

## 6. Equity

Discrimination was better among people with low educational attainment (AUC about 0.80) than among those with high educational attainment; the lower negative predictive value in the latter group is declared as an equity limitation. The education-adjusted cut-off avoids flagging only people with low educational attainment.

## 7. Limitations

Cross-sectional design; the outcome is a screening cut-off, not a clinical reference diagnosis. Voluntary sampling (women over-represented). Self-report (risk of anosognosia). Small 2024 validation (n = 146). No longitudinal follow-up.

## 8. Reproducibility and provenance

- **Training data:** `02_datos/09_FE_SAP_v2.2.csv` (de-identified, not published). SHA-256 and git SHA recorded in `model/training_report.json`.
- **Inference feature engineering** (`src/fe_sap.py`): the original raw-to-features script was not versioned; `fe_sap.py` reproduces it and is validated against the training file (`src/validate_fe.py`): the 21 features match exactly (correlation 1.0, maximum absolute difference 0.0 over the 774 rows matched one to one).
  - `reserva_cognitiva` is an exact affine function of (education, CHSC mental, CHSC social): 0.21112442623486138 x education + 1.3041487319636482 x mental + 1.4871760498940194 x social - 8.828883194872445, fitted by least squares on the 726 complete rows (maximum absolute difference 9.8e-15). `chsc_mental` is the mean of CHSC items 1, 17, 19, 20 and 21, excluding code 9.
  - `res_x_edad` = reserve x z(age), with z(age) = (age - 62.706815) / 17.719660; frozen linear equivalent 0.05643449222027676 x (reserve x age) - 3.5388272395751357 x reserve (maximum absolute difference 4.3e-15).
  - Constants are embedded, not recomputed on new data. This does not affect model parity, which is trained on the exact CSV.
- **Gate:** `python src/parity_test.py` reproduces AUC 0.802 (development), 0.626 (temporal) and threshold 0.216 (tolerance 0.02).

Freeze date, git SHA and hashes: see `model/training_report.json`.
