# Kaizen-MRCA: a self-report model to prioritize referral to a public memory unit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20602277.svg)](https://doi.org/10.5281/zenodo.20602277) License: MIT

Kaizen-MRCA (Modelo de Riesgo Cognitivo Autorreportable) is a first-level screening model that estimates global cognitive performance (ACE-III total score) from self-reported information and flags the risk of a low score, so that a public memory unit can prioritize referrals. Its guiding metric is the negative predictive value, that is, its ability to rule out. It is not a diagnostic tool.

Developed for the Memory and Brain Health Unit, Hospital Dr. Guillermo Rawson, San Juan, Argentina, with data from Neuromentia, the community dementia screening program led by Dr. Martín Bruno (Universidad Católica de Cuyo). Presented as a poster at the 16th Argentine Congress of Neuropsychology (SONEPSA, 2025; honorable mention, professionals category). Reported following TRIPOD+AI.

## Contents

```
kaizen-mrca/
  src/
    fe_sap.py          feature engineering (self-report to 21 features)
    train.py           trains on the 2023 wave and freezes the single artifact
    inference.py       predict(raw_inputs): the only inference entry point
    parity_test.py     gate that reproduces the reported metrics
    validate_fe.py     validates the feature engineering against the training file
  model/
    kaizen_mrca_v1.joblib   frozen artifact (pipeline, isotonic calibration, threshold, metadata)
    MODEL_CARD.md           model card in English (TRIPOD+AI); model_card.es.md in Spanish
    training_report.json    version, date, git SHA, data hash, metrics
  app/streamlit_app.py      user interface; imports only inference.predict()
  data/README.md            data policy (no participant-level data are published)
  data/sample_synthetic.csv eight fictitious rows for testing the inference
  requirements.txt          pinned versions
  CITATION.cff
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export KAIZEN_DATA=/path/to/data      # see data/README.md
python src/train.py                    # freezes model/kaizen_mrca_v1.joblib
python src/parity_test.py              # gate: AUC 0.802 (development), 0.626 (temporal validation), threshold 0.216
```

## Results

- Development, 2023 wave (n = 631), out-of-fold: AUC 0.80, negative predictive value 0.83, threshold 0.216.
- Temporal validation, 2024 wave (n = 146): AUC 0.63. The drop reflects a more educated population; calibration is recoverable with local recalibration.
- Discrimination was better among people with low educational attainment than among those with high educational attainment; the lower negative predictive value in the latter group is declared as an equity limitation.

## Intended use and limits

A first-level, self-report screening aid that requires local recalibration before use in any other population. It does not diagnose cognitive impairment or dementia, does not replace clinical assessment, and must not be used for self-diagnosis or to deny care.

## Data

No participant-level data are published. The repository contains code and the frozen model artifact, which holds only aggregated parameters (coefficients, scaling constants, the isotonic map and the threshold). Access to the de-identified dataset is governed by the Neuromentia program and its institutional procedures; requests can be sent to the author and are forwarded to the program. See `data/README.md`.

## Citation

See `CITATION.cff` and the Zenodo record (doi:10.5281/zenodo.20602277). Code under the MIT License.
