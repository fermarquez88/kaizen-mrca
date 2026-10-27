# Data policy

No participant-level data are published in this repository, which contains only code and the frozen model artifact.

## Availability

The de-identified dataset belongs to Neuromentia, the community dementia screening program led by Dr. Martín Bruno (Universidad Católica de Cuyo, San Juan, Argentina). Requests for access can be sent to the author (fmarquez.mum@gmail.com) and are forwarded to the program, which decides according to its own institutional and ethical procedures.

## Reproducing locally

`src/train.py`, `src/parity_test.py` and `src/validate_fe.py` expect, in a folder pointed to by the environment variable `KAIZEN_DATA`:

- `09_FE_SAP_v2.2.csv` and `09_FE_SAP_v2.2_manifest.json`, the pre-specified feature matrix;
- `06_neuromentia_mixta_anon_v1.0.csv`, the de-identified raw file, used only to fit and validate the inference feature engineering.

## Synthetic sample (`sample_synthetic.csv`)

`data/sample_synthetic.csv` contains eight fictitious rows generated with `random_state=42`. They do not correspond to any real person and only illustrate the input schema of `inference.predict()`:

```python
import sys; sys.path.insert(0, "src"); import inference, csv
flags = {"hipoacusia","deficit_visual","obesidad","diabetes","colesterol","fumador","vive_solo","sin_cloacas","sin_agua_red"}
for row in csv.DictReader(open("data/sample_synthetic.csv")):
    inp = {k: (v not in ("0", "")) if k in flags else (float(v) if k != "sexo" else v) for k, v in row.items()}
    r = inference.predict(inp); print(r["ace_estimado"], r["decision"])
```

## What the artifact contains

`model/kaizen_mrca_v1.joblib` holds aggregated parameters only: model coefficients, standardization means and deviations, the isotonic calibration map and the decision threshold. No individual record is stored. The SHA-256 hash of the training file is recorded in `model/training_report.json` for traceability without exposing the data.
