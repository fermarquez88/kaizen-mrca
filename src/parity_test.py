"""
parity_test.py — GATE de aceptación: el artefacto reproduce los números HONESTOS del paper.
============================================================================================
Reproduce la evaluación del paper (NO un score sobre datos de entrenamiento):
  - AUC out-of-fold en desarrollo 2023 (nested CV) y umbral congelado.
  - AUC en validación temporal 2024 (evaluada una sola vez) con el estimador final + isotónica del artefacto.
Aborta si algo cae fuera de tolerancia (causa más probable: versión de scikit-learn distinta).

Uso:  python src/parity_test.py    (exit 0 = pasa)
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, sklearn, joblib
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

import fe_sap, inference

warnings.filterwarnings("ignore")
RS = 42; np.random.seed(RS)
GRID = {"reg__alpha": [0.03, 0.1, 0.3, 1.0], "reg__l1_ratio": [0.2, 0.5, 0.8, 1.0]}
# objetivos del paper (números honestos)
TARGETS = {"dev_auc": 0.802, "dev_npv": 0.827, "umbral": 0.216, "val_auc": 0.626}
TOL_AUC, TOL_THR = 0.02, 0.02
cut = lambda e: np.where(np.asarray(e) >= 12, 85.0, 68.0)


def data_dir() -> Path:
    import os
    if os.environ.get("KAIZEN_DATA"):
        return Path(os.environ["KAIZEN_DATA"])
    for p in Path(__file__).resolve().parents:
        if (p / "02_datos" / "09_FE_SAP_v2.2.csv").exists():
            return p / "02_datos"
    raise FileNotFoundError("Definí KAIZEN_DATA o ubicá 02_datos/.")


def freeze(yv, p, f=0.80):
    bt, bs = 0.5, -1.0
    for t in np.unique(np.round(p, 4)):
        tn, fp, fn, tp = confusion_matrix(yv, (p >= t).astype(int), labels=[0, 1]).ravel()
        se = tp / (tp + fn) if tp + fn else 0; sp = tn / (tn + fp) if tn + fp else 0
        if se >= f and sp > bs:
            bs, bt = sp, t
    return float(bt)


def npv(yv, p, thr):
    tn, fp, fn, tp = confusion_matrix(yv, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return tn / (tn + fn) if tn + fn else None


def main():
    D = data_dir()
    FEAT = json.load(open(D / "09_FE_SAP_v2.2_manifest.json"))["model_features_wave_invariant"]
    df = pd.read_csv(D / "09_FE_SAP_v2.2.csv")
    y = df["y"].astype(int).values; base = df["base"].values
    ace = pd.to_numeric(df["ace_total"], errors="coerce").values
    edu = pd.to_numeric(df["edu_raw"], errors="coerce").values
    X = df[FEAT].astype(float)
    dev = base == 2023; val = base == 2024
    Xd, yd, aced, edud = X[dev], y[dev], ace[dev], edu[dev]

    pre = ColumnTransformer([("n", Pipeline([("i", IterativeImputer(random_state=RS, max_iter=10)),
                                             ("s", StandardScaler())]), FEAT)])
    inner = KFold(5, shuffle=True, random_state=RS)
    nested = lambda: GridSearchCV(Pipeline([("pre", pre), ("reg", ElasticNet(random_state=RS, max_iter=5000))]),
                                  GRID, scoring="neg_mean_squared_error", cv=inner, n_jobs=-1)
    splits = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(Xd, yd))

    # --- DEV 2023 OOF (nested) ---
    oof = cross_val_predict(nested(), Xd, aced, cv=splits, n_jobs=-1)
    risk = cut(edud) - oof
    iso = IsotonicRegression(out_of_bounds="clip").fit(risk, yd)
    p = iso.predict(risk); thr = freeze(yd, p)
    dev_auc = roc_auc_score(yd, p); dev_npv = npv(yd, p, thr)

    # --- VAL 2024 con el ARTEFACTO congelado (estimador final + isotónica del artefacto) ---
    art = joblib.load(inference.ARTIFACT_PATH)
    risk_v = cut(edu[val]) - art["pipeline"].predict(X[val])
    p_v = art["isotonic"].predict(risk_v)
    val_auc = roc_auc_score(y[val], p_v)

    print(f"sklearn={sklearn.__version__}  (paper: 1.6.1)")
    print(f"DEV 2023 OOF : AUC={dev_auc:.3f} (obj {TARGETS['dev_auc']})  VPN={dev_npv:.3f} (obj {TARGETS['dev_npv']})  umbral={thr:.3f} (obj {TARGETS['umbral']})")
    print(f"VAL 2024     : AUC={val_auc:.3f} (obj {TARGETS['val_auc']})")

    checks = {
        "dev_auc": abs(dev_auc - TARGETS["dev_auc"]) <= TOL_AUC,
        "dev_npv": abs(dev_npv - TARGETS["dev_npv"]) <= 0.03,
        "umbral":  abs(thr - TARGETS["umbral"]) <= TOL_THR,
        "val_auc": abs(val_auc - TARGETS["val_auc"]) <= TOL_AUC,
        "artefacto_umbral_coincide": abs(art["threshold"] - thr) <= 1e-6,
    }
    # smoke test de inferencia
    ej = dict(edad=78, sexo="Mujer", edu_anios=5, cqc_total=40, adlq_mean03=1.8, gds_total=8,
              chsc_fisica=2.0, chsc_nutri=2.5, chsc_social=1.5, chsc_sueno=0.9, chsc_mental=1.5,
              hipoacusia=True, deficit_visual=False, obesidad=True, diabetes=True, colesterol=False,
              fumador=False, vive_solo=True, sin_cloacas=True, sin_agua_red=False)
    out = inference.predict(ej)
    checks["inferencia_coherente"] = (40 <= out["ace_estimado"] <= 100) and (0.0 <= out["prob_calibrada"] <= 1.0)
    print(f"\nSmoke test inferencia: ACE={out['ace_estimado']} prob={out['prob_calibrada']} decision={out['decision']}")

    print("\n--- CHECKS ---")
    for k, v in checks.items():
        print(f"  [{'PASA' if v else 'FALLA'}] {k}")
    if all(checks.values()):
        print("\n✅ PARIDAD OK — el artefacto reproduce los números del paper.")
        return 0
    print("\n❌ PARIDAD FALLA — NO publicar/deployar. Causa probable: versión de scikit-learn (paper=1.6.1).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
