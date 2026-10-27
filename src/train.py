"""
train.py — Congela el modelo Kaizen-MRCA como artefacto único y reproducible.
==================================================================================
Entrena SOLO sobre desarrollo 2023 del CSV pre-especificado (09_FE_SAP_v2.2.csv),
reproduciendo el pipeline validado del paper (s4_pipeline_SAP.py):

  ColumnTransformer(IterativeImputer + StandardScaler) -> ElasticNet(ACE crudo)
  -> riesgo = corte(educación) - ACE_estimado -> IsotonicRegression -> umbral congelado.

Hiperparámetros FIJOS α=0,3 · l1_ratio=0,8 (confirmados como ganadores de la nested CV
sobre 2023; coeficientes idénticos a GridSearchCV.best_estimator_). El calibrador isotónico
y el umbral (≈0,216) se derivan del OOF anidado de 2023 (procedimiento exacto del paper).

Preprocesamiento dentro de fold (sin leakage). random_state=42. NO reajusta en pooled 2023+2024.

Salida: model/kaizen_mrca_v1.joblib   +   model/training_report.json
Uso:    python src/train.py
"""
from __future__ import annotations
import json, hashlib, subprocess, warnings, datetime, sys
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

import fe_sap

warnings.filterwarnings("ignore")
RS = 42; np.random.seed(RS); SENS_FLOOR = 0.80
ALPHA, L1 = 0.3, 0.8
GRID = {"reg__alpha": [0.03, 0.1, 0.3, 1.0], "reg__l1_ratio": [0.2, 0.5, 0.8, 1.0]}

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MODEL_DIR = REPO / "model"; MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT = MODEL_DIR / "kaizen_mrca_v1.joblib"

cut = lambda e: np.where(np.asarray(e) >= 12, 85.0, 68.0)


def data_dir() -> Path:
    import os
    if os.environ.get("KAIZEN_DATA"):
        return Path(os.environ["KAIZEN_DATA"])
    for p in HERE.parents:
        if (p / "02_datos" / "09_FE_SAP_v2.2.csv").exists():
            return p / "02_datos"
    raise FileNotFoundError("Definí KAIZEN_DATA o ubicá 02_datos/09_FE_SAP_v2.2.csv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a (repo aún no inicializado)"


def freeze_threshold(yv, p, floor=SENS_FLOOR):
    bt, bs = 0.5, -1.0
    for t in np.unique(np.round(p, 4)):
        tn, fp, fn, tp = confusion_matrix(yv, (p >= t).astype(int), labels=[0, 1]).ravel()
        se = tp / (tp + fn) if tp + fn else 0; sp = tn / (tn + fp) if tn + fp else 0
        if se >= floor and sp > bs:
            bs, bt = sp, t
    return float(bt)


def metrics(yv, p, thr):
    tn, fp, fn, tp = confusion_matrix(yv, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return dict(auc_roc=round(roc_auc_score(yv, p), 3),
                sens=round(tp / (tp + fn), 3), spec=round(tn / (tn + fp), 3),
                npv=round(tn / (tn + fn), 3) if tn + fn else None,
                ppv=round(tp / (tp + fp), 3) if tp + fp else None)


def main():
    D = data_dir()
    CSV = D / "09_FE_SAP_v2.2.csv"; MAN = D / "09_FE_SAP_v2.2_manifest.json"
    FEAT = json.load(open(MAN))["model_features_wave_invariant"]
    assert FEAT == fe_sap.FEATURES, "Desalineación: manifest vs fe_sap.FEATURES"

    df = pd.read_csv(CSV)
    y = df["y"].astype(int).values; base = df["base"].values
    ace = pd.to_numeric(df["ace_total"], errors="coerce").values
    edu = pd.to_numeric(df["edu_raw"], errors="coerce").values
    X = df[FEAT].astype(float)
    dev = base == 2023; val = base == 2024
    Xd, yd, aced, edud = X[dev], y[dev], ace[dev], edu[dev]
    print(f"n={len(df)} | dev2023={dev.sum()} | val2024={val.sum()} | features={len(FEAT)}")

    pre = ColumnTransformer([("n", Pipeline([("i", IterativeImputer(random_state=RS, max_iter=10)),
                                             ("s", StandardScaler())]), FEAT)])
    outer = StratifiedKFold(5, shuffle=True, random_state=RS)
    inner = KFold(5, shuffle=True, random_state=RS)

    # (1) OOF anidado en 2023 (procedimiento EXACTO del paper) -> calibrador + umbral
    def nested():
        return GridSearchCV(Pipeline([("pre", pre), ("reg", ElasticNet(random_state=RS, max_iter=5000))]),
                            GRID, scoring="neg_mean_squared_error", cv=inner, n_jobs=-1)
    splits = list(outer.split(Xd, yd))
    oof_ace = cross_val_predict(nested(), Xd, aced, cv=splits, n_jobs=-1)
    risk_dev = cut(edud) - oof_ace
    iso = IsotonicRegression(out_of_bounds="clip").fit(risk_dev, yd)
    p_dev = iso.predict(risk_dev)
    thr = freeze_threshold(yd, p_dev)
    dev_m = metrics(yd, p_dev, thr)
    print(f"DEV 2023 (OOF anidado): AUC={dev_m['auc_roc']} VPN={dev_m['npv']} | umbral={thr:.3f}")

    # confirmar que la nested CV aterriza en (0.3, 0.8) sobre 2023 completo
    gs_full = nested().fit(Xd, aced)
    best = gs_full.best_params_
    print(f"nested best_params (2023 completo): {best}")
    landed = (abs(best["reg__alpha"] - ALPHA) < 1e-9 and abs(best["reg__l1_ratio"] - L1) < 1e-9)

    # (2) Estimador FINAL con hiperparámetros FIJOS (≡ ganador) sobre 2023 completo
    final_est = Pipeline([("pre", pre), ("reg", ElasticNet(alpha=ALPHA, l1_ratio=L1,
                                                           random_state=RS, max_iter=5000))]).fit(Xd, aced)
    # coeficientes en puntos de ACE por DE (para contribuciones en inferencia)
    scaler = final_est.named_steps["pre"].named_transformers_["n"].named_steps["s"]
    coef = final_est.named_steps["reg"].coef_

    # validación temporal 2024 (UNA sola vez) con el estimador final + isotónica del paper
    risk_val = cut(edu[val]) - final_est.predict(X[val])
    p_val = iso.predict(risk_val)
    val_m = metrics(y[val], p_val, thr)
    print(f"VALIDACIÓN TEMPORAL 2024: AUC={val_m['auc_roc']} VPN={val_m['npv']}")

    # (3) Parámetros de FE para inferencia (ajustados desde el crudo de la cohorte) + validación FE
    raw = pd.read_csv(D / "06_neuromentia_mixta_anon_v1.0.csv", low_memory=False)
    adlq = fe_sap.adlq_columns(raw.columns)
    inputs = pd.DataFrame([fe_sap.raw_to_inputs(raw.iloc[i], adlq) for i in range(len(raw))])
    inputs["ace"] = pd.to_numeric(raw["ACE_TOTAL"], errors="coerce").values
    inputs["base"] = raw["base"].values
    cohort = inputs[inputs["ace"].notna() & inputs["edu_anios"].notna() & inputs["base"].isin([2023, 2024])]
    fe_params = fe_sap.fit_params(cohort.reset_index(drop=True))

    # (4) Artefacto único
    meta = {
        "name": "kaizen-mrca", "version": "v1",
        "fecha": datetime.date.today().isoformat(),
        "git_sha": git_sha(),
        "data_csv": CSV.name, "data_sha256": sha256(CSV),
        "manifest_sha256": sha256(MAN),
        "python": sys.version.split()[0], "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "random_state": RS, "alpha": ALPHA, "l1_ratio": L1,
        "nested_best_params": best, "fixed_equals_nested_winner": bool(landed),
        "n_dev_2023": int(dev.sum()), "n_val_2024": int(val.sum()),
        "metrics_dev_oof": dev_m, "metrics_val_temporal": val_m,
        "umbral_congelado": round(thr, 4),
        "outcome": "ACE-III por debajo del corte ajustado por educación usado en la práctica: <85 si ≥12 años; <68 si <12",
        "uso_no_previsto": "NO diagnóstico. Cribado de primer nivel; requiere recalibración local.",
    }
    artifact = {
        "pipeline": final_est, "isotonic": iso, "threshold": float(thr),
        "features": FEAT, "cut_rule": {"edu_ge_12": 85.0, "edu_lt_12": 68.0},
        "fe_params": fe_params, "fe_module": "fe_sap",
        "coef_ace_por_de": {f: float(c) for f, c in zip(FEAT, coef)},
        "feat_mean": dict(zip(FEAT, scaler.mean_.tolist())),
        "feat_scale": dict(zip(FEAT, scaler.scale_.tolist())),
        "intercept_ace": float(final_est.named_steps["reg"].intercept_),
        "meta": meta,
    }
    joblib.dump(artifact, ARTIFACT)
    json.dump(meta, open(MODEL_DIR / "training_report.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nArtefacto guardado: {ARTIFACT}")
    print(f"sklearn={sklearn.__version__} | umbral={thr:.4f} | fixed==nested winner: {landed}")
    return artifact


if __name__ == "__main__":
    main()
