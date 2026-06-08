"""
inference.py — Función ÚNICA de inferencia del modelo congelado Kaizen-MRCA.
================================================================================
La app Streamlit, el canal WhatsApp y cualquier consumidor deben llamar SOLO a
`predict(raw_inputs)`. Cero lógica de modelo o de FE duplicada en el front.

Carga el artefacto -> build_features (fe_sap + fe_params del artefacto) -> pipeline
-> ACE estimado -> corte por educación -> isotónica -> banda de riesgo + contribuciones.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
import joblib

import fe_sap

ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "model" / "kaizen_mrca_v1.joblib"


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None):
    return joblib.load(path or ARTIFACT_PATH)


def _cut(edu_anios: float, art) -> float:
    r = art["cut_rule"]
    return r["edu_ge_12"] if float(edu_anios) >= 12 else r["edu_lt_12"]


def predict(raw_inputs: dict, artifact_path: str | None = None) -> dict:
    """raw_inputs: dict con el autorreporte (ver fe_sap.inputs_to_features).
    Devuelve: ace_estimado, prob_calibrada, banda_riesgo, decision, margen_al_corte,
    corte, contribuciones (SHAP exacto del modelo lineal)."""
    art = load_artifact(artifact_path)
    FEAT = art["features"]
    Xdf = fe_sap.build_features(raw_inputs, art["fe_params"])  # 1x21 en orden exacto

    ace_est = float(art["pipeline"].predict(Xdf)[0])
    corte = _cut(raw_inputs["edu_anios"], art)
    riesgo = corte - ace_est
    prob = float(np.clip(art["isotonic"].predict([riesgo])[0], 0.0, 1.0))
    thr = float(art["threshold"])
    decision = "derivar" if prob >= thr else "descartar"
    if prob >= thr:
        banda = "alto"
    elif prob >= thr * 0.6:
        banda = "moderado"
    else:
        banda = "bajo"

    # contribuciones = SHAP exacto del modelo lineal: coef_i * (x_i - mean_i)/scale_i  (puntos de ACE)
    x = Xdf.iloc[0].to_dict()
    contribs = {}
    for f in FEAT:
        std = (x[f] - art["feat_mean"][f]) / (art["feat_scale"][f] or 1.0)
        contribs[f] = round(art["coef_ace_por_de"][f] * std, 3)
    contribs = dict(sorted(contribs.items(), key=lambda kv: -abs(kv[1])))

    return {
        "ace_estimado": round(ace_est, 1),
        "corte": corte,
        "margen_al_corte": round(ace_est - corte, 1),
        "prob_calibrada": round(prob, 3),
        "umbral": round(thr, 3),
        "decision": decision,
        "banda_riesgo": banda,
        "contribuciones_shap": contribs,
        "modelo_version": art["meta"]["version"],
        "nota": "Cribado de primer nivel, NO diagnóstico. Requiere recalibración local.",
    }


if __name__ == "__main__":
    ejemplo = dict(edad=78, sexo="Mujer", edu_anios=5, cqc_total=40, adlq_mean03=1.8, gds_total=8,
                   chsc_fisica=2.0, chsc_nutri=2.5, chsc_social=1.5, chsc_sueno=0.9, chsc_mental=1.5,
                   hipoacusia=True, deficit_visual=False, obesidad=True, diabetes=True, colesterol=False,
                   fumador=False, vive_solo=True, sin_cloacas=True, sin_agua_red=False)
    import json
    print(json.dumps(predict(ejemplo), ensure_ascii=False, indent=2))
