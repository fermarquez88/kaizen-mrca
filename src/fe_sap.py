"""
Kaizen-MRCA — Ingeniería de variables (FE) SAP v2.2
========================================================
Construye las 21 features wave-invariant del modelo congelado a partir del
autorreporte clínico crudo. Es la MISMA definición de FE con la que se generó
`02_datos/09_FE_SAP_v2.2.csv` (manifest `model_features_wave_invariant`).

IMPORTANTE (trazabilidad honesta):
- El modelo se ENTRENA sobre `09_FE_SAP_v2.2.csv` (las features exactas) → la paridad
  con el paper es exacta y NO depende de este módulo.
- Este módulo se usa solo para INFERENCIA de casos nuevos (app / WhatsApp): mapea
  respuestas crudas → 21 features. Cada fórmula fue validada por correlación contra el
  CSV de entrenamiento (ver `src/validate_fe.py`; corr ≈ 1,0 en todas las features).
- Los parámetros de estandarización de los índices compuestos (reserva, res_x_edad)
  se ajustan sobre la cohorte y viajan dentro del artefacto (`fe_params`).

Orden y nombres EXACTOS de las 21 features = manifest. No cambiar el orden.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

FEATURES = [
    "edad", "edad2", "sexo", "edu_anios", "cqc_total", "adlq_pct", "gds_total",
    "chsc_fisica", "chsc_nutri", "chsc_social", "chsc_sueno", "chsc_mental",
    "reserva_cognitiva", "res_x_edad", "carga_cardiometabolica", "obesidad",
    "hipoacusia", "fumador", "vive_solo", "saneamiento_deficit", "deficit_visual",
]
EDAD_CENTER = 63.0  # edad2 = (edad - 63)^2  (validado contra el CSV: diff máx 0)

# --- Constantes CONGELADAS de los índices compuestos (despejadas del CSV de entrenamiento) ---
# El z-scoring del FE original es lineal, por lo que reserva_cognitiva es una función AFÍN
# exacta de (edu, chsc_mental, chsc_social). Constantes ajustadas por mínimos cuadrados sobre
# las 726 filas completas de 09_FE_SAP_v2.2.csv -> max|diff| = 9.8e-15 (precisión de máquina).
# NO recalcular sobre datos nuevos: son constantes del modelo congelado.
RESERVA_AFFINE = (0.21112442623486138,   # peso edu_anios
                  1.3041487319636482,    # peso chsc_mental
                  1.4871760498940194,    # peso chsc_social
                  -8.828883194872445)    # intercepto
# res_x_edad = reserva * z(edad), con z(edad) = (edad - 62.706815) / 17.719660.
# Equivalente lineal exacto: res_x_edad = P*(reserva*edad) + Q*reserva  -> max|diff| = 4.3e-15.
RXE_AFFINE = (0.05643449222027676,       # P
              -3.5388272395751357)       # Q


def _z(x, mean, sd):
    sd = sd if sd and sd > 1e-9 else 1.0
    return (float(x) - mean) / sd


def inputs_to_features(inp: dict, P: dict) -> dict:
    """Cuestionario clínico (dict) -> dict con las 21 features en el orden del modelo.

    Campos esperados en `inp`:
      edad, sexo ('Mujer'/'Hombre'), edu_anios,
      cqc_total (0-96), adlq_pct (0-100) o adlq_mean03 (0-3),
      gds_total (0-15),
      chsc_fisica/chsc_nutri/chsc_social/chsc_mental (1-4), chsc_sueno (0-2),
      hipoacusia, deficit_visual, obesidad, diabetes, colesterol (bool),
      fumador, vive_solo, sin_cloacas, sin_agua_red (bool)
    """
    m, s = P["means"], P["sds"]
    f = {}
    edad = float(inp["edad"]); edu = float(inp["edu_anios"])
    f["edad"] = edad
    f["edad2"] = (edad - EDAD_CENTER) ** 2
    f["sexo"] = 1.0 if str(inp.get("sexo", "Mujer")) == "Mujer" else 0.0
    f["edu_anios"] = edu
    f["cqc_total"] = float(inp.get("cqc_total", 0.0))
    if inp.get("adlq_pct") is not None:
        f["adlq_pct"] = float(inp["adlq_pct"])
    else:  # desde promedio 0-3 -> % deterioro
        f["adlq_pct"] = float(inp.get("adlq_mean03", 0.0)) / 3.0 * 100.0
    f["gds_total"] = float(inp.get("gds_total", 0.0))

    chf = float(inp.get("chsc_fisica", m["fisica"]))
    chn = float(inp.get("chsc_nutri", m["nutri"]))
    chs = float(inp.get("chsc_social", m["social"]))
    chsu = float(inp.get("chsc_sueno", m["sueno"]))
    chm = float(inp.get("chsc_mental", m["mental"]))
    f["chsc_fisica"], f["chsc_nutri"], f["chsc_social"], f["chsc_sueno"], f["chsc_mental"] = chf, chn, chs, chsu, chm

    # reserva cognitiva: forma AFÍN exacta con constantes congeladas (edu, mental, social)
    A, B, C, D = RESERVA_AFFINE
    reserva = A * edu + B * chm + C * chs + D
    f["reserva_cognitiva"] = reserva
    # res_x_edad = reserva * z(edad) (equivalente lineal exacto con constantes congeladas)
    Pc, Qc = RXE_AFFINE
    f["res_x_edad"] = Pc * (reserva * edad) + Qc * reserva

    dia = 1.0 if inp.get("diabetes") else 0.0
    col = 1.0 if inp.get("colesterol") else 0.0
    obe = 1.0 if inp.get("obesidad") else 0.0
    f["carga_cardiometabolica"] = dia + col + obe
    f["obesidad"] = obe
    f["hipoacusia"] = 1.0 if inp.get("hipoacusia") else 0.0
    f["fumador"] = 1.0 if inp.get("fumador") else 0.0
    f["vive_solo"] = 1.0 if inp.get("vive_solo") else 0.0
    f["saneamiento_deficit"] = (1.0 if inp.get("sin_cloacas") else 0.0) + (1.0 if inp.get("sin_agua_red") else 0.0)
    f["deficit_visual"] = 1.0 if inp.get("deficit_visual") else 0.0
    return f


def build_features(raw_inputs: dict, P: dict) -> pd.DataFrame:
    """raw_inputs (dict) -> DataFrame de 1 fila con las 21 features en orden."""
    f = inputs_to_features(raw_inputs, P)
    return pd.DataFrame([[f[k] for k in FEATURES]], columns=FEATURES)


# =====================================================================
# Construcción desde el dataset crudo 06 (para ajustar fe_params y validar)
# =====================================================================
CHSC = {
    "fisica": ["CHSC_2_actfisica", "CHSC_3_actfisicatiempo", "CHSC_4_actfisica10años", "CHSC_5_actfisicavida"],
    "nutri":  ["CHSC_6_nutricion", "CHSC_7_nutricionvida"],
    "social": ["CHSC_8_contsocialamigos", "CHSC_9_contsocialfam", "CHSC_10_activsocial", "CHSC_18_activsocialcultural"],
    "mental": ["CHSC_1_habitossalud", "CHSC_17_esfuerzocognitivo", "CHSC_19_menteactiva10años",
               "CHSC_20_habitosexpect", "CHSC_21_habitosincorporar"],
    "sueno":  ["CHSC_11_sueñocalidad", "CHSC_12_sueñosomnol", "CHSC_13_sueñosiesta",
               "CHSC_14_sueñoregular", "CHSC_15_sueñohoras", "CHSC_16_sueñodif"],
}


def _num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan


def _mean_no9(vals):
    a = np.array([_num(v) for v in vals], dtype=float)
    a = a[~np.isnan(a)]; a = a[a != 9]
    return float(np.mean(a)) if len(a) else np.nan


def raw_to_inputs(row: pd.Series, adlq_cols: list) -> dict:
    """Fila del crudo 06 -> dict de inputs canónicos."""
    chsc = {k: _mean_no9([row.get(c) for c in cols]) for k, cols in CHSC.items()}
    A = np.array([_num(row.get(c)) for c in adlq_cols], dtype=float)
    A = A[~np.isnan(A)]; A = A[A != 9]
    adlq_pct = float(np.mean(A)) / 3.0 * 100.0 if len(A) else np.nan
    imc = _num(row.get("IMC"))
    return dict(
        edad=_num(row.get("edad")),
        sexo="Mujer" if str(row.get("sexo")) == "Mujer" else "Hombre",
        edu_anios=_num(row.get("ed_anos_completos")),
        cqc_total=_num(row.get("CQC_TOTAL")),
        adlq_pct=adlq_pct,
        gds_total=_num(row.get("Escala_Yesavage_TOTAL")),
        chsc_fisica=chsc["fisica"], chsc_nutri=chsc["nutri"], chsc_social=chsc["social"],
        chsc_sueno=chsc["sueno"], chsc_mental=chsc["mental"],
        hipoacusia=(_num(row.get("as_auditivo")) > 0),
        deficit_visual=(_num(row.get("as_visual")) == 1),
        obesidad=(imc >= 30),
        diabetes=(_num(row.get("diabetes")) > 0),
        colesterol=(_num(row.get("hipcolesterol")) > 0),
        fumador=(_num(row.get("tabaquismo")) > 0),
        vive_solo=(str(row.get("grupo_convivencia")).strip().lower() == "vive solo"),
        sin_cloacas=(_num(row.get("servicios_casa_cloacas")) == 2),
        sin_agua_red=(_num(row.get("servicios_casa_agua")) == 2),
    )


def fit_params(idf: pd.DataFrame) -> dict:
    """Ajusta medias/desvíos de la cohorte para los z-scores reproducibles.
    idf: DataFrame de inputs (de raw_to_inputs) sobre la cohorte."""
    def ms(col):
        s = pd.to_numeric(idf[col], errors="coerce")
        return float(s.mean()), float(s.std())
    means, sds = {}, {}
    for key, col in [("edu", "edu_anios"), ("mental", "chsc_mental"), ("social", "chsc_social"),
                     ("fisica", "chsc_fisica"), ("nutri", "chsc_nutri"), ("sueno", "chsc_sueno"),
                     ("edad", "edad")]:
        means[key], sds[key] = ms(col)

    def z(col, k):
        return (pd.to_numeric(idf[col], errors="coerce") - means[k]) / (sds[k] if sds[k] > 1e-9 else 1.0)
    res_sum = z("edu_anios", "edu") + z("chsc_mental", "mental") + z("chsc_social", "social")
    means["res_sum"], sds["res_sum"] = float(res_sum.mean()), float(res_sum.std())
    return {"means": means, "sds": sds, "edad_center": EDAD_CENTER}


def adlq_columns(raw_columns) -> list:
    return [c for c in raw_columns if c.startswith("ADLQ_") and re.search(r"_\([A-G]\)$", c)]
