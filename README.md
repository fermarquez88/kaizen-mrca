# Kaizen-MRCA — Modelo de Riesgo Cognitivo Autorreportable

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20602277.svg)](https://doi.org/10.5281/zenodo.20602277) · Licencia: MIT

Modelo de cribado **autorreportable** que estima el desempeño cognitivo global (ACE-III) y
señala el **riesgo de bajo desempeño** para **priorizar la derivación** a una Unidad de Memoria
pública (Hospital Dr. G. Rawson, San Juan, Argentina). Métrica norte: **valor predictivo
negativo (VPN)** / sensibilidad. **No es una herramienta diagnóstica.**

> Acompaña al manuscrito (Congreso Argentino de Neurología 2026). Reporte **TRIPOD+AI**.

## Qué hay acá
```
kaizen-mrca/
├── src/
│   ├── fe_sap.py        # ingeniería de variables (autorreporte → 21 features)
│   ├── train.py         # entrena en 2023 y CONGELA el artefacto único
│   ├── inference.py     # predict(raw_inputs) — ÚNICA función de inferencia
│   ├── parity_test.py   # GATE: reproduce los números del paper
│   └── validate_fe.py   # valida la FE contra el CSV de entrenamiento
├── model/
│   ├── kaizen_mrca_v1.joblib   # artefacto congelado (pipeline+isotónica+umbral+FE params+metadatos)
│   ├── model_card.md                # TRIPOD+AI
│   └── training_report.json         # versión, fecha, git SHA, hash de datos, métricas
├── app/streamlit_app.py # UI; importa SOLO inference.predict()
├── data/README.md       # política de datos (NO se publican datos individuales)
├── requirements.txt     # versiones EXACTAS (pinned)
├── LICENSE              # MIT
└── CITATION.cff
```

## Cómo reproducir
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (requiere el CSV anonimizado 09_FE_SAP_v2.2.csv; ver data/README.md)
export KAIZEN_DATA=/ruta/a/02_datos
python src/train.py          # congela model/kaizen_mrca_v1.joblib
python src/parity_test.py    # GATE: AUC dev 0.802 / temporal 0.626 / umbral 0.216
```

## Resultados honestos
- **Desarrollo 2023 (OOF):** AUC **0,80** · VPN **0,83** · umbral 0,216.
- **Validación temporal 2024:** AUC **0,63** (cae por población más educada; calibración recuperable con recalibración local).
- Mejor desempeño en **baja escolaridad** (perfil equitativo).

## Descargo
Herramienta de **cribado de primer nivel**, **no diagnóstica**. Requiere **recalibración local**.
No usar para autodiagnóstico ni para denegar atención.

## Datos
No se publican datos a nivel de participante. Disponibles **bajo solicitud razonable** al autor
(fmarquez.mum@gmail.com) y aprobación del Comité de Ética UCCuyo (protocolo 003/20). Ver `data/README.md`.

## Cita
Ver `CITATION.cff`. Licencia del código: **MIT** (`LICENSE`).
