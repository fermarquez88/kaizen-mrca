# Changelog — Kaizen-MRCA

Formato basado en [Keep a Changelog](https://keepachangelog.com/). Versionado semántico.

## [1.0.1] — 2026-06-08
Release archivado en **Zenodo** para obtener DOI citable. **Sin cambios** en el modelo,
la ingeniería de variables ni el código respecto de v1.0.0 (paridad idéntica: AUC dev 0,802 /
temporal 0,626 / umbral 0,216, sklearn 1.6.1).

## [1.0.0] — 2026-06-08
Primer congelado reproducible del modelo.

### Added
- Artefacto único congelado `model/kaizen_mrca_v1.joblib`: pipeline ajustado
  (IterativeImputer + StandardScaler → ElasticNet α=0,3 · l1_ratio=0,8 sobre ACE-III),
  calibración isotónica, **umbral congelado 0,216**, corte por educación (Bruno), las 21
  features wave-invariant en orden y metadatos (versión, fecha, hash SHA-256 de datos,
  sklearn 1.6.1, random_state=42).
- `src/train.py` — entrena en desarrollo 2023 y serializa el artefacto.
- `src/inference.py` — función única `predict(raw_inputs)`.
- `src/fe_sap.py` — ingeniería de variables crudo→21 features, **bit-exacta** vs el CSV de
  entrenamiento (constantes afines congeladas para `reserva_cognitiva` y `res_x_edad`).
- `src/parity_test.py` — gate de aceptación (reproduce AUC dev 0,802 / temporal 0,626 / umbral 0,216).
- `src/validate_fe.py` — valida la FE (21/21 features a max|diff| 0,0).
- `model/model_card.md` (TRIPOD+AI), `README.md`, `LICENSE` (MIT), `CITATION.cff`,
  `requirements.txt` (pinneado), `data/README.md` + `data/sample_synthetic.csv` (datos ficticios).

### Notes
- Entrenado **solo en 2023** (no pooled). Requiere **recalibración local**. **No diagnóstico**.
- No se publican datos a nivel de participante.
