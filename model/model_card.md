# Model Card — Kaizen-MRCA v1 (TRIPOD+AI)

**Modelo:** estimación autorreportable del riesgo de bajo desempeño cognitivo para **priorizar la derivación** a una Unidad de Memoria.
**Artefacto:** `model/kaizen_mrca_v1.joblib` · **Versión:** v1 · **Director:** Dr. Fernando Márquez (fmarquez.mum@gmail.com).
**Entorno de referencia:** Python 3.9.6 · scikit-learn **1.6.1** · numpy 2.0.2 · scipy 1.13.1 · pandas 2.3.3 · joblib 1.5.3 · `random_state=42`.

> ⚠️ Los artefactos de scikit-learn son **sensibles a la versión**. Cargar/entrenar con otra versión puede romper la paridad. Reproducir con `requirements.txt`. El gate es `src/parity_test.py`.

## 1. Uso previsto y NO previsto
- **Previsto:** cribado de **primer nivel**, autorreportado, para **priorizar/ordenar derivaciones** a la Unidad de Memoria y Salud Cerebral (Hospital Rawson, San Juan). Métrica norte: **VPN / sensibilidad** (valor de descarte).
- **NO previsto:** **no es diagnóstico** de deterioro cognitivo ni demencia; no reemplaza la evaluación clínica; no debe usarse para autodiagnóstico ni para denegar atención. **Requiere recalibración local** antes de su uso en otra población.

## 2. Población y desenlace
- **Desarrollo:** programa Neuromentia, San Juan, 2023 (**n=631**). **Validación temporal externa:** ola 2024 (**n=146**). Adultos ≥40 años, autorreporte + ACE-III disponible.
- **Desenlace:** ACE-III **por debajo del corte ajustado por educación** (Bruno): **<85** si ≥12 años de educación; **<68** si <12. Interpretado como **riesgo**, no diagnóstico.

## 3. Predictores (21, wave-invariant)
Solo variables autorreportables presentes en ambas olas (`09_FE_SAP_v2.2_manifest.json` → `model_features_wave_invariant`):
`edad, edad2, sexo, edu_anios, cqc_total, adlq_pct, gds_total, chsc_fisica, chsc_nutri, chsc_social, chsc_sueno, chsc_mental, reserva_cognitiva, res_x_edad, carga_cardiometabolica, obesidad, hipoacusia, fumador, vive_solo, saneamiento_deficit, deficit_visual`.
Las variables 2023-only (exposoma, multimorbilidad, casa/piso, complejidad ocupacional) se usan **solo en el análisis descriptivo de desigualdad**, **NUNCA como predictores**. *(El despliegue anterior incluía por error "NBI/privación material" en el gráfico de contribuciones: esa variable NO es del modelo.)*

## 4. Arquitectura (congelada)
`ColumnTransformer(IterativeImputer + StandardScaler)` → `ElasticNet(α=0.3, l1_ratio=0.8)` sobre **ACE-III crudo** → `riesgo = corte(educación) − ACE_estimado` → `IsotonicRegression` → **umbral congelado 0.216** (sensibilidad ≥0.80, fuera de fold).
- Hiperparámetros **fijos** confirmados como ganadores de la nested CV sobre 2023 (`best_params = {α:0.3, l1_ratio:0.8}`; coeficientes idénticos a `GridSearchCV.best_estimator_`).
- Calibrador isotónico y umbral derivados del **OOF anidado de 2023** (procedimiento exacto del paper).
- Preprocesamiento **dentro de fold** (sin leakage). Entrenado **solo en 2023** (no pooled).

## 5. Métricas honestas
| Conjunto | AUC | VPN | Umbral | Nota |
|---|---|---|---|---|
| **Desarrollo 2023 (OOF anidado)** | **0,802** | **0,827** | 0,216 | discriminación + descarte |
| **Validación temporal 2024 (1 vez)** | **0,626** | 0,722 | 0,216 | cohorte más educada |
| Validación 2024 **recalibrada** | 0,626 | — | — | calibración recuperable (slope→1) |

La caída en 2024 se debe a un **cambio de composición poblacional** (cohorte más educada, mayor reserva), no a variables faltantes. **Recalibración local obligatoria.**

## 6. Equidad
Mejor desempeño en **baja escolaridad** (AUC ~0,80) que en alta; el menor VPN en alta escolaridad se declara como **limitación de equidad**. El corte ajustado por educación evita re-detectar solo a personas de menor escolaridad.

## 7. Limitaciones
Diseño transversal; desenlace = corte de cribado, no diagnóstico clínico de referencia. Muestreo voluntario (sobrerrepresentación femenina). Autorreporte (riesgo de anosognosia). Validación 2024 pequeña (n=146). Sin seguimiento longitudinal.

## 8. Reproducibilidad y procedencia
- **Datos de entrenamiento:** `02_datos/09_FE_SAP_v2.2.csv` (anonimizado; **no se publica**). SHA-256 y git SHA registrados en `model/training_report.json`.
- **Ingeniería de variables para inferencia** (`src/fe_sap.py`): el script original de FE (crudo→features) corrió en el scratchpad de Cowork y no está versionado; `fe_sap.py` lo **reproduce y valida contra el CSV de entrenamiento** (`src/validate_fe.py`): **las 21 features quedan bit-exactas** (corr 1,0, max|diff| = 0,0 sobre las 774 filas emparejadas 1:1).
  - Detalle de los composites (constantes **congeladas**, despejadas del CSV; ver `src/fe_sap.py`):
    - `reserva_cognitiva` es una función **afín** exacta de (edu, chsc_mental, chsc_social):
      `0.21112442623486138·edu + 1.3041487319636482·mental + 1.4871760498940194·social − 8.828883194872445`.
      Constantes ajustadas por mínimos cuadrados sobre las **726 filas completas** → max|diff| = 9,8e-15.
      (El z-scoring original es lineal, por eso un ajuste afín lo recupera exactamente; `chsc_mental` = media de los ítems CHSC 1,17,19,20,21 excluyendo código 9.)
    - `res_x_edad` = `reserva · z(edad)` con `z(edad) = (edad − 62.706815) / 17.719660`; equivalente lineal congelado
      `0.05643449222027676·(reserva·edad) − 3.5388272395751357·reserva` → max|diff| = 4,3e-15.
  - **Criterio:** constantes embebidas (no recalculadas sobre datos nuevos). Esto NO afecta la paridad del modelo, que se entrena sobre el CSV exacto.
- **Gate:** `python src/parity_test.py` reproduce AUC dev 0,802 / temporal 0,626 / umbral 0,216 (tolerancia ±0,02).

**Fecha de congelado / git SHA / hashes:** ver `model/training_report.json`.
