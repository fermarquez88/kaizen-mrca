# Política de datos — Kaizen-MRCA

**No se publican datos a nivel de participante.** Por ética y por la regla del proyecto, este
repositorio incluye **solo código + el artefacto del modelo**. No contiene la base de datos.

## Disponibilidad
Los datos (anonimizados) están disponibles **bajo solicitud razonable** al autor
(**fmarquez.mum@gmail.com**) y aprobación del **Comité de Ética en Investigación de la
Universidad Católica de Cuyo (protocolo 003/20)**.

## Para reproducir localmente
`src/train.py`, `src/parity_test.py` y `src/validate_fe.py` esperan, en una carpeta `02_datos/`:
- `09_FE_SAP_v2.2.csv` + `09_FE_SAP_v2.2_manifest.json` (matriz de features pre-especificada).
- `06_neuromentia_mixta_anon_v1.0.csv` (crudo anonimizado; solo para ajustar/validar la FE de inferencia).

Indicá la ruta con la variable de entorno `KAIZEN_DATA=/ruta/a/02_datos`.

## Muestra sintética (`sample_synthetic.csv`)
`data/sample_synthetic.csv` contiene **8 filas FICTICIAS** (generadas con `random_state=42`, **no**
corresponden a ninguna persona real) con el esquema de entrada de `inference.predict()`. Sirve para
probar la inferencia sin datos reales:
```python
import sys; sys.path.insert(0, "src"); import inference, csv
for row in csv.DictReader(open("data/sample_synthetic.csv")):
    inp = {k: (v not in ("0", "")) if k in {"hipoacusia","deficit_visual","obesidad","diabetes",
            "colesterol","fumador","vive_solo","sin_cloacas","sin_agua_red"} else float(v) if k!="sexo" else v
           for k, v in row.items()}
    print(inference.predict(inp)["ace_estimado"], inference.predict(inp)["decision"])
```

## Lo que SÍ viaja en el artefacto
El artefacto (`model/kaizen_mrca_v1.joblib`) contiene **parámetros agregados**
(coeficientes del modelo, medias/desvíos de estandarización, mapa isotónico, umbral) y
**ningún registro individual**. El hash SHA-256 del CSV de entrenamiento queda registrado en
`model/training_report.json` para trazabilidad, sin exponer los datos.
