# PUBLISH_CHECKLIST — pasos manuales (los hace el Dr. Márquez)

> El repo ya está **commiteado** localmente con el tag **v1.0.0** preparado. **Nada se publicó.**
> Ejecutar estos pasos solo cuando decidas publicar. El árbol no contiene datos de pacientes.

## 0. Verificación previa (ya hecha, podés reconfirmar)
```bash
cd 08_despliegue_reproducibilidad/kaizen-mrca
git status            # árbol limpio
git log --oneline     # commit v1.0.0
git tag               # v1.0.0
find . -name "*.csv"  # solo data/sample_synthetic.csv (ficticio)
```

## a. Crear el repo público en GitHub
- GitHub → **New repository** → nombre sugerido: **`kaizen-mrca`** → **Public** →
  **sin** README/License/.gitignore (ya están en el repo local).

## b. Conectar y subir (main + tag)
```bash
cd 08_despliegue_reproducibilidad/kaizen-mrca
git remote add origin https://github.com/<TU_USUARIO>/kaizen-mrca.git
git push -u origin main
git push origin v1.0.0
```

## c. Crear el Release en GitHub
- GitHub → repo → **Releases** → **Draft a new release** → elegí el tag **v1.0.0** →
  título “Kaizen-MRCA v1.0.0” → pegá el resumen del `CHANGELOG.md` → **Publish release**.

## d. Conectar Zenodo ↔ GitHub para obtener DOI
1. https://zenodo.org → **Log in with GitHub** → **Settings → GitHub** → activá (toggle ON) el repo `kaizen-mrca`.
2. Volvé a GitHub y **cortá un nuevo release** (p. ej. **v1.0.1**), porque Zenodo solo archiva releases creados *después* de activar el repo.
3. Zenodo genera el **DOI**. Copialo (hay un DOI “de todas las versiones” y uno “de esta versión”).

## e. Devolver URL + DOI para el paper
Pegá de vuelta:
- **URL del repo:** `https://github.com/<TU_USUARIO>/kaizen-mrca`
- **DOI Zenodo:** `10.5281/zenodo.XXXXXXX`

Luego, en Cowork, actualizá:
- `CITATION.cff` → reemplazá los placeholders `repository-code` y `doi`.
- Manuscrito → sección **“Disponibilidad de código y datos”** con repo + DOI + URL de la demo + política de datos.

> ⚠️ Antes del push final, confirmá una vez más que no haya datos reales ni tokens. La demo
> pública (Streamlit) se enlaza **solo después** de que `parity_test.py` pase en el repo publicado.
