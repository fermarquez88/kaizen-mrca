"""
validate_fe.py — valida que build_features() reproduce las 21 features del CSV de
entrenamiento (09_FE_SAP_v2.2.csv). Empareja crudo 06 <-> CSV 09 (1:1) y reporta
correlación por feature. Uso: python src/validate_fe.py
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import fe_sap as fe


def data_dir() -> Path:
    env = os.environ.get("KAIZEN_DATA")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "02_datos" / "09_FE_SAP_v2.2.csv").exists():
            return p / "02_datos"
    raise FileNotFoundError("No encuentro 02_datos/09_FE_SAP_v2.2.csv (definí KAIZEN_DATA).")


def main():
    D = data_dir()
    feat = pd.read_csv(D / "09_FE_SAP_v2.2.csv")
    raw = pd.read_csv(D / "06_neuromentia_mixta_anon_v1.0.csv", low_memory=False)
    adlq = fe.adlq_columns(raw.columns)

    # construir inputs + features desde el crudo
    inputs = [fe.raw_to_inputs(raw.iloc[i], adlq) for i in range(len(raw))]
    idf = pd.DataFrame(inputs)
    # cohorte = filas con ACE y educación (las que están en el CSV)
    idf["ace"] = pd.to_numeric(raw["ACE_TOTAL"], errors="coerce").values
    idf["base"] = raw["base"].values
    cohort = idf[idf["ace"].notna() & idf["edu_anios"].notna() & idf["base"].isin([2023, 2024])].reset_index(drop=True)
    P = fe.fit_params(cohort)

    built = pd.DataFrame([fe.inputs_to_features(cohort.iloc[i].to_dict(), P) for i in range(len(cohort))])
    built["cqc_total"] = cohort["cqc_total"].values  # clave de match

    # emparejar 1:1 con el CSV por (base, edad, edu, ace, cqc)
    def key(df, e, ed, a, c, b="base"):
        return list(zip(df[b], df[e].round(2),
                        pd.to_numeric(df[ed], errors="coerce").round(2),
                        pd.to_numeric(df[a], errors="coerce").round(2),
                        pd.to_numeric(df[c], errors="coerce").round(2)))
    kB = key(built.assign(base=cohort["base"], ace=cohort["ace"]), "edad", "edu_anios", "ace", "cqc_total")
    kF = key(feat, "edad", "edu_raw", "ace_total", "cqc_total")
    built = built.assign(k=kB); feat = feat.assign(k=kF)
    import collections
    dup = {x for x, n in collections.Counter(kB).items() if n > 1} | {x for x, n in collections.Counter(kF).items() if n > 1}
    B = built[~built["k"].isin(dup)].set_index("k")
    F = feat[~feat["k"].isin(dup)].set_index("k")
    common = B.index.intersection(F.index)
    print(f"emparejados 1:1: {len(common)}")

    rows = []
    for col in fe.FEATURES:
        a = pd.to_numeric(B.loc[common, col], errors="coerce")
        b = pd.to_numeric(F.loc[common, col], errors="coerce")
        msk = a.notna() & b.notna()
        c = float(np.corrcoef(a[msk], b[msk])[0, 1]) if msk.sum() > 5 else np.nan
        maxabs = float((a[msk] - b[msk]).abs().max()) if msk.sum() else np.nan
        rows.append({"feature": col, "corr": round(c, 4), "max_abs_diff": round(maxabs, 4), "n": int(msk.sum())})
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    print(f"\ncorr media: {rep['corr'].mean():.4f} | mín: {rep['corr'].min():.4f}")
    return rep


if __name__ == "__main__":
    main()
