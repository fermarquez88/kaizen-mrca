"""
streamlit_app.py — UI del cribado Kaizen-MRCA.
===================================================
SOLO interfaz. Toda la lógica de modelo y de FE vive en el artefacto congelado y se
consume EXCLUSIVAMENTE vía inference.predict(). Cero lógica de modelo duplicada acá.
Ejecutar:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import inference  # noqa: E402
import fe_sap      # noqa: E402

st.set_page_config(page_title="Kaizen-MRCA · Riesgo cognitivo", page_icon="🧠", layout="wide")

art = inference.load_artifact()
META = art["meta"]; FEP = art["fe_params"]["means"]
DEV = META["metrics_dev_oof"]; VAL = META["metrics_val_temporal"]

st.markdown("""<style>
.block-container{max-width:900px;padding-top:1.2rem}
h1{font-size:1.5rem!important}.sub{color:#5b6675;font-size:.9rem}
.badge{display:inline-block;font-weight:700;padding:.35rem .8rem;border-radius:999px;font-size:1rem}
.ok{background:#e8f5ec;color:#1e7a3c;border:1px solid #b8e0c4}
.bad{background:#fdecea;color:#b3261e;border:1px solid #f4c4bf}
</style>""", unsafe_allow_html=True)

st.title("🧠 Estimación del riesgo cognitivo individual")
st.markdown('<div class="sub">Cribado autorreportable para <b>priorizar derivación</b> a la Unidad de '
            'Memoria (Hospital Rawson, San Juan). <b>No diagnostica</b>. Modelo congelado v1 · '
            f'AUC desarrollo {DEV["auc_roc"]} · VPN {DEV["npv"]} · validación temporal {VAL["auc_roc"]}.</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Datos de la persona")
    modo = st.radio("Modo", ["⚡ Rápido", "📋 Completo"], horizontal=True)
    completo = modo.startswith("📋")
    edad = st.number_input("Edad", 40, 100, 65)
    sexo = st.radio("Sexo", ["Mujer", "Hombre"], horizontal=True)
    edu = st.number_input("Años de educación", 0, 25, 9)
    cqc = st.slider("Quejas cognitivas (CQC total 0–96)", 0, 96, 18)
    adlq = st.slider("Dificultad funcional (ADLQ, 0–3)", 0.0, 3.0, 0.4, 0.1)
    gds = st.slider("Síntomas depresivos (GDS-15, 0–15)", 0, 15, 3)
    chm = st.slider("Actividad mental (CHSC 1–4)", 1.0, 4.0, 2.5, 0.25)
    chs = st.slider("Actividad social (CHSC 1–4)", 1.0, 4.0, 2.5, 0.25)
    hipo = st.checkbox("Hipoacusia")
    vsolo = st.checkbox("Vive solo/a")
    if completo:
        st.subheader("Hábitos y entorno")
        chf = st.slider("Actividad física (1–4)", 1.0, 4.0, float(round(FEP["fisica"], 1)), 0.25)
        chn = st.slider("Alimentación (1–4)", 1.0, 4.0, float(round(FEP["nutri"], 1)), 0.25)
        chsu = st.slider("Índice de sueño (CHSC)", 0.0, 2.0, float(round(FEP["sueno"], 2)), 0.1)
        visdef = st.checkbox("Déficit visual no corregido")
        obes = st.checkbox("Obesidad (IMC ≥ 30)")
        dia = st.checkbox("Diabetes"); col = st.checkbox("Colesterol elevado")
        fum = st.checkbox("Fuma actualmente")
        clo = st.checkbox("Vivienda SIN cloacas"); agu = st.checkbox("Vivienda SIN agua de red")
    else:
        chf, chn, chsu = FEP["fisica"], FEP["nutri"], FEP["sueno"]
        visdef = obes = dia = col = fum = clo = agu = False

raw_inputs = dict(edad=edad, sexo=sexo, edu_anios=edu, cqc_total=cqc, adlq_mean03=adlq, gds_total=gds,
                  chsc_fisica=chf, chsc_nutri=chn, chsc_social=chs, chsc_sueno=chsu, chsc_mental=chm,
                  hipoacusia=hipo, deficit_visual=visdef, obesidad=obes, diabetes=dia, colesterol=col,
                  fumador=fum, vive_solo=vsolo, sin_cloacas=clo, sin_agua_red=agu)

r = inference.predict(raw_inputs)
pos = r["decision"] == "derivar"
st.markdown((f'<span class="badge bad">⚑ DERIVAR — prioridad alta</span>' if pos
             else f'<span class="badge ok">✓ Descarte razonable</span>'), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ACE-III estimado", r["ace_estimado"])
c2.metric("Corte (educación)", int(r["corte"]))
c3.metric("Margen al corte", r["margen_al_corte"])
c4.metric("Prob. bajo desempeño", f"{r['prob_calibrada']*100:.0f}%")

st.caption(f"Umbral de decisión congelado: {r['umbral']*100:.0f}%.")

st.subheader("¿Qué pesa en esta estimación?")
LAB = {"edad": "Edad", "edad2": "Edad (no lineal)", "sexo": "Sexo", "edu_anios": "Educación",
       "cqc_total": "Quejas cognitivas", "adlq_pct": "Dificultad funcional", "gds_total": "Síntomas depresivos",
       "chsc_fisica": "Act. física", "chsc_nutri": "Alimentación", "chsc_social": "Act. social",
       "chsc_sueno": "Sueño", "chsc_mental": "Act. mental", "reserva_cognitiva": "Reserva cognitiva",
       "res_x_edad": "Reserva × edad", "carga_cardiometabolica": "Carga cardiometabólica",
       "obesidad": "Obesidad", "hipoacusia": "Hipoacusia", "fumador": "Tabaquismo", "vive_solo": "Vive solo",
       "saneamiento_deficit": "Saneamiento deficiente", "deficit_visual": "Déficit visual"}
items = [(LAB.get(k, k), v) for k, v in r["contribuciones_shap"].items() if abs(v) > 0.05][:8][::-1]
if items:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 0.34 * len(items) + 0.3), dpi=120)
    vals = [v for _, v in items]; names = [n for n, _ in items]
    ax.barh(names, vals, color=["#1f8a5b" if v > 0 else "#c8443a" for v in vals], height=0.6)
    ax.axvline(0, color="#888", lw=0.6); ax.set_xlabel("aporte al ACE-III (pts vs. promedio)", fontsize=8)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False, labelsize=8.5)
    fig.tight_layout(); st.pyplot(fig)
    st.caption("🟢 sube el ACE estimado (menor riesgo) · 🔴 lo baja (mayor riesgo).")

st.divider()
st.caption(f"Modelo congelado {META['version']} · ElasticNet(α={META['alpha']}, l1={META['l1_ratio']}) + "
           f"isotónica · sklearn {META['sklearn']} · {META['uso_no_previsto']}")
