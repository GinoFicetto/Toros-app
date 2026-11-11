"""
Toros 18 · ¿Qué tan bueno va a ser este toro a los 18 meses?
Aplicación del TP4 de Aprendizaje Automático y Grandes Datos (UNRaf, 2026).

Despliega los tres modelos elegidos en los TP2 y TP3, entrenados con los sumarios de padres 2026
de Braford y Brangus:
  - regresión lineal  -> DEP de peso a los 18 meses (kg)
  - regresión logística -> probabilidad de estar en el 10 % superior (RANK <= 10) a los 18 meses
  - K-Means (k = 3)   -> perfil del toro

Para correrla:  streamlit run app.py
"""
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st

CARPETA = Path(__file__).parent
RUTA_MODELOS = CARPETA / "modelos" / "modelos_toros.joblib"
RUTA_REGISTRO = CARPETA / "registros" / "predicciones.csv"
REPOSITORIO = "https://github.com/GinoFicetto/toros-app"

# Límites duros: valores fuera de esto no tienen sentido para una DEP o un año de nacimiento.
# Los límites "blandos" (lo que se observó en el sumario) se leen del modelo y solo generan un aviso.
LIMITES = {"PesoNacer_DEP": (-8.0, 8.0), "PesoDestete_DEP": (-60.0, 60.0), "AptitudMaterna_DEP": (-30.0, 30.0),
           "CircEscrotal_DEP": (-3.0, 5.0), "Anio_Nacimiento": (1960, date.today().year)}
PASO = {"PesoNacer_DEP": 0.01, "PesoDestete_DEP": 0.1, "AptitudMaterna_DEP": 0.1, "CircEscrotal_DEP": 0.01, "Anio_Nacimiento": 1}
AYUDA = {"PesoNacer_DEP": "Cuántos kg más (o menos) pesarán al nacer sus hijos que el promedio de la raza. Más alto puede traer partos difíciles.",
         "PesoDestete_DEP": "Kg de más (o de menos) de sus hijos al destete, por su genética de crecimiento.",
         "AptitudMaterna_DEP": "Kg de destete que aportan sus hijas como madres (producción de leche).",
         "CircEscrotal_DEP": "Centímetros de circunferencia escrotal de sus hijos; se asocia con fertilidad y precocidad.",
         "Anio_Nacimiento": "Año de nacimiento del toro."}
COLOR_PERFIL = {"Crecimiento": "#0055a5", "Materno": "#f0a800", "Generación anterior": "#2e8b57"}
BORDO, GRIS = "#7a1f1f", "#9ca3af"

st.set_page_config(page_title="Toros 18 · Braford y Brangus", page_icon="🐂", layout="wide")


# ------------------------------------------------------------------ carga del modelo
@st.cache_resource(show_spinner="Cargando los modelos...")
def cargar_modelos():
    """Carga el archivo serializado. Si no se puede abrir (por ejemplo, otra versión de scikit-learn),
    reentrena con el mismo código y los mismos datos, que tarda unos segundos."""
    try:
        paquete = joblib.load(RUTA_MODELOS)
        if paquete.get("version_sklearn") != sklearn.__version__:
            raise ValueError("versión distinta de scikit-learn")
        return paquete, None
    except Exception as error:                                   # noqa: BLE001 (cualquier falla -> reentrenar)
        from entrenar_modelos import entrenar_todo
        return entrenar_todo(CARPETA / "datos" / "toros_braford_brangus.csv"), str(error)


paquete, aviso_carga = cargar_modelos()
PRED, NOMBRES, UMBRAL = paquete["predictoras"], paquete["nombres"], paquete["umbral"]


# ------------------------------------------------------------------ funciones del modelo
def fmt(x, d=1):
    """Número con coma decimal, como se escribe en Argentina."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, d=0):
    return f"{fmt(100 * x, d)} %"


def analizar(tabla, R):
    """Corre los tres modelos sobre una tabla con las 5 predictoras. Devuelve una tabla con los resultados."""
    X = tabla[PRED].astype(float)
    kg = R["regresion"].predict(X)
    prob = R["clasificador"].predict_proba(X)[:, 1]
    Z = R["preparar"].transform(X)
    grupo = [R["nombres_grupo"][g] for g in R["kmeans"].predict(Z)]
    rareza = R["rareza"].score_samples(Z)
    pct_raro = np.array([(R["rareza_ref"] > s).mean() * 100 for s in rareza])       # % de toros de la raza más "normales"
    lo, hi = np.quantile(R["residuos"], [0.10, 0.90])                                # intervalo del 80 % con residuos reales
    return pd.DataFrame({"DEP 18 meses estimada (kg)": kg, "rango probable desde (kg)": kg + lo, "rango probable hasta (kg)": kg + hi,
                         "probabilidad top 10 %": prob, "decisión": [decision(p)[0] for p in prob], "perfil": grupo,
                         "rareza (percentil)": pct_raro}, index=tabla.index)


def decision(p):
    """Traduce la probabilidad a una frase y un color. El corte de 0,30 es el umbral elegido en el TP3."""
    if p >= 0.5:
        return "Muy probable top 10 %", "#2e8b57"
    if p >= UMBRAL:
        return "Candidato: vale la pena mirarlo", "#f0a800"
    return "Poco probable que sea top 10 %", "#c0392b"


def avisos_de_rango(fila, R):
    """Avisos (no errores) cuando un valor está fuera de lo observado en el sumario de la raza."""
    avisos = []
    for c in PRED:
        r = R["rangos"][c]
        if fila[c] < r["min"] or fila[c] > r["max"]:
            d_ = 0 if c == "Anio_Nacimiento" else 2
            avisos.append(f"**{NOMBRES[c]}** = {fmt(fila[c], d_)} está fuera de lo que se vio en el sumario ({fmt(r['min'], d_)} a {fmt(r['max'], d_)}): "
                          "el modelo está extrapolando.")
    return avisos


def registrar(filas, origen, razas):
    """Guarda cada análisis en registros/predicciones.csv (monitoreo básico). Si no se puede escribir, sigue igual."""
    try:
        RUTA_REGISTRO.parent.mkdir(exist_ok=True)
        filas = filas.copy()
        filas.insert(0, "fecha_hora", datetime.now().isoformat(timespec="seconds"))
        filas.insert(1, "origen", origen)
        filas.insert(2, "raza", razas)
        filas.to_csv(RUTA_REGISTRO, mode="a", header=not RUTA_REGISTRO.exists(), index=False)
    except OSError:
        pass


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("🐂 Toros 18")
    raza = st.radio("Raza del toro", ["Braford", "Brangus"], horizontal=True,
                    help="Cada raza tiene su propio modelo: las DEP de una raza no se comparan con las de la otra.")
    R = paquete["razas"][raza]
    origen = st.radio("¿De dónde salen los datos?", ["Cargar a mano", "Elegir un toro del sumario 2026"])
    toro_real = None
    if origen == "Elegir un toro del sumario 2026":
        ref = R["referencia"]
        nombre = st.selectbox("Toro", sorted(ref["Nombre"].unique()), help="Escribí parte del nombre para buscarlo.")
        toro_real = ref[ref["Nombre"] == nombre].iloc[0]
    st.divider()
    mr, mc = R["metricas_regresion"], R["metricas_clasificacion"]
    st.caption(f"**Modelos {raza}** (evaluados con toros que no vieron al entrenarse)")
    st.caption(f"• Kilos: error típico ±{fmt(mr['RMSE prueba'])} kg (R² {fmt(mr['R² prueba'], 2)})  \n"
               f"• Top 10 %: AUC {fmt(mc['AUC prueba'], 2)}; con el umbral 0,30 encuentra el "
               f"{pct(mc[f'umbral {UMBRAL}']['exhaustividad'])} de los toros top")
    st.caption(f"Datos: {R['n']} toros {raza} · scikit-learn {sklearn.__version__}")
    if aviso_carga:
        st.info("Los modelos se reentrenaron al iniciar porque el archivo guardado no se pudo usar en este equipo "
                f"({aviso_carga}). Los resultados son los mismos.")

st.title("¿Qué tan bueno va a ser este toro a los 18 meses?")
st.markdown(
    "Con las DEP que ya se conocen de un toro joven (nacimiento, destete, aptitud materna y circunferencia escrotal) "
    "la aplicación estima **su DEP de peso a los 18 meses**, la **probabilidad de que quede en el 10 % superior de su raza** "
    "y **a qué tipo de toro se parece**. Sirve como primer filtro para elegir pajuelas o toros antes de que su DEP de "
    "18 meses tenga una precisión alta.")

tab_uno, tab_lote, tab_info, tab_reg = st.tabs(["🐂 Analizar un toro", "📄 Muchos toros (CSV)", "ℹ️ Cómo funciona", "📝 Registro"])

# ================================================================== 1. UN TORO
with tab_uno:
    clave = f"{raza}_{toro_real['Nombre'] if toro_real is not None else 'mano'}"      # cambia el toro -> se recargan los valores
    base = {c: (toro_real[c] if toro_real is not None and pd.notna(toro_real[c]) else R["rangos"][c]["mediana"]) for c in PRED}
    with st.form("formulario"):
        st.subheader("Datos del toro")
        if toro_real is not None:
            st.caption(f"Valores de **{toro_real['Nombre']}** según el sumario 2026. Podés cambiarlos.")
        else:
            st.caption("Arranca con los valores de un toro típico de la raza (la mediana). Cargá los del toro que quieras analizar.")
        cols = st.columns(5)
        valores = {}
        for col, c in zip(cols, PRED):
            lo, hi = LIMITES[c]
            if c == "Anio_Nacimiento":
                valores[c] = col.number_input(NOMBRES[c], min_value=int(lo), max_value=int(hi), value=int(min(max(base[c], lo), hi)),
                                              step=1, help=AYUDA[c], key=f"{c}_{clave}")
            else:
                valores[c] = col.number_input(NOMBRES[c], min_value=lo, max_value=hi, value=float(np.clip(base[c], lo, hi)),
                                              step=PASO[c], format="%.2f", help=AYUDA[c], key=f"{c}_{clave}")
        enviado = st.form_submit_button("Analizar el toro", type="primary", width="stretch")

    if not enviado and "ultimo" not in st.session_state:
        st.info("Cargá los datos y tocá **Analizar el toro**.")
    if enviado:
        st.session_state["ultimo"] = {"valores": valores, "raza": raza, "toro": None if toro_real is None else toro_real["Nombre"]}
        fila_log = pd.DataFrame([valores])
        registrar(pd.concat([fila_log, analizar(fila_log, R)], axis=1), "formulario", raza)

    ultimo = st.session_state.get("ultimo")
    if ultimo and ultimo["raza"] == raza:
        fila = pd.DataFrame([ultimo["valores"]])
        res = analizar(fila, R).iloc[0]
        texto, color = decision(res["probabilidad top 10 %"])
        perfil = res["perfil"]; pf = R["perfiles"].loc[perfil]
        pct_dist = (R["referencia"]["Peso18Meses_DEP"] < res["DEP 18 meses estimada (kg)"]).mean() * 100

        st.subheader("Resultado")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("DEP de peso a los 18 meses (estimada)", f"{fmt(res['DEP 18 meses estimada (kg)'])} kg")
            st.caption(f"Rango probable: **{fmt(res['rango probable desde (kg)'])} a {fmt(res['rango probable hasta (kg)'])} kg** "
                       f"(8 de cada 10 toros quedan dentro). Supera al {fmt(pct_dist, 0)} % de los toros {raza} del sumario.")
        with c2:
            st.metric("Probabilidad de estar en el top 10 %", pct(res['probabilidad top 10 %']))
            st.markdown(f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;font-weight:600'>{texto}</span>",
                        unsafe_allow_html=True)
            st.caption(f"En la raza, el {pct(R['tasa_top'])} de los toros del sumario está en el top 10 %.")
        with c3:
            st.metric("Tipo de toro", perfil)
            st.caption(f"En ese grupo hay {int(pf['toros'])} toros del sumario y el {pct(pf['pct_top'])} es top 10 %.")

        # --- confiabilidad
        avisos = avisos_de_rango(fila.iloc[0], R)
        if res["rareza (percentil)"] >= 98:
            avisos.append(f"La combinación de DEP es **muy poco común**: es más rara que el {fmt(res['rareza (percentil)'], 0)} % de los toros "
                          f"{raza}. El modelo tiene pocos toros parecidos para aprender, así que el resultado es menos confiable.")
        for a in avisos:
            st.warning(a, icon="⚠️")
        if not avisos:
            st.success("Los valores están dentro de lo observado en el sumario y la combinación es habitual: el resultado es tan "
                       "confiable como dicen las métricas del modelo.", icon="✅")
        if ultimo["toro"]:
            real = R["referencia"][R["referencia"]["Nombre"] == ultimo["toro"]].iloc[0]
            st.info(f"**Valor real de {ultimo['toro']}** en el sumario: DEP 18 meses **{fmt(real['Peso18Meses_DEP'])} kg**, RANK "
                    f"{int(real['Peso18Meses_RANK'])} ({'top 10 %' if real['top10'] else 'no top 10 %'}). Ojo: este toro se usó para "
                    "entrenar el modelo, así que la comparación es optimista.", icon="📘")

        # --- gráficos de contexto
        g1, g2 = st.columns([1.35, 1])
        with g1:
            dist = R["referencia"]["Peso18Meses_DEP"]
            fig = px.histogram(dist, nbins=45, color_discrete_sequence=["#d6d3d1"])
            fig.add_vrect(x0=res["rango probable desde (kg)"], x1=res["rango probable hasta (kg)"], fillcolor=BORDO, opacity=0.12, line_width=0)
            fig.add_vline(x=res["DEP 18 meses estimada (kg)"], line_color=BORDO, line_width=3,
                          annotation_text="este toro", annotation_position="top")
            corte = R["referencia"].loc[R["referencia"]["top10"] == 1, "Peso18Meses_DEP"].min()
            fig.add_vline(x=corte, line_dash="dot", line_color="#555", annotation_text="desde acá, top 10 %", annotation_position="bottom right")
            fig.update_layout(title=f"Dónde cae entre los {R['n']} toros {raza} del sumario", showlegend=False, height=330, separators=",.",
                              xaxis_title="DEP de peso a los 18 meses (kg)", yaxis_title="toros", margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
        with g2:
            p = res["probabilidad top 10 %"]
            fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": " %", "valueformat": ".0f"},
                                         gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                                "steps": [{"range": [0, UMBRAL * 100], "color": "#f5d5d0"},
                                                          {"range": [UMBRAL * 100, 50], "color": "#fdeccc"},
                                                          {"range": [50, 100], "color": "#d5ecdc"}],
                                                "threshold": {"line": {"color": "#333", "width": 3}, "value": UMBRAL * 100}}))
            fig.update_layout(title="Probabilidad de estar en el top 10 %", height=330, margin=dict(t=60, b=10, l=30, r=30), separators=",.")
            st.plotly_chart(fig, width="stretch")
            st.caption("La raya negra marca el umbral de 30 %: desde ahí el toro se considera candidato.")

        # --- explicabilidad: aporte de cada DEP
        with st.expander("¿Por qué da este resultado? (aporte de cada dato)", expanded=True):
            z = R["preparar"].transform(fila[PRED].astype(float))[0]
            aporte_kg = np.array([R["coef_regresion"][c] for c in PRED]) * z
            aporte_log = np.array([R["coef_logistica"][c] for c in PRED]) * z
            base_kg = float(R["regresion"].named_steps["modelo"].intercept_)
            tabla_ap = pd.DataFrame({"dato": [NOMBRES[c] for c in PRED], "kg": aporte_kg,
                                     "chances": np.exp(aporte_log)}).sort_values("kg")
            fig = px.bar(tabla_ap, x="kg", y="dato", orientation="h", text=tabla_ap["kg"].map(lambda v: ("+" if v >= 0 else "−") + fmt(abs(v)) + " kg"),
                         color=np.where(tabla_ap["kg"] >= 0, "suma", "resta"), color_discrete_map={"suma": "#2e8b57", "resta": "#c0392b"})
            fig.update_layout(showlegend=False, height=300, xaxis_title="kg que suma o resta frente a un toro promedio de la raza",
                              yaxis_title=None, margin=dict(t=10, b=10, l=10, r=10))
            fig.update_layout(separators=",.")
            st.plotly_chart(fig, width="stretch")
            mayor = tabla_ap.iloc[tabla_ap["kg"].abs().argmax()]
            st.markdown(f"Un toro promedio de la raza tiene una DEP de 18 meses de **{fmt(base_kg)} kg**. A este toro se le suman o restan los "
                        f"aportes de cada dato; el que más pesa es la **{mayor['dato'].split(' (')[0]}** "
                        f"({'+' if mayor['kg'] >= 0 else '−'}{fmt(abs(mayor['kg']))} kg). "
                        "En la probabilidad de top 10 %, cada dato multiplica las chances frente al toro promedio: "
                        + ", ".join(f"{r.dato.split(' (')[0]} ×{fmt(r.chances, 2)}" for r in tabla_ap.sort_values('chances', ascending=False).itertuples()) + ".")
            st.caption("Los dos modelos son lineales, así que esta descomposición es exacta (coeficiente × valor estandarizado).")

# ================================================================== 2. MUCHOS TOROS
with tab_lote:
    st.subheader("Analizar muchos toros con un archivo CSV")
    st.markdown("El archivo tiene que tener una fila por toro y estas columnas (los nombres tal cual): "
                + ", ".join(f"`{c}`" for c in PRED) + ". Puede tener además `Nombre` y `Raza`; si no tiene `Raza`, se usa la elegida "
                "en la barra lateral. Los datos faltantes se completan con la mediana de la raza y se avisa.")
    plantilla = R["referencia"][["Nombre"] + PRED].head(5).assign(Raza=raza)
    st.download_button("Descargar un archivo de ejemplo", plantilla.to_csv(index=False).encode("utf-8"), "toros_ejemplo.csv", "text/csv")
    archivo = st.file_uploader("Subir el CSV", type=["csv"], help="Hasta 5000 toros. Separador coma o punto y coma.")
    if archivo is not None:
        try:
            crudo = archivo.getvalue().decode("utf-8-sig")
            lote = pd.read_csv(StringIO(crudo), sep=None, engine="python")
        except Exception as error:                                       # noqa: BLE001
            st.error(f"No se pudo leer el archivo como CSV ({error}).")
            lote = None
        if lote is not None:
            faltan = [c for c in PRED if c not in lote.columns]
            if faltan:
                st.error("Faltan columnas: " + ", ".join(f"`{c}`" for c in faltan) + ". Descargá el archivo de ejemplo para ver el formato.")
            elif len(lote) == 0 or len(lote) > 5000:
                st.error(f"El archivo tiene {len(lote)} filas; tiene que tener entre 1 y 5000.")
            else:
                obs = pd.Series("", index=lote.index)
                if "Raza" not in lote.columns:
                    lote["Raza"] = raza
                lote["Raza"] = lote["Raza"].astype(str).str.strip().str.capitalize()
                mala_raza = ~lote["Raza"].isin(["Braford", "Brangus"])
                obs[mala_raza] += "raza desconocida; "
                for c in PRED:
                    original = lote[c]
                    if pd.api.types.is_numeric_dtype(original):
                        num = original.astype(float)
                        escrito = original.notna()
                    else:                                                # texto: admite coma decimal
                        texto = original.astype("string").str.strip()
                        escrito = texto.notna() & (texto != "")
                        num = pd.to_numeric(texto.str.replace(",", ".", regex=False), errors="coerce").astype(float)
                    obs[escrito & num.isna()] += f"{c} no es un número; "
                    fuera = (num < LIMITES[c][0]) | (num > LIMITES[c][1])
                    obs[fuera] += f"{c} fuera de rango; "
                    lote[c] = num.mask(fuera)
                invalida = mala_raza | obs.str.contains("no es un número|fuera de rango")
                partes = []
                for rz in ["Braford", "Brangus"]:
                    sel = (lote["Raza"] == rz) & ~invalida
                    if sel.any():
                        Rz = paquete["razas"][rz]
                        sub = lote.loc[sel, PRED].copy()
                        for c in PRED:
                            obs[sel & lote[c].isna()] += f"{c} faltante (se usó la mediana); "
                        r_ = analizar(sub, Rz)
                        r_.loc[r_["rareza (percentil)"] >= 98, "perfil"] += " (combinación muy poco común)"
                        partes.append(r_)
                salida = lote.copy()
                if partes:
                    salida = salida.join(pd.concat(partes))
                salida.insert(1 if "Nombre" in salida else 0, "observaciones", obs.str.rstrip("; "))
                n_ok = int((~invalida).sum())
                aviso = st.warning if invalida.any() else st.success
                aviso(f"Se analizaron {n_ok} de {len(lote)} toros." + (f" {int(invalida.sum())} filas tienen errores y no se analizaron "
                                                                      "(ver la columna observaciones)." if invalida.any() else ""))
                vista = salida.copy()
                if "probabilidad top 10 %" in vista:
                    vista["probabilidad top 10 %"] = (vista["probabilidad top 10 %"] * 100).round(0)
                    vista = vista.rename(columns={"probabilidad top 10 %": "probabilidad top 10 % (%)"})
                st.dataframe(vista.round(1), width="stretch", hide_index=True)
                st.download_button("Descargar los resultados", salida.to_csv(index=False).encode("utf-8"),
                                   "toros_resultados.csv", "text/csv", type="primary")
                if partes:
                    ok = salida[~invalida]
                    fig = px.scatter(ok, x="DEP 18 meses estimada (kg)", y="probabilidad top 10 %", color=ok["perfil"].str.replace(" (combinación muy poco común)", "", regex=False),
                                     color_discrete_map=COLOR_PERFIL, hover_name="Nombre" if "Nombre" in ok else None,
                                     labels={"color": "perfil"})
                    fig.add_hline(y=UMBRAL, line_dash="dot", annotation_text="umbral 30 %")
                    fig.update_layout(height=380, yaxis_tickformat=".0%", separators=",.", title="Los toros del archivo: kilos estimados y probabilidad de top 10 %")
                    st.plotly_chart(fig, width="stretch")
                    ok_log = salida.loc[~invalida]
                    registrar(ok_log[PRED + ["DEP 18 meses estimada (kg)", "probabilidad top 10 %", "perfil"]], "csv", ok_log["Raza"].values)

# ================================================================== 3. CÓMO FUNCIONA
with tab_info:
    st.subheader("Cómo funciona")
    st.markdown(f"""
**El problema.** La DEP (diferencia esperada en la progenie) de peso a los 18 meses dice cuántos kg más que el promedio de la
raza pesarán los hijos de un toro a esa edad. En un toro joven esa DEP tiene poca precisión, porque todavía tiene pocos hijos
pesados a los 18 meses. En cambio, las DEP de nacimiento, destete, aptitud materna y circunferencia escrotal se conocen antes.
La aplicación usa esas DEP para anticipar la de 18 meses.

**Los datos.** Sumarios de padres 2026 de la Asociación Braford Argentina (1106 toros) y de la Asociación Argentina de Brangus
(1223 toros). Cada raza tiene su propio modelo.

**Los modelos.** Son los que mejor rindieron en los trabajos anteriores, comparados contra k vecinos y Random Forest:
""")
    filas = []
    for rz, Rz in paquete["razas"].items():
        a, b = Rz["metricas_regresion"], Rz["metricas_clasificacion"]
        filas.append({"raza": rz, "kilos: error típico (RMSE, kg)": fmt(a["RMSE prueba"], 2), "kilos: R²": fmt(a["R² prueba"], 3),
                      "top 10 %: AUC": fmt(b["AUC prueba"], 3),
                      "top 10 % con umbral 0,30: encuentra": pct(b[f"umbral {UMBRAL}"]["exhaustividad"]),
                      "top 10 % con umbral 0,30: acierta cuando dice candidato": pct(b[f"umbral {UMBRAL}"]["precisión"])})
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    st.markdown(f"""
- **Kilos:** regresión lineal múltiple. Fue la mejor en validación cruzada; Random Forest y k vecinos no la superaron.
- **Top 10 %:** regresión logística. Mejor AUC en validación cruzada. El umbral se bajó de 0,50 a **0,30** porque, para elegir
  pajuelas, perder un buen toro cuesta más que revisar uno de más.
- **Tipo de toro:** K-Means con 3 grupos (crecimiento, materno y generación anterior), el mejor de tres algoritmos de agrupamiento.
- Las métricas son sobre el 20 % de los toros que el modelo no vio. Para la aplicación, los modelos se reentrenaron con todos los
  toros de cada raza, con los mismos hiperparámetros.

**Qué no hace.** No reemplaza la DEP oficial ni la evaluación genética de la asociación: es un primer filtro. Con un toro de una
combinación muy poco común o con valores fuera de lo observado, la aplicación avisa que el resultado es menos confiable. Los
datos son de un solo año de sumario (2026); si cambia la evaluación genética, los modelos hay que reentrenarlos.

**Trabajo Práctico 4** · Aprendizaje Automático y Grandes Datos · Licenciatura en Agroinformática · UNRaf · 2026.
Autores: Gino Ficetto y Bautista Ferrer.
""" + (f"Código y datos: [{REPOSITORIO}]({REPOSITORIO})." if "USUARIO" not in REPOSITORIO else ""))

# ================================================================== 4. REGISTRO
with tab_reg:
    st.subheader("Registro de análisis (monitoreo)")
    st.caption("Cada análisis se guarda con fecha, datos de entrada y resultado. Sirve para ver cómo se usa la aplicación y si los "
               "toros que se cargan se parecen a los del entrenamiento (deriva de datos). En Streamlit Community Cloud el registro "
               "se borra cuando la aplicación se reinicia.")
    if RUTA_REGISTRO.exists():
        log = pd.read_csv(RUTA_REGISTRO)
        a, b, c = st.columns(3)
        a.metric("Toros analizados", len(log))
        b.metric("Desde el formulario", int((log["origen"] == "formulario").sum()))
        c.metric("Desde archivos CSV", int((log["origen"] == "csv").sum()))
        ref = paquete["razas"][raza]["referencia"]
        log_raza = log[log["raza"] == raza]
        deriva = pd.DataFrame({"dato": [NOMBRES[c] for c in PRED],
                               "media en el sumario": [ref[c].mean() for c in PRED],
                               "media de lo analizado": [pd.to_numeric(log_raza[c], errors="coerce").mean() for c in PRED]})
        deriva["diferencia (en desvíos)"] = [(m - ref[c].mean()) / ref[c].std() for m, c in zip(deriva["media de lo analizado"], PRED)]
        deriva["¿se aleja?"] = np.where(deriva["diferencia (en desvíos)"].abs() > 0.5, "sí, revisar", "no")
        st.markdown(f"**Deriva de los datos de entrada**: los {len(log_raza)} toros {raza} analizados frente al sumario "
                    "(más de medio desvío de diferencia es señal de alerta):")
        st.dataframe(deriva.round(2), hide_index=True, width="stretch")
        st.dataframe(log.tail(50).iloc[::-1].round(2), hide_index=True, width="stretch")
        st.download_button("Descargar el registro", log.to_csv(index=False).encode("utf-8"), "registro_predicciones.csv", "text/csv")
    else:
        st.info("Todavía no hay análisis registrados.")
