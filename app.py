import io, zipfile, urllib.request, datetime as dt
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DimensioVision AI", page_icon="📉", layout="wide")
st.markdown("""<style>
.stApp{background:#F8FAFC} [data-testid=stSidebar]{background:#0F172A}
[data-testid=stSidebar] *{color:#E2E8F0 !important}
h1,h2,h3{color:#0F172A} div[data-testid=stMetric]{background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:14px;box-shadow:0 1px 3px #0001}
.stButton>button,.stDownloadButton>button{background:#2563EB;color:#fff;border:0;border-radius:10px;font-weight:600}
.stButton>button:hover{background:#38BDF8;color:#0F172A}
.hero{background:linear-gradient(120deg,#0F172A,#2563EB);color:#fff;padding:28px;border-radius:16px}
.hero h1{color:#fff;margin:0}</style>""", unsafe_allow_html=True)

DESC = {"fixed acidity": "Acidez fija (ácido tartárico, g/dm³)", "volatile acidity": "Acidez volátil (ácido acético, g/dm³)",
 "citric acid": "Ácido cítrico (g/dm³)", "residual sugar": "Azúcar residual (g/dm³)", "chlorides": "Cloruros (cloruro de sodio, g/dm³)",
 "free sulfur dioxide": "Dióxido de azufre libre (mg/dm³)", "total sulfur dioxide": "Dióxido de azufre total (mg/dm³)",
 "density": "Densidad (g/cm³)", "pH": "Nivel de pH", "sulphates": "Sulfatos (g/dm³)", "alcohol": "Alcohol (% vol.)",
 "quality": "Calidad sensorial (0–10), etiqueta de referencia", "tipo": "Tipo de vino (tinto/blanco), añadido al descargar"}
UCI = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
SS = st.session_state

@st.cache_data(show_spinner=False)
def download_wine(kind):
    req = urllib.request.Request(UCI, headers={"User-Agent": "Mozilla/5.0"})
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=30).read()))
    frames = []
    for t, f in (("red", "winequality-red.csv"), ("white", "winequality-white.csv")):
        if kind in (t, "both"):
            d = pd.read_csv(z.open(f), sep=";"); d["tipo"] = t; frames.append(d)
    return pd.concat(frames, ignore_index=True)

def need(ok, msg):
    if not ok: st.warning(msg); st.stop()

# ---------- figuras ----------
def fig_var(evr, k):
    pcs = [f"PC{i+1}" for i in range(len(evr))]; cum = np.cumsum(evr) * 100
    f = go.Figure()
    f.add_bar(x=pcs, y=evr * 100, name="Varianza explicada (%)", marker_color=["#2563EB" if i < k else "#CBD5E1" for i in range(len(evr))])
    f.add_scatter(x=pcs, y=cum, name="Acumulada (%)", mode="lines+markers", line=dict(color="#38BDF8", width=3))
    f.update_layout(template="plotly_white", yaxis_title="%", xaxis_title="Componente", legend=dict(orientation="h"))
    return f

def fig_dim(d0, k):
    f = px.bar(x=["Original (variables)", "Reducida (componentes)"], y=[d0, k], color=["a", "b"],
               color_discrete_sequence=["#0F172A", "#38BDF8"], text=[d0, k])
    f.update_layout(template="plotly_white", showlegend=False, xaxis_title="", yaxis_title="Nº de columnas"); return f

def fig_scatter(scores, meta, color, px_, py_, feats_df):
    d = scores[[px_, py_]].copy()
    d["Registro"] = d.index
    hov = {}
    if color and color != "(ninguno)":
        c = meta[color]
        d[color] = c.astype(str) if (not pd.api.types.is_numeric_dtype(c) or c.nunique() <= 12) else c
        hov[color] = True
    for col in feats_df.columns: d[col] = feats_df[col].round(3)
    f = px.scatter(d, x=px_, y=py_, color=color if color and color != "(ninguno)" else None,
                   hover_data={"Registro": True, **hov, **{c: True for c in feats_df.columns}},
                   color_discrete_sequence=px.colors.qualitative.Safe, opacity=.7)
    f.update_layout(template="plotly_white"); return f

def fig_heat(L, feats):
    f = px.imshow(L, x=list(L.columns), y=feats, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto", text_auto=".2f")
    f.update_layout(coloraxis_colorbar_title="Carga"); return f

def top_var(R, pc):
    return R["loadings"][pc].abs().idxmax()

def local_wine():
    import os
    d = os.path.dirname(os.path.abspath(__file__)); fr = []
    for t in ("red", "white"):
        p = os.path.join(d, f"winequality-{t}.csv")
        if os.path.exists(p):
            x = pd.read_csv(p, sep=";"); x["tipo"] = t; fr.append(x)
    return pd.concat(fr, ignore_index=True) if fr else None

def compute_pca(k):
    P = SS["prep"]; maxk = min(len(SS["feats"]), len(P["X"]))
    pca = PCA(n_components=maxk, random_state=0).fit(P["Xs"])
    pcs = [f"PC{i+1}" for i in range(maxk)]
    SS["res"] = dict(k=k, evr=pca.explained_variance_ratio_, eig=pca.explained_variance_,
        loadings=pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=SS["feats"], columns=pcs),
        contrib=pd.DataFrame(pca.components_.T ** 2 * 100, index=SS["feats"], columns=pcs),
        scores=pd.DataFrame(pca.transform(P["Xs"])[:, :k], columns=pcs[:k], index=P["X"].index),
        cum_k=float(pca.explained_variance_ratio_[:k].sum() * 100))

def auto_run():
    """Carga el dataset y ejecuta preprocesamiento + PCA con valores por defecto (datos reales)."""
    try:
        if "df" not in SS:
            d = local_wine()
            if d is not None: SS["src"] = "Wine Quality – UCI ML Repository (archivos CSV del proyecto)"
            else: d = download_wine("both").copy(); SS["src"] = "Wine Quality – UCI ML Repository (tinto + blanco, descarga automática)"
            SS["df"] = d
        df = SS["df"]
        if "prep" not in SS:
            label = "quality" if "quality" in df.columns else "(ninguna)"
            feats = [c for c in df.select_dtypes("number").columns if c != label]
            if len(feats) < 2: return
            X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[feats]), columns=feats, index=df.index)
            if (X.std() == 0).any(): return
            SS.update(prep=dict(X=X, Xs=StandardScaler().fit_transform(X), label=label, std=True, imp="Imputar con mediana", dropped=0), feats=feats)
            SS.pop("res", None)
        if "res" not in SS:
            ev = PCA(random_state=0).fit(SS["prep"]["Xs"]).explained_variance_ratio_
            compute_pca(int(max(2, np.argmax(np.cumsum(ev) >= 0.8) + 1)))   # k mínimo que explica ≥80%
    except Exception as e:
        SS["load_err"] = str(e)

# ---------- páginas ----------
def go_to(p): SS["page"] = p

def page_inicio():
    st.markdown("<div class='hero'><h1>📉 DimensioVision AI</h1><p>Plataforma web de reducción de dimensionalidad mediante Machine Learning. "
                "Aplica Análisis de Componentes Principales (PCA) sobre datos reales, visualiza la varianza explicada y compara los datos antes y después.</p></div>", unsafe_allow_html=True)
    st.write("")
    df = SS.get("df"); R = SS.get("res")
    c = st.columns(4)
    c[0].metric("Registros", f"{len(df):,}" if df is not None else "—")
    c[1].metric("Variables", df.shape[1] if df is not None else "—")
    c[2].metric("Variables en el análisis", len(SS["feats"]) if "prep" in SS else "—")
    c[3].metric("Varianza explicada", f"{R['cum_k']:.1f}%" if R else "—")
    st.write("")
    st.button("🚀 Iniciar análisis", on_click=go_to, args=("5. Visualización",), type="primary")
    if R:
        st.success("Dataset cargado y PCA ejecutado automáticamente con datos reales. Resumen:")
        for x in conclusions(): st.markdown(f"- {x}")
        st.plotly_chart(fig_var(R["evr"], R["k"]), use_container_width=True)
    elif SS.get("load_err"):
        st.error(f"No se pudo cargar el dataset automáticamente ({SS['load_err']}). Ve a *Dataset* para cargar un CSV.")
    st.caption("Universidad Nacional de Trujillo · Ingeniería de Sistemas · Dataset: Wine Quality (UCI)")

def page_dataset():
    st.header("Dataset")
    src = st.radio("Fuente de datos", ["Wine Quality (UCI, descarga)", "Cargar mi CSV"], horizontal=True)
    if src.startswith("Wine"):
        kind = st.selectbox("Subconjunto", ["both", "red", "white"], format_func={"both": "Tinto + blanco (6 497)", "red": "Tinto (1 599)", "white": "Blanco (4 898)"}.get)
        if st.button("Descargar dataset de UCI"):
            try:
                with st.spinner("Descargando..."):
                    SS["df"] = download_wine(kind).copy(); SS["src"] = f"Wine Quality – UCI ML Repository ({kind})"
                SS.pop("prep", None); SS.pop("res", None)
            except Exception as e:
                st.error(f"No se pudo descargar ({e}). Revisa tu conexión o carga los CSV manualmente.")
    else:
        up = st.file_uploader("Archivo CSV", type=["csv"])
        if up is not None:
            try:
                d = pd.read_csv(up, sep=None, engine="python")
                if d.shape[0] < 3 or d.shape[1] < 2: raise ValueError("El archivo debe tener al menos 3 filas y 2 columnas.")
                SS["df"] = d; SS["src"] = f"Archivo cargado: {up.name}"; SS.pop("prep", None); SS.pop("res", None)
            except Exception as e:
                st.error(f"Error al leer el CSV: {e}")
    df = SS.get("df")
    need(df is not None, "Aún no hay datos. Descarga el dataset o carga un CSV.")
    st.success(f"**Fuente:** {SS['src']}")
    c = st.columns(3); c[0].metric("Registros", f"{len(df):,}"); c[1].metric("Variables", df.shape[1]); c[2].metric("Valores faltantes", int(df.isna().sum().sum()))
    st.subheader("Vista previa"); st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Columnas")
    st.dataframe(pd.DataFrame({"Columna": df.columns, "Tipo": df.dtypes.astype(str).values,
        "Descripción": [DESC.get(c, "—") for c in df.columns], "Faltantes": df.isna().sum().values}), use_container_width=True, hide_index=True)
    st.subheader("Estadísticas descriptivas"); st.dataframe(df.describe().T, use_container_width=True)

def page_prep():
    st.header("Preprocesamiento")
    df = SS.get("df"); need(df is not None, "Primero carga un dataset en la sección Dataset.")
    num = df.select_dtypes("number").columns.tolist()
    need(len(num) >= 2, "El dataset necesita al menos 2 columnas numéricas.")
    ids = [c for c in num if c.lower() in ("id", "index", "unnamed: 0")]
    label = st.selectbox("Variable de referencia (se excluye del cálculo; solo para colorear)", ["(ninguna)"] + list(df.columns),
                         index=(list(df.columns).index("quality") + 1) if "quality" in df.columns else 0)
    cand = [c for c in num if c != label and c not in ids]
    feats = st.multiselect("Variables numéricas de entrada", cand, default=cand)
    c1, c2 = st.columns(2)
    imp = c1.selectbox("Valores faltantes", ["Imputar con mediana", "Imputar con media", "Eliminar filas con faltantes"])
    std = c2.checkbox("Estandarizar (media 0, desviación 1) — recomendado", value=True)
    if st.button("Aplicar preprocesamiento"):
        if len(feats) < 2: st.error("Selecciona al menos 2 variables."); return
        X = df[feats].copy()
        if imp.startswith("Eliminar"): X = X.dropna()
        else: X = pd.DataFrame(SimpleImputer(strategy="median" if "mediana" in imp else "mean").fit_transform(X), columns=feats, index=X.index)
        if len(X) < 3: st.error("Quedan menos de 3 registros tras el tratamiento de faltantes."); return
        const = [c for c in feats if X[c].std() == 0]
        if const: st.error(f"Variables constantes (sin varianza): {const}. Quítalas de la selección."); return
        Xs = StandardScaler().fit_transform(X) if std else X.values - X.values.mean(0)
        SS.update(prep=dict(X=X, Xs=Xs, label=label, std=std, imp=imp, dropped=len(df) - len(X)), feats=feats); SS.pop("res", None)
    P = SS.get("prep")
    if P:
        st.success(f"Listo: {len(P['X'])} registros × {len(SS['feats'])} variables. Filas eliminadas: {P['dropped']}.")
        st.dataframe(pd.DataFrame(P["Xs"], columns=SS["feats"], index=P["X"].index).head(10).round(3), use_container_width=True)

def page_pca():
    st.header("Reducción de dimensionalidad (PCA)")
    P = SS.get("prep"); need(P is not None, "Primero aplica el preprocesamiento.")
    n = len(SS["feats"]); maxk = min(n, len(P["X"]))
    k = st.slider("Número de componentes principales", 2, maxk, SS["res"]["k"] if "res" in SS else min(2, maxk)) if maxk > 2 else 2
    if st.button("Ejecutar PCA"): compute_pca(k)
    R = SS.get("res")
    if R:
        st.success(f"PCA ejecutado con {R['k']} componentes sobre {n} variables. Varianza explicada acumulada: **{R['cum_k']:.2f}%**")
        st.dataframe(R["scores"].head(10).round(4), use_container_width=True)
        st.caption("Ver gráficos en *Visualización*.")

def page_viz():
    st.header("Visualización"); R = SS.get("res"); need(R is not None, "Ejecuta PCA primero.")
    P = SS["prep"]; k = R["k"]; d0 = len(SS["feats"])
    t = st.tabs(["Varianza explicada", "Dimensiones", "Proyección 2D", "Contribución de variables"])
    with t[0]:
        st.plotly_chart(fig_var(R["evr"], k), use_container_width=True)
        st.info(f"Los {k} componentes seleccionados (azul) capturan **{R['cum_k']:.2f}%** de la varianza total. Eigenvalores: " + ", ".join(f"{v:.2f}" for v in R["eig"][:k]))
    with t[1]:
        st.plotly_chart(fig_dim(d0, k), use_container_width=True)
        st.metric("Reducción dimensional", f"{(d0 - k) / d0 * 100:.1f}%")
    with t[2]:
        pcs = list(R["scores"].columns); c = st.columns(3)
        px_ = c[0].selectbox("Eje X", pcs, 0); py_ = c[1].selectbox("Eje Y", pcs, 1)
        meta = SS["df"].loc[R["scores"].index]
        opts = ["(ninguno)"] + [x for x in meta.columns if x not in SS["feats"] or x == P["label"]]
        color = c[2].selectbox("Colorear por", opts, index=opts.index(P["label"]) if P["label"] in opts else 0)
        st.plotly_chart(fig_scatter(R["scores"], meta, color, px_, py_, P["X"]), use_container_width=True)
        for pc in (px_, py_):
            s = R["loadings"][pc].sort_values(key=abs, ascending=False)
            st.markdown(f"**{pc}** ({R['evr'][int(pc[2:])-1]*100:.1f}% de la varianza): combinación lineal de las variables; sus mayores cargas son "
                        f"`{s.index[0]}` ({s.iloc[0]:+.2f}) y `{s.index[1]}` ({s.iloc[1]:+.2f}). Cada punto es un registro proyectado sobre estos ejes.")
    with t[3]:
        st.plotly_chart(fig_heat(R["loadings"].iloc[:, :k], SS["feats"]), use_container_width=True)
        pc = st.selectbox("Componente a detallar", list(R["scores"].columns))
        s = R["loadings"][pc].sort_values(key=abs)
        f = px.bar(x=s.values, y=s.index, orientation="h", color=s.values, color_continuous_scale="RdBu", range_color=[-1, 1])
        f.update_layout(template="plotly_white", xaxis_title=f"Carga en {pc}", yaxis_title=""); st.plotly_chart(f, use_container_width=True)
        st.dataframe(R["contrib"][pc].sort_values(ascending=False).rename("Contribución (%)").round(2).to_frame(), use_container_width=True)
        st.caption("Carga = autovector × √autovalor; con datos estandarizados equivale a la correlación entre la variable y el componente. "
                   "Contribución (%) = (coeficiente del autovector)² × 100: cuánto aporta cada variable a la definición del componente.")

def page_cmp():
    st.header("Comparación de resultados"); R = SS.get("res"); need(R is not None, "Ejecuta PCA primero.")
    P = SS["prep"]; d0 = len(SS["feats"]); k = R["k"]; red = (d0 - k) / d0 * 100
    c = st.columns(5); c[0].metric("D original", d0); c[1].metric("D reducida", k); c[2].metric("Componentes", k)
    c[3].metric("Varianza acumulada", f"{R['cum_k']:.2f}%"); c[4].metric("Reducción", f"{red:.1f}%")
    st.latex(rf"\text{{Reducción}}=\frac{{{d0}-{k}}}{{{d0}}}\times100={red:.2f}\%")
    st.warning("Reducir dimensiones no implica conservar toda la información. La varianza explicada mide la variabilidad capturada, "
               "no garantiza conservar la información útil para cualquier tarea (p. ej. clasificar).")
    n = st.slider("Registros de ejemplo", 5, 50, 10)
    st.markdown("**Original → nuevas coordenadas**")
    comp = pd.concat([P["X"].head(n).round(3), R["scores"].head(n).round(3)], axis=1); st.dataframe(comp, use_container_width=True)
    out = pd.concat([P["X"].add_prefix("orig_"), R["scores"]], axis=1)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Descargar componentes (CSV)", R["scores"].to_csv(index_label="registro").encode(), "pca_componentes.csv", "text/csv")
    c2.download_button("⬇️ Descargar originales + componentes (CSV)", out.to_csv(index_label="registro").encode(), "pca_originales_y_componentes.csv", "text/csv")

def conclusions():
    R = SS["res"]; k = R["k"]; d0 = len(SS["feats"]); cum = np.cumsum(R["evr"]) * 100
    def need_k(p): return int(np.argmax(cum >= p) + 1) if cum[-1] >= p else d0
    t = [f"Se redujo el dataset de {d0} variables a {k} componentes principales ({(d0-k)/d0*100:.1f}% de reducción dimensional).",
         f"Los {k} componentes explican {R['cum_k']:.2f}% de la varianza total (PC1: {R['evr'][0]*100:.2f}%, PC2: {R['evr'][1]*100:.2f}%).",
         f"Se necesitan {need_k(80)} componentes para ≥80% y {need_k(90)} para ≥90% de la varianza.",
         f"La variable con mayor carga absoluta en PC1 es «{top_var(R,'PC1')}» y en PC2 «{top_var(R,'PC2')}»."]
    return t

def report_html():
    R = SS["res"]; P = SS["prep"]; df = SS["df"]; k = R["k"]; d0 = len(SS["feats"])
    meta = df.loc[R["scores"].index]; col = P["label"] if P["label"] != "(ninguna)" else None
    figs = [fig_var(R["evr"], k), fig_dim(d0, k), fig_scatter(R["scores"], meta, col, "PC1", "PC2", P["X"]), fig_heat(R["loadings"].iloc[:, :k], SS["feats"])]
    h = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figs)]
    li = lambda xs: "<ul>" + "".join(f"<li>{x}</li>" for x in xs) + "</ul>"
    return f"""<html><head><meta charset='utf-8'><title>Informe DimensioVision AI</title><style>
body{{font-family:Segoe UI,Arial;max-width:960px;margin:auto;padding:24px;color:#0F172A;background:#F8FAFC}}h1,h2{{color:#2563EB}}
table{{border-collapse:collapse}}td,th{{border:1px solid #CBD5E1;padding:4px 8px}}</style></head><body>
<h1>DimensioVision AI – Informe de reducción de dimensionalidad con PCA</h1><p>Generado: {dt.datetime.now():%Y-%m-%d %H:%M}</p>
<h2>1. Dataset</h2><p>{SS['src']}. {len(df)} registros, {df.shape[1]} columnas. Procedencia: UCI Machine Learning Repository (Wine Quality), o archivo del usuario.</p>
<h2>2. Objetivo</h2><p>Transformar un conjunto con múltiples variables a una representación de menor dimensión conservando la mayor variabilidad posible, y visualizarla.</p>
<h2>3. Fundamento teórico</h2><p>La reducción de dimensionalidad proyecta datos de alta dimensión a un espacio de menor dimensión para facilitar la visualización, reducir redundancia y ruido.</p>
<h2>4. PCA</h2><p>Con datos centrados/estandarizados X, se calcula la matriz de covarianza C = XᵀX/(n−1) y su descomposición C = VΛVᵀ. Las columnas de V (autovectores) son los componentes; los autovalores λᵢ son sus varianzas. 
Varianza explicada del componente i = λᵢ/Σλⱼ. Proyección: Z = X·V<sub>k</sub>. Implementado con scikit-learn (SVD).</p>
<h2>5. Metodología y preprocesamiento</h2>{li([f"Variables seleccionadas ({d0}): " + ", ".join(SS['feats']), f"Valores faltantes: {P['imp']} (filas eliminadas: {P['dropped']})", "Estandarización: " + ("sí (StandardScaler)" if P['std'] else "no (solo centrado)"), f"Variable de referencia excluida del cálculo: {P['label']}", f"Componentes elegidos: {k}"])}
<h2>6. Resultados</h2>{li(conclusions())}
<h3>Varianza explicada</h3>{h[0]}<h3>Dimensiones</h3>{h[1]}<h3>Proyección PC1–PC2</h3>{h[2]}<h3>Cargas</h3>{h[3]}
<h2>7. Antes y después</h2><p>Reducción = (({d0}−{k})/{d0})×100 = {(d0-k)/d0*100:.2f}%.</p>{pd.concat([P['X'].head(8).round(3), R['scores'].head(8).round(3)], axis=1).to_html()}
<h2>8. Conclusiones</h2>{li(conclusions())}
<h2>9. Limitaciones</h2>{li(["PCA es lineal: no captura estructuras no lineales.","Es sensible a la escala y a valores atípicos.","Los componentes son combinaciones de variables y pueden ser difíciles de interpretar.","Varianza explicada ≠ información útil para una tarea concreta (p. ej., clasificación)."])}
</body></html>"""

def page_concl():
    st.header("Conclusiones"); R = SS.get("res"); need(R is not None, "Ejecuta PCA primero: las conclusiones se generan con los resultados reales.")
    for x in conclusions(): st.markdown(f"- {x}")
    st.caption("Limitaciones: PCA es lineal, sensible a escala y atípicos; varianza explicada no equivale a información útil para toda tarea.")
    st.download_button("📄 Descargar informe (HTML)", report_html().encode("utf-8"), "informe_dimensiovision.html", "text/html")

def page_meto():
    st.header("Metodología")
    st.markdown("""**Reducción de dimensionalidad:** transformar datos con *D* variables en *d < D* nuevas variables que conserven la estructura principal.

**PCA:** 1) estandarizar cada variable; 2) calcular la matriz de covarianza (o SVD); 3) obtener autovalores y autovectores; 4) ordenar por autovalor; 5) proyectar sobre los *k* primeros autovectores.""")
    st.latex(r"C=\tfrac{1}{n-1}X^TX=V\Lambda V^T,\quad \text{VE}_i=\frac{\lambda_i}{\sum_j\lambda_j},\quad Z=XV_k")
    st.markdown("""**Cargas:** autovector × √autovalor (correlación variable–componente si hay estandarización). **Contribución:** (autovector)² × 100.

**Reducción porcentual** = ((D_original − D_reducida) / D_original) × 100.

**Advertencia:** la varianza explicada mide variabilidad capturada, no garantiza conservar la información útil para cualquier tarea.""")

PAGES = {"1. Inicio": page_inicio, "2. Dataset": page_dataset, "3. Preprocesamiento": page_prep, "4. Reducción de dimensionalidad": page_pca,
         "5. Visualización": page_viz, "6. Comparación de resultados": page_cmp, "7. Conclusiones": page_concl, "8. Metodología": page_meto}
with st.spinner("Cargando dataset y ejecutando PCA..."): auto_run()
st.sidebar.title("📉 DimensioVision AI"); st.sidebar.radio("Menú", list(PAGES), key="page")
PAGES[SS["page"]]()
