import streamlit as st

st.set_page_config(
    page_title="Pulsar Score",  # Título del navegador
    layout="wide"
)

import pandas as pd
from PIL import Image

# -----------------------------
# Tipografía Montserrat
# -----------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif !important;
    }

    .stSelectbox label, .stSelectbox div[data-baseweb="select"] * {
        font-family: 'Montserrat', sans-serif !important;
    }

    .stButton button {
        font-family: 'Montserrat', sans-serif !important;
    }

    .stTextInput > label, .stNumberInput > label, .stDateInput > label {
        font-family: 'Montserrat', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Logo en esquina superior izquierda
# -----------------------------
logo = Image.open("logo.png")
st.image(logo, width=450)

# -----------------------------
# Espacio entre el logo y las pestañas
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

# -----------------------------
# Cargar datos desde Excel y limpiar columnas
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("df_totales_2.xlsx", header=0)

    # Limpiar espacios y caracteres invisibles en los nombres de columna
    df.columns = df.columns.str.strip().str.replace("\u00A0", "", regex=True)

    # Renombrar variantes comunes
    df.rename(columns={
        "colonia": "Colonia",
        "COLONIA": "Colonia"
    }, inplace=True)

    return df

df = load_data()

# Ver columnas disponibles (útil para depuración en pantalla)
#st.write("Columnas detectadas:", df.columns.tolist())

# -----------------------------
# TABS PRINCIPALES DEL DASHBOARD
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Resultados Generales", 
    "Operativos", 
    "Percepción", 
    "Alineación",
    "Mapa Alineación",
    "Mapa Probabilidad"
])

import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium

# -------------------------------
# CARGAR DATOS PARA MAPAS (GeoJSON y DataFrame)
# -------------------------------
@st.cache_data
def load_geojson():
    gdf = gpd.read_file("cuauhtemoc_colonias.geojson")
    gdf = gdf.rename(columns={'NOMUT': 'Colonia'})
    return gdf

@st.cache_data
def load_df_mapas():
    df = pd.read_excel("df_maps_2.xlsx")
    return df

gdf = load_geojson()
df_mapas = load_df_mapas()

# -------------------------------
# PALETA DE COLORES
# -------------------------------
colores = {
    "Crítica": "#d00000",
    "Baja": "#fca311",
    "Media": "#f9c74f",
    "Alta": "#00a300"
}

# -------------------------------
# FUNCIONES DE COLOR
# -------------------------------
def color_alineacion(value):
    if pd.isna(value): return '#d9d9d9'
    if value <= 21.45: return colores["Crítica"]
    elif value <= 35.20: return colores["Baja"]
    elif value <= 44.31: return colores["Media"]
    else: return colores["Alta"]

def color_probabilidad(value):
    if pd.isna(value): return '#d9d9d9'
    if value <= 20: return colores["Crítica"]
    elif value <= 35: return colores["Baja"]
    elif value <= 55: return colores["Media"]
    else: return colores["Alta"]

# -------------------------------
# FUNCIÓN GENÉRICA PARA MOSTRAR MAPAS
# -------------------------------
def render_map(merged_df, variable, color_fn, titulo):
    m = folium.Map(location=[19.4333, -99.1500], zoom_start=13, tiles=None)
    folium.TileLayer('cartodbpositron', control=False).add_to(m)

    def style_function(feature):
        value = feature['properties'].get(variable, None)
        return {
            'fillColor': color_fn(value),
            'color': 'black',
            'weight': 0.6,
            'fillOpacity': 0.7,
        }

    folium.GeoJson(
        merged_df,
        style_function=style_function,
        tooltip=GeoJsonTooltip(
            fields=["Colonia", variable],
            aliases=["Colonia:", f"{titulo}:"],
            localize=True
        )
    ).add_to(m)

    col1, col2 = st.columns([4, 1.5])
    with col1:
        st_folium(m, width=800, height=600)

    with col2:
        st.markdown(f"#### {titulo}")
        for nombre, color in colores.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;margin-bottom:8px;">'
                f'<div style="width:15px;height:15px;background-color:{color};margin-right:8px;"></div>'
                f'{nombre}'
                f'</div>',
                unsafe_allow_html=True
            )


# -----------------------------
# TAB 1: RESULTADOS GENERALES
# -----------------------------
with tab1:
    import plotly.express as px

    # --------------------------
    # FILTROS
    # --------------------------
    opciones_meses = sorted(df["Mes"].unique())
    opciones_colonia = sorted(df["Colonia"].unique())

    col1, col2 = st.columns(2)
    with col1:
        colonia_seleccionada = st.selectbox("Selecciona una colonia:", opciones_colonia, index=opciones_colonia.index("ALCALDIA"))
    with col2:
        mes_seleccionado = st.selectbox("Selecciona un mes:", opciones_meses, index=len(opciones_meses)-1)

    # --------------------------
    # GRÁFICO DE LÍNEA
    # --------------------------
    if colonia_seleccionada == "ALCALDIA":
        df_line = df[
            (df["Colonia"] == "ALCALDIA") &
            (df["Dimension"] == "GLOBAL")
        ].copy()
        titulo_grafico = "Evolución del Score Global - ALCALDIA CUAUHTÉMOC"
    else:
        df_line = df[
            (df["Colonia"] == colonia_seleccionada) &
            (df["Dimension"] == "TOTAL")
        ].copy()
        titulo_grafico = f"Evolución del Score Total - {colonia_seleccionada}"

    if df_line.empty:
        st.warning(f"No hay datos para {colonia_seleccionada} con esa configuración.")
    else:
        df_line = df_line[df_line["Mes"] <= mes_seleccionado].sort_values("Mes")

        fig = px.line(
            df_line,
            x="Mes",
            y="Score_Compuesto",
            markers=True,
            text="Score_Compuesto"
        )

        fig.update_traces(
            line=dict(color="#000000", width=3),
            marker=dict(color="#666666", size=8),
            textposition="top center",
            texttemplate="%{text:.1f}"
        )

        fig.update_layout(
            title=titulo_grafico,
            title_font=dict(family="Montserrat", size=20),
            xaxis_title="Mes",
            yaxis_title="Score",
            font=dict(family="Montserrat", size=14, color="#000000"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
            yaxis=dict(showgrid=True, gridcolor="#e0e0e0")
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # TARJETAS POR DIMENSIÓN
    # --------------------------
    if colonia_seleccionada == "ALCALDIA":
        df_dim = df[
            (df["Colonia"] == "ALCALDIA") &
            (df["Dimension"] != "GLOBAL")
        ].copy()
    else:
        df_dim = df[
            (df["Colonia"] == colonia_seleccionada) &
            (df["Dimension"] != "TOTAL")
        ].copy()

    meses_ordenados = sorted(df_dim["Mes"].unique())
    if mes_seleccionado not in meses_ordenados or mes_seleccionado == meses_ordenados[0]:
        st.warning("No hay datos anteriores para comparar este mes.")
    else:
        mes_actual = mes_seleccionado
        idx = meses_ordenados.index(mes_actual)
        mes_anterior = meses_ordenados[idx - 1]

        mes_actual_legible = pd.to_datetime(mes_actual).strftime("%B %Y").title()

        st.markdown(f"""
            <h3 style="font-family:Montserrat; font-size:20px; font-weight:600; margin-bottom:20px;">
                Comparativo por dimensión ({colonia_seleccionada} - mes actual <strong>{mes_actual_legible}</strong> vs anterior)
            </h3>
        """, unsafe_allow_html=True)

        df_actual = df_dim[df_dim["Mes"] == mes_actual].set_index("Dimension")
        df_anterior = df_dim[df_dim["Mes"] == mes_anterior].set_index("Dimension")

        cols = st.columns(3)

        for i, dimension in enumerate(df_actual.index):
            score_actual = df_actual.loc[dimension, "Score_Compuesto"]
            score_anterior = df_anterior.loc[dimension, "Score_Compuesto"]

            diferencia = score_actual - score_anterior
            flecha = "↑" if diferencia > 0 else "↓"
            color_hex = "#00a300" if diferencia > 0 else "#d00000"
            cambio = f"{'+' if diferencia > 0 else ''}{diferencia:.0f}"

            with cols[i % 3]:
                st.markdown(f"""
                    <div style="
                        background-color: white;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 16px;
                        margin-bottom: 24px;
                        text-align: center;
                        font-family: Montserrat;
                    ">
                        <div style="font-size: 16px; font-weight: 600; color: black; margin-bottom: 10px;">
                            {dimension}
                        </div>
                        <div style="font-size: 32px; font-weight: bold; color: black;">
                            {score_actual:.0f}
                        </div>
                        <div style="font-size: 18px; font-weight: 600; color: {color_hex}; margin-top: 4px;">
                            {flecha} {cambio}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# -----------------------------
# TAB 2: OPERATIVOS
# -----------------------------
with tab2:
    import plotly.graph_objects as go
    import plotly.express as px

    # -----------------------------
    # Cargar datos operativos desde archivo
    # -----------------------------
    @st.cache_data
    def load_operativos():
        df = pd.read_excel("df_operativos_2.xlsx")
        df["Mes"] = pd.to_datetime(df["Mes"])  # Asegurar tipo datetime
        return df

    df_operativos = load_operativos()

    # -----------------------------
    # Preparar opciones para filtros
    # -----------------------------
    meses_unicos = sorted(df_operativos["Mes"].dropna().unique())
    mes_labels = [mes.strftime("%Y-%m-%d") for mes in meses_unicos]
    mes_dict = dict(zip(mes_labels, meses_unicos))  # string → datetime

    opciones_colonia = sorted(df_operativos["Colonia"].unique())
    ultimo_mes_label = mes_labels[-1]

    # -----------------------------
    # Filtros visibles
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        colonia_seleccionada = st.selectbox(
            "Selecciona una colonia:",
            opciones_colonia,
            index=opciones_colonia.index("ALCALDIA"),
            key="op_colonia_tab2"
        )

    with col2:
        mes_label_seleccionado = st.selectbox(
            "Selecciona un mes:",
            mes_labels,
            index=mes_labels.index(ultimo_mes_label),
            key="op_mes_tab2"
        )
        mes_seleccionado = mes_dict[mes_label_seleccionado]

    # -----------------------------
    # GRÁFICO 1: Barras por acción
    # -----------------------------
    
    df_filtrado = df_operativos[
        (df_operativos["Colonia"] == colonia_seleccionada) &
        (df_operativos["Mes"] == mes_seleccionado)
    ].copy()

    if df_filtrado.empty:
        st.warning("No hay datos disponibles para esta combinación.")
    else:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_filtrado["Accion"],
            y=df_filtrado["Aporte_Accion"],
            name="Aporte de Acción",
            marker_color="#000000",
            yaxis="y1"
        ))

        fig.update_layout(
            title=f"Aportes por Acción - {colonia_seleccionada} ({mes_seleccionado.strftime('%B %Y')})",
            title_font=dict(family="Montserrat", size=20),
            font=dict(family="Montserrat", size=14, color="#000000"),
            xaxis=dict(title="Acción", tickangle=90),
            yaxis=dict(visible=False),
            barmode="group",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=600,
            margin=dict(l=0, r=0, t=60, b=100)
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # GRÁFICO 2: Serie de tiempo (líneas por dimensión)
    # -----------------------------
    df_lineas = df_operativos[
        (df_operativos["Colonia"] == colonia_seleccionada) &
        (df_operativos["Mes"] <= mes_seleccionado)
    ].copy()

    df_lineas["Mes"] = df_lineas["Mes"].dt.to_period("M").dt.to_timestamp()

    if not df_lineas.empty:
        ultimo_mes = df_lineas["Mes"].max()
        ranking_dim = (
            df_lineas[df_lineas["Mes"] == ultimo_mes]
            .groupby("Dimension")["Aporte_Dimension"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        df_lineas["Dimension"] = pd.Categorical(df_lineas["Dimension"], categories=ranking_dim, ordered=True)

        fig3 = px.line(
            df_lineas,
            x="Mes",
            y="Aporte_Dimension",
            color="Dimension",
            markers=True,
            title=f"Evolución mensual del Aporte por Dimensión - {colonia_seleccionada}"
        )

        fig3.update_layout(
            height=600,
            title_font=dict(family="Montserrat", size=20),
            font=dict(family="Montserrat", size=14, color="#000000"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend_title_text="Dimensión",
            xaxis=dict(
                title="Mes",
                showgrid=True,
                gridcolor="#e0e0e0",
                tickformat="%b %Y",
                tickangle=90,
                dtick="M1",
                ticklabelmode="instant"
            ),
            yaxis=dict(
                title="Aporte de Dimensión",
                showgrid=True,
                gridcolor="#e0e0e0"
            ),
            margin=dict(l=20, r=20, t=60, b=100)
        )

        fig3.for_each_trace(lambda t: t.update(line=dict(width=2)))
        st.plotly_chart(fig3, use_container_width=True)


# -----------------------------
# TAB 3: PERCEPCIÓN
# -----------------------------
with tab3:
    import plotly.express as px
    import pandas as pd

    # -----------------------------
    # Cargar datos de percepción
    # -----------------------------
    @st.cache_data
    def load_percepcion():
        df = pd.read_excel("df_percepcion_2.xlsx")
        df["Mes"] = pd.to_datetime(df["Mes"])
        return df

    df_percepcion = load_percepcion()

    # -----------------------------
    # Filtros: Colonia y Mes
    # -----------------------------
    meses_unicos = sorted(df_percepcion["Mes"].dropna().unique())
    mes_labels = [mes.strftime("%Y-%m-%d") for mes in meses_unicos]
    mes_dict = dict(zip(mes_labels, meses_unicos))

    opciones_colonia = sorted(df_percepcion["Colonia"].unique())
    ultimo_mes_label = mes_labels[-1]

    col1, col2 = st.columns(2)

    with col1:
        colonia_seleccionada = st.selectbox(
            "Selecciona una colonia:",
            opciones_colonia,
            index=opciones_colonia.index("ALCALDIA"),
            key="per_colonia_tab3"
        )

    with col2:
        mes_label_seleccionado = st.selectbox(
            "Selecciona un mes:",
            mes_labels,
            index=mes_labels.index(ultimo_mes_label),
            key="per_mes_tab3"
        )
        mes_seleccionado = mes_dict[mes_label_seleccionado]

    # -----------------------------
    # GRÁFICO 1: Barras por Dimensión
    # -----------------------------
    df_bar = df_percepcion[
        (df_percepcion["Colonia"] == colonia_seleccionada) &
        (df_percepcion["Mes"] == mes_seleccionado)
    ].copy()

    if df_bar.empty:
        st.warning("No hay datos disponibles para esta combinación.")
    else:
        fig_bar = px.bar(
            df_bar,
            x="Dimension",
            y="Score_Percepción",
            color_discrete_sequence=["#0a0a0a"],
            title=f"Score de Percepción por Dimensión - {colonia_seleccionada} ({mes_seleccionado.strftime('%B %Y')})"
        )

        fig_bar.update_traces(marker=dict(
            line=dict(width=0, color="#0a0a0a"),
            color="#0a0a0a"
        ))

        fig_bar.update_layout(
            title_font=dict(family="Montserrat", size=20),
            font=dict(family="Montserrat", size=14, color="#000000"),
            xaxis=dict(title="Dimensión", tickangle=90),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, showticklabels=False, ticks="", showline=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=500,
            margin=dict(l=20, r=20, t=60, b=100)
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # -----------------------------
    # GRÁFICO 2: Serie de tiempo por Dimensión
    # -----------------------------
    df_lineas = df_percepcion[
        (df_percepcion["Colonia"] == colonia_seleccionada) &
        (df_percepcion["Mes"] <= mes_seleccionado)
    ].copy()

    df_lineas["Mes"] = df_lineas["Mes"].dt.to_period("M").dt.to_timestamp()

    if not df_lineas.empty:
        ultimo_mes = df_lineas["Mes"].max()
        ranking_dim = (
            df_lineas[df_lineas["Mes"] == ultimo_mes]
            .groupby("Dimension")["Score_Percepción"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        df_lineas["Dimension"] = pd.Categorical(df_lineas["Dimension"], categories=ranking_dim, ordered=True)

        fig_line = px.line(
            df_lineas,
            x="Mes",
            y="Score_Percepción",
            color="Dimension",
            markers=True,
            title=f"Evolución mensual del Score de Percepción - {colonia_seleccionada}"
        )

        fig_line.update_layout(
            height=600,
            title_font=dict(family="Montserrat", size=20),
            font=dict(family="Montserrat", size=14, color="#000000"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend_title_text="Dimensión",
            xaxis=dict(
                title="Mes",
                showgrid=True,
                gridcolor="#e0e0e0",
                tickformat="%b %Y",
                tickangle=90,
                dtick="M1",
                ticklabelmode="instant"
            ),
            yaxis=dict(
                title="Score de Percepción",
                showgrid=True,
                gridcolor="#e0e0e0"
            ),
            margin=dict(l=20, r=20, t=60, b=100)
        )

        fig_line.for_each_trace(lambda t: t.update(line=dict(width=2)))
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("No hay datos disponibles para mostrar la evolución.")

# -----------------------------
# TAB 4: ALINEACIÓN
# -----------------------------
with tab4:
    import plotly.graph_objects as go

    # -----------------------------
    # Cargar y preparar datos
    # -----------------------------
    @st.cache_data
    def load_data_alineacion():
        df = pd.read_excel("df_alineacion_2.xlsx")
        df["Mes"] = pd.to_datetime(df["Mes"])

        # Usar los mismos niveles que el resto del dashboard
        bins = [0, 21.45, 35.20, 44.31, 100]
        labels = ["Crítica", "Baja", "Media", "Alta"]
        df["Nivel_Alineacion"] = pd.cut(
            df["Indice_Alineacion"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=True
        )

        return df

    df = load_data_alineacion()

    # -----------------------------
    # Filtros
    # -----------------------------
    colonias = sorted(df["Colonia"].unique())
    meses_unicos = sorted(df["Mes"].dropna().unique())
    mes_labels = [mes.strftime("%Y-%m-%d") for mes in meses_unicos]
    mes_dict = dict(zip(mes_labels, meses_unicos))

    col1, col2 = st.columns(2)
    with col1:
        colonia_sel = st.selectbox(
            "Selecciona una colonia:",
            colonias,
            index=colonias.index("CIUDAD") if "CIUDAD" in colonias else 0,
            key="ali_colonia"
        )
    with col2:
        mes_sel_label = st.selectbox(
            "Selecciona un mes:",
            mes_labels,
            index=len(mes_labels) - 1,
            key="ali_mes"
        )
        mes_sel = mes_dict[mes_sel_label]

    # -----------------------------
    # TÍTULO CENTRAL BONITO
    # -----------------------------
    titulo_legible = mes_sel.strftime("%B %Y").title()

    st.markdown(f"""
    <h3 style='font-family:Montserrat; font-size:20px; font-weight:bold; margin-top:20px;'>
        Índice de Alineación – {colonia_sel} ({titulo_legible})
    </h3>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Preparar DataFrame actual
    # -----------------------------
    df_actual = df[
        (df["Colonia"] == colonia_sel) &
        (df["Mes"] == mes_sel)
    ].groupby("Dimension", as_index=False).agg({
        "Indice_Alineacion": "mean",
        "Nivel_Alineacion": "first"
    })

    # -----------------------------
    # Crear tarjetas con gauge y bolita
    # -----------------------------
    cols = st.columns(3)

    for i, row in df_actual.iterrows():
        dimension = row["Dimension"]
        valor = round(row["Indice_Alineacion"], 1)
        nivel = row["Nivel_Alineacion"]
        color = colores.get(nivel, "#888888")

        gauge = go.Figure(go.Indicator(
            mode="gauge",
            value=valor,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 21.45], 'color': "#ffe5e5"},
                    {'range': [21.45, 35.2], 'color': "#fff0cc"},
                    {'range': [35.2, 44.31], 'color': "#fdf6d3"},
                    {'range': [44.31, 100], 'color': "#e5ffe5"}
                ],
            }
        ))

        gauge.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=230,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Montserrat", color="#000000")
        )

        with cols[i % 3]:
            with st.container():
                st.markdown(f"""
                    <div style="
                        background-color: white;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                        font-family: Montserrat;
                        margin-bottom: 10px;
                    ">
                        <div style="font-size: 16px; font-weight: 600; color: black; margin-bottom: 6px;">
                            {dimension}
                        </div>
                        <div style="font-size: 14px; color: {color}; font-weight: 600; margin-bottom: 10px;">
                            ● {nivel}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(gauge, use_container_width=True)



# -------------------------------
# TAB 5: Mapa Alineación
# -------------------------------
with tab5:
    meses_disponibles = sorted(df_mapas['Mes'].unique())
    mes_seleccionado = st.selectbox(
    "Selecciona el mes:",
    meses_disponibles,
    index=len(meses_disponibles) - 1,
    key="mes_selector_tab5"
    )
    df_mes = df_mapas[df_mapas['Mes'] == mes_seleccionado]
    merged = gdf.merge(df_mes, on='Colonia', how='left')
    render_map(merged, "Indice_Alineacion", color_alineacion, "Alineación Global")

# -------------------------------
# TAB 6: Mapa Voto
# -------------------------------
with tab6:
    meses_disponibles = sorted(df_mapas['Mes'].unique())
    mes_seleccionado = st.selectbox(
    "Selecciona el mes:",
    meses_disponibles,
    index=len(meses_disponibles) - 1,
    key="mes_selector_tab6"
    )
    df_mes = df_mapas[df_mapas['Mes'] == mes_seleccionado]
    merged = gdf.merge(df_mes, on='Colonia', how='left')
    render_map(merged, "Probabilidad", color_probabilidad, "Probabilidad de Voto")

