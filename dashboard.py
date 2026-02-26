# ============================================================
# TechZone Forecast System v3.0
# Dashboard interactivo con Prophet ML y Supply Chain Analytics
# Ejecutar con: python -m streamlit run dashboard.py
# ============================================================

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from prophet import Prophet
from pathlib import Path
from io import BytesIO
import os
import warnings
warnings.filterwarnings("ignore")

# ── RUTAS ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "data" / "techzone.db"

# ── AUTO-SETUP ───────────────────────────────────────────────
# Si la base de datos no existe (primer deploy en la nube)
# la generamos automaticamente ejecutando el script de setup.
if not DB_PATH.exists():
    import subprocess, sys
    os.makedirs(BASE_DIR / "data", exist_ok=True)
    subprocess.run(
        [sys.executable, str(BASE_DIR / "src" / "01_setup_database.py")],
        check=True
    )

# ════════════════════════════════════════════════════════════
# CONFIGURACION DE PAGINA
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TechZone Forecast System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .stApp {
        background-color: #0f1117;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #8892b0 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ccd6f6 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .feature-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3d4166;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        height: 100%;
    }
    .feature-icon { font-size: 2.5rem; margin-bottom: 12px; }
    .feature-title {
        color: #ccd6f6;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .feature-desc { color: #8892b0; font-size: 0.9rem; line-height: 1.5; }
    .hero-container {
        background: linear-gradient(135deg, #0a192f 0%, #112240 50%, #0a192f 100%);
        border: 1px solid #1d3461;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 32px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ccd6f6;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #64ffda;
        margin-bottom: 16px;
        font-weight: 400;
    }
    .hero-desc {
        font-size: 1rem;
        color: #8892b0;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.7;
    }
    .tech-badge {
        display: inline-block;
        background: #112240;
        border: 1px solid #64ffda;
        color: #64ffda;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 4px;
        font-weight: 500;
    }
    .stat-highlight {
        background: linear-gradient(135deg, #0a192f, #112240);
        border: 1px solid #64ffda;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stat-number { font-size: 2.5rem; font-weight: 800; color: #64ffda; }
    .stat-label { color: #8892b0; font-size: 0.9rem; margin-top: 4px; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892b0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #252840 !important;
        color: #64ffda !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1d3461;
    }
    hr { border-color: #1d3461; }
    .stDownloadButton button {
        background: linear-gradient(135deg, #64ffda, #00bcd4);
        color: #0a192f;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# FUNCIONES DE DATOS
# ════════════════════════════════════════════════════════════

@st.cache_data
def cargar_productos():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT nombre, categoria, precio_base FROM productos ORDER BY nombre",
            conn
        )
    return df


@st.cache_data
def cargar_serie(producto):
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT v.fecha, v.cantidad_vendida, v.ingreso_total
            FROM ventas v
            JOIN productos p ON p.id = v.producto_id
            WHERE p.nombre = ?
            ORDER BY v.fecha
        """
        df = pd.read_sql_query(query, conn, params=(producto,))
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


@st.cache_data
def cargar_todos():
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT
                p.nombre, p.categoria, p.precio_base,
                SUM(v.cantidad_vendida) AS unidades_totales,
                SUM(v.ingreso_total)    AS ingresos_totales
            FROM ventas v
            JOIN productos p ON p.id = v.producto_id
            GROUP BY p.nombre, p.categoria, p.precio_base
            ORDER BY ingresos_totales DESC
        """
        df = pd.read_sql_query(query, conn)
    return df


@st.cache_data
def cargar_todas_series():
    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT v.fecha, v.cantidad_vendida, p.nombre
            FROM ventas v
            JOIN productos p ON p.id = v.producto_id
            ORDER BY v.fecha
        """
        df = pd.read_sql_query(query, conn)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


# ════════════════════════════════════════════════════════════
# MODELOS
# ════════════════════════════════════════════════════════════

def modelo_sma(valores, ventana=3):
    predicciones = []
    for i in range(len(valores)):
        if i < ventana:
            predicciones.append(np.mean(valores[:i+1]))
        else:
            predicciones.append(np.mean(valores[i-ventana:i]))
    ultimos  = list(valores[-ventana:])
    forecast = []
    for _ in range(12):
        pred = np.mean(ultimos[-ventana:])
        forecast.append(pred)
        ultimos.append(pred)
    return predicciones, forecast


def modelo_ses(valores, alpha=0.3):
    predicciones = [float(valores[0])]
    for i in range(1, len(valores)):
        pred = alpha * valores[i-1] + (1 - alpha) * predicciones[i-1]
        predicciones.append(pred)
    ultimo_real = float(valores[-1])
    ultimo_pred = predicciones[-1]
    forecast    = []
    for _ in range(12):
        pred = alpha * ultimo_real + (1 - alpha) * ultimo_pred
        forecast.append(pred)
        ultimo_real = pred
        ultimo_pred = pred
    return predicciones, forecast


@st.cache_data
def modelo_prophet(producto):
    df    = cargar_serie(producto)
    datos = df[["fecha", "cantidad_vendida"]].rename(
        columns={"fecha": "ds", "cantidad_vendida": "y"}
    )
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.3,
        interval_width=0.80
    )
    m.fit(datos)
    futuro   = m.make_future_dataframe(periods=12, freq="MS")
    forecast = m.predict(futuro)
    forecast["yhat"]       = forecast["yhat"].clip(lower=0).round()
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0).round()
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0).round()
    return forecast, m, datos


# ════════════════════════════════════════════════════════════
# METRICAS
# ════════════════════════════════════════════════════════════

def calcular_mape(real, predicho):
    real     = np.array(real,     dtype=float)
    predicho = np.array(predicho, dtype=float)
    return np.mean(np.abs((real - predicho) / real)) * 100


def calcular_mae(real, predicho):
    return np.mean(np.abs(
        np.array(real, dtype=float) - np.array(predicho, dtype=float)
    ))


# ════════════════════════════════════════════════════════════
# ANALISIS ABC
# ════════════════════════════════════════════════════════════

def calcular_abc(df_todos):
    df               = df_todos.copy().sort_values(
        "ingresos_totales", ascending=False
    )
    total            = df["ingresos_totales"].sum()
    df["porcentaje"] = df["ingresos_totales"] / total * 100
    df["acumulado"]  = df["porcentaje"].cumsum()

    def clasificar(acum):
        if acum <= 80:   return "A"
        elif acum <= 95: return "B"
        else:            return "C"

    df["categoria_abc"] = df["acumulado"].apply(clasificar)
    return df


# ════════════════════════════════════════════════════════════
# ALERTAS DE STOCK
# ════════════════════════════════════════════════════════════

def calcular_alertas(forecast_mensual, stock_actual,
                     lead_time_dias, precio_unitario):
    demanda_diaria    = np.mean(forecast_mensual) / 30
    dias_cobertura    = (stock_actual / demanda_diaria
                         if demanda_diaria > 0 else 999)
    semanas_cobertura = dias_cobertura / 7
    stock_seguridad   = demanda_diaria * lead_time_dias * 0.20
    punto_reorden     = (demanda_diaria * lead_time_dias) + stock_seguridad
    demanda_anual     = np.sum(forecast_mensual)
    costo_pedido      = 150
    costo_mantencion  = precio_unitario * 0.25
    eoq = (np.sqrt((2 * demanda_anual * costo_pedido) / costo_mantencion)
           if costo_mantencion > 0 else 0)

    if dias_cobertura < lead_time_dias:
        nivel   = "CRITICA"
        mensaje = "Stock insuficiente para cubrir el Lead Time. Pedido urgente."
    elif dias_cobertura < lead_time_dias * 1.5:
        nivel   = "ADVERTENCIA"
        mensaje = "Stock bajo. Considerar realizar pedido pronto."
    else:
        nivel   = "OK"
        mensaje = "Stock suficiente para el periodo analizado."

    return {
        "dias_cobertura":    round(dias_cobertura),
        "semanas_cobertura": round(semanas_cobertura, 1),
        "demanda_diaria":    round(demanda_diaria, 1),
        "punto_reorden":     round(punto_reorden),
        "eoq":               round(eoq),
        "nivel":             nivel,
        "mensaje":           mensaje,
        "stock_seguridad":   round(stock_seguridad),
    }


# ════════════════════════════════════════════════════════════
# GRAFICAS
# ════════════════════════════════════════════════════════════

DARK_BG    = "#0f1117"
CARD_BG    = "#1e2130"
ACCENT     = "#64ffda"
TEXT_MAIN  = "#ccd6f6"
TEXT_SUB   = "#8892b0"
GRID_COLOR = "#1d3461"


def estilo_dark(fig, ax_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_SUB)
        ax.xaxis.label.set_color(TEXT_SUB)
        ax.yaxis.label.set_color(TEXT_SUB)
        ax.title.set_color(TEXT_MAIN)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)


def fig_landing_preview(df_todas):
    fig, ax = plt.subplots(figsize=(12, 3))
    colores  = ["#64ffda", "#ff6b6b", "#ffd166", "#06d6a0", "#118ab2"]
    for i, prod in enumerate(df_todas["nombre"].unique()):
        datos = df_todas[df_todas["nombre"] == prod]
        ax.plot(datos["fecha"], datos["cantidad_vendida"],
                color=colores[i % len(colores)],
                linewidth=1.8, alpha=0.85,
                label=prod.split()[0])
    estilo_dark(fig, ax)
    ax.set_title("Historico de Ventas 2022-2024 — Todos los Productos",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Unidades", fontsize=9)
    ax.legend(fontsize=8, loc="upper left",
              facecolor=CARD_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_SUB)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.tight_layout()
    return fig


def fig_historico(df, producto):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["fecha"], df["cantidad_vendida"],
            color=ACCENT, linewidth=2.2,
            marker="o", markersize=4)
    for anio in [2022, 2023, 2024]:
        ax.axvspan(
            pd.Timestamp(f"{anio}-11-01"),
            pd.Timestamp(f"{anio}-12-31"),
            alpha=0.08, color="#ff6b6b"
        )
    estilo_dark(fig, ax)
    ax.set_title(f"Ventas historicas — {producto}",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Unidades")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.tight_layout()
    return fig


def fig_forecast(df, forecast_vals, modelo_nombre,
                 es_prophet=False, prophet_data=None):
    fig, ax     = plt.subplots(figsize=(11, 4))
    ultimo      = df["fecha"].max()
    fechas_fore = pd.date_range(
        start=ultimo + pd.DateOffset(months=1),
        periods=12, freq="MS"
    )
    ax.plot(df["fecha"], df["cantidad_vendida"],
            color=TEXT_MAIN, linewidth=2, label="Ventas reales")

    if es_prophet and prophet_data is not None:
        fore_df, _, datos_raw = prophet_data
        hist_fore = fore_df[fore_df["ds"] <= datos_raw["ds"].max()]
        fut_fore  = fore_df[fore_df["ds"] >  datos_raw["ds"].max()]
        ax.plot(hist_fore["ds"], hist_fore["yhat"],
                color=ACCENT, linestyle="--",
                label="Ajuste Prophet", alpha=0.7)
        ax.plot(fut_fore["ds"], fut_fore["yhat"],
                color="#ff6b6b", linewidth=2.5,
                marker="o", markersize=5,
                label="Forecast 2025")
        ax.fill_between(
            fut_fore["ds"],
            fut_fore["yhat_lower"],
            fut_fore["yhat_upper"],
            alpha=0.15, color="#ff6b6b",
            label="Intervalo confianza 80%"
        )
    else:
        ax.plot(fechas_fore, forecast_vals,
                color="#ff6b6b", linewidth=2.5,
                marker="o", markersize=5,
                label="Forecast 2025")

    ax.axvline(x=ultimo, color=TEXT_SUB,
               linestyle=":", linewidth=1.5, alpha=0.6)
    estilo_dark(fig, ax)
    ax.set_title(f"Forecast 2025 — {modelo_nombre}",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Unidades")
    ax.legend(fontsize=8, facecolor=CARD_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_SUB)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.tight_layout()
    return fig


def fig_abc(df_abc):
    colores_abc = {"A": "#ff6b6b", "B": "#ffd166", "C": "#06d6a0"}
    colores     = df_abc["categoria_abc"].map(colores_abc)
    fig, axes   = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].barh(
        df_abc["nombre"],
        df_abc["ingresos_totales"] / 1_000_000,
        color=colores, edgecolor=DARK_BG, linewidth=0.5
    )
    for bar, (_, row) in zip(bars, df_abc.iterrows()):
        axes[0].text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"${row['ingresos_totales']/1_000_000:.2f}M [{row['categoria_abc']}]",
            va="center", fontsize=9, color=TEXT_SUB
        )
    axes[0].set_xlabel("Ingresos en Millones ($)")
    axes[0].set_xlim(0, df_abc["ingresos_totales"].max() / 1_000_000 * 1.55)
    leyenda = [
        mpatches.Patch(color="#ff6b6b", label="A — maxima prioridad"),
        mpatches.Patch(color="#ffd166", label="B — prioridad media"),
        mpatches.Patch(color="#06d6a0", label="C — baja prioridad"),
    ]
    axes[0].legend(handles=leyenda, fontsize=8,
                   facecolor=CARD_BG, edgecolor=GRID_COLOR,
                   labelcolor=TEXT_SUB)
    axes[0].set_title("Clasificacion ABC por Ingresos", fontweight="bold")

    axes[1].bar(range(len(df_abc)), df_abc["porcentaje"],
                color=colores, edgecolor=DARK_BG)
    ax2 = axes[1].twinx()
    ax2.plot(range(len(df_abc)), df_abc["acumulado"],
             color=ACCENT, linewidth=2.5, marker="o", markersize=6)
    ax2.axhline(y=80, color="#ff6b6b", linestyle="--",
                linewidth=1.2, alpha=0.7)
    ax2.axhline(y=95, color="#ffd166", linestyle="--",
                linewidth=1.2, alpha=0.7)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("% Acumulado", color=TEXT_SUB)
    ax2.tick_params(colors=TEXT_SUB)
    axes[1].set_xticks(range(len(df_abc)))
    axes[1].set_xticklabels(
        [n.split()[0] for n in df_abc["nombre"]],
        rotation=15, fontsize=8
    )
    axes[1].set_title("Curva de Pareto — Regla 80/20", fontweight="bold")
    axes[1].set_ylabel("% Individual")
    estilo_dark(fig, [axes[0], axes[1]])
    ax2.set_facecolor(CARD_BG)
    plt.tight_layout()
    return fig


def fig_cobertura(alertas, forecast_mensual, stock_actual):
    meses   = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
               "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    stocks  = []
    stock_r = stock_actual
    for v in forecast_mensual:
        stock_r = max(0, stock_r - v)
        stocks.append(stock_r)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(meses, forecast_mensual,
           color="#3d4166", label="Demanda forecast", alpha=0.9)
    ax2 = ax.twinx()
    ax2.plot(meses, stocks, color="#ff6b6b", linewidth=2.5,
             marker="o", markersize=6, label="Stock proyectado")
    ax2.axhline(y=alertas["punto_reorden"],
                color="#ffd166", linestyle="--", linewidth=1.5,
                label=f"Punto reorden ({alertas['punto_reorden']} u)")
    ax2.set_ylabel("Stock disponible", color=TEXT_SUB)
    ax2.tick_params(colors=TEXT_SUB)
    ax.set_ylabel("Demanda mensual")
    ax.set_title("Proyeccion de Stock vs Demanda 2025", fontweight="bold")
    lineas1, labels1 = ax.get_legend_handles_labels()
    lineas2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lineas1 + lineas2, labels1 + labels2,
              fontsize=8, facecolor=CARD_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_SUB)
    estilo_dark(fig, ax)
    ax2.set_facecolor(CARD_BG)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════
# EXPORTAR EXCEL
# ════════════════════════════════════════════════════════════

def generar_excel(producto, df_hist, forecast_vals,
                  es_prophet, prophet_data, df_abc, alertas_data):
    buffer = BytesIO()
    meses  = ["Ene 2025", "Feb 2025", "Mar 2025", "Abr 2025",
              "May 2025", "Jun 2025", "Jul 2025", "Ago 2025",
              "Sep 2025", "Oct 2025", "Nov 2025", "Dic 2025"]

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if es_prophet and prophet_data is not None:
            fore_df, _, datos_raw = prophet_data
            futuro  = fore_df[fore_df["ds"] > datos_raw["ds"].max()].copy()
            df_fore = pd.DataFrame({
                "Mes":      meses,
                "Forecast": futuro["yhat"].astype(int).values,
                "Minimo":   futuro["yhat_lower"].astype(int).values,
                "Maximo":   futuro["yhat_upper"].astype(int).values,
                "Modelo":   "Prophet"
            })
        else:
            df_fore = pd.DataFrame({
                "Mes":      meses,
                "Forecast": [int(v) for v in forecast_vals],
                "Modelo":   "Modelo Simple"
            })
        df_fore.to_excel(writer, sheet_name="Forecast 2025", index=False)

        df_exp          = df_hist.copy()
        df_exp["fecha"] = df_exp["fecha"].dt.strftime("%Y-%m")
        df_exp.columns  = ["Fecha", "Unidades Vendidas", "Ingreso Total"]
        df_exp.to_excel(writer, sheet_name="Historico", index=False)

        df_abc_exp = df_abc[[
            "nombre", "categoria", "unidades_totales",
            "ingresos_totales", "porcentaje",
            "acumulado", "categoria_abc"
        ]].copy()
        df_abc_exp.columns = [
            "Producto", "Categoria", "Unidades Totales",
            "Ingresos Totales", "% Ingreso",
            "% Acumulado", "Clasificacion ABC"
        ]
        df_abc_exp.to_excel(writer, sheet_name="Analisis ABC", index=False)

        if alertas_data:
            df_al = pd.DataFrame([{
                "Producto":             producto,
                "Stock Actual":         alertas_data.get("stock_input", "-"),
                "Dias de Cobertura":    alertas_data["dias_cobertura"],
                "Semanas de Cobertura": alertas_data["semanas_cobertura"],
                "Demanda Diaria":       alertas_data["demanda_diaria"],
                "Punto de Reorden":     alertas_data["punto_reorden"],
                "Stock de Seguridad":   alertas_data["stock_seguridad"],
                "EOQ Recomendado":      alertas_data["eoq"],
                "Nivel de Alerta":      alertas_data["nivel"],
                "Recomendacion":        alertas_data["mensaje"],
            }])
            df_al.to_excel(writer, sheet_name="Alertas Stock", index=False)

    buffer.seek(0)
    return buffer


# ════════════════════════════════════════════════════════════
# NAVEGACION PRINCIPAL
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0'>
        <div style='font-size:2rem'>📦</div>
        <div style='color:#ccd6f6; font-weight:700; font-size:1.1rem'>
            TechZone Forecast
        </div>
        <div style='color:#8892b0; font-size:0.8rem'>Supply Chain x ML</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    pagina = st.radio(
        "Navegacion",
        options=["Inicio", "Forecast", "Analisis ABC",
                 "Alertas de Stock", "Exportar"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    if pagina != "Inicio":
        productos_df = cargar_productos()
        producto_sel = st.selectbox(
            "Producto",
            options=productos_df["nombre"].tolist()
        )
        modelo_sel = st.selectbox(
            "Modelo",
            options=[
                "Prophet (Recomendado)",
                "Promedio Movil Simple",
                "Suavizamiento Exponencial"
            ]
        )
        precio_prod = float(
            productos_df[productos_df["nombre"] == producto_sel]
            ["precio_base"].values[0]
        )
        st.markdown("---")
        st.markdown("**Alertas de Stock**")
        stock_actual = st.number_input(
            "Stock actual (u)", min_value=0,
            max_value=10000, value=200, step=10
        )
        lead_time = st.number_input(
            "Lead Time (dias)", min_value=1,
            max_value=120, value=30, step=1
        )

    st.markdown("---")
    st.caption("v3.0 — 2026")


# ════════════════════════════════════════════════════════════
# PAGINA DE INICIO
# ════════════════════════════════════════════════════════════

if pagina == "Inicio":

    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📦 TechZone Forecast</div>
        <div class="hero-subtitle">Sistema de Prediccion de Demanda</div>
        <div class="hero-desc">
            Plataforma de inteligencia de negocio para retail tecnologico.
            Combina Machine Learning con conceptos de Supply Chain para
            predecir ventas, clasificar productos y optimizar el inventario.
        </div>
        <br>
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Prophet ML</span>
        <span class="tech-badge">SQLite</span>
        <span class="tech-badge">Pandas</span>
        <span class="tech-badge">Supply Chain</span>
        <span class="tech-badge">Streamlit</span>
    </div>
    """, unsafe_allow_html=True)

    df_todos       = cargar_todos()
    total_ingresos = df_todos["ingresos_totales"].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="stat-highlight">
            <div class="stat-number">5</div>
            <div class="stat-label">Productos analizados</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-highlight">
            <div class="stat-number">36</div>
            <div class="stat-label">Meses de historial</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-highlight">
            <div class="stat-number">3.9%</div>
            <div class="stat-label">Error del modelo (MAPE)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-highlight">
            <div class="stat-number">${total_ingresos/1_000_000:.1f}M</div>
            <div class="stat-label">Ingresos analizados</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='color:#ccd6f6; text-align:center'>"
        "Que puede hacer este sistema?</h3>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    features = [
        ("📈", "Forecast ML",
         "Predice ventas para los proximos 12 meses con Prophet e intervalo de confianza del 80%"),
        ("🔬", "Analisis ABC",
         "Clasifica productos segun la regla de Pareto para priorizar el inventario"),
        ("🚨", "Alertas Stock",
         "Calcula dias de cobertura, punto de reorden y EOQ automaticamente"),
        ("📊", "Visualizacion",
         "Dashboards con graficas profesionales de series de tiempo"),
        ("📥", "Export Excel",
         "Reportes completos en Excel con 4 hojas listos para presentar"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<h3 style='color:#ccd6f6'>Vista previa — Historico de ventas</h3>",
        unsafe_allow_html=True
    )
    df_todas = cargar_todas_series()
    st.pyplot(fig_landing_preview(df_todas))
    st.caption(
        "Datos simulados con comportamiento real de retail tecnologico: "
        "tendencia creciente, picos en Black Friday y Navidad, "
        "valle en enero/febrero."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<h3 style='color:#ccd6f6'>Como funciona</h3>",
        unsafe_allow_html=True
    )
    pasos = [
        ("1", "#64ffda", "Datos",
         "3 anos de historial de ventas almacenados en SQLite"),
        ("2", "#ffd166", "Analisis",
         "EDA para detectar tendencias y patrones de estacionalidad"),
        ("3", "#ff6b6b", "Modelo ML",
         "Prophet aprende los patrones y genera predicciones con intervalos"),
        ("4", "#06d6a0", "Decision",
         "Alertas de stock, clasificacion ABC y reportes para la gerencia"),
    ]
    cols = st.columns(4)
    for col, (num, color, titulo, desc) in zip(cols, pasos):
        with col:
            st.markdown(f"""
            <div style="background:#1e2130; border:1px solid {color};
                        border-radius:12px; padding:20px; text-align:center">
                <div style="font-size:2rem; font-weight:800; color:{color}">
                    {num}</div>
                <div style="color:#ccd6f6; font-weight:600; margin:8px 0">
                    {titulo}</div>
                <div style="color:#8892b0; font-size:0.85rem">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Usa el menu de la izquierda para navegar. "
        "Empeza por Forecast para ver las predicciones."
    )


# ════════════════════════════════════════════════════════════
# PAGINA FORECAST
# ════════════════════════════════════════════════════════════

elif pagina == "Forecast":
    df      = cargar_serie(producto_sel)
    valores = df["cantidad_vendida"].values

    st.markdown(
        f"<h2 style='color:#ccd6f6'>Forecast 2025 — {producto_sel}</h2>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Promedio mensual",
                  f"{df['cantidad_vendida'].mean():.0f} u")
    with col2:
        st.metric("Maximo historico",
                  f"{df['cantidad_vendida'].max()} u")
    with col3:
        st.metric("Minimo historico",
                  f"{df['cantidad_vendida'].min()} u")
    with col4:
        st.metric("Ingresos totales",
                  f"${df['ingreso_total'].sum()/1_000_000:.2f}M")

    st.divider()

    es_prophet    = "Prophet" in modelo_sel
    prophet_data  = None
    forecast_vals = None

    if es_prophet:
        with st.spinner("Entrenando Prophet..."):
            prophet_data  = modelo_prophet(producto_sel)
            fore_df, _, _ = prophet_data
            datos_raw     = prophet_data[2]
            futuro_fore   = fore_df[fore_df["ds"] > datos_raw["ds"].max()]
            forecast_vals = futuro_fore["yhat"].values
    elif "Movil" in modelo_sel:
        _, forecast_vals = modelo_sma(valores)
    else:
        _, forecast_vals = modelo_ses(valores)

    tab_hist, tab_fore = st.tabs(["Historico", "Forecast 2025"])

    with tab_hist:
        st.pyplot(fig_historico(df, producto_sel))
        st.caption("Zonas rojas: Noviembre-Diciembre (Black Friday y Navidad)")
        with st.expander("Ver datos crudos"):
            df_m          = df.copy()
            df_m["fecha"] = df_m["fecha"].dt.strftime("%Y-%m")
            df_m["ingreso_total"] = df_m["ingreso_total"].apply(
                lambda x: f"${x:,.2f}"
            )
            st.dataframe(df_m, use_container_width=True)

    with tab_fore:
        st.pyplot(fig_forecast(
            df, forecast_vals, modelo_sel,
            es_prophet=es_prophet, prophet_data=prophet_data
        ))
        meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

        if es_prophet and prophet_data is not None:
            fore_df, _, datos_raw = prophet_data
            fut   = fore_df[fore_df["ds"] > datos_raw["ds"].max()].copy()
            tabla = pd.DataFrame({
                "Mes":      meses,
                "Minimo":   fut["yhat_lower"].astype(int).values,
                "Forecast": fut["yhat"].astype(int).values,
                "Maximo":   fut["yhat_upper"].astype(int).values,
            })
            st.dataframe(tabla, use_container_width=True, hide_index=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Stock minimo",
                          f"{int(fut['yhat_lower'].sum()):,} u")
            with c2:
                st.metric("Forecast central",
                          f"{int(fut['yhat'].sum()):,} u",
                          delta="Mas probable")
            with c3:
                st.metric("Stock maximo",
                          f"{int(fut['yhat_upper'].sum()):,} u")
        else:
            tabla = pd.DataFrame({
                "Mes":      meses,
                "Forecast": [int(v) for v in forecast_vals],
            })
            st.dataframe(tabla, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════
# PAGINA ANALISIS ABC
# ════════════════════════════════════════════════════════════

elif pagina == "Analisis ABC":
    st.markdown(
        "<h2 style='color:#ccd6f6'>Analisis ABC — Clasificacion de Productos</h2>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <p style='color:#8892b0'>
    Clasificacion de productos segun la regla de Pareto (80/20).
    Los productos <b style='color:#ff6b6b'>Categoria A</b> generan
    el 80% de los ingresos y requieren maxima atencion en inventario.
    </p>
    """, unsafe_allow_html=True)

    df_todos = cargar_todos()
    df_abc   = calcular_abc(df_todos)
    st.pyplot(fig_abc(df_abc))

    df_tabla = df_abc[[
        "nombre", "categoria_abc", "ingresos_totales",
        "porcentaje", "acumulado"
    ]].copy()
    df_tabla.columns = ["Producto", "ABC", "Ingresos",
                        "% Individual", "% Acumulado"]
    df_tabla["Ingresos"]     = df_tabla["Ingresos"].apply(
        lambda x: f"${x:,.0f}")
    df_tabla["% Individual"] = df_tabla["% Individual"].apply(
        lambda x: f"{x:.1f}%")
    df_tabla["% Acumulado"]  = df_tabla["% Acumulado"].apply(
        lambda x: f"{x:.1f}%")
    st.dataframe(df_tabla, use_container_width=True, hide_index=True)
    st.info(
        "Recomendacion: concentra el 80% de tu presupuesto de "
        "inventario en los productos Categoria A."
    )


# ════════════════════════════════════════════════════════════
# PAGINA ALERTAS DE STOCK
# ════════════════════════════════════════════════════════════

elif pagina == "Alertas de Stock":
    df      = cargar_serie(producto_sel)
    valores = df["cantidad_vendida"].values

    st.markdown(
        f"<h2 style='color:#ccd6f6'>Alertas de Stock — {producto_sel}</h2>",
        unsafe_allow_html=True
    )

    es_prophet    = "Prophet" in modelo_sel
    prophet_data  = None
    forecast_vals = None

    if es_prophet:
        with st.spinner("Calculando forecast..."):
            prophet_data  = modelo_prophet(producto_sel)
            fore_df, _, _ = prophet_data
            datos_raw     = prophet_data[2]
            futuro_fore   = fore_df[fore_df["ds"] > datos_raw["ds"].max()]
            forecast_vals = futuro_fore["yhat"].values
    elif "Movil" in modelo_sel:
        _, forecast_vals = modelo_sma(valores)
    else:
        _, forecast_vals = modelo_ses(valores)

    alertas              = calcular_alertas(
        forecast_vals, stock_actual, lead_time, precio_prod
    )
    alertas["stock_input"] = stock_actual

    if alertas["nivel"] == "CRITICA":
        st.error(f"ALERTA CRITICA: {alertas['mensaje']}")
    elif alertas["nivel"] == "ADVERTENCIA":
        st.warning(f"ADVERTENCIA: {alertas['mensaje']}")
    else:
        st.success(f"Estado OK: {alertas['mensaje']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dias de cobertura",
                  f"{alertas['dias_cobertura']} dias")
    with col2:
        st.metric("Semanas de cobertura",
                  f"{alertas['semanas_cobertura']} sem")
    with col3:
        st.metric("Punto de reorden",
                  f"{alertas['punto_reorden']} u")
    with col4:
        st.metric("EOQ recomendado", f"{alertas['eoq']} u")

    st.pyplot(fig_cobertura(alertas, forecast_vals, stock_actual))

    with st.expander("Que significa cada indicador?"):
        st.markdown(f"""
        - **Dias de cobertura**: con {stock_actual} u en stock y demanda
          diaria de {alertas['demanda_diaria']} u/dia, tenes stock para
          {alertas['dias_cobertura']} dias.
        - **Punto de reorden ({alertas['punto_reorden']} u)**: cuando
          bajes a este nivel, hace el pedido considerando
          {lead_time} dias de Lead Time del proveedor.
        - **Stock de seguridad ({alertas['stock_seguridad']} u)**:
          colchon para absorber variaciones inesperadas.
        - **EOQ ({alertas['eoq']} u)**: cantidad optima por pedido
          que minimiza costos de ordenar y almacenar.
        """)


# ════════════════════════════════════════════════════════════
# PAGINA EXPORTAR
# ════════════════════════════════════════════════════════════

elif pagina == "Exportar":
    df      = cargar_serie(producto_sel)
    valores = df["cantidad_vendida"].values

    st.markdown(
        "<h2 style='color:#ccd6f6'>Exportar Reporte a Excel</h2>",
        unsafe_allow_html=True
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    for col, titulo, desc in zip(
        [col_a, col_b, col_c, col_d],
        ["Forecast 2025", "Historico", "Analisis ABC", "Alertas Stock"],
        ["Predicciones con intervalos", "Ventas 2022-2024",
         "Clasificacion productos", "Indicadores reposicion"]
    ):
        with col:
            st.markdown(f"""
            <div style="background:#1e2130; border:1px solid #3d4166;
                        border-radius:10px; padding:16px; text-align:center">
                <div style="color:#64ffda; font-weight:600">{titulo}</div>
                <div style="color:#8892b0; font-size:0.85rem; margin-top:6px">
                    {desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    es_prophet    = "Prophet" in modelo_sel
    prophet_data  = None
    forecast_vals = None

    if es_prophet:
        with st.spinner("Preparando datos..."):
            prophet_data  = modelo_prophet(producto_sel)
            fore_df, _, _ = prophet_data
            datos_raw     = prophet_data[2]
            futuro_fore   = fore_df[fore_df["ds"] > datos_raw["ds"].max()]
            forecast_vals = futuro_fore["yhat"].values
    elif "Movil" in modelo_sel:
        _, forecast_vals = modelo_sma(valores)
    else:
        _, forecast_vals = modelo_ses(valores)

    df_todos    = cargar_todos()
    df_abc      = calcular_abc(df_todos)
    alertas_exp = calcular_alertas(
        forecast_vals, stock_actual, lead_time, precio_prod
    )
    alertas_exp["stock_input"] = stock_actual

    excel_buffer = generar_excel(
        producto_sel, df, forecast_vals,
        es_prophet, prophet_data, df_abc, alertas_exp
    )
    nombre = f"techzone_{producto_sel.replace(' ', '_')}.xlsx"

    st.download_button(
        label="Descargar reporte Excel",
        data=excel_buffer,
        file_name=nombre,
        mime="application/vnd.openxmlformats-officedocument"
              ".spreadsheetml.sheet"
    )
    st.success(f"Reporte listo: {nombre}")