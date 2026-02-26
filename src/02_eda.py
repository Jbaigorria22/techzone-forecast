# ============================================================
# FASE 2: Análisis Exploratorio de Datos (EDA)
# TechZone Forecast System
# ============================================================

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import os

# ── CONFIGURACIÓN DE RUTAS ───────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DB_PATH   = BASE_DIR / "data" / "techzone.db"
OUT_PATH  = BASE_DIR / "outputs" / "graficas"
os.makedirs(OUT_PATH, exist_ok=True)

# ── ESTILO VISUAL ────────────────────────────────────────────
# Definimos un estilo consistente para todas las gráficas.
# En reportes logísticos reales, la consistencia visual
# transmite profesionalismo y facilita la comparación.
sns.set_theme(style="darkgrid")
COLORES = [
    "#2196F3", "#FF5722", "#4CAF50",
    "#FF9800", "#9C27B0"
]


# ── FUNCIÓN 1: CARGAR DATOS ──────────────────────────────────
# Traemos los datos de SQLite a un DataFrame de pandas.
# Convertimos la fecha a tipo datetime — esto es CRÍTICO
# para trabajar con series de tiempo en forecasting.

def cargar_datos():
    print("📂 Cargando datos desde la base de datos...")

    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT
                v.fecha,
                v.cantidad_vendida,
                v.ingreso_total,
                p.nombre    AS producto,
                p.categoria AS categoria
            FROM ventas v
            JOIN productos p ON p.id = v.producto_id
            ORDER BY v.fecha
        """
        df = pd.read_sql_query(query, conn)

    # Convertir fecha a datetime
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Extraer columnas de tiempo útiles para análisis
    df["anio"] = df["fecha"].dt.year
    df["mes"]  = df["fecha"].dt.month
    df["mes_nombre"] = df["fecha"].dt.strftime("%b")  # Ene, Feb, etc.

    print(f"✅ {len(df)} registros cargados.")
    print(f"   Período: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
    print(f"   Productos: {df['producto'].nunique()}")
    print()
    return df


# ── FUNCIÓN 2: EVOLUCIÓN DE VENTAS MENSUALES ────────────────
# CONCEPTO LOGÍSTICO: Serie de tiempo
# Una serie de tiempo es simplemente una variable medida
# a lo largo del tiempo en intervalos regulares.
# En supply chain, TODO es series de tiempo:
# ventas, inventario, lead times, costos de transporte.
# Aprender a leerlas es la habilidad #1 del analista logístico.

def grafica_evolucion_ventas(df):
    print("📊 Generando gráfica 1: Evolución de ventas mensuales...")

    fig, ax = plt.subplots(figsize=(14, 6))

    productos = df["producto"].unique()

    for i, producto in enumerate(productos):
        datos_prod = df[df["producto"] == producto].copy()
        datos_prod = datos_prod.sort_values("fecha")
        ax.plot(
            datos_prod["fecha"],
            datos_prod["cantidad_vendida"],
            label=producto,
            color=COLORES[i],
            linewidth=2,
            marker="o",
            markersize=3
        )

    # Marcamos los picos de Black Friday y Navidad
    # para que sean visibles en la gráfica
    for anio in [2022, 2023, 2024]:
        ax.axvspan(
            pd.Timestamp(f"{anio}-11-01"),
            pd.Timestamp(f"{anio}-12-31"),
            alpha=0.1, color="red",
            label="Black Friday / Navidad" if anio == 2022 else ""
        )

    ax.set_title(
        "Evolución de Ventas Mensuales por Producto (2022-2024)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Unidades Vendidas", fontsize=11)
    ax.legend(loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    ruta = OUT_PATH / "01_evolucion_ventas.png"
    plt.savefig(ruta, dpi=150)
    plt.show()
    print(f"   ✅ Guardada en: {ruta}\n")


# ── FUNCIÓN 3: ESTACIONALIDAD PROMEDIO ──────────────────────
# CONCEPTO LOGÍSTICO: Índice de Estacionalidad
# Muestra qué tan por encima o por debajo del promedio
# está cada mes. Un índice de 2.0 en diciembre significa
# que diciembre vende el doble que el promedio anual.
# Con esto podés planificar compras y personal con anticipación.

def grafica_estacionalidad(df):
    print("📊 Generando gráfica 2: Estacionalidad promedio...")

    # Calculamos el promedio de ventas por mes (todos los productos juntos)
    meses_orden = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    estacional = (
        df.groupby("mes")["cantidad_vendida"]
        .mean()
        .reset_index()
    )
    estacional["mes_nombre"] = pd.to_datetime(
        estacional["mes"], format="%m"
    ).dt.strftime("%b")

    # Calculamos el índice: cada mes dividido por el promedio general
    promedio_global = estacional["cantidad_vendida"].mean()
    estacional["indice"] = estacional["cantidad_vendida"] / promedio_global

    # Color: verde si está sobre el promedio, rojo si está debajo
    estacional["color"] = estacional["indice"].apply(
        lambda x: "#4CAF50" if x >= 1 else "#F44336"
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    bars = ax.bar(
        estacional["mes_nombre"],
        estacional["indice"],
        color=estacional["color"],
        edgecolor="white",
        linewidth=0.5
    )

    # Línea de promedio
    ax.axhline(y=1, color="gray", linestyle="--",
               linewidth=1.5, label="Promedio base (1.0)")

    # Etiquetas sobre cada barra
    for bar, val in zip(bars, estacional["indice"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}x",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    ax.set_title(
        "Índice de Estacionalidad Mensual — TechZone\n"
        "(1.0 = promedio, 2.0 = doble del promedio)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Mes", fontsize=11)
    ax.set_ylabel("Índice de Estacionalidad", fontsize=11)
    ax.legend()
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    ruta = OUT_PATH / "02_estacionalidad.png"
    plt.savefig(ruta, dpi=150)
    plt.show()
    print(f"   ✅ Guardada en: {ruta}\n")


# ── FUNCIÓN 4: INGRESOS POR PRODUCTO ────────────────────────
# CONCEPTO LOGÍSTICO: Análisis ABC
# Clasifica productos según su contribución a los ingresos:
# Categoría A → 20% de productos que generan 80% ingresos
# Categoría B → productos de importancia media
# Categoría C → muchos productos, poco impacto en ingresos
# Esto determina cuánto stock de seguridad mantener de cada uno.

def grafica_ingresos_producto(df):
    print("📊 Generando gráfica 3: Ingresos totales por producto...")

    ingresos = (
        df.groupby("producto")["ingreso_total"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    ingresos["ingreso_millones"] = ingresos["ingreso_total"] / 1_000_000
    ingresos["porcentaje"] = (
        ingresos["ingreso_total"] / ingresos["ingreso_total"].sum() * 100
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    bars = ax.barh(
        ingresos["producto"],
        ingresos["ingreso_millones"],
        color=COLORES,
        edgecolor="white"
    )

    # Etiquetas con valor y porcentaje
    for bar, (_, row) in zip(bars, ingresos.iterrows()):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"${row['ingreso_millones']:.2f}M  ({row['porcentaje']:.1f}%)",
            va="center", fontsize=9
        )

    ax.set_title(
        "Ingresos Totales por Producto 2022-2024 (Análisis ABC)",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Ingresos en Millones ($)", fontsize=11)
    ax.set_xlim(0, ingresos["ingreso_millones"].max() * 1.35)

    plt.tight_layout()
    ruta = OUT_PATH / "03_ingresos_producto.png"
    plt.savefig(ruta, dpi=150)
    plt.show()
    print(f"   ✅ Guardada en: {ruta}\n")


# ── FUNCIÓN 5: RESUMEN ESTADÍSTICO ──────────────────────────
def resumen_estadistico(df):
    print("=" * 55)
    print("📋 RESUMEN ESTADÍSTICO — TECHZONE 2022-2024")
    print("=" * 55)

    total_ingresos = df["ingreso_total"].sum()
    total_unidades = df["cantidad_vendida"].sum()
    mejor_mes = df.groupby("mes")["cantidad_vendida"].mean().idxmax()
    peor_mes  = df.groupby("mes")["cantidad_vendida"].mean().idxmin()

    meses = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",
             5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",
             9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}

    print(f"  💰 Ingresos totales:   ${total_ingresos:,.2f}")
    print(f"  📦 Unidades vendidas:  {total_unidades:,}")
    print(f"  📈 Mes más fuerte:     {meses[mejor_mes]}")
    print(f"  📉 Mes más débil:      {meses[peor_mes]}")
    print()

    print("  Top producto por ingresos:")
    top = df.groupby("producto")["ingreso_total"].sum().idxmax()
    val = df.groupby("producto")["ingreso_total"].sum().max()
    print(f"  🏆 {top}: ${val:,.2f}")
    print("=" * 55)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 TechZone Forecast System — Fase 2: EDA\n")

    df = cargar_datos()
    resumen_estadistico(df)
    grafica_evolucion_ventas(df)
    grafica_estacionalidad(df)
    grafica_ingresos_producto(df)

    print("✅ Fase 2 completada.")
    print(f"   Gráficas guardadas en: {OUT_PATH}")