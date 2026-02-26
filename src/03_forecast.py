# ============================================================
# FASE 3: Modelos de Forecasting
# TechZone Forecast System
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIGURACIÓN ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "techzone.db"
OUT_PATH = BASE_DIR / "outputs" / "graficas"
os.makedirs(OUT_PATH, exist_ok=True)

sns.set_theme(style="darkgrid")
PRODUCTO_ANALISIS = "Smartphone Samsung A55"


# ── FUNCIÓN 1: CARGAR SERIE DE TIEMPO ───────────────────────
# Cargamos solo el producto que vamos a analizar.
# Lo convertimos en una Serie de Tiempo mensual indexada por fecha.
# Esto es la estructura de datos base de TODO forecasting.

def cargar_serie(producto):
    print(f"📂 Cargando serie de tiempo: {producto}...\n")

    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT v.fecha, v.cantidad_vendida
            FROM ventas v
            JOIN productos p ON p.id = v.producto_id
            WHERE p.nombre = ?
            ORDER BY v.fecha
        """
        df = pd.read_sql_query(query, conn, params=(producto,))

    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha")
    df.index = df.index.to_period("M")

    print(f"✅ Serie cargada: {len(df)} meses de historial")
    print(f"   Desde: {df.index[0]}  →  Hasta: {df.index[-1]}")
    print(f"   Promedio mensual: {df['cantidad_vendida'].mean():.0f} unidades")
    print(f"   Máximo histórico: {df['cantidad_vendida'].max()} unidades")
    print(f"   Mínimo histórico: {df['cantidad_vendida'].min()} unidades\n")

    return df


# ── FUNCIÓN 2: PROMEDIO MÓVIL SIMPLE ────────────────────────
# CONCEPTO LOGÍSTICO:
# El Promedio Móvil Simple (SMA) es el modelo más básico.
# Toma los últimos 'ventana' meses y los promedia.
# Ventana de 3 meses = reacciona rápido pero es inestable.
# Ventana de 12 meses = muy estable pero reacciona lento.
# En logística se usa para productos con demanda estable
# sin tendencia ni estacionalidad marcada.

def promedio_movil_simple(serie, ventana=3):
    print(f"🔄 Calculando Promedio Móvil Simple (ventana={ventana})...")

    valores = serie["cantidad_vendida"].values
    predicciones = []

    # Para cada punto, predecimos con el promedio de los N anteriores
    for i in range(len(valores)):
        if i < ventana:
            # Sin suficiente historia, usamos lo que hay
            pred = np.mean(valores[:i+1])
        else:
            pred = np.mean(valores[i-ventana:i])
        predicciones.append(round(pred))

    # Forecast para los próximos 12 meses (2025)
    forecast_2025 = []
    ultimos = list(valores[-ventana:])

    for _ in range(12):
        pred = round(np.mean(ultimos[-ventana:]))
        forecast_2025.append(pred)
        ultimos.append(pred)

    return predicciones, forecast_2025


# ── FUNCIÓN 3: PROMEDIO MÓVIL PONDERADO ─────────────────────
# CONCEPTO LOGÍSTICO:
# El Promedio Móvil Ponderado (WMA) le asigna más peso
# a los meses recientes. Si la ventana es 3:
#   mes más reciente  → peso 3
#   hace 2 meses      → peso 2
#   hace 3 meses      → peso 1
#   total pesos       → 6
# Más sensible a cambios recientes de demanda.
# Útil cuando el mercado está en transición.

def promedio_movil_ponderado(serie, ventana=3):
    print(f"🔄 Calculando Promedio Móvil Ponderado (ventana={ventana})...")

    valores = serie["cantidad_vendida"].values
    pesos   = np.arange(1, ventana + 1)  # [1, 2, 3]
    predicciones = []

    for i in range(len(valores)):
        if i < ventana:
            p = np.arange(1, i + 2)
            pred = round(np.average(valores[:i+1], weights=p))
        else:
            pred = round(np.average(valores[i-ventana:i], weights=pesos))
        predicciones.append(pred)

    forecast_2025 = []
    ultimos = list(valores[-ventana:])

    for _ in range(12):
        pred = round(np.average(ultimos[-ventana:], weights=pesos))
        forecast_2025.append(pred)
        ultimos.append(pred)

    return predicciones, forecast_2025


# ── FUNCIÓN 4: SUAVIZAMIENTO EXPONENCIAL ────────────────────
# CONCEPTO LOGÍSTICO:
# El Suavizamiento Exponencial Simple (SES) es el modelo
# estándar en sistemas ERP para forecasting operativo.
# La fórmula es:
#
#   F(t) = alpha × Demanda(t-1) + (1-alpha) × F(t-1)
#
# Donde alpha (0 a 1) controla la "memoria" del modelo:
#   alpha alto (0.8) → reacciona rápido, olvida el pasado
#   alpha bajo (0.2) → reacciona lento, memoria larga
#
# SAP Business One usa alpha=0.3 como valor por defecto.
# Nosotros vamos a comparar tres valores de alpha.

def suavizamiento_exponencial(serie, alpha=0.3):
    print(f"🔄 Calculando Suavizamiento Exponencial (alpha={alpha})...")

    valores      = serie["cantidad_vendida"].values
    predicciones = [valores[0]]  # El primer valor es la semilla

    for i in range(1, len(valores)):
        pred = alpha * valores[i-1] + (1 - alpha) * predicciones[i-1]
        predicciones.append(round(pred))

    # Forecast 2025: seguimos aplicando la fórmula
    ultimo_real = valores[-1]
    ultimo_pred = predicciones[-1]
    forecast_2025 = []

    for _ in range(12):
        pred = round(alpha * ultimo_real + (1 - alpha) * ultimo_pred)
        forecast_2025.append(pred)
        ultimo_real = pred
        ultimo_pred = pred

    return predicciones, forecast_2025


# ── FUNCIÓN 5: MÉTRICAS DE ERROR ────────────────────────────
# CONCEPTO LOGÍSTICO:
# Estas métricas responden: ¿qué tan malo es nuestro modelo?
#
# MAE  → Error Absoluto Medio: promedio de cuántas unidades
#         nos equivocamos. Fácil de explicar al gerente.
#         "Nos equivocamos en promedio 25 unidades por mes."
#
# RMSE → Penaliza más los errores grandes. Si un mes
#         nos equivocamos por 200 unidades, eso es grave.
#
# MAPE → Error porcentual. "Nos equivocamos el 12% en promedio."
#         El más usado en reportes ejecutivos de supply chain.

def calcular_metricas(real, predicho, nombre_modelo):
    real    = np.array(real,    dtype=float)
    predicho = np.array(predicho, dtype=float)

    mae  = np.mean(np.abs(real - predicho))
    rmse = np.sqrt(np.mean((real - predicho) ** 2))
    mape = np.mean(np.abs((real - predicho) / real)) * 100

    print(f"   📏 {nombre_modelo}:")
    print(f"      MAE:  {mae:.1f} unidades")
    print(f"      RMSE: {rmse:.1f}")
    print(f"      MAPE: {mape:.1f}%")
    print()

    return {"modelo": nombre_modelo, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ── FUNCIÓN 6: GRÁFICA COMPARATIVA ──────────────────────────
def grafica_comparativa(serie, modelos_hist, modelos_fore):
    print("📊 Generando gráfica comparativa de modelos...")

    fechas_hist  = serie.index.to_timestamp()
    valores_real = serie["cantidad_vendida"].values

    # Generamos fechas para 2025
    ultimo = serie.index[-1]
    fechas_fore = pd.period_range(
        start=ultimo + 1, periods=12, freq="M"
    ).to_timestamp()

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    colores_modelos = {
        "SMA":  "#2196F3",
        "WMA":  "#FF9800",
        "SES":  "#4CAF50",
    }

    titulos = [
        "Modelo 1: Promedio Móvil Simple (SMA)",
        "Modelo 2: Promedio Móvil Ponderado (WMA)",
        "Modelo 3: Suavizamiento Exponencial (SES α=0.3)",
    ]
    claves = ["SMA", "WMA", "SES"]

    for i, ax in enumerate(axes):
        clave  = claves[i]
        color  = colores_modelos[clave]

        # Datos reales
        ax.plot(fechas_hist, valores_real,
                color="white", linewidth=2,
                label="Ventas reales", zorder=3)

        # Predicciones históricas
        ax.plot(fechas_hist, modelos_hist[clave],
                color=color, linewidth=1.5, linestyle="--",
                label=f"Ajuste {clave}", alpha=0.8)

        # Forecast 2025
        ax.plot(fechas_fore, modelos_fore[clave],
                color=color, linewidth=2.5, linestyle="-",
                marker="o", markersize=5,
                label="Forecast 2025", zorder=4)

        # Zona de forecast sombreada
        ax.axvspan(fechas_fore[0], fechas_fore[-1],
                   alpha=0.08, color=color)

        ax.set_title(titulos[i], fontsize=12,
                     fontweight="bold", pad=10)
        ax.set_ylabel("Unidades", fontsize=10)
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    axes[-1].set_xlabel("Fecha", fontsize=11)
    fig.suptitle(
        f"Comparación de Modelos de Forecast\n{PRODUCTO_ANALISIS}",
        fontsize=14, fontweight="bold", y=1.01
    )

    plt.tight_layout()
    ruta = OUT_PATH / "04_forecast_comparativo.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Guardada en: {ruta}\n")


# ── FUNCIÓN 7: TABLA FORECAST 2025 ──────────────────────────
def tabla_forecast_2025(forecasts):
    meses = ["Ene","Feb","Mar","Abr","May","Jun",
             "Jul","Ago","Sep","Oct","Nov","Dic"]

    df = pd.DataFrame({
        "Mes":  meses,
        "SMA":  forecasts["SMA"],
        "WMA":  forecasts["WMA"],
        "SES":  forecasts["SES"],
    })

    df["Promedio"] = df[["SMA","WMA","SES"]].mean(axis=1).round(0).astype(int)

    print("=" * 57)
    print(f"  📅 FORECAST 2025 — {PRODUCTO_ANALISIS}")
    print("=" * 57)
    print(f"  {'Mes':<6} {'SMA':>7} {'WMA':>7} {'SES':>7} {'Promedio':>9}")
    print("-" * 57)
    for _, row in df.iterrows():
        print(f"  {row['Mes']:<6} {row['SMA']:>7} "
              f"{row['WMA']:>7} {row['SES']:>7} {row['Promedio']:>9}")
    print("=" * 57)
    total = df["Promedio"].sum()
    print(f"  {'TOTAL':<6} {'':>7} {'':>7} {'':>7} {total:>9,}")
    print("=" * 57)

    return df


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 TechZone Forecast System — Fase 3: Forecasting\n")
    print("=" * 55)

    # 1. Cargar datos
    serie = cargar_serie(PRODUCTO_ANALISIS)

    # 2. Entrenar modelos
    sma_hist, sma_fore = promedio_movil_simple(serie,       ventana=3)
    wma_hist, wma_fore = promedio_movil_ponderado(serie,    ventana=3)
    ses_hist, ses_fore = suavizamiento_exponencial(serie,   alpha=0.3)
    print()

    # 3. Métricas de error
    real = serie["cantidad_vendida"].values
    print("📏 MÉTRICAS DE ERROR (qué tan bien ajusta cada modelo):")
    print("-" * 55)
    m_sma = calcular_metricas(real, sma_hist, "Promedio Móvil Simple")
    m_wma = calcular_metricas(real, wma_hist, "Promedio Móvil Ponderado")
    m_ses = calcular_metricas(real, ses_hist, "Suavizamiento Exponencial")

    # 4. Ganador
    metricas = [m_sma, m_wma, m_ses]
    ganador  = min(metricas, key=lambda x: x["MAPE"])
    print(f"🏆 Mejor modelo por MAPE: {ganador['modelo']} "
          f"({ganador['MAPE']:.1f}%)\n")

    # 5. Gráfica comparativa
    modelos_hist = {"SMA": sma_hist, "WMA": wma_hist, "SES": ses_hist}
    modelos_fore = {"SMA": sma_fore, "WMA": wma_fore, "SES": ses_fore}
    grafica_comparativa(serie, modelos_hist, modelos_fore)

    # 6. Tabla forecast 2025
    tabla_forecast_2025(modelos_fore)

    print("\n✅ Fase 3 completada.")
    print("   Siguiente paso: agregar modelos avanzados (Prophet / LSTM)")