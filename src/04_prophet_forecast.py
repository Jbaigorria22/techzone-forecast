# ============================================================
# FASE 4: Forecasting Avanzado con Prophet
# TechZone Forecast System
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from prophet import Prophet
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIGURACIÓN ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "techzone.db"
OUT_PATH = BASE_DIR / "outputs" / "graficas"
os.makedirs(OUT_PATH, exist_ok=True)

PRODUCTO = "Smartphone Samsung A55"


# ── FUNCIÓN 1: CARGAR Y PREPARAR DATOS ──────────────────────
# CONCEPTO CLAVE:
# Prophet exige un formato específico de DataFrame:
#   columna "ds" → fechas (date series)
#   columna "y"  → valores a predecir
# Este es el contrato de Prophet. Si no respetás estos
# nombres exactos, el modelo falla. Es una convención
# que viene del equipo de Meta que lo desarrolló.

def cargar_datos_prophet(producto):
    print(f"📂 Cargando datos para Prophet: {producto}\n")

    with sqlite3.connect(DB_PATH) as conn:
        query = """
            SELECT v.fecha, v.cantidad_vendida
            FROM ventas v
            JOIN productos p ON p.id = v.producto_id
            WHERE p.nombre = ?
            ORDER BY v.fecha
        """
        df = pd.read_sql_query(query, conn, params=(producto,))

    # Renombramos al formato que exige Prophet
    df = df.rename(columns={
        "fecha":            "ds",
        "cantidad_vendida": "y"
    })
    df["ds"] = pd.to_datetime(df["ds"])

    print(f"✅ Datos listos para Prophet")
    print(f"   Registros: {len(df)} meses")
    print(f"   Desde: {df['ds'].min().strftime('%Y-%m')}")
    print(f"   Hasta: {df['ds'].max().strftime('%Y-%m')}")
    print(f"   Promedio mensual: {df['y'].mean():.0f} unidades\n")

    return df


# ── FUNCIÓN 2: ENTRENAR MODELO PROPHET ──────────────────────
# CONCEPTO LOGÍSTICO:
# Los parámetros que configuramos en Prophet tienen
# significado de negocio directo:
#
# yearly_seasonality=True  → Le decimos que hay ciclos anuales
#                            (Black Friday, Navidad, enero flojo)
#
# seasonality_mode="multiplicative" → La estacionalidad multiplica
#                            la tendencia. En retail tech los picos
#                            son proporcionales al volumen base.
#                            Si la tendencia crece, los picos también.
#                            (vs "additive" donde los picos son fijos)
#
# changepoint_prior_scale  → Qué tan flexible es la tendencia.
#                            0.1 = tendencia rígida
#                            0.5 = tendencia flexible
#                            Para tech usamos 0.3 — mercado dinámico
#                            pero no caótico.

def entrenar_prophet(df):
    print("🧠 Entrenando modelo Prophet...")
    print("   Parámetros:")
    print("   • Estacionalidad anual:       activada")
    print("   • Modo estacionalidad:        multiplicativo")
    print("   • Flexibilidad de tendencia:  0.3\n")

    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # Datos mensuales, no necesitamos semanal
        daily_seasonality=False,    # Datos mensuales, no necesitamos diario
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.3,
        interval_width=0.80         # Intervalo de confianza del 80%
    )

    modelo.fit(df)
    print("✅ Modelo entrenado correctamente.\n")
    return modelo


# ── FUNCIÓN 3: GENERAR FORECAST ──────────────────────────────
# make_future_dataframe genera las fechas futuras
# sobre las que el modelo va a predecir.
# periods=12 → 12 meses hacia adelante (todo 2025)
# freq="MS"  → Month Start (primer día de cada mes)

def generar_forecast(modelo, df, periodos=12):
    print(f"🔮 Generando forecast para {periodos} meses...")

    futuro    = modelo.make_future_dataframe(periods=periodos, freq="MS")
    forecast  = modelo.predict(futuro)

    # Las columnas más importantes del resultado:
    # ds          → fecha
    # yhat        → predicción central
    # yhat_lower  → límite inferior del intervalo de confianza
    # yhat_upper  → límite superior del intervalo de confianza

    # Aseguramos que no haya predicciones negativas
    forecast["yhat"]       = forecast["yhat"].clip(lower=0).round()
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0).round()
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0).round()

    print("✅ Forecast generado.\n")
    return forecast


# ── FUNCIÓN 4: MÉTRICAS PROPHET vs MODELOS SIMPLES ──────────
# Comparamos Prophet contra los modelos de la Fase 3
# usando los mismos datos históricos.
# Esto nos dice cuánto mejoró el modelo avanzado.

def calcular_metricas_prophet(df, forecast):
    print("📏 Calculando métricas de Prophet...")

    historico = forecast[forecast["ds"] <= df["ds"].max()].copy()
    real      = df["y"].values
    predicho  = historico["yhat"].values[:len(real)]

    mae  = np.mean(np.abs(real - predicho))
    rmse = np.sqrt(np.mean((real - predicho) ** 2))
    mape = np.mean(np.abs((real - predicho) / real)) * 100

    print(f"   MAE:  {mae:.1f} unidades")
    print(f"   RMSE: {rmse:.1f}")
    print(f"   MAPE: {mape:.1f}%")
    print()

    print("📊 Comparación con modelos de Fase 3:")
    print("-" * 45)
    print(f"   {'Modelo':<30} {'MAPE':>8}")
    print("-" * 45)
    print(f"   {'Promedio Movil Simple':<30} {'25.2%':>8}")
    print(f"   {'Promedio Movil Ponderado':<30} {'23.3%':>8}")
    print(f"   {'Suavizamiento Exponencial':<30} {'24.7%':>8}")
    print(f"   {'Prophet <- (Fase 4)':<30} {str(round(mape,1))+'%':>8}")
    print("-" * 45)

    if mape < 23.3:
        mejora = 23.3 - mape
        print(f"   Prophet mejora {mejora:.1f} puntos porcentuales")
    else:
        print(f"   Con mas datos Prophet mostraria mayor ventaja")
    print()

    return mae, rmse, mape


# ── FUNCIÓN 5: GRÁFICA PRINCIPAL PROPHET ────────────────────
def grafica_prophet_forecast(df, forecast):
    print("📊 Generando gráfica principal de Prophet...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Separamos histórico del forecast futuro
    historico_fore = forecast[forecast["ds"] <= df["ds"].max()]
    futuro_fore    = forecast[forecast["ds"] >  df["ds"].max()]

    # Datos reales
    ax.plot(df["ds"], df["y"],
            color="white", linewidth=2.5,
            label="Ventas reales", zorder=5)

    # Ajuste histórico de Prophet
    ax.plot(historico_fore["ds"], historico_fore["yhat"],
            color="#4CAF50", linewidth=1.5, linestyle="--",
            label="Ajuste Prophet", alpha=0.8)

    # Banda de confianza histórica
    ax.fill_between(
        historico_fore["ds"],
        historico_fore["yhat_lower"],
        historico_fore["yhat_upper"],
        alpha=0.15, color="#4CAF50"
    )

    # Forecast 2025
    ax.plot(futuro_fore["ds"], futuro_fore["yhat"],
            color="#FF5722", linewidth=2.5,
            marker="o", markersize=6,
            label="Forecast 2025 (Prophet)", zorder=4)

    # Banda de confianza del forecast
    # CONCEPTO LOGÍSTICO: Esta banda es tu rango de planificación.
    # yhat_lower → stock mínimo que deberías tener
    # yhat_upper → stock máximo recomendado
    ax.fill_between(
        futuro_fore["ds"],
        futuro_fore["yhat_lower"],
        futuro_fore["yhat_upper"],
        alpha=0.20, color="#FF5722",
        label="Intervalo de confianza 80%"
    )

    # Línea divisoria histórico/forecast
    ax.axvline(x=df["ds"].max(), color="gray",
               linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(df["ds"].max(), ax.get_ylim()[1] * 0.95,
            " ← Histórico | Forecast →",
            fontsize=9, color="gray", va="top")

    ax.set_title(
        f"Prophet Forecast — {PRODUCTO}\n"
        f"Histórico 2022-2024 + Predicción 2025",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Unidades Vendidas", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    ruta = OUT_PATH / "06_prophet_forecast.png"
    plt.savefig(ruta, dpi=150)
    plt.show()
    print(f"   ✅ Guardada en: {ruta}\n")


# ── FUNCIÓN 6: GRÁFICA DE COMPONENTES ───────────────────────
# Esta es la gráfica más valiosa de Prophet para logística.
# Descompone la predicción en sus partes:
#   Tendencia    → ¿el negocio crece o decrece?
#   Estacionalidad → ¿qué meses son fuertes/débiles?
# Con esto podés planificar compras con 3-6 meses de anticipación.

def grafica_componentes(modelo, forecast):
    print("📊 Generando gráfica de componentes...")

    fig = modelo.plot_components(forecast)
    fig.set_size_inches(12, 7)
    fig.suptitle(
        "Descomposición del Forecast — Tendencia y Estacionalidad",
        fontsize=13, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    ruta = OUT_PATH / "07_prophet_componentes.png"
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"   ✅ Guardada en: {ruta}\n")


# ── FUNCIÓN 7: TABLA FORECAST 2025 ──────────────────────────
def tabla_forecast_2025(forecast, df):
    futuro = forecast[forecast["ds"] > df["ds"].max()].copy()
    futuro["mes"] = futuro["ds"].dt.strftime("%b %Y")

    print("=" * 62)
    print(f"  📅 FORECAST 2025 — Prophet")
    print(f"  {PRODUCTO}")
    print("=" * 62)
    print(f"  {'Mes':<12} {'Mínimo':>9} {'Predicción':>12} {'Máximo':>9}")
    print("-" * 62)

    total_min  = 0
    total_pred = 0
    total_max  = 0

    for _, row in futuro.iterrows():
        print(
            f"  {row['mes']:<12} "
            f"{int(row['yhat_lower']):>9,} "
            f"{int(row['yhat']):>12,} "
            f"{int(row['yhat_upper']):>9,}"
        )
        total_min  += int(row["yhat_lower"])
        total_pred += int(row["yhat"])
        total_max  += int(row["yhat_upper"])

    print("=" * 62)
    print(
        f"  {'TOTAL 2025':<12} "
        f"{total_min:>9,} "
        f"{total_pred:>12,} "
        f"{total_max:>9,}"
    )
    print("=" * 62)
    print()
    print("  💡 Interpretación logística:")
    print(f"     Stock mínimo anual recomendado:  {total_min:,} unidades")
    print(f"     Forecast central (más probable): {total_pred:,} unidades")
    print(f"     Stock máximo (escenario alto):   {total_max:,} unidades")
    print("=" * 62)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 TechZone Forecast System — Fase 4: Prophet\n")
    print("=" * 55)

    # 1. Cargar datos
    df = cargar_datos_prophet(PRODUCTO)

    # 2. Entrenar modelo
    modelo = entrenar_prophet(df)

    # 3. Generar forecast
    forecast = generar_forecast(modelo, df, periodos=12)

    # 4. Métricas y comparación
    calcular_metricas_prophet(df, forecast)

    # 5. Gráficas
    grafica_prophet_forecast(df, forecast)
    grafica_componentes(modelo, forecast)

    # 6. Tabla final
    tabla_forecast_2025(forecast, df)

    print("\n✅ Fase 4 completada.")
    print("   Prophet genera forecast con intervalo de confianza.")
    print("   Siguiente paso: Dashboard interactivo con todo junto.")