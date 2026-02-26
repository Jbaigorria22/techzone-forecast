# ============================================================
# FASE 1: Creación de BD y generación de datos históricos
# TechZone Forecast System
# ============================================================

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import os

# ── CONFIGURACIÓN DE RUTAS ───────────────────────────────────
# Path(__file__) obtiene la ubicación de ESTE archivo Python.
# .parent sube un nivel (de src/ a techzone_forecast/)
# .parent otra vez sube otro nivel — así encontramos la raíz
# sin importar en qué computadora corra el proyecto.

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "data" / "techzone.db"
os.makedirs(BASE_DIR / "data", exist_ok=True)


# ── FUNCIÓN 1: CREAR TABLAS ──────────────────────────────────
def crear_base_de_datos(conn):
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS productos (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre       TEXT    NOT NULL,
            categoria    TEXT    NOT NULL,
            precio_base  REAL    NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ventas (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            producto_id       INTEGER NOT NULL,
            fecha             TEXT    NOT NULL,
            cantidad_vendida  INTEGER NOT NULL,
            ingreso_total     REAL    NOT NULL,
            FOREIGN KEY (producto_id) REFERENCES productos(id)
        )
    """)

    conn.commit()
    print("✅ Tablas creadas correctamente.")


# ── FUNCIÓN 2: INSERTAR PRODUCTOS ───────────────────────────
def insertar_productos(conn):
    productos = [
        ("Smartphone Samsung A55",   "Smartphones", 750.00),
        ("Laptop Lenovo IdeaPad",    "Laptops",    1200.00),
        ("Auriculares Sony WH-1000", "Audio",        350.00),
        ("Tablet iPad Air",          "Tablets",      900.00),
        ("Cable USB-C Premium",      "Accesorios",    25.00),
    ]
    cursor = conn.cursor()
    cursor.executemany("""
        INSERT OR IGNORE INTO productos (nombre, categoria, precio_base)
        VALUES (?, ?, ?)
    """, productos)
    conn.commit()
    print(f"✅ {len(productos)} productos insertados.")


# ── FUNCIÓN 3: GENERAR DEMANDA ───────────────────────────────
# CONCEPTO LOGÍSTICO CLAVE:
# Demanda real = Tendencia + Estacionalidad + Ruido
#
# Tendencia:      el mercado tech crece ~15% anual
# Estacionalidad: Black Friday y Navidad duplican las ventas
# Ruido:          variación impredecible de cada mes
#
# Si tu modelo ignora cualquiera de estos tres componentes,
# sus predicciones van a fallar sistemáticamente.

def generar_demanda(base, tendencia_anual, mes, anio, ruido_std):
    anios_desde_base = anio - 2022
    factor_tendencia = tendencia_anual ** anios_desde_base

    estacionalidad = {
        1: 0.70, 2: 0.75, 3: 0.85, 4: 0.90,
        5: 0.95, 6: 1.00, 7: 1.05, 8: 1.00,
        9: 1.10, 10: 1.20, 11: 1.80, 12: 2.00
    }

    demanda_esperada = base * factor_tendencia * estacionalidad[mes]
    ruido = np.random.normal(0, ruido_std)
    return max(1, int(demanda_esperada + ruido))


# ── FUNCIÓN 4: POBLAR VENTAS ─────────────────────────────────
# Generamos 3 años completos (2022-2024).
# Regla en forecasting: necesitás al menos 2 ciclos anuales
# completos para que el modelo detecte la estacionalidad.

def poblar_ventas(conn):
    config = {
        1: {"base": 80,  "tendencia": 1.18, "ruido": 10},
        2: {"base": 40,  "tendencia": 1.12, "ruido": 6},
        3: {"base": 120, "tendencia": 1.20, "ruido": 15},
        4: {"base": 55,  "tendencia": 1.15, "ruido": 8},
        5: {"base": 300, "tendencia": 1.10, "ruido": 40},
    }

    cursor = conn.cursor()
    np.random.seed(42)
    total = 0

    for anio in range(2022, 2025):
        for mes in range(1, 13):
            fecha = f"{anio}-{mes:02d}-01"
            for prod_id, cfg in config.items():
                cantidad = generar_demanda(
                    cfg["base"], cfg["tendencia"],
                    mes, anio, cfg["ruido"]
                )
                cursor.execute(
                    "SELECT precio_base FROM productos WHERE id = ?",
                    (prod_id,)
                )
                precio = cursor.fetchone()[0] * np.random.uniform(0.95, 1.05)
                ingreso = round(cantidad * precio, 2)

                cursor.execute("""
                    INSERT INTO ventas
                        (producto_id, fecha, cantidad_vendida, ingreso_total)
                    VALUES (?, ?, ?, ?)
                """, (prod_id, fecha, cantidad, ingreso))
                total += 1

    conn.commit()
    print(f"✅ {total} registros de ventas generados.")


# ── FUNCIÓN 5: VERIFICAR ─────────────────────────────────────
def verificar(conn):
    query = """
        SELECT
            p.nombre                       AS producto,
            strftime('%Y', v.fecha)        AS anio,
            SUM(v.cantidad_vendida)        AS unidades,
            ROUND(SUM(v.ingreso_total), 2) AS ingresos
        FROM ventas v
        JOIN productos p ON p.id = v.producto_id
        GROUP BY p.nombre, anio
        ORDER BY p.nombre, anio
    """
    df = pd.read_sql_query(query, conn)
    print("\n📊 Resumen de ventas por producto y año:")
    print(df.to_string(index=False))


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Iniciando TechZone Forecast System — Fase 1\n")

    with sqlite3.connect(DB_PATH) as conn:
        crear_base_de_datos(conn)
        insertar_productos(conn)
        poblar_ventas(conn)
        verificar(conn)

    print(f"\n📁 Base de datos guardada en: {DB_PATH}")
    print("✅ Fase 1 completada. Listo para el análisis exploratorio.")