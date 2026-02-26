# 📦 TechZone Forecast System

Sistema de predicción de demanda para retail tecnológico construido
con Machine Learning y conceptos de Supply Chain.

## Demo en vivo
🔗 [Ver aplicación]

## ¿Qué hace este sistema?

Predice cuántas unidades va a vender una tienda de tecnología
en los próximos 12 meses, usando datos históricos y Machine Learning.

## Funcionalidades

- **Forecast ML** con Prophet — predicciones mensuales con intervalo de confianza del 80%
- **Análisis ABC** — clasificación de productos por importancia de ingresos (regla 80/20)
- **Alertas de Stock** — días de cobertura, punto de reorden y EOQ automático
- **Dashboard interactivo** — visualizaciones profesionales con diseño oscuro
- **Export Excel** — reportes completos en 4 hojas para presentar a gerencia

## Tecnologías utilizadas

| Tecnología | Uso |
|---|---|
| Python | Lenguaje principal |
| SQLite | Base de datos de historial de ventas |
| Pandas | Procesamiento y análisis de datos |
| Prophet (Meta) | Modelo de forecasting con estacionalidad |
| Streamlit | Dashboard web interactivo |
| Matplotlib | Visualizaciones y gráficas |
| OpenPyXL | Generación de reportes Excel |

## Resultados del modelo

| Modelo | MAPE | MAE |
|---|---|---|
| Promedio Móvil Simple | 25.2% | 25.4 u |
| Suavizamiento Exponencial | 24.7% | 25.9 u |
| **Prophet (seleccionado)** | **3.9%** | **3.2 u** |

Prophet redujo el error de predicción un **83%** respecto a los modelos simples.

## Conceptos de Supply Chain aplicados

- **Forecasting de demanda** — predicción basada en tendencia + estacionalidad + ruido
- **Análisis ABC** — clasificación de inventario por importancia económica
- **Punto de Reorden (ROP)** — cuándo hacer el próximo pedido al proveedor
- **EOQ** — cantidad económica de pedido que minimiza costos
- **Lead Time** — tiempo de reposición del proveedor
- **Stock de Seguridad** — colchón ante variaciones de demanda

## Estructura del proyecto
```
techzone_forecast/
├── data/
│   └── techzone.db          ← Base de datos SQLite
├── src/
│   ├── 01_setup_database.py ← Generación de datos históricos
│   ├── 02_eda.py            ← Análisis exploratorio
│   ├── 03_forecast.py       ← Modelos simples
│   └── 04_prophet_forecast.py ← Modelo Prophet
├── outputs/
│   └── graficas/            ← Gráficas generadas
├── dashboard.py             ← Aplicación Streamlit
├── requirements.txt         ← Dependencias
└── README.md                ← Este archivo
```

## Cómo ejecutar localmente
```bash
git clone https://github.com/TU_USUARIO/techzone_forecast
cd techzone_forecast
pip install -r requirements.txt
python src/01_setup_database.py
streamlit run dashboard.py
```

## Autor

Desarrollado como proyecto de aprendizaje de Supply Chain + ML.