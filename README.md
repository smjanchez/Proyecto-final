# Proyecto-final

# 🏠 Properati Price Predictor

Proyecto final del curso **Escuela de Datos Vivos – Batch 6**
Desarrollado por **Shirly Janchez**

---

## 🎯 Objetivo
Construir un modelo de *Machine Learning* capaz de predecir el **precio en USD** de propiedades publicadas en Properati, a partir de sus características estructurales y de ubicación.

---

## Tecnologías utilizadas
- Python 3.10
- pandas, numpy
- scikit-learn
- XGBoost
- Gradio
- Hugging Face Spaces

---

##  Fase 1 — EDA y preparación de datos

Se trabajó con propiedades de **Argentina**, filtrando solo aquellas con:
- `operation_type = "Venta"`
- `currency = "USD"`
- Zonas: **CABA** y **GBA**

### Variables seleccionadas:
- `rooms`, `bedrooms`, `bathrooms`
- `surface_total`, `surface_covered`
- `property_type`, `state_name`, `place_name`

### Feature engineering:
- Eliminación de duplicados y nulos irrelevantes
- Imputación de medianas en superficies y baños
- Creación de ratios: `surface_ratio`, `rooms_per_bath`

---

## 🤖 Fase 2 — Modelado y evaluación

Se probaron distintos modelos de regresión:

| Modelo | R² | RMSE (USD) | MAE (USD) | Observaciones |
|------------------------|------|-------------|------------|----------------|
| Linear Regression | 0.65 | 94,000 | 54,000 | Subestima precios altos |
| Random Forest | 0.81 | 76,700 | 42,400 | Buen equilibrio |
| Random Forest Tuned | **0.83** | **73,500** | **40,500** | Mejor performance |
| XGBoost | 0.81 | 75,900 | 43,300 | Similar rendimiento |

✅ **Modelo final elegido:** `RandomForestRegressor` afinado con `RandomizedSearchCV`
por su mejor trade-off entre precisión y estabilidad.

---

## 📊 Principales insights de negocio

- **CABA y Zona Norte del GBA** concentran los precios más altos del mercado.
- **Superficie cubierta y total** son los predictores más influyentes del precio.
- Propiedades en **Palermo, Tigre y Belgrano** presentan valores muy por encima del promedio.
- El modelo tiende a subestimar ligeramente los precios de propiedades de lujo, aunque mantiene estabilidad en el rango medio.

---

## 📈 Métricas finales del modelo
- **R²:** 0.83
- **RMSE:** 73,509 USD
- **MAE:** 40,509 USD

---

##  Fase 3 — Interfaz en Gradio

Se implementó una interfaz que permite ingresar los datos de una propiedad y obtener la predicción del precio estimado.

🔗 **Demo online:**
 [Properati Price Predictor — Hugging Face Space](https://huggingface.co/spaces/Smjanchez/properati-price-predictor)

📸 **Captura de la app:**
!<img width="1779" height="918" alt="image" src="https://github.com/user-attachments/assets/7b1c4895-da79-49a6-b6a3-aceb5b879326" />
)

---

##  Aprendizajes técnicos
- Creación de pipelines con *scikit-learn*
- Afinación de hiperparámetros con *RandomizedSearchCV* y *GridSearchCV*
- Implementación de interfaz web con *Gradio*
- Deploy completo en *Hugging Face Spaces*

---

##  Estructura del proyecto

```text
properati-price-predictor/
├── data/
│ └── processed_properati.csv
├── notebooks/
│ ├── 01_EDA_properati.ipynb
│ └── 02_Modelado_properati.ipynb
├── src/
│ ├── train_properati_final.py
│ ├── train_XGB.py
│ ├── graficos3.py
│ └── app.py
├── models/
│ └── model_properati_final.pkl
├── requirements.txt
└── README.md
