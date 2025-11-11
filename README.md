# Proyecto-final
# 🏠 Properati Price Predictor

Proyecto final del curso **Escuela de Datos Vivos – Batch 6**  
Desarrollado por **Shirly Janchez**

---

##  Objetivo
Construir un modelo de *Machine Learning* capaz de predecir el **precio en USD** de propiedades publicadas en Properati, a partir de sus características estructurales y de ubicación.

---

##  Tecnologías utilizadas
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
- Creación de variables derivadas:  
  - `surface_ratio = surface_covered / surface_total`  
  - `rooms_per_bath = rooms / bathrooms`

---

##  Principales insights de negocio

- **CABA y Zona Norte del GBA** concentran los precios más altos, impulsados por zonas como *Palermo, Belgrano, Recoleta y Tigre*.  
- La **superficie cubierta** es el factor con mayor peso predictivo: a mayor superficie, mayor precio.  
- Propiedades tipo **Casa y PH** presentan mayor variabilidad de precio; los **Departamentos** son más estables.  
- En zonas con gran densidad (CABA) el precio se explica más por **ubicación**, mientras que en GBA por **superficie**.  
- El modelo permite estimar precios razonables en contextos con alta heterogeneidad del mercado inmobiliario.

---

##  Fase 2 — Modelado y evaluación

Se probaron distintos modelos de regresión:

| Modelo                | R²   | RMSE (USD) | MAE (USD) | Observaciones |
|------------------------|------|-------------|------------|----------------|
| Linear Regression      | 0.65 | 94,000      | 54,000     | Subestima precios altos |
| Random Forest          | 0.81 | 76,700      | 42,400     | Buen equilibrio |
| Random Forest Tuned    | **0.83** | **73,500** | **40,500** | Mejor performance tras optimización |
| XGBoost                | 0.81 | 75,900      | 43,300     | Similar rendimiento, mayor costo de cómputo |

 **Modelo final elegido:** `RandomForestRegressor` afinado con `RandomizedSearchCV`  
por su mejor trade-off entre precisión, interpretabilidad y estabilidad.

---

## 📊 Métricas finales del modelo
- **R²:** 0.83  
- **RMSE:** 73,509 USD  
- **MAE:** 40,509 USD  

### Visualizaciones principales
📈 **Importancia de variables**
> `surface_covered` y `surface_total` son los mayores predictores del precio.

📉 **Precio real vs. predicho**
> El modelo sigue la diagonal correctamente, lo que indica buen ajuste en el rango medio.

📊 **Distribución de errores**
> La mayoría de las predicciones están concentradas alrededor de ±50.000 USD, lo que muestra estabilidad.

---

##  Fase 3 — Interfaz en Gradio

Se implementó una interfaz que permite ingresar los datos de una propiedad y obtener la predicción del precio estimado en USD.

🔗 **Demo online:**  
👉 [Properati Price Predictor — Hugging Face Space](https://huggingface.co/spaces/Smjanchez/properati-price-predictor)

📸 **Captura de la app:**  
![App Screenshot](app_screenshot.png)

---

##  Aprendizajes técnicos
- Creación de pipelines con *scikit-learn*  
- Afinación de hiperparámetros con *RandomizedSearchCV* y *GridSearchCV*  
- Implementación de interfaz web con *Gradio*  
- Deploy completo en *Hugging Face Spaces*  
- Evaluación de métricas y visualización de errores de predicción  

---

## 📑 Presentación del proyecto

La presentación con los principales resultados, visualizaciones e insights de negocio puede descargarse desde acá:
[Properati_Presentation.pptx](presentation/Properati_Presentation.pptx)

---

##  Estructura del proyecto

```text
properati-price-predictor/
├── data/
│   └── processed_properati.csv
├── notebooks/
│   ├── 01_EDA_properati.ipynb
│   └── 02_Modelado_properati.ipynb
├── src/
│   ├── train_properati_final.py
│   ├── train_XGB.py
│   ├── graficos3.py
│   └── app.py
├── models/
│   └── model_properati_final.pkl
├── presentation/
│   └── Properati_Presentation.pptx
├── requirements.txt
└── README.md
