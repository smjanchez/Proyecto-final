# Predicprecios.py
# EDA 
# train_properati_final.py
# Entrenamiento del modelo RandomForestRegressor con hiperparámetros afinados
# + análisis visual de resultados

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# 1) Cargar dataset completo procesado
data = pd.read_csv("processed_properati.csv")
print("Dataset completo utilizado. Shape:", data.shape)

# 2) Definir features y target
features = [
    "rooms",
    "bedrooms",
    "bathrooms",
    "surface_total",
    "surface_covered",
    "property_type",
    "state_name",
    "place_name"
]

X = data[features].copy()
y = data["price_usd"].copy()

numeric_features = ["rooms", "bedrooms", "bathrooms", "surface_total", "surface_covered"]
categorical_features = ["property_type", "state_name", "place_name"]

# 3) Train / Test split (para poder reportar métricas también en este modelo final)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

print("\nTamaño de train:", X_train.shape)
print("Tamaño de test:", X_test.shape)

# 4) Preprocesador
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 5) Modelo con hiperparámetros afinados
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=35,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=99,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf)
])

# 6) Entrenamiento
print("\nEntrenando modelo final RandomForestRegressor...")
model.fit(X_train, y_train)

# 7) Evaluación
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMétricas del modelo final en test:")
print(f"RMSE: {rmse:,.0f} USD")
print(f"MAE : {mae:,.0f} USD")
print(f"R²  : {r2:.3f}")

# 8) Guardar modelo final
filename = "model_properati_final.pkl"
with open(filename, "wb") as f:
    pickle.dump(model, f)

print(f"\nModelo final guardado como: {filename}")

# ==========================================================
# 9) Visualización de resultados
# ==========================================================

print("\nGenerando gráficos...")

# --- Importancia de variables ---
preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(
    x=importances[indices][:15],
    y=np.array(feature_names)[indices][:15],
    orient="h"
)
plt.title("Importancia de variables (Random Forest)")
plt.xlabel("Importancia relativa")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# --- Precio real vs predicho ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
plt.xlabel("Precio real (USD)")
plt.ylabel("Precio predicho (USD)")
plt.title("Precio real vs. precio predicho (Random Forest)")
plt.tight_layout()
plt.show()

# --- Histograma de errores ---
residuals = y_test - y_pred
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Error (real - predicho) [USD]")
plt.title("Distribución de errores del modelo Random Forest")
plt.tight_layout()
plt.show()

# --- Residuales vs predicción ---
plt.figure(figsize=(8, 4))
plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Precio predicho (USD)")
plt.ylabel("Error (real - predicho)")
plt.title("Errores vs precio predicho")
plt.tight_layout()
plt.show()

print("\nGráficos generados correctamente.")
import pandas as pd
import numpy as np

# 1) Cargar dataset
df = pd.read_csv("properati.csv")
print("Dataset cargado. Filas:", len(df))

# 2) Filtrar Argentina / Venta / USD
df = df.query("l1 == 'Argentina' and operation_type == 'Venta' and currency == 'USD'")
print("Filtro aplicado. Filas:", len(df))

# 3) Renombrar columnas relevantes
df = df.rename(columns={
    "price": "price_usd",
    "l2": "state_name",
    "l3": "place_name"
})

# 4) Seleccionar columnas de interés
cols = [
    "price_usd",
    "rooms",
    "bedrooms",
    "bathrooms",
    "surface_total",
    "surface_covered",
    "property_type",
    "state_name",
    "place_name"
]
df = df[cols].copy()

# 5) Eliminar registros con precio nulo o <= 0
df = df[df["price_usd"] > 0]

# 6) Quitar casos sin superficie total ni cubierta
df = df[~(df["surface_total"].isna() & df["surface_covered"].isna())]

# 7) Corregir relaciones ilógicas de superficie
df = df[
    (df["surface_covered"].isna())
    | (df["surface_total"].isna())
    | (df["surface_total"] >= df["surface_covered"])
]

# 8) Crear variable de apoyo: precio por m²
df["price_per_m2"] = df["price_usd"] / df["surface_covered"]

# 9) Filtrar precios por m² razonables (IQR)
q1 = df["price_per_m2"].quantile(0.25)
q3 = df["price_per_m2"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df[df["price_per_m2"].between(lower, upper)]
print("Eliminados outliers de precio/m². Filas restantes:", len(df))

# 10) Superficies razonables (10 m² a 500 m²)
df = df[df["surface_total"].between(10, 500) | df["surface_total"].isna()]
df = df[df["surface_covered"].between(10, 500) | df["surface_covered"].isna()]

# 10 bis) Ajuste adicional de valores extremos o ilógicos en rooms/bedrooms/bathrooms

# Rooms: valores razonables entre 1 y 10
df.loc[(df["rooms"] < 1) | (df["rooms"] > 10), "rooms"] = np.nan

# Bedrooms: entre 0 y 8
df.loc[(df["bedrooms"] < 0) | (df["bedrooms"] > 8), "bedrooms"] = np.nan

# Bathrooms: entre 1 y 6
df.loc[(df["bathrooms"] < 1) | (df["bathrooms"] > 6), "bathrooms"] = np.nan

# 11) Imputar valores faltantes numéricos con lógica por tipo de propiedad
for col in ["rooms", "bedrooms", "bathrooms", "surface_total", "surface_covered"]:
    df[col] = df.groupby("property_type")[col].transform(lambda x: x.fillna(x.median()))

# 12) Imputar categóricas
for col in ["property_type", "state_name", "place_name"]:
    df[col] = df[col].fillna("Desconocido")

# 13) Recalcular price_per_m2 (por si se modificó superficie)
df["price_per_m2"] = df["price_usd"] / df["surface_covered"]

# 14) Ver resumen final
print("\nDataset final limpio:")
print(df.describe(include="all").T)

print("\nPorcentaje de nulos:\n", df.isna().mean().round(3))

# 15) Guardar dataset limpio
df.to_csv("processed_properati.csv", index=False)
print("\nDataset limpio guardado como processed_properati.csv")
