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

# 9) Visualización de resultados


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
