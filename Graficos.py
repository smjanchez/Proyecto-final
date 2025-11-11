# prueba.py
# Exploración de correlaciones numéricas del dataset procesado

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Cargar el dataset procesado
df = pd.read_csv("processed_properati.csv")
print("Dataset cargado. Shape:", df.shape)
print("Dataset cargado. Shape:", df.shape)

# 2) Seleccionar columnas numéricas relevantes
num_cols = [
    "price_usd",
    "rooms",
    "bedrooms",
    "bathrooms",
    "surface_total",
    "surface_covered"
]

print("Columnas numéricas usadas:", num_cols)

# 3) Matriz de correlación
corr = df[num_cols].corr()
print("\nMatriz de correlación:")
print(corr)

# 4) Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlaciones numéricas")
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("processed_properati.csv")

# 5) Gráfico distribución de precios por tipo de propiedad
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="property_type", y="price_usd", scale="width", inner="quart")
plt.yscale("log")  # escala logarítmica para ver mejor los rangos
plt.title("Distribución de precios según tipo de propiedad")
plt.tight_layout()
plt.show()

# 6) Gráfico  precios por cantidad de habitaciones
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="rooms", y="price_usd")
plt.yscale("log")
plt.title("Precio vs cantidad de ambientes")
plt.tight_layout()
plt.show()

# 7) Gráfico de caja
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x="property_type", y="price_usd", showfliers=False)
plt.yscale("log")
plt.title("Distribución de precios (Boxplot)")

plt.tight_layout()
plt.show()

