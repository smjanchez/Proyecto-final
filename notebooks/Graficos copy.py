import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Cargar modelo final (pipeline)
with open("model_properati_final.pkl", "rb") as f:
    model = pickle.load(f)

# 2) Extraer preprocesador y modelo
preprocessor = model.named_steps["preprocessor"]
rf = model.named_steps["model"]

# 3) Nombres de columnas transformadas
numeric_features = preprocessor.transformers_[0][2]
cat_encoder = preprocessor.transformers_[1][1].named_steps["encoder"]
categorical_features = preprocessor.transformers_[1][2]
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)

feature_names = np.concatenate([numeric_features, cat_feature_names])
importances = rf.feature_importances_

feat_imp = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# 4) Agrupar por variable original:
#    - numéricas quedan igual
#    - categóricas se agrupan por prefijo (property_type, state_name, place_name)
def agrupar_nombre(col):
    # Si es una de las originales numéricas, la dejamos igual
    if col in numeric_features:
        return col
    # Si es categórica, viene como "property_type_Departamento", "state_name_Capital Federal", etc.
    # Nos quedamos con la parte antes del primer "_"
    return col.split("_")[0]

feat_imp["feature_group"] = feat_imp["feature"].apply(agrupar_nombre)

agg_imp = (
    feat_imp
    .groupby("feature_group", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)

print("\nImportancia agregada por variable original:")
print(agg_imp)

# 5) Gráfico de barras (todas las features agregadas)
plt.figure(figsize=(8, 5))
plt.barh(agg_imp["feature_group"][::-1], agg_imp["importance"][::-1], color="steelblue")
plt.title("Importancia de variables (agrupadas por feature original)", fontsize=12)
plt.xlabel("Importancia (suma de importancias de dummies)")
plt.ylabel("Variable original")
plt.tight_layout()
plt.savefig("feature_importance_agrupada.png", dpi=300)
plt.show()

# 6) Si querés solo el top 6 para el PPT:
top_n = 6
top6 = agg_imp.head(top_n)

plt.figure(figsize=(6, 4))
plt.barh(top6["feature_group"][::-1], top6["importance"][::-1], color="steelblue")
plt.title(f"Top {top_n} variables más importantes", fontsize=12)
plt.xlabel("Importancia")
plt.ylabel("Variable original")
plt.tight_layout()
plt.savefig("feature_importance_top6.png", dpi=300)
plt.show()

print("\nTop 6 variables más importantes:")
print(top6)
