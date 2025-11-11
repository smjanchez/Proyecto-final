import pickle
import numpy as np
import pandas as pd
import gradio as gr

# ================================
# 1) Cargar modelo y datos base
# ================================
MODEL_PATH = "model_properati_final.pkl"
DATA_PATH = "processed_properati.csv"

# Cargamos el modelo (pipeline completo: preprocesador + RandomForest)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Cargamos dataset procesado para armar los dropdowns
df = pd.read_csv(DATA_PATH)

features = [
    "rooms",
    "bedrooms",
    "bathrooms",
    "surface_total",
    "surface_covered",
    "property_type",
    "state_name",
    "place_name",
]

# Listas de opciones para los campos categóricos
property_types = sorted(df["property_type"].dropna().unique().tolist())
state_names = sorted(df["state_name"].dropna().unique().tolist())
place_names = sorted(df["place_name"].dropna().unique().tolist())

# Valores por defecto "razonables" si existen
default_property = "Departamento" if "Departamento" in property_types else property_types[0]
default_state = "Capital Federal" if "Capital Federal" in state_names else state_names[0]
default_place = "Palermo" if "Palermo" in place_names else place_names[0]


# ================================
# 2) Función de predicción
# ================================
def predecir_precio(
    rooms,
    bedrooms,
    bathrooms,
    surface_total,
    surface_covered,
    property_type,
    state_name,
    place_name,
):
    """
    Recibe los inputs del usuario, arma un DataFrame con las mismas columnas
    que se usaron para entrenar el modelo y devuelve el precio estimado.
    """
    data = pd.DataFrame(
        [
            {
                "rooms": rooms,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "surface_total": surface_total,
                "surface_covered": surface_covered,
                "property_type": property_type,
                "state_name": state_name,
                "place_name": place_name,
            }
        ]
    )

    pred = model.predict(data)[0]

    # Devolvemos texto formateado
    return f"USD {pred:,.0f}"


# ================================
# 3) Definición de la interfaz Gradio
# ================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏙️ Predictor de precios de propiedades (Properati Argentina)

        Proyecto final — Escuela de Datos Vivos  
        Modelo: **RandomForestRegressor** entrenado sobre datos históricos de Properati (CABA + GBA, ventas en USD).
        """
    )

    with gr.Row():
        with gr.Column():
            rooms_in = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=3,
                label="Ambientes (rooms)",
            )
            bedrooms_in = gr.Slider(
                minimum=0,
                maximum=8,
                step=1,
                value=2,
                label="Dormitorios",
            )
            bathrooms_in = gr.Slider(
                minimum=1,
                maximum=6,
                step=1,
                value=1,
                label="Baños",
            )
            surface_total_in = gr.Slider(
                minimum=10,
                maximum=500,
                step=1,
                value=80,
                label="Superficie total (m²)",
            )
            surface_covered_in = gr.Slider(
                minimum=10,
                maximum=500,
                step=1,
                value=70,
                label="Superficie cubierta (m²)",
            )
            property_type_in = gr.Dropdown(
                choices=property_types,
                value=default_property,
                label="Tipo de propiedad",
            )
            state_name_in = gr.Dropdown(
                choices=state_names,
                value=default_state,
                label="Provincia / zona",
            )
            place_name_in = gr.Dropdown(
                choices=place_names,
                value=default_place,
                label="Barrio / localidad",
            )

            btn = gr.Button(" Estimar precio")

        with gr.Column():
            output_price = gr.Textbox(
                label="Precio estimado (USD)",
                interactive=False,
                placeholder="El resultado aparecerá aquí...",
            )
            gr.Markdown(
                """
                ###  Notas
                * La estimación se basa en un modelo de **Machine Learning** entrenado con datos históricos de Properati.
                * Los valores son **referenciales** y no reemplazan una tasación profesional.
                * Mayor superficie y más baños/dormitorios aumentan fuertemente el precio, especialmente en zonas premium.
                """
            )

    btn.click(
        fn=predecir_precio,
        inputs=[
            rooms_in,
            bedrooms_in,
            bathrooms_in,
            surface_total_in,
            surface_covered_in,
            property_type_in,
            state_name_in,
            place_name_in,
        ],
        outputs=output_price,
    )

# Para correr localmente: python app.py
if __name__ == "__main__":
    demo.launch()
