import shap
import numpy as np
import pandas as pd

def generar_explicacion_shap(modelo, datos_contribuyente, feature_names):
    """
    Genera una explicación local utilizando valores Shapley para un caso individual.
    Integra lógica de normalización para manejar salidas de clasificación binaria.
    
    Parámetros:
        modelo: Objeto entrenado (ej. RandomForest).
        datos_contribuyente: DataFrame/Array con los datos del caso a auditar.
        feature_names: Lista con los nombres de las variables.
    """
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(datos_contribuyente)
    
    vals = None

    # 1. Normalización de formato de salida (Lista vs Array)
    # Algunos modelos devuelven una lista [Clase0, Clase1]. Tomamos la Clase 1 (Riesgo).
    if isinstance(shap_values, list):
        vals = shap_values[1] 
    else:
        vals = shap_values

    # 2. Aplanado de dimensiones (Flatten)
    vals = np.array(vals).flatten()

    # 3. Validación de dimensiones
    # Si el array contiene valores para ambas clases (N*2), extraemos la segunda mitad.
    if len(vals) == len(feature_names) * 2:
        vals = vals[len(feature_names):] 
    
    # 4. Estructuración del reporte
    importancia = pd.DataFrame({
        'Variable': feature_names,
        'Impacto_SHAP': vals
    }).sort_values(by='Impacto_SHAP', key=abs, ascending=False)
    
    return importancia.head(3) # Retorna los 3 factores determinantes

print("Funciones del Marco MEAAT cargadas correctamente.")
