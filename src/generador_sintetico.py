import numpy as np
import pandas as pd
from faker import Faker

def generar_poblacion_sintetica(n=10000, semilla=42):
    """
    Genera un dataset de contribuyentes sintéticos preservando la estructura
    estadística real (heterocedasticidad, correlaciones) e inyectando sesgos
    históricos controlados para pruebas de estrés.
    """
    np.random.seed(semilla)
    fake = Faker('es_ES')
    data = []

    for _ in range(n):
        # --- Variables Demográficas ---
        genero = np.random.choice(['M', 'F'], p=[0.5, 0.5])
        edad = int(np.random.normal(45, 12))
        edad = max(18, min(90, edad)) # Acotado al rango laboral
        region = np.random.choice(['Capital', 'Norte', 'Sur', 'Centro'], 
                                  p=[0.4, 0.2, 0.2, 0.2])

        # --- Variables Económicas ---
        # Distribución Log-Normal para simular riqueza realista
        ingresos = np.random.lognormal(mean=10.5, sigma=0.8)
        
        # Variabilidad: A mayor ingreso, mayor dispersión fiscal
        ingresos_var_z = np.random.normal(0, 1) + (ingresos / 100000) * 0.1
        
        # Deducciones condicionadas por edad
        base_deduccion = 0.2
        if edad > 60: base_deduccion += 0.1
        ratio_deducciones = np.random.beta(2, 5) + base_deduccion

        # --- Variable Latente (Ground Truth) ---
        # Probabilidad real de evasión (sin sesgo de género)
        prob_evasion_real = 0.05 + (0.1 if ratio_deducciones > 0.4 else 0)
        es_evasor = 1 if np.random.random() < prob_evasion_real else 0

        # --- Inyección de Sesgo Histórico ---
        # Simulación: Las mujeres fueron históricamente menos auditadas (Sesgo)
        prob_seleccion_historica = prob_evasion_real
        if genero == 'F':
            prob_seleccion_historica *= 0.6 

        fue_auditado_historico = 1 if np.random.random() < prob_seleccion_historica else 0

        data.append([genero, edad, region, ingresos, ratio_deducciones,
                     es_evasor, fue_auditado_historico])

    cols = ['genero', 'edad', 'region', 'ingresos', 'ratio_deducciones',
            'es_evasor_real', 'fue_auditado_historico']
    return pd.DataFrame(data, columns=cols)

