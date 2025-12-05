from src.generador_sintetico import generar_poblacion_sintetica
from src.metricas_fairness import calcular_metricas_fairness

from sklearn.ensemble import RandomForestClassifier

# 1. Generar población sintética
df = generar_poblacion_sintetica(n=5000, semilla=42)

# 2. Preparar datos para entrenar un modelo simple
X = df[['edad', 'ingresos', 'ratio_deducciones']]
y = df['es_evasor_real']

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# 3. Simular una “predicción de auditoría”
df['prediccion_auditoria'] = modelo.predict(X)

# 4. Calcular métricas de fairness por género
metricas = calcular_metricas_fairness(
    df_resultados=df,
    col_prediccion='prediccion_auditoria',
    col_grupo='genero'
)

print("Métricas de fairness calculadas:")
print(metricas)
