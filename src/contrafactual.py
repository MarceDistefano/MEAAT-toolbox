def generar_contrafactual(modelo, datos_contribuyente, variable_objetivo, paso=0.05):
    """
    Calcula el cambio mínimo necesario en una variable para revertir
    la decisión de auditoría (Frontera de decisión).
    """
    caso_simulado = datos_contribuyente.copy()
    
    # Validación del estado inicial
    prediccion_actual = modelo.predict(caso_simulado)[0]
    if prediccion_actual == 0:
        return "El caso ya está clasificado como de Bajo Riesgo."
    
    valor_original = caso_simulado[variable_objetivo].values[0]
    iteracion = 0
    max_iter = 50 

    # Bucle de simulación iterativa
    while iteracion < max_iter:
        # Reducción progresiva del valor de la variable
        nuevo_valor = caso_simulado[variable_objetivo].values[0] * (1 - paso)
        caso_simulado[variable_objetivo] = nuevo_valor
        
        # Re-evaluación del modelo
        nueva_prediccion = modelo.predict(caso_simulado)[0]
        
        if nueva_prediccion == 0:
            # Solución encontrada
            reduccion_pct = ((valor_original - nuevo_valor) / valor_original) * 100
            return (
                f"Solución Contrafactual: Para evitar la selección, la variable "
                f"'{variable_objetivo}' debe reducirse de {valor_original:.2f} "
                f"a {nuevo_valor:.2f} (Reducción del {reduccion_pct:.1f}%)."
            )
        
        iteracion += 1
        
    return "No se encontró una solución contrafactual simple en el rango evaluado."
