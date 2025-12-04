def calcular_metricas_fairness(df_resultados, col_prediccion, col_grupo='genero'):
    metricas = {}
    tasas_seleccion = df_resultados.groupby(col_grupo)[col_prediccion].mean()
    pdt_gap = tasas_seleccion.max() - tasas_seleccion.min()

    metricas['PDT_gap'] = round(pdt_gap, 4)
    metricas['PDT_detalle'] = tasas_seleccion.to_dict()

    solo_evasores = df_resultados[df_resultados['es_evasor_real'] == 1]
    tasas_tpr = solo_evasores.groupby(col_grupo)[col_prediccion].mean()
    iof_gap = tasas_tpr.max() - tasas_tpr.min()

    metricas['IOF_gap'] = round(iof_gap, 4)
    metricas['IOF_detalle'] = tasas_tpr.to_dict()

    return metricas

