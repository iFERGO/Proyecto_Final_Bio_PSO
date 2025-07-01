import numpy as np
import random
from typing import List
from math import sqrt
import matplotlib.pyplot as plt
import os
import pandas as pd # Importar pandas para crear y mostrar la tabla

#Desarrollado por: Fernando M. González e Irene R. Reyes

# Calcula el costo de una solución (peso de una viga) más penalización por restricciones violadas
def costo(x: List[float]) -> float:
    p = 6000
    l_viga = 14
    e = 30e6
    g_modulo = 12e6
    t_max = 13600
    s_max = 30000
    d_max = 0.25

    y = 1.10471 * x[0]**2 * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])

    m = p * (l_viga + x[1] / 2)
    r = sqrt(0.25 * (x[1] ** 2 + (x[0] + x[2]) ** 2))
    j = 2 * (sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 12 + 0.25 * (x[0] + x[2]) ** 2))
    p_c = (4.013 * e / (6 * l_viga ** 2)) * x[2] * x[3] ** 3 * (1 - 0.25 * x[2] * sqrt(e / g_modulo) / l_viga)

    t1 = p / (sqrt(2) * x[0] * x[1])
    t2 = m * r / j
    t_esfuerzo = sqrt(t1**2 + t1 * t2 * x[1] / r + t2**2)

    s_esfuerzo = 6 * p * l_viga / (x[3] * x[2] ** 2)
    d_deflexion = 4 * p * l_viga**3 / (e * x[3] * x[2] ** 3)

    g1 = t_esfuerzo - t_max
    g2 = s_esfuerzo - s_max
    g3 = x[0] - x[2] 
    g4 = 0.10471 * x[0] ** 2 + 0.04811 * x[2] * x[3] * (14.0 + x[1]) - 5.0
    g5 = x[1] - x[3]
    g6 = d_deflexion - d_max
    g7 = p - p_c

    restricciones = [g1, g2, g3, g4, g5, g6, g7]

    w_magnitud = 100
    w_cantidad = 100

    g_redondeado = [round(val, 6) for val in restricciones]
    phi = sum(max(val, 0) for val in g_redondeado)
    num_violaciones = sum(val > 0 for val in g_redondeado)

    penalizacion = w_magnitud * phi + w_cantidad * num_violaciones

    return y + penalizacion

# Representa una partícula en PSO (posición, velocidad y su mejor valor personal)
class Particula:
    def __init__(self, posicion, velocidad):
        self.posicion = np.array(posicion)
        self.velocidad = np.array(velocidad)
        self.mejor_posicion_personal = self.posicion.copy()
        self.mejor_valor_personal = costo(self.posicion)

# Inicializa un conjunto de partículas con posiciones y velocidades aleatorias
def inicializar_particulas(num_particulas: int, num_dimensiones: int, limites_inferiores: List[float], limites_superiores: List[float]) -> List[Particula]:
    particulas = []
    for _ in range(num_particulas):
        posicion = [random.uniform(limites_inferiores[i], limites_superiores[i]) for i in range(num_dimensiones)]
        velocidad = [random.uniform(-1, 1) for _ in range(num_dimensiones)]
        particulas.append(Particula(posicion, velocidad))
    return particulas

# Actualiza la velocidad de una partícula considerando inercia, componente cognitivo y social
def actualizar_velocidad(particula: Particula, mejor_posicion_global: np.ndarray, inercia_w: float, coeficiente_cognitivo_c1: float, coeficiente_social_c2: float, num_dimensiones: int):
    r1 = np.random.rand(num_dimensiones)
    r2 = np.random.rand(num_dimensiones)
    inercia = inercia_w * particula.velocidad
    cognitivo = coeficiente_cognitivo_c1 * r1 * (particula.mejor_posicion_personal - particula.posicion)
    social = coeficiente_social_c2 * r2 * (mejor_posicion_global - particula.posicion)
    particula.velocidad = inercia + cognitivo + social

# Actualiza la posición de una partícula y aplica límites si excede el rango permitido
def actualizar_posicion(particula: Particula, limites_inferiores: List[float], limites_superiores: List[float], num_dimensiones: int):
    nueva_posicion = particula.posicion + particula.velocidad
    for i in range(num_dimensiones):
        if nueva_posicion[i] < limites_inferiores[i]:
            nueva_posicion[i] = limites_inferiores[i]
        elif nueva_posicion[i] > limites_superiores[i]:
            nueva_posicion[i] = limites_superiores[i]
    particula.posicion = nueva_posicion

# Ejecuta el algoritmo PSO para una corrida, devolviendo el mejor valor y su historial
def ejecutar_pso(
    num_particulas: int,
    max_iteraciones: int,
    inercia_w: float,
    coeficiente_cognitivo_c1: float,
    coeficiente_social_c2: float,
    semilla_aleatoria: int,
    limites_inferiores: List[float],
    limites_superiores: List[float],
    num_dimensiones: int
):
    random.seed(semilla_aleatoria)
    np.random.seed(semilla_aleatoria)

    particulas = inicializar_particulas(num_particulas, num_dimensiones, limites_inferiores, limites_superiores)

    mejor_global = {'posicion': None, 'valor': np.inf}
    historial_mejor_global = []

    for particula in particulas:
        if particula.mejor_valor_personal < mejor_global['valor']:
            mejor_global['posicion'] = particula.mejor_posicion_personal.copy()
            mejor_global['valor'] = particula.mejor_valor_personal

    for _ in range(max_iteraciones):
        for particula in particulas:
            actualizar_velocidad(particula, mejor_global['posicion'], inercia_w, coeficiente_cognitivo_c1, coeficiente_social_c2, num_dimensiones)
            actualizar_posicion(particula, limites_inferiores, limites_superiores, num_dimensiones)

            valor_actual = costo(particula.posicion)
            if valor_actual < particula.mejor_valor_personal:
                particula.mejor_posicion_personal = particula.posicion.copy()
                particula.mejor_valor_personal = valor_actual

        for particula in particulas:
            if particula.mejor_valor_personal < mejor_global['valor']:
                mejor_global['posicion'] = particula.mejor_posicion_personal.copy()
                mejor_global['valor'] = particula.mejor_valor_personal

        historial_mejor_global.append(mejor_global['valor'])

    return mejor_global, historial_mejor_global

# Genera una gráfica del historial del mejor costo global y la guarda como imagen
def guardar_grafica_historial_gbest(historial_gbest: List[float], filename: str, titulo: str = "Historial del Mejor Costo Global (gbest)"):
    plt.figure(figsize=(10, 6))
    plt.plot(historial_gbest, marker='o', linestyle='-', markersize=4)
    plt.title(titulo)
    plt.xlabel("Iteración")
    plt.ylabel("Mejor Costo Global (gbest)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Ejecuta múltiples corridas de PSO con diferentes parámetros y guarda los resultados
def ejecutar_multiples_corridas_pso(parametros_corridas: List[dict]):
    resultados_corridas = []
    num_dimensiones = 4
    limites_inferiores = [0.125, 0.1, 0.1, 0.125]
    limites_superiores = [10.0, 10.0, 10.0, 10.0]
    semillas = [42, 123, 789, 555, 321, 987, 654, 111, 212, 999]
    main_results_dir = "PSO_Resultados"
    os.makedirs(main_results_dir, exist_ok=True)

    tabla_datos = [] # Lista para almacenar los datos de la tabla

    for i, params in enumerate(parametros_corridas):
        print(f"\n--- Iniciando Corrida {i+1} ---")
        print(f"Parámetros: {params}")
        run_path = os.path.join(main_results_dir, f"Corrida_{i+1}")
        os.makedirs(run_path, exist_ok=True)
        resultados_semillas = []
        costos_por_corrida = [] # Para almacenar los costos de las 10 ejecuciones de esta corrida

        for semilla in semillas:
            print(f"  > Ejecutando con semilla: {semilla}")
            mejor_global, historial_mejor_global = ejecutar_pso(
                num_particulas=params['num_particulas'],
                max_iteraciones=params['max_iteraciones'],
                inercia_w=params['inercia_w'],
                coeficiente_cognitivo_c1=params['coeficiente_cognitivo_c1'],
                coeficiente_social_c2=params['coeficiente_social_c2'],
                semilla_aleatoria=semilla,
                limites_inferiores=limites_inferiores,
                limites_superiores=limites_superiores,
                num_dimensiones=num_dimensiones
            )

            resultados_semillas.append({
                'semilla': semilla,
                'mejor_global': mejor_global,
                'historial_mejor_global': historial_mejor_global
            })
            costos_por_corrida.append(mejor_global['valor']) # Agrega el mejor costo de esta semilla

            plot_title = (
                f"Historial gbest - Ejecución {i+1} (Semilla: {semilla})\n"
                f"Mejor Costo (gbest): {mejor_global['valor']:.4f}\n"
                f"Parámetros: np={params['num_particulas']}, iter={params['max_iteraciones']}, w={params['inercia_w']}, c1={params['coeficiente_cognitivo_c1']}, c2={params['coeficiente_social_c2']}"
            )
            image_filename = os.path.join(run_path, f"corrida_{i+1}_semilla_{semilla}.png")
            guardar_grafica_historial_gbest(historial_mejor_global, image_filename, plot_title)

        mejor_semilla = min(resultados_semillas, key=lambda x: x['mejor_global']['valor'])
        resultados_corridas.append({
            'parametros': params,
            'resultados_semillas': resultados_semillas,
            'mejor_resultado': mejor_semilla
        })
        
        # Calcula los valores para la tabla
        costo_promedio = np.mean(costos_por_corrida)
        costo_mejor = np.min(costos_por_corrida)
        costo_peor = np.max(costos_por_corrida)
        desviacion_estandar = np.std(costos_por_corrida)

        tabla_datos.append({
            'No de configuración': i + 1,
            'Costo (promedio)': costo_promedio,
            'Costo (mejor)': costo_mejor,
            'Costo (peor)': costo_peor,
            'Desviación estándar': desviacion_estandar
        })

    mejor_corrida = min(resultados_corridas, key=lambda x: x['mejor_resultado']['mejor_global']['valor'])
    print("\n--- Mejor Ejecución General ---")
    print(f"   Parámetros: {mejor_corrida['parametros']}")
    print(f"   Semilla: {mejor_corrida['mejor_resultado']['semilla']}")
    print(f"   Mejor Valor: {mejor_corrida['mejor_resultado']['mejor_global']['valor']:.6f}")
    print(f"   Posición: [{', '.join(f'{x:.4f}' for x in mejor_corrida['mejor_resultado']['mejor_global']['posicion'])}]")

    # Imprimir la tabla final
    print("\n" + "="*80)
    print("Resultados del Análisis de Configuraciones de PSO".center(80))
    print("="*80)
    df_resultados = pd.DataFrame(tabla_datos)
    # Formatear las columnas numéricas para una mejor lectura
    df_resultados['Costo (promedio)'] = df_resultados['Costo (promedio)'].map('{:.6f}'.format)
    df_resultados['Costo (mejor)'] = df_resultados['Costo (mejor)'].map('{:.6f}'.format)
    df_resultados['Costo (peor)'] = df_resultados['Costo (peor)'].map('{:.6f}'.format)
    df_resultados['Desviación estándar'] = df_resultados['Desviación estándar'].map('{:.6f}'.format)
    print(df_resultados.to_string(index=False))
    print("="*80)


# Función principal para definir parámetros y lanzar las corridas de PSO
def main():
    parametros_para_analisis = [
        # Parámetros para las corridas de PSO Metodo Anova
        # num_particulas=[30, 50, 100]
        # inercia = [0.7, 0.8, 0.9]
        # coeficiente_cognitivo = [1.0, 1.5, 2.0]
        # coeficiente_social = [1.0, 1.5, 2.0]

        {
            #Inercia alta
            'num_particulas': 50,
            'max_iteraciones': 70,
            'inercia_w': 0.9,
            'coeficiente_cognitivo_c1': 1.5,
            'coeficiente_social_c2': 1.5
        },
        {
            #inercia baja
            'num_particulas': 50,
            'max_iteraciones': 70,
            'inercia_w': 0.7,
            'coeficiente_cognitivo_c1': 1.5,
            'coeficiente_social_c2': 1.5
        },
        {
            #Numero de particulas alto
            'num_particulas': 100,
            'max_iteraciones': 70,
            'inercia_w': 0.8,
            'coeficiente_cognitivo_c1': 1.5,
            'coeficiente_social_c2': 1.5
        },
        {
            #Coeficiente social alto
            'num_particulas': 50,
            'max_iteraciones': 70,
            'inercia_w': 0.8,
            'coeficiente_cognitivo_c1': 1.0,
            'coeficiente_social_c2': 2.0
        },
        {
            #Coeficiente cognitivo alto
            'num_particulas': 50,
            'max_iteraciones': 70,
            'inercia_w': 0.8,
            'coeficiente_cognitivo_c1': 2.0,
            'coeficiente_social_c2': 1.0
        }
    ]
    ejecutar_multiples_corridas_pso(parametros_para_analisis)

if __name__ == "__main__":
    main()