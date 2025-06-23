## Discretización
El env.gameState() devuelve un estado continuo. Se realizó una discretización de características sobre dicho estado de la siguiente forma:

| Variable                      | Descripción                                         | Rango aproximado | Bins usados |
|------------------------------|-----------------------------------------------------|------------------|-------------|
| `player_y`                   | Posición vertical del pájaro                        | 0 a 512 px       | 20          |
| `player_vel`                 | Velocidad vertical del pájaro                       | -20 a 20         | 10          |
| `next_pipe_dist_to_player`   | Distancia horizontal al próximo tubo                | 0 a 288 px       | 15          |
| `next_pipe_top_y`            | Altura del borde superior del próximo tubo          | 0 a 512 px       | 20          |
| `next_pipe_bottom_y`         | Altura del borde inferior del próximo tubo          | 0 a 512 px       | 20          |

Cada una de estas variables fue discretizada dividiendo su rango en un número fijo de *bins* (segmentos) de igual tamaño.

Se decidió no incluir las variables referidas a next_next_pipe ya que se evidenció que el agente funcionaba bien sólamente mirando a la siguiente tubería. Incluso, el incluirlas hacía el entrenamiento más lento y a veces provocaba confusión en el agente (tomaba decisiones basándose en la "siguiente-siguiente" tubería antes de pasar la "siguiente" tubería)

# Análisis y comparación de los resultados obtenidos para los diferentes agentes.

## El QAgent
Este agente primero hace su etapa exploratoria, con 20.000 iteraciones con la cual conforma la QTable. Luego, el agente toma las decisiones en base a esa QTable en el *test_agent*.
En el entrenamiento, además de la recompensa proveída por la cátedra, se agrega una penalización igual a:
```python
gap_center = (pipe_top + pipe_bottom) / 2
dist_to_center = abs(gap_center - player_y)
norm_distance = dist_to_center / 512
reward -= norm_distance * 0.2
```
Es decir, el 20% de la distancia del agente al centro de la siguiente tubería. Esto se hace para que el mismo busque pasar por el centro de la tubería, minimizando pérdidas de partida por pegar en los bordes de las mismas. Teniendo así una performance mas "limpia" y fluída.

Este QAgent, en el test ha logrado pasar un máximo de 100 tuberías.

## El NN_Agent
Este agente toma sus decisiones en base a un modelo entrenado. Dicho modelo, es una red neuronal que toma un estado y predice que tan bueno es saltar o no saltar para el agente, finalmente el agente toma la mejor decisión para cada estado en base a la predicción del modelo.
El modelo fue entrenado con la QTable armada en el entrenamiento del QAgent, obteniendo un loss en set de train de **0.1820** y en set de val **0.3280**.

Si bien parece estar algo sobreajustado, la perfomance del modelo en el *test_agent* da resultados óptimos, pasando un valor máximo de 80 tuberías.

## Comparaciones.

| Agente  | Técnica                         | Entrenamiento                                                          | Máximo de tubos superados |
| ------- | ------------------------------- | ---------------------------------------------------------------------- | ------------------------- |
| QAgent  | Q-learning tabular              | 20.000 episodios con exploración + penalización por distancia al hueco | **100**                   |
| NNAgent | Aproximación Q con red neuronal | Supervisado desde Q-table                                              | **80**                    |

Ambos agentes muestran un comportamiento eficaz en el entorno, aunque con diferencias en su enfoque. El QAgent, al utilizar la tabla completa y la exploración, logra un rendimiento levemente superior.

El NNAgent, por su parte, permite generalizar el conocimiento aprendido y ofrece una inferencia más eficiente en tiempo real, a costa de una ligera pérdida de precisión.


