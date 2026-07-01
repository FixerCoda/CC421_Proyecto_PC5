# Guion de la Presentación: Optimización de Políticas en Tetris

**Duración estimada:** ~15 minutos.  
**Dinámica:** Tono profesional, explicativo, matemático pero fluido. No se debe leer el texto tal cual, sino usarlo como base para la narración.

---

## Diapositiva 1: El Desafío Combinatorio
**Orador:** Aldo  
**Duración aproximada:** 2.5 minutos

"Buenos días al jurado y a todos los presentes. Hoy les presentamos nuestro proyecto sobre la 'Optimización de Políticas en Tetris mediante Aprendizaje por Refuerzo y el Método de Entropía Cruzada (CEM)'. 

Antes de entrar en los algoritmos, es crucial mencionar un esfuerzo fundamental de este proyecto: no usamos un entorno preexistente. Construimos nuestro propio motor de Tetris desde cero en Python. Diseñamos este motor no para que lo juegue un humano, sino para que un agente de IA pueda interactuar con él millones de veces por segundo. Prescindimos de interfaces gráficas pesadas durante el entrenamiento para enfocarnos en la representación pura de los estados matemáticos.

¿Por qué Tetris? Tetris no es solo un juego; desde la perspectiva de la IA, es un problema de decisión secuencial con una explosión combinatoria masiva. El número de estados posibles del tablero es astronómico, lo que hace que cualquier método de búsqueda exhaustiva sea computacionalmente inviable. 

Ante este desafío, la tendencia actual en la industria es lanzar fuerza bruta: Deep Reinforcement Learning, redes neuronales masivas y días de entrenamiento en GPUs. Nosotros tomamos un camino diferente. Proponemos una solución elegante, basada en abstracciones matemáticas sólidas, extracción de características geométricas y optimización directa, demostrando que con el modelado correcto, la fuerza bruta no siempre es necesaria."

---

## Diapositiva 2: Formulación MDP y Abstracción Temporal
**Orador:** Pedro  
**Duración aproximada:** 1.5 minutos

"Para resolver el problema computacional, el primer paso fue modelar el entorno formalmente como un Proceso de Decisión de Markov (MDP). Sin embargo, tomamos una decisión de diseño crítica: la abstracción temporal.

Si controlamos el juego 'frame a frame', decidiendo si mover la pieza a la izquierda, derecha o rotarla en cada instante, el agente se pierde en micro-decisiones. En lugar de eso, abstrajimos el control a **colocaciones**. Una acción para nuestro agente consiste simplemente en elegir el destino final de la pieza: su rotación y la columna de caída. La pieza cae instantáneamente por *hard-drop*.

Esto reduce drásticamente el factor de ramificación: pasamos de infinitas trayectorias posibles a solo unas 40 acciones válidas por turno. Además, esto nos permite usar el concepto de **Afterstate**. La transición es determinista: dado un tablero y una colocación, obtenemos un único tablero resultante tras fijar la pieza y limpiar las líneas. El agente solo necesita evaluar la calidad de este *afterstate*."

---

## Diapositiva 3: Extracción de Características y Aceleración JIT
**Orador:** Pedro  
**Duración aproximada:** 2 minutos

"Como el espacio de *afterstates* sigue siendo gigante, no podemos usar una tabla para almacenar el valor de cada tablero. Por ello, mapeamos cada *afterstate* a un vector compacto de 6 características geométricas clave: 
1. La altura agregada de todas las columnas.
2. Los huecos (espacios vacíos atrapados bajo bloques).
3. La rugosidad (la diferencia de alturas entre columnas adyacentes).
4. Las líneas eliminadas en ese turno.
5. La altura máxima de la pila.
6. Los pozos profundos.

Con esto, usamos una aproximación lineal clásica: el valor del estado es el producto punto entre estas características y un vector de pesos que el agente debe aprender.

Pero aquí nos encontramos con un cuello de botella: evaluar estas 6 características para 40 acciones posibles, millones de veces durante el entrenamiento, es muy costoso en Python puro. La solución de ingeniería fue compilar toda la lógica de extracción de características *Just-In-Time* (JIT) a nivel de C utilizando el decorador `@njit` de la librería Numba. Esta inyección de velocidad fue lo que hizo viable entrenar a nuestros agentes en minutos en lugar de horas."

---

## Diapositiva 4: Diferencia Temporal y Estabilidad Numérica
**Orador:** Aldo  
**Duración aproximada:** 2 minutos

"Con el entorno y las características listas, el primer algoritmo que usamos para encontrar los pesos óptimos fue el Aprendizaje por Diferencia Temporal, específicamente TD(0) con actualizaciones semi-gradiente.

Hicimos un barrido de hiperparámetros y encontramos una política que limpiaba en promedio 409 líneas. Aquí descubrimos algo vital: un factor de descuento $\gamma$ de 0.999 fue esencial. Un agente miope muere rápido en Tetris; se necesita planificar a largo plazo.

Sin embargo, observamos un problema teórico clásico en la práctica: la inestabilidad numérica. Al combinar *bootstrapping* de TD con una función de aproximación, la magnitud de los pesos divergía hacia el infinito, llegando al orden de $10^5$. 
¿Por qué no colapsó la política de inmediato? Porque la elección de la acción se basa en un $\arg\max$, el cual es invariante a la escala. A la política solo le importan los signos y proporciones relativas de los pesos, no su tamaño absoluto. Aún así, esta divergencia es una bomba de tiempo algorítmica."

---

## Diapositiva 5: Optimización mediante CEM
**Orador:** Diego  
**Duración aproximada:** 2 minutos

"Para resolver la inestabilidad del TD, cambiamos de paradigma. En lugar de usar *bootstrapping* y gradientes, implementamos el Método de Entropía Cruzada (o CEM). CEM es un método de optimización directa de la política basado en poblaciones.

El mecanismo es elegante: mantenemos una distribución Gaussiana sobre el espacio de los pesos. En cada generación, muestreamos decenas de agentes candidatos y los ponemos a jugar. Evaluamos su rendimiento, seleccionamos al 20% mejor (la élite), y ajustamos nuestra campana de Gauss para que se centre en ellos.

Para garantizar robustez y comparaciones justas, usamos *Common Random Numbers*, es decir, todos los candidatos de una generación enfrentan exactamente la misma secuencia de piezas. Y lo más importante: CEM nos permite aplicar una restricción explícita. Forzamos a que la norma del vector de pesos sea exactamente 1 ($||w|| = 1$), eliminando por completo la inestabilidad que sufríamos en TD."

---

## Diapositiva 6: La Ilusión del Corto Plazo (Evaluación a 1000 pasos)
**Orador:** Diego  
**Duración aproximada:** 2 minutos

"Llegado el momento de comparar ambos algoritmos frente a baselines, diseñamos un protocolo estricto: 20 partidas con las mismas semillas aleatorias. 

Inicialmente, pusimos un horizonte máximo de 1000 colocaciones para que el experimento no durara eternamente. Los resultados fueron los siguientes: el agente aleatorio hizo 0 líneas. El heurístico 184. TD alcanzó 338 líneas y CEM 371.

A simple vista, CEM parece solo un poco mejor. Pero nos dimos cuenta de que estábamos ante un 'efecto techo'. Ambos agentes son lo suficientemente buenos para sobrevivir 1000 piezas sin perder. Así que este límite artificial solo estaba midiendo qué tan eficientemente empaquetaban las líneas en esos primeros 1000 pasos, pero ocultaba su verdadera capacidad de supervivencia."

---

## Diapositiva 7: Desenmascarando la Inestabilidad (Evaluación a 5000 pasos)
**Orador:** Aldo  
**Duración aproximada:** 1.5 minutos

"Decidimos levantar ese techo y extender la evaluación a 5000 colocaciones. Aquí es donde se revela la verdadera naturaleza de ambos algoritmos.

Bajo estrés prolongado, la divergencia del TD finalmente pasa factura. Los pesos desproporcionados de TD (cerca de $152,000$) empiezan a amplificar pequeños errores numéricos, forzando al agente a tomar malas decisiones que resultan en un *Game Over* prematuro, promediando 563 líneas.

En contraste, la estabilidad del CEM brilla. Gracias a sus pesos acotados y su aprendizaje sin gradientes ($||w|| = 1.0$), el agente CEM casi duplica ese rendimiento, alcanzando 1048 líneas en promedio. CEM no solo empaca líneas eficientemente, sino que toma decisiones consistentes que garantizan la supervivencia a largo plazo."

---

## Diapositiva 8: Conclusiones
**Orador:** Diego  
**Duración aproximada:** 1.5 minutos

"Para concluir, este proyecto nos deja lecciones muy claras sobre el diseño de sistemas de Inteligencia Artificial.

Primero, demostramos que una representación matemática rigurosa, la abstracción del tiempo mediante un MDP por colocaciones y la aceleración JIT a bajo nivel, nos permitieron entrenar agentes altamente competentes en un hardware estándar en minutos.

Segundo, evidenciamos experimentalmente los límites teóricos del aprendizaje por Diferencia Temporal en este contexto, y cómo métodos de optimización estocástica como CEM ofrecen garantías de estabilidad muy superiores.

Al final, nuestro mensaje es que una abstracción inteligente combinada con el algoritmo de optimización adecuado suele ser mucho más poderosa y eficiente que lanzar simplemente poder de cómputo bruto contra un problema. 

Muchas gracias por su atención. Estamos abiertos a cualquier pregunta."