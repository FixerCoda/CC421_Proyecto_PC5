# Prompt para Generación de Presentación: Optimización de Políticas en Tetris

Actúas como un experto diseñador de presentaciones técnicas y científicas de alto impacto (estilo TED/Keynote). Tu objetivo es crear el contenido para una presentación dirigida a un jurado experto en Inteligencia Artificial.

El proyecto se titula: **"Optimización de Políticas en Tetris mediante Aprendizaje por Refuerzo y CEM"**.

## Instrucciones Críticas de Estilo
*   **Minimalismo Extremo:** Las diapositivas deben tener el mínimo texto posible. Ni bloques de texto ni largos párrafos. Usa palabras clave, cifras grandes y conceptos de alto nivel.
*   **Cero Notas del Orador:** No redactes un guion ni notas del orador. Enfócate exclusivamente en lo que se proyecta visualmente.
*   **Impacto Visual:** Para cada diapositiva, sugiere *detalladamente* un recurso visual (un gráfico específico, un diagrama esquemático, un fragmento de código muy breve, o la disposición de los datos) que complemente el minimalismo textual. Debes asegurarte de que los detalles visuales propuestos usen la simbología matemática y de código estricta del proyecto.
*   **Flujo y Memorabilidad:** Que ninguna diapositiva se sienta igual a la anterior. Cada una debe transmitir un concepto clave utilizando los datos reales del proyecto.

---

## Estructura y Contenido Requerido por Diapositiva

Genera el contenido siguiendo exactamente esta estructura. Para cada diapositiva entrega:
1. **Título (Corto y directo)**
2. **Texto en Pantalla (Mínimo, viñetas cortas o cifras)**
3. **Elemento Visual Propuesto (Descripción detallada de la imagen/gráfico a incluir. DEBE incluir las fórmulas, nombres de variables o conceptos específicos listados en cada diapositiva).**

### Diapositiva 1: El Desafío Combinatorio
*   **El concepto:** El inmenso espacio de estados de Tetris imposibilita la búsqueda exhaustiva.
*   **Diferenciador:** Frente a enfoques de Deep RL de alto costo computacional, empleamos una formulación basada en extracción de características y optimización directa.
*   **Detalle Visual Exigido:** Esquema comparativo sobrio entre una arquitectura de red neuronal densa frente a nuestra representación compacta mediante características geométricas del tablero.

### Diapositiva 2: Formulación MDP y Abstracción Temporal
*   **El concepto:** El control no se ejerce a nivel de frame, sino mediante acciones discretas correspondientes a colocaciones válidas.
*   **Dato clave:** El factor de ramificación se reduce a $\sim 40$ acciones posibles por pieza.
*   **El núcleo:** La transición determinista al *Afterstate* (el tablero posterior al *hard-drop* y limpieza de líneas).
*   **Detalle Visual Exigido:** Un esquema que muestre el estado original $s = (b, p)$ (tablero y pieza actual), una transición etiquetada con la acción macro $a = (r, c)$ (rotación y columna), que conduce al *afterstate* $\sigma = f(s, a)$.

### Diapositiva 3: Extracción de Características y Aceleración JIT
*   **El concepto:** Mapeo del *afterstate* a un vector de 6 características geométricas: altura agregada, huecos, rugosidad, líneas eliminadas, altura máxima y pozos.
*   **Dato clave:** Aproximación lineal de la función de valor $V(\sigma) = w^T\varphi(\sigma)$.
*   **Aceleración computacional:** Para viabilizar el entrenamiento rápido sobre millones de estados, la extracción de características se compila *Just-In-Time* (JIT).
*   **Detalle Visual Exigido:** Un tablero de Tetris ilustrando las 6 características para que el diseño visual sea riguroso. Para graficarlas, ten en cuenta sus definiciones formales:
    - `aggregate_height`: La suma de las alturas de todas las columnas.
    - `holes`: Celdas vacías bloqueadas desde arriba por al menos una celda ocupada.
    - `bumpiness`: La suma de las diferencias absolutas de altura entre columnas adyacentes (los desniveles de la superficie).
    - `lines_cleared`: Filas completas que se van a eliminar.
    - `max_height`: La columna más alta del tablero.
    - `wells`: Columnas vacías o valles flanqueados por columnas más altas a ambos lados (tratando las paredes del tablero como barreras de altura infinita).
    A un lado del gráfico, incorpora un fragmento de código ilustrativo mostrando el decorador `@njit(cache=True, fastmath=True)`, seguido de la fórmula de valor $V(\sigma) = w^T\varphi(\sigma)$.

### Diapositiva 4: Diferencia Temporal y Estabilidad Numérica
*   **El concepto:** El entrenamiento TD(0) produce una política competitiva pero numéricamente inestable.
*   **Dato de impacto:** El mejor hiperparámetro logra $409$ líneas promedio ($\gamma=0.999$, $\alpha=0.0005$).
*   **Inestabilidad numérica:** La magnitud de los pesos diverge ($||w|| \approx 10^5$). La política sobrevive gracias a la invarianza de escala de la selección $\arg\max$.
*   **Detalle Visual Exigido:** Dos gráficos analíticos. A la izquierda, una curva de aprendizaje (líneas eliminadas vs episodios). A la derecha, un gráfico ilustrando la divergencia de la norma de los pesos ($||w|| \approx 10^5$). En el centro, la fórmula de actualización $w \leftarrow w + \alpha \delta_t \varphi(\sigma_t)$ y la política $\arg\max_a w^T\varphi(f(s,a))$ evidenciando la invarianza de escala.

### Diapositiva 5: Optimización mediante CEM
*   **El concepto:** El Método de Entropía Cruzada optimiza directamente la política mediante búsqueda estocástica poblacional.
*   **El mecanismo:** Distribución Gaussiana $\to$ Muestreo de candidatos $\to$ Selección de élite $\to$ Reajuste de parámetros.
*   **Clave de robustez:** Uso de ruido decreciente y evaluación justa mediante *Common Random Numbers* (mismas semillas).
*   **Detalle Visual Exigido:** Una distribución normal $\mathcal{N}(\mu, \Sigma)$ con el percentil superior (la élite $\mathcal{E}$) sombreado. Una flecha indicando el desplazamiento de $\mu$ hacia dicha élite. Incluir la restricción de calibración final del modelo: normalización estricta $\lVert w \rVert = 1$.

### Diapositiva 6: La Ilusión del Corto Plazo (Evaluación a 1000 pasos)
*   **El concepto:** Bajo un horizonte artificial estricto (1000 colocaciones), el rendimiento de ambos agentes entrenados parece similar.
*   **Los datos a mostrar:** Aleatorio: 0 | Heurístico: 184 | TD(0): 338 | CEM: 371.
*   **Conclusión Parcial:** Ocurre un "efecto techo". Ambos agentes logran sobrevivir el límite de los 1000 pasos; la métrica aquí solo está capturando ligeras variaciones en la eficiencia de empaquetado, no la capacidad real de supervivencia.
*   **Detalle Visual Exigido:** Un gráfico de barras minimalista. La parte superior del gráfico debe estar visualmente seccionada o truncada por una línea punteada que diga "Límite artificial: 1000 pasos", mostrando cómo las barras de TD(0) y CEM "chocan" contra este techo.

### Diapositiva 7: Desenmascarando la Inestabilidad (Evaluación a 5000 pasos)
*   **El concepto:** Al extender el horizonte a 5000 colocaciones, se revela la verdadera fragilidad de la divergencia en TD(0). Los errores acumulados fuerzan un estado terminal prematuro, mientras CEM sostiene la supervivencia.
*   **Los datos a mostrar:** TD(0): 563 ($\pm 341$, máx. 1441) | CEM: 1048 ($\pm 679$, máx. 1999).
*   **El remate analítico:** La factura matemática del método semi-gradiente. $||w||_{TD} = 152,514.1$ frente a la estabilidad garantizada de $||w||_{CEM} = 1.0$.
*   **Detalle Visual Exigido:** Un nuevo gráfico de barras, ahora sin techo, mostrando cómo la barra de CEM duplica a la de TD(0). En un espacio destacado al lado, contrastar tipográficamente las normas: un enorme "$||w||_{TD} \approx 1.5 \times 10^5$" en rojo o tono de alerta, contra un elegante "$||w||_{CEM} = 1.0$" en el color principal de la presentación.

### Diapositiva 8: Conclusiones
*   **El concepto:** Cierre de la presentación.
*   **Mensaje:** Una representación matemática compacta acoplada al método de optimización adecuado constituye una alternativa efectiva a los métodos masivos en problemas de alta complejidad.
*   **Detalle Visual Exigido:** Fondo oscuro, texto en fuente sans-serif geométrica en el centro, y una figura sutil (por ejemplo, el tetrominó "O" o "I") minimalista como marca de agua.

---
Genera la presentación ahora.