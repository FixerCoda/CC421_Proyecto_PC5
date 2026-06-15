# Tetris RL - Optimización de Políticas de Juego mediante Aprendizaje por Refuerzo y Extracción de Características

Proyecto del curso de **CC421 - Inteligencia Artificial**. Un agente aprende a jugar Tetris (1 jugador, tablero 20×10, 7 piezas estándar) optimizando una **función de valor lineal sobre características del tablero** `V(σ) = wᵀφ(σ)` mediante el **Cross-Entropy Method (CEM)**.

## Estructura del repositorio

```
src/tetris_rl/      código fuente (env, features, data, agents, model, eval, utils)
scripts/            puntos de entrada (generate_data, train_cem, evaluate)
notebooks/          EDA y análisis de resultados
data/               trayectorias self-play y features (generadas)
models/             pesos w aprendidos por CEM
docs/               documentación matemática en LaTeX (informe)
```

## Requisitos

```
python >= 3.10
pip install -r requirements.txt
```

## Uso

```
python scripts/generate_data.py     # genera datos self-play (baselines)
python scripts/train_cem.py         # entrena la política con CEM
python scripts/evaluate.py          # evalúa random vs heurístico vs CEM
```
