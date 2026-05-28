# Career Path LLM — Planificador de trayectoria académica con variantes y análisis

Proyecto de **planificación de cursos** donde, dada una instancia con cursos, prerrequisitos, habilidades iniciales y objetivo de habilidades, se busca una **secuencia válida de cursos** que cumpla el objetivo minimizando el coste (créditos). El proyecto incluye:

- Algoritmos base (por ejemplo `greedy`, `a_star`, `exact_small`).
- Variantes de ejecución **A–D** (incluyendo variantes asistidas por LLM cuando aplica).
- Un _runner_ de experimentos determinista (semillas) que exporta resultados a `results/`.
- Un notebook de análisis reproducible que genera tablas y figuras en `report/`.

> Este README está enfocado en cómo **ejecutar** el proyecto (un caso, experimento completo y análisis) y en cómo interpretar la estructura de archivos.

---

## Requisitos

- Python 3 (recomendado 3.10+).
- Dependencias instalables vía `pip` (ver `requirements.txt`).
- Opcional: **Ollama** si se desean ejecutar variantes que usen LLM (según configuración en `config/llm_config.json`).

---

## Instalación

Crea y activa un entorno virtual (opcional pero recomendado), e instala dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si tu configuración de PowerShell no permite activar scripts, puedes usar `cmd.exe` o ajustar la política de ejecución.

---

## Punto de entrada (CLI)

El punto de entrada unificado es `main.py` en la **raíz** del repo:

```powershell
python .\main.py --help
```

Subcomandos disponibles:

- `run`: ejecuta **una sola instancia** con una variante y algoritmo.
- `experiments`: ejecuta la **matriz de experimentos** y genera artefactos en `results/`.
- `check-ollama`: valida la conectividad/config del endpoint de Ollama.
- `paths`: imprime rutas clave del repo.

### Ver rutas relevantes

```powershell
python .\main.py paths
```

---

## Ejecutar un solo caso (una instancia)

Ejecuta la variante A (sin LLM) sobre una instancia pequeña:

```powershell
python .\main.py run --instance .\data\instances\synthetic_10_courses_01.json --variant A --algorithm greedy --objective "python"
```

Guardar el resultado en JSON (trayectoria por corrida):

```powershell
python .\main.py run --instance .\data\instances\synthetic_10_courses_01.json --variant A --algorithm greedy --objective "python" --output .\results\trajectories\demo.json
```

Notas:

- Para variantes **B/C/D** normalmente necesitas `--use_ollama`.
- `--objective`:
  - Variante **B**: texto libre (se interpreta con el LLM).
  - Variantes **A/C/D**: habilidades separadas por comas.

---

## Runner de experimentos (matriz completa)

El runner vive en `experiments/run_experiments.py` y escribe un CSV consolidado + JSONs por corrida en `results/`.

Ejecución típica (se recomienda correr desde la raíz del repo):

```powershell
python .\main.py experiments
```

Para ver las opciones del runner:

```powershell
python .\main.py experiments -- --help
```

Salidas generadas (según configuración del runner):

- `results/experiment_results.csv`
- `results/experiment_results.json`
- `results/run_metadata.json`
- `results/trajectories/*.json` (uno por ejecución)
- `results/experiments.log`

---

## Configuración del LLM (Ollama)

1. Revisa/ajusta `config/llm_config.json` (endpoint/modelo/timeout/etc.).
2. Verifica conectividad y modelo:

```powershell
python .\main.py check-ollama
```

Si falla, asegúrate de tener Ollama corriendo (por ejemplo `ollama serve`) y de que el modelo esté disponible.

---

## Análisis y reporte

El análisis reproducible está en:

- `report/analysis.ipynb`

Entradas típicas:

- `results/experiment_results.csv`
- `results/run_metadata.json`
- `results/trajectories/*.json`

Salidas típicas:

- Tablas: `report/tables/*.csv`
- Figuras: `report/figures/*.png`

Recomendación:

- Abre `report/analysis.ipynb` en VS Code/Jupyter y ejecuta celdas en orden para regenerar el reporte.

---

## Estructura del repositorio

- `src/`: implementación de algoritmos, variantes y utilidades.
- `data/instances/`: instancias (JSON) para experimentos.
- `experiments/`: runner de experimentos y utilidades de ejecución.
- `scripts/`: scripts auxiliares (por ejemplo validación de Ollama).
- `results/`: artefactos generados por experimentos (CSV, logs, trajectories).
- `report/`: notebook de análisis y artefactos exportados (tablas/figuras).
- `tests/`: pruebas automatizadas.

---

## Formato de instancias

El formato está documentado en `data/README.md`. En resumen, cada JSON contiene:

- `skills`: habilidades disponibles.
- `courses`: lista de cursos con atributos (id, nombre, créditos, prerequisitos, skills granted/required, etc.).
- `initial_skills`: habilidades iniciales.
- `target_skills`: objetivo.

---

## Pruebas

Ejecuta pruebas con `pytest` (si está disponible en tu entorno):

```powershell
pytest -q
```
