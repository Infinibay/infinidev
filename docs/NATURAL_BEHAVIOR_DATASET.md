# Natural Behavior Dataset and Mini-Head

Fecha: 2026-08-11

Estado: pipeline y campaña E2E completos; modelo seleccionado para shadow
offline, no para intervenciones productivas.

## Corpus natural

`bench.behavior_natural_corpus` extrae ventanas de `run.json` reales. No guarda
chain-of-thought, argumentos de `think`, cuerpos de resultados, excerpts de
código, secretos, emails ni paths personales. Las etiquetas automáticas sólo
se aprueban cuando existe evidencia observable; el resto va a un fichero de
revisión separado.

La extracción histórica produjo:

- 56 runs, 9 familias de proyecto y 4 modelos;
- 150 ventanas aprobadas y 51 para revisión;
- 33 negativos difíciles;
- 118 `uncategorized`, 19 `healthy_progress`, 7 `excessive_exploration`,
  5 `retry_loop` y 1 `command_timeout`.

Se detectó y corrigió un falso positivo del extractor: varias lecturas de
offsets distintos en el mismo archivo compartían firma y parecían un retry.
La firma ahora incluye `offset`, `start_line` y `limit`. Un exit code no cero
también cuenta como fallo aunque el hook no lo haya marcado.

## Holdout C/C++ con MiniMax

Los repositorios fijados por commit están en
[`behavior_natural_oss.sources.json`](../bench/behavior_natural_oss.sources.json):
jsmn (C) y cxxopts (C++). Ninguna de sus ventanas participó en fit o selección
de umbrales.

Se ejecutaron cuatro tareas en copias limpias: dos implementaciones en español
y portugués, y dos reviews read-only en francés e inglés.

| Tarea | Resultado | Tool calls | Prompt tokens | Completion tokens |
| --- | ---: | ---: | ---: | ---: |
| jsmn: control chars | pass | 25 | 394.040 | 8.967 |
| cxxopts: bool case-insensitive | pass | 29 | 429.983 | 9.187 |
| jsmn: review incremental | pass | 19 | 207.557 | 9.573 |
| cxxopts: review bool | pass | 23 | 403.214 | 9.476 |

El holdout resultante contiene 15 ventanas aprobadas: 3 positivas, 12
`uncategorized` y 8 negativos difíciles. Otras 8 ventanas ambiguas permanecen
en revisión.

Una primera ejecución read-only descubrió que `TreeEngine` tenía presupuestos
independientes y podía escapar el límite del benchmark: llegó a 27 tools y 19
llamadas LLM sólo dentro del árbol. Esa ejecución se abortó y se excluyó. El
runner acotado ahora pasa `allow_explore=False`; la misma tarea se repitió y
terminó dentro del contador principal.

## Mini-modelo y ablaciones

Los splits son por familia de proyecto:

- calibration: 87 ventanas;
- validation: 62 ventanas;
- holdout ciego: 15 ventanas.

La selección de arquitectura y umbrales usó sólo validation. El encoder
evaluado fue exactamente `ken/static-qwen3-r512-v2`, espacio
`ken/static-qwen3-r512-v2:1024:e6ab79ad2462d447`.

| Arquitectura | Precisión selectiva holdout | Recall positivo | FP neutrales | Exact match |
| --- | ---: | ---: | ---: | ---: |
| prototipos semánticos anteriores | 0% | 0% | 66,7% | 26,7% |
| embedding + head | N/A (0% coverage) | 0% | 0% | 80,0% |
| embedding + observables | 75% | 100% | 8,3% | 93,3% |
| observables solamente | 100% | 100% | 0% | 100% |

El híbrido y observables-only empataron en validation. La regla predeclarada
elige la alternativa más simple, por lo que el artefacto seleccionado usa 11
features y ocupa 1.271 bytes. No requiere una inferencia de embedding para
detectar estos estados.

Esto no invalida el embedding para intención o claims semánticos. Sí demuestra
que, para loops, progreso y verificación, una representación estructurada es
más barata y generalizó mejor en este holdout.

## Reproducción

```bash
uv run python -m bench.behavior_natural_corpus bench/runs \
  --output /tmp/natural-approved.jsonl \
  --review-output /tmp/natural-review.jsonl \
  --repository-root .

uv run python -m bench.behavior_natural_head \
  /tmp/natural-approved.jsonl \
  /tmp/natural-holdout-approved.jsonl \
  --artifact /tmp/natural-head.npz
```

Los runs y corpora derivados permanecen locales. El extractor, contratos,
fixtures, tests y hashes son versionables.

## Gate pendiente

El resultado no autoriza producción. Hay sólo dos familias holdout, tres
positivos y ningún `retry_loop` positivo natural. El siguiente gate exige:

- más incidentes reales, especialmente retries equivalentes;
- revisión humana de las 51 ventanas ambiguas;
- nuevas familias de proyecto que no participen en fit ni thresholding;
- replay shadow que confirme cero falsas intervenciones;
- E2E que pruebe mejora causal, no sólo clasificación offline.
