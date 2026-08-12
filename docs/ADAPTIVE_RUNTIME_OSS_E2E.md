# Adaptive Runtime: MiniMax M3 OSS E2E

Fecha: 2026-08-11

Estado: campaña exploratoria aprobada; producción permanece en shadow/opt-in.

## Qué se implementó

El runtime observa únicamente transcript, herramientas, resultados, plan,
diffs y tests. No captura ni infiere chain-of-thought privado. Las señales
deterministas actuales son:

- `excessive_discovery`: al menos ocho llamadas, cuatro de descubrimiento y dos
  lecturas, sin edición ni evidencia real de test;
- `command_timeout`;
- `tool_schema_mismatch`;
- `premature_completion`.

Sólo `excessive_discovery` y `command_timeout` pueden emitir una intervención.
La intervención es corta, se entrega en la siguiente llamada inmediata sin
otra request LLM, tiene deduplicación y cuota, y puede reducir temporalmente el
contexto de archivos abiertos de 48.000 a 16.000 caracteres.

Los defaults productivos son:

```text
ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED=true
ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE=true
ADAPTIVE_RUNTIME_SEMANTIC_SHADOW_ENABLED=false
```

## Fixtures y contrato

Se usaron copias frescas de Requests/Python, p-map/JavaScript y
ripgrep/Rust, fijadas por commit en
[`behavior_runtime_oss.sources.json`](../bench/behavior_runtime_oss.sources.json).
Cada condición recibió el mismo request, Task estructurado y verificador
declarado. Los source clones permanecieron intactos. Los verificadores fueron:

- Requests: 33 tests;
- p-map: `npm test`, 52 tests baseline y 54 candidate;
- ripgrep: 11 tests del filtro `grep-cli human`.

El comparador excluye caches, dependencias, `target` y `.egg-info` generados.

## Resultado pareado

| Métrica | Baseline | Candidate | Cambio |
| --- | ---: | ---: | ---: |
| verificadores correctos | 3/3 | 3/3 | igual |
| prompt tokens | 1.095.305 | 755.000 | -31,1% |
| completion tokens | 31.985 | 22.864 | -28,5% |
| tool calls | 71 | 66 | -7,0% |
| latencia | 582,4 s | 483,3 s | -17,0% |
| scope regressions | 0 | 0 | igual |

Ninguna tarea aumentó tool calls. p-map recibió una intervención
`excessive_discovery` y redujo prompt tokens 26,0%, completion 25,2%, tool calls
de 26 a 25 y latencia 10,3%, manteniendo `npm test` en verde. Requests y
ripgrep no recibieron intervención: sus diferencias no prueban causalidad y
pueden incluir variación de MiniMax.

Reproducción del reporte:

```bash
uv run python -m bench.behavior_runtime_eval \
  bench/behavior_runtime_oss.conditions.json \
  bench/runs/20260811-behavior-oss-adaptive-v2-pmap/observations.jsonl \
  bench/runs/20260811-behavior-oss-adaptive-v2-rest/observations.jsonl \
  --intervention-review bench/behavior_runtime_oss.intervention_review.json
```

Los runs locales no forman parte del paquete; el dataset, manifiesto, runner y
evaluador sí son versionables.

## Resultado del mini-modelo

El encoder exacto fue `ken/static-qwen3-r512-v2`, espacio observado
`ken/static-qwen3-r512-v2:1024:e6ab79ad2462d447`.

Se evaluaron 112 ejemplos holdout multilingües, separados de 252 ejemplos de
calibración y 112 de validación:

| Clasificador | Tamaño adicional | Cobertura | Precisión selectiva | Falso positivo neutral |
| --- | ---: | ---: | ---: | ---: |
| similitud a prototipos | sin head | 58,9% | 60,6% | 5,36% |
| ridge lineal | 23,6 KB | 28,6% | 28,1% | 0,89% |

El head lineal sobreajustó plantillas/idiomas y el clasificador por prototipos
confundió especialmente `verification_gap` y `speculative_claim`. Ninguno está
habilitado para intervenir. Agregar más ejemplos repetitivos no resolvería el
problema: el próximo corpus necesita trayectorias naturales, más diversidad de
paráfrasis por clase, negativos difíciles y splits por familia/proyecto.

## Límite de la conclusión

Tres tareas sirven para falsificar fallos obvios, no para aprobar rollout. El
próximo gate exige repeticiones con orden contrabalanceado, más proyectos e
idiomas, y revisión humana de falsos positivos. Hasta entonces el detector
registra eventos por defecto y las intervenciones requieren opt-in explícito.
