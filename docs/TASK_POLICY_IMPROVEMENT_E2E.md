# Conditional Task Policies: improvement gate

Fecha: 2026-08-11

## Resultado

El compositor ya no se considera útil por el solo hecho de reemplazar un
prompt fijo. La inyección es prompt-only, se evalúa por fragmento y modelo, y
debe conservar calidad y scope mientras mejora coste medible.

Para MiniMax M3, solamente `refactor.developer@1` está aprobado. El router
sigue detectando y registrando las demás categorías, pero sus fragmentos no se
inyectan hasta ganar su propio E2E. Rutas de modelo desconocidas también fallan
cerradas cuando `TASK_POLICIES_EVIDENCE_GATED` está activo.

## Aislamiento

Antes de repetir la campaña se eliminaron tres contaminaciones:

- el `TaskProfile` dejó de duplicarse en el XML entregado al modelo;
- el perfil dejó de modificar implícitamente el catálogo de herramientas;
- baseline y candidate usan la misma identidad developer filtrada por las
  herramientas realmente disponibles.

En el par refactor, ambas condiciones recibieron 1.903 caracteres de user
prompt y 17.139 caracteres de schemas. La única diferencia efectiva fue el
fragmento dinámico de 460 caracteres; los dos caracteres de diferencia en el
prefijo medido son los separadores del cache breakpoint.

## Par refactor

Petición semántica sin verbo de categoría: `Make warning badges easier to scan
while keeping all other output stable.` La cabeza y el retriever acordaron
`refactor.preserve_behavior` con score 0,825 y sin fallback LLM.

| Métrica | Baseline | Candidate | Reducción |
| --- | ---: | ---: | ---: |
| verificador verde | sí | sí | igual |
| prompt tokens | 286.320 | 49.916 | 82,6 % |
| completion tokens | 10.156 | 1.805 | 82,2 % |
| tool calls | 26 | 8 | 69,2 % |
| latencia | 132,24 s | 32,34 s | 75,5 % |
| cambios fuera de scope | 0 | 0 | igual |

Este par demuestra una mejora concreta para una familia y una ruta de modelo;
no demuestra todavía una mejora general.

## Calibración bugfix

Después de corregir el gate de progreso del loop, baseline y candidate
arreglaron `normalize_tags`, modificaron solamente `src/tags.py` y pasaron los
cuatro tests. La política no ganó:

| Métrica | Baseline | Candidate v1 |
| --- | ---: | ---: |
| prompt tokens | 76.964 | 91.508 |
| completion tokens | 2.091 | 3.375 |
| tool calls | 8 | 12 |
| latencia | 53,01 s | 66,96 s |

Una versión más corta tampoco ganó en calibración: 191.660 prompt tokens, 6.717
completion tokens, 18 tools y 111,45 s. Por eso `bugfix.developer` queda
detectable pero no aprobado para MiniMax M3. Reescribirlo otra vez sobre la
misma familia sería tuning leakage.

## Abstención negativa

Una petición que solo pedía responder `LISTO` hizo abstener a la cabeza por
margen insuficiente. Baseline y candidate recibieron el mismo system core,
user prompt y schemas; no hubo fragmentos ni cambios de archivos. Ambos
terminaron correctamente. La diferencia de coste entre las dos trayectorias es
varianza del proveedor, no efecto del tratamiento.

El retriever de prototipos ya no puede forzar una política después de una
abstención del mini-modelo. `uncategorized` y los márgenes bajos son ahora
decisiones finales.

## Corrección del loop descubierta por el E2E

El benchmark detectó un deadlock independiente del prompt: si el modelo creaba
un Step y editaba en la misma ventana, o si la edición ocurría antes de que ese
Step quedara activo, el fingerprint podía capturarse después del cambio. El
motor concluía erróneamente que no había editado y repetía el Step.

El loop conserva ahora el fingerprint previo a la ventana y permite cerrar un
Step model-authored cuando existe una edición real y un test verde asociado al
fingerprint actual. Un test anterior a una edición nueva no satisface el gate.

## Gate reproducible

```bash
uv run python -m bench.task_policy_improvement_eval \
  bench/task_policy_improvement.gates.json \
  /path/to/refactor-observations.jsonl \
  /path/to/abstention-observations.jsonl
```

El gate exige cero regresiones de éxito/scope, aislamiento prompt-only,
fragmentos esperados, ninguna regresión agregada de tokens/tools y al menos
dos métricas con una mejora mínima del 5 %. En esta campaña pasaron los nueve
gates; las reducciones agregadas fueron 76,0 % prompt tokens, 71,7 % completion
tokens, 65,4 % tools y 65,0 % latencia.

Los artefactos E2E permanecen fuera del repo porque contienen trayectorias y
outputs completos del proveedor. Los datasets, manifests, gates y verificadores
sí están versionados para poder repetir la campaña.

## Seguimiento bugfix por modelo

Una nueva versión outcome-focused fue evaluada con GPT-5.6 Terra y MiniMax M3.
Terra redujo prompt 16,3 %, completion 12,8 %, tools 5,9 % y latencia 25,7 %;
MiniMax produjo un resultado mixto y no fue aprobado. Véase
[`TASK_POLICY_BUGFIX_MODEL_E2E.md`](TASK_POLICY_BUGFIX_MODEL_E2E.md).
