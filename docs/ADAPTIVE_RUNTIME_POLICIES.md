# Adaptive Runtime Policies

Estado: detector determinista integrado en shadow; mini-modelo de reasoning
visible integrado con intervenciones evidence-gated. Véase
[`CONDITIONAL_REASONING_PROMPTS.md`](CONDITIONAL_REASONING_PROMPTS.md).

La implementación y el E2E reproducible están resumidos en
[`ADAPTIVE_RUNTIME_OSS_E2E.md`](ADAPTIVE_RUNTIME_OSS_E2E.md). La campaña inicial
pasó los gates de calidad y coste, pero sigue siendo evidencia exploratoria:
una sola de las tres trayectorias candidate recibió una intervención.

La segunda campaña, con trayectorias naturales y proyectos C/C++ ciegos, está
en [`NATURAL_BEHAVIOR_DATASET.md`](NATURAL_BEHAVIOR_DATASET.md). Su resultado
favorece una cabeza de observables de 1,3 KB para estados de ejecución; el
embedding queda reservado para categorías donde la semántica demuestre valor.

## Idea

Conditional Task Policies decide el método inicial a partir del request. La
extensión adaptativa observa la trayectoria real y activa instrucciones cortas
solamente cuando pueden corregir la siguiente decisión del modelo.

```text
request del usuario
  -> TaskProfile estable para toda la tarea

eventos observables del runtime
  -> BehaviorProfile temporal
  -> una intervención acotada para el próximo turno
  -> observar transición
  -> expirar, mantener con histéresis o escalar
```

Esto generaliza el conditional prompting sin crear un system prompt distinto
para cada combinación. El prefijo estable, permisos e instrucciones del repo
no cambian. Las intervenciones se insertan después de ese prefijo y antes del
estado dinámico del Step.

## Qué se observa

El controlador no infiere chain-of-thought privado. Puede usar reasoning o
summaries que la API expone explícitamente, además de evidencia auditable:

- mensajes visibles y summaries de Steps;
- plan y transiciones de estado;
- nombres y argumentos normalizados de herramientas;
- errores y cambios materiales entre retries;
- fingerprint del diff neto;
- comandos y resultados de tests;
- archivos leídos/editados y límites de scope;
- tokens, iteraciones y tiempo desde el último progreso.

Una ventana puede combinar el embedding de los últimos mensajes con features
deterministas pequeñas. Ejemplo:

```json
{
  "message_embedding": "ken/static-qwen3-r512-v2:1024:...",
  "same_step_windows": 3,
  "net_diff_changed": false,
  "new_test_fingerprint": false,
  "equivalent_tool_retry": true,
  "outside_scope_paths": 0
}
```

## Taxonomía inicial

| Señal | Evidencia mínima | Intervención posible |
| --- | --- | --- |
| `healthy_progress` | diff/test/plan cambia de forma útil | ninguna |
| `no_progress` | ventanas repetidas sin diff ni test nuevo | limitar una ventana a acciones de progreso |
| `retry_loop` | misma operación sin cambio material | exigir cwd/argumentos/estrategia distintos |
| `overplanning` | planes crecientes sin ejecución | reducir el horizonte a 1-3 Steps |
| `verification_gap` | intento de completar tras editar sin check relevante | exigir verificación enfocada |
| `scope_drift` | path o objetivo fuera del Task | reinyectar scope y exigir evidencia de necesidad |
| `speculative_claims` | claims severos sin evidencia accesible | separar hechos, inferencias y preguntas |
| `premature_completion` | criterios pendientes o evidencia ausente | bloquear cierre y enumerar la evidencia faltante |
| `uncategorized` | ninguna señal supera sus gates | ninguna |

Las clases describen una intervención útil, no cada comportamiento imaginable.
Una categoría nueva necesita evidencia de que cambia el resultado y una
política distinta; no basta con que tenga un nombre diferente.

## Modelo pequeño

El clasificador puede reutilizar el mismo encoder estático que el router de
tareas, pero sólo cuando una ablación demuestra que aporta. La arquitectura
evaluada mantiene el runtime simple:

```text
embedding agregado de mensajes (1024)
+ features deterministas normalizadas
  -> head lineal multi-label
  -> threshold por clase + margen + abstención
```

Un head lineal ocupa entre uno y decenas de KB y se ejecuta con NumPy. En la
campaña natural inicial, los observables solos superaron al embedding y al
híbrido para `healthy_progress`/`excessive_exploration`. Si un MLP pequeño
mejora un held-out real, puede usarse una capa de 16-32 unidades, pero el modelo
más complejo no sustituye un dataset mejor.

### Memoria jerárquica, no contexto crudo

El mini-modelo no necesita una secuencia de tokens de todo el run. Cada evento
se comprime una vez a un vector `float16` de 1024 dimensiones (~2 KB):

```text
16-32 eventos recientes
+ centroides/resúmenes por Step
+ evidencia crítica fijada (diff, tests, errores, scope)
  -> similitud con el estado actual
  -> top-k 4 eventos
  -> pooling + features deterministas
  -> head
```

Esto es una forma de atención sparse externa y auditable: cada vector conserva
un puntero al evento original. Con 32 eventos, el buffer ocupa alrededor de 64
KB y la atención densa entre eventos también sería barata.

DeepSeek V4 combina compresión y DeepSeek Sparse Attention para contexto de
hasta un millón de tokens dentro de un transformer grande. La idea inspira la
jerarquía/top-k, pero incorporar ese mecanismo completo al clasificador pequeño
añadiría un indexer y complejidad sin resolver un bottleneck medido. Solo se
consideraría un tiny transformer sparse si un benchmark futuro demuestra que
el orden fino de cientos de eventos mejora respecto al pooling.

`uncategorized` se representa como ninguna activación, no como un softmax que
obliga a elegir. El dataset sí etiqueta explícitamente negativos para auditar
su diversidad: conversación, status, explicación de texto citado, tareas
saludables, contenido fuera de dominio y ventanas ambiguas.

## Ciclo de vida de una intervención

1. Una señal necesita evidencia determinista o dos ventanas coherentes; un
   embedding aislado no modifica el runtime.
2. Conflictos se resuelven por severidad y especificidad; normalmente se
   inyecta una sola política.
3. La política se aplica al rol y fase pertinentes durante una ventana.
4. Después existe un cooldown para evitar oscilación o prompt piling.
5. Progreso material retira la intervención inmediatamente.
6. Reincidencia requiere nueva evidencia; no se conserva por inercia.

Las políticas pueden restringir acciones o exigir evidencia, pero nunca
ampliar autoridad, scope, permisos, commit o publicación.

## Dataset

La unidad de clasificación es una ventana de trayectoria, no una frase
aislada. Cada ejemplo necesita:

- request y `TaskProfile` relevantes;
- 2-4 eventos anteriores en orden;
- features deterministas y sus valores originales;
- etiqueta multi-label o `uncategorized` explícito;
- acción correctiva esperada;
- explicación revisada de por qué esa intervención aplica;
- modelo/proveedor y versión de prompts;
- familia de escenario y split;
- estado `draft`, `approved` o `rejected` con reviewer.

Las familias completas pertenecen a un solo split. Variantes de la misma
trayectoria, repo o fallo no pueden cruzar calibration, validation y held-out.
El corpus debe contener ejecuciones sanas en cantidad comparable: entrenar solo
con fallos haría que el controlador interrumpa progreso válido.

El modo de recolección recomendado es aprendizaje activo en shadow mode:

- disagreements entre head, recuperación y `Task.kind`;
- abstenciones de bajo margen;
- intervenciones que no cambiaron el estado;
- runs con alto coste, scope drift o cierre rechazado;
- muestras aleatorias de progreso sano para controlar falsos positivos.

Antes de almacenar texto natural se eliminan secrets, paths personales y datos
del usuario. Los ejemplos sintéticos complementan cobertura, pero no sustituyen
trazas aprobadas.

## Evaluación

El clasificador se mide primero offline:

- precision/recall, coverage y abstención por señal;
- falsos positivos sobre progreso sano;
- estabilidad por idioma, proveedor y tipo de repo;
- family leakage, duplicados y balance de modelos;
- calibración de score y margen.

El controlador se mide después E2E:

- tiempo y tokens hasta progreso material;
- pass rate y criterios de aceptación;
- scope y autoridad;
- retries equivalentes;
- suficiencia de verificación;
- falsos bloqueos de runs que baseline completa;
- coste adicional por intervención.

La matriz incluye baseline, intervención correcta, intervención errónea y
shadow. Un detector preciso no se despliega si su prompt no mejora resultados.

## Relación con mecanismos existentes

El control de estancamiento actual ya sigue este patrón estrecho: combina
similitud de summaries con ausencia de diff y tests nuevos, usa histéresis y
reduce temporalmente la superficie de herramientas. Debe evolucionar como una
política del nuevo controlador, no duplicarse en otro detector independiente.

`TaskProfile` continúa estable y explica qué se está intentando hacer.
`BehaviorProfile` es efímero y explica qué problema observable requiere una
corrección ahora.

## Gate de rollout

No activar intervenciones de producción hasta cumplir simultáneamente:

- dataset natural aprobado y auditado;
- held-out por familia nunca usado para tuning;
- precisión muy alta sobre señales restrictivas;
- falsos positivos cercanos a cero sobre progreso sano;
- mejora E2E por modelo/proveedor sin regresión de autoridad o scope;
- replay y event log suficientes para explicar cada activación.
