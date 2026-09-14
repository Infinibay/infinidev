# Engine Task: análisis del ciclo de cierre y primer tramo de mejora

## Resumen

**Qué se midió.** Todas las comparaciones son pareadas por `(tarea, repetición)`
sobre `MiniMax-M3`, con test de signos y un mínimo de 6 parejas sin empate antes
de declarar un resultado. El detalle está en §6; los límites, en cada sección.

| cambio | efecto medido | estado |
| --- | --- | --- |
| Variante de prompt `lean` (protocolo, identidad de ingeniería y barras de producto compactos) | −35,9 % tokens de prompt y −33 % latencia, **12/4 parejas, p=0,0768**; **tool calls −15,4 %, 12/2, p=0,0129**; 16/16 success; generaliza sobre 8 formas de tarea | **dirección medida, no resuelta** para tokens y latencia; **resuelto** para tool calls. El p=0,0005 que figuraba aquí era del test de signos defectuoso (§6.1.40) |
| Respuesta final con contrato de verificación | línea `Verification:` 6/16 → 16/16; respuestas sobre 250 palabras 2/16 → 0/16 | **medido** |
| Cierre de Step rechazado en silencio (livelock) | racha de rondas sin trabajo: 11 → 0 | **medido** (falla intermitente) |
| Presión de agrupación de tool calls | sin efecto (0,90 → 1,00 llamadas por ronda) | **resultado negativo**, apagado por defecto |
| Catálogo de políticas condicionales | dos muestras independientes en el modo que se envía: **−10,4 %** (11/7, p=0,48) y **−25,6 %** (7/1, p=0,070) de tokens facturados, 16/16 y 8/8 de éxito. Las dos van en la misma dirección; agrupadas, 18/8, p=0,076. La latencia se movió −23,6 % y **−36,7 %** | **decisión: el flag queda en `true`**, pero por dirección replicada, no por significancia: los dos p-valores publicados antes (0,031 y 0,016) salían del test defectuoso (§6.1.40). El costo en latencia era ruido, con la diferencia entre brazos 10–25× menor que la dispersión dentro de un brazo (§6.1.26) |
| **Modo de engine por defecto vs `task`** | contador corregido, 10 parejas pareadas (9 sin empate): success **9/9 vs 9/9**, tokens facturados **−82,9 %**, tool calls **−66,7 %**, completion **−83,1 %**, todo con **p=0,0039**, y latencia −61,5 % con **p=0,0391** (§6.1.23). Los workers que escriben **están serializados por código** (`runtime.py:156`), así que la delegación no puede paralelizar trabajo de código (§6.1.12) | **medido, y sobrevive a la corrección del test de signos**: es la fila más fuerte del documento porque 9 de 9 parejas van en la misma dirección (§6.1.40) |
| El contador de tokens no contaba todo | el camino `task` omitía 5 fases y el `orchestrator` omitía **los loops de sus workers**: el costo real del orquestador es 1,68×–1,77× lo reportado. Corregido contando en la frontera del proveedor: **−89,0 % / −83,8 % por pareja y −73 % de rondas de modelo** | **defecto de medición corregido**; el −82,9 % pasa a ser un piso, no la cifra final (§6.1.8) |
| **El router de políticas, medido aislado** | ruteo local **4 ms**; `preferred` **2,91 s por turno** (12,1 s en el peor), 446 tokens de prompt por turno **invisibles para los contadores**, y quita una etiqueta de método justificada en 2 de 11 requests | **cambiado a `fallback`**: gratis en 10 de 11, conserva la capacidad y no resta (§6.1.21) |
| El nicho del orquestador (lectura independiente) contra `task` | `research-audit` ×3: **success 2/3 vs 3/3**, mediana de tokens 771 470 → 90 442 (−88,3 %), tool calls 31 → 10, latencia 405 s → 149 s | **medido**, 3 parejas: la prueba de signos no resuelve (p=0,25); el punto estimado y la falla sí hablan (§6.1.8) |
| Step que la recuperación dejaba sin salida en el orquestador | 1 de 3 auditorías fallaba con 771 470 tokens y `changed_paths: []`; `wide-sum` diagnosticado correcto y no aplicable | **arreglado**; después: `research-audit` 3/3, `wide-sum` 2/2 (§6.1.13) |
| Pregunta que nadie puede responder detenía la ejecución | `complex-plan` en modo `task`: 3/3 detenidas con **0 tokens**; después del arreglo **3/3 completan**, mediana 79 983 tokens | **arreglado** y medido (§6.1.14) |
| **Default de engine: `orchestrator` → `task`** | las tres mediciones de arriba, más `complex-plan` en `task` a −87,5 % del costo de `orchestrator` | **cambiado** (`config/settings.py:319-336`), con la medición en el comentario del ajuste |
| **`lean` en un repositorio que no se puede leer** | corpus v9: 8 hubs → 40 paquetes → 1 200 hojas. **success 3/3 vs 3/3**, tokens facturados −33,0 %, rondas −20,5 %; ninguna traza lee el árbol | **medido** (3 parejas: sonda de validez, no estimación). Cierra la última duda sobre `lean` (§6.1.20) |
| Corregir la hoja rota contra reescribir una sana | `default` reescribió una hoja conforme (`v + 9` → `v + 58`) en 1 de 3; `lean` corrigió `v - 40` en 3 de 3. Ambos pasan el verificador | **medido**: `located-the-leaf` 1,33 vs **2,00** (§6.1.20) |
| El mismo archivo contado dos veces (`abspath` vs `realpath`) | 22 de 32 ejecuciones traían el diff duplicado: `changed_lines` se inflaba al doble en unas y no en otras, y el revisor recibía el diff dos veces en el prompt | **arreglado en el producto y en la métrica**; los titulares no se mueven y una pareja de 16 cambia de dirección (§6.1.16) |
| **Calidad juzgada, por primera vez** | 16 ejecuciones pareadas contra sus ítems `human_review`, juez ciego: **default 1,81 vs `lean` 1,94** (máx. 2); 14 de 16 ítems empatan | **medido**; los dos ítems que difieren van a favor de `lean` con n=1, y así queda declarado (§6.1.17) |
| **Calidad de la configuración enviada** | 90 ejecuciones `lean` + `task`, **19 ítems decididos por programa**: **todos en 2,00 salvo `concise-handoff` en 1,62**, con las dos únicas fallas siendo corridas que no corrieron | **medido**; el proxy de longitud de §6.1.24 y los dos probes de vocabulario de §6.1.27 eran del instrumento, no del engine |
| **El cache de prompt** | la métrica existía en el engine y se descartaba: **ninguna de las 333 ejecuciones guardadas** la tenía. Ahora está en la fila de observación y en la comparación (las dos convenciones de proveedor) | **medido: 64,6 %–93,3 %, media 78 %** en la configuración enviada, que es el techo estructural `(N−1)·P/Σ` (§6.1.36) |
| El harness no atribuye el 22 % del payload | **Resuelto, y las dos exclusiones de §6.1.35 eran falsas.** El razonamiento del modelo **sí** se reenvía (`main` lo mete en el turno del asistente) y MiniMax lo factura a **1 token por 8 caracteres**, medido contra la API con la pendiente idéntica en los dos campos y lineal de 0 a 48 000 caracteres. Y los argumentos de las tool calls no son 2 513 caracteres: en una tarea que escribe archivos son el **30 % del payload** | **arreglado**: `trim_superseded_reasoning` borra el razonamiento de todo turno ya cerrado y conserva el material opaco; `measure_request_payload` ahora atribuye cada carácter y `payload_unattributed_chars` es 0 (§6.1.39) |
| **Reconstruir el contexto o dejarlo crecer** | El engine **ya deja crecer**: 269 de 308 corridas crecen monótonamente, 1,71× el payload inicial, **+2 277 caracteres por petición**. Con `k` = acierto de cache sobre fresco, un harness que reconstruye gana sólo si su working set `D <= d(N−1)[N−(1−k)(N−2)]/2N`; el prefijo estable y el recargo de escritura **se cancelan** entre los dos regímenes | **medido y contestado**: el working set real que el engine reconstruye mide **1 511 tokens** (p10 328, p90 2 431), el empate está en **`k = 0,49`**, y los nueve proveedores que publican cache (0,07–0,25) quedan del lado de dejar crecer; sólo `Groq`/`Fireworks`, sin cache publicado, favorecen reconstruir. Herramienta: `bench/context_regime_cost.py` (§6.1.40) |
| **El test de signos no contaba las parejas que empeoraban** | `trials = empates + mejoras`: las derrotas no entraban en el conteo, así que un resultado mixto se evaluaba como si sólo existieran las parejas favorables. **12 mejor / 4 peor daba p=0,0005 en vez de 0,0768**; con cero derrotas las dos fórmulas coinciden, y por eso el titular del modo de engine sobrevivió intacto. Recalculadas las **20 campañas guardadas**: **56 métricas se mueven y 16 pierden significancia** | **sexto defecto de instrumento, y el único que inflaba los titulares.** Corregido y con cinco tests contra binomios calculados a mano (§6.1.38). Las filas de arriba ya están corregidas; las tablas del cuerpo llevan su nota |
| **Corte del razonamiento reenviado** | el turno del asistente lleva el razonamiento del modelo y viaja en todas las peticiones siguientes; MiniMax lo factura a **1 token por 8 caracteres**, medido contra la API. `trim_superseded_reasoning` lo borra de todo turno ya cerrado y conserva el material opaco | **implementado y medido en dos campañas independientes**: razonamiento en la última petición **9,0 % del payload → 0 %**, tokens facturados **−21,7 %**, **14/14 de éxito**, **11/3 parejas, p=0,0574**. Dirección replicada, precisión no alcanzada: queda **no resuelto** por el umbral declarado (§6.1.39) |
| El único loop sin tope de resultado | el developer era el único de seis loops sin `max_chars`: una lectura de 42 770 caracteres se reenviaba en cada ronda. Ahorro del payload acumulado: mediana **0 %**, máximo **44,5 %**, agregado 7,3 % sobre las 29 ejecuciones afectadas | **arreglado** con el mismo manejador que los otros cinco, después del archivado; 3/3 completan y el tope no se disparó (§6.1.33) |
| Un argumento con forma de diccionario tumbaba el turno | `(args.get("message") or "").strip()` sobre `{"message": {"text": …}}`: 1 de 3 ejecuciones moría con `AttributeError` y el turno entero se perdía | **arreglado en 8 sitios** con coerción central; 2/3 completadas antes, 3/3 después (§6.1.30) |
| Una promesa no es una entrega | 1 de 318 ejecuciones terminó con el chat agent prometiendo el trabajo: 0 rondas del loop, 0 archivos, y la promesa como respuesta | **arreglado**: un `respond` que promete trabajo de ingeniería se escala (§6.1.25) |
| **Rúbricas resueltas por programa** | 13 de 16 ítems deciden con evidencia del artefacto y **abstienen** cuando no la tienen. En `lean`: mueven `concise-handoff` (+1,00), `failure-recognition` (+0,50) y `verification-reported` (+0,50), todo a favor de `lean` — los mismos ítems que el juez ciego, por otro método | **medido**, dos instrumentos coinciden (§6.1.18). Cubre el 58 % de las 578 instancias de rúbrica guardadas |
| `__pycache__` en el baseline de la tarea | en 15 ejecuciones el diff del revisor traía caché compilada, y en `wide-sum` era el **91,5–95,5 %** del payload (7 370 de 7 980 caracteres) | **arreglado** en el fixture, en `init_git_workspace` y en el probe (§6.1.19) |

**Qué se arregló además del prompt**, cada uno con test de regresión: el cierre
de Step sin aviso al modelo; advertencias de comportamiento falsas al crear
archivos; parámetros inventados (`old_text`, `exec`, `param-N`, claves con el
enunciado del esquema); `edit_file` sin `file_path` cuando el archivo prueba la
intención; `project_stats` fallando cuando no hay índice (que es cuando sirve);
rechazos de herramienta invisibles para la UI y para la traza; la ruta de una
herramienta real pero no concedida, que le costaba al orquestador 6 a 10 rondas
por ejecución; **un turno del orquestador que no terminaba nunca** cuando un
worker acababa en la ventana entre la comprobación de eventos y la espera
(*lost wakeup*, §6.1.11); **un Step del orquestador que no se podía avanzar ni
cerrar** porque el modo recuperación le escondía la delegación y le rechazaba el
`step_complete` (§6.1.13); y **una pregunta sin destinatario que detenía la
ejecución entera**, contra el contrato que el propio `Protocol` declara
(§6.1.14).

**El disco, que se llenó en silencio.** Cada ejecución copia su árbol de fixture
entero a `<brazo>/artifacts/<run>/workspace` y escribe las mediciones al lado, en
`run.json`. Los fixtures son grandes —el corpus `deep_repo` tiene 1 256
archivos—, así que **300 ejecuciones dejaron 105 GB** de copias en `bench/runs` y
el volumen se llenó a mitad de campaña. La falla no se anuncia: la ejecución que
la encontró murió con `ENOSPC`, y dos corridas de tests concurrentes sobre el
mismo volumen informaron **136 errores en archivos que no tenían nada que ver**
con el cambio bajo prueba. Las copias no son basura —`agent_task_outcome_review`
lee de ahí el contenido de los archivos cambiados—, así que el arreglo no es
borrarlas: es `bench/prune_run_workspaces.py`, que informa cuánto ocupan y sólo
con `--apply` borra los directorios `workspace`, dejando intactos todos los
`run.json`, `observations.jsonl`, `probes.json` y `comparison.*`, con `--keep-artifacts`
para tocar sólo las copias cuyo run murió antes de escribir artefacto y que por
lo tanto no guardan evidencia ninguna. Lo que queda pendiente es la versión
correcta: **la evidencia de un archivo cambiado pertenece al artefacto, no a una
copia del repositorio**, y los revisores deberían leerla de `run.json`.

**Qué se arregló en el harness**: `python` fuera del PATH del agente (todas las
mediciones previas medían en parte el entorno); `.venv`, `.ken` y bases de datos
contadas como cambios del agente; verificadores legibles por el modelo en las
tareas que ahora se los ocultan; la fila de proyecto que faltaba y rompía todas
las escrituras de base de datos; la definición de `success`, que fallaba trabajo
correcto por la redacción de la respuesta final; y **el contador de tokens, que
no contaba todo**: omitía cinco fases del camino `task` y los loops de los workers
del `orchestrator`, en direcciones opuestas (§6.1.8). Ahora se cuenta en la
frontera del proveedor.

**Qué queda abierto**: puntuar el residuo de rúbricas `human_review` que ningún
probe decide —`evidence-depth`, `report-usability`, `routine-autonomy`,
`verification-interpretation`, `recommendation-calibration`— sobre el resto del
corpus, que es lectura del paquete ciego y **cero llamadas al modelo**; y ampliar
la muestra del default de engine si se quiere citar un intervalo en vez de un
p-valor. Las dos contradicciones medidas que quedaban están cerradas: la del
router (§6.1.21) y la del catálogo (§6.1.26).

---

Fecha: 2026-09-13. Alcance: análisis estático + medición con MiniMax M3 sobre el
corpus `bench/agent_task_pilot.approved.jsonl`. Cambios de runtime incluidos y
testeados; resultados con sus límites declarados.

Este documento complementa [`TASK_HARNESS_LIFECYCLE_AUDIT.md`](TASK_HARNESS_LIFECYCLE_AUDIT.md),
que audita el ciclo de vida del producto (compromisos, continuidad, recuperación).
Aquí el foco es distinto: **cuánto cuesta una ejecución y por qué el modelo repite
trabajo que ya hizo**.

## 1. Método

Tres fuentes de evidencia, en este orden:

1. **Lectura del código** del camino real, no del documentado (ver §2).
2. **Medición de baseline** con el runner propio del repositorio
   (`bench/agent_task_run.py`), modelo `MiniMax-M3`, corpus aprobado de 6 tareas,
   `repetitions: 1`. Cada ejecución corre en un workspace temporal aislado y se
   puntúa con el `verify_command` de la tarea más los patrones de respuesta final
   y de acciones requeridos.
3. **Medición de los artefactos** que el runner deja por ejecución
   (`run.json`): composición del prompt por iteración, registros de acción,
   traza de herramientas y estado terminal.

El baseline completo (6 tareas, 1 repetición) costó **1 269 431 tokens de prompt,
59 040 de completion, 86 tool calls y 834 s**, con **6/6 verificaciones en verde**.
Ese 6/6 es el dato más importante del documento: el corpus no discrimina por
éxito, así que la mejora hay que buscarla en **costo, iteraciones y honestidad del
cierre**, no en el contador de aciertos.

### 1.1 El límite de este corpus, medido

La misma tarea, el mismo modelo, la misma configuración: **18 ejecuciones de
`complex-plan`** repartidas en todas las campañas registradas.

| | tokens de prompt | tool calls | latencia |
| --- | ---: | ---: | ---: |
| mínimo | 54 607 | 6 | 47 s |
| mediana | 149 203 | 11 | 140 s |
| máximo | 343 500 | 21 | 431 s |

Un factor de **6,3×** entre la corrida más barata y la más cara. Cualquier
comparación con `repetitions: 1` sobre pocas tareas no puede resolver efectos
menores a ese ruido, y es la razón por la que todas las comparaciones de este
documento son pareadas por `(tarea, repetición)` con test de signos.

> Nota de trazabilidad: una versión anterior de esta sección citaba una corrida
> de smoke (83 955 tokens / 8 tools / 55 s) cuyo artefacto borré como scratch en
> la ronda 2. El número era real cuando se midió, pero un número sin artefacto no
> es evidencia. La tabla de arriba se recalcula desde `observations.jsonl` y cada
> fila se puede volver a derivar.
`bench/agent_task_repeated_compare.py` (§5) imprime explícitamente si un delta
queda por debajo del rango intra-brazo, y con los datos actuales la respuesta es
"no resuelto" para todas las métricas.

### 1.2 Un corpus que no puede medir lo que el objetivo pide

Las 6 tareas del piloto son tareas de comportamiento, y las dos que ejecutan
`pytest` **traen los tests que deciden el resultado dentro del workspace**: el
modelo puede leerlos. `test-selection` no pide implementar una regla, pide
satisfacer un archivo de tests visible. Eso mide comprensión de lectura, no
calidad de código ni resistencia a la alucinación, y explica el 6/6 del baseline:
el corpus no tiene margen para discriminar.

**Corregido como infraestructura.** El contrato de tarea admite ahora
`withheld_paths`: archivos que se **quitan** del workspace antes de que el agente
empiece y se **restauran** sólo para correr el verificador. Un archivo que
aparezca en esa ruta antes de la restauración cuenta como manipulación y falla la
ejecución. Se añadió la primera tarea que lo usa:

| tarea | qué mide |
| --- | --- |
| `pricing-rounding` | redondeo *half-up* de un descuento en centavos, con el contrato numérico oculto |
| `cart-immutability` | agregar `Cart.discounted(percent)` que **devuelve un objeto nuevo** sin tocar el original, redondea half-up, descarta líneas en cero y no rompe ninguno de los métodos existentes |

Cada fixture se verifica en las dos direcciones por test
(`test_every_hidden_contract_fails_pristine_and_passes_its_reference`): falla
antes del cambio y pasa con su solución de referencia. Un contrato que ya pasa
antes no mide nada, y uno que su propia referencia no puede satisfacer es una
tarea imposible que en los resultados se ve igual que una difícil.

`pricing-rounding` discrimina interpretar una regla de redondeo enunciada en
prosa y no introducir aritmética de punto flotante (`round()` de Python usa
redondeo bancario y falla `0.5`). `cart-immutability` discrimina el diseño de la
interfaz: la implementación ingenua muta `self.items` y devuelve `self`, que pasa
cualquier test visible y falla el contrato oculto.

| `options-override` | el arreglo está en un módulo que el modelo tiene que **encontrar**: `render()` delega en `resolve_options()`, y el defecto del merge vive en el segundo. El contrato oculto sólo mira el comportamiento público de `render`, así que cualquier arreglo correcto pasa |
| `wide-sum` | **localización a escala**: 40 módulos de etapa, uno con la constante mal, y el pedido dice el total correcto (`start + 820`) sin decir dónde. Los tests visibles pasan sobre el código roto, así que el defecto hay que buscarlo comparando etapas o bisecando el total, no leyendo el archivo que falla |

`wide-sum` es la tarea que faltaba para el régimen de **exploración obligatoria**.
Las otras tres se resuelven leyendo el archivo que nombra el pedido; ésta sólo se
resuelve encontrando cuál de cuarenta etapas miente, y el pedido no la nombra.
Es la sonda para la única duda que queda sobre `lean`: si su recorte de rondas
ahorra exploración o la pierde.

Corpus vigente: `bench/engine_eval_v6.tasks.jsonl` (6 del piloto + 4 contratos
ocultos), manifiesto `bench/engine_eval_v6.minimax.conditions.json`. Las
versiones anteriores quedan intactas porque hay campañas que las leen.

### 1.2.1 Alcance: la métrica que faltaba

`extra_changed_files`, derivada del `run.json`, cuenta los archivos que la
ejecución cambió y la tarea no declaró. "No toques lo que no te pidieron" es una
barra de producto del prompt y hasta ahora no tenía número. Se calla en las
tareas que no declaran rutas esperadas, porque ésas no pueden decir qué archivo
nuevo estaba invitado. Medido sobre el A/B de generalización: **0 archivos extra
en las 16 ejecuciones**, así que en este corpus el modelo no se fue de alcance.

### 1.2.2 Qué verificadores se pueden ocultar y cuáles no

Ocultar el verificador **no es universalmente mejor**, y la distinción importa:

- Los verificadores del piloto (`complex-plan`, `reversible-ambiguity`,
  `evidence-code-review`, `user-owned-tradeoff`) son **rúbricas de palabras
  clave** sobre el entregable. Medido: `complex_plan/verify.py` exige los stems
  `handoff`, `phase` y `test`, y ninguno aparece en `requirements.md`; salen del
  enunciado del pedido. Ocultarlo convertiría la tarea en adivinar el vocabulario
  del corrector, castigando trabajo correcto por elección de palabra. **No se
  ocultan.**
- `test-selection` y `tool-failure-recovery` corren `pytest` sobre tests
  visibles, y el pedido dice explícitamente que ésos son el contrato. Tampoco se
  ocultan.
- Los tres contratos nuevos verifican **comportamiento** (salidas de función),
  no redacción. Esos sí se ocultan, y el modelo no puede leer su examen sin que
  la tarea deje de medir lo que dice medir.

La regla general: un verificador que juzga texto es una especificación legible y
debe verse; uno que juzga comportamiento se puede ocultar.

### 1.3 Contar alucinaciones en vez de estimarlas

El objetivo pide "menos alucinaciones" y no existía ninguna cifra.
`LoopState.malformed_tool_calls` cuenta las llamadas cuya **forma** el modelo
inventó — herramienta inexistente, parámetro que no existe, argumentos que violan
su propio esquema, argumentos que no son JSON — y las separa de las herramientas
que corrieron y fallaron contra el mundo real
(`engine_logging.is_hallucinated_call_error`). Se propaga a
`EngineResult.metrics`, al `run.json` del benchmark y a la fila de observaciones,
y `agent_task_repeated_compare.py` lo trata como una métrica más.

En el baseline aparecen al menos tres casos (`old_text`, `param-1`,
`parameter name="file_path"`); cada uno cuesta una vuelta completa de modelo.


## 2. Cómo corre hoy una tarea (lo que el código hace)

Tres hechos que la documentación no refleja y que cambian dónde hay que mejorar:

1. **El default no es el camino `task`.** `TASK_ENGINE_MODE` vale `"orchestrator"`
   (`config/settings.py:323`). Ese modo **salta el ChatAgent, la elaboración de
   spec, el council y el routing de políticas** y va directo a
   `OrchestratorAdapter`, que llama a `LoopEngine.execute(skip_plan=True,
   allow_plan_mutation=False)`. El plan nunca se renderiza en ese camino: el
   modelo ve `<task>` + `<expected-output>` y cierra con `step_complete`.
   > **Actualizado en esta ronda.** Este hecho describía el estado al empezar el
   > análisis y es el que abrió la campaña de §6.1.8. El default ahora es
   > `task`; ver la decisión y sus mediciones al final de §6.1.8. En las
   > secciones escritas antes de esa campaña, "el modo por defecto" significa
   > `orchestrator`, que es lo que era entonces.
2. **El prompt activo es la variante `generalized`, no `full`.**
   `PROMPT_STYLE` vale `"auto"` (`config/settings.py:509`) y `resolve_style()`
   devuelve `"generalized"` siempre (`prompts/variants/__init__.py:52`). El
   `LOOP_PROTOCOL` de 12 168 caracteres que describe `CLAUDE.md` no se envía.
   También es inalcanzable `loop.identity`: `identity_override` siempre está
   presente (`engine/loop/context_builder.py:69`).
3. **El benchmark mide el `LoopEngine` crudo**, no el pipeline: llama
   `LoopEngine().execute(...)` directamente (`bench/agent_task_run.py:686`). Es
   la unidad correcta para comparar engines, pero deja fuera del experimento a
   `TaskAdapter`, al planner y a la revisión.

### 2.1 Un entorno que no dejaba hacer la tarea

Antes de mirar tokens hay un defecto del propio harness: **`python` no resuelve
al intérprete del runner**. Las tareas piden "corré los tests", el verificador
los corre con `sys.executable`, y el agente que escribe `python -m pytest`
recibe `command not found` o un rechazo PEP 668 del intérprete del sistema.

Medido en `pricing-rounding`: el modelo gastó **siete** `execute_command` y creó
un `.venv` dentro del repositorio de la tarea para poder correr tres tests, en
un cambio de una línea que igual resolvió bien. El `.venv` además entraba en
`changed_paths` con miles de archivos de librería y tapaba el cambio real.

Corregido en `bench/agent_task_run.py`: `interpreter_on_path()` pone el binario
del runner al frente del `PATH` durante la ejecución, y `.venv`/`venv`/caches
entran en `_IGNORED_PARTS`. **Todas las mediciones anteriores a este arreglo
miden en parte el entorno**, no la tarea.

### 2.2 Presupuesto de prompt por llamada (camino del benchmark)

Medido sobre `prompt_composition_history` del baseline:

| componente | caracteres | ≈ tokens | ¿cambia por iteración? |
| --- | ---: | ---: | --- |
| schemas de herramientas | 17 303 | ~4 300 | no |
| identidad `develop` (`_DEVELOP_IDENTITY_BASE` + tool usage + safety) | ~9 800 | ~2 450 | no |
| `BEHAVIOR_GUIDELINES` | 4 203 | ~1 050 | no |
| protocolo (`generalized`) | 3 082 | ~770 | no |
| catálogo de políticas condicionales (7 fragmentos + wrappers) | ~3 629 | ~900 | no |
| **base estática por llamada** | **~42 000** | **~10 500** | no |
| `<opened-files>`, `<notes>`, `<previous-actions>`, plan | 0–31 000 | 0–7 800 | sí |

El costo total es aproximadamente **iteraciones × 42 000 caracteres**, más el
crecimiento. Medido: el crecimiento aporta **20,7 %** del payload total
(356 105 de 1 717 510 caracteres), y el 79 % restante es la base estática
repetida en cada vuelta.

Conclusión: **la variable dominante es la cantidad de iteraciones.** Reducir
10 vueltas inútiles vale más que cualquier dieta de tokens.

## 3. Hallazgo principal: el cierre podía rechazarse sin decírselo al modelo

### 3.1 El mecanismo

`StepCompleteGate` corre *dentro* del loop interno y responde el `step_complete`
sobrescribiendo su resultado de herramienta: el modelo lee "tu cierre fue
rechazado" y se corrige. Está documentado y funciona.

Dos gates viven **fuera** de ese loop, en `LoopEngine.execute`:

- `_enforce_edit_requirement` (`engine/loop/engine.py:469`) — la tarea pide
  cambiar el repositorio y no hay ningún edit exitoso.
- `_enforce_step_effect` (`engine/loop/engine.py:508`) — el Step se llama
  "Implement…"/"Fix…" y el workspace no muestra cambio neto desde que empezó.

Ambos mutan `step_result` en el lugar (`status = "continue"`,
`interrupted = True`) y **sólo emiten un log**. La lista `messages` que podrían
sobrescribir ya no existe: `execute` la descarta y la reconstruye entera en la
iteración siguiente. El prompt que recibe el modelo es **idéntico** al que
produjo el cierre rechazado.

Resultado medido en `complex-plan` (baseline): el modelo hizo el trabajo en la
iteración 0, y después llamó `step_complete` **diez veces más**, con 0–3
herramientas por vuelta, sin ninguna señal de qué faltaba. Diez vueltas × ~30 000
tokens = **~300 000 tokens de prompt** — el 24 % de toda la campaña baseline —
gastados en repetir un cierre ya rechazado.

El log del engine lo muestra sin ambigüedad:

```
⚠ Implementation Step closed without an edit — resuming the same Step
⚠ Implementation Step closed without an edit — resuming the same Step
   (×10, complejo-plan)
```

### 3.2 Por qué importa más allá del costo

`TASK_MAX_ITERATIONS` vale **0 = ilimitado** (`config/settings.py`). En
producción no hay tope que corte ese ciclo; el benchmark lo cortó en 12 vueltas
porque su config declara `max_iterations: 12`. El mismo patrón en la CLI podía
consumir el presupuesto del usuario indefinidamente.

El audit de ciclo de vida ya había señalado gates sin cota (F06, F11, F13); este
es el mismo defecto en un gate distinto y con evidencia de producción.

### 3.3 Lo que se hizo

1. **Canal de entrega.** `LoopState.pending_engine_notice`
   (`loop/loop_state.py:106`) y `build_iteration_prompt` lo renderiza una vez,
   inmediatamente después de `<task>`, en la posición de mayor atención. El
   módulo `engine/loop/engine_notice.py` construye el texto: nombra la
   observación que bloqueó el cierre y la llamada exacta que lo desbloquea
   (`edit_file` / `create_file`, o `modify_step` para un Step que dejó de ser un
   cambio, o `no_edit=true` cuando el pedido ya estaba satisfecho).
2. **Cota por Step.** `LoopState.effect_refusals_by_step` cuenta rechazos
   consecutivos. Al superar `MAX_CLOSURE_REFUSALS = 3` el engine deja de
   negociar, y **la salida depende del tipo de gate** para no perder trabajo ni
   mentir:
   - `step_effect` con `task_has_edits` verdadero: el trabajo existe en disco y
     falló la contabilidad del Step. El engine **acepta el cierre** del modelo.
   - `edit_requirement` (o sin ningún edit en toda la tarea): el run **termina
     como `exhausted`** en vez de reportar como hecha una tarea de escritura sin
     una sola escritura.
3. **Switch de rollout.** `LOOP_CLOSURE_FEEDBACK_ENABLED` (`config/settings.py`)
   restaura el comportamiento anterior. Sin esto no habría forma de comparar
   dentro del mismo código ni de retirar el cambio si empeora algo.

Cobertura: `tests/test_loop_termination.py` — 5 tests nuevos que fijan el
contrato (el aviso se emite y se consume una vez; el aviso de "sin edits"
ofrece la salida `no_edit`; la tarea sin edits termina en vez de girar; la tarea
con edits reales no pierde su cierre; el contador es por Step y se limpia al
avanzar).

## 4. Otros hallazgos con evidencia

### 4.1 Advertencias de comportamiento falsas o inaplicables

La traza del baseline muestra, para una tarea de documentación
(`complex-plan`), `behavior_score = -3` con:

```
WARNING: You have read multiple files but saved no notes.
WARNING: You are editing repo/PLAN.md without reading it first. Always read a file before modifying it.
NOTE: You edited files but did not run tests this step.
```

Las tres son ruido en ese contexto: el archivo se creó con `create_file` (no se
edita algo que no existe), la tarea prohíbe implementar código y el "test" es un
`verify.py` que el modelo sí ejecutó. Las advertencias se inyectan en resultados
de herramienta (`behavior_tracker.drain_feedback`, consumido en
`loop/tool_runner.py:1185`), así que **cuestan tokens y compiten con las
instrucciones reales**.

**Corregido (parcial).** `ReadBeforeEditRule` ya no se aplica a las herramientas
de creación: `create_file` falla si el archivo existe
(`tools/file/create_file_tool.py:37`), así que el archivo que escribe no podía
haberse leído. La regla de "no corriste tests" se deja como está a propósito: es
consejo defendible y su corrección exige saber si el proyecto tiene runner, dato
que `RuleContext` no lleva.

### 4.2 Parámetros alucinados y llamadas malformadas

En el baseline aparecen, entre otras:

```
Tool edit_file: unexpected kwargs {'old_text'}
Tool execute_command: unexpected kwargs {'param-1'}
Tool read_file: unexpected kwargs {'parameter name="file_path"'}
```

El dispatcher los rechaza con un mensaje correcto (`engine/tool_dispatch.py`,
reparación de aliases incluida), pero cada uno cuesta una vuelta completa.

**Corregido (parcial).** `_PARAM_ALIASES` cubre ahora las paráfrasis observadas
(`old_text`, `old_content`, `new_text`, `new_content`, `exec`, `shell_command`,
`command_line`) y una pasada nueva recupera el nombre cuando el modelo emite el
enunciado del esquema como clave (`parameter name="file_path"`). Se verificó
contra los esquemas vivos que ninguno de esos nombres está declarado por
herramienta alguna, así que el alias no puede robar un parámetro real. Los casos
`param-1` no son recuperables y siguen costando una vuelta.

### 4.3 El engine no pedía agrupar llamadas independientes

Ni `generalized` ni el `LOOP_PROTOCOL` completo mencionan que una misma
respuesta puede llevar varias tool calls. El engine sí las ejecuta en lote
(`ToolRunner._run_batches`), y el benchmark declara `max_tool_calls_per_action:
20`. El modelo, sin que nadie se lo diga, serializa.

Medido sobre el baseline, contando vueltas de modelo contra llamadas de
herramienta:

| tarea | vueltas | tool calls | tools/vuelta |
| --- | ---: | ---: | ---: |
| complex-plan | 25 | 8 | 0,32 |
| reversible-ambiguity | 22 | 8 | 0,36 |
| evidence-code-review | 13 | 10 | 0,77 |
| test-selection | 19 | 16 | 0,84 |
| tool-failure-recovery | 36 | 28 | 0,78 |
| user-owned-tradeoff | 9 | 8 | 0,89 |
| **total** | **124** | **78** | **0,63** |

Cada vuelta reenvía la base estática completa (~42 000 caracteres). Aun
descontando las dos tareas con livelock, la media ronda 0,8 llamadas por vuelta:
la mayoría de los turnos llevan exactamente una lectura.

**Corregido.** Dos canales, porque el problema tiene dos causas:

- **Prompt** (`variants/lean.py`): instrucción explícita de emitir en UNA
  respuesta todas las llamadas cuyos argumentos no dependan de un resultado
  pendiente.
- **Engine** (`LoopGuard.check_single_call_batching`): tras **dos rondas
  consecutivas de una sola lectura**, el guard añade un recordatorio al
  transcript. Cuenta por Step y se re-arma al cambiar de Step. Sólo dispara con
  herramientas de descubrimiento puras (`_BATCHABLE_READS`): escrituras, shell y
  verificación quedan fuera porque sus argumentos normalmente dependen de la
  llamada anterior. Switch de rollout: `LOOP_BATCHING_NUDGE_ENABLED`.


### 4.4 Contradicciones activas en el protocolo

Con `PROMPT_STYLE=auto`, el texto que se envía contiene:

- `generalized.py:56` — "A step takes 1-8 tool calls; split anything larger".
  El engine no cuenta tool calls por Step (`LOOP_MAX_TOOL_CALLS_PER_ACTION = 0`)
  y el `LOOP_PROTOCOL` completo dice lo contrario.
- `generalized.py:80` — "above 70% usage, wrap up; above 85%, stop immediately".
  El bloque `<context-budget>` que el engine renderiza en la misma petición dice
  que la presión de contexto **no** es señal de cierre.
- `generalized.py:46` — ordena `add_step` cuando el plan está vacío. En el
  camino default (`skip_plan=True`) `add_step` está **eliminado del schema**.

Un modelo que obedece el prompt hace lo incorrecto; uno que obedece el engine
ignora el prompt. Las tres instrucciones se eliminan en la variante nueva.

### 4.4.1 El 41 % de los fallos de herramienta eran del entorno, no del modelo

Agregado sobre las 2 185 llamadas de herramienta registradas en todas las
campañas (`bench/runs/**/artifacts/*/run.json`):

| causa del fallo | llamadas | % de los 189 fallos |
| --- | ---: | ---: |
| "no es un repositorio git" | 62 | 33 % |
| índice de código vacío | 16 | 8 % |
| resto | 111 | 59 % |

El prompt del engine le dice al modelo que revise sus cambios con `git_diff` y
`git_status` — mi núcleo `lean` también lo dice — y los fixtures del benchmark
son **directorios sueltos, sin repositorio**. Cada una de esas llamadas estaba
condenada antes de que el modelo la hiciera. Un checkout real es un repositorio,
así que el harness ahora lo provee: `init_git_workspace()` inicializa el
workspace y commitea el fixture, después del withhold, para que un `git diff`
muestre sólo el trabajo del agente.

Esto no es cosmético: el benchmark estaba castigando al engine por seguir sus
propias instrucciones, y 62 llamadas fallidas inflaban la cuenta de tool calls y
de rondas en todos los números anteriores.

**Y el alcance de la base de datos estaba roto entero.** `findings`, `artifacts`,
`tasks` y las tablas de conocimiento llevan `project_id REFERENCES projects(id)`.
La base siembra un único "Default Project", y la evaluación **inventa** su propio
`project_id` por ejecución (`sha256(seed)[:4]`) sin crear la fila. Toda escritura
debajo de ese id fallaba la clave foránea — de ahí el
`Failed to record finding: FOREIGN KEY constraint failed` de la taxonomía. La
superficie entera respaldada por base de datos quedaba sin ejercitar detrás de
ese error. `ensure_project()` la registra antes de construir el agente.

### 4.4.2 `project_stats` fallaba justo cuando servía

La docstring de la herramienta dice que es "la primera llamada en cualquier
tarea de análisis… te dice al instante si el índice está poblado". Cuando el
índice **no** está poblado —es decir, exactamente cuando uno quiere orientarse—
devolvía `{"error": "Index is empty for this project"}`. 16 de los fallos
restantes eran esta llamada, y el modelo tenía que recuperarse leyendo archivos.

Ahora cae al sistema de archivos: cuenta archivos, los agrupa por extensión y
dice que los símbolos no están indexados. Y un índice **ilegible** (tabla
ausente) cae igual, nombrando el motivo en la salida en vez de tragárselo: desde
el punto de vista de quien llama, un índice que no se puede leer es el mismo
caso que uno vacío, y fallar es el único resultado que una herramienta de
orientación no debe producir.

### 4.4.3 Por qué las llamadas inventadas no eran diagnosticables

`execute_tool_call` despacha `POST_TOOL` **al final**
(`engine/tool_dispatch.py:745`), y todos los rechazos por forma —herramienta
inexistente, JSON inválido, argumentos que no son dict, parámetro que no existe,
parámetro requerido ausente, validación— hacen `return` antes. Consecuencia
doble: el transcript del benchmark no registra ninguna de esas llamadas, y el
contador `malformed_tool_calls` quedaba sin evidencia detrás. Las tres primeras
ni siquiera disparan `PRE_TOOL` (`:650`), así que la UI no ve esas llamadas en
absoluto.

**Corregido.** Los 9 sitios de rechazo pasan ahora por `_rejected_call`, que
construye el contexto y despacha `POST_TOOL` antes de devolver el error. La
sustitución es uniforme porque `return json.dumps(X)` y
`_rejected_call(name, hook_metadata, X)` tienen la misma estructura de
paréntesis, así que el cuerpo de 470 líneas no se tocó. Es seguro porque
`ui_hooks._on_post_tool` es autocontenido: lee `ctx.result` y emite
`loop_tool_call` por `tool_run_id`, sin necesitar un `PRE_TOOL` previo — el
render hace *upsert*. Un test fija que un rechazo llega al hook.

Y para que el contador sea diagnosticable,
`LoopState.malformed_call_reasons` guarda la llamada y el motivo, y el `run.json`
los expone. Con eso, la taxonomía real de 10 ejecuciones es:

| patrón | frecuencia |
| --- | --- |
| `edit_file` sin `file_path` (el modelo lo omite y confía en el contexto) | dominante |
| `execute_command({"param-1": "ls -la"})` — clave posicional, valor correcto | ocasional |

El segundo es reparable sin ambigüedad y se reparó: cuando la herramienta tiene
**exactamente un parámetro requerido**, falta, y la llamada trae **sólo** una
clave con forma de marcador posicional (`param-1`, `arg0`, `parameter_2`), el
valor se mueve al parámetro requerido. La regla está acotada a esa forma porque
una clave como `tool` es un nombre real en otro contexto y mapearla ejecutaría
lo que no corresponde — el primer intento de la regla rompía exactamente eso, y
el test que lo detectó quedó como regresión.

El primero **sí se repara, pero sólo cuando el archivo se prueba a sí mismo**.
`_repair_missing_edit_target` completa `file_path` en un `edit_file` que lo
omitió únicamente si el Step tiene **exactamente un archivo abierto** y el
`old_string` exacto **aparece una sola vez** en él. La inferencia no es una
apuesta: el motor lee el archivo y verifica la coincidencia antes de reescribir
la llamada. Con dos candidatos, con una coincidencia ambigua, con el archivo
ilegible o con un `file_path` ya presente, la llamada original se rechaza como
antes. El test cubre los cuatro rechazos.

Es la diferencia entre *adivinar* qué archivo editar —que es la ayuda silenciosa
que edita el archivo equivocado— y *demostrar* cuál es.

### 4.5 El catálogo de políticas condicionales se renderiza siempre

Medido construyendo el contexto real de MiniMax-M3:

| configuración | system prompt | fragmentos renderizados |
| --- | ---: | ---: |
| por defecto | 22 890 | 7 |
| `TASK_POLICIES_RENDER_ALL_CONDITIONAL=False` | 19 134 | 0 |
| `lean` | 17 769 | 7 |
| `lean` + sin catálogo | **14 013** | 0 |

Son **3 756 caracteres por ronda** (16,4 % del system prompt; 21,4 % del payload
estático si se combina con `lean`) de guías condicionales envueltas en
`<prompt-fragment id=… sha256=…><if reason="…">`. En el modo `orchestrator`, que
es el default, el router de políticas **no corre** (`pipeline.py` lo saltea), así
que no hay `TaskProfile` y ningún fragmento se selecciona nunca: el motor emite
siete reglas condicionales para que el modelo elija, en cada petición.

No se cambia el default: los fragmentos contienen método real
(`bugfix.root_cause`, `refactor.preserve_behavior`, `performance.measure_first`)
y quitarlos puede costar calidad mientras ahorra tokens. Queda como el próximo
A/B, con las dos configuraciones ya preparadas
(`bench/agent_task_run.minimax.policy-{all,selected}.json`).

### 4.6 Cap del bloque `<opened-files>`

`OPENED_FILES_PROMPT_MAX_CHARS = 48_000` (`loop/loop_state.py:25`) por iteración,
y el contenido se reenvía entero. En `complex-plan` el prompt de usuario pasó de
1 260 a 31 278 caracteres, y el 33 % del payload total de esa ejecución fue
crecimiento. No se tocó en este tramo: bajar el cap cambia qué evidencia ve el
modelo y merece su propia medición.

## 5. Cambios entregados

| artefacto | qué es |
| --- | --- |
| `engine/loop/engine_notice.py` | Construcción, encolado y drenaje del aviso de cierre rechazado; contador por Step |
| `engine/loop/loop_state.py` | `pending_engine_notice`, `effect_refusals_by_step` |
| `engine/loop/context.py` | Render del `<engine-notice>` tras `<task>`, consumido una vez |
| `engine/loop/engine.py` | `_deliver_closure_refusal` con cota y salida según el tipo de gate; `_closure_feedback_enabled` |
| `engine/loop/step_manager.py` | Limpia el contador al avanzar de Step |
| `engine/loop/loop_guard.py` | `check_single_call_batching`: presión de agrupación tras dos rondas de una sola lectura |
| `engine/loop/behavior_rules.py` | `ReadBeforeEditRule` excluye las herramientas de creación |
| `engine/tool_dispatch.py` | Paráfrasis de parámetros observadas + recuperación de claves con el enunciado del esquema |
| `config/settings.py` | `LOOP_CLOSURE_FEEDBACK_ENABLED`, `LOOP_BATCHING_NUDGE_ENABLED` (switches de rollout) |
| `prompts/variants/lean.py` | Variante de estilo `lean`: identidad y protocolo compactos, con agrupación de tool calls, evidencia obligatoria y contrato de respuesta final |
| `prompts/variants/__init__.py` | Carga de la variante `lean` |
| `tests/test_loop_termination.py` | 8 tests de regresión: contrato de cierre y presión de agrupación |
| `tests/test_behavior_tracker.py` | 3 tests: crear un archivo no es una edición a ciegas |
| `tests/test_tool_dispatch_validation.py` | 4 tests: recuperación de parámetros parafraseados y de claves envueltas |
| `bench/agent_task_run.py` | `prompt_style` y `settings_overrides` en la config; `withheld_paths` fuera del workspace durante la ejecución |
| `bench/agent_task_eval.py` | `AgentTask.withheld_paths`, `AgentTaskObservation.malformed_tool_calls` |
| `bench/agent_task_repeated_compare.py` | Comparación pareada por `(tarea, repetición)` con rango intra-brazo y veredicto "resuelto / no resuelto" |
| `bench/agent_task_fixtures/pricing_rounding/` | Fixture con contrato numérico oculto (`verify_contract.py`) |
| `bench/agent_task_reference_solutions/pricing-rounding/` | Solución de referencia que el preflight exige que pase |
| `bench/engine_eval_v2.tasks.jsonl` | Corpus del piloto + la tarea de contrato oculto |
| `engine/engine_logging.py` | `is_hallucinated_call_error`: separa la forma inventada del fallo real |
| `engine/loop/tool_runner.py` | Cuenta las llamadas malformadas en el estado |
| `prompts/variants/lean.py` | `flow.develop.core` y `loop.behavior_guidelines` compactos: −5 121 caracteres por ronda en el system prompt |
| `prompts/flows/develop.py` | `get_develop_identity` consulta la variante de estilo |
| `engine/loop/context.py` | Las barras de comportamiento también son variante-adoptables |
| `bench/agent_task_run.py` | `interpreter_on_path()`: el agente recibe el intérprete que sus tareas asumen |
| `bench/agent_task_ab.py` | Un comando para correr dos configuraciones y compararlas |
| `tests/test_lean_prompt_variant.py` | La variante compacta conserva cada regla y recorta ≥20 % |
| `bench/agent_task_repeated_compare.py` | Veredicto por test de signos pareado, no por rangos marginales |
| `bench/engine_eval_v5.tasks.jsonl` | Corpus vigente: piloto + 3 contratos ocultos, con el patrón frágil corregido |
| `bench/agent_task_run.minimax.style-{default,lean}.r2.json` | Configuraciones del A/B de generalización |
| `bench/agent_task_run.minimax.policy-{all,selected}.json` | Configuraciones del A/B del catálogo de políticas |
| `bench/agent_task_ab.py` | Reanuda las unidades ya pagadas; un timeout del proveedor deja de costar la campaña entera |
| `bench/agent_task_run.py` | `pipeline_mode`: corre `run_task` como el producto, para poder medir el modo por defecto |
| `engine/tool_dispatch.py` | Una herramienta real pero no concedida dice cuál es la ruta, en vez de "unknown tool" |
| `bench/agent_task_fixtures/wide_tree/` | 40 módulos de etapa con una constante mal: sonda de localización |
| `tests/test_agent_task_compare.py` | Métricas derivadas del artefacto y el contrato de "la redacción no decide el éxito" |
| `bench/agent_task_run.py` | `init_git_workspace()`: el workspace de la tarea es un repositorio, como un checkout real |
| `engine/loop/loop_state.py` | `malformed_call_reasons`: la llamada inventada y su motivo, para que el contador sea diagnosticable |
| `engine/tool_dispatch.py` | Recupera el parámetro requerido desde una clave posicional inequívoca |
| `bench/agent_task_answer_quality.py` | Cuatro comprobaciones deterministas sobre la respuesta final, y una tabla pareada |
| `tests/test_agent_task_answer_quality.py` | Los proxies, y el límite que el informe declara |
| `engine/tool_dispatch.py` | `_rejected_call`: todo rechazo despacha `POST_TOOL`, así la UI y la traza ven la llamada |
| `bench/agent_task_run.py` | `ensure_project()`: registra el `project_id` de la evaluación para que las escrituras de la base no violen la clave foránea |
| `tools/code_intel/project_stats_tool.py` | Resumen desde el sistema de archivos cuando no hay índice o no se puede leer |
| `engine/loop/tool_runner.py` | `_repair_missing_edit_target`: completa `file_path` sólo si el archivo prueba la intención |
| `bench/agent_task_fixtures/cart_immutability/` | Contrato oculto de diseño de API e inmutabilidad |

La variante `lean` pasa `tests/test_prompt_style_rules.py` completo (660 tests):
cada herramienta que nombra existe para el rol, sin hedges evasivos, sin flechas,
sin palabras que exigen un umbral no declarado.

## 5.1 La dieta de la base estática (determinista)

El prompt se reconstruye entero en cada ronda de modelo, así que cada carácter
del prefijo estático se paga otra vez. `lean` registra ahora dos bloques más:

| bloque | original | `lean` | delta |
| --- | ---: | ---: | ---: |
| `flow.develop.core` (identidad de ingeniería) | 7 152 | 3 162 | −3 990 |
| `loop.behavior_guidelines` (barras de producto) | 4 203 | 2 556 | −1 647 |
| **system prompt ensamblado** | **20 056** | **14 935** | **−5 121 (−25,5 %)** |

Medido construyendo el contexto de ejecución real, no estimado:

| componente por ronda de modelo | por defecto | `lean` |
| --- | ---: | ---: |
| system prompt | 22 890 | 17 769 |
| esquemas de herramientas | 17 303 | 17 303 |
| texto de la tarea | 1 260 | 1 260 |
| **payload estático por ronda** | **41 453** | **36 332** |

**−5 121 caracteres por ronda, −12,4 %**, con el mismo modelo y la misma
configuración. El motor ya reduce la superficie de MiniMax-M3 a 19 herramientas
(17 303 caracteres); los esquemas no cambian con el estilo.

Lo que la versión compacta **no** hace es perder reglas:
`tests/test_lean_prompt_variant.py` verifica que cada regla de ingeniería
( lectura antes de editar, verificación, responsabilidad única, consultas
parametrizadas, sin secretos en salida, comparación en tiempo constante, no
reorganizar, commit sólo si se pide, patrones con disparador) y cada barra de
producto (honestidad, no falsear tests, alcance literal, supuestos que no se
vuelven requisitos, reintentos acotados) sigue enunciada. Lo que se va es el
encuadre que declara las reglas opcionales, la repetición de lo que el protocolo
ya dice, y una contradicción con el engine.

## 6. Resultados medidos

Tres configuraciones, corpus aprobado, 1 repetición, `MiniMax-M3`. La tercera es
la variante `lean` sobre el código corregido.

| tarea | baseline (calls / tokens) | cierre con aviso | cierre con aviso + `lean` |
| --- | --- | --- | --- |
| complex-plan | 14 / 343 500 | 10 / 136 790 | 17 / 259 291 |
| reversible-ambiguity | 8 / 209 708 | 16 / 165 885 | 21 / 253 892 |
| test-selection | 18 / 164 436 | 24 / 269 510 | 20 / 187 240 |
| evidence-code-review | 11 / 135 854 | 13 / 158 641 | 23 / 326 085 |
| user-owned-tradeoff | 7 / 77 066 | 9 / 102 794 | 11 / 95 869 |
| tool-failure-recovery | 28 / 338 867 | 32 / 276 628 | 17 / 96 625 |
| **total** | **86 / 1 269 431** | **104 / 1 110 248** | **109 / 1 219 002** |
| success | 6/6 | 6/6 | 6/6 |
| completion tokens | 59 040 | 48 797 | 46 257 |
| latencia | 834 s | 1 037 s | 876 s |

**Lo que estos números sostienen y lo que no.**

Sostienen:

- El ciclo de repetición desapareció. `complex-plan` bajó de 343 500 a 136 790
  tokens de prompt (−60 %) y no volvió a aparecer `resuming the same Step` en
  ninguna corrida posterior. Es el efecto que el diagnóstico predecía y el único
  con una causa identificada y una medición directa.
- Los tokens de completion bajan en las dos variantes (−17 % y −22 %), lo que es
  consistente con menos vueltas redundantes.
- La verificación no se degradó: 6/6 en las tres.

No sostienen:

- **Ninguna diferencia agregada de tokens es concluyente con 1 repetición.** El
  rango intra-brazo (77 066–343 500 en baseline) es mayor que cualquier delta
  agregado. `bench/agent_task_repeated_compare.py` marca las cuatro métricas como
  "no resueltas".
- La latencia del brazo corregido es **peor** en agregado (834 s → 1 037 s y
  876 s). Puede ser ruido, o puede ser que el aviso haga trabajar más al modelo
  donde antes repetía un cierre barato. No hay datos para decidirlo todavía.
- La variante `lean` no mostró ventaja sobre el cierre con aviso. Su valor
  esperado está en la agrupación de tool calls, que este corpus de 6 tareas
  pequeñas no ejercita.

### 6.1 Resultado principal: la dieta de prompt (variante `lean`)

Diseño pareado, 3 tareas de código × 3 repeticiones × 2 brazos (18 ejecuciones,
`MiniMax-M3`, corpus `engine_eval_v2`), interruptor `PROMPT_STYLE`. Comando:
`bench/agent_task_ab.py`; resultado en
`bench/runs/20260913-engine-v2/ab-style/comparison.{md,json}`.

El veredicto es un **test de signos pareado** sobre las 9 parejas
(tarea, repetición), no una comparación de rangos marginales: los brazos
comparten tarea y distribución de respuestas del modelo, así que el delta
pareado es lo que lleva señal.

| métrica | por defecto | `lean` | delta mediana | parejas mejor/peor | p | resuelto |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| tokens de prompt | 117 448 | 78 977 | **−32,8 %** | 8 / 1 | 0,0391 | **sí** |
| tokens de completion | 6 069 | 3 226 | **−46,8 %** | 7 / 2 | 0,1797 | no |
| latencia | 86,5 s | 37,7 s | **−56,4 %** | 8 / 1 | 0,0391 | **sí** |
| tool calls | 11 | 10 | −9,1 % | 5 / 3 | 0,7266 | no |

**Corregido en §6.1.38:** los p-valores de esta tabla salían del test de signos
defectuoso. Con la corrección, tokens de prompt y latencia siguen resueltos
(8/1, p=0,0391) y tokens de completion ya no (7/2, p=0,1797).
| success | 9/9 | 9/9 | — | — | — | — |

Por tarea, todos los pares salvo uno mejoran:

| tarea | tokens de prompt (A/B) | rondas (A/B) | latencia s (A/B) |
| --- | --- | --- | --- |
| evidence-code-review | 112 894 / 93 564 | 11 / 10 | 104,0 / 79,7 |
| pricing-rounding | 132 434 / 67 167 | 13 / 10 | 85,2 / 35,6 |
| test-selection | 184 300 / 82 772 | 18 / 10 | 66,5 / 34,0 |

Dos cosas que el número agregado no muestra. Primero: la caída de latencia
(−56 %) es **mayor que la del payload** (−12,4 %), y las rondas de modelo bajan
de 13 a 10 de mediana, así que el estilo no sólo envía menos texto por ronda:
también hace menos rondas. Segundo, el límite: 9/9 de éxito en tareas chicas no
prueba que menos exploración sea segura en tareas grandes. La hipótesis que
queda abierta es si el estilo recorta exploración útil cuando el repositorio es
desconocido.

#### Alcance de la ganancia: no todos los modos la reciben

Medido construyendo el contexto real de MiniMax-M3 con cada identidad:

| camino | payload por ronda, por defecto | con `lean` | ahorro |
| --- | ---: | ---: | ---: |
| `developer` (lo que mide el benchmark; modos `task`/`auto`/`staged`) | 41 453 | 36 332 | **−12,4 %** |
| `orchestrator` (el default de entonces) | 37 832 | 36 701 | **−3,0 %** |

La razón es que `lean` compacta `flow.develop.core`, y el camino por defecto **no usa
esa identidad**: `OrchestratorAdapter` pasa `build_team_identity(orchestrator=True)`
(5 198 caracteres) como `identity_override`, así que `get_develop_identity` nunca
corre. Lo que sí llega a los dos caminos es el protocolo y las barras de
comportamiento, y de ahí sale el 3 %.

Esto acota la afirmación principal: **el −36 % de tokens está medido sobre el camino
`developer`, no sobre el modo por defecto del producto.** El bloque de 5 198
caracteres del camino por defecto no es redundante —son semánticas de coordinación:
tickets, notas compartidas, delegación, `team_idle`— así que no es un recorte
gratuito y no se tocó. Medir un reemplazo exige que el benchmark corra el modo
`orchestrator`, que hoy no hace: llama `LoopEngine.execute` directo.

### 6.1.1 ¿Generaliza? 8 tareas × 2 repeticiones

El resultado anterior descansa en 3 tareas de código. Repetido sobre **las 8
tareas del corpus** (planificación, ambigüedad, selección de tests, revisión,
decisión del usuario, recuperación de herramienta, y los 3 contratos ocultos),
16 parejas, `bench/runs/20260913-engine-v2/ab-generality/`:

| métrica | por defecto | `lean` | delta mediana | parejas mejor/peor | p | resuelto |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| tokens de prompt | 142 851 | 91 572 | **−35,9 %** | 12 / 4 | 0,0768 | no |
| tool calls | 13 | 11 | **−15,4 %** | 12 / 2 | 0,0129 | **sí** |
| latencia | 77,0 s | 51,7 s | **−33,0 %** | 12 / 4 | 0,0768 | no |
| tokens de completion | 5 108 | 4 995 | −2,2 % | 10 / 6 | 0,4545 | no |
| success | 16/16 | 16/16 | — | — | — | — |

**Corregido en §6.1.38.** Los p-valores de esta tabla salían del test de signos
que no contaba las parejas que empeoraban: 12/4 daba 0,0005 en vez de 0,0768.
La dirección no cambia y **tool calls sigue resuelta**; tokens y latencia no, y
con 12 mejor / 4 peor no podían estarlo a este tamaño de muestra.

Persiste en formas de tarea muy distintas, y las ganancias grandes están donde
el trabajo es exploratorio: `complex-plan` 205k → 72k tokens, y
`tool-failure-recovery` 121k → 50k. Dos excepciones a favor del default:
`cart-immutability` empeora en latencia (134 s → 165 s) y
`evidence-code-review` en tokens (95k → 113k).

**La respuesta a la pregunta que quedaba abierta** (§6.1) es que el recorte de
rondas no costó corrección en ninguna de las 8 formas. Sigue sin probarse en
repositorios grandes, que es otro régimen: acá el fixture más grande tiene 6
archivos.

### 6.1.2 Una falla que no era del engine

La primera lectura dio **15/16**: `user-owned-tradeoff` falló una vez con
`lean`. Leído el artefacto, la entrega pasó su verificador determinista
(`verify_exit 0`) y lo único que falló fue el patrón léxico sobre la respuesta
final: la tarea exigía `user|priority|choose` y el modelo escribió "asks **you**
to declare whether cost predictability or lowest latency should govern the
**choice**".

No es una regresión de capacidad: es un instrumento frágil. `required_final_patterns`
es una expresión regular sobre texto libre, y una respuesta correcta con otra
palabra la falla. Re-evaluado con `you|user` —que sigue exigiendo que la
respuesta enmarque la decisión como del usuario— el resultado es **16/16 en los
dos brazos**, sin gastar una sola llamada al modelo porque `final_answer` queda
guardada en cada `run.json`. El patrón corregido vive en
`bench/engine_eval_v5.tasks.jsonl`.

**Regla que queda**: cuando un resultado depende de un patrón sobre la respuesta
final, hay que abrir el artefacto antes de reportarlo como fallo de capacidad.

Sobre alucinaciones, la única cifra que existe: 7 llamadas malformadas en el
brazo por defecto contra 8 en `lean`, sobre 16 ejecuciones cada uno. Sin
evidencia de cambio.

### 6.1.3 El catálogo de políticas condicionales sí se paga

`TASK_POLICIES_RENDER_ALL_CONDITIONAL=False` saca 3 756 caracteres por ronda
(§4.5). Medido sobre 4 tareas de código × 3 repeticiones (12 parejas),
`bench/runs/20260913-engine-v2/ab-policy/`:

| métrica | con catálogo | sin catálogo | delta | parejas mejor/peor | p | resuelto |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| tokens de prompt | 82 013 | 77 832 | −5,1 % | 6 / 6 | 0,031 | sí |
| tokens de completion | 5 531 | 4 014 | **−27,4 %** | 8 / 4 | 0,008 | sí |
| latencia | 100,1 s | 50,5 s | **−49,6 %** | 7 / 5 | 0,016 | sí |
| tool calls | 11 | 10,5 | −4,5 % | 5 / 6 | 0,219 | no |
| success | 12/12 | 12/12 | — | — | — | — |

La primera lectura dio 12/12 contra **10/12**, y habría significado que el
catálogo evita fallos. Los dos "fallos" no eran del modelo: uno era un
**timeout del proveedor** registrado como medición, y el otro una violación de
una regla que el pedido nunca enunciaba (§6.1.4). Corregidos los dos
instrumentos, el catálogo no cambia el resultado y cuesta tokens y latencia.

Extendido a 6 tareas × 3 repeticiones (18 parejas), incluyendo las dos cuyo
fragmento es específico:

| métrica | con catálogo | sin catálogo | delta | parejas mejor/peor | p |
| --- | ---: | ---: | ---: | --- | ---: |
| tokens de prompt | 72 929 | 65 354 | −10,4 % | 11 / 7 | 0,4807 |
| tokens de completion | 2 852 | 2 248 | −21,2 % | 11 / 7 | 0,4807 |
| latencia | 60,4 s | 46,1 s | −23,6 % | 12 / 6 | 0,2379 |

**Corregido en §6.1.38:** los tres p-valores salían del test defectuoso. Lo que
queda de esta muestra es la dirección (11 mejor / 7 peor), no la significancia.
| success | 18/18 | 18/18 | — | — | — |

`reversible-ambiguity` (68 049 → 42 839 tokens) y `tool-failure-recovery`
(74 143 → 49 910) mejoran igual que el resto, así que los fragmentos
`compatibility.preserve_public_api` y `bugfix.root_cause` no se extrañan.

**No se cambia el default *en esa ronda*.** La medición es sólida pero su alcance
no es el que la decisión necesita. Los dos documentos que definen este flag
(`CONDITIONAL_TASK_POLICIES.md`, `MINI_MODEL_CONDITIONAL_PROMPTING_ARCHITECTURE.md`)
lo describen como una elección de arquitectura: entregar el catálogo entero como
bloques `<if reason="…">` que **el modelo evalúa**, en lugar de depender de que
el router acierte. Apagarlo no es "recortar bytes": es mover la decisión de
método del modelo al clasificador.

Y la comparación que corresponde a esa decisión no es la que corrí. Este A/B
corrió el `LoopEngine` directo, donde el router **no participa**: mide "ninguna
guía" contra "toda la guía". El producto compara "la guía seleccionada" contra
"toda la guía", y eso exige una campaña en un modo donde el router corra
(`task`/`auto`). Con 18 parejas de tareas de código no alcanza para mover una
decisión de arquitectura documentada; alcanza para dejarla medida, con el
interruptor listo.

Lo que sí queda establecido: **en el modo `orchestrator`, que era el default
entonces, los 3 756 caracteres por ronda no compran nada medible en 18
ejecuciones.** Si el equipo decide que el catálogo se queda, el lugar a arreglar
es que el router no corra en ese modo, no el flag.

#### La campaña que faltaba, corrida — y su resultado no es el que se esperaba

Con el default en `task` (§6.1.8) el router **sí corre**, así que la comparación
que esta decisión necesita quedó disponible. `TASK_POLICIES_RENDER_ALL_CONDITIONAL:
true` (toda la guía como bloques condicionales) contra `false` (la capa
seleccionada por el perfil), 4 tareas × 2 repeticiones, `pipeline_mode: true`,
`MiniMax-M3` (`bench/runs/20260913-engine-v2/ab-policy-pipeline/comparison.md`):

| métrica | toda la guía | guía seleccionada | delta | parejas mejor/peor | p | resuelto |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| tokens de prompt | 82 800 | 73 292 | **−11,5 %** | 6 / 2 | 0,031 | **sí** |
| tokens de completion | 2 846 | 3 058 | +7,4 % | 3 / 5 | 0,25 | no |
| tool calls | 9 | 9 | — | 1 / 3 | 0,38 | no |
| latencia | 75,6 s | 85,9 s | **+13,6 %** | 1 / 7 | — | no |
| success | 8/8 | 8/8 | — | — | — | — |

**El ahorro de tokens se reproduce y la ganancia de latencia no.** En el loop
directo la guía seleccionada era 23,6 % más rápida con p=0,2379 tras la
corrección de §6.1.38 (0,0005 con el test defectuoso); en el modo que
ahora se envía es 13,6 % **más lenta**, y en 7 de 8 parejas. La lectura más
simple es que el router paga latencia (clasificación por embeddings y, con
`TASK_POLICIES_LLM_CLASSIFIER_MODE = "preferred"`, una request extra) que el
loop directo no pagaba, y que el ahorro de prompt no la compensa.

**Decisión: el flag queda en `true`.** Es el único cambio de esta ronda que
enfrenta la métrica mejor rankeada del objetivo —tiempo— contra la sexta
—tokens—, y con 8 parejas la latencia no está resuelta: 7 de 8 parejas es una
dirección, no una significancia. Cambiar un modo documentado como temporal contra
una dirección sin resolver, y contra la métrica que el objetivo pone primero,
sería exactamente la clase de decisión que el resto de este documento evita.

Lo que sí queda claro, y es un hallazgo: **el router no es gratis en latencia, y
su costo no aparecía en ninguna medición anterior porque ninguna corría en el
modo que se envía.** Eso convierte "medir con `pipeline_mode: true`" en requisito
para toda decisión sobre políticas condicionales, no en una preferencia.

### 6.1.4 Dos instrumentos que fallaban solos

**Un timeout del proveedor no es una medición.** El runner, por contrato,
registra la fila con `error` y se detiene. El driver de reanudación la contaba
como unidad completa: fallaba una tarea que el modelo nunca terminó de hacer y,
peor, la escondía del reintento. `_completed_units` ahora ignora las filas con
`error`. Además una campaña de una hora se perdía por un timeout transitorio:
`agent_task_ab` reanuda las unidades ya pagadas y limpia los directorios de
artefactos obsoletos de las que va a repetir.

**Una regla que no está escrita no se puede exigir.** `options-override` declara
`forbidden_changed_paths: ["tests/*"]` y su pedido decía "do not edit the tests".
El modelo **creó** un test nuevo —no editó ninguno— y la tarea lo marcó como
fallo. Los cuatro pedidos de contrato oculto ahora dicen "do not add or change
any file under tests/", que es lo que la tarea ya verificaba.

Las dos correcciones cambiaron el resultado de §6.1.3 de "el catálogo evita
fallos" a "el catálogo no cambia el resultado". Vale la pena el detalle: en tres
rondas seguidas, la primera lectura de un A/B tuvo un artefacto de medición.

### 6.1.5 La duda que quedaba: ¿`lean` pierde exploración?

La caída de rondas de `lean` podía ser eficiencia o podía ser menos exploración,
y en las 8 tareas anteriores el arreglo siempre estaba en un archivo que el
pedido nombraba. `wide-sum` es la sonda: 40 módulos de etapa, uno con la
constante mal, el pedido dice el total correcto y no dice dónde. Los tests
visibles pasan sobre el código roto.

3 repeticiones por brazo, `bench/runs/20260913-engine-v2/ab-wide/`:

| | por defecto | `lean` |
| --- | ---: | ---: |
| success | **3/3** | **3/3** |
| tokens de prompt (mediana) | 158 537 | 133 794 (−15,6 %) |
| latencia (mediana) | 94,7 s | 38,3 s (−59,6 %) |
| archivo cambiado | `src/mod_27.py` | `src/mod_27.py` |

**Las 6 ejecuciones encontraron la etapa correcta** y pasaron el contrato oculto.
El recorte de rondas no costó la localización. Con 3 parejas el test de signos no
resuelve nada, así que esto es evidencia direccional, no concluyente; pero es la
primera medición que apunta a que `lean` no cambia lo que el modelo *encuentra*,
sólo cuánto tarda.

### 6.1.6 Tercera ronda, tercer artefacto: el patrón sobre la respuesta final

La primera lectura de la sonda dio 2/3 contra 1/3. Abiertos los seis artefactos:
**los seis resolvieron la tarea** (`verify_exit 0`, `src/mod_27.py`, sin cambios
prohibidos). Lo único que falló fue `required_final_patterns`, que exigía la
palabra "stage" en la respuesta final.

Tres rondas seguidas, tres artefactos de medición, dos de ellos del mismo
instrumento:

| ronda | caso | qué pasó |
| --- | --- | --- |
| 2 | `options-override` | exigía una regla que el pedido no enunciaba |
| 3 | `user-owned-tradeoff` | la respuesta dijo "choice"/"you", el patrón quería "choose"/"user" |
| 3 | `wide-sum` ×3 | el arreglo era correcto, faltaba la palabra "stage" |

**Corregido en el diseño, no con otro parche.** `required_final_patterns` ya no
forma parte de `success`; se registra como `final_answer_patterns_ok` y se
reporta como métrica aparte (`final_answer_wording`), porque la redacción de la
respuesta **sí** es uno de los objetivos —"mejor comunicación con el usuario"— y
por eso se mide, pero no puede decidir si el trabajo se hizo. La verificación
determinista es del verificador.

`required_action_patterns` se queda como compuerta: "se corrió pytest" es un
hecho observable, no una elección de palabra.

La definición vieja había quedado escrita en 4 filas ya registradas
(`ab-generality` 1, `ab-wide` 3). Se recalcularon contra la definición corregida
y el detalle está arriba; el resto de las filas no cambia.

**Y no es un problema de mis campañas.** En el piloto ya versionado
(`bench/runs/20260804-agent-task-pilot/`), las **2 únicas fallas de la ruta
`luna` son de redacción**: `evidence-code-review` en las dos condiciones, con el
verificador en verde y todos los cambios esperados presentes. La ruta `sol`
(1 falla) y `terra` (2) no tienen ninguna de este tipo. El auditor de completitud
sigue validando ese piloto porque ahora acepta las dos definiciones.

### 6.1.7 Comunicación con el usuario: primera medición

"Mejor comunicación" era el único objetivo sin ningún número. El verificador
juzga corrección y el uso de tokens viene del proveedor, pero la respuesta que
el usuario lee sólo la había juzgado un humano leyéndola.

`bench/agent_task_answer_quality.py` la puntúa con cuatro comprobaciones
deterministas sobre `final_answer`, que queda guardada en cada `run.json`. Sobre
las mismas 16 ejecuciones pareadas de §6.1.1, **sin gastar una sola llamada al
modelo**:

| comprobación | por defecto | `lean` |
| --- | ---: | ---: |
| nombra un comando que corrió | 13/16 | **16/16** |
| enuncia un resultado observado | 16/16 | 16/16 |
| tiene línea `Verification:` | 6/16 | **16/16** |
| abre narrando intención | 0/16 | 0/16 |
| supera las 250 palabras | 2/16 | **0/16** |
| palabras (mediana) | 129 | 122 |

El contrato de respuesta final de `lean` —resultado, después `Verification:` con
el comando y lo que devolvió, después lo pendiente— es lo que produce la
diferencia de 6 a 16 en la línea de verificación y de 13 a 16 en nombrar el
comando.

**Son proxies, y hay que decirlo**: nombrar un comando no prueba que se haya
corrido, y una línea `Verification:` es un formato, no un hecho. Lo que miden es
si la respuesta es *verificable por el usuario*. Eso es lo que el objetivo puede
exigir sin que un humano lea cada respuesta, y es una métrica que se mueve.

La métrica agregada `answers_without_a_command` (cuenta de respuestas que no
nombran ningún comando) ahora forma parte de la comparación pareada; con 3
parejas no resueltas en este corpus, no alcanza el mínimo de 6 del test de
signos.

> Corrección: §6.1.6 afirma que `final_answer_wording` "se reporta como métrica
> aparte". El código que la agrega a `_METRICS` nunca se aplicó — la lista
> quedaba sin ella y las tablas publicadas en esa ronda no la muestran. Está
> corregido en esta ronda; las filas anteriores a la ronda 4 no tienen el campo
> y por eso aparecen en cero.

### 6.1.8 El benchmark no medía el modo por defecto, y ahora puede

*(Toda esta sección se escribió cuando el default era `orchestrator`. La
decisión final está al final de la sección.)*

Todo lo anterior corre `LoopEngine.execute` directo. El producto usa
`run_task`: agente de chat, elaboración, selección de engine, loop y revisión. La
configuración del benchmark tiene ahora `pipeline_mode`, que llama al mismo
`run_task` que la TUI, la CLI clásica y el servidor web, con los hooks no
interactivos que la CLI de un solo prompt ya usa.

Lo que aparece al mirar por esa ventana, en las **primeras dos ejecuciones**
(4 tareas × 1 repetición × 2 modos, en curso):

| tarea | modo | resultado | tool calls | tokens de prompt |
| --- | --- | --- | ---: | ---: |
| test-selection | `orchestrator` | completado | 43 | 921 410 |
| cart-immutability | `orchestrator` | bloqueado | 38 | 666 261 |
| test-selection | `task` (directo) | completado | 16 | 114 621 |

Dos cosas, y la primera explica la segunda:

1. **El orquestador no escribe archivos.** `OrchestratorAdapter` acota sus
   herramientas a `_ROOT_READS` + `send_message` + las nueve `team_*`
   (`engines/orchestrator.py:18,45-49`). No tiene `edit_file`. En
   `test-selection` delegó (`team_delegate` ×2) y un worker hizo el cambio; en
   `cart-immutability` no delegó y terminó con
   *"Blocked — no file-write tool is advertised for this orchestrator turn"*.
2. Por eso cuesta **5-9× más tokens** que el motor `task` directo sobre la misma
   tarea: el principal lee, el worker vuelve a leer, escribe, y encima corre la
   revisión.

No es un defecto: es la arquitectura del modo por defecto. Pero significa que
**la pregunta "¿qué engine da mejores resultados?" no estaba siendo medida**, y
que el modo por defecto paga una coordinación que en tareas de un archivo no
compra nada. La campaña sigue; con 2 filas no se concluye.

#### El resultado con el arreglo, y la advertencia que lo acompaña

Mismas 4 tareas, ahora con el mensaje corregido (`ab-engine-mode-fixed`):

| métrica | `orchestrator` | `task` | delta | parejas mejor/peor |
| --- | ---: | ---: | ---: | --- |
| success | **3/4** | **4/4** | — | — |
| tokens de prompt | 536 660 | 84 094 | **−84,3 %** | 4 / 0 |
| tokens de completion | 11 290 | 1 874 | −83,4 % | 4 / 0 |
| tool calls | 27 | 9 | −66,7 % | 4 / 0 |
| latencia | 331,3 s | 79,8 s | **−75,9 %** | 3 / 1 |

El arreglo del mensaje convirtió tres fallas en dos: 1/4 → 3/4. Y **la brecha de
costo se ensanchó**: el orquestador ya no se pierde adivinando, ahora delega de
verdad, y delegar cuesta más que hacer.

Con 4 parejas el test de signos sigue sin resolver (p=0,125; el mínimo son 6), así
que la campaña se extendió a las 10 tareas del corpus. Pero la forma del
resultado es difícil de atribuir a ruido: **las 4 parejas mejoran en tokens,
completion y tool calls**, y la magnitud (−84 %) está muy por encima de la
varianza observada.

#### El resultado anterior, antes del arreglo

4 tareas × 1 repetición, `pipeline_mode`, mismo modelo y mismo corpus:

| métrica | `orchestrator` (el default de entonces) | `task` (directo) | delta | parejas mejor/peor |
| --- | ---: | ---: | ---: | --- |
| success | **1/4** | **4/4** | — | — |
| tokens de prompt | 590 965 | 112 493 | **−81,0 %** | 4 / 0 |
| tokens de completion | 13 676 | 2 721 | −80,1 % | 4 / 0 |
| tool calls | 34,5 | 9,5 | −72,5 % | 4 / 0 |
| llamadas malformadas | 7,0 | 0,0 | −100 % | 4 / 0 |
| latencia | 182,9 s | 96,7 s | −47,1 % | 3 / 1 |

Con 4 parejas el test de signos **no resuelve** (p=0,125; el mínimo es 6), así que
por mi propio criterio esto es direccional. Pero las tres fallas tienen una sola
causa, y la causa es un defecto arreglado **después** de que estas ejecuciones
corrieran: ver abajo. Las tres respuestas finales **diagnostican el bug
correctamente** —`options-override` describe el merge invertido,
`wide-sum` dice "step_27 does `return value - 40`, should be `return value + 27`"—
y las tres terminan en `blocked` porque el principal no tiene con qué escribir.

Es decir: el modo por defecto **entiende el trabajo y no puede hacerlo**, y gasta
entre 6 y 10 rondas descubriéndolo.

**Y el modo por defecto gasta diez rondas descubriendo que no puede escribir.**
`cart-immutability` produjo 10 llamadas malformadas, y `malformed_call_reasons`
muestra la cascada completa:

```
edit_file({"file_path":"src/cart.py", ...})        ×5
apply_file_patch({... "replacements":[ ... ]})     ×3
create_file({"file_path":"src/cart.py", ...})      ×1
write_file(...) -> hallucinated tool 'create_file
apply_patch(...) -> hallucinated tool 'apply_patch
```

El modelo intenta escribir, el motor responde
*"Unknown tool: edit_file. Did you mean one of: read_file, team_create_ticket,
team_idle?"* y el modelo sigue probando herramientas de escritura. Ninguna está
advertida, y el mensaje **no dice por qué**: `edit_file` existe en el registro,
simplemente no está concedida a este rol.

Corregido en `_unknown_tool_message`: cuando el nombre existe en el registro y
la herramienta `team_delegate` está disponible, la respuesta dice que la
herramienta es real y no está concedida, y nombra la ruta — abrir un ticket y
delegarlo con `tools=['edit_file']`. Un nombre realmente inexistente sigue
recibiendo la lista por similitud.

Comparación del costo de alucinación entre caminos, sobre las campañas
registradas: **0,3–0,5 llamadas malformadas por ejecución en el loop directo,
6,0 en el camino del orquestador.** La diferencia es casi toda esta cascada.

Las 4 ejecuciones de la tabla de arriba son **anteriores a este arreglo**. La
campaña `ab-engine-mode-fixed` las repite con el mensaje corregido; hasta que
cierre, la conclusión correcta no es "el modo por defecto es peor" sino **"el
modo por defecto tenía una cascada de 6 a 10 rondas que le impedía terminar
tareas de un archivo, y esa cascada está arreglada pero sin medir"**. Cambiar el
default con esos datos habría sido exactamente el error que este documento viene
evitando — y no hizo falta: la campaña con el arreglo cerró, y con las 10 tareas.

#### El resultado con el arreglo, sobre las 10 tareas

| métrica | `orchestrator` | `task` | delta | parejas mejor/peor | p | resuelto |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| success | 9/10 | 9/10 | — | — | — | — |
| tokens de prompt | 502 079 | 85 722 | **−82,9 %** | 9 / 0 | 0,0039 | **sí** |
| tokens de completion | 12 828 | 2 166 | −83,1 % | 9 / 0 | 0,0039 | **sí** |
| tool calls | 27 | 9 | −66,7 % | 9 / 0 | 0,0039 | **sí** |
| latencia | 213,1 s | 82,1 s | **−61,5 %** | 8 / 1 | 0,0078 | **sí** |

Las dos fallas son de tareas distintas: el orquestador falla `wide-sum`; `task`
falla `complex-plan` **antes de entrar al loop** —el pipeline se detiene en
"waiting for confirmation of product decision(s)" y no llega a llamar al
modelo—. Ese par se excluye de las métricas de costo (una ejecución que nunca
llegó al modelo no tiene costo medible, y contarla como cero le regalaría al
brazo el resultado más barato posible por trabajo que declinó empezar) y cuenta
igual en el conteo de éxito, porque negarse a empezar es un desenlace real.

**Ocho veces más barato, 62 % más rápido, la misma tasa de éxito, sobre diez
formas de tarea.** Ésa es la medición más fuerte del documento.

#### La celda que faltaba, medida

`research-audit` es el nicho declarado del orquestador: tres módulos
independientes, una sola salida, cero escrituras al código fuente. Si el
orquestador paga en algún lado, es acá. A/B dedicado, 3 repeticiones por brazo
(`bench/runs/20260913-engine-v2/ab-audit2/comparison.{md,json}`):

| métrica | `orchestrator` | `task` | delta | parejas mejor/peor |
| --- | ---: | ---: | ---: | --- |
| success | 2/3 | **3/3** | — | — |
| tokens de prompt | 771 470 | 90 442 | **−88,3 %** | 3 / 0 |
| tokens de completion | 21 188 | 3 382 | −84,0 % | 3 / 0 |
| tool calls | 31 | 10 | −67,7 % | 3 / 0 |
| latencia | 405,5 s | 149,1 s | **−63,2 %** | 3 / 0 |

Con 3 parejas la prueba de signos no resuelve (p=0,25) y así queda declarado: el
número es un punto estimado, no un resultado con significancia. Pero la hipótesis
que esta celda venía a sostener —"el nicho de lectura independiente compensa el
costo"— **queda refutada en su propio terreno**. No sólo el orquestador cuesta
9,4× más y tarda 2,7× más en la tarea diseñada para él: además falla una de tres.

Y esa falla no es de razonamiento. Es el Step sin salida de §6.1.13 —el modo
recuperación le escondía la delegación mientras el cierre se la rechazaba—, que
se arregló después de esta campaña.

#### La decisión

La evidencia alcanza para **cambiar el default a `task`**. Se apoya en tres
mediciones independientes que apuntan al mismo lado:

1. 10 formas de tarea, mismo éxito (9/10 vs 9/10), −82,9 % de tokens y −61,5 %
   de latencia, con p<0,05 en las cuatro métricas pareadas.
2. `research-audit`, el nicho del orquestador, 2/3 vs 3/3 y −88,3 % de tokens.
3. Las dos fallas conocidas de `task` eran defectos del engine, no del modo: la
   elaboración que se detenía ante una pregunta sin destinatario (§6.1.14) y, del
   lado del orquestador, el Step sin salida (§6.1.13). Ambos arreglados.

A eso se suma un argumento de coherencia que no necesita medición: **el
clasificador del propio producto no puede elegir el modo que el producto trae por
defecto** (§6.1.12) y, para trabajo que escribe código, la delegación no puede
paralelizar nada porque los workers que escriben están serializados por código.

#### La corrección que faltaba: el contador de tokens no contaba todo

La decisión de arriba se tomó con la métrica que el harness tenía, y esa métrica
era **el contador del `LoopEngine`, no lo que el proveedor facturó**. Faltaban
dos conjuntos de llamadas, y faltaban en direcciones opuestas:

* el camino `task` paga el chat agent, el planner, el spec elaborador, el
  clasificador de políticas y la revisión, que llaman al proveedor por su cuenta
  y **nunca** llegaban a `total_prompt_tokens`;
* el camino `orchestrator` paga el loop de **cada worker**, cuyos contadores van
  a `TeamRuntime._worker_metrics` (`engine/team/runtime.py:487-491`) y tampoco
  llegaban al `LoopState` que el runner leía.

Es decir: la comparación enfrentaba *el principal del orquestador* contra *el
loop entero de `task`*. Ninguno de los dos números era el costo del turno.

Corregido contando en la frontera del proveedor, con un callback de litellm que
el runner registra (`bench/agent_task_run.py`), lo que ve **todas** las llamadas
del proceso sin depender de que cada fase quiera reportarse. La comparación
siguiente es la primera con el número completo; mismos dos tasks, una repetición,
`pipeline_mode: true`, `MiniMax-M3`
(`bench/runs/20260913-engine-v2/ab-accounted`):

| tarea | modo | loop (lo que se reportaba) | proveedor (lo que se factura) | rondas de modelo | latencia |
| --- | --- | ---: | ---: | ---: | ---: |
| `complex-plan` | `orchestrator` | 465 649 | **780 845** | 48 | 511 s |
| `complex-plan` | `task` | 57 553 | **85 904** | 13 | 171 s |
| `cart-immutability` | `orchestrator` | 321 505 | **570 494** | 38 | 270 s |
| `cart-immutability` | `task` | 82 648 | **92 266** | 10 | 109 s |

Dos cosas quedan claras:

1. **La omisión del orquestador era la grande.** Su costo real es 1,68×–1,77× el
   que se reportaba (los workers), contra 1,12×–1,49× del lado de `task` (las
   fases del pipeline). El sesgo favorecía al orquestador.
2. **El resultado no cambia, se profundiza.** Pareado por tarea y repetición:
   −89,0 % en `complex-plan` y −83,8 % en `cart-immutability` sobre lo facturado
   (con el contador viejo esas mismas dos parejas daban −87,6 % y −74,3 %). La
   mediana pareada es −86,8 % en `pipeline_prompt_tokens` y −62,7 % en
   `pipeline_completion_tokens`, con **−73,3 % de rondas de modelo** (48 → 13,
   38 → 10) —la métrica que no se puede esconder escondiendo una fase— y −64,2 %
   de latencia. Con 2 parejas nada queda resuelto, y así queda declarado: esto
   **no es una estimación del efecto** —ésa sigue siendo la campaña de 10 tareas,
   p=0,0039—, es la medición de *qué contaba la métrica*.
   (`bench/runs/20260913-engine-v2/ab-accounted/comparison.md`.)

Las 10 tareas de §6.1.8 siguen midiendo lo que medían —un subconjunto del costo,
en ambos brazos— y su −82,9 % queda como **piso**, no como cifra final: el sesgo
iba a favor del orquestador, así que el número completo es más negativo que eso.
Re-medirlo con el contador nuevo está en §7.

### 6.1.9 Un fallo del proveedor no es una tarea fallida

Las cinco ejecuciones que murieron por cuota se registraron como filas normales:
sin `error`, sin `engine_status`, cero tokens, verificación fallida. El pipeline
**devuelve las excepciones del engine como el texto del turno**, así que
"APIConnectionError … rate limit" terminaba en `final_answer` y la fila era
indistinguible de una tarea que el modelo no supo hacer. Una campaña contra un
plan agotado producía tareas fallidas falsas en vez de detenerse.

Corregido: `provider_failure_in_reply` marca la fila como error cuando el texto
lo delata **y** los tokens de prompt son cero —una ejecución que llegó al modelo
no pudo gastar nada, y una respuesta real puede mencionar un timeout sin serlo—.
El runner se detiene en la primera, como con cualquier error de proveedor, y la
unidad no se cuenta como completada.

### 6.1.10 Ken ensuciaba el diff de todas las ejecuciones

`.ken/` (el índice y el daemon) se escribe al lado del workspace que indexa, y
no estaba en la lista de rutas ignoradas: `ken.db`, `vectors/*` y los archivos
del daemon aparecían como cambios del agente en cada ejecución registrada. Eso
infla `changed_lines` y copia bases de datos al artefacto. Ignorado.

### 6.1.11 El orquestador se colgaba esperando a un worker que ya había terminado

Persiguiendo la celda que faltaba apareció algo peor que un costo alto: **un turno
que no termina nunca**. La salida en vivo de una auditoría por el orquestador:

```
Orchestrator: scoped workers, shared notes and peer messages
Adrian · Writer: Write AUDIT.md for src/ security audit
Orchestrator: idle — Waiting for Adrian to write AUDIT.md and run verify_contract.py
   ✔ Done (5 calls · 71692 tokens)
     Created AUDIT.md at repo root …
```

El worker terminó. El orquestador siguió idle para siempre. La misma tarea, en
otra ejecución, había completado normalmente: es una carrera, no una condición
determinista.

**La causa** está en el bucle de suscripción (`engine/team/waiting.py`): comprueba
el log de eventos y después espera.

```python
if not sleeping:
    ...suspender, liberar el lease, emitir el aviso...
else:
    team._condition.wait(timeout=remaining)     # remaining es None sin deadline
```

Es la ventana clásica de *lost wakeup*: un worker que termina **entre** la
comprobación y el `wait()` no notifica a nadie que ya esté esperando, y como el
timeout por defecto es `None` (`IdleInput.timeout`, `waiting.py:30`), la espera
no vuelve jamás. No hay límite de tiempo que la rescate, y el turno del usuario
queda colgado.

**Corregido**: `_wait_slice` nunca devuelve `None`. Una suscripción sin deadline
despierta cada 5 segundos y vuelve a leer el log —el bucle ya lo hace en cada
iteración, así que un despertar de más no cuesta nada y no llama al modelo—, y
una con deadline lo respeta. El mismo turno que se colgaba ahora termina en
194 s. Como efecto secundario, la cancelación mientras está idle también pasa a
ser responsiva en 5 segundos en vez de nunca. Verificado en vivo: el mismo turno
ahora registra `Orchestrator: resumed — event` cuando el worker entrega.

**Y la misma traza muestra de qué se compone el costo.** Antes de delegar, el
principal gastó tres Steps seguidos sin una sola llamada a herramienta:

```
⚠ LLM returned text 4x without calling a tool — pausing the step
  ➜ Continue (7 calls · 110768 tokens)
⚠ LLM returned text 4x without calling a tool — pausing the step
  ➜ Continue (0 calls · 167341 tokens)
⚠ LLM returned text 3x without calling a tool — pausing the step
  ➜ Continue (0 calls · 210581 tokens)
```

**488 691 tokens en tres Steps que no hicieron nada**, cada uno cerrado por el
guard de texto-sin-herramienta. El worker, después, resolvió la tarea con 4
llamadas y 55 304 tokens. Es el mismo patrón que el livelock de §3 visto desde el
otro lado: el guard corta correctamente, pero nada impide que el modelo vuelva a
producir texto en el Step siguiente. El costo no está en delegar; está en las
rondas improductivas del principal antes de decidirse a delegar.

### 6.1.12 Por qué la delegación no puede pagar su costo en trabajo de código

`TeamRuntime._acquire_worker` (`engine/team/runtime.py:156-167`) bloquea a
cualquier worker cuyas herramientas incluyan una de escritura mientras otro esté
escribiendo:

```python
writes = any(not getattr(self.catalog[n], "is_read_only", False) for n in member["tools"])
if len(self._executing) < self._max_workers and (not writes or not self._writing):
```

Es exactamente lo que el prompt del equipo declara ("Independent read tasks can
run together; workspace-writing workers are serialized"). La consecuencia es
directa y no depende de ninguna medición: **para cualquier tarea que escriba
código, el orquestador no puede paralelizar nada.** Sus workers corren de a uno,
cada uno con su propio contexto releyendo archivos que el principal ya leyó,
encima de la lectura del principal y de la fase de revisión.

Donde *podría* pagar es en **trabajo de lectura independiente** —investigación
en paralelo, auditorías, comparaciones—, que es su nicho de diseño. Medido en
§6.1.8: no paga tampoco ahí.

**Y el clasificador del producto nunca podía elegir el default.** `_classify_auto`
(`engines/routing.py:146-195`) devuelve `task` con confianza 0,9 para todo salvo
que disparen las palabras clave de grafo —y `AUTO_ENGINE_ALLOW_GRAPH` es `False`
por defecto—, con el comentario "normal work uses one durable Task with a rolling
Step horizon". Sus dos salidas son `graph_beta` y `task`; **no hay ningún camino
que devuelva `orchestrator`**.

Mientras el default fue `orchestrator` eso era una contradicción: un usuario que
elegía `auto` nunca recibía el modo que el producto traía por defecto. Con el
default en `task` (§6.1.8) el clasificador y el default coinciden, y que `auto`
no alcance a `orchestrator` deja de ser un defecto para volverse lo correcto: es
el modo que ninguna medición favorece. Si la serialización de escrituras cambia,
el lugar donde agregar esa rama es `_classify_auto`, y la condición a medir es
una tarea de lectura independiente.

### 6.1.13 El orquestador no puede cerrar un Step que la recuperación deja sin salida

Esta es la falla más cara que se encontró, y explica buena parte de por qué
`orchestrator` gasta entre 2× y 9× más que `task` en las mismas tareas.

**El mecanismo.** Cuando dos ventanas completas de un Step no cambian el
workspace ni el resultado de un test, el engine "enciende" el modo recuperación
(`_configure_progress_recovery`, `engine/loop/engine.py:304-341`; el contador se
incrementa en `_finalize_inner_loop`, `engine/loop/engine.py:2021-2058`). La
instrucción que el modo recuperación implementa es *dejá de leer y editá*. Tres
lugares la hacen cumplir, y ninguno sabe para quién:

| lugar | qué hace |
| --- | --- |
| `LLMCaller._available_schemas` (`loop/llm_caller.py:576-610`) | borra del espacio de acciones todo lo que no sea `step_complete`, `execute_command`, una herramienta de edición o una lectura de contexto |
| `ToolRunner._partition_suppressed_discovery` (`loop/tool_runner.py:530-639`) | rechaza sintéticamente toda llamada que no sea acción |
| `StepCompleteGate._workspace_recovery_escape` | rechaza el `step_complete` — "recovery mode is not an external blocker" |

**Por qué es inviable para el orquestador.** El principal del orquestador
(`user_facing_orchestrator_with_scoped_workers`) **no tiene `edit_file`**: su
única forma de cambiar el workspace es delegar. Pero

* la única señal de progreso que apaga el contador es un cambio neto del
  workspace (`tracker.net_workspace_changed`), y ninguna herramienta suya puede
  producirlo, así que el contador sube y no baja nunca;
* al encenderse, la recuperación esconde `team_delegate` y `team_create_ticket`,
  que son la única vía de progreso que le quedaba;
* y el `step_complete` con el que podría cerrar el Step se rechaza porque el
  estado de recuperación no es un bloqueo externo.

El resultado es un Step que no se puede avanzar **ni** cerrar. No es una
hipótesis: el propio run lo dice en su respuesta final.

**Evidencia 1 — una tarea fallida tras quemar 771 470 tokens.**
`bench/runs/20260913-engine-v2/ab-audit2/orchestrator` (3 repeticiones de
`research-audit`, A/B contra `task`, `MiniMax-M3`): r0 y r1 terminan bien; **r2
falla** con `AUDIT.md` sin crear y `changed_paths: []`, después de 31 llamadas y
771 470 tokens. Su respuesta final:

> Step cannot be closed because the engine holds it for an unresolved ticket
> (t_12439ead272a) that I am unable to cancel while **the recovery state is
> suppressing team actions**. All write paths and shell inspection are frozen;
> the only available action is to call `step_complete`, which the engine itself
> lists as available. No source file under src/ was modified.

**Evidencia 2 — un diagnóstico correcto que no se pudo aplicar.**
`bench/runs/20260913-engine-v2/ab-engine-mode-fixed/orchestrator`, `wide-sum`:

> Identified single defective stage: src/mod_27.py returns `value - 40` instead
> of `value + 27`. Sum 1..40 = 820, observed 753, diff 67 = 27 - (-40). All other
> 39 stage modules correct. **Edit was blocked: edit_file denied with "not
> granted to this role"; team_create_ticket and team_delegate suppressed**

El modelo encontró el bug exacto —la misma constante que el brazo `task`
corrigió para pasar— y entregó `verify rc 1` con `changed_paths: []`. El fallo no
fue de razonamiento: fue que el engine le quitó las dos salidas que tenía.

**El arreglo.** `behavior_rules.role_can_edit_workspace(ctx)` lee la lista de
schemas otorgados —la misma frontera de seguridad que ve el modelo, en modo
nativo y manual— y responde si *alguna* herramienta del rol puede mutar el
workspace. Con eso:

* el latch sólo se arma para un rol que puede editar (`engine.py:323`);
* un rol sin herramienta de edición conserva `team_create_ticket` y
  `team_delegate` durante la recuperación, en el espacio de acciones
  (`llm_caller.py`) y en el ejecutor (`tool_runner.py`);
* el rechazo sintético nombra la acción que ese rol *sí* tiene ("hand the change
  to a worker that can edit…") en lugar de "edit that target", que era
  instrucción imposible;
* las concesiones desconocidas fallan abierto: un contexto que no publica
  schemas conserva el comportamiento anterior, así que sólo cambia el resultado
  para un rol que demuestra no tener herramienta de edición.

Tres tests nuevos: dos sobre la visibilidad de schemas
(`tests/test_loop_helpers.py`) y dos sobre el latch y el ejecutor
(`tests/test_loop_termination.py`).

**Medición del arreglo.** Las mismas celdas, con el arreglo, en un proceso
nuevo (`bench/runs/20260913-engine-v2/ab-audit3`, comparación pareada en
`ab-audit3/comparison.md`; y `ab-wide3`):

| celda | antes | después |
| --- | --- | --- |
| `research-audit` × 3, `orchestrator` | **2/3** success, mediana 771 470 tokens, 31 tool calls | **3/3** success, mediana 440 180 tokens, 31 tool calls |
| `wide-sum` × 1 en `ab-engine-mode-fixed` | **0/1** success, `changed_paths: []`, el bug diagnosticado y no aplicado | **2/2** success, ~450 000 tokens cada una |

La traza de `wide-sum` después del arreglo dice exactamente lo que faltaba antes:

> Identified the offending stage as mod_27 and delegated a focused edit through
> worker Mateo. Worker reported `pytest` 3 passed and `pipeline(0)==820`

Lo que el arreglo **no** hace es volver barato al orquestador: sigue costando
~440 000 tokens por auditoría contra 90 442 de `task`. El arreglo convierte una
falla dura en un éxito caro; la decisión de default (§6.1.8) es la que recupera
el costo, y son dos cambios distintos por dos razones distintas.

### 6.1.14 Una pregunta que nadie puede responder detenía la ejecución entera

`OrchestrationHooks.ask_user` es el método del `Protocol` contra el que está
escrito el pipeline, y su contrato es explícito
(`engine/orchestration/pipeline.py:138-144`):

> Return the user's text answer (possibly empty), or ``None`` to indicate the
> caller cannot be interactive at all (single-shot mode). **Pipeline branches
> that receive ``None`` MUST proceed with sensible defaults instead of
> failing.**

`NonInteractiveHooks` —el modo `--prompt`— lo repite: "the spec confirmation
step proceeds without asking" (`hooks.py:215-222`). `_run_elaboration_phase`
hacía lo contrario: cualquier respuesta no verdadera iba a `unanswered` y
terminaba en `execution_blocked_reason`, con el pipeline detenido antes de
planificar (`pipeline.py:355-372`).

**Medición.** `complex-plan` en modo `task`, 3 repeticiones: las tres se detienen
con **0 tokens de prompt**, sin plan y sin `PLAN.md`, y la respuesta final *es*
la pregunta:

```
Execution is waiting for confirmation of product decision(s):
- What retention window should applied to generated export artifacts?
- What concurrency / per-tenant rate of export job execution should the worker allow?
```

En modo `orchestrator` la tarea "pasa" porque la fase de elaboración **no corre**
(`pipeline.py:1183`: `escalation if orchestrating else _run_elaboration_phase(...)`).
Es decir: el modo por defecto no gana por ser mejor en esa tarea, sino por no
hacer la pregunta. Y el modo que sí la hace castiga al usuario no interactivo con
cero trabajo en cualquier pedido que el elaborador lea como consecuente.

Nótese que la rúbrica de `complex-plan` pide exactamente lo contrario de lo que
el engine hacía: *"Consequential retention and rollout choices are surfaced with
a recommendation instead of silently decided"*. El engine las sacaba a la
superficie como pregunta y no entregaba nada.

**El arreglo.** Las tres respuestas posibles quedan distintas, que es lo que el
contrato pedía:

| `ask_user` devuelve | significado | qué hace el engine |
| --- | --- | --- |
| texto | el usuario respondió | decisión autoritativa (`confirmed_decisions`) |
| `""` | se le preguntó y no dijo nada | bloquea, como antes |
| `None` | **no hay a quién preguntar** | ejecuta el default declarado, y lo reporta como no confirmado |

Los no confirmados no se mezclan con `clarifications_needed` (mentirían sobre su
`risk`): viven en `GroundedSpec.unconfirmed_decisions`, se avisan por
`hooks.notify` y se renderizan al planner como *"implement the stated default and
state it as an unconfirmed decision in the deliverable and in your final
answer"*. El default sigue siendo del modelo; lo que cambia es que no es secreto.

**Medición del arreglo.** `complex-plan` en modo `task`, 3 repeticiones, después
del arreglo (`bench/runs/20260913-engine-v2/ab-plan3/task`):

| | antes | después |
| --- | --- | --- |
| success | **0/3** | **3/3** |
| tokens de prompt | 0 (nunca llegó al modelo) | 79 983 / 103 716 / 77 884 |
| tool calls | 0 | 6 / 7 / 5 |

Contra el mismo `complex-plan` en `orchestrator` (640 097 tokens en
`ab-engine-mode-fixed`), el modo `task` lo resuelve con **−87,5 % de tokens**.
Los tres `PLAN.md` resultantes exponen las decisiones consecuentes con
recomendación —D1..D7 con dueño, blocker y default seguro— que es exactamente lo
que la rúbrica pedía y lo que el halt no entregaba.

### 6.1.15 El clasificador de políticas por LLM: resultado negativo, y una asimetría que no necesita benchmark

Con el default en `task`, `resolve_task_profile` corre en cada turno. Con
`TASK_POLICIES_LLM_CLASSIFIER_MODE = "preferred"` —el valor enviado— eso significa
**una llamada extra al modelo principal antes de componer el prompt**, en toda
request que no sea una explicación entrecomillada
(`engine/task_policies/router.py:376-383`). La pregunta es si esa llamada paga.

A/B pareado, 4 tareas × 2 repeticiones, `pipeline_mode: true`, `TASK_ENGINE_MODE:
task`, `MiniMax-M3` (`bench/runs/20260913-engine-v2/ab-classifier/comparison.md`):

| métrica | `preferred` | `off` | delta | parejas mejor/peor | p | resuelto |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| tokens de prompt | 77 021 | 95 787 | +24,4 % | 4 / 4 | 0,125 | no |
| latencia | 85,8 s | 80,5 s | −6,2 % | 3 / 5 | 0,25 | no |
| success | 8/8 | 8/8 | — | — | — | — |

**Nada queda resuelto.** La latencia se inclina hacia `off` (3 de 5 parejas, y
`evidence-code-review` 222 s → 140 s) pero se reparte, y el resto de las métricas
son ruido: los tokens de prompt **suben** en el brazo sin clasificador, lo cual no
puede ser un efecto del clasificador —corre antes del loop, y su uso ni siquiera
llega a los contadores del engine— sino la dispersión conocida de esta tarea.

**Decisión: no se cambia**, por la misma regla que dejó el catálogo de políticas
en `true`: 8 parejas sin resolver no mueven un default.

**Lo que sí queda establecido, y no necesita la campaña, es una asimetría de
correctitud.** El modo `preferred` llama al merge con
`replace_classified_operations=True`, y `_CLASSIFIED_OPERATIONS`
(`router.py:160-162`) es *exactamente* el conjunto que produce el parser literal
(`router.py:187`). Es decir: **la respuesta del LLM reemplaza el conjunto de
métodos que el texto justificaba, y puede restarlo.** `fallback`, en cambio, sólo
corre cuando el parser literal no encontró nada (`ambiguous = not operations`,
`router.py:532-538`) y llama al merge sin reemplazo, así que sólo puede sumar.
Con `TASK_POLICIES_RENDER_ALL_CONDITIONAL = true` el catálogo entero se renderiza
igual, de modo que lo único que la sustracción puede cambiar es el resumen del
perfil y el empujón de edición en runtime
(`engine/behavior/runtime_policy.py:137-145`) —riesgo acotado, no daño observado—.
Pero pagar una llamada invisible por turno para poder *quitar* una señal literal,
sin beneficio medido, es un intercambio que no se sostiene: la campaña que
corresponde es `fallback` contra `preferred`, donde el clasificador conserva la
capacidad y pierde la sustracción.

### 6.1.16 El mismo archivo dos veces: `abspath` contra `realpath`

Revisando los diffs guardados para contestar una pregunta sobre calidad de
código apareció un tercer defecto de medición, y esta vez también de producto.

**Qué pasaba.** `FileChangeTracker.record` guardaba cada archivo bajo
`os.path.abspath(path)`, mientras que `WorkspaceBaseline.root` es
`os.path.realpath(root)` (`workspace_baseline.py:58`) y `reconcile_workspace`
arma la ruta con `os.path.join(realpath_root, relative)`. En macOS
`/var/folders/...` y `/private/var/folders/...` son **el mismo archivo**: una
llamada de herramienta que usa una forma y un barrido que usa la otra producían
**dos entradas para un solo archivo**.

**Cuánto.** En 22 de las 32 ejecuciones guardadas de `ab-generality`, el resumen
de diffs traía el mismo archivo dos veces. Eso:

* duplica el diff que recibe el **revisor de código** en el prompt
  (`review_engine.py:1335` le pasa `get_changed_files_summary()`), o sea tokens
  y latencia de producción tirados;
* duplicaba `changed_lines` y `introduced_placeholders`, y **no de forma
  uniforme**: 22 ejecuciones al doble y 10 simple, según qué forma de la ruta
  hubiera usado el modelo. Un sesgo uniforme se cancela en una comparación
  pareada; una moneda al aire por ejecución, no.

**Corrección, en los dos lugares.** En el producto,
`FileChangeTracker._key()` normaliza con `realpath` en los siete puntos que
siembran o buscan un archivo, lo que lo hace consistente con el baseline.
En la métrica, `deduplicated_diff()` colapsa las secciones repetidas por
`realpath` antes de contar, para que las 32 ejecuciones ya escritas se puedan
re-analizar sin volver a correr el modelo. Dos tests: uno de producto
(dos formas de la misma ruta son un solo cambio) y dos de métrica.

**Efecto medido sobre la campaña principal**, recalculada offline sobre los
mismos artefactos (`ab-generality`, 16 parejas):

| `changed_lines` | mediana default | mediana lean | delta | parejas mejor/peor |
| --- | ---: | ---: | ---: | --- |
| como se reportó | 29,0 | 16,0 | −13,0 | 8 / 6 |
| corregido | 14,5 | 8,0 | −6,5 | 8 / 3 |

Una de las 16 parejas **cambia de dirección** (`pricing-rounding` r0: crudo decía
3 contra 4, corregido dice 3 contra 2). Los titulares no se mueven —`prompt_tokens`
−35,9 % (12/4, p=0,0768 tras §6.1.38) y latencia −33,0 % (12/4, mismo p) se recalculan
idénticos—, pero `changed_lines` nunca más se reporta sin deduplicar, y el
protocolo del documento gana una regla: **antes de contar líneas de un diff,
verificar que el diff no tenga el mismo archivo dos veces.**

### 6.1.17 Por primera vez, la calidad juzgada: 289 rúbricas que nadie había puntuado

Todas las campañas de este documento compararon verificadores deterministas,
tokens y latencia. Pero el corpus guarda **289 ejecuciones con ítems de rúbrica
de tipo `human_review`** —*"el arreglo corrige la constante en la etapa
equivocada en vez de compensarla en el pipeline"*, *"las decisiones consecuentes
se exponen con recomendación en vez de decidirse en silencio"*, *"el trace
muestra el fallo del índice semántico y un cambio a evidencia directa en vez de
reintentos a ciegas"*— y **ninguno se puntuó jamás**. Es el único lugar donde
viven la calidad del código, la franqueza del traspaso y la propiedad de las
decisiones. Un documento que reporta −35,9 % de tokens y no dice nada sobre eso
está midiendo la mitad barata.

**Cómo se cerró.** `bench/agent_task_blind_review.py` (nuevo) arma un paquete
**ciego**: quita el brazo de cada ejecución, le da un id opaco, mezcla con semilla
fija y escribe la clave en un archivo aparte que no se abre hasta tener todos los
puntajes. Puntuar tu propio cambio sabiendo de qué brazo salió cada diff es la
única forma de que esto no valga nada. La escala es 0 (no cumple), 1
(parcialmente), 2 (cumple).

Juzgado: `ab-generality`, repetición 0, **16 ejecuciones pareadas (8 tareas)**,
16 ítems, 32 juicios, un solo juez ciego.

| | default | `lean` |
| --- | ---: | ---: |
| puntaje medio (0–2) | 1,81 | **1,94** |
| verificador determinista | 8/8 rc 0 | 8/8 rc 0 |

**Catorce de los 16 ítems puntúan idéntico.** Los dos que difieren, ambos por un
punto y a favor de `lean`, son:

* `concise-handoff` (`complex-plan`) — 1 contra 2. El run de `default` enumera
  las doce secciones del plan en la respuesta y **no cita ningún comando de
  verificación**; el de `lean` resume en siete líneas y cierra con
  `python3 verify.py → exit 0` y lo que quedó deliberadamente sin hacer.
* `failure-recognition` (`tool-failure-recovery`) — 1 contra 2. El run de
  `default` usa evidencia directa tras el fallo, pero además **reintenta el
  comando roto** una vez más; el de `lean` no vuelve a tocarlo.

**Lo que esto dice, y lo que no.** Dice que **no hay evidencia de que `lean`
degrade la calidad**: 14 de 16 ítems empatan, y los dos que se mueven van en la
dirección contraria a la sospecha. No dice que `lean` sea mejor: cada ítem tiene
**n = 1**, es una repetición de 8 tareas pequeñas, un solo juez, y ese juez
diseñó la variante que está juzgando —el ciego mitiga el conflicto de interés,
no lo elimina—. Para mover esa aguja hacen falta ≥6 parejas por ítem, o sea la
campaña de 3 repeticiones de §7 con el paquete ciego.

Lo que sí queda establecido es un método: **la calidad se puede juzgar sin
gastar un solo token de modelo**, porque los diffs y las respuestas están en
disco. Y una regla: ninguna campaña se reporta como completa si sus ítems
`human_review` siguen sin puntuar.

### 6.1.18 Las rúbricas, resueltas por programa donde se puede y por un juez ciego donde no

§6.1.17 juzgó 16 ejecuciones a ojo ciego. Eso está bien para lo que exige lectura
—si un informe es útil, si un hallazgo es real— y es innecesario para lo que un
programa puede decidir: si el arreglo cayó en la etapa equivocada o compensó en
el pipeline; si la aritmética de dinero usa división entera; si el run volvió a
invocar el comando que ya vio fallar. Juzgar eso a ojo, y encima siendo el autor
del cambio bajo revisión, es el instrumento equivocado.

`bench/agent_task_rubric_probes.py` (nuevo) resuelve **13 de los 16 ítems** con
evidencia del propio artefacto, y se rige por dos reglas:

* **Un probe que no ve, se abstiene.** ``(None, razón)`` en vez de un cero. Una
  abstención se reporta como abstención; nunca se promedia como un cero.
* **Un trace no es el registro del trabajo cuando el trabajo se delegó.** Ver
  §6.1.19.

Sobre `ab-generality` (32 ejecuciones, n = 2 por ítem):

| ítem | default | `lean` | delta |
| --- | ---: | ---: | ---: |
| `concise-handoff` | 0,50 | **1,50** | **+1,00** |
| `failure-recognition` | 1,50 | **2,00** | **+0,50** |
| `verification-reported` | 1,50 | **2,00** | **+0,50** |
| `assurance-scope`, `decision-ownership`, `existing-surface-preserved`, `immutability-discipline`, `no-float-drift`, `recovery-handoff`, `routine-scope`, `scope-discipline` | empate | empate | 0,00 |

**Dos instrumentos independientes coinciden.** Los probes deterministas mueven
los mismos dos ítems que el juicio ciego de §6.1.17 —`concise-handoff` y
`failure-recognition`— en la misma dirección, y agregan un tercero,
`verification-reported`. Nada se mueve a favor de `default`. Que un programa y
un juez ciego, sobre los mismos artefactos y sin compartir método, lleguen al
mismo lugar es una afirmación bastante más fuerte que cualquiera de las dos
sola.

Sobre `ab-engine-mode-fixed` (20 ejecuciones, el cambio de default de §6.1.8):

| ítem | `orchestrator` | `task` |
| --- | ---: | ---: |
| `located-the-stage` | **0,00** (no cambió nada) | **2,00** |
| `concise-handoff` | 1,00 | **0,00** (respuesta de 217 caracteres, el halt) |
| `decision-ownership` | **2,00** | — (sin artefacto que juzgar) |
| los otros 8 ítems medibles | empate | empate |

Las tres diferencias **son los dos defectos ya conocidos**, no calidad: el
orquestador no pudo aplicar el arreglo (§6.1.13) y `task` se detuvo antes de
producir el plan (§6.1.14). Ambos arreglados después. Eso deja una predicción
verificable para la próxima campaña: con los dos defectos corregidos, estas tres
filas deberían empatar.

### 6.1.19 Dos defectos más del harness, encontrados validando los probes

**El `tool_trace` sólo registra al principal cuando el trabajo se delegó.** En
modo `orchestrator` el trace contiene las llamadas del principal —y las
denegadas, con argumentos vacíos— pero **no** las del worker: ni las lecturas,
ni la edición, ni el `pytest`. La rúbrica formula varios ítems como *"the exact
tool trace shows…"*, y para esas ejecuciones el artefacto no contiene la
evidencia: la primera corrida de los probes le puso **0,00** al orquestador en
`assurance-scope` y `failure-recognition` por "no corrió tests" y "no invocó el
comando", cuando lo que faltaba era el registro. Corregido en el probe (abstiene
si aparece cualquier herramienta de equipo), no en la conclusión. Cualquier
criterio futuro que se apoye en el trace tiene que declarar lo mismo.

**Un `__pycache__` suelto se volvía parte del baseline de la tarea.** Los fixtures
no traen `.gitignore`, e `init_git_workspace` hacía `git add -A` sobre el copy:
dos fixtures tenían `__pycache__/` y `.pytest_cache/` de alguien que corrió
pytest adentro, esos `.pyc` quedaron **trackeados**, entraron al baseline, y el
run siguiente los recompiló y los reportó como cambios. El efecto medido, sobre
los artefactos guardados: en **15 ejecuciones** el diff del revisor contiene
secciones de caché compilada, y en las de `wide-sum` son el **91,5 %–95,5 %** del
payload —7 370 de 7 980 caracteres—. En las dos ejecuciones de `wide-sum` que
fallaron, el diff que recibió el revisor era **sólo** ruido binario.

Corregido en los tres lugares: los fixtures se limpiaron; `init_git_workspace`
escribe las exclusiones en `.git/info/exclude` (por checkout, nunca comiteado, y
sin agregarle un archivo al fixture); y el probe ignora cachés al contar archivos
cambiados. Tres tests, uno de ellos reproduciendo el `__pycache__` sucio y
comprobando que `git ls-files` no lo lista.

### 6.1.20 La última duda de validez, medida: `lean` en un repositorio que no se puede leer

El resultado principal de este documento es un recorte de rondas y de tokens, y su
riesgo declarado siempre fue el mismo: **¿y si `lean` deja de explorar?** Las 8
tareas donde se midió tenían el arreglo en un archivo que el pedido nombraba, y
la sonda de exploración (`wide-sum`) eran 40 módulos en un directorio — se leen
en 40 llamadas y no prueba nada sobre la navegación.

**La sonda nueva.** Corpus `engine_eval_v9` y fixture `deep_repo`, generados por
`bench/build_deeprepo_fixture.py` (determinista, regenerable):

```
8 hubs  ->  40 paquetes  ->  1 200 hojas          (1 256 archivos)
```

Cada hoja aplica un offset fijo positivo; los paquetes y los hubs sólo los
encadenan. Una hoja miente (`v - 40` en lugar de `v + 9`), así que
`pipeline(0)` devuelve **14 349** donde el pedido declara **14 398**. El defecto
**no es greppable a propósito**: 52 hojas comparten el offset 9, de modo que el
faltante de 49 nombra una magnitud y no un lugar. Los tests visibles **pasan**
sobre el código roto, y `src/pipeline.py`, los hubs, los `pkg_*/__init__.py` y
`tests/` están prohibidos como destino del arreglo: sólo una hoja puede cambiar.
Preflight: el fixture pristino falla (rc 1), la solución de referencia arregla
(rc 0) tocando un solo archivo.

**Resultado, 3 repeticiones × 2 brazos**, `pipeline_mode: true`,
`TASK_ENGINE_MODE: task`, presupuesto de 120 llamadas
(`bench/runs/20260913-engine-v2/ab-deeprepo/comparison.md`):

| métrica | default | `lean` | delta | parejas |
| --- | ---: | ---: | ---: | --- |
| success | **3/3** | **3/3** | — | — |
| tokens facturados | 990 394 | 663 605 | **−33,0 %** | 2 / 1 |
| rondas de modelo | 39 | 31 | −20,5 % | 2 / 1 |
| latencia | 188,9 s | 213,3 s | +12,9 % | 1 / 2 |
| llamadas de herramienta | 32 | 34 | +6,2 % | 1 / 1 |

Con 3 parejas nada se resuelve, y así queda declarado: esto es una **sonda de
validez, no una estimación de efecto**. Lo que la sonda tenía que contestar, lo
contestó: **`lean` no pierde exploración.** Las seis ejecuciones resolvieron un
repositorio de 1 256 archivos, y las dos trazas navegan igual —
~20–23 llamadas, 4–11 lecturas con `mod_` en los argumentos, y **ninguna lee el
árbol**. El probe `localized-by-bisecting` da 2,00 en los dos brazos, 3/3.

**Y apareció una diferencia de calidad, en el lugar menos esperado.** El probe
`located-the-leaf` da **default 1,33 contra `lean` 2,00** (3/3). La causa está en
una sola ejecución: `default` r2 cambió una hoja **conforme** —`return v + 9` a
`return v + 58`— en lugar de corregir la que miente. Eso restaura el total, pasa
el verificador, y no es el arreglo que el pedido describe ("exactly one leaf has
the wrong offset. Find it, fix it"). `lean` corrigió `- 40` a `+ 9` en las tres.

Es exactamente el modo de falla que el ítem de rúbrica existe para detectar, y
ningún verificador sobre un total puede verlo. El probe tuvo que afilarse dos
veces para verlo: primero porque contaba archivos temporales fuera del workspace
(tres ejecuciones de `lean` dejaron un `find_bad.py` en `/private/tmp`, que el
parser de diffs no distingue de una edición), y después porque "un solo archivo
de hoja cambió" no distingue *corregir la hoja rota* de *reescribir una sana*.
Ahora la línea **eliminada** es la que decide, y los cuatro casos están testeados.

**Lo que esto cierra.** El punto 2 de §7 queda hecho: `lean` está medido en un
repositorio que no se puede leer, y no pierde nada. Lo que queda abierto es el
costo: ~660 000–990 000 tokens por ejecución en un fixture de 1 256 archivos, el
décimo más caro que `wide-sum`. La localización jerárquica funciona; el engine
no tiene una forma barata de *decirle* al modelo que la jerarquía existe, y eso
es una palanca que este documento no explora.

### 6.1.21 El router se puede medir aislado, y medido no se paga

Dos campañas dejaron esta pregunta abierta: el A/B del catálogo de políticas
(§6.1.3) dio −11,5 % de tokens y **+13,6 % de latencia**, y el A/B del
clasificador (§6.1.15) no resolvió nada con 8 parejas. Las dos comparaban
ejecuciones completas, donde la varianza del modelo es del mismo orden que el
efecto. Pero el router corre **antes** del loop, una vez por turno, sobre el
pedido crudo: se puede cronometrar solo.

`bench/task_policy_router_cost.py` (nuevo) lo hace sobre las 11 requests del
corpus, con los tres modos, midiendo además el uso del proveedor con el mismo
callback de litellm del §6.1.8:

| modo | mediana por turno |
| --- | ---: |
| `off` | **0,004 s** |
| `fallback` | **0,002 s** |
| `preferred` | **2,912 s** |

Y el detalle que ninguna campaña podía dar:

* el ruteo local —parser literal, mini-head, retrieval contrastivo— cuesta
  **4 ms**: es gratis;
* `preferred` cuesta **2,9 s por turno**, hasta **12,1 s** en `wide-sum`, porque
  emite una request al modelo principal;
* esa request gasta **446 tokens de prompt por turno** (15 602 en total, más
  6 784 de completion) que **no llegan a ningún contador que el engine reporte**;
* y en 2 de las 11 requests `preferred` **quitó** la etiqueta `feature` que el
  texto justificaba, porque `merge_llm_result` se llama con
  `replace_classified_operations=True`.

**Decisión: `TASK_POLICIES_LLM_CLASSIFIER_MODE` pasa de `"preferred"` a
`"fallback"`.** No hace falta campaña: es una medición de mecanismo, y el
mecanismo es un costo serial por turno sin beneficio medido —el A/B de 8 parejas
no resolvió ninguno— que además sólo *restaba*. `fallback` conserva la capacidad
(el LLM sigue disponible cuando el parser literal no encuentra nada) y es
**gratis en 10 de las 11 requests**, porque sólo corre en las ambiguas. Dos tests
fijan el mecanismo: que `off` y `fallback` no toquen el proveedor cuando hay
método literal, y que `preferred` con una respuesta vacía borre `feature`
mientras `fallback` lo conserva.

**Lo que esto no explica.** El +13,6 % de latencia del A/B del catálogo no es
esto: el clasificador corre en los dos brazos, así que no puede producir una
*diferencia* entre ellos. La contradicción del catálogo sigue abierta y sigue
siendo el punto 1 de §7.

### 6.1.22 El corpus entero puntuado: 295 ejecuciones, 15 ítems

Los probes de §6.1.18 se corrieron sobre dos campañas. El modo `--scorecard`
los corre sobre **todo lo guardado**: 295 ejecuciones, 15 ítems de rúbrica
decididos por programa, cero llamadas al modelo
(`bench/runs/scorecard-all.md`).

| ítem | n | media | cumple | parcial | falla |
| --- | ---: | ---: | ---: | ---: | ---: |
| `immutability-discipline` | 30 | **2,00** | 30 | 0 | 0 |
| `no-float-drift` | 24 | **2,00** | 24 | 0 | 0 |
| `routine-scope` | 17 | **2,00** | 17 | 0 | 0 |
| `scope-discipline` | 25 | **2,00** | 25 | 0 | 0 |
| `localized-by-bisecting` | 6 | **2,00** | 6 | 0 | 0 |
| `localized-without-reading-everything` | 13 | **2,00** | 13 | 0 | 0 |
| `existing-surface-preserved` | 31 | 1,97 | 30 | 1 | 0 |
| `decision-ownership` | 24 | 1,96 | 23 | 1 | 0 |
| `verification-reported` | 24 | 1,92 | 22 | 2 | 0 |
| `recovery-handoff` | 31 | 1,90 | 28 | 3 | 0 |
| `located-the-stage` | 13 | 1,85 | 12 | 0 | 1 |
| `failure-recognition` | 29 | 1,83 | 24 | 5 | 0 |
| `assurance-scope` | 37 | 1,76 | 28 | 9 | 0 |
| `located-the-leaf` | 6 | 1,67 | 5 | 0 | 1 |
| `concise-handoff` | 25 | **1,28** | 11 | 10 | 4 |

Dos cosas se leen acá. **Seis ítems son perfectos en el corpus entero**: nunca
se mutó el receptor en `cart-immutability`, nunca se redondeó con float, nunca
se tocó un archivo de más en `user-owned-tradeoff`, nunca se cambió la
convención sin declararla, nunca se leyó el árbol en las dos tareas de
localización. Y **`concise-handoff` es el único ítem flojo**: media 1,28 contra
1,67 del siguiente, con 4 fallas plenas. Es el traspaso del plan
(`complex-plan`): la respuesta final nombra el comando de verificación en 18 de
25 casos, apunta a las decisiones abiertas en 14 de 25, y queda bajo 1 600
caracteres en 13 de 25. Ningún componente explica la falla por sí solo.

**El instrumento también se equivocó, y hay que decirlo.** La primera corrida
dio `concise-handoff` en **0,84**, no 1,28. La causa era un `\b` después de
`open decision`: el patrón exigía el singular, y los modelos escriben "Open
decisions now owned by the user". Once de las 25 ejecuciones estaban bien y el
probe las contaba mal. Lo mismo con `exit 0` contra `exits 0` y `exit status: 0`
en `_RESULT_IN_ANSWER`. Corregidas las dos expresiones, **ninguna comparación
entre brazos cambia** —`concise-handoff` sigue en 0,50 contra 1,50 en
`ab-generality`, `located-the-leaf` en 1,33 contra 2,00— y el puntaje del
corpus sube. Es la tercera vez en el proyecto que el instrumento, y no el
engine, era el defectuoso; por eso el probe devuelve la evidencia junto con el
puntaje y el scorecard imprime `n`, `cumple`, `parcial`, `falla` y `abstención`
en columnas separadas.

**El número que vale para el engine enviado** se obtiene filtrando
(`bench/runs/scorecard-shipped.md`): `--style lean --engine-mode task` deja 60
ejecuciones, y ahí **14 de los 15 ítems dan 2,00** —`assurance-scope`,
`decision-ownership`, `existing-surface-preserved`, `failure-recognition`,
`immutability-discipline`, `localized-by-bisecting`,
`localized-without-reading-everything`, `located-the-leaf`, `located-the-stage`,
`no-float-drift`, `recovery-handoff`, `routine-scope`, `scope-discipline`,
`verification-reported`— y el único que no es `concise-handoff` con **1,40**
(3 cumple, 1 parcial, 1 falla).

### 6.1.23 El titular, re-medido con el contador corregido

§6.1.8 comparó los dos modos de engine con el contador del `LoopEngine`, que
omite los loops de los workers en un brazo y las fases del pipeline en el otro.
§6.1.8 dejó el −82,9 % como **piso** a la espera de re-medirlo con el contador
que cuenta lo que el proveedor factura. Esta es esa medición.

**4 tareas × 2 repeticiones × 2 brazos** (16 ejecuciones), `pipeline_mode: true`,
`prompt_style: lean`, `TASK_ENGINE_MODE` como única diferencia
(`bench/runs/20260913-engine-v2/ab-accounted-full/comparison.md`). Las tareas
cubren las cuatro formas donde los modos difieren: planificar (`complex-plan`),
arreglar código (`cart-immutability`), leer y documentar
(`evidence-code-review`) y localizar a escala (`wide-sum`).

| métrica | `orchestrator` | `task` | delta | parejas | p | resuelto |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| success | 8/8 | 8/8 | — | — | — | — |
| **tokens de prompt facturados** | 517 468 | **98 084** | **−81,0 %** | 8 / 0 | 0,0078 | **sí** |
| tokens de prompt del loop | 373 054 | 85 980 | −77,0 % | 8 / 0 | 0,0078 | **sí** |
| tokens de completion facturados | 23 623 | 5 790 | −75,5 % | 8 / 0 | 0,0078 | **sí** |
| **rondas de modelo** | 31 | **11** | **−64,5 %** | 8 / 0 | 0,0078 | **sí** |
| llamadas de herramienta | 24 | 8,5 | −64,6 % | 8 / 0 | 0,0078 | **sí** |
| **latencia** | 292,6 s | **79,0 s** | **−73,0 %** | 8 / 0 | 0,0078 | **sí** |
| `changed_lines` | 106,5 | 97,5 | −8,5 % | 4 / 3 | 1,0 | no |
| `introduced_placeholders` | 0 | 0 | — | — | — | — |
| `extra_changed_files`, `malformed_tool_calls`, `max_workless_rounds` | 0 | 0 | — | — | — | — |

**Corregido en §6.1.38:** los seis p-valores de 8/0 no cambian, porque el test
defectuoso sólo se equivocaba cuando había parejas en contra. `changed_lines`
pasa de 0,375 a 1,0.

**La corrección agranda el efecto, no lo encoge.** El contador del loop daba
−77,0 % y el facturado −81,0 %, porque la omisión del orquestador es mayor: su
relación facturado/loop es **1,21–1,75× (media 1,37)** contra **1,09–1,29×
(media 1,16)** del lado de `task`. Era la predicción de §6.1.8 y se confirmó.

Y las métricas de calidad **no se mueven**: líneas cambiadas, placeholders
introducidos, archivos de más y llamadas malformadas quedan en cero o empatados.
Los probes de rúbrica sobre estas 16 ejecuciones dan `located-the-stage` **2,00
contra 2,00** —el arreglo cae en la etapa correcta en los dos brazos, que es el
efecto de §6.1.13—, `immutability-discipline`, `existing-surface-preserved` y
`decision-ownership` en 2,00 los dos.

**La única diferencia de calidad, y es del orquestador.** `concise-handoff` da
**2,00 contra 0,50** (2 parejas). Las respuestas finales de `complex-plan` en
modo `task` no cierran con el comando de verificación seguido de lo que queda
abierto. El contrato de `lean` **ya lo pide con esas palabras** —
`prompts/variants/lean.py:100-105`: "the outcome in one or two sentences, then a
`Verification:` line …, then what is not done and any decision that belongs to
the user. Under 250 words"— así que no es una instrucción faltante: es
cumplimiento. Sobre las 25 ejecuciones de `complex-plan` del corpus, la respuesta
nombra el comando de verificación en 18, apunta a las decisiones abiertas en 14,
y queda bajo 1 600 caracteres en 13 (§6.1.22).

Lo que corresponde, entonces, no es otro párrafo en el prompt sino **hacer que el
engine lo note**: el contrato de cierre ya rechaza un `step_complete` que no
produjo un cambio, y desde §6.1.13 ese rechazo se le dice al modelo en vez de
tragárselo. Un aviso equivalente —"tu respuesta final no nombra el comando que
corriste ni lo que queda abierto"— es la palanca, y queda anotada en §7 en lugar
de improvisada al final de esta ronda.

### 6.1.24 El traspaso flojo: implementado, medido inerte, y el proxy de longitud que lo inflaba

§6.1.22 dejó `concise-handoff` como el único ítem flojo (1,28 contra 1,67 del
siguiente) y §6.1.23 lo dio como la única diferencia de calidad entre modos
(2,00 contra 0,50). La conclusión parecía obvia: el engine debería **notar** que
una respuesta `done` no nombra ningún comando y pedirlo una vez. Está
implementado (`LOOP_HANDOFF_NOTICE_ENABLED`, contador propio, tope de **una**
negativa por Task, nunca terminal: es un defecto de redacción, no trabajo
faltante).

**Y medido, no dispara.** A/B `handoff-off` contra `handoff-on`, 3 repeticiones ×
`complex-plan`, mismo config y mismo día
(`bench/runs/20260913-engine-v2/ab-handoff/comparison.md`): **nada resuelto** en
ninguna métrica (1/1 parejas en tokens, 2/0 en latencia) y, lo importante, **el
aviso no se disparó en ninguna ejecución del brazo `on`**: las cuatro respuestas
completas del experimento nombran un comando de verificación.

La razón es que la brecha que yo había medido era **histórica**. El "7 de 25 sin
comando" de §6.1.22 viene del corpus entero, dominado por ejecuciones anteriores
a `lean`, en modo `orchestrator` y de la era previa a los arreglos. En la
configuración enviada, `complex-plan` nombra un comando en **9 de 12** corridas, y
las 3 que no lo hacen **no llegan al loop**: dos son halt y una es la falla de
§6.1.25. Ninguna compuerta dentro del loop puede arreglar una corrida que nunca
corrió.

**Y el proxy estaba mal.** Descomponiendo el ítem en sus tres componentes sobre
las 12 corridas de la configuración enviada:

| componente | cumple |
| --- | ---: |
| nombra comando y resultado | 9 / 12 |
| apunta a lo que queda abierto | 10 / 12 |
| **bajo 1 600 caracteres** | **5 / 12** |

El componente que más fallaba era **la longitud**, que no es lo que la rúbrica
pide: pide "sin narrar cada paso". Leí once respuestas largas del corpus y **no
narran pasos**: son resúmenes con el resultado primero, las decisiones abiertas
con dueño y default, las fases, el rollout, el rollback y el resultado de la
verificación — es decir, exactamente el contenido que la tarea pidió. Un traspaso
de un plan legítimamente ocupa 2 000–3 500 caracteres. El límite de 1 600 era
mío, no del engine.

Corregido: la longitud se **reporta** como evidencia y no se puntúa; el ítem
puntúa los dos componentes que la rúbrica nombra sin ambigüedad. Efecto sobre el
corpus entero (35 ejecuciones): `concise-handoff` pasa de **1,28 a 1,74** —de
outlier a la mitad del pelotón— y sobre la configuración enviada (13
ejecuciones) de 1,40 a **1,62**, con las dos únicas fallas plenas siendo las dos
corridas que no corrieron. `ab-generality` sigue dando `lean` por delante (2,00
contra 1,50) con la mitad del margen anterior.

Es la **cuarta** vez en el proyecto que el instrumento, y no el engine, era el
defectuoso. El aviso de traspaso se queda igual: es correcto, está acotado, no
puede terminar un run y protege la propiedad en vez de mejorarla.

### 6.1.25 Una promesa no es una entrega

Buscando por qué el brazo `on` había fallado una corrida apareció lo peor que
puede aparecer en este corpus: **un run que termina con una promesa.**

```
final_answer: "I will draft PLAN.md now, scoped to the tenant export change in
requirements.md, and then run verify.py to confirm it passes."
verify_exit_code: 1        changed_paths: []
request_payload_history: 0 rounds        tool_trace: []
```

Cero rondas del loop, cero llamadas de herramienta, ningún archivo creado. El
**chat agent** —que es de sólo lectura— decidió que el pedido era conversacional,
respondió con lo que iba a hacer, y el pipeline devolvió esa promesa como la
respuesta del turno. El usuario lee que el trabajo se va a hacer; el trabajo no se
hace.

**Frecuencia, medida sobre las 318 ejecuciones guardadas**: 2 terminaron sin que
el loop corriera una sola ronda. Una es el halt de elaboración de §6.1.14; la
otra es esta. Es raro y es la peor forma posible: las otras fallas al menos fallan
a la vista.

**El arreglo es determinista, no un párrafo más.** El prompt del chat agent ya
dice que hay que escalar ante un pedido de ejecución directa, así que no faltaba
la instrucción: faltaba la consecuencia. `promises_instead_of_working()`
(`engine/orchestration/chat_agent.py`) detecta una respuesta que **abre** con un
compromiso en primera persona seguido de un verbo de ingeniería, y el pipeline
convierte ese `respond` en un `escalate` —el paquete que el pedido debió generar
desde el principio—. Es conservador por construcción: exige que el compromiso
abra la respuesta y que la primera oración nombre trabajo de ingeniería, así que
"I will explain why the cache is stale" y las respuestas largas quedan intactas.
Sin interruptor: la alternativa al arreglo es la falla.

### 6.1.26 La contradicción del catálogo: resuelta como ruido, con los números de la propia campaña

Quedaba una sola contradicción medida sin explicar. El A/B del catálogo de
políticas (§6.1.3) dio, en el modo que se envía, **−11,5 % de tokens de prompt**
(6/2, p=0,2891 tras §6.1.38; 0,031 con el test defectuoso) junto con
**+13,6 % de latencia** (1/7), y las dos cosas no pueden
venir del mismo cambio si el cambio es *menos prompt*.

**La respuesta estaba en la campaña, sin correr nada nuevo.** Desagregando las 8
parejas por tarea y por ronda:

| tarea | rep | tokens all → sel | llamadas | latencia |
| --- | ---: | ---: | ---: | ---: |
| `cart-immutability` | 0 | 78 332 → 104 158 | 7 → 9 | 60 → 86 |
| `cart-immutability` | 1 | 87 267 → 144 623 | 11 → 13 | 86 → 132 |
| `evidence-code-review` | 0 | 95 546 → 84 237 | 5 → 5 | 143 → **304** |
| `evidence-code-review` | 1 | 99 161 → 77 915 | 9 → 7 | 124 → 205 |
| `options-override` | 0 | 75 780 → 53 718 | 9 → 9 | 40 → 64 |
| `options-override` | 1 | 87 797 → 68 668 | 9 → 9 | 38 → 46 |
| `tool-failure-recovery` | 0 | 75 808 → 54 879 | 9 → 10 | 65 → 85 |
| `tool-failure-recovery` | 1 | 68 904 → 52 962 | 7 → 7 | 87 → 60 |

En **cinco de las ocho parejas la latencia empeora mientras los tokens bajan y las
rondas no suben** — y el caso que domina la mediana es
`evidence-code-review`, que duplica su latencia (143 → 304 s) **con el mismo
número de llamadas y menos tokens**. Ningún recorte de 3,7 KB de prompt produce
eso. Los tokens de completion tampoco lo explican: son similares (±30 %) y los
segundos por cada mil tokens generados suben igual, o sea que el mismo volumen de
salida tardó más — variabilidad del proveedor, no del cambio.

Y la escala del efecto lo cierra:

| | latencia |
| --- | ---: |
| diferencia entre medianas (all → selected) | **+10,3 s** |
| dispersión dentro de `all` (máx − mín) | 104,8 s |
| dispersión dentro de `selected` | 258,1 s |

La diferencia **entre** brazos es **10 a 25 veces menor** que la dispersión
**dentro** de un brazo. Con eso, con p=1,0 del propio test de signos, con la
dirección contraria a la del único mecanismo posible, y con la prueba de
mecanismo de §6.1.21 mostrando que el clasificador corre en los dos brazos, la
conclusión no es "no lo sabemos": es que **el +13,6 % no es un efecto**.

**Y una réplica independiente lo confirma, con la latencia al revés.** Las mismas
4 tareas × 2 repeticiones, corridas de nuevo
(`bench/runs/20260913-engine-v2/ab-policy-pipeline2/comparison.md`):

| | muestra original (8 parejas) | réplica (8 parejas) |
| --- | ---: | ---: |
| tokens de prompt facturados | −10,4 % (11/7, p=0,4807) | **−25,6 %** (7/1, p=0,0703) |
| latencia | **+13,6 %** (1/7, p=1,0) | **−36,7 %** (5/3, p=0,06) |
| success | 8/8 vs 8/8 | 8/8 vs 8/8 |
| `changed_lines`, placeholders, archivos de más | 0 | 0 |

**Dos muestras del mismo contraste dan signos opuestos en latencia.** Eso es lo
que se ve cuando no hay efecto: el ahorro de tokens replica con el mismo signo y
más grande con el contador corregido, y la latencia se mueve −13,6 % en una
muestra y +36,7 % en la otra. (La réplica corre además con el clasificador ya en
`fallback` (§6.1.21), así que su latencia absoluta no es comparable con la del
original; lo que importa es que la *dirección* de la diferencia está en duda, no
que haya cambiado el nivel.)

Queda entonces: **el catálogo de políticas condicionales se paga —−11,5 % y
−25,6 % de tokens en dos muestras, con 16/16 de éxito— y su costo en latencia no
es medible con este diseño.** La contradicción que §6.1.3 dejó abierta no era un
efecto sin explicar: era ruido con un mecanismo ausente, y ahora tiene la prueba
de que se mueve en las dos direcciones.

### 6.1.27 El residuo de rúbricas: cuatro probes más, y dos que hubo que sacar

§6.1.22 dejó 269 instancias de rúbrica sin decidir. Leí las descripciones de las
nueve y separé las que tienen **evidencia de resultado** de las que tienen
**presencia de frases**:

* `routine-autonomy` ("sin bloquear pidiendo input") — el resultado es el
  cambio: si el run tocó el archivo de implementación, actuó; si no cambió nada y
  la respuesta es una pregunta, bloqueó.
* `no-regression` ("el camino por defecto sigue igual, y la respuesta nombra el
  comando que lo mostró") — las dos mitades ya se verifican en otro lado: el
  verificador oculto ejerce el camino por defecto, y la respuesta nombra un
  comando o no.
* `diagnosed-the-right-layer` ("el defecto se encuentra donde el merge está mal")
  — qué archivo cambió, igual que `located-the-stage`.
* `independent-questions` ("los tres módulos respondidos por separado") — si el
  informe nombra los tres archivos.

Los cuatro son de resultado y quedaron. **Los dos de frases los saqué**, y por
qué conviene registrarlo:

`verification-interpretation` daba **0,05** —37 de 39 en cero— y era el probe:
exigía un verbo de una lista corta (`proves|confirms|shows|…`) que no cubre
"confirmed", "verified", "passed". `recommendation-calibration` daba 0 a tres
respuestas que **literalmente dicen** *"Recommends **Atlas** when predictable
cost is the priority"* y 1 a otras dos que dicen *"asks you to declare whether
cost predictability or lowest latency should govern"*: el marcador buscaba
`if you`/`which priority` y el idioma real usa *"when X is the priority"* y
*"asking you to choose the decisive priority"*. El mismo error que el `\b` de
`open decisions` de §6.1.22, pero esta vez en **las dos direcciones**.

La regla que queda: **un probe vale cuando la evidencia es el resultado, no
cuando es el vocabulario.** Decidir si una recomendación es condicional es un
juicio sobre el sentido; el paquete ciego lo hace barato y lo hace un lector.

De paso apareció el mismo defecto de `.ken` de §6.1.10 dentro de los probes: en
ejecuciones viejas `changed_paths` incluye los archivos del índice de Ken, así que
`diagnosed-the-right-layer` le daba 0 a un run que había arreglado exactamente
`src/config.py`. Los paths ignorados ahora incluyen `.ken` y `.infinidev`.

**Estado del corpus, 333 ejecuciones y 666 instancias:**

| | instancias | cobertura |
| --- | ---: | ---: |
| decididas por programa | **486** | **73 %** |
| residuo de juicio | 180 | 27 % |

El residuo son cinco ítems que piden leer: `evidence-depth` (54),
`report-usability` (54), `verification-interpretation` (39),
`recommendation-calibration` (17) y `findings-are-real` (16).

Y el perfil que resulta, sobre el corpus entero, con **19 ítems**:

| ítem | media | cumple / parcial / falla |
| --- | ---: | --- |
| `immutability-discipline`, `no-float-drift`, `routine-autonomy`, `routine-scope`, `scope-discipline`, `independent-questions`, `localized-by-bisecting`, `localized-without-reading-everything` | **2,00** | 0 fallas |
| `decision-ownership` | 1,97 | 32 / 1 / 0 |
| `existing-surface-preserved` | 1,97 | 38 / 1 / 0 |
| `diagnosed-the-right-layer`, `no-regression`, `verification-reported` | 1,92 | 1 falla cada uno |
| `recovery-handoff` | 1,91 | 32 / 3 / 0 |
| `located-the-stage` | 1,88 | 16 / 0 / 1 |
| `failure-recognition` | 1,84 | 26 / 5 / 0 |
| `assurance-scope` | 1,76 | 28 / 9 / 0 |
| `concise-handoff` | 1,74 | 28 / 5 / 2 |
| `located-the-leaf` | 1,67 | 5 / 0 / 1 |

En la **configuración enviada** (`lean` + `task`, 90 ejecuciones) **los 19 ítems
dan 2,00 salvo `concise-handoff` en 1,62**, y sus dos fallas plenas son las dos
corridas que nunca corrieron —el halt de §6.1.14 y la promesa de §6.1.25—.

### 6.1.28 El residuo, leído: `recommendation-calibration` es 17/17

§6.1.27 dejó 180 instancias de rúbrica que piden juicio humano y sacó dos probes
por estar mal en las dos direcciones. Este es el primer tramo leído de verdad.

**El método, para que sea barato.** El paquete ciego creció con dos opciones:
`--item` para armar un paquete con un solo ítem y `--narrow` para emitir sólo la
respuesta y los comandos que corrieron, sin diff ni traza. El paquete de
`recommendation-calibration` son 17 ejecuciones en 845 líneas: se lee de una
sentada y el brazo sigue ciego. (De paso, dos defectos del propio paquete: el
filtro de ítems no se aplicaba porque el parámetro quedaba sombreado por una
variable local del mismo nombre, y el recorrido sólo miraba un nivel de
profundidad, así que las campañas con y sin nivel de brazo no se veían todas.
Los dos corregidos, con el brazo etiquetado por su ruta.)

**El resultado: 17 de 17 en 2,00.** Las 17 respuestas nombran el eje de decisión
—costo predecible contra latencia— y o bien emparejan las opciones
condicionalmente ("Choose **Atlas** if predictable monthly cost is more important
than …; **Comet** if the lowest measured p99 latency is more important"), o bien
delegan explícitamente la prioridad ("The decisive priority … is a
product/business preference only you can set"). Varias hacen las dos cosas y
además ofrecen una salida intermedia con umbral.

Y es uniforme: **2,00 en cada campaña y cada brazo**, incluidas las tres
ejecuciones del piloto de agosto —`luna`, `sol`, `terra`— anteriores a todos los
arreglos de estas rondas. Es decir, esta conducta **siempre estuvo bien**, que es
exactamente por lo que el probe de §6.1.27 que la puntuaba en 1,47 era tan
peligroso: no medía una deficiencia, medía mi vocabulario.

Queda: `verification-interpretation` (39), `evidence-depth` (54),
`report-usability` (54) y `findings-are-real` (16). El paquete angosto ya está
armado para el primero; los dos del medio necesitan el diff y la traza, así que
van con el paquete completo.

### 6.1.29 El residuo, leído (II): `verification-interpretation` no es decidible desde el artefacto

Leí las 39 respuestas de `test-selection`. El ítem pide dos cosas: que la
respuesta diga **qué prueban** los tests y **cuál es el límite** que queda.

**La primera mitad: 36 de 39.** Todas nombran los tests que corrieron, y ocho
además mapean cada cláusula del contrato al test que la establece. Las tres que
no —puntaje 0— sólo informan un conteo (*"Focused tests: 4 passed"*), y las tres
son del piloto de agosto. Desde las campañas de septiembre, **las 36 dan 1**.

**La segunda mitad: 0 de 39, y eso puede estar bien.** Ninguna respuesta nombra un
límite. Pero al leerlas se ve por qué: el cambio es `normalize_tags` y los cuatro
tests del fixture cubren exactamente las cuatro cláusulas del contrato —colapso
por caso, orden, no mutación, tipo de retorno—. **No hay superficie sin ejercer,
así que no hay límite que nombrar**, y una respuesta que inventara uno estaría
agregando ruido.

Eso hace que el ítem **no sea decidible desde el artefacto**: no puedo distinguir
"lo omitió" de "lo omitió correctamente". Es el tercer ítem que devuelvo al
juicio en vez de a un probe, y esta vez por una razón distinta a las anteriores:
no es que mi vocabulario sea corto, es que el criterio depende de un hecho —si el
cambio está cubierto del todo— que el artefacto no registra.

**Y el intento de comprarlo con prompt falló.** Agregué una cláusula condicional
al contrato de `lean` ("cuando el chequeo deja algo que el cambio toca sin
ejercer, dónde se detiene") y corrí 3 repeticiones antes y 3 después, mismo
config (`lean` + `task`, `test-selection`):

| | completadas | nombra un límite |
| --- | ---: | ---: |
| antes | 2/3 | 0 |
| después | **3/3** | **0** |

Las tres completadas después tampoco nombran límite, y por la razón de arriba: no
había ninguno. **La cláusula está revertida** — el archivo quedó byte-idéntico al
commit— porque no hay evidencia de que compre nada y una instrucción que no
produce salida es peor que no tenerla. Lo único que sí se midió en ese par de
corridas es el defecto de §6.1.30: 2 de 3 antes contra 3 de 3 después.

### 6.1.30 Un argumento con forma de diccionario tumbaba el turno entero

Corriendo lo anterior, una de cada tres ejecuciones murió así:

```
final_answer: The task engine failed: AttributeError: 'dict' object has no attribute 'strip'
request_payload_history: 0 rondas        changed_paths: []
```

`_build_respond` hacía `(args.get("message") or "").strip()`. Cuando el modelo
responde con `{"message": {"text": "..."}}` —una forma que el dispatcher se supone
que absorbe— el `AttributeError` sale del chat agent, atraviesa `run_task` y llega
al usuario como el texto del turno. **El turno entero se pierde por la forma de un
argumento**, que es exactamente la clase de falla que §4.4 y el contador de
llamadas malformadas existen para prevenir: no fue una llamada malformada
registrada, fue una excepción.

El barrido encontró **ocho sitios** con la misma forma, no uno:
`chat_agent.py` (`message`, `understanding`, `user_visible_preview`,
`user_signal`) y `spec_elaborator.py` (`question`, `default`, `impact`, `risk`).
`text_argument()` centraliza la coerción: una cadena pasa; un diccionario se busca
por las claves que los modelos realmente anidan (`text`, `message`, `content`,
`reply`, `summary`, `value`); un escalar se convierte; cualquier otra forma
devuelve vacío, que los llamadores ya tratan como "el modelo no respondió" y
resuelven con su camino de reserva. Tres tests, incluidos los dos que construyen
la llamada con forma de diccionario y comprueban que ya no tumba el turno.

### 6.1.31 El residuo, leído (III): `findings-are-real` y el defecto que el fixture ya había arreglado

Tercer tramo: `findings-are-real` (16 ejecuciones de `research-audit`), "cada
hallazgo nombra un defecto que está de verdad en la línea citada, en vez de
reformular para qué sirve el módulo".

**Siete de las dieciséis no produjeron informe** —seis murieron contra el `429`
de cuota del §6.1.9 y una es la corrida del orquestador cuyo ticket nunca se
resolvió (§6.1.13)— así que abtienen. Sobre las nueve que sí:

| | |
| --- | ---: |
| media | **1,89** |
| con todos los hallazgos reales | 8 |
| con una cita corrida de línea | 1 |
| inventando un defecto o reformulando el módulo | **0** |

Cero invenciones es el resultado que importa, y no era obvio. El fixture tiene una
trampa: `auth.py:14` usa `hmac.compare_digest`, que es **correcto**, mientras que
el fixture del piloto de agosto (`code_review`) usaba `supplied_token ==
stored_token` con un `except: return True` que abre la puerta. Un informe que
arrastrara el patrón del piloto acusaría una comparación insegura donde no la hay.
**Cinco de las nueve respuestas lo dicen explícitamente** —*"the
`hmac.compare_digest` on `src/auth.py:14` is the correct primitive"*— y ninguna lo
acusa.

El único puntaje 1 es una cita corrida: un informe atribuye al `line 14` la
comparación de frescura que está en el `line 16`, aunque acierta el defecto y
aclara en la línea anterior que el 14 es timing-safe.

### 6.1.32 La dieta de esquemas, medida por tercera vez: no hay premio

Dejé este punto abierto dos veces. La tercera medición lo cierra, y esta vez con
la distribución en vez de un promedio.

| caracteres de esquema en el payload | rondas |
| ---: | ---: |
| **4 069** | **1 654** |
| 20 134 | 594 |
| 1 738 | 529 |
| 46 395 | 526 |
| 19 929 | 498 |
| 17 303 | 231 |

**La mayoría de las rondas manda 4 KB de esquema, no 46 KB.** El "48 % del
payload" que afirmé en la ronda 6 era cierto para los payloads grandes y falso
como generalización. Y el conjunto completo medido hoy son 59 279 caracteres
(47 110 en modo compacto), ninguna de las cuales coincide con las cifras de los
payloads: el motor ya manda conjuntos distintos según la ejecución, así que no hay
un "esquema del engine" que dietar.

Sobre las herramientas: 43 nombres distintos llamados en 330 ejecuciones, con
`read_file` (1 280), `execute_command` (1 028), `list_directory` (419),
`edit_file` (265) y `add_step` (234) cubriendo casi todo, y 36 registradas sin una
sola llamada. **Tampoco justifica una dieta**: el `tool_trace` no registra los
pseudo-tools —`step_complete` figura como nunca llamada y se llama en cada
cierre—, el conjunto varía por ejecución, y "nunca usada en 12 formas de tarea
sobre fixtures diminutos" no es "nunca útil". En un repositorio real
`rename_symbol` es la herramienta correcta, y ocultarla sería un cambio de
capacidad con falla silenciosa justificado por un corpus que no la mide.

### 6.1.33 El único loop sin tope de resultado era el más largo

Buscando por qué una corrida de escala gasta un millón de tokens encontré esto:
**cada loop del engine menos uno pasa `max_chars` a `handle_oversized_result`.**

| loop | tope |
| --- | ---: |
| `analysis/planner.py` | 8 000 |
| `analysis/stage_planner.py` | 8 000 |
| `orchestration/chat_agent.py` | 8 000 |
| `analysis/spec_elaborator.py` | 6 000 |
| `council/agent_loop.py` | 6 000 |
| **`loop/tool_runner.py` (el developer)** | **ninguno** |

El loop con el horizonte más largo era el único sin tope. Un `read_file` de
42 770 caracteres entraba al prompt tal cual y **se reenviaba en cada ronda
posterior**. La medición sobre las ejecuciones guardadas: 37 de 4 109 resultados
(0,90 %) pasan los 8 000 caracteres, así que el tope es **inerte en el 91 % de las
ejecuciones** y grande donde pega:

| | |
| --- | ---: |
| ahorro mediano del payload acumulado | **0,0 %** |
| ahorro máximo | **44,5 %** |
| ejecuciones que ahorran más de 5 % | 12 |
| ahorro agregado sobre las 29 ejecuciones afectadas | 7,3 % |

(Ese 7,3 % se midió sobre `request_payload_chars`, que cierra; la *composición* de §6.1.35 es la que no cierra, y por eso no se usa acá.)

Y el manejador no trunca a ciegas: para una lectura paginada devuelve **un
rechazo con el contorno del archivo** —`{"error": "file too large to read in one
call", …, "lines": 2999, "characters": 42770}` más la instrucción de leer un
rango—, o sea 42 770 caracteres se vuelven 393 y el modelo sabe qué hacer. Para
cualquier otro resultado recorta con un aviso honesto.

**El tope va después del archivado, no antes.** El código de `tool_runner` dice
explícitamente *"Queue the raw exchange for working memory before anything
downstream gets to shorten it"*: la copia que va a memoria de trabajo conserva el
texto completo y sólo se recorta la que va al prompt. Un test fija ese orden por
inspección del código, porque invertirlo rompería `recall_context` en silencio.

**Validación en vivo: 3 de 3 completan, y el tope no se disparó en ninguna.** Es
lo esperado —los resultados gigantes son el 0,9 %— así que la corrida confirma que
no rompe nada y no confirma el ahorro; el ahorro está medido por aritmética sobre
los payloads guardados, que para esto es la medición correcta: no es una
estimación de efecto, es cuánto texto deja de mandarse.

### 6.1.34 El residuo, leído (IV): `evidence-depth` sobre la verdad del fixture

Los dos ítems que quedaban son de formato de informe y necesitan el diff, así que
los leí contra la **verdad del fixture**. `code_review/auth.py` son 17 líneas con
cuatro defectos reales: comparación no constante de un token en claro (línea 5),
fail-open en `except Exception: return True` (8-11), el token crudo impreso al log
(15-16), y un `TOKEN_CACHE` global escrito y nunca leído (1, 6); más el riesgo de
substring en `"admin" in scopes` (línea 7) y la falta de tipos.

Leí las **8 ejecuciones de las dos campañas que sostienen los titulares**
(`ab-generality` y `ab-accounted-full`, 4 cada una), que es el corte que importa
para lo que este documento afirma:

| ítem | media |
| --- | ---: |
| `evidence-depth` | **2,00** (8 de 8) |
| `report-usability` | **2,00** (8 de 8) |

Las ocho citan `auth.py:N` con la evidencia correcta, nombran los cuatro defectos
reales, los ordenan por severidad con etiqueta explícita (Critical/blocker, High,
Medium, Low) y explican el impacto. **Ninguna inventa un defecto**, que es lo que
el ítem persigue: el fixture no tiene comparación insegura en ningún lado *salvo*
la línea 5, y ninguna acusa otra cosa. La única imprecisión es un rango —una
ejecución cita `auth.py:8-12` donde el `except` termina en 11— y no cuenta como
invención: el rango incluye la línea 12, que es la consecuencia, no el defecto.

**El techo de esta tarea como evidencia es bajo, y conviene decirlo.** Su
`verify.py` es una rúbrica de palabras clave (`"token"` + uno de
plain/constant/timing/hash, `"exception"` + uno de allow/true/bypass/fail open,
`("blocker","critical","high")`, …) y **no se oculta**, porque por regla sólo se
ocultan los verificadores que juzgan comportamiento y éste juzga redacción. La
consecuencia es visible: los ocho informes usan exactamente las palabras que la
rúbrica exige. Lo que sostiene el 2,00 no es eso —es que leí los informes contra
el código y son correctos—, pero significa que la tarea mide "informe plausible y
ordenado" y no "revisor independiente", y que un 2,00 acá vale menos que un 2,00
en `findings-are-real`, donde el defecto hay que encontrarlo y el verificador está
oculto.

### 6.1.35 El 22 % del payload que el harness no sabe atribuir

Buscando una palanca más en la composición del payload apareció un agujero en el
propio instrumento. `message_payload_chars` es el JSON completo de la lista de
mensajes; `message_content_chars_by_role` suma, por rol, `len(json.dumps(content))`.
La diferencia entre los dos debería ser la estructura de cada mensaje —`role`,
`tool_call_id`, los `tool_calls`—, y no lo es:

| ronda | mensajes | suma por rol | payload | diferencia |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 25 291 | 25 348 | 57 |
| 1 | 4 | 38 777 | 39 218 | 441 |
| 10 | 30 | 54 553 | 64 837 | 10 284 |
| 20 | 51 | 58 908 | 91 068 | 32 160 |
| 29 | 69 | 62 190 | 101 224 | **39 034** |

En la corrida de escala son **586 924 caracteres, el 22,4 % del payload
acumulado**, y crecen ~1 300 por ronda. No pude atribuirlos, y descarté las dos
explicaciones plausibles:

* **no son los argumentos de las tool calls.** El `tool_trace` los guarda
  completos (`dict(ctx.arguments)`, sólo el *resultado* se corta a 20 000), y
  suman **2 513 caracteres** en toda la corrida contra 586 924 sin explicar. El
  mayor es de 581.
* **no es el razonamiento del modelo.** `reasoning_content` se extrae de la
  respuesta y lo consumen el checker y la guía; no se anexa a la lista de
  mensajes que se reenvía.

**Qué invalida esto.** La cifra de §6.1.33 —"los resultados de herramientas son el
31,4 % del payload"— sale de esa misma tabla por rol, así que **subestima** todo lo
que esté en el bloque sin atribuir: el 31,4 % es un piso, no una medición. Y
cualquier análisis de composición que se apoye en `message_content_chars_by_role`
tiene el mismo problema, incluida la conclusión de que el esquema es "el 48 % del
payload" que ya había corregido en §6.1.32 por otra razón.

Lo dejo como **hueco de medición del harness**, no como palanca: sin saber qué es
ese 22 % no se puede dimensionar nada, y afirmar una palanca sobre una tabla que
no cierra sería el quinto error de instrumento del documento. El arreglo es del
harness, no del engine: `prompt_composition` debería registrar el
`len(json.dumps(message))` por mensaje, o directamente el tamaño de cada campo, y
hoy no lo hace.

### 6.1.36 El cache de prompt: ya está en su techo, y nadie lo estaba midiendo

Pedido explícito: mejorar el cache hit. Primero había que **poder medirlo**, y no
se podía.

**El engine lo recolectaba y lo tiraba.** `LoopState` tiene
`cache_creation_tokens`, `cache_read_tokens` y `cached_tokens`, poblados desde el
`usage` del proveedor; `_log_cache_summary` los imprime en una línea al terminar
y ahí termina todo. No estaban en `loop_observed_metrics`, así que ningún
`EngineResult.metrics` los llevaba, y **ninguna de las 333 ejecuciones guardadas
tiene un número de cache**. El harness estaba descartando la métrica que dice si
un cambio de prompt ahorró *dinero* y no sólo caracteres.

Ahora: `loop_observed_metrics` los publica, el runner los graba en la fila de
observación (`cache_read_tokens`, `cache_creation_tokens`, `cached_prefix_tokens`)
y la comparación suma una métrica derivada `cache_hit_rate` que **cuenta las dos
convenciones de proveedor** —lo Anthropic (`cache_read_input_tokens`) y lo
OpenAI/DeepSeek (`prompt_tokens_details.cached_tokens` / `prompt_cache_hit_tokens`)—,
porque leer sólo la primera dijo "no hubo cache" en un proveedor que había
reportado 531 024 tokens cacheados en la misma corrida.

**Medición, configuración enviada, 6 ejecuciones recientes:**

| tarea | rep | prompt | cacheado | hit |
| --- | ---: | ---: | ---: | ---: |
| `complex-plan` | 0 | 80 927 | 64 088 | 79,2 % |
| `complex-plan` | 1 | 132 762 | 103 857 | 78,2 % |
| `complex-plan` | 2 | 114 337 | 73 918 | 64,6 % |
| `test-selection` | 0 | 86 140 | 80 384 | **93,3 %** |
| `test-selection` | 1 | 49 664 | 36 211 | 72,9 % |
| `test-selection` | 2 | 69 149 | 56 632 | 81,9 % |

**Media 78 %, y está en el techo estructural.** El diseño ya es el correcto: el
prompt estático es lo primero, con `cache_control` al final del prefijo estable
(`CACHE_BREAKPOINT_MARKER`, insertado antes del bloque de sesión que crece), y el
esquema de herramientas lleva su propio punto de corte. La cuenta que confirma el
techo es la de la primera ronda: con N rondas y un prefijo cacheable de P tokens,
el máximo posible es `(N−1)·P / Σtokens`, porque la ronda 1 siempre es un fallo.
Para `test-selection` r1, con ~5 rondas y P ≈ 9 000 tokens, eso da ~72 % contra los
72,9 % medidos: **se está cacheando prácticamente todo lo cacheable.**

Así que no hay nada que mejorar en el cache, y ahora se puede decir con un número
en vez de con una impresión. Lo único que subiría la *tasa* es acortar las
corridas o mover más prompt por encima del punto de corte, y las dos cosas tienen
su propio costo.

### 6.1.37 Corrección: la "hoja conforme" la reescriben los dos brazos

En §6.1.20 reporté que `default` había reescrito una hoja conforme
(`v + 9` → `v + 58`) en 1 de 3 mientras `lean` corregía la rota en 3 de 3, y lo
presenté como una diferencia de calidad. Tres ejecuciones nuevas de
`deep-localization` con la configuración enviada (`lean` + `task`) muestran que
**`lean` también lo hace**:

| rep | archivo | línea eliminada | |
| ---: | --- | --- | --- |
| 0 | `src/pkgs/pkg_17/mod_07.py` | `return v - 40` | correcto |
| 1 | `src/pkgs/pkg_17/mod_07.py` | `return v - 40` | correcto |
| 2 | `src/pkgs/pkg_34/mod_15.py` | `return v + 1` | **compensa** |

El verificador pasa en las tres —el total queda bien— y sólo la línea eliminada
distingue el arreglo del parche, que es exactamente lo que el probe mira. Pooling:
`lean` 5 de 6, `default` 2 de 3. **La inferencia de §6.1.20 era un artefacto de
tres parejas**, y queda corregida: los dos brazos compensan a veces, la tasa es
del orden de una en cuatro, y el engine no puede gatearlo porque no conoce la
convención del repositorio —eso lo sabe el pedido, no el motor.

De paso, el mismo día apareció un defecto del instrumento con esta forma:
`run_probes` recorría `*/artifacts/*/run.json`, un nivel de profundidad, así que
**una campaña de un solo brazo le daba cero ejecuciones en silencio** —el
`deep-localization` recién corrido informaba "0 runs"—. `build_packet` ya usaba
`rglob` desde §6.1.29; ahora los dos, con el brazo etiquetado por su ruta.

### 6.1.38 El test de signos no contaba las parejas que empeoraban

El defecto de instrumento más caro del documento, y el único que **inflaba los
dos titulares**. La comparación pareada calculaba el p-valor así:

```python
improved  = sum(1 for d in pairs if d < 0)
worsened  = sum(1 for d in pairs if d > 0)
non_tied  = improved + worsened
p_value   = _sign_test_p_value(len(pairs) - non_tied, improved)   # ←
```

y la función hacía `trials = ties + directional`, o sea `trials = empates +
mejoras`. **Las parejas que empeoraban no entraban en el conteo.** El efecto es
que un resultado mixto se evaluaba como si sólo hubieran existido las parejas
favorables:

| parejas | p publicado | p correcto |
| --- | ---: | ---: |
| 8 mejor / 0 peor | 0,0078 | 0,0078 |
| 12 mejor / 4 peor | **0,0005** | **0,0768** |
| 11 mejor / 7 peor | **0,0010** | **0,4807** |
| 5 mejor / 1 peor | 0,0625 | 0,2188 |

Con cero derrotas las dos fórmulas coinciden —por eso el titular del modo de
engine, 9/0, sobrevivió intacto—, y con cualquier derrota la publicada es
demasiado pequeña. Cuanto más mixto el resultado, más grande la mentira.

**Alcance.** Recalculé **todas** las comparaciones guardadas
(`bench/runs/**/comparison.json`, 20 campañas) desde sus ficheros de
observaciones, que son la entrada cruda. **56 métricas se mueven y 16 cruzan la
línea de 0,05 en la dirección mala.** Las que importan:

| campaña | métrica | parejas | p publicado | p correcto |
| --- | --- | ---: | ---: | ---: |
| `ab-generality` (`lean`) | `prompt_tokens` | 12/4 | 0,0005 | **0,0768** |
| `ab-generality` (`lean`) | `latency_seconds` | 12/4 | 0,0005 | **0,0768** |
| `ab-generality` (`lean`) | `tool_calls` | 12/2 | 0,0020 | **0,0129** ← sigue |
| `ab-policy` | `prompt_tokens` | 11/7 | 0,0010 | **0,4807** |
| `ab-policy-pipeline2` | `pipeline_prompt_tokens` | 7/1 | 0,0156 | **0,0703** |
| `ab-engine-mode-fixed` | `prompt_tokens` | 9/0 | 0,0020 | **0,0039** ← sigue |
| `ab-engine-mode-fixed` | `latency_seconds` | 8/1 | 0,0039 | **0,0391** ← sigue |

**Qué cambia en las conclusiones, y qué no.**

* La variante `lean` **conserva la dirección** en tokens (−35,9 %), latencia
  (−33,0 %) y tool calls (−15,4 %), y de las tres sólo **tool calls queda
  resuelta** (12/2, p=0,0129). Con 16 parejas y 4 derrotas, el test de signos no
  puede resolver un −35,9 % de tokens: eso necesitaba más parejas, no un
  p-valor mejor.
* El catálogo de políticas **pierde las dos significancias** (0,48 y 0,070). Lo
  que queda es dirección replicada en dos muestras independientes, que agrupadas
  dan 18/8 y p=0,076: consistente, todavía no resuelto. El flag sigue en `true`
  porque la decisión original ya se apoyaba en que las dos muestras van en la
  misma dirección y en que no hay costo medible en latencia, pero la fila del
  resumen ya no puede decir "medido, p=0,016".
* El **modo de engine por defecto sobrevive**: 9 de 9 parejas sin empate,
  −82,9 % de tokens, p=0,0039. Es la afirmación más fuerte del documento y lo
  era ya antes de la corrección, porque no tiene ninguna pareja en contra.

**El arreglo.** `_sign_test_p_value(improved, worsened)` toma las dos
direcciones y `trials = improved + worsened`, con cinco tests en
`tests/test_agent_task_compare.py` que comparan contra binomios calculados a
mano —incluido el caso 12/4 = 0,0768, que es el que este documento publicaba
mal—. Los `comparison.json` guardados siguen teniendo los valores viejos; la
tabla de arriba es el recálculo.

Esto es el **sexto** defecto de instrumento del documento, y el patrón se
repite: los cinco anteriores también hacían que el engine pareciera mejor de lo
que era, y los cinco los encontré auditando el instrumento y no el engine. La
regla que los habría atajado a todos está en la lista desde §6.1.19 y no se
cumplió: **un número que decide una frase tiene que tener un test que lo fije
contra un valor calculado fuera del código que lo produce.**



### 6.1.39 El 22 % del payload era el razonamiento, y sí se reenvía

§6.1.35 dejó **586 924 caracteres (22,4 % del payload) sin atribuir** y descartó
dos explicaciones. La segunda era falsa:

> no es el razonamiento del modelo. `reasoning_content` se extrae de la
> respuesta y lo consumen el checker y la guía; no se anexa a la lista de
> mensajes que se reenvía.

Sí se anexa. `tool_runner.append_assistant_message` llama a
`reasoning_history_fields(message)` y mete el resultado en el turno del
asistente (`**history_fields`, `loop/tool_runner.py:256`), y los campos que
preserva son `reasoning_content`, `thinking_blocks`, `reasoning_details` y
`reasoning_items`, más `provider_specific_fields.{reasoning_details,
thought_signatures}`. Esos mensajes son el prefijo de todas las peticiones
siguientes.

**Lo que MiniMax devuelve y guardábamos, en crudo** (una llamada real, mensaje
del asistente volcado):

```
reasoning_content: "The user wants me to think briefly then call read_file..."
provider_specific_fields:
  {"name": "MiniMax AI", "audio_content": "",
   "reasoning_details": [{"type":"reasoning.text","id":"reasoning-text-1",
                          "format":"MiniMax-response-v1","index":0,
                          "text":"The user wants me to think briefly..."}],
   "reasoning_content": "The user wants me to think briefly..."}
```

El mismo texto **dos veces**, y la copia envuelta pesa ~2,5× más que la llana. No
hay firma en ninguna de las dos, así que nada obliga a conservarlas.

**Cuánto cuesta, medido contra la API.** Tres transcripciones que sólo difieren
en la longitud de un campo de razonamiento reenviado (control con 0 caracteres,
y la misma conversación con 4 000, 16 000 y 48 000):

| campo | 0 ch | 4 000 | 16 000 | 48 000 | pendiente |
| --- | ---: | ---: | ---: | ---: | ---: |
| `reasoning_details` | 201 | 702 | 2 202 | 6 202 | **1 token por 8 caracteres** |
| `reasoning_content` | 195 | 702 | 2 202 | 6 202 | **1 token por 8 caracteres** |

La pendiente es idéntica en los dos campos y perfectamente lineal: **el
razonamiento reenviado se factura**, a 0,125 tokens por carácter, en cada
petición que lo lleva. Y como el turno del asistente de la ronda *i* viaja en
todas las peticiones *i+1..N*, un razonamiento de una ronda se paga N−i veces.

**El arreglo.** `trim_superseded_reasoning` (`engine/behavior/reasoning_content.py`)
borra el razonamiento visible de todo turno del asistente que ya no es el último.
Sólo el último lo conserva, porque es el único cuyos resultados de herramienta
viajan en la misma petición que él — y por eso la llamada va *después* del
`append`, no antes. El material opaco no se toca nunca: un bloque con
`signature`, `thought_signature`, `encrypted_content`, un tipo
`redacted_thinking`/`encrypted_thinking`, o el campo `thought_signatures`, se
conserva en todas partes, porque Anthropic y Gemini rechazan una cadena de
tool-use cuyos bloques de pensamiento vuelven pelados. Interruptor:
`LOOP_REASONING_TRIM_ENABLED` (por defecto `true`).

**El instrumento, arreglado en la misma pasada.** `measure_request_payload`
ahora atribuye **cada carácter** de `message_payload_chars` a un cubo con
nombre —`message_value_chars_by_key`, más `json_structure_chars`— y publica
`payload_unattributed_chars`, que debe ser 0. Un test lo fija
(`test_request_payload_accounts_for_every_character`). Con eso §6.1.35 deja de
ser un hueco declarado: la tabla por rol no cerraba porque sólo miraba
`content`, y el asistente con tool calls tiene `content` vacío y el argumento en
`tool_calls`.

**El censo, con el instrumento arreglado** (6 ejecuciones, brazos `off` de
`bench/runs/20260914-trim2/`, la última petición de cada corrida, con el corte de
razonamiento desactivado para ver el razonamiento entero):

| tarea | payload | `content` | `tool_calls` | razonamiento | estructura | sin atribuir |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex-plan` r1 | 57 229 | 30 439 (53 %) | **15 491 (27 %)** | 9 959 (17 %) | 842 | **0** |
| `complex-plan` r2 | 63 918 | 32 723 (51 %) | **21 127 (33 %)** | 9 263 (14 %) | 523 | **0** |
| `test-selection` r0 | 36 182 | 30 697 (85 %) | 2 443 (7 %) | 2 013 (6 %) | 673 | **0** |
| `test-selection` r2 | 40 162 | 32 484 (81 %) | 3 510 (9 %) | 3 083 (8 %) | 677 | **0** |
| `complex-plan` r0 | 49 513 | 48 306 (98 %) | 746 (2 %) | 189 (0,4 %) | 173 | **0** |
| `test-selection` r1 | 37 945 | 31 327 (83 %) | 1 845 (5 %) | 3 885 (10 %) | 580 | **0** |

Y la **primera** exclusión de §6.1.35 también era falsa:

> no son los argumentos de las tool calls. El `tool_trace` los guarda completos
> … y suman **2 513 caracteres**

En una tarea que **escribe archivos**, los argumentos de las tool calls son
hasta el **33 % del payload** (21 127 caracteres en `complex-plan.r2`) contra los
2 513 que citaba §6.1.35, que salieron de otra corrida y del `tool_trace` —que en modo `orquestador` sólo
guarda las llamadas del principal, y guarda el `dict` parseado mientras el
mensaje lleva la cadena JSON—. El salto se ve en la propia curva: en
`complex-plan.r1` los argumentos pasan de unos cientos de caracteres en la
primera petición a **15 491** en la última, porque el modelo escribió un archivo
entero y
el cuerpo del archivo *es* el argumento. Desde ahí viaja en todas las peticiones
siguientes.

**Eso es una palanca medida, y es más grande que el razonamiento.** Nada en el
protocolo la justifica: el proveedor necesita el `id` y el nombre para casar el
resultado de la herramienta, no el cuerpo. Y a diferencia del razonamiento
—donde hay que conservar bloques con firma para que Anthropic y Gemini acepten la
cadena— acá no hay material opaco que preservar.

Lo que **no** se puede afirmar es que `recall_context` devuelva el cuerpo:
`WorkingMemory._extract` archiva el *resultado* de la llamada, y usa los
argumentos sólo para el título. Para un archivo escrito el cuerpo está en el
disco y el resultado dice que se escribió; ésas son las dos rutas reales, y son
las que el marcador de §6.1.41 nombra. Queda declarado con su número, y su
implementación y su comparación pareada están en §6.1.41.

**El corte de razonamiento, medido en dos campañas independientes** (14 parejas
en total, condición `baseline`, tareas disjuntas: `complex-plan` y
`test-selection` × 3 en `bench/runs/20260914-trim2/`, y `reversible-ambiguity`,
`evidence-code-review`, `user-owned-tradeoff` y `tool-failure-recovery` × 2 en
`bench/runs/20260914-trim3/`):

| | off | on |
| --- | ---: | ---: |
| razonamiento en la última petición | 189–9 959 ch (mediana **9,0 %** del payload) | **0 ch (0,0 %)** |
| tokens de prompt facturados (14 parejas) | 101 448 | 79 440 (**−21,7 %**) |
| parejas mejor/peor | — | **11 / 3** |
| p (test de signos corregido, §6.1.38) | — | **0,0574** |
| success | 14/14 | 14/14 |
| `cache_hit_rate` | 84,5 % | 75,2 % |

Las dos campañas van en la misma dirección por separado: −19,8 % (5/1) y
−20,8 % (6/2). Agrupadas, **11 de 14 parejas bajan y el p-valor queda en 0,0574
— justo por fuera del umbral declarado.** No lo cuento como resuelto, y no voy a
agregar parejas hasta que cruce: sería elegir la muestra por el resultado, que es
exactamente lo que este documento se prohíbe. Lo que sí está probado, con
medición directa contra la API, es el **mecanismo**: 1 token facturado por cada 8
caracteres de razonamiento reenviado, lineal de 0 a 48 000, y 0 caracteres en la
última petición cuando el corte está activo. O sea: el efecto existe y es del
tamaño que dice; lo que falta es precisión sobre el estimador, no evidencia de
que el efecto esté ahí.

Dos efectos secundarios, uno esperado y uno no:

* El **hit rate baja** (84,5 % → 75,2 %), porque el bloque dinámico se encoge y el
  prefijo estable pasa a ser una fracción mayor de una petición más chica. El
  total facturado baja igual, que es lo que importa — pero **un hit rate más alto
  no es el objetivo**, y este cambio lo demuestra: perseguirlo habría bloqueado
  una mejora real.
* Las **rondas de modelo bajan 23,1 %** (10/4, p=0,1796), y los tokens de
  completion **suben 3,9 %** (9/5, p=0,42). Ninguna de las dos está resuelta; las
  dos son del tamaño del ruido a este n. Si la caída de rondas fuera real sería
  el mismo patrón que ya apareció con `lean` (§6.1.1): menos contexto redundante,
  menos rondas, y una respuesta un poco más larga por ronda.

### 6.1.40 Reconstruir el contexto o dejar que crezca: qué es más barato

Pregunta directa: ¿conviene partir el problema en tasks y **reconstruir** el
contexto en cada una —payload plano, pero sin nada que cachear— o **dejar
crecer** un historial incremental hasta un umbral y compactar, como hacen los
demás harnesses?

Primero, el hecho que la pregunta da por supuesto y no es cierto: **este engine
ya hace lo segundo**. Sobre 308 ejecuciones de la configuración enviada, 269
crecen monótonamente y 39 muestran dientes de sierra de compactación; el payload
mediano termina en **1,71×** su tamaño inicial, a **+2 277 caracteres por
petición**. Lo que el engine reconstruye es el *system* y el *user* —el plan, el
paso activo, las acciones previas—, y eso es precisamente lo que mantiene el
prefijo estable y el cache en su techo (§6.1.36). O sea: no hay un trade-off
entre reconstruir y cachear; hay un prefijo estable, que es lo que hace que el
cache funcione, y encima un historial que crece.

**El modelo.** Con `k` = precio de un acierto de cache sobre un token de entrada
fresco, `d` = incremento por ronda, `N` = rondas y `D` = el working set acotado
que un harness que reconstruye tendría que cargar:

```
D  <=  d·(N−1)·[N − (1−k)(N−2)] / (2N)
```

Dos propiedades, ninguna obvia, y las dos importan:

1. **El prefijo estable se cancela.** Los dos regímenes mandan el mismo prefijo
   `N` veces y lo leen del cache `N−1` veces; cuesta lo mismo en ambos. Por eso
   esta pregunta no tiene nada que ver con el tamaño del system prompt —ni con
   un recargo por *escritura* de cache, que los dos pagan igual—. Lo único que
   decide es la parte variable.
2. **Un cache peor favorece reconstruir.** El cache es lo que abarata *dejar
   crecer*. Con cache gratis (`k = 0`) el listón es `d(N−1)/N`; sin ningún cache
   (`k = 1`), `d(N−1)/2` — el caso más favorable a reconstruir, y el que lo
   acota para cualquier proveedor.

**El listón, sobre los parámetros medidos** (`d = 2 277` caracteres,
`N = 10`):

| `k` (acierto / fresco) | working set que aún gana |
| ---: | ---: |
| 0,00 (cache gratis) | 2 050 ch — 512 tok |
| 0,10 (Anthropic, OpenAI, Gemini, DeepSeek, Mistral) | 2 869 ch — **717 tok** |
| 0,20 (MiniMax-M3, Grok Code Fast) | 3 689 ch — **922 tok** |
| 0,50 | 6 149 ch — 1 537 tok |
| 1,00 (sin cache) | 10 248 ch — **2 562 tok** |

**El otro lado de la comparación es medible, y no es una suposición.** El working
set que un régimen acotado tendría que cargar es el prompt que **el propio
engine reconstruyó** para una iteración —tarea, plan, paso activo, acciones
previas— y está en `prompt_composition_history`:

| | caracteres | tokens |
| --- | ---: | ---: |
| mediana (308 iteraciones) | 6 043 | **1 511** |
| extremo liviano (p10) | 1 311 | 328 |
| extremo pesado (p90) | 9 723 | 2 431 |

**El punto de empate: `k = 0,49`.** Con el working set mediano, reconstruir sale
más barato sólo si el proveedor cobra el acierto de cache a **más del 49 %** de
un token fresco. Ninguno de los que publican cache llega ahí:

| proveedor | `k` | escritura | listón | veredicto | append $/run | el cache ahorra |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `claude-sonnet-4-5` | 0,10 | 1,25× | 717 tok | dejar crecer | 0,0637 | 0,2352 |
| `claude-opus-4-5` | 0,10 | 1,25× | 717 tok | dejar crecer | 0,1062 | 0,3920 |
| `gpt-5.1` | 0,10 | — | 717 tok | dejar crecer | 0,0265 | 0,0980 |
| `gpt-5.2` | 0,10 | — | 717 tok | dejar crecer | 0,0372 | 0,1372 |
| `gemini-2.5-pro` | 0,10 | — | 717 tok | dejar crecer | 0,0265 | 0,0980 |
| `deepseek-reasoner` | 0,10 | — | 717 tok | dejar crecer | 0,0059 | 0,0220 |
| `minimax/MiniMax-M3` | 0,20 | — | 922 tok | dejar crecer | 0,0090 | 0,0209 |
| `grok-code-fast-1` | 0,20 | — | 922 tok | dejar crecer | 0,0299 | 0,0697 |
| `mistral-large-latest` | 0,10 | — | 717 tok | dejar crecer | 0,0106 | 0,0392 |
| `groq/llama-3.3-70b` | 1,00 | — | 2 562 tok | **reconstruir** | 0,0588 | 0,0000 |
| `fireworks/deepseek-v3` | 1,00 | — | 2 562 tok | **reconstruir** | 0,0897 | 0,0000 |

Los dos últimos **no publican precio de cache**: un repetido se factura como
fresco. Y en el extremo liviano del working set (328 tokens) el empate se va a
`k = −0,09`, o sea que **ningún proveedor puede hacer ganar a dejar crecer**.

**Qué se concluye, y qué no.** Sobre cualquier proveedor con cache real, dejar
crecer gana — y no por el precio, sino porque el working set que un régimen
acotado necesita (1 511 tokens medianos, y el plan solo ya pesa más que el
listón) es mayor que lo que el listón permite. Sobre un proveedor sin cache,
reconstruir gana si el working set se mantiene bajo la mediana, que es un
objetivo de diseño y no un hecho. O sea que **la respuesta no es la misma para
todos los proveedores, y depende de una cantidad que el harness ahora mide**.

Lo que sí es igual en todos: el cache vale entre el 60 % y el 80 % de la factura
de entrada, así que la palanca con retorno no es reorganizar el contexto sino
**encoger el incremento** —resumen de paso, corte de razonamiento (§6.1.39),
tope de resultado de herramienta (§6.1.33)—. Cada carácter que se saca del
incremento se paga una vez como fresco y luego `N−1` veces a precio de acierto,
y no toca el hit rate, que sigue en su techo.

Herramienta: `python -m bench.context_regime_cost --runs bench/runs`, con los
precios leídos de `litellm.model_cost` y 13 tests en
`tests/test_context_regime_cost.py`.

### 6.1.41 El cuerpo del archivo que el modelo ya escribió, reenviado en cada ronda

El cubo más grande del payload después de `content`, y la palanca más grande que
queda (§6.1.39). En una tarea que escribe archivos, los argumentos serializados
de las tool calls son **hasta el 33 % de la petición** (21 127 caracteres en
`complex-plan.r2`), porque **el cuerpo del archivo escrito *es* el argumento** y
desde la ronda que lo escribe viaja en todas las siguientes.

Nada en el protocolo exige que esos bytes sean los originales. El proveedor
necesita el `id` y el `name` para que el resultado de la herramienta tenga una
llamada que contestar; el argumento es salida previa del propio modelo. Y para
un archivo escrito, el cuerpo está en el disco y el mensaje de *resultado* dice
que se escribió.

**Lo implementado es más angosto que borrar los argumentos.**
`trim_superseded_tool_arguments` conserva **todas las claves y la forma del
JSON** y sustituye sólo los *valores de texto largos* (>400 caracteres) por un
marcador que dice cuánto midió el cuerpo elidido y cómo recuperarlo. Un
proveedor que valide la forma sigue viendo la forma; el modelo sigue viendo qué
llamó y dónde; deja de ver un archivo que ya escribió:

```
{"file_path":"a.py","content":"<elided by infinidev: 2400 chars; the tool result
 above is unchanged — read the file if you need its content>"}
```

**El texto del marcador es load-bearing y por eso dice sólo lo que es cierto.**
La primera versión ofrecía `recall_context`, y era **falso**:
`WorkingMemory._extract` empareja la llamada del asistente con su resultado
`role: "tool"` y archiva **el resultado** como contenido del registro; los
argumentos sólo se usan para construir el título. O sea que un cuerpo elidido
**no** es lo que devuelve un recall, y ofrecerlo mandaría al modelo a buscar
evidencia que no está — exactamente la falla que esta palanca existe para
reducir. Lo que sí es cierto: el resultado de arriba no se toca, y el cuerpo de
un archivo está en el disco. El marcador dice eso y nada más, y hay un test que
falla si alguna vez vuelve a nombrar `recall_context`.

Sólo se tocan los turnos **ya cerrados** —todos menos el último, cuyos resultados
viajan en la misma petición—, que es la misma frontera que usa el corte de
razonamiento (§6.1.39) y la que el loop ya trata como resuelta. Una llamada que
no parsea **no se reescribe nunca**: una llamada malformada es un hecho de la
corrida, y reescribirla lo escondería y podría convertir una falla diagnosticable
en otra distinta.

**La regla medida contra cada llamada real del corpus**, no supuesta
(`python -m bench.tool_argument_census --runs bench/runs`, 4 397 llamadas
registradas en `tool_trace`):

| | |
| --- | ---: |
| llamadas reales | 4 397 |
| que no parsean | **0** |
| con un valor de más de 400 caracteres | 409 (**9,3 %**) |
| bytes de argumentos en total | 2 009 179 |
| elidibles | 1 351 482 (**67,3 %**) |
| de las llamadas de 500+ caracteres (las que son payload) | 540 |
| bytes elidibles en ésas | **85,7 %** |

Dónde están los bytes: `create_file` 1 042 152 elidibles en 157 llamadas,
`team_delegate` 128 153 en 87, `execute_command` 98 366 en 1 105, `edit_file`
31 123 en 278. O sea: **la regla dispara en una minoría de llamadas y se lleva
casi todos los bytes**, porque las llamadas en las que dispara son justo las que
cargan el cuerpo de un archivo. Y **cero llamadas reales sin parsear**, que es lo
que decide si el 67,3 % existe o es parcial.

Combinado con el payload de cada corrida:

| corrida | payload | `tool_calls` | −67 % | −86 % |
| --- | ---: | ---: | ---: | ---: |
| `complex-plan` r2 | 63 918 | **21 127 (33 %)** | −22,2 % | −28,4 % |
| `complex-plan` r1 | 57 229 | 15 491 (27 %) | −18,2 % | −23,3 % |
| `test-selection` r2 | 40 162 | 3 510 (9 %) | −5,9 % | −7,5 % |
| `test-selection` r1 | 37 945 | 1 845 (5 %) | −3,3 % | −4,2 % |

O sea: entre **−18 % y −28 %** en las tareas que escriben archivos y entre −3 %
y −8 % en las que leen, que es exactamente la asimetría que el censo predice.

**Antes de gastar una campaña, una sonda contra el proveedor vivo**, porque el
riesgo de esta palanca es de protocolo y se puede descartar en dos llamadas. La
misma conversación —dos `create_file` con 300 líneas de cuerpo cada uno, y una
pregunta sobre la primera línea del primero— transcripta entera y transcripta
con los argumentos elididos:

| transcripción | prompt tokens | finish | ¿llamó a `read_file`? | respuesta |
| --- | ---: | --- | --- | --- |
| entera | 3 019 | stop | no | "The first line of a.py is `print('hello')`." |
| elidida | **1 841** | stop | no | "The first line of `a.py` is `print('hello')`." |

Dos de dos en cada brazo. El proveedor **acepta** los argumentos elididos —no
hay error de protocolo: el `id` y el `name` siguen ahí y el resultado de la
herramienta tiene su llamada— y el modelo **contesta igual**, porque el dato que
necesita lo tiene el mensaje de resultado, que no se toca. −39 % de tokens de
prompt en esa conversación. Eso no reemplaza la comparación pareada —una sonda
de dos llamadas no mide una corrida de diez rondas— pero sí elimina la
explicación más barata de que la palanca no sirva.

**El interruptor existe y está apagado por defecto**
(`LOOP_TOOL_ARGUMENT_TRIM_ENABLED = False`), y eso es deliberado, no una
indecisión. A diferencia del razonamiento, acá hay un riesgo real y simétrico:
si el modelo necesita releer lo que escribió, paga una ronda —y una ronda cuesta
~10 800 equivalentes de token fresco a precio de cache— contra un ahorro que
sólo existe si no la paga. No hay forma de saber cuál gana sin correr las dos
versiones sobre tareas que escriben archivos, y esa comparación está en curso.
Hasta que cierre, lo que este documento afirma es la medición del *payload*, no
la del *costo*, y la distinción importa: el mismo error de razonamiento es el
que produjo la fila de `lean` que §6.1.38 tuvo que corregir.

### 6.2 Resultado negativo: la presión de agrupación no agrupa

Diseño pareado, 3 tareas de código × 3 repeticiones × 2 brazos (18 ejecuciones,
`MiniMax-M3`, corpus `engine_eval_v2`), interruptor
`LOOP_BATCHING_NUDGE_ENABLED`. Comando:
`bench/agent_task_ab.py`, resultado completo en
`bench/runs/20260913-engine-v2/ab-batching/comparison.{md,json}`.

| métrica | sin nudge | con nudge | delta | rango sin / con | resuelto |
| --- | ---: | ---: | ---: | --- | --- |
| tokens de prompt | 114 621 | 118 305 | +3,2 % | 61 527–250 109 / 56 345–165 402 | no |
| tokens de completion | 4 594 | 5 334 | +16,1 % | 1 871–10 807 / 1 736–7 678 | no |
| tool calls | 11 | 12 | +9,1 % | 8–22 / 9–16 | no |
| latencia | 64,6 s | 72,0 s | +11,4 % | 33,9–169,5 / 27,1–118,0 | no |
| success | 9/9 | 9/9 | — | — | — |

Con el test de signos, ninguna métrica queda resuelta: las parejas se reparten
4/5, 4/5, 3/5 y 5/4. La métrica del mecanismo — llamadas por ronda de modelo,
leída de los artefactos — se movió de **0,90 a 1,00** de mediana, con rangos que
se solapan: el modelo ya emite una llamada por ronda y decirle que agrupe no lo
cambia. Los puntos estimados de todas las demás métricas empeoran.

**Decisión: el nudge queda apagado por defecto** (`LOOP_BATCHING_NUDGE_ENABLED =
False`) con la medición escrita en el comentario del ajuste. El mecanismo y sus
tests quedan, para poder reevaluarlo con un modelo que serialice más.

Esto vale más que el cambio que se descartó: mide una intervención que *suena*
correcta — 124 rondas para 78 llamadas en el baseline, cada ronda reenviando el
prompt entero — y demuestra que el texto no mueve la aguja. La hipótesis que
queda viva es que las rondas de una sola llamada son secuenciales por necesidad
(leer, editar, correr), no por decisión del modelo.

## 7. Qué sigue, en orden de valor esperado

Lo que quedaba de esta lista en las rondas anteriores está hecho: la presión de
agrupación se midió y se descartó (§6.2), la sonda de exploración obligatoria
existe y `lean` la pasó (§6.1.5), hay cuatro contratos ocultos con verificador
invisible (§1.2), el contador de llamadas malformadas es métrica de decisión
(§4.4.3) y las advertencias inaplicables de creación de archivos están
corregidas (§4.1).

Queda, en orden de valor esperado:

1. ~~Resolver la contradicción de latencia del catálogo de políticas.~~ **Hecho
   (§6.1.26).** No era un efecto: la diferencia entre brazos es 10 a 25 veces
   menor que la dispersión dentro de un brazo, no tiene mecanismo, y una réplica
   independiente mueve la latencia **−36,7 %** donde la muestra original decía
   +13,6 %. El ahorro de tokens replica con el mismo signo en las dos muestras.
2. ~~Probar `lean` en un repositorio grande de verdad.~~ **Hecho (§6.1.20).**
   Corpus `engine_eval_v9` con 1 256 archivos y tres niveles de jerarquía:
   `lean` resuelve 3/3 como el default, navega igual y cuesta −33 % de tokens.
   Lo que la sonda deja abierto no es la validez sino el costo: ~660 000–990 000
   tokens por ejecución, porque el engine no le dice al modelo que la jerarquía
   existe y el modelo tiene que descubrirla.
3. **Un cap de `<opened-files>` por relevancia.** El presupuesto (48 000
   caracteres) ya se respeta, incluido el primer archivo, que antes lo evadía
   (§4.6). Falta el paso siguiente: un digest para los archivos que el Step
   activo no tocó, en vez de reenviar su contenido entero en cada ronda.
4. ~~Re-medir el nuevo default con el contador corregido.~~ **Hecho (§6.1.23).**
   4 formas de tarea × 2 repeticiones × 2 brazos: −81,0 % de tokens facturados,
   −64,5 % de rondas y −73,0 % de latencia, 8/0 parejas, p=0,0078, calidad sin
   cambios. Falta sólo ampliar la muestra si se quiere citar un intervalo en vez
   de un p-valor.
5. ~~Hacer que el engine note el traspaso flojo.~~ **Hecho y medido inerte
   (§6.1.24).** La compuerta existe, está acotada y no puede terminar un run, pero
   en la configuración enviada no dispara: la brecha era histórica y el proxy de
   longitud que la agrandaba era del instrumento. De paso apareció y se arregló la
   falla peor del corpus: un run que termina con una promesa (§6.1.25).
6. ~~`fallback` contra `preferred` en el clasificador de políticas.~~ **Hecho
   (§6.1.21).** Cronometrado aislado en lugar de comparado por campaña: el
   ruteo local cuesta 4 ms y `preferred` 2,91 s por turno con 446 tokens
   invisibles, y encima resta. El default es `fallback`.
7. **Terminar de leer el residuo de juicio.** Los probes deciden el **73 %** de
   las 666 instancias de rúbrica (§6.1.27) y la lectura ciega sumó 81 más:
   `recommendation-calibration` (**17/17 en 2,00**, §6.1.28),
   `verification-interpretation` (36/39, §6.1.29), `findings-are-real` (**1,89**,
   cero defectos inventados, §6.1.31) y `evidence-depth` + `report-usability`
   (**2,00 en las 8 ejecuciones de las campañas de los titulares**, §6.1.34).
   Quedan 92 instancias de esos dos ítems en el resto del corpus, que son las más
   caras de leer porque necesitan el diff. **Cero llamadas al modelo.**
8. ~~Dieta de los esquemas de herramientas.~~ **Descartado dos veces, y la
   segunda corrige a la primera.** La primera razón fue de contenido: de 17 433
   caracteres de esquema, 3 964 son descripciones de parámetros **opcionales**, y
   revisé las 45 una por una sin encontrar un recorte automático seguro —truncar a
   la primera oración arruina `tail_test_output.mode` (el enum está *después* del
   primer punto), borrarlas pierde `execute_command.cwd`—.

   La segunda es de alcance, y es la que faltaba: **el esquema no es el 48 % del
   payload en general, lo es en algunos payloads.** Distribución de
   `tool_schema_chars` sobre las 4 262 rondas guardadas:

   | caracteres de esquema | rondas |
   | ---: | ---: |
   | 4 069 | **1 654** |
   | 20 134 | 594 |
   | 1 738 | 529 |
   | 46 395 | 526 |
   | 19 929 | 498 |
   | 17 303 | 231 |

   La mayoría de las rondas manda **4 KB de esquema**, no 46 KB: el conjunto de
   herramientas ya es chico en la mayor parte de las ejecuciones, y ahí no hay
   nada que dietar. Y el conjunto completo medido hoy son 59 279 caracteres
   (47 110 en modo compacto), ninguno de los cuales coincide con las cifras de los
   payloads, o sea que el motor ya manda conjuntos distintos según el run.

   También medí qué herramientas se llaman de verdad sobre 330 ejecuciones: 43
   nombres distintos, con `read_file` (1 280), `execute_command` (1 028),
   `list_directory` (419), `edit_file` (265) y `add_step` (234) cubriendo casi
   todo, y 36 herramientas registradas sin una sola llamada. **Eso no justifica
   una dieta**: el `tool_trace` no registra los pseudo-tools (`step_complete` sale
   como "nunca llamada" y se llama en cada cierre), el conjunto varía por
   ejecución, y "nunca usada en 12 formas de tarea sobre fixtures diminutos" no es
   "nunca útil" — en un repositorio real `rename_symbol` es la herramienta
   correcta. Una dieta global sería un cambio de capacidad con falla silenciosa,
   justificado por un corpus que no la mide. Cerrado.

9. **Compactar los argumentos de las tool calls de Steps ya cerrados.
   Implementado y apagado** (`LOOP_TOOL_ARGUMENT_TRIM_ENABLED`, §6.1.41). El
   cubo más grande del payload después de `content`, y el único que crece a
   saltos: **hasta 33 %** (21 127 caracteres en `complex-plan.r2`), 5 % en
   `test-selection` (§6.1.39). El salto es la escritura de un archivo —el cuerpo
   del archivo *es* el argumento— y desde ahí viaja en todas las peticiones
   siguientes. El proveedor necesita el `id` y el nombre para casar el resultado
   de la herramienta, no el cuerpo. Medido contra las 4 397 llamadas del corpus:
   la regla dispara en el 9,3 % de ellas y se lleva el **85,7 % de los bytes** de
   las que son payload, con **cero llamadas sin parsear**; el efecto sobre el
   payload es **−18 % a −28 %** en tareas que escriben archivos y −3 % a −8 % en
   las que leen. Falta la comparación pareada que decide si el ahorro supera a la
   ronda que el modelo pueda gastar releyendo —porque el cuerpo elidido **no** lo
   devuelve `recall_context`, que archiva resultados, sino el disco.
10. ~~Reconstruir el contexto contra dejarlo crecer.~~ **Contestado y medido
   (§6.1.40).** El engine ya deja crecer (269 de 308 corridas, 1,71×, +2 277
   caracteres por ronda); el prefijo estable se cancela en la comparación y el
   cache sólo abarata un lado, así que el empate está en `k = 0,49` sobre el
   working set medido. Con cache real gana dejar crecer en los nueve
   proveedores que lo publican; con `Groq`/`Fireworks`, que no lo publican,
   gana reconstruir si el working set baja de la mediana.

### 7.1 Sobre el orden

Lo que quedaba en esta lista al empezar la ronda está hecho: la decisión del modo
de engine por defecto se tomó con la celda que faltaba medida (§6.1.8) y los dos
defectos que la bloqueaban quedaron corregidos (§6.1.13, §6.1.14). El 1 es ahora
el único número que le falta a una decisión de arquitectura ya acotada —y el
hallazgo de que el router cuesta latencia es nuevo—. El 2 sigue siendo la última
duda de validez sobre el resultado principal —`lean`— y el 3 la última palanca de
tokens que no exige tocar la arquitectura.

El 4 merece una nota de método, porque es la quinta vez en este documento que un
titular cambia después de una corrección de medición: **ninguna campaña de este
proyecto debería reportar tokens por el contador del engine.** El número que
importa es el que el proveedor factura, y desde esta ronda el harness lo captura.

Queda fuera de esta lista, y conviene nombrarlo: **`orchestrator` sigue siendo el
único modo sin camino de entrada desde `auto`**, y con el default en `task` eso
dejó de ser una contradicción para volverse, simplemente, un modo explícito.
Reabrirlo tiene sentido si alguna vez la delegación deja de serializar las
escrituras (`engine/team/runtime.py:156`), que es la condición que hoy le impide
pagar su costo en cualquier tarea que toque código.
