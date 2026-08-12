# Conditional Task Policies

Estado: clasificación activa e inyección evidence-gated. La intención usa el
embedding `ken/static-qwen3-r512-v2`, un head jerárquico empaquetado y acuerdo
contrastivo. El fallback LLM sigue desactivado.

La arquitectura unificada y el procedimiento para ampliarla están en
[`MINI_MODEL_CONDITIONAL_PROMPTING_ARCHITECTURE.md`][architecture]
y [`MINI_MODEL_DATASET_PLAYBOOK.md`][dataset-playbook].

[architecture]: MINI_MODEL_CONDITIONAL_PROMPTING_ARCHITECTURE.md
[dataset-playbook]: MINI_MODEL_DATASET_PLAYBOOK.md

## Implementación actual

La primera versión está integrada en el runtime:

- `engine/task_policies/models.py` define el `TaskProfile` multi-eje y los
  registros de selección/rechazo;
- `engine/task_policies/fragments.py` contiene fragmentos canónicos,
  versionados y acotados;
- `engine/task_policies/router.py` implementa la cascada aprendida, autoridad
  literal, veto contrastivo y fallback LLM estructurado;
- `engine/task_policies/linear_classifier.py` carga el gate discursivo y el
  head de método v3; valida schema, labels, shapes y el `space_id` exacto;
- `engine/task_policies/semantic.py` usa exclusivamente el artefacto bundled
  `ken/static-qwen3-r512-v2`, prototipos positivos/negativos, margen y
  abstención;
- `engine/task_policies/semantic_prototypes.py` mantiene esos prototipos fuera
  del registro de prompts, con al menos 20 positivos y 20 negativos por clase;
- el pipeline resuelve el perfil una vez desde `EscalationPacket.user_request`;
- `Task` y `GoalSpec` transportan el mismo perfil por Task, ReAct, Staged y
  Graph;
- planner, developer y reviewer reciben una capa `task-policy` filtrada por
  rol y fase;
- `engine/task_policies/rollout.py` habilita únicamente fragmentos y versiones
  con evidencia positiva para una ruta de modelo;
- cada fragmento puede ser genérico, inclusivo para una familia/ruta de modelo
  o excluyente; los ajustes específicos se suman al núcleo en vez de
  reemplazarlo y se aprueban de forma independiente;
- `task_profile_resolved` persiste evidencia, scores, candidatas rechazadas y
  uso del fallback para replay;
- `bench/task_policy_eval.py` materializa 240 requests held-out y calcula
  métricas por política y autoridad.

El perfil se calcula por defecto, pero inyectar un fragmento requiere además un
gate de rollout. El mini-modelo decide método, pero no autoridad: modificar,
commit, push, publicación y negaciones dependen del texto literal. En
paráfrasis sin categoría explícita, el prompt sólo se activa si el head y el
recuperador contrastivo independiente coinciden. Una abstención del gate
discursivo es final. Una candidata de método del tier relajado tampoco puede
activarse sola: necesita evidencia literal o contrastiva coincidente.

## Compositor condicional activo

La clasificación ya no agrega el mismo párrafo a todos los agentes. El runtime
mantiene un núcleo estable y selecciona entre 22 fragmentos de método:

```text
núcleo estable: identidad + autoridad + seguridad + protocolo + herramientas
  -- cache breakpoint --
fragmentos por operación + rol + fase
  -> objetivo y evidencia de la iteración
```

Cada fragmento declara `policy_id`, roles, fases, operaciones, constraints y
autoridad requeridas, operaciones/constraints excluidas, prioridad, versión y
presupuesto UTF-8. La selección falla cerrada si la política no fue aceptada
por el `TaskProfile`, si falta autoridad o si un fragmento excede el budget.

Las variantes actuales son específicas para planner, developer y reviewer;
research agrega también una variante `researcher/investigate`. Por ejemplo,
una feature recibe contrato y aceptación en planning, un slice completo en
development y mapping de criterios a evidencia en review. Ninguna de esas
instrucciones entra en un refactor o una investigación.

El ejemplo fijo de bugfix fue eliminado del núcleo del developer y el ejemplo
de reparación del planner fue reemplazado por un ejemplo neutral de schema. En
una medición local, el núcleo developer quedó estable en 7.450 caracteres y la
capa condicional por tarea agregó entre 290 y 354 caracteres. El fallback de
function calling conserva la composición existente en vez de reconstruir un
prompt genérico y perder la política activa.

`prompt_composition_history` registra por iteración el ID, versión, hash y
policy de cada fragmento, además del tamaño estable/dinámico. No persiste el
contenido del prompt en esa telemetría.

El `TaskProfile` es estado interno: ya no se duplica en el XML del user prompt
ni modifica por sí mismo el catálogo de herramientas. Esto deja el tratamiento
aislado al fragmento de system prompt y reduce contexto repetido.

El rollout actual habilita `refactor.developer@1` únicamente para MiniMax M3 y
`bugfix.developer@3` únicamente para GPT-5.6 Terra. El detalle del segundo E2E
está en [`TASK_POLICY_BUGFIX_MODEL_E2E.md`](TASK_POLICY_BUGFIX_MODEL_E2E.md).

## Resumen

Infinidev ya adapta parte de su comportamiento al modelo, la fase de ejecución
y algunas señales explícitas del request. La propuesta es extender ese enfoque
con políticas de trabajo seleccionadas según la intención del usuario.

Una petición de refactorización, una investigación, una corrección de bug y una
implementación nueva tienen objetivos, riesgos y métodos de validación
diferentes. En lugar de cargar todas esas instrucciones en cada prompt,
Infinidev seleccionaría únicamente los fragmentos relevantes para la tarea
actual.

El nombre propuesto es **Conditional Task Policies** y no “conditional system
prompting”, porque:

- el system prompt y los guardrails globales deben permanecer estables;
- las políticas describen cómo abordar una tarea concreta;
- varias políticas pueden combinarse;
- una política nunca debe cambiar la autoridad concedida por el usuario.

## Motivación

Actualmente existen mecanismos relacionados pero independientes:

- políticas de ejecución específicas para ciertos modelos, como MiniMax M3;
- señales deterministas para distinguir requests informativos de requests de
  ejecución;
- selección dinámica de herramientas mediante patrones;
- prompts especializados para bugs, features y refactors;
- clasificación LLM en algunos flows;
- una estructura `Task` con tipo y criterios de aceptación;
- capas tipadas de comportamiento, ejecución, objetivo y evidencia.

Estas piezas pueden llegar a interpretar una misma petición de forma diferente.
La propuesta introduce un perfil de tarea compartido que se calcula una vez y
es reutilizado por el planner, el developer, el reviewer, el routing de
herramientas y los mecanismos de verificación.

## Principios

1. **La petición literal conserva la autoridad.** Una política no puede ampliar
   el alcance ni convertir una consulta en autorización para modificar.
2. **Las políticas añaden restricciones; no eliminan guardrails.**
3. **La clasificación es multi-label.** Una petición puede requerir investigar,
   implementar y preservar compatibilidad al mismo tiempo.
4. **Los prompts son fragmentos canónicos.** Los ejemplos sirven para
   seleccionar una política, pero nunca se inyectan directamente.
5. **Los casos claros no requieren una llamada LLM.**
6. **Los casos ambiguos usan una sola clasificación estructurada**, no una
   colección creciente de preguntas binarias.
7. **Toda decisión es observable.** El runtime debe registrar qué políticas
   seleccionó, mediante qué evidencia y con qué versión.
8. **La selección se evalúa offline antes de modificar comportamiento de
   producción.**

## Perfil de tarea

La clasificación no debería reducirse a una única categoría. El perfil puede
representar varios ejes independientes:

| Eje | Ejemplos |
| --- | --- |
| Operaciones | `bugfix`, `feature`, `refactor`, `research`, `review` |
| Autoridad | `answer`, `diagnose`, `modify`, `commit`, `publish` |
| Restricciones | `preserve_behavior`, `preserve_public_api`, `read_only` |
| Riesgos | `security`, `migration`, `destructive`, `external_write` |
| Resultado | `code`, `report`, `plan`, `recommendation` |
| Secuencia | investigar, implementar, verificar, publicar |

Ejemplo:

> Investiga por qué falla y después corrígelo sin cambiar la API pública.

```json
{
  "operations": ["research", "bugfix"],
  "authority": ["diagnose", "modify"],
  "constraints": ["preserve_public_api"],
  "risks": [],
  "result": ["code"],
  "sequence": ["investigate", "implement", "verify"]
}
```

Este perfil sería una extensión o un derivado versionado de `Task`, no una
segunda interpretación libre del objetivo.

## Pipeline de clasificación

La selección debería usar una cascada de menor a mayor coste:

```text
request
  -> mini-head Qwen: método o uncategorized
  -> señales literales: autoridad, negación y constraints
  -> acuerdo contrastivo para paráfrasis ambiguas
  -> resolución de negaciones y conflictos
  -> clasificación LLM estructurada solo si continúa ambiguo
  -> TaskProfile compartido
```

### 1. Señales deterministas

Los comandos y contratos explícitos tienen prioridad:

- `/refactor` selecciona la operación de refactor;
- “no cambies archivos” establece autoridad de solo lectura;
- “commit y push” autoriza esas operaciones externas concretas;
- una ruta o símbolo mencionado aporta grounding, pero no nueva autoridad;
- el `Task.kind` ya validado debe reutilizarse en lugar de reclasificarse.

Estas reglas deben ser conservadoras y de alta precisión.

### 2. Recuperación semántica

Los embeddings recuperan las políticas más próximas a la petición. Cada
política debería tener:

- varios ejemplos positivos;
- ejemplos negativos y contraejemplos;
- ejemplos en los idiomas soportados;
- casos con negación;
- casos mixtos con más de una operación;
- una descripción canónica independiente de los ejemplos.

No debería compararse el request contra una sola frase como “quiero
refactorizar”. Varios prototipos reducen la sensibilidad a una redacción
particular.

La recuperación semántica propone candidatas; no concede autoridad ni activa
por sí sola guardrails críticos.

### 3. Resolución de conflictos

Antes de usar un LLM deben resolverse patrones como:

- “No refactorices; solo explícame el problema.”
- “Investiga y luego impleméntalo.”
- “Revisa el PR, pero no cambies archivos.”
- “Optimiza esta función sin modificar la API.”
- “El error dice ‘refactor required’, ¿qué significa?”

La negación, el texto citado, la autoridad explícita y el orden de las acciones
son señales distintas de la similitud temática.

### 4. Fallback LLM

Si las candidatas tienen poco margen, se contradicen o la petición es
genuinamente compuesta, una única llamada LLM devuelve el perfil completo con
un schema cerrado.

No se deberían hacer llamadas separadas para preguntar si la tarea es refactor,
research, implementación o review. Además de aumentar coste y latencia, esas
respuestas pueden ser mutuamente incompatibles.

La confianza declarada por el modelo no debe autorizar acciones sensibles. El
runtime conserva las reglas deterministas de permisos y autoridad.

## Registro de políticas

Cada política podría definirse con un contrato semejante a:

```yaml
id: refactor.preserve_behavior
version: 1
operations: [refactor]
prompt_layer: task-policy
roles: [planner, developer, reviewer]
phases: [investigate, plan, execute, review]
priority: 50
max_utf8_bytes: 1200
incompatible_with: []
requires:
  authority: [modify]
content: |
  Preserve observable behavior. Establish a baseline, identify callers and
  tests, make one structural change at a time, and rerun the narrowest
  relevant verification after each boundary changes.
```

Campos importantes:

- identidad y versión estables;
- operaciones y restricciones que permiten seleccionarla;
- roles y fases donde es relevante;
- precedencia y políticas incompatibles;
- límite de tamaño;
- fragmento canónico;
- ejemplos positivos y negativos almacenados aparte;
- hash para evaluación y reproducibilidad.

## Composición del prompt

Las políticas de tarea deben ser una capa explícita, separada de la calibración
por modelo:

```text
guardrails y permisos invariantes
  -> política operativa del modelo
  -> instrucciones del repositorio
  -> políticas de la tarea
  -> objetivo literal y criterios de aceptación
  -> evidencia dinámica
```

Esto permite combinar, por ejemplo:

```text
MiniMax M3 execution policy
+ refactor.preserve_behavior
+ compatibility.preserve_public_api
```

sin crear un prompt monolítico específico para cada combinación de modelo y
tarea.

Las políticas:

- no reescriben el objetivo;
- no agregan trabajo no solicitado;
- no pueden relajar seguridad o permisos;
- no deberían duplicar instrucciones ya presentes;
- deben tener un presupuesto total pequeño;
- deben limitarse normalmente a dos o tres fragmentos por request.

La parte estable del system prompt debería conservar su posición para no perder
los beneficios del prompt caching. La capa dinámica puede insertarse después
del prefijo cacheable y antes del objetivo de la tarea.

## Políticas iniciales

La primera versión debería ser deliberadamente pequeña:

### `bugfix.root_cause`

- reproducir o establecer evidencia del fallo;
- distinguir síntoma de causa;
- realizar el cambio mínimo;
- añadir o ejecutar una regresión enfocada.

### `feature.contract_first`

- concretar comportamiento y criterios de aceptación;
- revisar integraciones y compatibilidad;
- evitar ampliar arquitectura sin necesidad;
- verificar el flujo nuevo de extremo a extremo.

### `refactor.preserve_behavior`

- establecer baseline;
- localizar consumidores y tests;
- preservar comportamiento observable;
- aplicar cambios estructurales incrementales.

### `research.evidence_first`

- separar hechos, inferencias y preguntas abiertas;
- priorizar fuentes primarias;
- citar evidencia;
- no modificar archivos salvo autorización explícita.

### `review.read_only`

- buscar defectos, regresiones y riesgos concretos;
- priorizar hallazgos por impacto;
- no implementar correcciones;
- aprobar explícitamente cuando no haya defectos materiales.

Después podrían añadirse políticas para migraciones, seguridad, performance,
documentación, publicación y operaciones destructivas.

## Evaluación

La clasificación y los prompts deben evaluarse por separado.

### Dataset

El dataset debería contener cientos de requests held-out:

- español, inglés y lenguaje mixto;
- paráfrasis;
- instrucciones breves y extensas;
- negaciones;
- texto citado;
- operaciones múltiples;
- cambios de autoridad;
- casos que no pertenecen a ninguna política;
- familias no vistas durante el ajuste de thresholds.

Cada ejemplo necesita labels multi-eje y una explicación revisada de la
autoridad concedida.

### Métricas de routing

- precision, recall y F1 por política;
- tasa de falsas activaciones;
- exactitud de autoridad de escritura;
- conflictos no resueltos;
- porcentaje que necesita fallback LLM;
- latencia y coste;
- estabilidad ante paráfrasis.

Las falsas activaciones de políticas restrictivas o sensibles deben ponderarse
más que no seleccionar una recomendación opcional.

### Métricas de tarea

El objetivo no es únicamente clasificar bien, sino mejorar ejecuciones reales:

- cumplimiento de criterios de aceptación;
- expansión indebida de alcance;
- cambios no autorizados;
- calidad y suficiencia de tests;
- número de llamadas y tokens;
- tiempo hasta el primer cambio material;
- repeticiones y estancamiento;
- resultado por proveedor y modelo.

La comparación debe incluir baseline sin políticas, política correcta, política
incorrecta deliberada y variantes de redacción. Los thresholds se calibran con
datos offline y se congelan antes de validar en el conjunto held-out.

## Observabilidad

Cada run debería registrar algo equivalente a:

```json
{
  "task_profile_version": 1,
  "selected_policies": [
    {
      "id": "refactor.preserve_behavior",
      "version": 1,
      "source": "embedding",
      "evidence": ["example:refactor-es-07"],
      "score": 0.84
    }
  ],
  "rejected_candidates": [
    {
      "id": "review.read_only",
      "reason": "explicit modify authority"
    }
  ],
  "llm_fallback_used": false
}
```

Los scores sirven para análisis y replay, no para presentarlos al modelo como
certeza ni para conceder permisos.

## Rollout y feature flags

- `INFINIDEV_TASK_POLICIES_ENABLED`: calcula y transporta el perfil.
- `INFINIDEV_TASK_POLICIES_SHADOW_MODE`: registra decisiones sin inyectar el
  perfil ni los fragmentos en prompts.
- `INFINIDEV_TASK_POLICIES_EMBEDDINGS_ENABLED`: habilita recuperación
  semántica de candidatas.
- `INFINIDEV_TASK_POLICIES_LLM_FALLBACK_ENABLED`: habilita una sola
  clasificación estructurada para ambigüedad real.
- `INFINIDEV_TASK_POLICIES_EMBEDDING_MIN_SCORE` y
  `INFINIDEV_TASK_POLICIES_EMBEDDING_MIN_MARGIN`: thresholds congelables.
- `INFINIDEV_TASK_POLICIES_MAX_SELECTED` y
  `INFINIDEV_TASK_POLICIES_MAX_UTF8_BYTES`: límites de composición.
- `INFINIDEV_TASK_POLICIES_SHOW_SELECTION`: muestra una línea de estado
  opcional en la UI.
- `INFINIDEV_TASK_POLICIES_EVIDENCE_GATED`: exige aprobación exacta de modelo,
  fragmento y versión antes de inyectar; está activo por defecto.

El artifact jerárquico v3 separa primero `task` de `uncategorized` y luego
clasifica el método. En el holdout histórico del head subió la precisión
selectiva de 82,0 % a 88,3 % y redujo falsas activaciones de 10 a 4, a costa de
menor cobertura. Ese holdout ya fue observado durante el desarrollo y no debe
tratarse como prueba futura de generalización. La ruta completa sobre 112
paráfrasis conserva 100 % de precisión selectiva, cero falsas activaciones y
cero falsa autoridad; su cobertura actual es 57,1 %. El benchmark literal
obtiene 0 % de falsa autoridad de escritura, 90 % de coincidencia exacta de
autoridad y 80 % de coincidencia exacta de políticas sobre 240 casos. Estos
números son una regresión local, no una afirmación de calidad cross-model.

El piloto prompt-only con MiniMax M3 está documentado en
[`TASK_POLICY_MINIMAX_M3_PILOT.md`](TASK_POLICY_MINIMAX_M3_PILOT.md). Sus dos
pares utilizables mejoraron el score mecánico de método, pero también aumentaron
latencia y un tercer escenario agotó el presupuesto sin respuesta completa; no
se usa como prueba de calidad end-to-end.

La evaluación anterior con loop, herramientas, cambios y verificadores reales está en
[`TASK_POLICY_MINIMAX_M3_E2E.md`](TASK_POLICY_MINIMAX_M3_E2E.md). Baseline y
candidate pasaron 2/2, mientras candidate consumió 22,2 % más prompt tokens,
29,9 % más completion tokens y 20 % más herramientas. Esta evidencia motivó
el diseño conservador del acuerdo doble.

La evaluación prompt-only actual está en
[`TASK_POLICY_IMPROVEMENT_E2E.md`](TASK_POLICY_IMPROVEMENT_E2E.md). El par
refactor conservó resultado y scope mientras redujo 82,6 % de prompt tokens y
69,2 % de tools. Bugfix empeoró coste y fue retirado del rollout de MiniMax; el
negativo `uncategorized` abstuvo sin inyección. Esto aprueba solamente
`refactor.developer@1` para MiniMax M3, no el sistema completo.

El contrato de datos, el pool draft de 672 ejemplos multilingües y sus límites
de leakage están en [`TASK_POLICY_DATASET.md`](TASK_POLICY_DATASET.md). El
auditor conserva `release_ready=false` hasta aislar familias de frase y aprobar
las filas manualmente.

## Riesgos

### Explosión combinatoria

Crear un prompt completo por modelo, tipo de tarea y fase no escala. Las
políticas deben ser ortogonales y composables.

### Prompt inflation

Seleccionar demasiadas instrucciones puede empeorar el seguimiento. Debe haber
deduplicación, prioridades y un presupuesto estricto.

### Falsa autoridad

Una coincidencia semántica nunca debe autorizar escrituras, commits,
publicaciones o acciones destructivas.

### Clasificación rígida

Una taxonomía cerrada puede deformar requests poco comunes. El perfil admite
varias operaciones y un estado sin política aplicable.

### Leakage de evaluación

Los ejemplos usados para crear embeddings, elegir thresholds o redactar
políticas no deben aparecer en la validación held-out.

### Diferencias entre modelos

Una política que ayuda a un modelo puede perjudicar a otro. La selección de
tarea y la calibración por modelo son ejes separados, y el efecto combinado
debe medirse por ruta de proveedor/modelo.

## Decisiones adoptadas

- `TaskProfile` es un artefacto derivado que forma parte de `Task` y
  `GoalSpec`; no reemplaza el objetivo literal.
- la selección semántica se fija al espacio exacto
  `ken/static-qwen3-r512-v2` y registra su `space_id`; no acepta silenciosamente
  otro backend de igual dimensión;
- señales literales de alta precisión activan políticas sin una llamada LLM;
- la secuencia es una lista ordenada y las acciones externas requieren su
  autoridad literal correspondiente;
- la capa dinámica se inserta después del prompt estable y la calibración por
  modelo, antes del objetivo/evidencia de la iteración;
- planner, developer y reviewer reciben únicamente fragmentos aplicables a su
  rol y fase;
- la selección es silenciosa por defecto y observable en el event log; la
  línea del TUI es opt-in.

## Criterio de éxito

La propuesta tiene éxito si las políticas condicionales mejoran la ejecución de
tareas especializadas sin:

- modificar la autoridad literal del usuario;
- aumentar cambios fuera de alcance;
- inflar sustancialmente el prompt;
- duplicar clasificaciones en distintos subsistemas;
- depender de una llamada LLM para requests explícitos;
- degradar modelos o rutas que funcionaban mejor con el baseline.

## Extensión: políticas adaptativas durante la ejecución

La misma arquitectura puede clasificar ventanas de trayectoria y activar una
intervención corta en el momento donde resulte útil: estancamiento, retries
equivalentes, scope drift, ausencia de verificación o claims especulativos.
Esta extensión usa únicamente mensajes y estado observables, no chain-of-thought
privado, e incluye histéresis, cooldown y expiración. El diseño completo está
en [`ADAPTIVE_RUNTIME_POLICIES.md`](ADAPTIVE_RUNTIME_POLICIES.md).
