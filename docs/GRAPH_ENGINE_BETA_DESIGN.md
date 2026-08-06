# Graph Engine beta: orquestación adaptativa para Infinidev

**Estado:** implementada y activada como beta
**Fecha:** 2026-08-06
**Objetivo:** sustituir el plan rígido como única forma de trabajo por un sistema que pueda
elegir entre ReAct, Staged y un grafo dinámico, conservando control, trazabilidad y capacidad
de reanudar.

## 1. Conclusión

La idea es viable y encaja bien con el problema, pero el núcleo no debería ser simplemente
un árbol de tareas que admite enlaces adicionales. Debería ser un **grafo de trabajo tipado,
versionado y respaldado por un registro de eventos append-only**.

La arquitectura recomendada conserva tres motores:

- **ReAct** para tareas pequeñas, locales y de bajo riesgo.
- **Staged** para trabajo predecible que se beneficia de hitos y aprobación explícita.
- **Graph beta** para problemas no lineales, ambiguos o cuyos requisitos pueden cambiar
  mientras se trabaja.
- **Auto** como coordinador que elige el motor, explica la decisión y puede cambiarlo con
  límites claros.

Graph no debe reemplazar Staged desde el primer día. Conviene introducir una interfaz común,
añadir primero ReAct y el registro de eventos, y activar Graph detrás de una opción beta.

## 2. Problema que se quiere resolver

Un plan de stages, phases y steps es útil cuando el trabajo puede anticiparse. Pierde calidad
cuando:

- aparecen dependencias o restricciones durante la exploración;
- dos ramas comparten evidencia o artefactos;
- una hipótesis queda invalidada;
- el usuario modifica el objetivo a mitad de ejecución;
- una tarea debe suspenderse para desbloquear otra;
- el agente necesita explicar después por qué tomó una decisión;
- una tarea trivial paga el coste de una maquinaria de planificación excesiva.

Dar libertad total tampoco es suficiente. Un agente sin estructura puede repetir búsquedas,
abrir demasiadas ramas, gastar tokens sin reducir incertidumbre o declarar éxito sin haber
cerrado los requisitos importantes. El diseño debe permitir exploración local sin perder
invariantes globales.

## 3. Estado actual relevante

La arquitectura existente ya contiene piezas reutilizables:

- El motor Staged representa dependencias, evidencia, ejecución secuencial, reanudación y una
  compuerta de finalización.
- El loop de desarrollo admite mensajes inyectados y steps inferidos por el modelo.
- La sesión conserva tanto conversación compacta como mensajes visibles y eventos de runtime.
- Working memory y contextos relacionados ya usan recuperación vectorial.
- Ken ya resuelve navegación semántica y referencias al código.

También hay límites que esta propuesta debe corregir:

- La escalación termina siempre en Staged; no hay una selección de motor configurable.
- GoalSpec, StageSpec y TaskSpec se comportan como una especificación esencialmente congelada.
  La nueva orientación del usuario no tiene todavía una semántica formal de revisión e
  invalidación.
- El TreeEngine actual sirve para exploración o brainstorming con profundidad controlada. No
  es todavía un grafo persistente de ejecución.
- Los eventos de runtime no forman por sí solos una memoria causal consultable.
- Las salidas completas de comandos tienen restricciones de visibilidad y retención que deben
  respetarse al indexar historial.

Por tanto, la nueva implementación debe reutilizar contratos y persistencia existentes, sin
confundir el TreeEngine actual con el Graph Engine propuesto.

## 4. Modelo conceptual: un grafo en varias capas

Un único grafo mezclando intención, tareas, código y tool calls se volvería inmanejable. Es
preferible separar capas conectadas:

1. **Grafo de intención**
   - revisiones del objetivo;
   - requisitos;
   - criterios de aceptación;
   - restricciones y preferencias del usuario.
2. **Grafo de trabajo y decisiones**
   - preguntas;
   - hipótesis;
   - decisiones;
   - unidades de trabajo;
   - verificaciones;
   - bloqueos y alternativas.
3. **Grafo de evidencia y ejecución**
   - resultados de tools;
   - observaciones;
   - artefactos producidos;
   - checkpoints;
   - eventos del agente.
4. **Referencias al grafo de código**
   - símbolos, archivos, rutas, llamadas o resultados de Ken;
   - identificadores y hashes, no copias completas del código.

El nodo raíz inicial es el prompt del usuario, pero cada modificación posterior crea una nueva
revisión del objetivo. La raíz lógica es entonces la identidad del objetivo; sus revisiones
forman una secuencia causal.

### 4.1 Tipos de nodo iniciales

- goal_revision
- requirement
- question
- hypothesis
- decision
- work
- verification
- evidence
- blocker
- artifact_ref
- code_ref

Conviene empezar con pocos tipos y extenderlos sólo cuando exista una diferencia real de
semántica. Un nodo guarda, como mínimo:

- id estable;
- tipo;
- título y objetivo;
- expected outcome verificable;
- estado de ciclo de vida;
- veredicto, si aplica;
- revisión del objetivo que lo gobierna;
- prioridad y presupuesto;
- autor, timestamps y versión;
- checkpoint resumido;
- referencias a evidencia y artefactos.

### 4.2 Tipos de arista

- decomposes_into
- requires
- alternative_to
- blocks
- satisfies
- supports
- contradicts
- produced_by
- targets
- supersedes
- invalidates

Las aristas también deben poder tener autor, versión, confianza y evidencia. La frase
OAuthService es usado por AuthMiddleware es una relación semántica; la frase verificar todos
los endpoints requiere obtener la lista de endpoints es una dependencia dura de ejecución.
No deben tratarse igual.

### 4.3 Estado, veredicto y frescura

No conviene concentrar todo en pending, in_progress y completed. Son dimensiones diferentes:

- **Ciclo de vida:** proposed, ready, active, suspended, resolved, abandoned.
- **Veredicto:** unknown, confirmed, rejected, inconclusive.
- **Frescura:** current, stale, invalidated.

Un nodo puede estar resolved pero stale después de un cambio del usuario. La evidencia puede
estar confirmed y, aun así, haber sido invalidada por cambios posteriores en el repositorio.

### 4.4 Ciclos permitidos y ciclos prohibidos

El grafo semántico puede tener ciclos. El subgrafo de dependencias duras debe ser un DAG. El
reducer rechazará una arista requires que cree un ciclo o la degradará a relación informativa,
según el caso.

Esto permite usar:

- orden topológico para trabajo ejecutable;
- componentes fuertemente conexos para detectar ciclos semánticos densos;
- alcance inverso para invalidar trabajo afectado;
- búsqueda best-first para seleccionar el siguiente nodo;
- trazado causal para responder por qué se hizo algo.

No se recomienda comenzar con MCTS o búsquedas especulativas costosas. ReAct, Graph of
Thoughts y LATS ofrecen ideas útiles, pero Infinidev necesita primero un grafo de estado fiable,
presupuestos explícitos y resultados reproducibles.

## 5. Scheduler: libertad local con control global

El agente puede moverse a otro nodo sin terminar el actual, pero ese movimiento debe producir
un checkpoint. El scheduler elige entre nodos ready usando una función explicable, por ejemplo:

1. instrucciones nuevas del usuario;
2. trabajo autorizado y directamente solicitado;
3. nodos que desbloquean requisitos críticos;
4. ganancia de información esperada dividida por coste;
5. afinidad con el contexto ya cargado;
6. antigüedad, para evitar inanición.

La puntuación exacta puede evolucionar. Lo importante es persistir la razón de selección y no
permitir que el modelo cambie directamente el estado global sin pasar por operaciones
validadas.

### 5.1 Un solo escritor

Al principio debe existir un único escritor del grafo. La exploración y las tools pueden ser
concurrentes, pero las mutaciones se serializan mediante un reducer transaccional. Esto evita
estados imposibles y hace que replay, resume y depuración sean deterministas.

### 5.2 Checkpoint al suspender un nodo

Antes de abandonar temporalmente un nodo activo se registra:

- qué se intentaba conseguir;
- qué se aprendió;
- qué falta;
- qué evidencia se produjo;
- qué supuestos siguen abiertos;
- por qué se suspende;
- cuál sería el siguiente paso seguro;
- qué archivos, símbolos o tool calls son relevantes.

Así, saltar entre ramas no depende de conservar todo el transcript en la ventana del modelo.

### 5.3 Context capsule por nodo

Cada activación construye un NodeContextCapsule con:

- revisión vigente del objetivo;
- objetivo y expected outcome del nodo;
- ancestros autoritativos;
- resumen de dependencias;
- evidencia relevante;
- checkpoint previo;
- cambios recientes del usuario;
- referencias de Ken o ContextRank;
- presupuesto restante;
- vecinos relevantes, no el grafo completo.

El contexto sigue conceptualmente el stack de exploración, pero se materializa como una cápsula
reconstruible. Esto permite volver a un nodo desde otro camino y aprovechar evidencia
compartida sin duplicar todos sus ancestros.

### 5.4 Profundidad lógica ilimitada, recursos limitados

No hace falta fijar un límite de profundidad semántica. Sí hacen falta límites operativos:

- fan-out máximo por expansión;
- ramas abiertas simultáneas;
- revisitas por nodo;
- tokens y tool calls por nodo;
- presupuesto total de coste y tiempo;
- tamaño máximo de cápsula;
- umbral mínimo de utilidad para abrir una rama.

Agotar un presupuesto nunca equivale a completar el objetivo. El resultado correcto es
suspended, blocked o needs_user_input, con una explicación.

## 6. Mutaciones mediante protocolo

El modelo no debería editar filas o estructuras internas libremente. Debe emitir operaciones
de dominio equivalentes a:

~~~text
graph_patch(
  add_nodes=[...],
  add_edges=[...],
  update_nodes=[...],
  rationale="...",
  based_on_revision=17
)

checkpoint_node(node_id="...", reason="dependency_unblocked_elsewhere")
resolve_node(node_id="...", evidence_ids=[...], outcome="...")
resolve_goal(revision_id="...", evidence_ids=[...])
~~~

El reducer valida invariantes:

- los ids existen;
- la revisión base no está obsoleta;
- no se crean ciclos duros;
- los nodos resueltos poseen evidencia suficiente;
- no se abandona un requisito activo sin una revisión que lo autorice;
- toda mutación produce eventos auditables;
- sólo puede haber una operación de escritura aplicada por versión.

## 7. Cambios del usuario durante la ejecución

Una intervención del usuario no debe reescribir el pasado. Debe crear una GoalRevision y
clasificarse como:

- aclaración;
- restricción adicional;
- nuevo requisito;
- eliminación de requisito;
- cambio de prioridad;
- contradicción;
- reemplazo total del objetivo;
- solicitud de pausa o cancelación.

Después se calcula el alcance inverso desde los requisitos afectados:

1. marcar nodos dependientes como stale o invalidated;
2. conservar trabajo y evidencia que continúen siendo válidos;
3. crear nodos de reparación o reverificación;
4. reordenar la frontera ejecutable;
5. preguntar al usuario sólo si existe un conflicto que no puede resolverse con la autoridad
   ya disponible.

No se borran nodos históricos. Se usan supersedes, invalidates y tombstones para conservar la
explicación causal.

## 8. Los motores y cuándo utilizarlos

### 8.1 ReAct

Adecuado para:

- preguntas y cambios locales;
- una o pocas tools;
- bajo riesgo;
- expected outcome inmediato;
- tareas donde construir un plan cuesta más que ejecutarlo.

Ventajas: mínima latencia y overhead. Riesgo: improvisación excesiva cuando el alcance crece.

### 8.2 Staged

Adecuado para:

- entregables previsibles;
- trabajo con hitos claros;
- migraciones con orden conocido;
- tareas donde el usuario necesita revisar o aprobar etapas;
- ejecución donde la transparencia del plan es prioritaria.

Ventajas: estructura y legibilidad. Riesgo: replanning tosco ante cambios frecuentes.

### 8.3 Graph beta

Adecuado para:

- requisitos incompletos o volátiles;
- dependencias descubiertas durante el trabajo;
- investigación y ejecución entrelazadas;
- evidencia compartida entre ramas;
- contradicciones, alternativas y reverificación;
- tareas largas con interrupciones.

Ventajas: adaptación y navegación no lineal. Riesgo: overhead, crecimiento sin control y falsa
sensación de progreso si no existen buenas compuertas de finalización.

### 8.4 Auto

Auto no es un cuarto loop completo. Es un coordinador que clasifica el trabajo, propone un
motor y conserva la autoridad para hacer transiciones permitidas.

La decisión debe generar un EscalationPacket estructurado:

~~~json
{
  "engine": "react",
  "confidence": 0.86,
  "reasons": ["single_local_change", "low_uncertainty"],
  "risks": ["scope_may_expand_after_repository_inspection"],
  "reconsider_if": ["more_than_three_components", "new_requirement"],
  "estimated_overhead": "low"
}
~~~

El usuario debe poder ver una explicación breve y cambiar la elección.

### 8.5 Transiciones entre motores

Transiciones iniciales permitidas:

- ReAct a Staged cuando aparece una secuencia larga pero predecible.
- ReAct a Graph cuando aparecen ramas, contradicciones o requisitos volátiles.
- Staged a Graph cuando una revisión invalida parte importante del plan.
- Graph a ReAct para ejecutar una hoja pequeña, conservándola como nodo del grafo.

Para evitar oscilación:

- máximo de cambios por run;
- umbral de confianza;
- tiempo mínimo de permanencia;
- razón persistida;
- confirmación del usuario para cambios de alto impacto.

Graph puede usar ReAct como ejecutor de hojas. Eso no significa abandonar el grafo; el
resultado del episodio ReAct vuelve como evidencia y eventos del nodo activo.

## 9. Configuración y TUI

Configuración inicial propuesta:

~~~text
TASK_ENGINE_MODE=auto|react|staged|graph_beta
AUTO_ENGINE_ALLOW_GRAPH=false
ENGINE_SHOW_SELECTION_REASON=true
GRAPH_MAX_OPEN_BRANCHES=8
GRAPH_MAX_NODE_REVISITS=4
GRAPH_NODE_TOKEN_BUDGET=...
GRAPH_RUN_TOOL_BUDGET=...
~~~

En /settings:

- selector del motor por defecto;
- etiqueta Beta para Graph;
- explicación de ventajas y costes;
- activación separada de Graph dentro de Auto;
- presupuestos avanzados en una sección desplegable;
- indicación clara de si el cambio aplica al siguiente turno o al run actual.

Cambiar el valor por defecto no debe alterar silenciosamente una ejecución activa. Un cambio
durante el run se representa como evento, por ejemplo /engine graph, y pasa por las reglas de
transición.

La vista Graph debería priorizar legibilidad, no dibujar el grafo completo:

- motor actual y razón de elección;
- nodo enfocado;
- breadcrumb causal;
- contadores de ready, active, blocked, stale y resolved;
- ramas críticas;
- vecinos y dependencias inmediatas;
- revisión vigente del objetivo;
- acción Pause graph.

El TreeEngine existente debería renombrarse gradualmente a ExplorationTreeEngine para evitar
confusión conceptual y mantener compatibilidad mientras se migra.

## 10. Event log e historial consultable

El grafo es una proyección; la fuente canónica debe ser un registro de eventos append-only.
Esto permite reconstrucción, auditoría, migraciones y respuestas causales.

Entidades persistentes sugeridas:

- engine_runs;
- execution_events;
- graph_nodes;
- graph_edges;
- history_entries;
- history_digests.

graph_nodes y graph_edges son una proyección actualizada por el reducer. execution_events
conserva la verdad histórica.

Eventos relevantes:

- run_started;
- engine_selected;
- engine_switched;
- goal_revised;
- graph_patched;
- node_activated;
- node_checkpointed;
- tool_requested;
- tool_started;
- tool_progressed;
- tool_finished;
- evidence_attached;
- node_resolved;
- node_invalidated;
- run_paused;
- run_resumed;
- run_cancelled;
- digest_created.

Cada evento debe contener run_id, sequence, timestamp, actor, parent_event_id, goal_revision,
node_id opcional, visibilidad, payload versionado y hash del contenido cuando corresponda.

### 10.1 Tools de historial

Un grupo pequeño es preferible a muchas tools solapadas:

#### history_search

Búsqueda híbrida semántica y estructurada.

Filtros:

- session_id o run_id;
- rango temporal;
- tipo de evento;
- tool;
- nodo;
- archivo o símbolo;
- visibilidad;
- límite;
- número de eventos alrededor del resultado.

Devuelve snippets, score, ids estables y un motivo de coincidencia.

#### history_read

Lee eventos o mensajes concretos por id, incluyendo una ventana antes y después. Puede
recuperar el payload completo sólo cuando la política de seguridad lo permite.

#### history_trace

Reconstruye una cadena causal para preguntas como:

- por qué se editó este archivo;
- qué evidencia llevó a esta decisión;
- qué cambió después del mensaje del usuario;
- qué tool produjo este artefacto;
- por qué quedó invalidada una verificación.

Las respuestas de estas tools deben usar referencias como history:event-id. No deben
reinyectarse como si fueran tool calls nativas del proveedor, porque eso puede romper el
protocolo de conversación.

### 10.2 Episodio de recuperación efímero

Cuando falta contexto, el motor abre un episodio ReAct de sólo lectura:

1. formula la pregunta de recuperación;
2. llama a history_search, history_read o history_trace;
3. sintetiza un digest con afirmaciones y referencias;
4. descarta del contexto activo los resultados intermedios;
5. incorpora sólo el digest al nodo o turno padre.

Los eventos del episodio se guardan con visibilidad archive_only. No se indexan para
recuperación hasta cerrar el episodio, evitando que una búsqueda encuentre sus propias
búsquedas y cree un bucle de contexto.

### 10.3 Seguridad y retención

No debe incrustarse ciegamente todo:

- secretos detectados se redactan antes de persistir o indexar;
- salidas privadas permanecen privadas;
- contenido binario se representa mediante metadatos;
- resultados enormes se fragmentan y deduplican;
- cada entrada conserva procedencia, hash y política de retención;
- el acceso de las history tools respeta las mismas reglas de sesión, workspace y permisos
  que el contenido original.

Los embeddings son un índice, no la fuente de verdad. El registro original y sus ids permiten
verificar cada afirmación.

### 10.4 Relación con el embedder de Ken

Ken ya expone las primitivas útiles para incrustar pasajes y consultarlos, pero Infinidev no
debería depender implícitamente de una copia privada bajo un home directory.

Hay dos opciones:

1. dependencia opcional y fallback a la implementación actual de Infinidev;
2. extraer el embedder y su protocolo de índices a un paquete o servicio compartido.

La segunda es más limpia a medio plazo. En ambos casos cada vector debe guardar:

- embedding_model;
- dimensión;
- space_version;
- estrategia de chunking;
- hash del texto normalizado.

Cambiar el modelo o la dimensión requiere un nuevo espacio y una reindexación explícita; no se
mezclan vectores incompatibles en la misma búsqueda.

## 11. Pausa, cancelación y salida

Hay que distinguir cuatro acciones:

- **Cancelar tool en foreground:** detiene sólo la tool activa y notifica al agente.
- **Pausar run:** checkpoint del nodo, digest del run y estado reanudable.
- **Cancelar run:** termina la ejecución, conserva historia y explica el estado incompleto.
- **Abandonar objetivo:** cierra la intención activa, sin borrar el registro.

La orden ambigua Detente debería significar Pause por defecto, salvo que el contexto indique
claramente una cancelación total. El agente confirma en una frase qué quedó detenido.

Antes de salir se crea un RunDigest:

- objetivo original y revisión vigente;
- motor usado y transiciones;
- trabajo completado;
- trabajo activo, suspendido y bloqueado;
- decisiones y alternativas;
- cambios de archivos y artefactos;
- verificaciones y resultados;
- errores y riesgos;
- razones causales principales;
- próximos pasos;
- referencias a eventos, nodos y evidencia.

El digest facilita reanudar y contestar preguntas rápidas, pero no sustituye el event log. Ante
Por qué hiciste eso, history_trace recupera la cadena causal y el modelo responde desde hechos
referenciables.

## 12. Refactor propuesto

Una organización posible:

~~~text
src/infinidev/engine/orchestration/
  coordinator.py
  contracts.py
  routing.py
  transitions.py

src/infinidev/engine/engines/
  base.py
  react.py
  staged_adapter.py
  graph/
    domain.py
    reducer.py
    scheduler.py
    context.py
    completion.py
    persistence.py

src/infinidev/engine/history/
  events.py
  store.py
  projection.py
  retrieval.py
  digest.py
  redaction.py

src/infinidev/tools/history/
  search.py
  read.py
  trace.py
~~~

Primero se crean adaptadores alrededor del comportamiento actual. No conviene mover todo el
código antes de estabilizar los contratos.

EngineResult debería normalizar, como mínimo:

- status;
- user_message;
- summary;
- artifacts;
- evidence;
- resume_token;
- engine_name;
- transition_request;
- metrics.

El Graph Engine no debe tener Stage o Step como objetos de dominio obligatorios. Puede
presentar vistas de stages para compatibilidad o UI. De igual forma, un episodio ReAct puede
aparecer como ejecución de una hoja Graph sin convertirse en un plan.

## 13. Plan de implementación

### Fase 1: router en shadow mode

- Extraer características de tareas reales.
- Pedir al clasificador que elija motor sin cambiar la ejecución.
- Registrar decisión, confianza y resultado observado.
- Medir cuánto se habría equivocado.

### Fase 2: contratos y settings

- Introducir Engine, EngineResult, EscalationPacket y transiciones.
- Añadir /settings con react, staged, graph_beta y auto.
- Envolver Staged en StagedAdapter sin alterar su comportamiento.

### Fase 3: ReAct normal

- Implementar loop simple con presupuesto y compuerta de finalización.
- Usarlo manualmente primero.
- Añadir escalación a Staged o Graph.

### Fase 4: event log e history tools

- Definir schemas versionados.
- Emitir eventos desde los motores existentes.
- Implementar history_search, history_read y history_trace.
- Añadir digest efímero y reglas de redacción.

### Fase 5: dominio Graph puro

- Nodo, arista, revisión y eventos.
- Reducer determinista.
- Validación de invariantes.
- Replay y proyecciones.
- Sin LLM ni ejecución real todavía.

### Fase 6: scheduler y context capsules

- Frontera ready.
- scoring explicable;
- checkpoints;
- presupuestos;
- invalidación;
- completion gate.

### Fase 7: ejecución real

- Graph Engine ejecuta hojas mediante ReAct.
- Evidencia y artefactos vuelven al grafo.
- Soporte de pausa, resume y cambios del usuario.

### Fase 8: TUI y restauración

- Vista enfocada del grafo.
- Revisiones y estados.
- Reanudación completa desde event log y digest.
- Herramientas de inspección y exportación.

### Fase 9: Auto con Graph

- Activación beta explícita.
- Política conservadora.
- Histeresis y límites de transición.
- Comparación continua con elección manual.

## 14. Pruebas y experimento

Casos representativos:

1. pregunta trivial;
2. edición local de un archivo;
3. bug conocido con ruta de solución predecible;
4. agregar JWT a todos los endpoints;
5. el usuario cambia de JWT a sesiones a mitad del trabajo;
6. una evidencia contradice una hipótesis;
7. se cancela una tool y se reanuda el nodo;
8. el usuario pregunta por qué se editó un archivo;
9. la aplicación se reinicia y reanuda el run;
10. una dependencia propuesta crearía un ciclo;
11. una verificación queda stale después de editar código relacionado.

Métricas:

- tasa de éxito;
- falsos completados;
- tokens y tool calls;
- tiempo y coste;
- búsquedas repetidas;
- ramas abiertas y abandonadas;
- churn del grafo;
- cambios de motor;
- tiempo hasta detectar bloqueo;
- recuperación después de cambio del usuario;
- exactitud tras resume;
- exactitud de respuestas históricas;
- router regret frente al mejor motor observado.

El criterio de éxito no es que Graph gane siempre. Es que:

- ReAct gane en tareas simples;
- Staged siga siendo competitivo en trabajo lineal;
- Graph mejore tareas no lineales o cambiantes;
- Auto se acerque al mejor motor sin imponer overhead significativo.

## 15. Decisiones abiertas

### Solubles dentro del diseño

- El grafo tipado y versionado.
- El reducer y scheduler deterministas.
- La revisión e invalidación por cambios del usuario.
- La pausa, reanudación y explicación causal.
- El episodio efímero de recuperación.

### Necesitan decisión de producto o seguridad

- Qué contenido histórico puede incrustarse.
- Cuánto tiempo se retienen outputs completos.
- Si el embedder compartido será paquete, proceso o servicio.
- Qué información de razonamiento se guarda. Deben persistirse decisiones, observaciones y
  razones operativas, no depender de almacenar razonamiento privado bruto.
- Cuándo Auto puede activar Graph sin consentimiento explícito.

## 16. Primer corte vertical recomendado

El primer corte no debería intentar implementar el Graph Engine completo. La secuencia más útil
es:

1. settings de selección;
2. coordinator común;
3. StagedAdapter;
4. ReAct simple;
5. event log;
6. history_search, history_read y history_trace;
7. digest al pausar.

Este corte ya resuelve el overkill de Staged para tareas simples y el problema de perder
contexto histórico. Después puede construirse el reducer y scheduler Graph sobre una base
observable y testeable.

## 17. Nota operativa sobre Ken

La versión copiada del embedder debe integrarse mediante una interfaz estable, no importando
archivos directamente desde una ruta personal. Si se adopta el modelo estático previsto, la
reindexación debe ser explícita, por ejemplo:

~~~text
ken reembed --model ken/static-qwen3-r512-v2
~~~

El objetivo es mantener búsqueda multilingüe, la misma dimensión de vectores y menor coste de
inferencia, sin volver a parsear el código si sólo cambia el espacio de embeddings.

## 18. Referencias conceptuales

- ReAct: https://iclr.cc/virtual/2023/poster/11003
- Graph of Thoughts: https://ojs.aaai.org/index.php/AAAI/article/download/29720/31236
- LATS: https://proceedings.mlr.press/v235/zhou24r.html
- Discusión crítica sobre búsqueda y agentes: https://aclanthology.org/2024.acl-long.738/

Estas referencias inspiran estrategias de exploración, pero no sustituyen los contratos de
persistencia, seguridad, revisión del objetivo y finalización que Infinidev necesita.
