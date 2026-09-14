# Auditoría del ciclo de vida del harness Task

Fecha: 2026-09-13. Alcance: análisis y propuestas; sin cambios al runtime.

## Resultado del análisis

Task es una buena unidad para conducir una ejecución y comprimir su contexto. No es
todavía un contrato completo de continuidad de proyecto: su identidad se reconstruye
desde cada pedido, los mensajes nuevos no versionan una especificación compartida y
las evidencias de finalización tienen semánticas distintas según el camino de cierre.

La propuesta principal es conservar su ejecución adaptativa y separar tres cosas:
**compromisos del usuario, acciones para cumplirlos y evidencia de que se cumplieron**.
Los Steps seguirían siendo útiles como unidades de trabajo y memoria. El estado del
proyecto y la decisión de entregar no dependerían de que esa lista esté vacía.

Se registran 16 hallazgos y 64 situaciones de producto. Hay 11 probes locales que
caracterizan conductas del código con modelos/colaboradores simulados. No son una
medición de frecuencia de fallos de un LLM ni una comparación empírica de engines.
La preferencia del usuario por Task se toma como experiencia de uso, no como benchmark.

Se estudia la selección explícita `task`. El default declarado en
`config/settings.py:308` es `orchestrator`; no se leyó configuración privada del
operador ni se presupone que ese default sea el modo que usa habitualmente.

Las referencias de código son relativas a `src/infinidev/`, salvo indicación contraria.
Los números de línea corresponden al checkout auditado el 2026-09-13.

## Recorrido actual auditado

```text
El cliente recibe el mensaje.
  Si está ocupado:
    TUI lo inyecta al engine; Web lo inyecta solo durante execute.
    En otras fases Web lo difiere a un nuevo turno.
  Si no está ocupado:
    llama a run_task y abre un TaskRuntime para esa entrada.

run_task agrega contexto de sesión/repo y pasa por ChatAgent.
  Si responde: devuelve el texto y cierra el turno.
  Si escala: elaboración opcional, council si fue solicitado y políticas de tarea.

Con selección task, TaskAdapter construye el contrato desde el pedido literal.
  Crea un Step inicial sin llamada de planner.
  Configura horizonte de tres Steps, sin límites por defecto de Steps/tools.
  Ejecuta LoopEngine, que reconstruye contexto y procesa herramientas dentro del Step.

Cuando el modelo llama step_complete:
  varios gates pueden rechazarlo;
  se reconcilian pasos pendientes y evidencia de edición;
  se actualiza el plan, se resume y se archivan resultados;
  se decide continuar, terminar, bloquear o salir por límites/cancelación.

Si el loop termina completed:
  con cambios, verificación y revisión de código con rework;
  sin cambios, revisión informativa con rework.

TaskAdapter devuelve EngineResult.
El pipeline procesa hooks/continuación autónoma, guarda resumen y cierra el runtime.
El cliente muestra el texto final y procesa su cola pendiente.
```

| Pieza | Fuente |
|---|---|
| Orquestación entre fases | [pipeline.py](../src/infinidev/engine/orchestration/pipeline.py) |
| Adaptador Task | [task.py](../src/infinidev/engine/engines/task.py) |
| Ejecución y terminación | [engine.py](../src/infinidev/engine/loop/engine.py) |
| Gates de Step | [step_complete_gate.py](../src/infinidev/engine/loop/step_complete_gate.py) |
| Plan, archivo y resumen | [step_manager.py](../src/infinidev/engine/loop/step_manager.py) |
| Revisión de cambios | [review_engine.py](../src/infinidev/engine/analysis/review_engine.py) |
| Revisión informativa | [evidence_review.py](../src/infinidev/engine/analysis/evidence_review.py) |
| Continuidad estructural | [resume_checkpoint.py](../src/infinidev/engine/loop/resume_checkpoint.py) |

La documentación general de `CLAUDE.md` conserva detalles anteriores como un máximo
de cuatro tools por Step y pasos iniciales tratados globalmente como aprobados. Para
esta auditoría prevalece el código de Task y las pruebas de la versión actual.

## Modelo ideal independiente de la implementación

Un harness mantiene compromisos y coordina decisiones, ejecución y evidencia en un
mundo cambiante. El objetivo no es reproducir un organigrama mediante un agente por
puesto. Producto define el resultado y los límites; ingeniería reduce incertidumbre;
desarrollo construye; QA intenta refutar; operaciones valida entrega y recuperación.
Estas responsabilidades pueden asumirlas un modelo, herramientas deterministas o
colaboradores separados, según el trabajo.

### Invariantes deseables

1. Un mensaje nuevo tiene relación explícita con el trabajo existente; no lo reemplaza
   accidentalmente ni se pierde al cerrar una fase.
2. Un compromiso conserva identidad aunque cambie su redacción o su plan.
3. Toda ampliación o reducción de alcance conserva autor y motivo. Una táctica del
   modelo no se convierte por accidente en un requisito del usuario.
4. Cada resultado afirma solamente lo que respalda la evidencia vigente.
5. Una comprobación deja de respaldar el resultado cuando cambian sus dependencias.
6. Agotar tiempo, tokens o reintentos no demuestra ni éxito ni bloqueo externo.
7. Las acciones con efectos conservan intención, resultado y estado incierto cuando
   el proceso se interrumpe. Reanudar no significa repetir a ciegas.
8. Una decisión del usuario que cambia el objetivo invalida las acciones y revisiones
   preparadas contra la versión anterior en los puntos de ejecución correspondientes.
9. El trabajo pequeño paga poca coordinación; el grande conserva una visión global
   sin planificar prematuramente cada paso.
10. La entrega distingue construir, verificar, integrar, publicar y observar. Solo
    incluye los hitos necesarios para el alcance autorizado.

### Esqueleto de comportamiento

```text
Primero el usuario pide X.
  Si hay un compromiso activo:
    decidir si X aclara, corrige, consulta, agrega, reemplaza, pausa o cancela;
    registrar la relación y actualizar solamente lo afectado;
    si es una consulta de estado, responder y continuar el trabajo existente.
  Si no hay un compromiso activo:
    recuperar el contexto relevante del proyecto y abrir el compromiso necesario.

Luego establecer qué resultado observable satisface el pedido.
  Si falta un dato que puede obtenerse con herramientas, investigarlo.
  Si falta una decisión del usuario indispensable, preguntar y avanzar lo independiente.
  Si una elección es reversible y está autorizada, asumirla y registrar el supuesto.

Luego elegir la próxima acción por reducción de incertidumbre, progreso, riesgo y costo.
  Si el trabajo es pequeño, ejecutar y verificar directamente.
  Si es grande, conservar los entregables globales y concretar solo el próximo tramo.
  Si hay trabajo independiente que justifica coordinación, delegarlo con contrato claro.

Antes de un efecto, comprobar que sigue siendo válido y autorizado para la versión actual.
Después de un efecto, registrar qué ocurrió y qué evidencia quedó invalidada.
Cuando llega una novedad, reconciliar objetivo, trabajo y evidencia antes de continuar.

Cuando parece terminado:
  si falta un compromiso, seguir o explicar el impedimento;
  si faltan pruebas necesarias, obtenerlas o declarar la limitación;
  si cambió algo después de verificarlo, repetir solo las comprobaciones afectadas;
  si hace falta una decisión de entrega, presentar el resultado concreto;
  si todo lo necesario está satisfecho, entregar evidencia y cerrar.

Si se interrumpe:
  conservar el estado y los efectos conocidos;
  al retomar, reconciliar lo persistido con el mundo real antes de repetir acciones.
```

## Cómo coordinaría un equipo excelente

### Responsabilidades, sin obligar a multiplicar agentes

| Perspectiva | Pregunta que tiene que quedar resuelta | Resultado que se conserva |
|---|---|---|
| Usuario/producto | ¿Qué problema resolvemos y para quién? | Resultado deseado, límites, criterios explícitos y supuestos |
| Project manager | ¿Qué falta, qué bloquea y qué puede avanzar? | Compromisos pendientes, dependencias reales y decisiones necesarias |
| Ingeniería | ¿Qué desconocemos que podría cambiar el enfoque? | Hipótesis contrastadas, alternativas y decisiones justificadas |
| Developer | ¿Cuál es el próximo cambio útil y reversible? | Cambio concreto y relación con el resultado esperado |
| Tester | ¿Cómo podría estar mal aunque parezca funcionar? | Casos negativos, reproducción, regresiones y evidencia observable |
| Reviewer | ¿La evidencia respalda el cambio y el pedido? | Objeciones trazables y resolución, sin inventar alcance |
| Operaciones | ¿Se puede integrar, entregar y recuperar? | Estado de entrega, health checks y recuperación cuando corresponde |
| Mantenimiento | ¿Qué necesitará saber quien continúe? | Decisiones, limitaciones, pruebas pendientes y referencias vigentes |

Estas responsabilidades se activan según necesidad. Corregir un typo no requiere una
reunión de producto; cambiar persistencia probablemente sí exige pensar en datos,
compatibilidad, pruebas y recuperación. Una misma instancia del modelo puede cubrir
varias perspectivas. Otro agente se justifica por independencia útil, especialidad o
trabajo separable, no porque falte una casilla del organigrama.

### Cuatro recorridos completos

**Proyecto nuevo: «Quiero una aplicación para turnos».** Primero separar decisiones
de negocio indispensables de detalles que pueden elegirse provisionalmente. Explorar
el repositorio si existe. Proponer una primera experiencia completa y comprobable:
crear disponibilidad, reservar, impedir doble reserva y ver la confirmación. Mantener
visibles los compromisos globales, pero concretar solamente el próximo tramo. Entregar
una porción usable, observar sus fallos y extenderla. No confundir tener login y tres
pantallas sueltas con una aplicación que resuelve el flujo. Publicar solo si forma
parte del pedido o se obtiene la decisión necesaria sobre una entrega concreta.

**Proyecto en curso: «Agregá exportación CSV».** Recuperar decisiones de permisos y
formato existentes. Si la función ya está implementada, verificarla y entregar la
evidencia sin fabricar un diff. Si falta, reproducir el recorrido, implementarlo,
comprobar encoding, permisos, datos vacíos y tamaño relevante. Los casos necesarios
dependen del producto; no imponer todos a todos los cambios. Si el usuario dice
«solo los registros filtrados», revisar la implementación y los tests contra esa
nueva versión del pedido. El revisor recibe exactamente esa misma versión.

**Incidente: «No se puede guardar».** Primero reproducir y establecer impacto. Un
test rojo que reproduce el incidente es evidencia útil, no fracaso de la investigación.
Formular hipótesis, ejecutar la prueba más discriminante y cambiar de hipótesis si
queda refutada. Si el arreglo depende de una credencial inaccesible, preparar lo
independiente y pedir el dato concreto. Si la intervención es operativa, comprobar
el estado real antes de repetirla tras un timeout. Cerrar cuando el recorrido vuelve
a funcionar y las comprobaciones pertinentes respaldan ese resultado.

**Interrupción: «Pará, sin cambiar la base de datos»; al día siguiente, «seguí».** El
primer mensaje revoca inmediatamente la validez de las acciones preparadas que toquen
esa base. Se conserva lo ya ocurrido y cualquier efecto incierto. «Seguí» reanuda el
compromiso identificado, con la restricción nueva; no reinventa un objetivo cuyo texto
es únicamente «seguí». Antes de continuar, confrontar archivos y servicios con el
checkpoint. Si alguien modificó el mismo código, invalidar la evidencia afectada y
adaptar la solución. Cancelar definitivamente y pausar son decisiones diferentes.

## Matriz de situaciones

Cada fila define comportamiento esperado antes de evaluar si el código lo soporta.
«Parcial» significa que existe ayuda en prompts o piezas del mecanismo, pero no se
observó una garantía completa en el camino Task. No implica un fallo observado en vivo.

### Entrada y continuidad

| Caso | Si ocurre… | Entonces debería… | Contraste con Task |
|---|---|---|---|
| C01 | Pedido pequeño y claro | Resolver y comprobar con mínima coordinación | Horizonte corto favorable; costo de fases sigue siendo relevante |
| C02 | Pedido grande pero entendible | Conservar entregables globales y concretar el próximo tramo | Objetivo literal presente; falta seguimiento estructurado de entregables |
| C03 | Repositorio desconocido | Obtener solo la orientación que cambia decisiones | Herramientas y recuperación disponibles |
| C04 | Proyecto parcialmente terminado | Distinguir trabajo existente de faltante | Depende de investigación y contexto recuperado |
| C05 | Usuario dice «seguí» | Recuperar compromiso e identidad, sin reinterpretar todo | Reanudación estructural no enlaza por relación entre turnos: F03 |
| C06 | Dos compromisos podrían ser «eso» | Desambiguar solo si elegir cambia materialmente el trabajo | No hay relación explícita de mensaje a compromiso: F04 |
| C07 | Pedido ya satisfecho | Demostrarlo y terminar sin cambios artificiales | Existe `no_edit`; conservar esta capacidad |
| C08 | Pedido mezcla consulta y acción | Responder lo útil y ejecutar lo autorizado | Clasificación existe; no hay obligaciones separadas del plan: F14 |

### Mensajes durante el trabajo

| Caso | Si ocurre… | Entonces debería… | Contraste con Task |
|---|---|---|---|
| C09 | «¿Cómo venís?» | Responder estado y continuar el objetivo | Inyección obliga a reconocer el mensaje; relación queda al modelo |
| C10 | Corrección durante una llamada LLM | Reconciliar antes del siguiente efecto incompatible | Se drena antes de llamadas y al cierre, no hay revisión de versión: F04 |
| C11 | Corrección durante revisión | Actualizar especificación y revisión actuales | Web difiere; TUI inyecta a una cola que el reviewer no consume: F04 |
| C12 | Nuevo alcance compatible | Incorporarlo con procedencia e invalidación selectiva | Texto llega al loop; contrato de Task no se versiona |
| C13 | «No toques X» | Impedir nuevos efectos incompatibles | Cancelación explícita tiene evento; una restricción textual no: F04 |
| C14 | Usuario reemplaza el objetivo | Retirar compromisos reemplazados y preservar trabajo útil | No hay transición explícita de reemplazo |
| C15 | Usuario adjunta captura al finalizar | Conservar imagen y texto juntos | Dos rutas consumen solo texto: F04 |
| C16 | Dos mensajes rápidos y diferentes | Conservar orden, identidad y relación de cada uno | Cola FIFO interna; Web une los diferidos en un nuevo texto |

### Producto, alcance y decisiones

| Caso | Si ocurre… | Entonces debería… | Contraste con Task |
|---|---|---|---|
| C17 | Falta un dato obtenible del repo | Investigar antes de preguntar | Soportado por herramientas; eficacia depende del modelo |
| C18 | Falta una decisión indispensable | Preguntar y avanzar lo independiente | Hooks de preguntas disponibles; Steps no expresan dependencias |
| C19 | Detalle reversible no especificado | Elegir razonablemente y registrar supuesto | Posible por prompt, sin registro tipado de supuestos |
| C20 | Reviewer propone una mejora ajena | Mantenerla opcional, sin imponerla | Separación de autoridad en prompts es una fortaleza |
| C21 | Se contradicen dos requisitos | Identificar conflicto y decisión que lo resuelve | No existe reconciliación estructurada de requisitos |
| C22 | Se descubre un requisito implícito necesario | Explicar relación causal con el pedido | Criterios derivados están separados, pero poco trazables: F14 |
| C23 | Se piden varias prestaciones pequeñas | Conservar cada compromiso aunque solo planifique tres Steps | Horizonte no equivale a inventario de compromisos: F14 |
| C24 | Usuario reduce alcance | Retirar obligaciones y pruebas dependientes de esa versión | Plan mutable; contrato y verificación no tienen revisiones |

### Desarrollo e investigación

| Caso | Si ocurre… | Entonces debería… | Contraste con Task |
|---|---|---|---|
| C25 | Hipótesis inicial falsa | Registrar descarte y cambiar enfoque | Notas, archivo y resúmenes ayudan |
| C26 | Investigación responde el pedido | Entregar hallazgos, sin exigir implementación | Bootstrap de discovery pide un cambio incluso para investigación: F10 |
| C27 | Hace falta insertar un prerrequisito | Reordenar sin borrar la obligación dependiente | Inserción protege pasos activos; no hay dependencias semánticas |
| C28 | Un paso ya quedó cubierto por otro | Retirarlo por evidencia de resultado equivalente | Deduplicación y retiro usan títulos/rutas como aproximación: F10 |
| C29 | Cambia el enfoque arquitectónico | Modificar tácticas y conservar criterios de éxito | Horizonte mutable útil; no separa por completo criterios y tácticas |
| C30 | Ciclo editar/deshacer | Detectar ausencia de progreso neto | Fingerprints y tests existentes lo cubren |
| C31 | Error de nombre de herramienta | Corregir y reintentar con la herramienta disponible | Gate específico de error recuperable lo cubre |
| C32 | Trabajo independiente justifica ayuda | Delegar con entradas, resultado y criterio de integración | Extensión de diseño; no requiere reemplazar Task por un equipo permanente |

### Pruebas y evidencia

| Caso | Si ocurre… | Entonces debería… | Contraste con Task |
|---|---|---|---|
| C33 | Test rojo reproduce el bug | Considerarlo éxito diagnóstico | StepVerification admite exit esperado; Task no puede autorarlo: F05/F11 |
| C34 | Test rojo preexistente no relacionado | Registrar baseline, no ampliar alcance automáticamente | Gate del último test no distingue causalidad: F11 |
| C35 | Test específico verde, suite relevante roja | Conservar ambas observaciones y explicar cobertura | Último test/preferido puede reducir el control: F11 |
| C36 | Cambio después de tests verdes | Invalidar verificaciones afectadas | Hay fingerprints locales, pero cierre de review tiene hueco: F07 |
| C37 | No hay runner conocido | Informar qué se verificó y qué sigue desconocido | Cero comandos devuelve `passed=True`: F06 |
| C38 | Verificación falla repetidamente | Detener ese intento sin convertir FAIL en éxito | Gate por Step abre al superar el cap: F06 |
| C39 | Reviewer indisponible | Entrega con revisión pendiente o pausa según criticidad | Error interno puede convertirse en SKIPPED: F06 |
| C40 | Reporte incluye afirmación sin fuente | Corregirla o entregarla como incertidumbre explícita | Revisión informativa tiene F01/F02 |

### Interrupciones y recuperación

| Caso | Si ocurre… | Entonces debería… | Contraste con Task |
|---|---|---|---|
| C41 | Se corta durante una lectura | Recuperar observación sin repetir todo | Checkpoint guarda intercambios como texto seguro |
| C42 | Se corta después de un efecto externo | Consultar resultado antes de reintentar | No se observa journal de intención/efecto en Task: F12 |
| C43 | Usuario pausa | Guardar estado reanudable y motivo | `cancelled` no permite restauración automática del checkpoint: F03 |
| C44 | Usuario cancela definitivamente | Retirar compromiso sin perder auditoría | Cancelación existe; semántica no separada de pausa |
| C45 | Se alcanza presupuesto | Pausar con avance, costo y siguiente acción | `exhausted` se normaliza a `blocked`: F15 |
| C46 | Credencial externa falta | Aislar dependencia y pedir la acción concreta | Posible textual; Step bloqueado es terminal sin dependencia formal |
| C47 | Archivos cambian durante la pausa | Reconciliar estado y descartar evidencia obsoleta | Identidad del checkpoint no incluye revisión del workspace: F12 |
| C48 | Falla guardar checkpoint | Avisar continuidad degradada sin fingir durabilidad | `_checkpoint` solo escribe log debug: F12 |

### Revisión, integración y entrega

| Caso | Si ocurre… | Entonces debería… | Contraste con Task |
|---|---|---|---|
| C49 | Paso obligatorio quedó bloqueado | Mantener obligación pendiente de resolución | `undischarged` ignora blocked: F08 |
| C50 | Reviewer rechaza y rework corrige | Verificar el artefacto final, no una versión anterior | Rework conserva tracker/estado; orden parcial de comprobaciones: F07 |
| C51 | Se completó código, falta documentación pedida | Seguir hasta satisfacer ambos compromisos | Depende de cobertura del plan y juicio textual: F14 |
| C52 | Usuario solo pidió revisar | Entregar observaciones, no implementar sugerencias | Autoridad literal y criterios derivados ayudan; bootstrap conflictivo |
| C53 | Entrega requiere permiso final | Preparar artefacto concreto antes de pedir decisión | Permisos existen, falta hito de entrega separado |
| C54 | Deploy termina pero health check falla | Reportar entrega fallida y recuperar según autorización | Diseño pendiente para ciclo operativo completo |
| C55 | Tests verdes pero recorrido de usuario roto | Ejecutar aceptación sobre la experiencia real | La abstracción actual no expresa cobertura por recorrido |
| C56 | Estado final falló, texto decía «listo» | Generar respuesta consistente con el desenlace | Normalización corrige solo texto vacío/genérico: F15 |

### Memoria, costo y evolución

| Caso | Si ocurre… | Entonces debería… | Contraste con Task |
|---|---|---|---|
| C57 | Cierra un paso con resumen suficiente | Registrar sin otra llamada si no aporta valor | Summarizer adicional habilitado por defecto: F13 |
| C58 | Se necesita un detalle antiguo | Recuperarlo por fuente y vigencia | Working memory y recall son fortalezas |
| C59 | Descarta un Step posterior y cierra el actual | Atribuir resumen y evidencia al Step realmente ejecutado | Se elige el último cerrado por índice: F09 |
| C60 | Proyecto dura muchos turnos | Conservar compromisos y decisiones sin reenviar todo | Hay memoria de sesión; no contrato durable de proyecto |
| C61 | Se cambia a modelo de contexto menor | Reducir material conservando invariantes y referencias | Probes de capacidades y perfiles compactos existen |
| C62 | Modo autónomo consume tokens | Contar todas las fases contra el presupuesto | `record_outcome` recibe cero tokens en el pipeline: F13 |
| C63 | No hay trabajo autorizado restante | Cerrar o esperar; no inventar una mejora obligatoria | Continuación reflexiva sugiere «next improvement»: F16 |
| C64 | Cambia el harness | Comparar resultados completos y fallos de transición | Suite extensa; faltan varias composiciones reproducidas aquí |

## Hallazgos detallados

Prioridades propuestas: **P1** afecta continuidad, cierre veraz o coordinación central;
**P2** afecta robustez, costo o una ruta condicionada. No se asigna P0: no se demostró
un incidente generalizado en producción. Las propuestas arquitectónicas se distinguen
de defectos deterministas y de riesgos cuyo impacto requiere evaluación con modelos.

### F01 — El presupuesto ilimitado bloquea el rework informativo [P1, reproducido]

Task pasa `TASK_MAX_TOOL_CALLS=0` como ilimitado. El guard de
`analysis/evidence_review.py:292` compara `used_calls >= max_total_tool_calls` sin
excluir valores no positivos. Con cinco llamadas utilizadas, cero dispara agotamiento.

Probe `evidence_budget`: `0` produce cero reworks y conserva REJECTED; `None` y `100`
permiten un rework y llegan a APPROVED con exactamente el mismo rechazo inicial.
No depende de cómo razone el modelo: es una contradicción de contratos de presupuesto.

**Mejora:** normalizar límites una sola vez (`None` ilimitado internamente), usar un
presupuesto compartido por todas las fases y probar cero/negativo/ausente/positivo.

### F02 — Rechazar una respuesta informativa no invalida su finalización [P1, reproducido]

`orchestration/pipeline.py:628` recibe `result, _` del rework informativo. Si no hay
cambios de archivos, retorna el texto sin consultar el último rechazo. TaskAdapter
obtiene nuevamente `_last_status`, que puede seguir siendo `done`.

Probe `rejected_evidence_status`: un REJECTED explícito deja `engine_status=done` y
la respuesta rechazada intacta. F01 hace especialmente visible esta composición con
la configuración ilimitada, pero F02 también existe tras agotar reworks positivos.

**Mejora:** devolver una evaluación estructurada de cierre; un rechazo no resuelto
debe impedir afirmar aceptación. Definir si corresponde corregir, entregar parcial
o esperar una decisión. No basta con notificar un rechazo en un mensaje de progreso.

### F03 — La tarea durable no tiene identidad estable entre pedidos [P1, reproducido]

`loop/resume_checkpoint.py:46` identifica por título, descripción y tipo, y
`resume_state_for_task:215` exige igualdad de esa clave. TaskAdapter reconstruye esos
campos desde el nuevo pedido literal. `pipeline.py:892` además crea un TaskRuntime
nuevo por entrada; sus eventos son persistidos, pero esta ruta no reconstruye desde
ellos el compromiso activo. `resume_token=session_id` no resuelve esa asociación.

Probe `checkpoint_identity`: restaura la misma Task, no el follow-up; una Task marcada
`cancelled` tampoco se restaura aunque coincida literalmente. Esto último es una
política explícita del checkpoint, en tensión con comentarios que llaman reanudable
a la cancelación. El chat puede recuperar contexto y reconstruir trabajo: no se afirma
que desaparezca todo el historial ni que «seguí» siempre fracase funcionalmente.

**Mejora:** identidad estable del compromiso, revisiones separadas de su redacción y
relaciones `continues/amends/replaces`. Distinguir pausar de cancelar definitivamente.
Un solo checkpoint de sesión tampoco debe ser la identidad de todos sus compromisos.

### F04 — Los mensajes nuevos no actualizan un contrato común a todas las fases [P1]

`loop/user_message_injector.py:63` agrega texto al contexto de ejecución y lo reconoce;
no versiona Task ni propaga cambios al contrato del reviewer. El contexto siguiente se
reconstruye en `loop/context_builder.py:419`. La continuidad de esa corrección queda
dependiendo de notas, resúmenes y atención del modelo a información de distinta edad.

En Web, `server/runtime.py:256` solo inyecta con fase `execute`; durante review acumula
en `deferred` y ejecuta un turno posterior. En TUI, `ui/app.py:1047` inyecta mientras
el engine esté corriendo, incluso si está revisando. Si no hay otro execute que drene
esa cola, el mensaje puede quedar pendiente hasta una tarea posterior. La persistencia
del transcript visible no equivale a consumo por la fase que debe actuar.

Además, `drain()` descarta adjuntos al devolver únicamente strings; el rechazo de cierre
tardío también usa solo `item.text`. `attachment_delivery` reproduce ambas pérdidas de
payload. La ruta `inject_mid_step` sí conserva imágenes cuando soporta visión.

**Mejora:** inbox durable con IDs y revisión aplicada, compartido por fases y clientes.
Una consulta de estado no cambia el objetivo; una corrección sí. Antes de un efecto
preparado o de publicar el resultado, verificar si sigue vigente su revisión. Acusar
recibo no prueba que la corrección haya sido incorporada. Pruebas de carreras reales
de UI quedan pendientes; la pérdida de adjuntos sí está reproducida localmente.

### F05 — Task no conecta su plan móvil con las verificaciones ejecutables [P1]

El bootstrap de `engines/task.py:63` crea `PlanStepSpec` sin `verify`.
`tools/meta/plan_tools.py:31`, `:61` y `loop/step_operation.py:11` no permiten declararlo;
`loop/loop_plan.py:439` agrega pasos sin verificación. Por ese camino una Task fresca
no puede poblar los checks que consumirían el gate y la reverificación de objetivos.

Probe `task_check_authoring` confirma el bootstrap y los campos ausentes. No significa
que Task no ejecute tests: el developer puede llamarlos y Review ejecuta verificaciones
generales. Significa que **el control por objetivo existe, pero esta ruta no lo arma**.

**Mejora:** permitir proponer comprobaciones tipadas asociadas a resultados, incluyendo
diagnósticos con exit esperado no cero. Proteger los criterios del usuario sin congelar
una comprobación defectuosa propuesta por el modelo: corregirla requiere motivo y
trazabilidad. Mejor aún, asociar checks al compromiso y no a la táctica que puede borrarse.

### F06 — La incertidumbre de verificación puede parecer éxito [P1/P2, reproducido]

Hay tres contratos diferentes:

- `loop/step_complete_gate.py:500` deja cerrar tras superar tres fallos; anota una
  advertencia. `verification_fail_open` devuelve `[True, True, True, False]`.
- `analysis/verification_engine.py:99` devuelve `passed=True` con cero comandos.
  `empty_verification` lo reproduce. No tener un test aplicable puede ser razonable,
  pero no es lo mismo que haber verificado una condición necesaria.
- `analysis/review_engine.py:681` captura fallos del proveedor como `SKIPPED`;
  `run_review_rework_loop` retorna ese resultado sin cambiar el estado. El wrapper
  captura excepciones propagadas como failed, pero este fallo ya fue convertido a dato.
  `skipped_review_on_error` reproduce SKIPPED ante indisponibilidad del reviewer.

La primera tiene menor exposición inmediata en Task por F05; importa al conectar checks.
La revisión posterior puede rescatar algunos fallos, pero no unifica sus significados.

**Mejora:** separar `pass/fail/unknown/not_applicable` y política de obligatoriedad.
Agotar reintentos deja FAIL o UNKNOWN; no abre una puerta a éxito. Una revisión opcional
puede quedar pendiente sin bloquear una tarea pequeña, siempre de manera explícita.

### F07 — Reparar objetivos puede invalidar tests sin volver a ejecutarlos [P2, reproducido]

`analysis/review_engine.py:1434` corre tests, después repara objetivos y luego revisa
texto. Si la reparación de objetivos modifica el programa y el reviewer aprueba, no se
repite la verificación general para ese nuevo estado. Las rondas por rechazo textual
repiten el mismo orden, así que también pueden terminar con una reparación posterior.

Probe `objectives_can_stale_tests` controla los colaboradores y obtiene la secuencia
`tests_pass → objective_rework_mutates → review_approves`, con una sola ejecución de
tests. Demuestra el hueco de secuenciación; no ejecuta una regresión real contra un LLM.
En Task normal depende de que haya checks poblados, actualmente limitado por F05.

**Mejora:** invalidar checks por revisión del artefacto y alcanzar un estado estable:
todo lo obligatorio está verificado contra la versión final. Cualquier rework vuelve
a la reconciliación de evidencia. No repetir pruebas no afectadas por reflejo.

### F08 — Un Step bloqueado deja de contar como trabajo sin resolver [P1, reproducido]

`loop/loop_plan.py:170` considera pendientes solo `pending/active`.
`StepManager.reconcile_task_completion:234` consulta esa lista; `engine.py:1440`
permite avanzar desde un Step bloqueado si hay otro pendiente. Si luego el segundo
se declara done, el primero no impide la finalización. La rama implícita de cierre
también puede terminar con una lista sin pendientes que contiene blocked.

Probe `implicit_done_with_blocked_step` caracteriza ambos caminos. Un reviewer podría
detectar el faltante, pero no existe una obligación estructurada que exija resolverlo.
Tampoco todo Step bloqueado debería impedir terminar: una táctica alternativa puede
satisfacer el mismo resultado. Falta justamente esa relación de sustitución por evidencia.

**Mejora:** separar estado de una tentativa y cumplimiento del compromiso. Registrar
«esta alternativa resolvió aquello» o «el usuario retiró ese alcance» antes de darlo
por satisfecho. No convertir todos los bloqueos en un gate eterno.

### F09 — Resúmenes y evidencia pueden atribuirse al Step incorrecto [P1, reproducido]

`loop/step_manager.py:386` infiere el paso cerrado con `closed[-1].index` después de
avanzar el plan. La lista está ordenada por índice, no por momento de cierre. Si se
descartó un Step posterior, ese será el último aunque acabe de completarse uno anterior.

Probe `summary_step_identity`: Step 1 completado, 2 activo, 3 descartado; el resumen
de la ejecución de 1 queda guardado con `step_index=3`. Ese mismo índice se usa al
archivar y registrar el resultado, afectando recuperación y trazabilidad posterior.

**Mejora:** transportar explícitamente el ID del Step ejecutado en StepResult/evento de
transición. Los IDs no deberían ser posiciones reordenables. El archivo debe vincular
su evidencia por identidad, sin inferirla desde una vista del plan ya mutada.

### F10 — El lenguaje del título condiciona semántica y puede orientar mal [P2]

`loop/loop_plan.py:74` infiere discover/change/verify por regex; esa fase participa en
activación, deduplicación y gates de efecto. La cobertura idiomática es desigual. El
nombre legible del trabajo funciona también como señal de control, aunque dos títulos
equivalentes puedan activar reglas diferentes.

Más directo: `_bootstrap_step` pide «add or modify one concrete change Step» para todo
discovery, incluida una Task de investigación. El probe lo reproduce. La protección de
investigación frente a exigir edits existe en otros lugares; aquí hay una instrucción
contradictoria que el modelo debe resolver, no un bypass probado de permisos.

**Mejora:** tipo de acción y resultado esperado explícitos; título libre para la persona.
Un discovery puede desembocar en cambio, reporte, decisión o cierre. Las heurísticas
pueden completar metadatos faltantes, sin gobernar obligaciones por sí solas.

### F11 — «El último test» es una aproximación insuficiente a aceptación [P2]

`loop/step_complete_gate.py:224` bloquea done ante el último test rojo sin distinguir
reproducción intencional, baseline o fallo causado por el cambio. Un test verde posterior
quita ese veto aunque sea distinto. `analysis/review_engine.py:1104` elige un comando
preferido; `VerificationEngine._detect_test_command` lo reutiliza para respetar el
entorno y no introducir suites irrelevantes, lo cual tiene sentido pero no prueba cobertura.

**Caso:** investigar una regresión y demostrarla no debería obligar a arreglarla cuando
se pidió solamente diagnóstico. Tampoco ejecutar un test trivial verde debe borrar la
relevancia de una suite necesaria roja. Las estructuras por comando y fingerprints ya
guardan más información de la que usa este gate.

**Mejora:** evidencia por propósito, objetivo, revisión y baseline. Diferenciar resultado
del proceso de resultado esperado de la comprobación. Seleccionar pruebas por impacto
y compromisos, no por orden temporal exclusivamente. Falta medir incidencia con tareas reales.

### F12 — Checkpoint de contexto no equivale a recuperación de efectos [P2, brecha de diseño]

El checkpoint conserva plan y resultados observados, y `tool_runner.py:1231` lo persiste
tras registrar herramientas. Es valioso. Pero un proceso puede terminar después del
efecto y antes de ese registro. El snapshot de Task no contiene un journal previo con
intención, clave de idempotencia y estado incierto de cada operación externa.

También puede restaurarse contexto de archivos ya modificados por otra persona.
Hay fingerprints locales; no se observa en `resume_state_for_task` una reconciliación
del estado externo ni una validación completa del workspace al aceptar el snapshot.
`engine.py:2010` convierte fallos al persistir en logs debug.

**Mejora:** registrar intención antes de efectos relevantes, resultado después y usar
consultas de estado o idempotencia cuando la herramienta lo permita. Reanudar desde
observación del mundo actual, con continuidad degradada visible si no se pudo guardar.
No es necesario envolver cada lectura de archivos en una transacción distribuida.

### F13 — El costo de coordinación no está representado como presupuesto completo [P2]

La compresión de Steps archiva evidencia y evita reenviar todo: conservarla. Sin embargo,
`step_manager.py:406` llama al resumidor en cada Step completo si está habilitado;
`step_summarizer.py:183` hace otra completion y no incorpora su usage en LoopState.
No se demostró que esta llamada sea inútil: sí que debe medirse su ganancia neta.

En la cadena autónoma, `pipeline.py:1311` llama `record_outcome` sin `tokens_used`;
`orchestration/autonomous.py:165` solo incrementa consumo si recibe ese dato. El fuse
de tokens de esa cadena no refleja el gasto por esta ruta. Tampoco esos topes de cadena
interrumpen una Task ilimitada en curso: se evalúan tras volver del engine.

**Mejora:** contabilidad por llamada y por fase, agregada al compromiso. Medir entrada,
salida, caché, costo y latencia sin confundir tokens lógicos con dinero facturado.
Resumen estructurado barato desde eventos; llamada adicional cuando aporta continuidad
que los eventos no capturan. Mantener presupuesto independiente de longitud del prompt.

### F14 — El horizonte corto no mantiene por sí solo todos los compromisos [P2, diseño]

Task conserva la descripción completa; no pierde automáticamente todo lo que queda
fuera de sus tres Steps. Pero esa descripción y el juicio del modelo siguen siendo el
respaldo de prestaciones no representadas aún. `_goal_from_escalation` no rellena
`acceptance_criteria`; devuelve checks derivados y el esquema usa un criterio placeholder
cuando faltan condiciones explícitas. Los criterios no tienen IDs, evidencia ni revisión.

No conviene resolverlo obligando a un plan exhaustivo inicial. Eso agregaría rigidez y
tokens antes de conocer el proyecto. Hace falta un inventario compacto de resultados
comprometidos, separado del horizonte táctico y actualizable con procedencia.

**Mejora:** registrar los compromisos importantes al descubrirlos; conservar requisitos
literales y supuestos por separado. Si una Task es pequeña, un solo compromiso alcanza.
Para un proyecto grande, vistas de hitos y dependencias, sin expandir todos sus Steps.

### F15 — Estados y respuesta final no expresan toda la situación [P2]

`engines/base.py:27` normaliza `exhausted` a `blocked`; el runtime dispone de cuatro
estados terminales. Se pierde la distinción entre necesitar al usuario, esperar una
operación, agotar recursos o sufrir un error recuperable. El texto y logs conservan
algunos motivos, pero las siguientes decisiones no disponen de una semántica común.

Un rechazo de revisión puede cambiar el estado a blocked y conservar el texto de éxito
del developer. `normalize_terminal_message` corrige vacío y dos frases genéricas, no
una respuesta detallada que quedó obsoleta. El estado estructurado puede ser correcto
mientras el texto final sostiene algo incompatible.

**Mejora:** estado de ejecución separado de estado de aceptación y de entrega. Producir
el mensaje final desde ese resultado compuesto, incluyendo lo conseguido, limitación
actual y siguiente acción cuando haga falta. No agregar estados solo para multiplicar enums.

### F16 — Continuar autónomamente puede perder el ancla precisa del proyecto [P2]

`pipeline.py:1347` crea un nuevo pedido interno que pide identificar y ejecutar la
próxima mejora. Se reingresa por ChatAgent y reconstrucción del Task, dependiendo del
contexto de sesión para preservar alcance. El resumen de trabajo habitual se guarda
después de la rama que retorna por continuación; ese camino no ejecuta ese guardado
en la pasada anterior, aunque sí hay otros archivos/eventos de memoria.

No se afirma que toda continuación amplíe alcance: la política de autorización y la
voluntad del usuario importan. El problema es no distinguir formalmente «terminar este
compromiso», «trabajar sobre este backlog» y «proponer/mejorar indefinidamente».

**Mejora:** continuar por identidad y pendientes autorizados. Si no quedan, cerrar o
esperar según el modo acordado. Una mejora nueva es candidata de backlog, no obligación
nacida del prompt de continuación. Persistir la transición antes de reingresar.

## Qué preservaría

- El horizonte corto que evita pagar un plan largo y frágil antes de observar el repo.
- La descripción literal y la separación explícita de criterios derivados y autoridad.
- Working memory con evidencia recuperable; comprimir no debería destruir las fuentes.
- Cancelación como estado real, comprobada antes y después de llamadas al modelo.
- Rework que conserva el tracker y estado de Task, para no reiniciar artificialmente costos.
- Fingerprints de cambios netos y resultados de tests, en vez de equiparar actividad a progreso.
- Rechazo de cierre ante mensajes tardíos; extenderlo a una reconciliación completa.
- Permisos compartidos para herramientas y verificaciones, y capacidades compatibles con modelos locales.

## Una alternativa: ejecución guiada por compromisos y evidencia

No propongo una reescritura inmediata ni un grafo obligatorio para todo. Propongo cambiar
qué entidad decide el rumbo: los resultados comprometidos, no la secuencia de Steps.
La ejecución puede seguir siendo el LoopEngine actual al principio.

### Estado mínimo

| Registro | Contenido mínimo | Para qué sirve |
|---|---|---|
| Compromiso | ID, revisión, pedido fuente, resultado, restricciones, estado | Saber qué se debe al usuario |
| Decisión | Pregunta/supuesto, autor, elección, razón, elementos afectados | Resolver ambigüedad sin reabrir todo |
| Acción | ID estable, compromiso, precondiciones, efecto, estado y resultado | Ejecutar y recuperar intentos |
| Evidencia | Afirmación/check, fuente, resultado esperado/observado, alcance, revisión | Saber qué se ha demostrado y cuándo deja de valer |
| Evento | Mensaje, cambio, resultado o interrupción con identidad y secuencia | Reconstruir transiciones y proyectar UI/contexto |

Estos registros pueden vivir inicialmente en las tablas y archivos existentes. Un motor
de eventos nuevo no aporta nada si duplica autoridades o crea dos estados divergentes.
Primero definir quién escribe y quién proyecta cada hecho; migrar almacenamiento solo
cuando el contrato esté probado.

Separaría tres dimensiones: **ejecución** (activa, esperando, pausada, cancelada),
**aceptación** (pendiente, satisfecha, rechazada, evidencia insuficiente) y **entrega**
(no requerida, preparada, pendiente de decisión, realizada, verificada). Así un deploy
en curso no se confunde con falta de permiso, ni falta de un check opcional con código roto.

### Control del próximo movimiento

1. Consumir novedades y aplicar cambios al compromiso antes de decidir acciones.
2. Reconciliar lo observado con el estado del workspace y servicios relevantes.
3. Detectar obligaciones insatisfechas, evidencia inválida y decisiones pendientes.
4. Elegir la acción con mejor balance entre avance, reducción de incertidumbre, riesgo
   y costo. No hace falta calcular una puntuación matemática ficticia.
5. Ejecutar bajo autoridad vigente; registrar efectos y actualizar evidencia.
6. Mantener una pequeña cola de próximos movimientos, extensible según observaciones.
7. Intentar el cierre solo si no quedan compromisos necesarios sin resolver.

Un grafo de dependencias puede emerger en proyectos grandes. El caso trivial sigue
siendo un compromiso, una acción y una comprobación. Los roles especializados son
estrategias de trabajo sobre ese mismo estado, no copias del objetivo en prompts aislados.

### Memoria y economía de tokens

Mantendría tres capas con contratos distintos:

- **Núcleo estable:** objetivo vigente, restricciones, decisiones importantes y mapa
  compacto de compromisos. Cambia por hechos de producto, no con cada tool call.
- **Contexto de la acción:** archivos, errores, hipótesis y pruebas necesarios ahora;
  material reciente relevante permanece literal aunque cierre un Step.
- **Archivo recuperable:** intercambios completos, resultados grandes, versiones y
  decisiones antiguas con IDs y fuentes. Se carga por necesidad y vigencia.

El fin del Step sería una buena oportunidad de ordenar, no una obligación de olvidar
todo y hacer una llamada de resumen. Comprimir ante presión, cambio de foco o redundancia;
extraer datos de herramientas de forma determinista cuando sea posible. Pedir síntesis
al modelo cuando haya conclusiones semánticas que no se pueden obtener de los eventos.
Mantener los hechos críticos del usuario fuera de una cadena de resúmenes que puede degradarlos.

La selección también debe considerar incertidumbre: evidencia contradictoria y supuestos
sin resolver pueden ser más importantes que el último archivo leído. El contexto largo
de un modelo SOTA permite retenerlos, pero no elimina el problema de relevancia. El modo
compacto sigue siendo necesario para compatibilidad; no determina toda la arquitectura.

**Hipótesis a medir:** retener contexto útil entre Steps y reducir llamadas de resumen
podría bajar latencia y relecturas. También podría subir tokens o distraer al modelo.
No se afirma ahorro antes de medir. El objetivo sería menor costo por resultado correcto,
no menor prompt a cualquier precio.

### Comparación de opciones

| Opción | Ventaja | Riesgo | Decisión sugerida |
|---|---|---|---|
| Reparar Task actual | Bajo cambio, arregla defectos concretos | Conserva varias ambigüedades de estado | Hacer primero |
| Task + compromisos/evidencia | Conserva el loop y mejora continuidad/cierre | Duplicar contratos si no se define un dueño | Prototipo preferido |
| Sustituir por un scheduler de compromisos | Mayor libertad de planificación y espera | Migración amplia y mayor superficie de bugs | Evaluar después del prototipo |
| Equipo multiagente permanente | Varias perspectivas disponibles | Costo, coordinación y errores correlacionados | No usar como solución universal |
| Contexto largo sin Steps | Menor fricción de planificación | Continuidad, medición y recuperación quedan débiles | Baseline experimental, no reemplazo asumido |

## Plan de mejora y pruebas de aceptación

### Tramo 1: cerrar contradicciones verificables

Corregir F01/F02 y F09; unificar el significado de review rechazada/desconocida; asegurar
que el mensaje final concuerde con el estado. Conectar verificaciones de Task (F05)
junto con la política FAIL/UNKNOWN y el orden de reverificación (F06/F07). No activar
más checks sin resolver antes las rutas que abren el gate o reutilizan resultados viejos.

**Aceptación:** cero ilimitado permite rework; REJECTED no entrega completed; los
resúmenes conservan su Step real; un artefacto mutado no usa un PASS anterior como
evidencia final. Tests focalizados con esas invariantes, no con textos exactos del prompt.

### Tramo 2: una sola relación entre mensajes y trabajo

Introducir ID/revisión del compromiso e inbox compartido entre UI, ejecución, review y
cierre. Resolver explícitamente continuar/corregir/reemplazar/consultar/pausar. Conservar
adjuntos. Añadir un punto de reconciliación antes de efectos y antes de entregar.

**Aceptación:** mensajes idénticos en CLI/TUI/Web producen transiciones equivalentes;
«seguí» restaura identidad y restricciones; una corrección en review no espera a que se
publique una respuesta inválida; cada mensaje aceptado queda pendiente o aplicado, nunca
consumido sin trazabilidad. Inyectar eventos en cada borde de fase para probar carreras.

### Tramo 3: compromisos independientes del plan y memoria por vigencia

Mantener obligaciones y checks aunque se modifiquen Steps; distinguir tentativa bloqueada
de compromiso insatisfecho. Reutilizar archivo y fingerprints existentes. Cambiar la
política de resumen detrás de una configuración experimental, manteniendo baseline.

**Aceptación:** un proyecto con más de tres entregables no pierde ninguno al replanificar;
un resultado puede satisfacerse por una alternativa sin reactivar una táctica abandonada;
los cambios invalidan solamente evidencia dependiente y los requisitos del usuario
sobreviven compacción, rework y reinicio.

### Tramo 4: recuperación y colaboración cuando el producto lo necesite

Agregar journal de operaciones relevantes, reconciliación tras reinicio e integración
de colaboradores sobre un único contrato. Delimitar permisos y efectos por acción;
esperar libera recursos y se reanuda por un evento, no por polling continuo del modelo.

**Aceptación:** una interrupción entre efecto y persistencia no duplica una publicación;
un resultado de un colaborador no cierra automáticamente un compromiso sin validación
de integración; falta de un secreto no detiene trabajo independiente autorizado.

### Cómo comparar sin confundir impresión con mejora

Crear una campaña con repositorios pequeños verificables y conversaciones multivuelta.
Reusar los mismos pedidos, estados iniciales y puntos de interrupción para baseline y
candidatos. Separar casos usados para diseñar del conjunto reservado para evaluar.

Medir como resultados principales: compromisos realmente satisfechos, entregas falsas,
regresiones, efectos fuera de alcance, mensajes perdidos y recuperación correcta.
Como costos: tokens de todas las fases, costo efectivo, latencia, tool calls, relecturas,
cantidad de preguntas evitables y trabajo rehecho. Reportar por tipo de tarea y tamaño;
un promedio puede esconder que un candidato mejora cambios simples y rompe proyectos largos.

Incluir fallos deliberados: reviewer no disponible, test flaky, baseline rojo, cancelación
durante tool batch, mensaje durante review, workspace cambiado durante pausa, evidencia
antigua, tool timeout después de un efecto y cambio de alcance tras una primera entrega.
Los probes deterministas prueban invariantes; una evaluación con modelo prueba si la
estrategia lleva a mejores decisiones. Son evidencias distintas y ambas hacen falta.

No se ejecutaron campañas pagas ni comparaciones de modelos en esta auditoría. No hay
fundamento para prometer porcentajes de ahorro ni adjudicar la superioridad a más agentes.

## Evidencia reproducible y límites

- [Probes offline](audits/task_harness_lifecycle_probes.py): 11 escenarios de caracterización.
- [Observaciones JSON](audits/task_harness_lifecycle_observations.json): salida de la ejecución.
- Comando: `uv run --offline python docs/audits/task_harness_lifecycle_probes.py`.
- El script no llama al proveedor: bloquea `litellm.completion`; simula colaboradores y
  evita modificar la aplicación o bases de datos reales. Sus assertions confirman el
  comportamiento defectuoso observado, no el contrato deseado para una futura suite.
- 276 tests existentes de coordinator, plan inicial, extracción, review, checkpoints,
  terminación, runtime y Web pasaron. Que pasen no refuta los huecos de composición:
  los probes introducen combinaciones que esos tests no exigen actualmente.
- Suite completa: `uv run --offline pytest -q` produjo **5086 passed, 12 skipped,
  1 failed** en 148,18 segundos. El único fallo fue
  `tests/test_notifications.py::TestChannels::test_deliver_webhook_posts_json`:
  el sandbox impidió abrir un socket en `127.0.0.1` (`PermissionError` en `socket.bind`).
  Se repitió únicamente ese test con permiso ampliado: **1 passed** en 0,58 segundos.
  Esto no constituye una segunda ejecución completa; son la suite y su reintento focalizado.
- Se usó `UV_CACHE_DIR=/private/tmp/infinidev-audit-uv` para mantener la caché de uv en
  una ubicación escribible. `--offline` impide resolver dependencias desde red; no es
  un bloqueo general de red del código ejecutado. Los probes bloquean el proveedor aparte.

La auditoría combina lectura estática y ejecuciones deterministas de fronteras. No se
hizo un playtest interactivo de las carreras de UI ni una campaña de un proyecto completo
con LLM real. Las brechas de diseño describen capacidades deseables; no deben contarse
como incidentes ya observados. No se modificó código de producción.
