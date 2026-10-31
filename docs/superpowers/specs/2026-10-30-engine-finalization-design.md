# Cierre verificable e interrumpible del engine

## Intención

El usuario pidió otra sesión de mejoras importantes y limpieza del engine. Mantiene
la delegación de elección e implementación y la prioridad previa de evitar peores
resultados, pérdida de contexto y confusión. Las decisiones de este documento son
del implementador. No se cambia el modelo objetivo ni se justifica el diseño por
limitaciones de modelos pequeños.

## Hallazgos y alternativas

El guardrail devuelve solo texto: el llamador no distingue aceptación, rechazo,
excepción o presupuesto agotado y siempre finaliza con `done`. Las correcciones
ejecutan tools sin el ámbito de cancelación del engine. Además, el cierre por plan
vacío duplica el cierre explícito y convierte dos iteraciones sin Steps pendientes
en éxito. Un mensaje recibido durante resumen/corrección puede quedar en cola
mientras se publica una respuesta para la instrucción anterior.

| Alternativa | Ventaja | Coste |
| --- | --- | --- |
| Añadir checks en cada `return` | Cambios pequeños | Mantiene dos cierres y la ambigüedad del string |
| Reemplazar todos los loops por uno | Reducir duplicación ampliamente | Mezcla estado del Step y corrección final; migración mayor |
| Resultado tipado y una frontera de cierre | Estados honestos, cancelación y entradas nuevas explícitas | Cambia contratos privados y requiere un shim de compatibilidad |

Elijo la tercera alternativa. También se reutiliza la serialización del transcript
del ToolRunner; se conserva el micro-loop correctivo acotado en lugar de introducir
otro motor o una dependencia nueva.

## A. Resultado de validación

- Introducir `GuardrailOutcome(text, status, reason)` en `guardrail_runner.py`.
  `status` usa etiquetas del loop: `done`, `blocked`, `exhausted`, `failed`, `cancelled`.
- El camino normal usa `evaluate_guardrail(ctx, result, *, hooks,
  build_context_messages, run_tool, is_cancelled)`. La configuración ya pertenece a
  ExecutionContext: el engine no vuelve a pasar una docena de campos por separado.
- Sin guardrail, o tras una aceptación booleana/tupla, devolver `done`. Un booleano
  falso o un último resultado rechazado devuelve `blocked` con el feedback
  disponible. Un validador o ejecución que lanza devuelve `failed`, con razón.
- Un retorno del validador que no sea `bool` o una tupla `(bool, feedback)` es
  `failed`; la veracidad de un string o diccionario no prueba aceptación.
- Fuses de tokens, tools o rondas agotados sin validar el resultado devuelven
  `exhausted`; nunca equivalen a validación positiva. Un último resultado corregido
  se valida antes de decidir, aunque ya se haya usado el último intento.
- Conservar el texto más reciente como resultado parcial. La presentación de un
  outcome distinto de `done` explicita la razón y que el texto es parcial.
- Mantener `apply_guardrail` con su firma y retorno string actuales como shim
  legacy: devuelve `outcome.text`, incluso en rechazo/excepción. Construye una
  copia superficial del contexto con sus argumentos explícitos, sin mutar la
  configuración del llamador ni copiar el estado de ejecución.
- El método privado del engine pasa a `_validate_final_result(ctx, result)` y
  devuelve el outcome. Se migran todos los consumidores privados encontrados.

## B. Una sola decisión de cierre

- La Task se completa por un `step_complete(status="done")` aceptado por los gates
  y la validación final. Mantener las comprobaciones actuales de evidencia inicial.
- El plan sin Steps pendientes no completa automáticamente la Task. Tras la misma
  observación de dos iteraciones, el contador solo encola un aviso para pedir una
  decisión explícita sobre el resultado global o nuevos Steps; se entrega una vez
  por racha. Los guards de no-progreso y los límites reales siguen funcionando.
- Reunir la finalización explícita en una operación privada que comprueba parada
  e instrucciones pendientes antes y después de la validación y mapea el outcome
  a `StepManager.finish`. La cancelación gana a una respuesta positiva tardía.
- Si hay una indicación pendiente, no publicar `done`: conservarla en el estado y
  checkpoint, dejarla en cola para entregar el turno urgente y sus imágenes en la
  próxima iteración, y marcar la continuación como interrumpida. Los presupuestos
  duros mantienen autoridad para detener el run como `exhausted`.
- En workers, `TeamRuntime.pending_user_guidance(actor)` toma un snapshot sin
  consumir el cursor de eventos. La barrera conserva las indicaciones con sus IDs
  `team:{id}`; el poll normal conserva la entrega de imágenes y evidencia de peers.

## C. Cola de usuario y retención idempotente

- Usar un `deque` protegido por un único Lock para inyección, drenaje y snapshot;
  eliminar el bucle `Queue.empty()/get_nowait()/Empty` y sus locks implícitos.
- Cada `InjectedUserMessage` recibe un ID runtime estable. `UserGuidance` conserva
  `message_id` con default vacío compatible con checkpoints previos.
- `retain_pending(state)` toma un snapshot sin consumir la cola y conserva las
  indicaciones. `drain_for_state` conserva otra vez de forma idempotente por ID.
  Mensajes con texto idéntico e IDs distintos siguen siendo eventos diferentes.
- `has_pending` consulta bajo el mismo lock. Solo el worker modifica LoopState;
  el thread de UI únicamente encola. No persistir píxeles ni data URLs.
- Esto cierra ventanas largas durante resumen/corrección. No se promete una
  transacción atómica entre la UI y la publicación del último byte de respuesta.

## D. Corrección cooperativa y transcript común

- Comprobar `is_cancelled` antes/después de validación, petición al modelo y cada
  ejecución de tool. Ninguna acción posterior debe empezar después de observarla.
- El engine suministra `run_tool(name, arguments)`: abre/cierra su ámbito normal
  de foreground en `try/finally`, comprueba cancelación y despacha con el mismo
  evento cooperativo de cancelación. Así funcionan tanto la parada de Task como
  la parada de la tool activa, sin dejar el indicador de actividad pegado.
- Reutilizar `ToolRunner.append_assistant_message` para conservar reasoning y
  campos opacos según el proveedor. Normalizar contenido con el helper existente
  sin mutar el objeto de respuesta del proveedor.
- El modo manual usa el parser de tools existente y respuestas de texto; no envía
  schemas FC ni turnos `tool`. El modo FC conserva un resultado por call ID.
- Responder también las llamadas no ejecutadas cuando termina el presupuesto o
  se recibe `step_complete`; ninguna escritura posterior al terminal se ejecuta.
  Un `step_complete` vacío es una salida del intento, no ocho rondas repetidas.
- Un terminal correctivo `blocked` o `explore` devuelve `blocked`: pasar un check
  de formato no resuelve un bloqueo ni realiza una exploración pendiente.
  `continue` conserva el comportamiento de propuesta correctiva (también es el
  default del parser); su texto debe pasar el validador antes de aceptar el cierre
  `done` que ya solicitó el loop principal.
- Mantener contabilidad, permisos, seguimiento de archivos, caché actualizada y
  contexto literal ya implementados en la ronda anterior.

## Limpieza y compatibilidad

No agregar settings ni flags de política. No modificar StepCompleteGate ni los
criterios del plan del usuario. Preservar todos los cambios locales previos y el
lockfile. Las firmas públicas `LoopEngine.execute` y `apply_guardrail` permanecen;
los consumidores privados se migran junto a sus tests. El cambio deliberado de
producto es que una validación fallida y un plan vacío dejan de anunciar éxito.
Los adaptadores existentes ya normalizan blocked/exhausted/failed/cancelled.

## Validación

Regresiones RED→GREEN de rechazo/excepción anunciados como éxito, cancelación en
validador/modelo/tool y batch, instrucciones nuevas durante resumen y corrección,
retención idempotente y checkpoints, plan vacío, razonamiento requerido, modo
manual, argumentos/contenido estructurados, batch terminal y completion vacío.
Comprobar la interfaz legacy, consumidores de estados, eventos terminales, scopes
de tools siempre cerrados y conservación de contexto/caché. Ejecutar suite
relacionada, revisión independiente y `uv run --frozen --no-sync pytest`.
