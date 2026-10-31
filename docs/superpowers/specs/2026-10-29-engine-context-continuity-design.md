# Continuidad del contexto en el motor principal

## Intención y autoridad

El usuario pidió brainstorming y mejoras del main engine, delegando el número y
la elección de cambios. Su prioridad explícita es evitar comportamientos que
empeoren resultados, pierdan contexto o confundan al agente. Las decisiones de
implementación de este documento son del implementador.

El objetivo es conservar la diferencia entre instrucciones literales, planes
derivados y observaciones históricas durante Steps, rework y reanudaciones de una
misma Task. Se mantienen el ciclo plan–execute–summarize, los contratos de
verificación existentes y el soporte de modelos locales y SOTA de contexto largo.

## Hallazgos del código actual

1. Los mensajes en vivo se drenan desde `UserMessageInjector`. Se muestran en el
   Step actual, pero no quedan en `LoopState`; el siguiente prompt se reconstruye
   desde cero y puede perder una restricción que el resumen no haya conservado.
2. El drenaje de inicio de Step convierte mensajes a texto y descarta imágenes.
   El rechazo tardío de `step_complete` también conserva solo texto.
3. `guardrail_runner` recibe descripción, resultado esperado y estado, pero sus
   peticiones correctivas solo contienen el sistema, feedback y respuesta
   rechazada. El corrector pierde Task, plan, notas y evidencia contextual.
4. Un checkpoint restaura cuerpos de archivos y búsquedas. El prompt afirma que
   todo está actualizado y prohíbe releerlo, aunque el disco pudo cambiar durante
   la interrupción.

## Enfoques considerados

| Enfoque | Ventaja | Riesgo |
| --- | --- | --- |
| Repetir todo el transcript | Retención amplia | Mezcla observación y autoridad; arrastra protocolo de proveedor y payloads grandes |
| Pedir mejores resúmenes/notas | Cambio pequeño | Una paráfrasis sigue pudiendo omitir una restricción literal o una imagen |
| Separar instrucciones durables, corrección contextual y snapshots | Mantiene la información y su procedencia | Requiere conectar varios colaboradores del motor |

Se elige el tercero. Los mensajes nuevos exigen una respuesta inmediata una vez;
las instrucciones retenidas se vuelven a mostrar como contexto, sin exigir
acknowledgements repetidos en cada Step.

## A. Instrucciones del usuario conservadas en la Task

- Añadir un modelo `UserGuidance` con `text: str` y
  `attachment_paths: tuple[str, ...] = ()`, y un campo
  `LoopState.user_guidance: list[UserGuidance]` con valor vacío por defecto.
- Conservar cada mensaje en orden de recepción al drenarlo en cualquiera de las
  tres rutas: inicio de Step, mitad de Step y cierre tardío. No deduplicar por texto:
  dos mensajes iguales enviados por el usuario son dos eventos legítimos.
- Los textos se conservan literalmente. Las rutas/nombres de adjuntos son
  metadatos del runtime, no contenido que el modelo haya visto en una imagen.
  No persistir data URLs, base64 ni objetos de imagen en este estado.
- Renderizar estas indicaciones en cada prompt, independientemente del perfil
  opcional de prompting. Mantener cronología y expresar que una indicación
  posterior puede actualizar una anterior. Escapar XML en texto y metadatos.
- El contexto literal permanece separado de notas, resúmenes y requisitos
  derivados. No modificar automáticamente el plan por palabras clave.
- `drain() -> list[str]` sigue existiendo por compatibilidad. La construcción
  real del prompt usa un drenaje estructurado para conservar adjuntos.
- Los adjuntos nuevos se entregan en un turno `user` cuando hay visión; sin
  visión se indican sus rutas. Las imágenes no se reenvían automáticamente en
  todos los Steps. Tras reiniciar solo se recuperan sus referencias, con una
  indicación explícita de que los píxeles no fueron persistidos.
- Una llamada tardía reescribe el único resultado de `step_complete` y añade
  las imágenes como turno `user` separado, manteniendo válido el protocolo FC y
  su fallback de texto.
- Las indicaciones literales reenviadas a un worker de equipo se conservan del
  mismo modo cuando llegan a su ExecutionContext; la evidencia de colaboradores
  sigue siendo una observación separada, sin autoridad de usuario.
- El aviso urgente deja de afirmar que todos los Steps son aprobados e
  inmutables: guía al agente a adaptar tácticas derivadas y preservar únicamente
  requisitos del usuario que el mensaje nuevo no haya cambiado. Una consulta de
  estado no autoriza por sí sola terminar la tarea.
- La retención pertenece a `LoopState`: sobrevive a rework que conserva estado y
  checkpoints de la misma Task, sin filtrar instrucciones a otra Task nueva.
  Propagar indicaciones a todos los Tasks de un Goal es otro contrato, fuera de
  este cambio.
- Guardar checkpoint después de entregar instrucciones nuevas, antes de la
  siguiente llamada al modelo, tanto al construir el prompt como a mitad de Step
  o al rechazar el cierre tardío.

## B. Checkpoints y procedencia de archivos

- Los checkpoints antiguos siguen siendo válidos: los campos nuevos son
  opcionales y no requieren una migración ni bump del formato.
- Incluir `user_guidance` entre los campos esenciales de un checkpoint. No
  resumir ni recortar instrucciones para hacerlo caber. Si el alcance literal
  no entra en el límite existente, se rechaza el checkpoint como ya ocurre con
  un plan irreducible, en vez de fingir que se guardó completo.
- Añadir `OpenedFile.requires_revalidation: bool = False`. Al restaurar estado,
  marcar todos los cuerpos retenidos para revalidación sin borrar su contenido.
  Esto aplica también a las observaciones de símbolos y búsquedas.
- El prompt identifica esos cuerpos como snapshots anteriores a la reanudación;
  permite releer antes de usarlos como evidencia del estado actual. El encabezado
  general deja de garantizar actualidad absoluta de cualquier caché.
- Una lectura o refresh real elimina la marca solo de la entrada actualizada.
  Preservar TTL, pinned y datos históricos; no hacer lecturas de disco nuevas
  durante la construcción del prompt.

## C. Correcciones del guardrail con contexto

- Mantener el retorno string y la política pública actual de fallback del
  guardrail; este cambio mejora la información del corrector.
- Añadir a `apply_guardrail` un callback opcional `build_context_messages` que
  construye mensajes nuevos antes de cada intento correctivo, únicamente si
  la validación falló y queda presupuesto para corregir.
- `LoopEngine._apply_guardrail` proporciona su constructor normal de contexto:
  conserva Task, plan, notas, corpus, conocimiento del proyecto, indicaciones
  retenidas y posibles mensajes/adjuntos que llegaron mientras trabajaba.
- Añadir feedback y respuesta rechazada después de ese contexto sin mutar la
  lista que devuelve el callback. Las rondas de tools de una corrección siguen
  ampliando su propia conversación.
- El constructor de corrección también sondea las indicaciones y los adjuntos
  reenviados al worker, como lo hace el inner loop ordinario.
- Aplicar el bookkeeping de caché existente después de tools correctivos para
  que otro intento no vuelva a presentar el código anterior. Los comandos de
  shell se reconcilian incluso cuando terminan con error, porque pueden haber
  escrito parcialmente. Resolver rutas relativas desde el workspace del agente
  antes de recurrir al contexto del thread o al cwd del proceso.
- Los consumidores directos sin callback reciben un contexto mediante el
  renderer normal a partir de descripción, esperado y `LoopState`; con estado
  ausente se usa un estado vacío. No inventar evidencia.
- Mantener los límites de herramientas/tokens, uso contabilizado, hooks y
  seguimiento de cambios ya existentes. No llamar al constructor ni al modelo
  cuando el resultado se valida sin corrección.

## D. Avisos consumibles y ventanas de historial

Al inspeccionar la proyección de historial se reprodujo otro defecto: el renderer
consume avisos en una copia profunda de LoopState y deja intactos los campos del
estado real. Al siguiente Step vuelven a aparecer avisos de cierre rechazado,
guías o intervenciones que debían entregarse una sola vez.

Mantener la proyección de historial para lectura y pasar el estado real como
`pending_messages_state` al renderer. Solo este último es dueño del consumo de
`pending_engine_notice`, `pending_guidance` y `pending_runtime_intervention`.
Los consumidores anteriores omiten el argumento y siguen consumiendo su propio
estado. No recortar ni mutar el historial original.

## Límites

Sin dependencias nuevas, sin cambios a proveedor, budgets, modelo objetivo,
algoritmo de scheduling, criterios de éxito o política de permisos. Preservar
todos los cambios locales previos, especialmente `StepCompleteGate.evaluate`.
No hacer commits, merge ni publicación. No persistir razonamiento ni imágenes
codificadas con esta funcionalidad.

## Verificación

- Reproducciones RED→GREEN de una instrucción que desaparece en el siguiente
  prompt, imágenes perdidas al inicio/cierre y correcciones sin Task/evidencia.
- Pruebas del orden literal, repetición legítima, perfiles compactos, referencias
  de imágenes, FC/manual, no duplicación de tool results y fallback sin visión.
- Roundtrip de checkpoint, compatibilidad con estado viejo, compactación hasta
  campos esenciales, rechazo por instrucciones irreducibles y aislamiento de
  Task. Comprobar que una guía entregada llega al checkpoint antes del modelo.
- Snapshots restaurados conservan cuerpo y flags, se marcan para revalidación y
  vuelven a ser observaciones nuevas solo tras read/refresh.
- Corrección contextual real, callback perezoso por intento, contexto no mutado,
  presupuestos y fallback existentes.
- Suite relacionada, revisión independiente y `uv run --frozen --no-sync pytest`.
