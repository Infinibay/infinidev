# Equipos de investigación

Escribe una tarea en un mensaje normal, por ejemplo: «Investiga por qué el cache pierde
gradientes y comprueba la causa». El agente principal recibe el pedido y decide cómo
resolverlo: puede responder o inspeccionar fuentes directamente, o crear tickets y delegar
especialistas cuando hacen falta ejecución, investigaciones independientes o una auditoría.
Él organiza el trabajo, revisa la evidencia y responde al usuario.

La conversación normal utiliza este flujo por defecto. No hace falta un comando, pedir
subagentes ni administrar tickets. `/engine` queda como selección avanzada para ejecutar
flujos de compatibilidad; una elección explícita guardada de otro engine sigue vigente.

El orquestador utiliza el modelo configurado y el LoopEngine existente. Sus trabajadores
usan ese mismo modelo y las mismas reglas del proyecto y perfiles de prompt. Cada
delegación añade una especialidad explícita y una lista exacta de herramientas. Los
esquemas de las otras herramientas quedan fuera del contexto del trabajador y de su
dispatch. La selección no concede permisos nuevos de filesystem. Conceder shell o un
intérprete permite las operaciones normales de esa herramienta dentro del sandbox.

## Flujo

1. El principal mantiene el pedido literal y decide si puede resolverlo con información
   establecida y consultas directas. Si delega, crea tickets con pregunta, entregable,
   aceptación, restricciones y dependencias.
2. Consulta `team_tool_catalog` y utiliza `team_delegate` para asignar un ticket,
   `system_prompt`, nombre y herramientas. No carga todos los esquemas en su propio prompt.
3. Los trabajadores inspeccionan fuentes, ejecutan comprobaciones autorizadas, publican
   notas y preguntan a compañeros. Reciben también los adjuntos del turno.
4. Un reporte entregado deja el ticket en `review`. El principal inspecciona su evidencia
   y usa `team_review_ticket` para aceptarlo, pedir correcciones o cancelar trabajo obsoleto.
5. Las dependencias se desbloquean al aceptar sus tickets. El runtime impide terminar con
   éxito mientras queden trabajadores activos o tickets sin resolver.

La aceptación es una decisión registrada del orquestador. El runtime verifica la
transición de estado; no demuestra automáticamente que una afirmación científica sea
verdadera. El prompt exige contrastar evidencia y conservar los resultados negativos.

## Conversación y notas

`team_send_message` acepta el ID o nombre de un compañero, `orchestrator` o `all`.
Devuelve inmediatamente un ID. Ejemplo de interacción:

```text
Investigador: Che, ¿puedes averiguar si la escritura del cache hace detach?
Auditor: Sí. En cache.py:42 se llama detach antes de guardar. Inspeccioné el código;
         todavía no ejecuté una prueba de autograd. [reply_to: ID de la pregunta]
```

Una pregunta nueva puede reactivar a un trabajador inactivo. Las respuestas con
`reply_to` llegan a los agentes activos y quedan en el historial; no reactivan por sí
solas a los inactivos, para evitar conversaciones infinitas de acuses de recibo. Cada
agente puede consultar conversaciones anteriores con `team_read(view="messages")`.
Los mensajes de compañeros están etiquetados como evidencia de colaboradores, separados
de los mensajes nuevos del usuario. Una petición de un compañero no amplía el alcance.

`team_write_note` registra autor, fecha, referencias, ticket y tipo (`hypothesis`,
`observation`, `decision`, `handoff`). El autor proviene del contexto real del agente,
no de un argumento que el modelo pueda inventar. `supersedes` crea una revisión; la nota
anterior conserva su contenido y expone `superseded_by`. `team_read` permite paginar
notas, mensajes y eventos con `after`/`next_after`. Todos los miembros ven el historial.

Ken conserva findings duraderos revisados. Las notas del equipo conservan coordinación,
hipótesis y estado cambiante. Antes de publicar un finding se contrasta con los artefactos
actuales y se incluyen referencias y autores: la API de memoria de Ken observada no tiene
un campo independiente de autor y reutilizar un topic reemplaza su contenido.

## Persistencia y ejecución

El tablero y los eventos viven en la base de Infinidev, separados por proyecto, workspace
y sesión. Las actualizaciones usan transacciones de SQLite a través del helper existente.
Reabrir esa sesión recupera tickets, notas y conversaciones. Un trabajador que quedó
marcado en ejecución tras morir el proceso pasa a `interrupted`; su proceso no se
considera vivo por el texto guardado. La delegación puede reintentarse explícitamente.

Cada trabajador tiene ID y sesión de checkpoint propios. Las lecturas independientes
pueden ejecutarse juntas. Los trabajadores con herramientas de escritura se serializan
porque comparten workspace y baselines de cambios; MCP sin declaración de lectura se
trata conservadoramente como escritura. Esto evita atribuir o revertir cambios simultáneos
con el tracker actual. No proporciona worktrees separados.

Valores iniciales, todos configurables con prefijo `INFINIDEV_`:

| Setting | Valor |
| --- | --- |
| TEAM_MAX_WORKERS | 3 |
| TEAM_MAX_AGENTS | 12 identidades, reutilizables en nuevos tickets |
| TEAM_MAX_FOLLOWUPS | 8 reactivaciones automáticas por trabajador y turno |
| TEAM_WORKER_MAX_ITERATIONS | 12 |
| TEAM_WORKER_MAX_TOOL_CALLS | 80 |

Los trabajadores no delegan otros trabajadores. La cancelación del principal se propaga
a todos; el cierre espera su cancelación cooperativa antes de liberar la sesión. Los
tiempos máximos de las llamadas de red siguen dependiendo del timeout configurado.
Las métricas del resultado suman tokens, herramientas e iteraciones del principal y de
todos los trabajadores, incluidas sus reactivaciones. `/think` y `GATHER_ENABLED`
conservan la recopilación de contexto previa cuando el usuario la solicita.
No es un canal para contactar procesos Codex arbitrarios del sistema.

## Prompts y verificación

Las guías siguen [PROMPTING.md](PROMPTING.md): autoridad de entradas, hechos del runtime,
criterios de resultado, métodos con condiciones, continuidad y ejemplo de fallo/corrección.
`team.orchestrator_guidance` y `team.worker_guidance` usan el snapshot de perfiles del turno;
su catálogo se encuentra en `45-team.json`. Las identidades especializadas también se
conservan en el camino compacto para modelos locales.

`tests/test_research_team.py` cubre atribución, revisiones, carreras de mensajes,
aislamiento, recuperación, cancelación, paralelismo de lectura, serialización de escritura,
grants efectivos, composición de prompts y veto de éxito sin revisión. También recorre la
entrada de un mensaje normal, sin comando de modo, tanto con respuesta directa como con
delegación y revisión. Las decisiones del modelo se simulan para verificar ese recorrido.
La [investigación](AGENT_REASONING_RESEARCH.md) distingue estos contratos de ingeniería
de una evaluación de calidad del modelo en investigación científica.
