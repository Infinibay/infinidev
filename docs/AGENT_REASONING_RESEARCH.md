# Investigación: cómo deciden los agentes y qué debe observar Infinidev

Fecha: 2026-09-06. Alcance: revisión del código de Infinidev, estudios existentes del
repositorio, documentos operativos de SENN/new_research, API de memoria de Ken y fuentes
primarias. No se ejecutaron nuevos entrenamientos de SENN ni una campaña de llamadas a
modelos para esta revisión.

## Pregunta y conclusión operativa

La pregunta útil para el producto es qué información modifica una decisión observable:
qué herramienta se elige, qué evidencia se consulta, cuándo se delega, cómo se corrige
una hipótesis y qué justifica terminar. No tenemos acceso directo a los estados internos
de los modelos hospedados. Una explicación verbal y un campo de thinking expuesto por
una API tampoco certifican la causa completa de una decisión.

ReAct estudia la alternancia entre razonamiento, acciones y observaciones del entorno:
consultar una fuente puede cambiar el plan siguiente. Sus resultados respaldan estudiar
el ciclo completo, dentro de las tareas evaluadas, no sólo una respuesta aislada.
[Yao et al., ReAct](https://arxiv.org/abs/2210.03629).

Un estudio de Anthropic encontró decisiones influidas por pistas que los modelos no
siempre reconocían en el razonamiento expresado. Por eso, para Infinidev, interpretar el
texto de thinking como prueba de obediencia o causalidad sería una inferencia injustificada.
Las acciones, las entradas disponibles y los resultados verificables deben conservarse.
[Chen et al., 2025](https://www.anthropic.com/research/reasoning-models-dont-say-think).

## Evidencia local

| Fuente inspeccionada | Observación | Implicación para el diseño |
| --- | --- | --- |
| [Estudio de GPT-5.6 Sol](GPT_5_6_SOL_PROMPT_COMPREHENSION_MENTAL_MAP.md) | 672 respuestas estructuradas; confianza declarada media 0,9744; reconstrucciones expansivas y asimilación de instrucciones del envoltorio | Una reconstrucción fluida y confiada no basta para autorizar acciones ni cerrar trabajo |
| [Estudio de GLM 5.2](GLM_5_2_PROMPT_COMPREHENSION_MENTAL_MAP.md) | 671 respuestas estructuradas de 672; confianza media 0,901; patrones de cierre anticipado y expansión de alcance ambiguo | El sistema debe conservar el pedido literal y distinguir criterios derivados de autoridad |
| [Método de investigación de prompts](PROMPT_COMPREHENSION_RESEARCH_METHOD.md) | Separa comprensión, ejecución y preferencia; requiere variantes controladas y confirmación en ejecución | Un cambio de prompt se evalúa por un efecto observable en una tarea, no por su plausibilidad editorial |
| [Prompts de razonamiento condicional](CONDITIONAL_REASONING_PROMPTS.md) | Existen resultados de calibración/validación y casos de ejecución; no equivalen a garantía general | Las intervenciones se condicionan a señales observables y se contrastan con errores que introducen |

Las cifras anteriores describen los informes existentes, no una nueva medición ni una
comparación causal entre proveedores. Las campañas difieren en detalles; no deben usarse
como ranking universal. Los artefactos y límites de cada estudio permanecen en sus informes.

## Lo que aportó SENN

Se inspeccionaron `AGENTS.md`, `new_research/CONTINUE.md`, `README.md`, `HYPOTHESES.md`,
`STUDY_LOG.md`, `results/README.md` y el cambio de objetivo
`LEARNING_PER_WALL_TIME_OBJECTIVE_2026-09-06.md` bajo `/Users/andres/Projects/SENN`.
También se consultaron findings mediante el CLI de Ken desde ese proyecto.

El proyecto exige un lead que delega trabajo acotado y verifica los resultados. Las
líneas de investigación prueban separación entre pesos de razonamiento y conocimiento
externo, incluidos controles contrafactuales al cambiar ese conocimiento. Los resultados
se documentan con hipótesis, protocolo, revisión, comandos, artefactos, métricas y límites.

La lectura mostró un problema de continuidad concreto: documentos extensos y findings
conservan campañas históricas descritas como activas aunque registros posteriores las
cierran. Además, el objetivo más reciente prioriza aprendizaje útil por tiempo de pared,
por encima del objetivo histórico de throughput. Un agente que toma el primer resumen
como estado actual puede continuar la campaña equivocada.

En el resultado negativo más reciente inspeccionado, mejorar una métrica de formato no
confirmaba la hipótesis experimental; aparecía un comportamiento degenerado en respuestas.
La lección de producto es conservar los controles y el veredicto negativo, junto a la
métrica. Los experimentos A/B prerregistrados requerían ejecución secuencial por recursos
y comparabilidad; los análisis independientes podían avanzar en paralelo.

En Ken, `remember` reemplaza contenido por topic y conserva anclajes/timestamps, pero la
implementación inspeccionada no registra autor separado. Por eso el estado cambiante del
equipo necesita notas con procedencia y revisiones propias. Los findings revisados siguen
siendo útiles para conocimiento duradero, con referencias que permiten revalidarlo.

## Hipótesis de diseño e intervención implementada

**H1: delegaciones con pregunta, aceptación y herramientas acotadas reducen duplicación y
omisiones.** El patrón lead/trabajadores tiene evidencia de ingeniería en el sistema de
Research de Anthropic. Ese informe también señala mayor consumo de tokens y dificultades
en tareas con dependencias fuertes. No se extrapolan sus porcentajes a Infinidev: aquí la
consecuencia es delegar por dependencias reales y medir el costo total del equipo.
[Informe del sistema multiagente](https://www.anthropic.com/engineering/multi-agent-research-system).

**H2: separar entrega y aceptación reduce el cierre basado en afirmaciones sin comprobar.**
Se implementaron tickets `review`, aceptación explícita, dependencias que requieren esa
aceptación y un veto mecánico al cierre mientras haya trabajo pendiente. La calidad del
juicio del orquestador sigue siendo una variable que medir.

**H3: procedencia y revisión de notas reducen reutilización de estados caducados.** Se
implementó historial con autor ligado al agente, referencias y supersesión. Esto hace
posible inspeccionar quién afirmó qué y qué observación lo reemplazó; no hace verdadera
una nota automáticamente.

**H4: mensajes asíncronos permiten resolver preguntas cruzadas sin perder la tarea.** Se
implementaron destinatarios, respuestas enlazadas, entrega entre llamadas y reactivación
acotada. Los mensajes son evidencia de colaboradores y no se inyectan como nuevas órdenes
del usuario. La cancelación y el aislamiento de checkpoints son parte del contrato.

**H5: esfuerzo correcto por API evita confundir parámetros inválidos con debilidad del
modelo.** El mismo nombre de esfuerzo no representa el mismo presupuesto ni efecto entre
proveedores. La [matriz de soporte](model-support.md) y las pruebas de payload permiten
separar errores de transporte de resultados de calidad.

## Confirmación experimental pendiente

Las pruebas de regresión verifican transiciones, procedencia, carreras y composición;
no miden si los modelos hacen mejores investigaciones. Para confirmar H1–H5, la siguiente
campaña debe usar el método existente del repositorio y mantener separadas comprensión y
ejecución. Protocolo propuesto, todavía no ejecutado:

1. Congelar casos y rúbricas antes de modificar prompts: estados históricos contradictorios,
   resultado negativo con métrica favorable, dependencia experimental secuencial, pregunta
   cruzada sobre gradients, mensaje de compañero que sugiere ampliar alcance y fallo de tool.
2. Comparar `task` y `orchestrator` con el mismo modelo, esfuerzo, fuentes y presupuesto total
   del equipo. Separar una condición con notas de otra sin notas; no cambiar todo a la vez.
3. Guardar revisión, hash de prompts, grants, entradas, llamadas, eventos, resultados,
   tokens de todos los agentes y tiempo de pared. No exigir cadenas privadas de pensamiento.
4. Puntuar exactitud sustentada, cobertura del pedido, fidelidad de citas, preservación de
   negativos, trabajo duplicado, errores de alcance, cierres falsos y costo por caso resuelto.
5. Mantener familias reservadas que no se usen para ajustar el prompt. Usar adjudicación
   independiente de las explicaciones del agente y revisar desacuerdos con fuentes originales.

La decisión de producto se apoya en los contratos ya comprobados y en el encaje con el
flujo real de SENN. Una afirmación de mejora de calidad científica necesita esa comparación
controlada; esta revisión no la da por demostrada.
