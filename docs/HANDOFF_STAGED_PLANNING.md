# Handoff — reemplazo del plan estático por planning adaptativo por Stages

> Destinatario: el modelo que continúa este trabajo. Fecha: 2026-08-05. Worktree: trabajo en progreso sin commitear sobre `main`.

Continúa el trabajo en `/home/andres/infinidev`, termínalo y haz una revisión crítica del resultado. No te limites a diagnosticar ni a escribir un diseño: implementa, prueba y corrige lo necesario.

Antes de modificar:

1. Lee `AGENTS.md` y sigue sus instrucciones, especialmente el uso de ken.
2. Inspecciona el worktree actual. Contiene trabajo en progreso que debes preservar y revisar; no lo reviertas ni lo reemplaces ciegamente.
3. Lee los brain maps y la política de prompting:
   - `docs/GPT_5_6_SOL_PROMPT_COMPREHENSION_MENTAL_MAP.md`
   - `docs/MINIMAX_M3_PROMPT_COMPREHENSION_MENTAL_MAP.md`
   - `docs/PROMPTING.md`
   - `tests/test_prompt_style_rules.py`

## Objetivo del cambio

Reemplazar conceptualmente el plan estático por un sistema adaptativo:

```
Goal
└── Stage 1..N
    ├── Task 1..M
    │   └── Step 1..K
    └── Task ...
```

El Goal es el resultado estable controlado por el usuario.

El Stage representa el siguiente horizonte planificable. Después de ejecutar y medir un Stage, el sistema vuelve a evaluar el Goal y decide entre:

- Goal completado con evidencia.
- Goal bloqueado por una decisión del usuario, autoridad ausente o estado externo.
- Emitir el siguiente Stage.

Cada Stage contiene múltiples Tasks con dependencias. Cada Task se convierte en Steps concretos mediante el Task Planner y se ejecuta usando LoopEngine.

Debe ser un sistema único:

- Un pedido pequeño y claro puede terminar con un Stage, un Task y pocos Steps.
- Un objetivo claro pero largo o incierto puede requerir muchos Stages.
- Un objetivo cuyo final no sea decidible necesita aclaración o bloqueo; no debe confundirse “objetivo ambiguo” con “camino todavía desconocido”.
- El número de Stages, Tasks o Steps no debe decidirse mediante límites arbitrarios.

## Principios de prompting

Los prompts deben funcionar como guías de decisión, no como catálogos de prohibiciones.

Solo deben formularse como contratos rígidos:

- La autoridad y el alcance del usuario.
- La veracidad de la evidencia y de las afirmaciones de completitud.
- Los límites reales del rol.
- Los hechos de la máquina y schemas de herramientas.

Los métodos deben expresarse como:

“Preferí X porque produce Y; apartate cuando la evidencia muestre Z.”

Evita palabras que escondan el criterio, como:

- relevant
- sufficient
- reasonable
- meaningful progress
- appropriate
- complex
- as needed

Cuando se use un concepto como progreso, debe tener significado observable: cambió el estado de un criterio, apareció evidencia nueva o se eliminó una incógnita que impedía decidir.

Los ejemplos deben declarar que ilustran granularidad o formato. Sus paths, comandos y acciones no son evidencia ni autorización para el trabajo real.

Preserva estas conclusiones de los brain maps:

- Separar `USER_LITERAL`, `DERIVED` y `OBSERVED_EVIDENCE`.
- Un plan derivado no puede convertirse en requisito del usuario.
- No usar confidence como gate.
- No ampliar un target singular ambiguo a todos sus candidatos.
- No interpretar el wrapper o los ejemplos como scope.
- Mantener prompts concisos: Sol expande instrucciones compactas.
- Usar schemas simples para MiniMax y modelos menos fiables con estructuras profundas.
- Los fallos materiales y la verificación deben reportarse con honestidad.
- Los comandos de verificación escritos por el planner son output no confiable y deben respetar permisos.

## Trabajo ya realizado

Se agregaron:

- `src/infinidev/prompts/analyst/planning_vocabulary.py`
- `src/infinidev/prompts/analyst/stage_planner_prompt.py`
- `src/infinidev/prompts/analyst/task_planner_prompt.py`
- `src/infinidev/tools/planner/stage_decision_tools.py`
- `tests/test_staged_planning_prompts.py`

También se modificaron:

- `src/infinidev/engine/analysis/plan.py`
- `src/infinidev/engine/analysis/planner.py`
- `src/infinidev/prompts/analyst/__init__.py`
- `src/infinidev/prompts/analyst/planner_prompt.py`
- `src/infinidev/tools/__init__.py`
- `src/infinidev/tools/planner/__init__.py`
- `src/infinidev/tools/planner/emit_plan_tool.py`
- `tests/test_planner.py`
- `tests/test_prompt_style_rules.py`
- `tests/test_specialized_prompt_contracts.py`

Estado conceptual actual:

- Existe vocabulario semántico compartido.
- Existe un Stage Planner prompt.
- Existe un Task Planner prompt.
- El planner actual usa `emit_task_plan`.
- Hay roles separados `stage_planner` y `task_planner`.
- El Stage Planner tiene terminales:
  - `emit_stage`
  - `complete_goal`
  - `block_goal`
- El schema de Stage valida IDs, dependencias desconocidas, dependencias propias y ciclos.
- Los Steps se describen como tácticas model-inferred adaptables.
- Los criterios producidos por el Task Planner se denominan `derived_verification_criteria`.
- `Plan.acceptance_criteria` sigue existiendo como nombre de compatibilidad, pero el pipeline lo trata como verificación derivada.

Importante: el Stage Planner todavía no está conectado al pipeline. El runtime sigue ejecutando esencialmente una solicitud raíz con un Task Plan. No confundas la existencia del prompt y sus tools con la implementación completa del ciclo por Stages.

## Trabajo restante

### 1. Revisa críticamente el diff actual.

Busca:

- Contradicciones entre prompts y schemas.
- Herramientas documentadas pero no disponibles para el rol.
- Métodos presentados falsamente como invariantes.
- Palabras sin criterio observable.
- Campos que puedan promover checks derivados a requisitos del usuario.
- Compatibilidad con el planner y pipeline actuales.
- Complejidad innecesaria en los schemas.

### 2. Completa los artefactos de dominio.

La recomendación es:

- `GoalSpec`: contrato raíz inmutable.
- `StageSpec`: horizonte activo y sus criterios de salida.
- `TaskSpec`: entregable dentro de un Stage.
- `TaskPlan`: Steps tácticos de una Task.
- Decisiones tipadas del Stage Planner.

Actualmente `engine/orchestration/task_schema.py::Task` representa el pedido global. Evita la colisión con las nuevas Tasks internas. Puedes introducir `GoalSpec` y mantener un alias temporal `Task = GoalSpec` si eso reduce el riesgo de migración.

### 3. Implementa el Stage Planner runtime.

Debe:

- Recibir Goal, historial de Stages, ledger de evidencia y estado actual.
- Exponer solamente herramientas read-only más sus tres terminales.
- Interpretar exactamente una decisión terminal.
- Tener recuperación tolerante para modelos sin function calling.
- No convertir agotamiento de presupuesto en éxito.
- No declarar completitud por queue vacía o plan terminado.

### 4. Implementa el orquestador Stage → Tasks → Steps.

Flujo esperado:

```
Stage Planner
- complete_goal: revisión final y cierre.
- block_goal: handoff honesto al usuario.
- emit_stage:
  - validar DAG;
  - seleccionar Tasks dependency-ready;
  - ejecutar Task Planner para cada Task;
  - ejecutar sus Steps mediante LoopEngine;
  - verificar resultado de cada Task;
  - agregar evidencia del Stage;
  - volver al Stage Planner.
```

No fuerces paralelismo si el runtime no puede preservar correctamente estado, evidencia y persistencia. Las dependencias deben representar flujo de resultados, no simple preferencia de orden.

### 5. Mantén separadas las comprobaciones:

- Step complete: acción local verificada.
- Task complete: entregable de la Task verificado.
- Stage complete: hito o aprendizaje del Stage verificado.
- Goal complete: criterios globales respaldados por evidencia.

“No quedan Steps” nunca significa automáticamente “Goal completado”.

### 6. Maneja estancamiento sin umbrales semánticos arbitrarios.

Compara cada nuevo Stage con el historial:

- ¿Movió un criterio?
- ¿Produjo una observación nueva?
- ¿Eliminó una incógnita que impedía decidir?
- ¿Cambió inputs o método de modo que repetir una acción pueda producir otro resultado?

Si no existe una ruta in-scope que pueda producir nueva evidencia, bloquea honestamente. Los límites de tokens, tools o Stages son límites de recursos, no criterios de éxito.

### 7. Completa persistencia, resume y TUI en proporción al runtime nuevo.

El estado debería poder reconstruir:

- Goal.
- Stage activo e historial.
- Tasks, dependencias y estados.
- Task Plan y Steps.
- Evidencia y decisiones terminales.
- Bloqueos y trabajo incompleto.

La UI debe mostrar la jerarquía sin mezclar Goal, Stage, Task y Step.

## Pruebas requeridas

Agrega regresiones para:

- Pedido pequeño: un Stage, una Task, Steps concretos.
- Goal claro con camino largo: múltiples Stages.
- Goal con final decidible pero ruta desconocida.
- Goal cuyo final depende de una decisión del usuario: bloqueo.
- Target singular ambiguo: no ampliar a todos los candidatos.
- Tasks con dependencias.
- Dependencia desconocida, propia o cíclica.
- Task bloqueada durante ejecución.
- Stage que produce evidencia y causa una estrategia posterior distinta.
- Intento de declarar complete sin evidencia.
- Queue vacía que no completa el Goal.
- Reanudación de una sesión a mitad de Stage.
- Separación entre requisitos del usuario y checks derivados.
- Herramientas correctas para cada rol.
- Prompts sin palabras threshold-free, contradicciones ni herramientas retiradas.

## Validación conocida

Antes de los últimos ajustes de redacción se ejecutó la suite completa:

`3017 passed, 1 skipped, 1 warning`

Después de los últimos ajustes se ejecutaron las pruebas enfocadas:

`312 passed in 8.19s`

Se inició nuevamente la suite completa sobre el estado final, pero el usuario interrumpió el turno cerca del 7 %. El proceso ya no está corriendo. No presentes esa última suite como completada.

`git diff --check` está limpio.

Al terminar ejecuta, sobre el estado final exacto:

- `uv run pytest`
- `git diff --check`

Si la suite completa no puede terminar, reporta exactamente qué se ejecutó y qué quedó pendiente.

## Entrega final

Revisa el resultado como si fueras un reviewer adversarial:

- Señala y corrige problemas encontrados.
- No ocultes incompatibilidades.
- No afirmes que el Stage loop está implementado si solo existen prompts o schemas.
- Resume los archivos y comportamiento final.
- Incluye los comandos de prueba y sus resultados reales.
- No hagas commit ni push salvo que el usuario lo pida explícitamente.
