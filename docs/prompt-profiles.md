# Perfiles de prompts

Infinidev permite ajustar fragmentos de prompt ya existentes sin editar el código ni
cambiar nada para usar los valores predeterminados. Un perfil es un archivo JSON del
proyecto en `.infinidev/prompts.json`. Por tanto se puede versionar junto al proyecto,
compartirlo, o publicar un perfil para un modelo concreto.

> El perfil selecciona, desactiva o anota fragmentos existentes; no permite introducir
> texto de prompt arbitrario. Esto conserva las reglas y la composición que Infinidev
> incluye por defecto.

## Formato

La raíz debe ser un objeto JSON. Cada sección de fase contiene identificadores de
fragmento y uno de estos valores:

- `true`: conserva el fragmento habilitado (equivale al valor predeterminado).
- `false`: elimina el fragmento de la composición.
- Un objeto cuyos valores son cadenas o números: conserva el fragmento y añade esos
  parámetros escalares al final del fragmento como metadatos XML
  `<prompt-profile>`. Los nombres de los parámetros deben ser cadenas. No se aceptan
  booleanos, listas, objetos anidados ni `null` como valores de parámetros.

```json
{
  "develop": {
    "loop.identity": false,
    "loop.protocol": {"verbosity": "compact", "max_examples": 1}
  },
  "execute": {
    "phase.feature.execute": {"focus": "tests"}
  }
}
```

Una configuración vacía, o la ausencia de `.infinidev/prompts.json`, conserva
exactamente la composición incorporada.

## Perfiles compartidos por modelo

La clave reservada `models` permite declarar las mismas secciones para un proveedor o
para un modelo exacto. Las claves usan el proveedor configurado (`provider`) o
`provider/model`:

```json
{
  "develop": {
    "loop.protocol": {"detail": "normal"}
  },
  "models": {
    "anthropic": {
      "develop": {
        "loop.protocol": {"detail": "concise"}
      }
    },
    "anthropic/claude-sonnet-4": {
      "develop": {
        "loop.protocol": false
      }
    }
  }
}
```

Para cada fragmento, Infinidev busca una entrada en este orden:

1. `models["provider/model"][phase]`;
2. `models["provider"][phase]`;
3. la sección general `[phase]` de la raíz;
4. el valor incorporado.

La primera entrada que nombre el fragmento gana; las entradas no se mezclan entre sí.
Esto permite que un perfil general siga funcionando para otros modelos y que una
configuración específica sustituya solamente lo que necesita.

## Validación y recuperación

- JSON inválido, una raíz que no sea un objeto, `models` que no sea un objeto y un
  valor inválido en una sección que se consulte producen un error de perfil. Corrija el
  archivo antes de ejecutar.
- Las secciones que la composición no consulta y los nombres de fragmento que esta no
  solicita no afectan al resultado. Esto permite que un archivo compartido contenga
  ajustes para otras versiones o flujos.
- Los perfiles no cambian el contenido incorporado de ningún prompt. Al habilitar un
  fragmento sin parámetros, el resultado es el mismo fragmento (o la variante de
  estilo activa) que se usaría sin perfil.

## Catálogo estable actual

Los identificadores son nombres con puntos. Los siguientes son los fragmentos que la
composición actual resuelve mediante perfiles. El catálogo contiene 101 identificadores:
71 bloques con nombre fijo y 30 bloques de estrategia (cinco tipos de tarea por tres
fases, con guía e identidad independientes).

### Bucle de desarrollo (`develop`)

| Grupo | Identificadores |
| --- | --- |
| Sistema | `loop.identity`, `loop.protocol`, `loop.behavior_guidelines`, `loop.technology_guidance`, `loop.project_instructions`, `loop.critic_guidance`, `loop.session_context` |
| Contexto de iteración | `iteration.smart_summary`, `iteration.project_knowledge`, `iteration.context_corpus`, `iteration.context_rank`, `iteration.workspace`, `iteration.background_completions`, `iteration.background_tasks`, `iteration.reactive_guidance`, `iteration.opened_files`, `iteration.session_notes`, `iteration.working_notes`, `iteration.note_nudge`, `iteration.previous_actions`, `iteration.anti_patterns`, `iteration.behavior_summary`, `iteration.next_actions`, `iteration.context_budget` |

El objetivo de la tarea, el plan activo, la acción actual, su salida esperada y los
contratos de terminación/herramientas no son fragmentos independientes: forman el
estado mínimo que permite al engine avanzar con seguridad.

### Planificadores (`plan`)

| Grupo | Identificadores |
| --- | --- |
| Task planner | `task_planner.identity`, `task_planner.methodology`, `task_planner.planning_vocabulary`, `task_planner.handoff_guidance`, `task_planner.decomposition_guidance`, `task_planner.verification_guidance`, `task_planner.examples` |
| Stage planner | `stage_planner.identity`, `stage_planner.methodology`, `stage_planner.planning_vocabulary`, `stage_planner.authority_guidance`, `stage_planner.horizon_guidance`, `stage_planner.decision_guidance`, `stage_planner.decomposition_guidance`, `stage_planner.examples` |

Los hechos de máquina y los contratos de salida de `emit_task_plan`, `emit_stage`,
`complete_goal` y `block_goal` permanecen siempre activos.

### Evaluación (`review`)

| Grupo | Identificadores |
| --- | --- |
| Revisor | `reviewer.identity`, `reviewer.input_guidance`, `reviewer.authority_guidance`, `reviewer.evaluation_guidance`, `reviewer.severity_guidance` |
| Extracción y juicio | `extractor.identity`, `judge.identity`, `judge.input_guidance`, `judge.authority_guidance`, `judge.evaluation_guidance`, `judge.severity_guidance` |
| Evidencia | `evidence.identity`, `evidence.evaluation_guidance`, `adversarial.identity`, `adversarial.evaluation_guidance` |

Los esquemas JSON, las reglas de extracción y los formatos de veredicto son contratos
atómicos y no se deshabilitan por perfil.

### Chat, consejo, recopilación y resúmenes

| Sección | Identificadores |
| --- | --- |
| `chat` | `chat.identity`, `chat.language_guidance`, `chat.council_guidance`, `chat.followup_guidance`, `chat.project_instructions`, `chat.model_guidance` |
| `council` | `council.seed_identity`, `council.member_identity`, `council.judge_identity`, `council.synthesis_identity`, `council.language_guidance`, `council.persona_palette` |
| `gather` | `gather.identity_guidance`, `gather.classifier_guidance`, `gather.synthesis_guidance`, `gather.question_guidance` |
| `summarize` | `summary.step_guidance` |

En estas familias permanecen obligatorios el routing `respond`/`escalate`, los
terminadores y transiciones del consejo, el modo de investigación de solo lectura,
las taxonomías necesarias y todos los formatos de salida parseables.

### Fases de estrategia

Use la sección de fase que aparece en la primera columna. Cada tipo de tarea dispone
de una guía y una identidad independientes; estos son los 30 identificadores exactos:

| Sección | Tipo | Identificadores |
| --- | --- | --- |
| `investigate` | `bug` | `phase.bug.investigate`, `phase.bug.investigate_identity` |
| `investigate` | `feature` | `phase.feature.investigate`, `phase.feature.investigate_identity` |
| `investigate` | `refactor` | `phase.refactor.investigate`, `phase.refactor.investigate_identity` |
| `investigate` | `other` | `phase.other.investigate`, `phase.other.investigate_identity` |
| `investigate` | `sysadmin` | `phase.sysadmin.investigate`, `phase.sysadmin.investigate_identity` |
| `plan` | `bug` | `phase.bug.plan`, `phase.bug.plan_identity` |
| `plan` | `feature` | `phase.feature.plan`, `phase.feature.plan_identity` |
| `plan` | `refactor` | `phase.refactor.plan`, `phase.refactor.plan_identity` |
| `plan` | `other` | `phase.other.plan`, `phase.other.plan_identity` |
| `plan` | `sysadmin` | `phase.sysadmin.plan`, `phase.sysadmin.plan_identity` |
| `execute` | `bug` | `phase.bug.execute`, `phase.bug.execute_identity` |
| `execute` | `feature` | `phase.feature.execute`, `phase.feature.execute_identity` |
| `execute` | `refactor` | `phase.refactor.execute`, `phase.refactor.execute_identity` |
| `execute` | `other` | `phase.other.execute`, `phase.other.execute_identity` |
| `execute` | `sysadmin` | `phase.sysadmin.execute`, `phase.sysadmin.execute_identity` |

Por ejemplo, para retirar la identidad de la fase de planificación de una tarea de
funcionalidad:

```json
{
  "plan": {
    "phase.feature.plan_identity": false
  }
}
```

Las preguntas iniciales y los límites numéricos de cada estrategia no son fragmentos
configurables independientes. Las identidades de `flows` fuera de la composición
anterior tampoco lo son. Declarar nombres no incluidos en este catálogo no cambia el
runtime hasta que una composición los exponga explícitamente.

## Materialización por ejecución

Al comenzar una tarea, el pipeline lee y valida `.infinidev/prompts.json` exactamente
una vez y construye un snapshot efectivo inmutable. La misma instancia se comparte con
chat, council, planners, gather, developer, review y los ciclos de corrección o reentrada
de esa tarea. Ningún compositor vuelve a consultar el archivo durante el recorrido.

Editar el JSON mientras una tarea está en marcha no modifica sus prompts: el cambio se
aplica al comienzo de la siguiente tarea. Cuando uno de esos subsistemas se invoca de
forma independiente —fuera del pipeline— compila igualmente un solo snapshot para esa
invocación y lo reutiliza hasta que termina.

La precedencia se resuelve al compilar el snapshot, no al componer cada fragmento:
configuración general, después proveedor y finalmente modelo exacto. Dentro del snapshot
resultante, el valor más específico sustituye al menos específico para el mismo
identificador; los parámetros de varias capas no se fusionan.

## Relación con los estilos de prompt

El ajuste `PROMPT_STYLE` sigue seleccionando la variante incorporada (`full`,
`generalized`, `coding` o `extra_simple`; `auto` resuelve a `generalized`). El perfil
se aplica después de escoger esa variante. Por tanto un mismo `prompts.json` puede
compartirse entre estilos, y deshabilitar un identificador tiene el mismo efecto para
la variante que esté activa.
