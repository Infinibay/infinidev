# Perfiles de prompts

Infinidev permite ajustar fragmentos de prompt ya existentes sin editar el código ni
cambiar nada para usar los valores predeterminados. Los perfiles compartidos viven como
archivos JSON en `~/.infinidev/prompts/`; el perfil opcional del proyecto permanece en
`.infinidev/prompts.json`. Así se puede mantener un catálogo personal reutilizable y,
a la vez, versionar ajustes específicos junto al proyecto.

> El perfil selecciona, desactiva o anota fragmentos existentes; no permite introducir
> texto de prompt arbitrario. Esto conserva las reglas y la composición que Infinidev
> incluye por defecto.

## Texto incorporado de las fases

Los prompts base de `PhaseEngine` también están separados del código Python. Viven
como recursos empaquetados en `src/infinidev/prompts/phase_resources/`:

    phase_resources/
    |-- shared.json
    |-- bug.json
    |-- feature.json
    |-- refactor.json
    `-- other.json

Cada texto se representa como un array de líneas para que los cambios sean legibles en un
diff. `shared.json` contiene los contratos comunes de investigación y edición, la
identidad del planificador y el prompt de seguimiento. Los otros cuatro archivos contienen
las preguntas, identidades y prompts de investigación, plan y ejecución de cada tipo de
tarea. La estrategia `sysadmin` continúa heredando los textos de
`other.json`.

El cargador valida la versión del esquema, todos los campos y los marcadores compartidos,
y conserva los finales de línea y los placeholders de runtime. Los módulos de
`prompts/phases/*.py` sólo mantienen los nombres de importación históricos para no
romper consumidores. Estos recursos definen el texto incorporado; los perfiles descritos
debajo siguen siendo la capa de activación y parámetros del usuario.

## Formato

La raíz debe ser un objeto JSON. Cada sección de fase contiene identificadores de
fragmento y uno de estos valores:

- `true`: conserva el fragmento habilitado (formato heredado).
- `false`: elimina el fragmento de la composición (formato heredado).
- Un objeto plano cuyos valores son cadenas o números: conserva el fragmento y añade
  esos parámetros escalares al final del fragmento como metadatos XML
  `<prompt-profile>` (formato heredado).
- Un objeto estructurado puede usar `enabled_by_default` para declarar de forma explícita
  el estado de fábrica, `enabled` para sustituir el estado efectivo y `parameters` para
  adjuntar un objeto plano de parámetros escalares. También se aceptan las formas concisas
  `{"enabled": true}` y `{"parameters": {...}}`; cuando falta `enabled_by_default`, su
  valor implícito es `true`, y cuando falta `enabled`, se usa el estado de fábrica.

Los nombres de parámetros deben ser cadenas. No se aceptan booleanos, listas, objetos
anidados ni `null` como valores de parámetros, y los objetos estructurados no admiten
campos desconocidos.

```json
{
  "develop": {
    "loop.identity": {
      "enabled_by_default": true,
      "enabled": false
    },
    "loop.protocol": {
      "enabled_by_default": true,
      "parameters": {"verbosity": "compact", "max_examples": 1}
    }
  },
  "execute": {
    "phase.feature.execute": {"focus": "tests"}
  }
}
```

Una configuración vacía conserva exactamente la composición incorporada.

## Catálogo, primera ejecución y precedencia de archivos

En la primera carga predeterminada, Infinidev crea `~/.infinidev/prompts/`. Si el
directorio está vacío, materializa siete perfiles iniciales. Los primeros seis exponen los
101 fragmentos activos del runtime. Cada entrada declara `"enabled_by_default": true` de
forma visible y mantiene habilitada la conducta incorporada, para mostrar tanto el estado
de fábrica como una organización editable por capacidad:

```text
~/.infinidev/prompts/
├── 10-development.json
├── 20-planning.json
├── 30-review.json
├── 40-collaboration.json
├── 50-investigation.json
├── 60-execution.json
└── 90-optional-capabilities.json
```

El séptimo archivo, `90-optional-capabilities.json`, ofrece un catálogo de 42 guías
adicionales para `debugging`, pruebas, revisión, seguridad, documentación, performance,
accesibilidad,
migraciones de datos, claridad de requisitos, impacto arquitectónico, contratos de API,
concurrencia, ciclo de vida de recursos, recuperación de errores, calidad de pruebas, corrección
algorítmica, observabilidad, preparación de releases, cambios de dependencias y configuración,
compatibilidad entre plataformas, experiencia CLI, estados completos de UI, verificación en
navegador, localización, handoff, gestión de contexto, integridad de datos, privacidad, respuesta a
incidentes, límites de autoridad, integridad de verificación, preservación de cambios, disciplina
de progreso, investigación con evidencia, refactorización, limpieza de código obsoleto, higiene
Git, control de costo, mentoría, análisis de causa raíz y evaluación de sistemas de IA. Todas declaran
`"enabled_by_default": false`: no cambian el prompt hasta que el usuario las active.

La forma recomendada de descubrirlas y administrarlas, tanto en la TUI como en el modo
clásico, es el comando integrado:

```text
/prompts
/prompts search trust boundaries
/prompts show debugging
/prompts enable debugging
/prompts disable debugging
/prompts reset debugging
```

`/prompts` muestra el estado efectivo de cada capacidad y `show` permite leer su condición
y guía incorporada antes de activarla. Los cambios se guardan en
`~/.infinidev/prompts/99-user-overrides.json` y se aplican desde la tarea siguiente. `reset`
elimina sólo el override indicado para volver a heredar el estado de los demás perfiles, sin
alterar otras preferencias. También se aceptan identificadores completos, por ejemplo
`capability.debugging`. Como todos los
perfiles compartidos se cargan por nombre, un archivo posterior a `99-user-overrides.json`
puede contradecir el cambio; `.infinidev/prompts.json` siempre tiene aún más precedencia. En
ambos casos, el comando informa la discrepancia y muestra el estado efectivo.

La materialización nunca reemplaza una ruta existente. Completa por separado cada nombre
inicial que falte, aunque el directorio contenga perfiles personalizados,
`99-user-overrides.json` o una versión editada de otro inicial. Así, una actualización puede
incorporar archivos iniciales nuevos sin alterar ni un byte de los archivos existentes, y un
override creado por `/prompts` antes de la primera carga no deja el catálogo incompleto. Para
personalizar el catálogo, edite esos archivos o añada otros con extensión `.json`.

La carga predeterminada combina los `*.json` del catálogo en orden lexicográfico. Cada
archivo posterior sustituye únicamente los identificadores que vuelve a declarar; no
borra las otras fases ni entradas del archivo anterior. Por último, si existe
`.infinidev/prompts.json` en el proyecto actual, se aplica como capa de mayor precedencia.
Por tanto, de menor a mayor precedencia entre archivos:

1. archivos de `~/.infinidev/prompts/*.json`, ordenados por nombre;
2. `.infinidev/prompts.json` del proyecto actual.

Pasar una ruta explícita a la API de perfiles sigue leyendo sólo ese archivo, sin crear ni
combinar el catálogo del usuario.

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

Los identificadores son nombres con puntos. Los siguientes son los 101 fragmentos activos
que la composición actual resuelve mediante perfiles: 71 bloques con nombre fijo y 30
bloques de estrategia (cinco tipos de tarea por tres fases, con guía e identidad
independientes). Las capacidades opcionales se enumeran por separado más adelante.

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
terminadores y transiciones del consejo, el modo de investigación de sólo lectura,
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

## Capacidades opcionales curadas

Las capacidades opcionales son fragmentos incorporados y versionados con Infinidev, no texto
arbitrario descargado de terceros. Cada prompt vive en un recurso independiente bajo
`src/infinidev/prompts/capabilities/`. Los archivos se cargan por nombre; su prefijo numérico
mantiene un orden estable. El contenido usa este esquema:

```json
{
  "id": "capability.debugging",
  "reason": "the current task requires diagnosing incorrect behavior",
  "guidance": [
    "Reproduce the symptom or establish the failing contract before changing code.",
    "Form a narrow cause hypothesis from observed evidence."
  ]
}
```

El paquete valida campos obligatorios, IDs, texto no vacío y duplicados al cargar el catálogo.
Estos JSON contienen el texto incorporado; los perfiles de `~/.infinidev/prompts/` siguen
controlando únicamente su estado y sus parámetros.

Las capacidades se inspiran en patrones comunes de agentes de programación: reproducir antes
de depurar, pruebas ligadas al contrato, revisión basada en hallazgos, análisis de límites de
confianza, documentación verificable, optimización medida, interfaces accesibles y migraciones
recuperables. La adaptación local mantiene las políticas del proyecto: condiciones explícitas,
alcance literal, evidencia antes de afirmar resultados, reintentos acotados y separación entre
revisión y modificación.

| ID | Se aplica cuando | Adaptación principal |
| --- | --- | --- |
| `capability.debugging` | Hay comportamiento incorrecto que diagnosticar | Reproductor, hipótesis estrecha y mismo check tras el arreglo |
| `capability.testing` | Se añade o cambia comportamiento ejecutable | Prueba mínima ligada al contrato y expansión sólo por evidencia |
| `capability.review` | El usuario pide review o auditoría | Solo lectura, severidad e impacto; no autoriza correcciones |
| `capability.security` | Se tocan inputs no confiables o límites sensibles | Flujo entrada→sink, mínimo privilegio y abuso observado |
| `capability.documentation` | El resultado incluye documentación | Describe conducta existente y no promesas sin probar |
| `capability.performance` | La tarea exige mejorar performance | Baseline y comparación de la misma carga |
| `capability.accessibility` | Cambia una interfaz interactiva | Teclado, etiquetas, foco y rutas no visuales verificables |
| `capability.data_migration` | Cambian datos persistidos o formatos serializados | Compatibilidad, recuperación y prueba con datos viejos y nuevos |
| `capability.requirements_clarity` | El resultado es ambiguo o admite interpretaciones incompatibles | Separa hechos, supuestos y decisiones que pertenecen al usuario |
| `capability.architecture_impact` | El cambio cruza módulos, procesos o contratos públicos | Traza consumidores, datos, propiedad y compatibilidad entre límites |
| `capability.api_contracts` | Cambia una API, protocolo, esquema, CLI o integración | Verifica productores, consumidores y semántica de errores |
| `capability.concurrency` | Se tocan tareas, colas, locks, reintentos o cancelación | Invariantes de orden y contabilidad con sincronización determinista |
| `capability.resource_lifecycle` | Se adquieren procesos, archivos, watchers u otros recursos | Limpieza idempotente, inicio parcial, parada repetida y recuperación |
| `capability.error_recovery` | Un flujo puede fallar parcialmente o reintentar operaciones | Clasificación, causa original, reintentos acotados y estado recuperable |
| `capability.test_quality` | Las pruebas son frágiles, flaky o muy simuladas | Contratos observables, no determinismo controlado y lógica real |
| `capability.algorithmic_correctness` | Hay algoritmos no triviales o sensibles a escala | Invariantes, terminación, complejidad y entradas adversariales |
| `capability.observability` | Cambia una ruta usada para diagnóstico operativo | Eventos distinguibles, correlación, contexto de fallo y datos seguros |
| `capability.release_readiness` | Se prepara un release, despliegue o handoff productivo | Revisión, artefactos, migraciones, rollback y verificación de la superficie final |
| `capability.dependency_change` | Se añade, elimina, actualiza o reemplaza una dependencia | Necesidad, compatibilidad, lockfile y rutas que cargan la dependencia |
| `capability.configuration_change` | Cambian configuración, defaults, flags o variables de entorno | Precedencia, validación, secretos y valores ausentes o heredados |
| `capability.cross_platform` | La conducta depende de OS, shell, filesystem, terminal, encoding o arquitectura | APIs portables, ramas aisladas y cobertura de plataformas declarada con precisión |
| `capability.cli_ux` | Cambia un comando, opción, interacción terminal, exit status o salida para scripts | Uso no interactivo, errores accionables, cancelación y semántica estable de salida |
| `capability.ui_state_completeness` | Cambia una pantalla o componente interactivo | Estados loading, vacío, error, disabled, overflow, layout estrecho y recuperación |
| `capability.browser_verification` | La conducta web depende de render, navegación, storage, consola o red | Navegador real cuando esté disponible y límites explícitos si no lo está |
| `capability.localization` | Cambian traducciones, formatos por locale, mensajes o encoding | Texto separable, placeholders, fallback, expansión y locales efectivamente comprobados |
| `capability.handoff` | El trabajo continuará entre sesiones, agentes o mantenedores | Estado, decisiones, evidencia, riesgos y siguiente verificación sin mezclar supuestos |
| `capability.context_management` | Una tarea larga puede perder restricciones o acumular contexto irrelevante | Requisitos durables, evidencia resumida y relectura antes de cambios importantes |
| `capability.data_integrity` | Un cambio puede perder, duplicar, reordenar o corromper datos | Atomicidad, idempotencia, orden, interrupción y lectura durable |
| `capability.privacy` | Se manejan datos personales, sensibles, retenidos o transmitidos | Minimización, exposición trazada, redacción y autoridad para transmitir |
| `capability.incident_response` | Hay una caída activa, regresión severa o incidente productivo | Contención autorizada, evidencia, impacto, recuperación y seguimiento causal |
| `capability.authority_boundaries` | El pedido limita alcance o efectos laterales | Autoridad explícita antes de escribir, delegar o actuar externamente |
| `capability.verification_integrity` | El resultado afirma que algo funciona | Cada afirmación se liga a un check observado y sus límites |
| `capability.change_preservation` | Una actualización acotada toca un artefacto existente | Lectura previa, transformación mínima y revisión del diff |
| `capability.progress_discipline` | Hay reintentos o continuación autónoma | Sólo repetir con evidencia nueva y detener bucles sin progreso |
| `capability.research_evidence` | Se piden hechos actuales, comparaciones o investigación externa | Fuentes primarias, citas por afirmación, inferencias separadas y lagunas explícitas |
| `capability.refactoring_discipline` | Se reestructura código conservando comportamiento | Contrato previo, movimientos incrementales y checks por límite afectado |
| `capability.dead_code_cleanup` | Se retiran código, flags, dependencias o scaffolding obsoletos | Entradas dinámicas y empaquetadas comprobadas antes de eliminar |
| `capability.git_hygiene` | Hay commits, rebases, conflictos, branches o preparación de PR | Diff exacto, historial protegido y cambios ajenos o binarios preservados |
| `capability.cost_awareness` | Una operación puede tener costo monetario o computacional material | Estimación, operación representativa acotada y aprobación de gasto |
| `capability.mentoring` | El usuario pide aprender, no sólo recibir el resultado | Nivel adaptado, conceptos graduales y pistas progresivas sin autoridad implícita |
| `capability.root_cause_analysis` | Un fallo se repite, cruza componentes o sólo tiene explicación sintomática | Cadena causal comprobada y reparación de la primera causa demostrada |
| `capability.ai_system_evaluation` | Se evalúan prompts, agentes, retrieval, routing o uso de tools | Casos y fallos definidos, condiciones iguales y harness separado del modelo |

Para activar sólo depuración, edite `90-optional-capabilities.json`:

```json
{
  "develop": {
    "capability.debugging": {
      "enabled_by_default": false,
      "enabled": true
    }
  }
}
```

Cada cuerpo conserva una condición `<if reason="…">`: habilitar una capacidad ofrece el
método, pero no inventa que la condición se cumpla, amplía el pedido ni concede permiso. Esta
colección adapta la taxonomía y los patrones descritos por Anthropic en
[Building effective agents](https://www.anthropic.com/research/building-effective-agents),
la guía de [prompt engineering de OpenAI](https://platform.openai.com/docs/guides/prompt-engineering)
y las categorías de tareas reproducibles de [SWE-bench](https://www.swebench.com/), contrastados
con los estudios internos de comprensión de prompts bajo `docs/*_PROMPT_COMPREHENSION_MENTAL_MAP.md`.

## Materialización por ejecución

Al comenzar una tarea, el pipeline lee y valida el catálogo del usuario y el override
del proyecto exactamente una vez, y construye un snapshot efectivo inmutable. La misma
instancia se comparte con chat, council, planners, gather, developer, review y los ciclos
de corrección o reentrada de esa tarea. Ningún compositor vuelve a consultar archivos
durante el recorrido.

Editar un JSON mientras una tarea está en marcha no modifica sus prompts: el cambio se
aplica al comienzo de la siguiente tarea. Cuando uno de esos subsistemas se invoca de
forma independiente —fuera del pipeline— compila igualmente un sólo snapshot para esa
invocación y lo reutiliza hasta que termina.

La precedencia se resuelve al compilar el snapshot, no al componer cada fragmento:
configuración general, después proveedor y finalmente modelo exacto. Dentro del snapshot
resultante, el valor más específico sustituye al menos específico para el mismo
identificador; los parámetros de varias capas no se fusionan.

## Relación con los estilos de prompt

El ajuste `PROMPT_STYLE` sigue seleccionando la variante incorporada (`full`,
`generalized`, `coding` o `extra_simple`; `auto` resuelve a `generalized`). El perfil
se aplica después de escoger esa variante. Por tanto un mismo perfil JSON puede
compartirse entre estilos, y deshabilitar un identificador tiene el mismo efecto para
la variante que esté activa.
