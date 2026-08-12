# Mini-model architecture for conditional prompting

Estado: arquitectura de referencia para extender el sistema. Resume las piezas
activas y los contratos que deben conservar futuras categorías, heads y
fragmentos.

Documentos relacionados:

- [`CONDITIONAL_TASK_POLICIES.md`](CONDITIONAL_TASK_POLICIES.md): clasificación
  inicial del request y prompts por tipo de tarea;
- [`CONDITIONAL_REASONING_PROMPTS.md`](CONDITIONAL_REASONING_PROMPTS.md):
  clasificación del reasoning que la API expone;
- [`ADAPTIVE_RUNTIME_POLICIES.md`](ADAPTIVE_RUNTIME_POLICIES.md): observables,
  intervenciones temporales y memoria de trayectoria;
- [`MINI_MODEL_DATASET_PLAYBOOK.md`](MINI_MODEL_DATASET_PLAYBOOK.md): cómo agregar
  datos, categorías y evaluaciones sin contaminar los splits.

## Objetivo

El sistema usa clasificadores locales muy pequeños para decidir qué
instrucción corta puede ayudar en un momento concreto. No intenta reemplazar al
modelo principal, leer chain-of-thought privado ni construir un system prompt
completamente distinto para cada request.

Las decisiones que queremos adaptar son distintas:

1. **Qué tarea pidió el usuario:** bugfix, feature, refactor, research, review,
   performance u otra operación.
2. **Qué restricciones aplican:** preservar comportamiento o API, solo lectura,
   seguridad, migración y scope.
3. **Qué patrón muestra la trayectoria:** progreso sano, exploración excesiva,
   retry loop, verificación faltante o cierre prematuro.
4. **Qué intervención funciona para la ruta de modelo:** un prompt genérico,
   un ajuste adicional para una familia/modelo o ninguna intervención.

## Vista completa

```text
request literal del usuario
  |-- parser determinista: autoridad, negación, acciones externas
  |-- embedding ken/static-qwen3-r512-v2
  |     -> gate discursivo: task vs uncategorized
  |     -> mini-head de método + abstención estricta
  |     -> candidata de menor confianza, solo utilizable con acuerdo independiente
  |     -> recuperador contrastivo independiente
  `-- resolución de conflictos
        -> TaskProfile estable
        -> fragmento genérico por tarea
        -> ajuste opcional por familia/modelo

respuesta del proveedor + eventos del loop
  |-- reasoning o summary explícitamente expuesto
  |-- mensajes visibles y tool calls normalizadas
  |-- diff, tests, errores, scope, tiempo y progreso
  `-- embedding/pooling + features observables
        -> mini-head de patrón de trayectoria
        -> veto observable
        -> intervención temporal para el siguiente turno

cada fragmento/intervención
  -> telemetría con ID, versión, hash y evidencia
  -> E2E baseline/candidate en una ruta inmutable
  -> rollout fail-closed por fragmento + versión + modelo
```

## Separación de responsabilidades

### Autoridad

La autoridad nunca proviene del embedding ni del mini-modelo. Solo el request
literal puede autorizar:

- modificar archivos;
- commit o push;
- publicación o deploy;
- acciones destructivas o escrituras externas.

Una predicción `bugfix` sin verbo operativo literal puede proponer un método,
pero no activar un fragmento que requiera escritura. Negación, texto citado y
modo read-only tienen prioridad sobre similitud semántica.

### Intención y método

El mini-modelo clasifica el método de trabajo. Puede distinguir una capacidad
nueva de la restauración de un contrato, o una revisión read-only de una
corrección. Cuando el request usa una categoría explícita, esa señal también
puede desempatar la primera candidata del head sin añadir una operación nueva.

Para paráfrasis sin señal literal se exige:

```text
mini-head por encima de threshold y margen
  + recuperador contrastivo que elige la misma política
  + autoridad literal compatible
  = política seleccionable
```

Una abstención del head es final; el retriever no puede forzar una categoría.

### Estado de ejecución

El patrón de trayectoria no debe confundirse con el tipo de tarea. Un bugfix
puede progresar sanamente o caer en exploración excesiva; un refactor puede
tener un verification gap. Por eso `TaskProfile` dura toda la tarea y el
`BehaviorProfile` expira después de una ventana o cuando aparece progreso.

## Familia de mini-modelos

Todos pueden reutilizar el encoder congelado
`ken/static-qwen3-r512-v2`, pero son artefactos separados:

| Head | Entrada | Salida | Estado |
| --- | --- | --- | --- |
| acto discursivo | embedding del request | task o uncategorized | activo |
| método de tarea | embedding aceptado | política, candidata o abstención | activo |
| patrón de reasoning | ventanas expuestas + observables | patrón o abstención | activo |
| estado observable | features de loop + embedding opcional | patrón temporal | evaluado |
| mensaje visible | mensaje + contexto del Step | intención comunicativa | futuro |
| trayectoria | pooling de eventos y features | intervención útil | futuro |

No hace falta una única red que resuelva todo. Heads pequeños y especializados
permiten datasets, thresholds, versiones y rollouts diferentes. Si se comparte
un encoder, cada artefacto debe verificar exactamente su `space_id`, dimensión
y hash de corpus.

## Fuentes de entrada

### Request

Se normaliza el texto del usuario, excluyendo texto citado cuando corresponde.
El embedding completo evita depender de regexp para paráfrasis, pero las reglas
literales conservan autoridad, negación y operaciones inequívocas.

### Reasoning expuesto

Solo se procesa texto que el proveedor devuelve de forma visible mediante
`reasoning_content`, summaries, thinking blocks no opacos o equivalentes. No se
intentan decodificar firmas, contenido cifrado ni `redacted_thinking`.

El texto se divide en ventanas acotadas y se combina con evidencia del loop.
Un patrón semántico por sí solo no debe interrumpir una trayectoria saludable.

### Mensajes y herramientas

Los mensajes assistant visibles, planes, summaries y llamadas de herramientas
pueden convertirse en eventos normalizados. Son especialmente útiles para
detectar:

- anuncios de finalización sin evidencia;
- repetición de la misma hipótesis;
- preguntas al usuario cuando existe una decisión reversible segura;
- tool calls equivalentes con argumentos apenas reformateados;
- afirmaciones que exceden el output observado.

### Observables

Siempre que exista un observable más fiable que lenguaje, tiene prioridad:

- fingerprint del diff;
- test ejecutado y fingerprint del workspace que verificó;
- error normalizado;
- archivos leídos o modificados;
- Step actual y transiciones;
- tiempo, iteraciones y tool calls desde el último progreso;
- cambios fuera de scope.

La campaña natural mostró que features observables pueden superar al embedding
para estados como progreso sano o exploración excesiva.

## Compositor de prompts

Una política no tiene que elegir entre un prompt universal y uno específico.
La composición es aditiva:

```text
núcleo estable global
  + fragmento genérico de tarea
  + ajuste opcional de familia/modelo
  + intervención temporal de trayectoria
```

Cada `ConditionalPromptFragment` puede declarar:

- rol y fase;
- operaciones, constraints y autoridad requeridas;
- operaciones o constraints excluidas;
- rutas de modelo incluidas;
- rutas de modelo excluidas;
- prioridad, versión y presupuesto UTF-8.

Un selector `openai_subscription:gpt-5.6` incluye Sol, Terra y Luna. Un selector
`openai_subscription:gpt-5.6-terra` afecta solo Terra. Los fragmentos sin
selector son genéricos. Si falta una ruta para un fragmento específico, la
selección falla cerrada.

Los ajustes específicos no reemplazan silenciosamente al fragmento genérico.
Ambos aparecen con IDs y hashes distintos en telemetría y deben ganar sus
propios E2E. También es válido que una ruta use solo el genérico o ninguno.

## Adaptación por modelo

Los mapas mentales de MiniMax y GPT-5.6 describen tendencias observables, no
reglas infalibles. Sirven para proponer candidatos, nunca para aprobarlos.

Ejemplos actuales:

- MiniMax suele expandir requisitos, riesgos y verificación; un prompt más
  descriptivo puede aumentar exploración.
- GPT-5.6 Sol construye contratos detallados y suele beneficiarse de objetivos
  compactos, límites y evidencia en lugar de microprocedimientos.
- Terra y Luna pueden responder de forma distinta aun compartiendo familia;
  una victoria en Terra no autoriza automáticamente Sol o Luna.

La campaña bugfix confirmó este principio: el núcleo genérico mejoró Terra,
quedó mixto en MiniMax y un ajuste MiniMax intuitivamente razonable empeoró
todas las métricas. El ajuste se eliminó.

## Abstención

`uncategorized` no es un error; es una salida necesaria. Debe cubrir:

- conversación y acknowledgements;
- preguntas conceptuales sin autorización de actuar;
- texto de acciones dentro de citas o logs;
- status y seguimiento;
- contenido fuera de dominio;
- mezclas ambiguas o scores sin margen;
- trayectorias saludables que no necesitan corrección.

Para categorías restrictivas importa más evitar falsos positivos que maximizar
coverage. Una política correcta aplicada en el momento incorrecto puede ser
peor que no inyectar nada.

## Artefactos y contratos

Cada head empaquetado debe contener como mínimo:

```json
{
  "schema_version": 1,
  "model": "nombre-versionado",
  "embedding_space_id": "ken/static-qwen3-r512-v2:1024:hash",
  "labels": ["...", "uncategorized"],
  "features": ["..."],
  "parameters": {"thresholds": {}, "margins": {}},
  "calibration_sha256": "...",
  "validation_sha256": "...",
  "holdout_sha256": "..."
}
```

El loader falla cerrado ante un label, shape, feature, schema o espacio de
embedding distinto. Cambiar dataset, output labels o regla de decisión requiere
una nueva versión del artefacto.

## Telemetría mínima

Cada decisión debe poder reconstruirse sin guardar el prompt completo:

- head y versión;
- `space_id`;
- score, threshold y runner-up margin;
- primera candidata incluso si hubo abstención;
- evidencia literal, semántica, contrastiva u observable;
- políticas rechazadas y motivo;
- fragmentos renderizados con ID, versión y hash;
- provider, model y model identity;
- intervención, cooldown y transición posterior.

La primera candidata es importante: permite distinguir “el head eligió otra
clase” de “eligió la correcta pero abstuvo por margen”.

## Estado actual

| Capacidad | Estado de producción |
| --- | --- |
| clasificación de task method | head jerárquico v3 activo, con gate discursivo y doble acuerdo |
| autoridad/negación | literal y fail-closed |
| prompt de refactor | aprobado solo para MiniMax M3 |
| prompt de bugfix v3 | aprobado solo para GPT-5.6 Terra |
| reasoning-pattern head | activo con veto observable |
| ajustes específicos por modelo | soportados; ninguno se conserva sin E2E ganador |
| clasificación general de mensajes | diseño futuro |
| memoria jerárquica de trayectoria | diseño/evaluación incremental |

El retriever contrastivo usa un registro separado y versionado con un piso de
20 prototipos positivos y 20 negativos difíciles por cada una de las seis
categorías operativas. Esos textos se embeben localmente; no se inyectan al
modelo principal. Los contratos y contenidos de prompt permanecen aislados en
`registry.py`.

## Qué falta

Prioridad inmediata:

1. congelar un holdout natural nuevo por familia causal; los conjuntos usados
   para desarrollar v3 ya no prueban generalización futura;
2. separar método primario de calificadores combinables como seguridad,
   migración, documentación, tests y configuración;
3. clasificar inputs largos por ventanas y pooling para que logs, código citado
   o contexto histórico no diluyan la petición activa;
4. calibrar thresholds por clase e idioma, preservando un gate común de falsa
   activación cero;
5. mover los embeddings de los prototipos a un artifact precomputado y validado
   por hash para evitar reconstruir el índice en cada proceso;
6. crear una cola de aprendizaje activo con desacuerdos, abstenciones costosas,
   falsos positivos y muestras sanas aleatorias.

Siguiente nivel:

- un head multi-label de calificadores, independiente del método;
- un head de acto comunicativo para mensajes visibles: progreso, cierre,
  pregunta evitable, claim sin evidencia y pedido legítimo de información;
- un head temporal que clasifique ventanas de trayectoria, no frases aisladas;
- selección de intervención por utilidad observada: detectar un patrón no basta,
  también hay que demostrar que el prompt correctivo mejora el siguiente estado;
- detección open-set/OOD para idiomas, dominios o combinaciones no cubiertas;
- calibración y rollout exactos por proveedor/modelo, sin herencia implícita.

## Cómo expandir esta arquitectura

Antes de añadir una categoría o ajuste:

1. demostrar que representa una decisión/intervención diferente;
2. reunir ejemplos naturales, negativos difíciles y `uncategorized`;
3. congelar familias completas en calibration, validation y holdout;
4. evaluar head, abstención y falsos positivos offline;
5. escribir un fragmento corto y versionado;
6. ejecutar baseline/candidate prompt-only por ruta de modelo;
7. aprobar únicamente el fragmento, versión y ruta que ganaron;
8. observar en shadow antes de ampliar el rollout.

Los detalles operativos están en
[`MINI_MODEL_DATASET_PLAYBOOK.md`](MINI_MODEL_DATASET_PLAYBOOK.md).
