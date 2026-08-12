# Mini-model dataset and evaluation playbook

Este documento define el procedimiento para ampliar clasificadores de tasks,
reasoning, mensajes o trayectorias sin convertir ejemplos observados en una
prueba circular.

La taxonomía previa a la autoría manual de requests de Task Policy está en
[`TASK_POLICY_MANUAL_DATASET_DESIGN.md`](TASK_POLICY_MANUAL_DATASET_DESIGN.md).
Ese corpus prohíbe generadores de templates: los scripts pueden validar y
auditar filas, pero no escribir su contenido.

## Principio central

El objetivo no es maximizar accuracy sobre frases sintéticas. El objetivo es
seleccionar una intervención que mejore trabajo real sin ampliar autoridad,
romper scope ni interrumpir progreso sano.

Por eso hay dos productos distintos:

1. un clasificador con abstención calibrada;
2. un prompt/intervención que gana un E2E por modelo.

Un buen clasificador no justifica desplegar un prompt perjudicial. Un prompt
ganador tampoco compensa falsas activaciones frecuentes.

## Unidad de datos

### Task method

```json
{
  "id": "project-family--case",
  "text": "request natural completo",
  "operations": ["bugfix"],
  "constraints": ["preserve_public_api"],
  "authority": ["answer", "diagnose", "modify"],
  "policy": "bugfix.root_cause",
  "uncategorized_reason": null,
  "language": "es",
  "project_family": "queue-worker",
  "programming_languages": ["Go"],
  "split": "validation",
  "source": "reviewed_natural_request",
  "review_status": "approved",
  "reviewer": "...",
  "rationale": "restaura un contrato existente; no crea capacidad"
}
```

### Reasoning o mensaje

```json
{
  "id": "trajectory-family--window",
  "request_summary": "...",
  "task_profile": ["bugfix"],
  "visible_text": "summary/reasoning expuesto y sanitizado",
  "events_before": ["read", "read", "test_failure"],
  "features": {
    "same_step_windows": 3,
    "net_diff_changed": false,
    "new_test_fingerprint": false,
    "equivalent_tool_retry": true
  },
  "label": "retry_loop",
  "expected_intervention": "change strategy",
  "provider": "minimax",
  "model": "MiniMax-M3",
  "split": "holdout",
  "review_status": "approved"
}
```

Nunca se almacenan firmas opacas, razonamiento cifrado, secrets, paths
personales o datos no necesarios para la etiqueta.

## Qué categorías admitir

Una categoría nueva debe cumplir las cuatro condiciones:

1. tiene una definición observable y distinguible;
2. necesita un método o intervención diferente;
3. existe evidencia suficiente para positivos y negativos;
4. puede evaluarse con un outcome E2E.

No crear categorías solo porque aparecen palabras diferentes. Por ejemplo,
`stuck`, `slow_progress` y `too_much_reading` podrían ser el mismo
`excessive_exploration` si activan exactamente la misma corrección.

## Fuentes de ejemplos

Orden de valor:

1. requests o trayectorias naturales con opt-in y sanitización;
2. issues y tareas open source reformulados, sin copiar texto protegido;
3. errores reales revisados de E2E y shadow mode;
4. ejemplos sintéticos para rellenar combinaciones faltantes;
5. perturbaciones controladas de ejemplos aprobados.

Los sintéticos sirven para cobertura lingüística y tests de contrato, pero no
deben dominar el holdout. Agregar muchas reformulaciones de la misma plantilla
infla métricas sin añadir diversidad.

## Positivos

Para cada clase incluir:

- distintas formulaciones, longitudes y registros;
- varios idiomas y code switching;
- múltiples lenguajes de programación y dominios de proyecto;
- requests directos e implícitos;
- requests con constraints adicionales;
- tareas compuestas y secuencia explícita;
- errores con síntomas distintos, no solo nombres de categoría.

Un dataset de bugfix necesita regresiones, crashes, stale state, off-by-one,
duplicados, violaciones de protocolo, compatibilidad y concurrencia. Repetir
“fix the bug” con veinte objetos no cubre esas familias.

El retriever de producción mantiene un piso de 20 prototipos positivos y 20
negativos difíciles por categoría operativa. Es un gate de cobertura mínima,
no un dataset suficiente ni una licencia para clonar plantillas. Los prototipos
viven separados del registro de prompts, tienen versión propia y deben crecer
con familias causales distintas.

## Negativos difíciles

Los negativos deben compartir vocabulario con los positivos:

| Positivo | Negativo difícil |
| --- | --- |
| corregir una regresión | explicar un issue titulado “fix regression” |
| restaurar contrato | feature que cambia deliberadamente el contrato |
| investigar y corregir | diagnóstico read-only sin autorización de editar |
| optimizar latencia | bug de timeout que devuelve datos incorrectos |
| review | request que pide review y luego implementación explícita |
| verification gap | modelo todavía implementando, sin intentar cerrar |
| retry loop | segundo intento con hipótesis o argumentos materialmente distintos |

También incluir negación, citas, logs, ejemplos, condicionales futuros y
referentes ambiguos.

## `uncategorized`

No usar una sola plantilla neutral. Mantener razones explícitas:

- `conversation`;
- `acknowledgement`;
- `quoted_action`;
- `conceptual_question`;
- `read_only_status`;
- `out_of_domain`;
- `ambiguous_method`;
- `healthy_progress`;
- `insufficient_observable_evidence`.

Reportar métricas por razón. Una accuracy global puede ocultar que el modelo
activa políticas sobre texto citado o conversación.

## Splits y leakage

La unidad de split es la familia causal, no la fila. Deben permanecer juntos:

- mismo proyecto o fixture;
- mismo issue o bug raíz;
- variantes lingüísticas;
- perturbaciones de wording;
- baseline/candidate de una trayectoria;
- ventanas vecinas del mismo run.

Usar como mínimo:

- `calibration`: fit;
- `validation`: thresholds, margins y elección de arquitectura;
- `holdout`: una sola evaluación final;
- `challenge`: negativos adversariales renovables que no autorizan tuning del
  holdout.

Una vez observado un holdout para corregir el modelo, pasa a development. Se
debe congelar una familia nueva antes de volver a afirmar generalización.

## Entrenamiento

Orden recomendado:

1. recuperador contrastivo sin head;
2. head lineal sobre embedding congelado;
3. head lineal + features observables;
4. thresholds y margins por clase;
5. MLP de 16-32 unidades si gana un held-out natural;
6. modelo secuencial solo si el orden de eventos demuestra valor.

No introducir sparse attention o un tiny transformer por tamaño del contexto
si pooling/top-k ya cabe en memoria. Con 32 embeddings `float16` de 1024
dimensiones, el buffer ronda 64 KB.

## Calibración y abstención

Medir por clase:

- precision, recall y F1;
- coverage y abstention rate;
- selective precision;
- runner-up margin;
- falsos positivos por tipo de negativo;
- calibración por provider/model;
- estabilidad ante pequeñas perturbaciones.

Threshold global y margin global son un baseline, no una obligación. Si bugfix
y refactor necesitan separaciones distintas, usar parámetros por clase siempre
que validation lo justifique. No bajar un margen porque un ejemplo observado
quedó cerca: eso es tuning leakage.

Las señales literales pueden desempatar una candidata coincidente porque no
agregan una categoría nueva. No deben rescatar una candidata semántica
conflictiva ni conceder autoridad.

## Evaluación del router

Evaluar por separado:

1. operaciones detectadas;
2. autoridad literal;
3. policy seleccionada;
4. motivo de rechazo;
5. abstención y primera candidata;
6. fragmentos que realmente se renderizaron.

Esto evita confundir:

- “el mini-modelo no detectó bugfix”;
- “detectó bugfix pero faltó autoridad literal”;
- “la policy fue seleccionada pero rollout la bloqueó”;
- “el prompt se inyectó y perjudicó al modelo principal”.

## Diseño del fragmento

Un buen candidato expresa:

- outcome específico;
- scope que debe preservarse;
- evidencia mínima de éxito.

Evitar duplicar identidad, seguridad, protocolo del loop o instrucciones
genéricas. Para modelos que ya expanden tareas, evitar microprocedimientos como
“haz exactamente una lectura”, “prueba una vez” o “termina inmediatamente”.

Composición permitida:

```text
fragmento genérico de bugfix
  + ajuste MiniMax opcional
  + ajuste GPT-5.6 opcional
```

Cada ajuste debe tener ID y versión propios. Un ajuste específico puede
excluirse de una ruta o limitarse a una familia. Si pierde, se elimina aunque
parezca coherente con un mapa mental.

## E2E prompt-only

Baseline y candidate deben compartir:

- request y user prompt exactos;
- identidad estable;
- tools y schemas;
- workspace y verifier;
- modelo, provider, versión y reasoning effort;
- permisos y budgets;
- dataset y condition manifest.

La única diferencia debe ser el fragmento esperado. Registrar en la primera
iteración:

- `stable_system_chars`;
- `dynamic_system_chars`;
- `user_chars`;
- `tool_schema_chars`;
- IDs, versiones y hashes de fragmentos.

Gates mínimos:

- cero regresiones de success, verifier, scope o autoridad;
- baseline sin fragmentos;
- fragmentos candidate exactos;
- ninguna regresión agregada de tokens o tools;
- latencia dentro del límite;
- al menos dos métricas con mejora material;
- revisión de tool failures y claims finales.

Una repetición puede rechazar una regresión grande, pero una mejora pequeña
necesita repeticiones. Reportar siempre varianza y tamaño de campaña.

## Matriz por modelo

No heredar resultados por nombre de familia:

| Ruta | Fragmento genérico | Ajuste específico | Decisión |
| --- | --- | --- | --- |
| MiniMax M3 | medir | medir separado | rollout exacto |
| GPT-5.6 Sol | medir | opcional | rollout exacto |
| GPT-5.6 Terra | medir | opcional | rollout exacto |
| GPT-5.6 Luna | medir | opcional | rollout exacto |

Si `bugfix.developer@3` gana Terra, Sol y Luna permanecen fail-closed hasta su
propia evidencia.

## Shadow y aprendizaje activo

Recolectar prioritariamente:

- head/retriever/literal en desacuerdo;
- primera candidata correcta con margen bajo;
- falsas activaciones sobre `uncategorized`;
- intervenciones que no producen transición útil;
- runs caros con success verde;
- tool-call schema errors;
- muestras aleatorias de runs sanos.

No reentrenar automáticamente en producción. Las muestras pasan por
sanitización, revisión, versionado y una campaña offline.

## Checklist de expansión

- [ ] Definición y outcome distintos.
- [ ] Positivos naturales de varias familias.
- [ ] Negativos difíciles con vocabulario compartido.
- [ ] `uncategorized` por razón.
- [ ] Splits sin leakage de proyecto/trayectoria.
- [ ] Artefacto versionado con hashes y `space_id`.
- [ ] Métricas por clase y abstención.
- [ ] Cero autoridad semántica.
- [ ] Fragmento corto, versionado y auditable.
- [ ] E2E prompt-only por ruta.
- [ ] Rollout exacto o shadow; nunca herencia implícita.
- [ ] Documentación de fallos y límites.
