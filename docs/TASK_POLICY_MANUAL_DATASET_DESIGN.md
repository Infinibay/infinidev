# Diseño del corpus manual de Task Policies

Estado: taxonomía y protocolo de autoría aprobables; ejemplos todavía `draft`.

Este documento precede deliberadamente a la escritura del nuevo corpus. Su
objetivo es evitar que la cantidad de ejemplos o la comodidad de una plantilla
decidan qué diversidad existe. Cada request se redactará y justificará de forma
individual. No se usarán expansores de templates, traducción automática en
serie ni scripts que fabriquen filas.

## Qué debe aprender el modelo

El mini-modelo propone métodos de trabajo, no permisos ni una identidad del
usuario. La salida es multi-label y puede contener:

- `bugfix.root_cause`;
- `feature.contract_first`;
- `refactor.preserve_behavior`;
- `research.evidence_first`;
- `review.read_only`;
- `performance.measure_first`;
- ninguna activación, registrada con una razón `uncategorized`.

Autoridad, negaciones, commit, push, publicación y escrituras externas siguen
siendo deterministas. Las dimensiones que siguen sirven para diversificar el
lenguaje y medir sesgos; no son labels que el modelo deba adivinar.

## Tipos de usuario

Un mismo método se expresa de forma muy diferente según experiencia, rol y
relación con el proyecto. El corpus cubrirá estos arquetipos sin inferirlos en
producción:

| ID | Arquetipo | Rasgos que deben aparecer en el lenguaje |
| --- | --- | --- |
| `solo_maintainer` | mantenedor individual | contexto implícito, continuidad, preocupación por regresiones |
| `oss_contributor` | contribuidor nuevo | referencia a issue/PR, alcance pequeño, dudas sobre convenciones |
| `junior_developer` | desarrollador junior | descripción por síntomas, términos imprecisos, pedidos de explicación |
| `senior_developer` | desarrollador senior | invariantes, límites de módulos, compatibilidad y tradeoffs explícitos |
| `staff_architect` | staff/arquitecto | efectos entre subsistemas, migración, contratos y rollout |
| `product_engineer` | ingeniería orientada a producto | flujo de usuario, aceptación, UI/API observable |
| `qa_engineer` | QA/test | reproducción, fixtures, matrices y regresiones verificables |
| `sre_oncall` | SRE/on-call | incidente, síntomas operativos, mitigación y evidencia temporal |
| `security_reviewer` | seguridad/auditoría | trust boundaries, explotación demostrable, pedido frecuentemente read-only |
| `performance_engineer` | rendimiento | workload, baseline, percentiles, CPU/memoria y reproducibilidad |
| `release_engineer` | release/build | compatibilidad, packaging, tags, CI y rollback |
| `data_ml_engineer` | datos/ML | pipelines, schemas, reproducibilidad, drift y costo de inferencia |
| `technical_writer` | documentación | explicación, ejemplos, audiencia; a menudo `uncategorized` para estos métodos |
| `researcher_evaluator` | investigación/evaluación | fuentes, comparación, incertidumbre y diseño experimental |
| `nontechnical_stakeholder` | usuario no técnico | outcome y síntomas de negocio, poca terminología de implementación |

No se clonará una misma petición cambiando solamente el rol. Cada arquetipo
debe aportar una situación y una forma de razonar propias.

## Tipos de proyecto

Las familias de proyecto se asignan completas a un split. Los nombres concretos
son ficticios; un proyecto open source puede inspirar el dominio, pero no se
copian issues ni texto.

| ID | Dominio | Tecnologías representativas | Fallos y cambios naturales |
| --- | --- | --- | --- |
| `cli_devtool` | CLI/herramienta de desarrollo | Python, Rust, Go | parsing, exit codes, config, terminal |
| `web_api` | API web | Python, Go, Java, C# | contratos HTTP, auth, serialización, retries |
| `web_frontend` | frontend web | TypeScript, JavaScript | estado, accesibilidad, rendering, bundles |
| `mobile_client` | aplicación móvil | Kotlin, Swift, Dart | lifecycle, offline, permisos, sincronización |
| `database_storage` | base de datos/storage | Rust, C++, Java | transacciones, índices, snapshots, recovery |
| `queue_stream` | colas y streaming | Go, Java, Elixir | ordering, ack, backpressure, delivery |
| `compiler_language` | compilador/parser/LSP | Rust, C++, TypeScript | AST, diagnóstico, incrementalidad, ABI |
| `sdk_protocol` | SDK/cliente de protocolo | Python, Java, Go | compatibilidad, paginación, wire format |
| `build_monorepo` | build/package/monorepo | Starlark, TypeScript, Shell | caché, resolución, reproducibilidad, releases |
| `infra_observability` | infraestructura/observabilidad | Go, Terraform, YAML | rollout, métricas, cardinalidad, alertas |
| `embedded_firmware` | firmware/embedded | C, C++, Rust | memoria, timing, interrupciones, hardware |
| `data_pipeline` | ETL/analytics | Python, SQL, Scala | schemas, particiones, idempotencia, drift |
| `ml_inference` | serving/inferencia ML | Python, C++, CUDA | batching, memoria, latencia, determinismo |
| `security_identity` | auth/crypto/identity | Rust, Go, Java | tokens, permisos, validación, rotación |
| `game_mod_plugin` | juego/mod/plugin | C#, Java, Lua | eventos, saves, compatibilidad, tick loop |
| `desktop_editor` | editor/desktop | Rust, C++, TypeScript | plugins, documentos, IPC, undo/redo |
| `scientific_library` | cálculo científico | Python, C++, Fortran | precisión, estabilidad numérica, vectorización |
| `legacy_business` | sistema empresarial legado | Java, C#, COBOL/SQL | compatibilidad, migraciones, reglas ocultas |

La evaluación reportará también runtime, framework y lenguaje de programación,
pero no confundirá esos atributos con el método de trabajo.

## Familias lingüísticas

La cobertura no se limita al código ISO del idioma. Cada fila declara idioma,
variante y fenómeno lingüístico.

### Idiomas y variantes

- inglés internacional, estadounidense y británico;
- español rioplatense, latinoamericano general y peninsular;
- portugués brasileño y europeo;
- francés;
- alemán;
- italiano.

El primer release prioriza inglés, español y portugués. Francés, alemán e
italiano funcionan como familias de transferencia y challenge; no se afirmará
paridad mientras su cobertura sea menor.

### Fenómenos

- terminología técnica nativa;
- préstamos en inglés: “hacer rollback”, “pushear”, “cachear”;
- code switching dentro de una oración;
- sujeto omitido y referencias pronominales;
- elipsis dependiente del turno anterior;
- errores ortográficos y falta de tildes sin caricaturizar al usuario;
- puntuación mínima o mensajes fragmentados;
- negación de alcance;
- condicional e hipotético;
- ironía leve o frustración, sin usar sentimiento como label;
- citas de logs, tickets, documentación o terceros.

Una traducción de una fila sigue perteneciendo a la misma familia de escenario
y al mismo split. Preferiremos escribir un escenario distinto en cada idioma.

## Estilos de petición

Cada método debe aparecer en varios de estos estilos:

| ID | Estilo |
| --- | --- |
| `terse_imperative` | orden breve y directa |
| `conversational` | pedido natural con contexto informal |
| `issue_report` | expected/actual o reproducción tipo issue |
| `acceptance_criteria` | outcome con criterios verificables |
| `incident_report` | síntomas operativos y línea temporal |
| `evidence_first` | logs, mediciones o test fallido antes del pedido |
| `outcome_only` | describe resultado deseado sin nombrar el método |
| `question_directive` | pregunta que en realidad autoriza trabajo |
| `polite_indirect` | pedido indirecto o tentativo pero ejecutable |
| `fragmented_followup` | continuación corta dependiente del contexto |
| `compound_sequenced` | varias operaciones en orden explícito |
| `compound_unordered` | varios outcomes compatibles sin orden claro |
| `scope_limited` | rutas, componentes o límites explícitos |
| `constraint_heavy` | compatibilidad, API, dependencias o presupuesto |
| `negated_alternative` | autoriza una vía y prohíbe otra |
| `quoted_or_reported` | contiene una acción que no es una orden del usuario |
| `hypothetical` | escenario futuro sin autorización actual |
| `meta_classification` | pregunta qué tipo de tarea sería, no pide hacerla |

## Familias semánticas por método

Estas familias son strata de cobertura, no nuevas labels.

### Bugfix

- regresión respecto de una versión anterior;
- crash o excepción sobre input válido;
- output incorrecto o stale state;
- boundary/off-by-one/paginación;
- concurrencia, carrera o cancelación;
- violación de protocolo o serialización;
- compatibilidad hacia atrás;
- fuga de recursos con comportamiento incorrecto;
- corrupción o pérdida de datos;
- error de permisos respecto del contrato existente.

### Feature

- nuevo flujo de usuario;
- endpoint/comando/evento inexistente;
- nuevo formato o integración;
- configuración que antes no era expresable;
- modo offline/batch/streaming;
- nueva capacidad de UI o accesibilidad;
- soporte para plataforma o credencial nueva;
- extensión compatible de protocolo.

### Refactor

- separar responsabilidades;
- eliminar duplicación;
- aclarar state machine;
- invertir o aislar dependencias;
- reorganizar módulos privados;
- reemplazar representación interna;
- reducir complejidad preservando outputs;
- preparar un seam sin añadir todavía una feature.

### Research

- comparar tecnologías o diseños;
- interpretar un estándar/protocolo;
- investigar compatibilidad y mantenimiento;
- evaluar licencia, seguridad o adopción;
- reunir evidencia antes de una decisión;
- investigar una causa sin autorización para corregir;
- diseñar un experimento o benchmark.

### Review

- revisar diff/PR;
- auditoría read-only de seguridad;
- revisión de arquitectura;
- buscar regresiones o gaps de tests;
- evaluar compatibilidad;
- revisar rendimiento sin optimizar;
- validar claims contra código o especificación.

### Performance

- latencia p50/p95/p99;
- throughput;
- CPU o allocations;
- memoria pico/residente;
- tiempo de startup;
- tamaño de bundle/binario;
- consultas o I/O;
- rendimiento bajo concurrencia;
- costo de inferencia;
- energía/timing en embedded.

## Razones de `uncategorized`

`uncategorized` significa “no activar ninguno de estos métodos”, no “texto sin
sentido”. Debe ser la parte más diversa del corpus.

- `acknowledgement`: agradecimiento, aprobación o cierre;
- `status_only`: pide estado o evidencia ya obtenida;
- `conceptual_question`: explicación técnica sin trabajo sobre el repo;
- `explanation_only`: explicar código o comportamiento actual;
- `quoted_action`: acción dentro de una cita, log, ticket o traducción;
- `hypothetical_future`: cambio posible, explícitamente no actual;
- `meta_method`: pregunta cómo clasificar o abordar una tarea;
- `ambiguous_authority`: hay un problema, pero no queda claro si debe actuar;
- `out_of_domain`: conversación no relacionada con desarrollo;
- `unsupported_method`: docs, tests, migración o configuración cuando ninguna
  policy activa aporta un método específico;
- `ambiguous_method`: existe autoridad para actuar, pero el request no permite
  elegir de forma segura entre dos o más métodos incompatibles;
- `continuation_without_task`: “ok”, “y?”, “listo” sin nueva operación;
- `conflicting_request`: constraints incompatibles que requieren aclaración;
- `insufficient_context`: referente imposible de resolver incluso con contexto;
- `reported_third_party_request`: cuenta lo que otra persona pidió sin adoptarlo;
- `healthy_existing_plan`: confirma un plan ya activo sin cambiar su método.

## Autoridad y constraints

Aunque no se aprendan semánticamente, cada fila los anota para detectar
activaciones inseguras.

Autoridad observable:

- `answer`, `diagnose`, `modify`, `commit`, `push`, `publish`, `external_write`;
- `read_only` y negaciones explícitas se registran aparte;
- una petición puede autorizar `modify` sin autorizar `commit`.

Constraints representativos:

- preservar comportamiento, API pública, ABI, schema, datos o wire format;
- limitar archivos, módulo, dependencias o plataforma;
- mantener compatibilidad hacia atrás;
- no acceder a red o no escribir externamente;
- no refactorizar, no optimizar o no implementar todavía;
- conservar presupuesto de latencia/memoria;
- exigir rollback, migración gradual o reproducibilidad.

## Cardinalidad y combinaciones

La matriz debe contener salidas con cero, una, dos y ocasionalmente tres
políticas. Las combinaciones prioritarias son:

- bugfix + refactor;
- bugfix + research;
- bugfix + performance;
- feature + refactor;
- feature + research;
- feature + performance;
- research + review;
- review + performance cuando se pide auditar un artefacto y además reproducir
  o validar mediciones, sin optimizarlo;
- research + feature + performance en tareas realmente secuenciadas.

No toda coocurrencia verbal es multi-label. “Investiga el bug” puede ser un
bugfix normal; research aplica cuando se requiere evidencia externa, comparación
o una fase investigativa distinguible. “Código lento y equivocado” puede ser
bugfix, performance o ambos según los outcomes pedidos.

## Contrato semántico de las etiquetas

Las etiquetas se asignan por el método que el prompt condicional realmente debe
activar, no por palabras del ticket ni por la forma que tomó un patch upstream:

- `bugfix` restaura un comportamiento que contradice un contrato existente;
- `feature` crea o cambia una capacidad, API o conducta observable;
- `refactor` cambia estructura interna y conserva deliberadamente la conducta
  observable; deprecaciones, upgrades y migraciones no son refactor por defecto;
- `performance` exige medición representativa. Puede ser una optimización con
  autoridad de cambio o un análisis read-only que termina en un informe;
- `research` produce evidencia externa, comparación o un experimento como
  outcome independiente, no sólo el diagnóstico normal previo a un fix;
- `review` evalúa un artefacto concreto y entrega hallazgos sin modificarlo;
- una salida vacía es correcta cuando ninguna policy agrega un método útil.

Una salida multi-label sólo es válida si cada policy cambia de manera
independiente el workflow y sus restricciones son compatibles. `review` puede
combinarse con `research` o con medición de `performance`; no puede combinarse
con `bugfix`, `feature` o `refactor`, porque esas policies requieren cambios.

## Pares de contraste obligatorios

Cada ejemplo positivo debe tener al menos un negativo conceptual cercano en el
corpus, no necesariamente una reescritura literal:

| Frontera | Contraste que debe aprenderse |
| --- | --- |
| bugfix / feature | restaurar contrato existente frente a crear capacidad |
| bugfix / performance | resultado incorrecto frente a resultado correcto pero caro |
| bugfix / refactor | cambio observable frente a estructura interna únicamente |
| feature / refactor | nuevo outcome frente a preparar/reorganizar sin outcome nuevo |
| research / conceptual | decisión basada en evidencia frente a explicación general |
| review / bugfix | entregar hallazgos sin editar frente a corregirlos |
| performance / review | optimizar medición frente a informar problemas solamente |
| task / quoted action | orden actual frente a texto mencionado |
| task / hypothetical | autoridad presente frente a posibilidad futura |
| single / multi-label | vocabulario secundario frente a dos outcomes independientes |

## Niveles de dificultad

- `D0_explicit`: método nombrado y autoridad clara;
- `D1_paraphrase`: intención clara sin palabra de categoría;
- `D2_overlap`: vocabulario compartido con una categoría vecina;
- `D3_composed`: dos o más operaciones, constraints o secuencia;
- `D4_pragmatic`: cita, negación, ironía, elipsis o pregunta-directiva;
- `D5_contextual`: sólo se resuelve con el turno anterior sanitizado.

Holdout no estará compuesto sólo por D4/D5; debe medir uso normal y casos
difíciles. Las métricas se reportan por nivel.

## Split y holdout sellado

La unidad indivisible es `scenario_family`, que agrupa root cause, proyecto,
conversación, paráfrasis, traducciones y negativos derivados. Una familia vive
en un solo split.

- `calibration`: autoría y fit;
- `validation`: arquitectura, thresholds y abstención;
- `challenge`: adversariales renovables, nunca usados como holdout final;
- `holdout_sealed`: evaluación única después de congelar código y parámetros.

El manifiesto del holdout contendrá IDs y hash, pero sus textos permanecerán
fuera del flujo de desarrollo. Al abrirlo, pasa automáticamente a
`development_archive`; para otra afirmación de generalización se escribe un
holdout nuevo.

La asignación separará simultáneamente:

- familias de proyecto;
- familias causales concretas;
- lotes de autoría;
- estilos de frase;
- conversaciones multi-turn completas.

No intentaremos que cada split sea una copia porcentual exacta. Validation y
holdout deben contener combinaciones nuevas pero plausibles, no huecos
artificiales imposibles de aprender.

## Matriz de cobertura inicial

Los números son presupuestos de autoría, no un objetivo estadístico definitivo:

- 60 requests individuales por cada uno de los seis métodos;
- 18 requests por cada combinación multi-label prioritaria;
- 240 `uncategorized`, repartidos entre sus razones;
- al menos 40% de negativos difíciles con vocabulario de una policy;
- al menos 35% de filas D2-D5;
- al menos 45% en idiomas distintos del inglés;
- ningún tipo de proyecto con más del 10% del corpus;
- ningún estilo, usuario o familia causal con más del 15%.

La cobertura se controla como una matriz dispersa: no se construye el producto
cartesiano de todas las dimensiones. Cada fila debe justificar qué hueco real
cubre.

## Contrato de una fila manual

Cada línea JSONL contiene:

- `id`, `text` y, para D5, `context_before`;
- `policies` multi-label o `uncategorized_reason`;
- `scenario_family` y `contrast_family`;
- `user_type`, `project_type` y `programming_languages`;
- `language`, `locale` y `linguistic_features`;
- `style` y `difficulty`;
- `authority` y `constraints` observados literalmente;
- `split`, `source`, `author` y `review_status`;
- `rationale`: por qué aplica la etiqueta;
- `contrast_note`: cuál es la confusión peligrosa más cercana.

No se permite una fila `approved` si faltan rationale, contraste y revisor. El
autor no puede ser el único revisor del holdout.

## Proceso de autoría manual

1. Elegir un hueco de la matriz antes de escribir texto.
2. Imaginar un escenario técnico concreto con contrato y outcome propios.
3. Redactar como el arquetipo de usuario, sin insertar palabras de label por
   conveniencia.
4. Asignar políticas y autoridad leyendo solamente el request terminado.
5. Escribir rationale y la alternativa más confundible.
6. Buscar duplicación conceptual con filas existentes.
7. Marcar `draft`; una segunda revisión puede aprobar, corregir o rechazar.
8. Congelar familias completas antes de entrenar.

Está prohibido producir N filas sustituyendo componente, idioma o sinónimo en
una frase base. Un auditor puede contar, validar schema y buscar duplicados,
pero nunca generar el contenido del dataset.

## Orden de construcción

1. calibration: lote balanceado de casos simples y fronteras principales;
2. calibration: multi-label, code switching y negativos pragmáticos;
3. validation: nuevas familias de proyecto y autoría independiente;
4. challenge: citas, negaciones, ambigüedad y secuencias adversariales;
5. holdout sellado: familias inéditas redactadas al final;
6. entrenamiento E5-small contrastivo y cabeza con abstención;
7. una única apertura de holdout y posterior E2E del prompt seleccionado.

No se redactará holdout mientras todavía estemos decidiendo labels, schema o
arquitectura. Primero deben estabilizarse calibration, validation y challenge.

## Estado de construcción

Los lotes `boundary-01`, `causal-02`, `pragmatic-03`, `register-04` y
`mechanisms-05` contienen 201 filas de calibration escritas manualmente:

- 126 requests con una policy, 27 con dos, 3 con tres y 45 `uncategorized`;
- entre 30 y 33 apariciones de cada una de las seis policies;
- las 15 razones planificadas de `uncategorized` más `ambiguous_method`, que
  separa una discusión explícita de métodos de la mera falta de autoridad;
- 18 familias de proyecto, 15 tipos de usuario y 18 estilos;
- inglés, español, portugués, francés, alemán e italiano;
- 67,2% de filas fuera del inglés;
- 76,1% de dificultad D2-D5;
- máximo 7,0% para una familia de proyecto, 10,9% para un tipo de usuario y
  12,4% para un estilo;
- cero IDs, textos o familias de escenario duplicados y cero placeholders de
  template.

La auditoría reproducible en `bench.task_policy_manual_audit` tampoco encontró
duplicados claros: no hubo pares con coseno E5-small mayor o igual que 0,95. El
máximo fue 0,9428 entre dos casos distintos que comparten correctamente
`bugfix + performance`. Los vecinos con labels diferentes exponen fronteras
útiles como refactor/bugfix. La similitud sirve para enviar pares a revisión,
pero nunca para rechazarlos automáticamente.

```bash
uv run python -m bench.task_policy_manual_audit
uv run python -m bench.task_policy_manual_audit --semantic --top-k 20
```

Todas permanecen `draft`. El lote prueba el contrato y la disciplina de
autoría, no suficiencia estadística. Todavía no existe validation ni un nuevo
holdout; por lo tanto no corresponde reentrenar o publicar métricas de modelo.
