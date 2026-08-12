# Ledger de cobertura del corpus manual

Este ledger convierte la taxonomía general en lotes concretos de autoría. No
contiene templates ni texto destinado a expandirse automáticamente. Cada ítem
es un hueco que debe convertirse en un escenario original, con rationale y
contraste propios.

## Estado antes de `mechanisms-05`

- 140 filas de calibration;
- labels: bugfix 22, feature 22, refactor 21, research 23, review 21,
  performance 21;
- 31 negativos con 16 razones;
- cardinalidad: 89 single-label, 19 double-label, 1 triple-label;
- 18 proyectos, 15 usuarios, 18 estilos y 6 idiomas;
- ninguna familia de escenario repetida.

## `mechanisms-05`

Presupuesto inicial: 60 escenarios individuales. Cierre real: 61. La auditoría
de distribución dejó `review` en 29 apariciones; se añadió un review móvil
genuino en vez de rebajar el gate o forzar una etiqueta dudosa. Objetivo:
mecanismos todavía débiles, no paráfrasis de los lotes anteriores.

### Bugfix: seis escenarios

- zona horaria/DST que altera una recurrencia existente;
- deadlock por orden de locks;
- índice secundario que queda inconsistente tras recovery;
- use-after-free o lifetime en un callback nativo;
- locale/normalización que rompe identificadores existentes;
- pérdida de foco/estado accesible tras re-render.

### Feature: seis escenarios

- webhook/evento nuevo con contrato de reintentos;
- permisos delegados de alcance limitado;
- soporte de plataforma nueva;
- búsqueda guardada/compartida;
- importación incremental resumible;
- extensión de plugin declarativa.

### Refactor: seis escenarios

- seam de reloj para test sin alterar timing real;
- adapter para aislar API legacy;
- separar parseo de validación;
- unificar builder de consultas interno;
- extraer view-model preservando accesibilidad;
- encapsular ownership/lifetime nativo.

### Research: seis escenarios

- threat model comparativo;
- diseño de benchmark antes de medir;
- build-versus-buy con mantenimiento/licencia;
- compatibilidad normativa regional;
- estrategia de migración basada en evidencia;
- comparación de formatos con integridad y recuperación.

### Review: siete escenarios

- gaps de fixtures y tests;
- documentación versus comportamiento real;
- plan de migración de datos;
- permisos/IAM de infraestructura;
- cambio de ABI o FFI;
- patch de accesibilidad/UI.
- routing/deep links y autenticación móvil.

### Performance: seis escenarios

- N+1/plan de consultas;
- conexiones/reuso de red;
- tiempo de compilación incremental;
- ocupación y transferencias GPU;
- compresión versus CPU;
- lock contention bajo concurrencia.

### Multi-label: diez escenarios

- bugfix + refactor;
- feature + refactor;
- bugfix + performance;
- feature + research;
- bugfix + research;
- feature + performance;
- research + review;
- review + performance read-only;
- feature + research + performance;
- bugfix + refactor + performance.

### `uncategorized`: catorce escenarios

- dos acknowledgements/continuaciones;
- status-only;
- explicación de código actual;
- pregunta conceptual;
- dos acciones citadas o reportadas;
- futuro hipotético;
- meta-method;
- autoridad ambigua;
- tarea soportada por el agente pero no por estas policies;
- conflicto de constraints;
- contexto insuficiente;
- conversación fuera de dominio.

## Criterios de cierre del lote

- 61 IDs nuevos y 61 familias de escenario nuevas, incluida la corrección de
  balance documentada;
- ninguna sustitución mecánica de componente, idioma o sinónimo;
- al menos 45% fuera del inglés;
- cada policy termina con 30 o más apariciones acumuladas;
- máximo 10% por proyecto y 15% por usuario o estilo;
- cardinalidad 0/1/2/3 conservada;
- todos los casos permanecen `draft`;
- revisión de schema, duplicados literales, vecinos E5 y suite completa.

## Presupuesto de cierre de calibration

La auditoría posterior a `mechanisms-05` separó apariciones de labels de
ejemplos single-label. No se considerará terminada calibration contando un
ejemplo multi-label tres veces.

- 60 ejemplos single-label por cada policy;
- 18 ejemplos para cada una de las nueve combinaciones prioritarias;
- 240 `uncategorized`, con al menos 12 por razón principal y más para las
  fronteras abiertas (`conceptual_question`, `quoted_action`,
  `ambiguous_authority`, `unsupported_method` y continuaciones);
- cardinalidad 3 tratada como challenge de composición, no como reemplazo de
  los pares;
- cada escenario y texto escrito individualmente y asignado a una sola familia.

Estado tras 277 filas:

| Policy | Single-label | Total de apariciones |
| --- | ---: | ---: |
| bugfix | 30 | 42 |
| feature | 30 | 42 |
| refactor | 33 | 40 |
| research | 29 | 43 |
| review | 34 | 40 |
| performance | 30 | 42 |

Las combinaciones tienen entre 2 y 4 filas y `uncategorized` tiene 61. Por lo
tanto el dataset todavía no está terminado y no corresponde entrenar.

## `frontiers-06`

Presupuesto: 60 escenarios single-label, diez por policy. Se priorizan
fronteras ausentes: fallos de caché y cancelación, features de colaboración y
extensibilidad, refactors de boundaries e invariantes, research de protocolos
y decisiones, review de artefactos diversos y performance de I/O, serialización
y recursos. Ninguna fila de este lote puede ser multi-label o uncategorized.

Cierre: 60 escenarios, sin duplicados. Cada policy ganó diez single-label. La
auditoría posterior dejó 30/30/33/29/34/30 casos single-label para bugfix,
feature, refactor, research, review y performance respectivamente.

## `abstention-07`

Primer sublote: 16 negativos difíciles, uno por cada razón `uncategorized`
existente. Se escribieron como shard separado para que cada iteración manual
sea revisable sin reescribir el JSONL histórico. Incluye vocabulario literal de
las policies dentro de citas, preguntas conceptuales, reportes de terceros,
continuaciones, estado, autoridad o método ambiguos y tareas reales que ninguna
policy especializada debe interceptar.

Estado: 61 negativos, 277 filas totales y cero IDs, textos o familias de
escenario duplicadas. El próximo sublote debe profundizar las razones abiertas;
una pasada uniforme no sustituye los negativos adversariales específicos de
cada frontera.

## `composition-08`

Primer sublote dirigido por cross-validation con `E5-base`: 20 escenarios
multi-label nuevos. Se priorizaron `feature+research` y `bugfix+performance`,
seguidos de `research+review`, `review+performance`, `feature+refactor`,
`bugfix+refactor`, `feature+performance` y dos triples.

Estado tras el lote: 297 filas; cardinalidad 0/1/2/3 = 61/186/45/5. Las
combinaciones prioritarias tienen ahora entre 4 y 8 ejemplos, todavía por
debajo del presupuesto de 18. Cero IDs, textos o familias duplicadas. El lote
no cierra composition: su evaluación out-of-fold confirmó que 2–5 positivos
visibles por combinación siguen siendo insuficientes.

## `composition-09`

Segundo sublote dirigido: 40 escenarios multi-label nuevos en dos shards. Se
evitaron marcadores explícitos como “esto es feature y research”; cada caso
combina operaciones causalmente separables en proyectos, idiomas y estilos
distintos.

Estado tras el lote: 337 filas; cardinalidad 0/1/2/3 = 61/186/77/13. Las
combinaciones acumuladas son:

| Combinación | Filas |
| --- | ---: |
| bugfix + performance | 12 |
| bugfix + refactor + performance | 6 |
| bugfix + refactor | 9 |
| bugfix + research | 8 |
| feature + performance | 9 |
| feature + research + performance | 7 |
| feature + refactor | 9 |
| feature + research | 12 |
| performance + review | 8 |
| research + review | 10 |

La auditoría conserva cero IDs, textos y familias duplicadas, y ningún par E5
con similitud de 0.95 o más. El máximo por proyecto es 8.6%, por estilo 14.2%
y por tipo de usuario 11.9%; 69.7% de los casos no están en inglés.

La cross-validation muestra que el siguiente lote no debe repartir ejemplos
uniformemente. Debe contrastar las falsas activaciones de autoridad/método
ambiguo, requests conflictivos y acciones reportadas, además de completar los
pares que aún tienen menos de 12 casos. `composition-09` mejora la señal para
un head independiente, pero no cierra calibration ni autoriza validation.

## `singles-14`

Primer lote de cierre de las políticas individuales: 60 escenarios originales,
diez por policy. Se escribieron como fronteras semánticas explícitas: resultados
incorrectos frente a costos altos, capacidades nuevas frente a decisiones todavía
abiertas, estructura interna sana frente a reparación, y revisión de un artefacto
concreto frente a investigación abierta.

Estado tras 527 filas: bugfix 49, feature 58, performance 47, refactor 52,
research 55 y review 57 ejemplos single-label. El lote incorpora 42 tipos de
proyecto, 41 tipos de usuario y 44 estilos adicionales; conserva seis idiomas,
71.5% de ejemplos no ingleses y cero IDs, textos o familias repetidas.

Faltan 42 singles para el presupuesto de 60 por policy: bugfix 11, feature 2,
performance 13, refactor 8, research 5 y review 3. Esta cifra es un gate de
cobertura, no una predicción de que el modelo alcanzará 95% al completarla.

## `singles-15`

Cierre del presupuesto single-label con 42 escenarios: bugfix 11, feature 2,
performance 13, refactor 8, research 5 y review 3. Las seis políticas quedan en
60 casos individuales cada una. El corpus suma 569 filas, 135 tipos de proyecto,
125 tipos de usuario, 152 estilos, 72.1% de texto no inglés y ninguna repetición
literal de ID, texto o familia causal.

El cierre de singles no cierra calibration: todavía faltan 44 composiciones para
llevar las combinaciones prioritarias a 18 y 169 negativos para llegar a 240.

## `composition-16`

Cierre de 44 composiciones dirigidas. Las diez firmas prioritarias quedan con
18 o más escenarios: se añadieron dos bugfix+performance, ocho triples
bugfix+refactor+performance, cinco bugfix+refactor, seis bugfix+research, cinco
feature+performance, siete triples feature+research+performance, cinco
feature+refactor y seis review+performance.

Cada triple expresa tres outcomes verificables, no tres nombres yuxtapuestos.
Por ejemplo, separar causa y restauración, una frontera interna que debe
preservarse, y un presupuesto medido. El corpus suma 613 filas, 146 pares y 36
triples; calibration todavía requiere 169 negativos difíciles.

## `abstention-17`

Primer lote del cierre negativo: 40 casos manuales repartidos entre las 16
razones. Incluye asentimientos, autoridad y método ambiguos, preguntas
conceptuales, continuaciones con antecedente no resoluble, discurso citado o
reportado, estado, progreso sano y acciones externas no soportadas. El corpus
queda en 653 filas y 111 negativos; faltan 129 para el presupuesto de 240.

## Evidencia incremental con folds estables

El asignador round-robin anterior movía filas históricas al ampliar el corpus,
por lo que no permitía comparar un `fold 0` entre iteraciones. Desde
`stable-family-hash-v2`, cada familia conserva su fold aunque se agreguen datos;
los contrastes emparejados continúan juntos.

Sobre E5-small con las doce capas entrenables:

| Corpus | Subconjunto comparable | Exact match | Errores |
| --- | ---: | ---: | ---: |
| cierre de singles, 569 filas | 147 históricos | 72.1% | 41 |
| cierre de composición, 613 filas | los mismos 147 | 76.2% | 35 |
| `abstention-17`, 653 filas | 160 históricos de la iteración anterior | 76.9% | 37 |

Los 16 negativos nuevos asignados al test de `abstention-17` se clasificaron
correctamente. La evaluación completa de ese fold fue 79.0% exact match, 90.0%
macro F1 y tres falsas activaciones. Sigue siendo diagnóstico de desarrollo,
no evidencia de holdout ni autorización para inyección automática.

## `abstention-18`

Segundo lote negativo: 43 escenarios adicionales. Aumenta especialmente
continuaciones con antecedentes múltiples, autoridad atribuida a terceros,
preguntas que citan nombres de policies y acciones externas que contienen
vocabulario de ingeniería. El corpus suma 696 filas y 154 negativos; faltan 86
para cerrar el presupuesto de 240.

## `abstention-19`

Tercer lote negativo: 46 escenarios manuales. Refuerza autoridad atribuida,
continuaciones cuyo referente se perdió o nunca fue seleccionado, texto técnico
citado para traducción o edición, y acciones externas que mencionan bugs,
benchmarks o reviews sin pedir trabajo sobre código. El corpus suma 742 filas y
200 negativos; faltan 40 para cerrar exactamente 240.

## `abstention-20` y cierre de calibration

Último lote: 40 negativos manuales. Calibration queda en 782 escenarios:

- 240 negativos, con las 16 razones representadas por al menos 12 casos;
- 60 ejemplos single-label para cada una de las seis policies;
- 146 pares y 36 triples, con 18 o más casos por firma prioritaria;
- 241 tipos de proyecto, 264 tipos de usuario y 363 estilos;
- 71.1% de texto no inglés;
- cero IDs, textos o familias causales repetidos literalmente.

Este cierre satisface el presupuesto de autoría; no prueba por sí mismo el gate
de 95%. La selección de arquitectura, validación natural y holdout permanecen
como pasos separados.

## `precision-boundaries-21` y `performance-negatives-22`

El criterio de aceptación quedó definido como **accuracy binaria individual por
policy mayor a 95%**, no exact-match del conjunto multi-label ni precision/PPV.
Se añadieron 72 contrastes escritos manualmente para fronteras que el análisis
de errores mostraba inestables: review frente a research, research con medidas
comparativas frente a performance, y vocabulario de offsets, compactación,
rotación, tamaños y scheduling usado en bugfix o refactor sin intención de
optimizar. El corpus suma 854 escenarios y conserva cero IDs, textos o familias
duplicados literalmente.

La evaluación usa cuatro folds estables separados por familia. Cada fold parte
del encoder publicado, entrena una cabeza de atención por policy y selecciona
checkpoint, precision floor y thresholds exclusivamente en validation. Las 854
predicciones out-of-fold agregadas dieron:

| Policy | Accuracy | Precision | Recall |
| --- | ---: | ---: | ---: |
| bugfix | 98.01% | 91.50% | 97.22% |
| feature | 96.84% | 87.41% | 93.28% |
| refactor | 96.84% | 90.24% | 88.10% |
| research | 96.25% | 86.14% | 94.08% |
| review | 98.24% | 90.91% | 96.49% |
| performance | 95.43% | 88.28% | 85.33% |

El exact-match global fue 83.26% y se conserva como diagnóstico secundario; no
es el gate solicitado. Esta evidencia supera el gate individual en calibration,
pero performance tiene un margen pequeño y el corpus es sintético. Antes de
afirmar generalización hay que repetirlo en validation natural sellada, separada
por repositorio y familia, y después verificar si el prompting condicionado
mejora respuestas E2E, tokens, tool calls, tiempo, errores y regresiones de los
modelos objetivo.
