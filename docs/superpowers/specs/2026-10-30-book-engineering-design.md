# Criterio de ingeniería derivado de los skills basados en libros

## Intención y autoridad

El usuario pidió leer los skills basados en libros, extraer sus ideas importantes
con Superpowers y aplicarlas a los prompts o la lógica de Infinidev. Mantiene la
delegación previa de las decisiones de implementación. Este diseño documenta
decisiones del implementador; no atribuye al usuario la elección de cada regla.

El objetivo es mejorar las decisiones del agente: comprender contratos, proteger
invariantes, verificar cambios y manejar fallos. No convertir cada tarea en una
ceremonia de arquitectura ni agregar otra capa de clasificación.

## Lecturas y síntesis

Se leyeron los SKILL.md y las reglas mini indicadas por los siguientes skills
locales. Los documentos extensos quedan como referencia de profundización; no se
afirma haber leído los libros completos.

| Skill / fuente | Idea útil para Infinidev | Aplicación elegida |
| --- | --- | --- |
| A Philosophy of Software Design — Ousterhout | Reducir carga cognitiva; interfaces pequeñas que ocultan decisiones | Cohesión y profundidad por encima de dividir funciones mecánicamente |
| Clean Architecture — Martin | Política independiente de mecanismos volátiles | Dueño de invariantes y traducción en fronteras; sin capas de reenvío gratuitas |
| Clean Code — Martin | Nombres precisos, efectos visibles y razonamiento local | Una terminología por concepto, errores explícitos, pruebas del comportamiento |
| Code Complete — McConnell | Construcción incremental, límites e invariantes verificables | Datos inválidos en fronteras, pruebas de casos límite y fallos alcanzables |
| Designing Data-Intensive Applications — Kleppmann | Fuente de verdad, estado derivado, reintentos y éxito desconocido | Guía condicional de persistencia, proyecciones y efectos repetidos |
| Domain-Driven Design — Evans | Lenguaje y modelo que expresan reglas reales | Conceptos e invariantes con propietario, sin introducir DDD por defecto |
| Domain-Driven Design Distilled — Vernon | Invertir modelado donde existe complejidad de dominio | Un flujo sencillo puede seguir siendo una función sencilla |
| Implementing Domain-Driven Design — Vernon | Transiciones válidas y traducción de modelos externos | Pruebas de invariantes y contratos en la frontera que los posee |
| Patterns of Enterprise Application Architecture — Fowler | Elegir estructura por las fuerzas del problema | Coordinar casos de uso sin absorber reglas ni esconder transacciones |
| Refactoring — Fowler | Cambiar estructura preservando comportamiento | Baseline, pasos reversibles, distinguir cambio funcional de limpieza |
| Refactoring.Guru | Diagnosticar el olor y definir cuándo parar | Causa del coste de mantenimiento y condición de salida concreta |
| Release It! — Nygard | Fallos parciales, límites y recuperación explícitos | Reintento seguro y acotado; timeout no demuestra que no hubo efecto |
| The Pragmatic Programmer — Hunt/Thomas | Un dueño por conocimiento; feedback de un corte real | Reutilizar catálogo, probar una ruta completa y mantener decisiones reversibles |
| Working Effectively with Legacy Code — Feathers | Obtener control antes de reestructurar código incierto | Caracterización y seam mínimo; no afirmar que tests antiguos son toda la especificación |

También se revisaron Machine Learning Engineering y Deprecation and Migration.
Del primero sirve medir el cuello de botella y reproducir en pequeño, ya cubierto
por la política de performance; recetas de GPU, SLURM o entrenamiento no pertenecen
al prompt genérico. Del segundo sirve separar cambios aditivos y retirada de
formatos usados. Una recuperación puede ser rollback o reparación hacia delante;
no se promete que todo cambio de datos pueda deshacerse.

## Estado actual y problemas concretos

- `ENGINEERING_JUDGMENT` ya reúne reglas comunes de autoridad, alternativas,
  contratos, comunicación y continuidad. Se conserva su responsabilidad.
- `TASK_METHOD_FRAGMENTS` ya separa planner/developer/reviewer y condiciones por
  operación. El modo por defecto presenta el catálogo condicional; el modo
  seleccionado conserva gates por modelo/versión.
- Los fragments `bugfix.developer@3` y `refactor.developer@1` tienen aprobaciones
  de evaluación específicas. No se cambia su contenido ni se reutilizan sus
  aprobaciones para nuevas instrucciones.
- Los cores full y lean obligan a simular dependencias y proponen divisiones por
  etapas mecánicas. Esto puede ocultar fallos de integración o aumentar el número
  de interfaces sin reducir complejidad. Generalized también propone dividir por
  cantidad de acciones en vez de por responsabilidad.
- El compositor recorta el contenido de un fragmento a su límite individual;
  el renderer vuelve a recortarlo. Una regla final de autoridad o recuperación
  puede desaparecer aunque el ID/hash anuncie el documento completo.

## Alternativas

1. **Copiar todos los libros al core.** Disponibilidad universal, pero repetición,
   conflictos y reglas irrelevantes incluso para tareas sencillas. Se descarta.
2. **Síntesis corta y métodos condicionales existentes.** Un pequeño criterio
   común, métodos por rol y guías opt-in para riesgos especializados. Reutiliza
   contratos, presupuesto y perfiles existentes. Es la elección del implementador.
3. **Nuevo router/checklist obligatorio.** Podría automatizar más decisiones, pero
   exige nuevos estados, calibración y excepciones sin demostrar valor adicional.
   Se descarta para esta ronda.

## Diseño

### A. Criterio común y cores coherentes

Ampliar moderadamente el bloque común existente: dueño de la regla/fuente de
verdad, vocabulario consistente, invariantes en el propietario y detalles volátiles
en fronteras. Expresar los principios como criterios para escoger la solución más
simple que protege el contrato. No exigir clases, agregados, repositorios ni
interfaces a un CRUD o arreglo local.

En full, lean y generalized corregir únicamente reglas de construcción afectadas:
funciones cohesivas y módulos profundos, pruebas observables, fakes donde hagan
falta para aislar y checks reales de integración donde el contrato sea la frontera.
Caracterizar comportamiento incierto sin canonizar un defecto conocido. Mantener
las reglas de autorización, evidencia, seguridad y scope.

### B. Métodos por rol y capacidades existentes

En los métodos actuales de planner/reviewer para bugfix/refactor y en los de
feature/performance, incorporar en pocas frases las decisiones pertinentes:

- planner: contrato, incertidumbre, ejemplo que lo contradiga y corte verificable;
- developer: corte completo, dueño de invariantes y efectos/recovery en la frontera;
- reviewer: defecto o contrato demostrable, equivalencia, prueba que detecta el
  fallo; ninguna preferencia de arquitectura se vuelve obligación del usuario.

Mantener IDs, operaciones, autoridad, prioridades y filtros; incrementar la versión
de cada fragmento cuyo contenido cambie. No tocar los dos fragments developer
con aprobación previa. Las capacidades opt-in de arquitectura, migración, errores,
concurrencia, recursos, test quality y refactoring reciben las reglas especializadas
que no corresponden al core. Siguen apagadas por defecto. No agregar capabilities
ni cambiar catálogos, settings, modelos o perfiles del usuario.

### C. Fragmentos indivisibles en el compositor

Un fragmento es una unidad de instrucción, no texto de salida abreviable. Los dos
selectores conservan su elegibilidad y orden actuales. Tras filtrar:

- texto vacío: omitir con razón `empty`;
- longitud UTF-8 completa mayor que `fragment.max_utf8_bytes`: omitir con razón
  `fragment-budget`, sin escoger un prefijo;
- fragmento completo que no cabe en el presupuesto de contenido existente:
  omitir con razón `budget`;
- en otro caso, seleccionar íntegro y contabilizarlo sin recortar.

Eliminar el helper de clipping y los recortes repetidos del renderer. El límite
global conserva su significado actual; esta ronda no redefine la contabilidad de
wrappers XML. La firma pública de selectores y renderers no cambia. Los bloques
del catálogo real deben caber enteros en sus límites y todos los de cada rol deben
seguir cabiendo en el presupuesto por defecto. Un bloque no aprobado en el modo
seleccionado continúa omitido por su gate de evidencia.

## Límites y decisiones del implementador

No imponer una metodología única ni atribuir doctrina a los usuarios de Infinidev.
No usar los libros como prueba de mejoras de calidad del modelo. No cambiar el
engine de finalización, la cola de usuario, persistencia, APIs de tools ni planes
del usuario. No instalar dependencias ni modificar skills personales. Preservar
todos los cambios anteriores del checkout. Sin commits ni publicaciones.

## Validación

Reproducir primero el recorte inseguro con fragmentos sintéticos que terminan en
una condición esencial, para selección por perfil y catálogo completo; comprobar
UTF-8, límite exacto, omitidos y renderizado íntegro. Cubrir que los fragments
reales caben, conservan filtros de rol/fase/autoridad, gates y cache prefix.

Verificar los prompts realmente compuestos y perfiles opt-in, con checks de
contrato focalizados, sin una batería que congele cada palabra del prompt.
Ejecutar tests relacionados, revisión independiente y suite completa mediante
`uv run --frozen --no-sync pytest`. No usar credenciales ni llamar modelos remotos
para afirmar una mejora empírica que no se ha medido.
