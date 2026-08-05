# Continuación: batería para entender cómo interpreta prompts cada modelo

## Objetivo

Construir un framework reutilizable que permita observar **qué significado reconstruye un modelo**
cuando recibe distintos tipos de prompts y usar esa evidencia para mejorar cómo Infinidev se comunica
con cada modelo.

La intención no es encontrar un “prompt mágico” universal ni asignarle una personalidad mediante
unos pocos números. Queremos conservar las respuestas concretas del modelo y construir un mapa de su
comportamiento observable por categoría:

- qué cree que se le pidió;
- qué considera objetivo, entregable y restricciones;
- qué acciones interpreta como autorizadas o no autorizadas;
- qué decisiones cree que pertenecen al usuario;
- qué contradicciones y ambigüedades detecta;
- cómo resuelve precedencia, excepciones y alcance;
- qué evidencia cree necesaria antes de afirmar que terminó;
- cuándo entiende que puede continuar y cuándo debe detenerse.

Esto permite adaptar Infinidev a distintos modelos —OpenAI, Anthropic, Kimi, MiniMax u otros— y a
distintas preferencias de usuario. No existe una única conducta óptima: algunos usuarios priorizan
velocidad y autonomía; otros, control, interacción o verificación adicional.

“Entender cómo piensa” significa aquí construir un mapa empírico de sus **interpretaciones y
decisiones observables**. No significa afirmar acceso a chain-of-thought privado ni a un estado
mental literal.

## Distinción fundamental entre prompts

Los resultados anteriores mostraron que hay que separar responsabilidades que antes podían quedar
mezcladas:

1. **Behavior prompt**
   - Identidad del agente y relación con Infinidev y su harness.
   - Estilo general de colaboración, autonomía, interacción, cautela y comunicación.
   - Preferencias dependientes del usuario.
   - No debería inventar el objetivo de una tarea ni ordenar acciones específicas para todos los
     casos.

2. **Execution-policy prompt**
   - Cómo ejecutar una clase de tarea: desarrollo, planificación, review, testing, research, etc.
   - Uso de herramientas, evidencia, fases, recuperación y criterios de terminación.
   - Puede variar según el tipo de trabajo y el modelo.

3. **Objective prompt**
   - Lo que el usuario quiere conseguir en esta tarea concreta.
   - Entregables, restricciones, prioridades y condiciones de aceptación.

4. **Context/evidence prompt**
   - Repositorio, archivos, historial relevante, resultados de herramientas y demás evidencia.

Las preguntas multiple-choice anteriores aportan información sobre preferencias y behavior. No
demuestran que un modelo interprete correctamente un objetivo o una política de ejecución. Esta
nueva batería existe para estudiar esa segunda cuestión por separado.

En runtime estas responsabilidades ya tienen representación tipada en
`src/infinidev/engine/prompt_layers.py`. Los perfiles calibrados de behavior no deben entrar
silenciosamente como reglas operativas de ejecución.

## Pregunta de investigación principal

> ¿Qué necesita saber Infinidev sobre la interpretación de un modelo para formularle objetivos,
> políticas y contexto de manera que mejore el resultado real?

No debemos comenzar preguntando “¿prefiere listas o párrafos?” ni generando una cuadrícula por el
simple hecho de alcanzar cientos de preguntas. El orden correcto es:

1. **¿Qué problema observamos en Infinidev trabajando con el modelo actual?**
2. **¿Qué información necesitamos entender del modelo para resolver ese problema?**
3. **¿Qué hipótesis alternativas podrían explicar el comportamiento?**
4. **¿Qué preguntas o ejemplos controlados distinguen esas hipótesis?**
5. **¿Qué evidencia concreta debemos conservar de la respuesta?**
6. **¿Qué intervención podría seguir?**
   - behavior prompt;
   - execution policy;
   - plantilla del objetivo;
   - disposición del contexto;
   - evaluador;
   - routing de modelos;
   - o ninguna modificación.
7. **¿La intervención resuelve el problema original en casos held-out y tareas reales?**

El mapa inicial de problemas está en `docs/INFINIDEV_LLM_PROBLEM_MAP.md`.

## Por qué usar familias controladas

Preguntarle directamente al modelo “¿qué formato preferís?” produce sólo un self-report. Es una
evidencia débil: puede no coincidir con su rendimiento real.

La batería usa familias de prompts:

- **Anchor:** versión de referencia.
- **Equivalent:** cambia una propiedad superficial pero conserva exactamente el significado
  revisado. Si la reconstrucción cambia, hay sensibilidad o fragilidad.
- **Contrast:** cambia un operador semántico y, por lo tanto, una parte específica de la clave
  esperada.
- **Adversarial:** introduce contradicción, ruido, ejemplos engañosos, ambigüedad o alcance difícil.

Ejemplo:

- `NEVER modify tests.`
- `Do not modify tests.` — debería ser equivalente en fuerza semántica.
- `Tests are sometimes stale; edit one only when repository evidence proves the contract changed.`
  — introduce una excepción real y acotada.

Así se puede distinguir si mayúsculas cambian la interpretación, si el modelo comprende
`sometimes`, y si mantiene estrecha la excepción. Sólo después se prueba si esa interpretación se
traduce en mejor ejecución.

## Qué debe recolectar cada respuesta

La respuesta es híbrida, no sólo multiple-choice:

- `understanding`: reconstrucción libre en palabras del modelo;
- `objective`;
- `deliverables`;
- `constraints`;
- `user_owned_decisions`;
- `authorized_actions`;
- `unauthorized_actions`;
- `verification`;
- `ambiguities`;
- `stop_conditions`;
- `conflicts`;
- `priority_resolution`;
- `interpretation_risks`;
- `confidence`.

El texto libre es central porque revela omisiones, énfasis e inferencias concretas. Los campos
estructurados permiten comparar familias, pero no deben reemplazar las respuestas ni reducirlas a
un score universal.

## Estado actual

### Infraestructura implementada

- `bench/prompt_comprehension.py`
  - contrato de casos y observaciones;
  - composición tipada de condiciones;
  - condición `raw` sin system prompt;
  - parsing de reconstrucción libre y campos estructurados.
- `bench/run_prompt_comprehension.py`
  - llamadas aisladas, sin historial compartido;
  - ejecución estrictamente secuencial;
  - lock global single-flight;
  - sin retries ni cache de LiteLLM;
  - detención ante el primer 429;
  - intervalo por defecto de 1 segundo y mínimo configurable de 0,75 segundos.
- `bench/prompt_comprehension_report.py`
  - conserva respuestas crudas y reconstrucciones concretas;
  - separa por categorías y familias;
  - usa números sólo para salud, cobertura y coste de la colección.
- `src/infinidev/engine/prompt_layers.py`
  - separación runtime de behavior, execution policy, objective y context/evidence.

### Batería materializada

El archivo principal es:

`bench/prompt_comprehension_battery.draft.jsonl`

Contiene actualmente:

- **672 casos únicos**;
- **224 familias controladas**;
- **3 variantes por familia**;
- **18 fenómenos lingüísticos/pragmáticos**;
- **8 dominios de tareas**;
- **336 casos de calibración**;
- **336 casos de validación**.

Cada fenómeno lingüístico tiene 24 casos y cada dominio tiene 84. Los 240 casos adicionales
materializan 10 dimensiones de execution policy en los ocho dominios, con tres variantes por
pregunta de investigación.

Fenómenos cubiertos:

1. registro formal, semiformal e informal;
2. párrafo, lista y tabla;
3. lenguaje directo, cortés e indirecto;
4. `NEVER`, `ALWAYS`, `SOMETIMES`, `must`, `should`, preferencias y defaults;
5. negación y alcance;
6. cuantificadores como `all`, `each` y `some`;
7. excepciones con `unless`, `except` y autorización condicionada;
8. contradicción y precedencia de instrucciones;
9. vaguedad y falta de especificación;
10. referencias ausentes o resolubles por contexto;
11. ruido e información irrelevante;
12. ejemplos consistentes o incompatibles con reglas;
13. posición de instrucciones al principio, medio o final;
14. pragmática, implicaturas y cortesía;
15. typos, gramática imperfecta y code-switching español/inglés;
16. formatos y schemas de salida;
17. condiciones temporales y cambios de estado;
18. solicitudes compuestas y alcance anidado.

Dominios cubiertos:

1. planificación;
2. implementación;
3. testing y verificación;
4. code review;
5. research en la web;
6. interacción con el usuario;
7. ayuda para decisiones;
8. autorización y acciones con estado externo.

### Trazabilidad y auditoría

- `bench/prompt_comprehension_family_registry.json` conserva para cada una de las 224 familias:
  problema de Infinidev, pregunta de investigación, utilidad para el producto, información buscada,
  hipótesis, evidencia, intervenciones posibles y confirmación held-out.
- `bench/prompt_comprehension_battery.audit.json` contiene la auditoría machine-readable.
- `bench/generate_prompt_comprehension_battery.py` regenera determinísticamente la batería.
- `bench/prompt_comprehension_battery_audit.py` verifica:
  - cantidades y cobertura;
  - IDs y prompts únicos;
  - familias atómicas por split;
  - claves idénticas para equivalentes;
  - claves diferentes para contrastes;
  - campos semánticos completos;
  - dimensiones del estímulo completas;
  - trazabilidad problema→investigación→intervención.

La auditoría estructural actual pasa. Los casos permanecen intencionalmente como `draft`.

## Qué falta implementar

### 1. Completar la revisión semántica humana

La auditoría automática no puede demostrar que las 672 claves sean conceptualmente correctas.
Falta revisar las 224 familias, idealmente con revisión independiente y ciega respecto de la clave:

- confirmar que cada equivalente conserva realmente el significado;
- confirmar que cada contraste cambia una sola variable relevante;
- rechazar frases artificiales, ambiguas accidentalmente o poco realistas;
- comprobar que la clave no contiene información ausente del prompt;
- comprobar que `authorized_actions` no confunde una preferencia con permiso;
- comprobar que conflictos y excepciones se resuelven con la autoridad correcta;
- verificar que calibración y validación no tengan leakage semántico indebido.

El flujo formal ya existe en `bench/prompt_comprehension_review.py`: exporta un packet ciego sin las
claves, exige una reconstrucción completa por variante, genera después un dossier que revela ambas
interpretaciones y aplica sólo adjudicaciones explícitas a familias completas. El packet actual está
materializado en `bench/prompt_comprehension_battery.review-packet.json`.

Lo pendiente es humano: obtener revisiones independientes de las 224 familias y adjudicar sus
diferencias. No se deben convertir los 672 casos a `approved` automáticamente ni por auto-revisión
del autor.

Antes de la revisión completa ya está preparado un piloto determinista de 16 familias/48 casos, con
dos familias por dominio y 16 dimensiones de investigación distintas. Sus artefactos son
`bench/prompt_comprehension_pilot.review-packet.json`,
`bench/prompt_comprehension_pilot.reviews.template.jsonl` y
`bench/prompt_comprehension_pilot.manifest.json`; las instrucciones están en
`docs/PROMPT_COMPREHENSION_PILOT_REVIEW.md`.

El piloto también está dividido en cuatro asignaciones de cuatro familias bajo
`bench/prompt_comprehension_pilot_shards/`. El comando `check` valida completitud, hashes, variantes,
duplicados y placeholders antes de aceptar la entrega de un revisor, pero deliberadamente no evalúa
acuerdo semántico ni aprueba casos.

MiniMax-M3 completó una primera revisión ciega de 16/16 familias. Esa revisión produjo cinco cambios
de contenido y dos cambios del protocolo; la evidencia original quedó congelada bajo
`bench/runs/minimax-m3-comprehension-review/`. Las siete familias afectadas fueron regeneradas y
revisadas nuevamente bajo el nuevo hash en `bench/runs/minimax-m3-comprehension-review-v2/`: las siete
recibieron `accept`, con 40 checks `pass`, dos `not_applicable_by_design` y cero `fail`. El análisis
está en `docs/MINIMAX_M3_PROMPT_COMPREHENSION_PILOT.md`. Esto valida las correcciones dirigidas, pero
no convierte automáticamente el dataset completo a `approved`.

GPT-5.6 Sol completó las 16 familias actuales mediante `openai_subscription`, aisladas, secuenciales
y sin system prompt. Marcó 15 como `revise` y una como `accept`, pero 13 fallos provienen en parte de
usar `requests_are_self_contained` como si midiera preparación para ejecutar en vez de comprensión
semántica. También encontró nueve posibles confounds de variable única que MiniMax-M3 no había
detectado. La comparación y provenance están en
`docs/GPT_5_6_SOL_PROMPT_COMPREHENSION_PILOT.md` y los brutos en
`bench/runs/gpt-5.6-sol-comprehension-review/`. El siguiente cambio metodológico debe separar
completitud semántica de suficiencia de contexto para ejecutar.

GLM-5.2 revisó las nueve familias donde Sol señaló confounds usando el endpoint Subscription/Coding
Plan. Aceptó 9/9, con 52 checks `pass` y dos `fail` intencionales en el caso de referentes ausentes.
MiniMax y GLM son más permisivos; Sol ofrece objeciones textuales más finas. El desacuerdo no debe
resolverse por mayoría: la adjudicación debe inspeccionar cada cláusula adicional. El reporte está en
`docs/GLM_5_2_PROMPT_COMPREHENSION_REVIEW.md` y los brutos en
`bench/runs/glm-5.2-comprehension-review/`.

### 2. Aumentar diversidad dentro de cada dominio si la revisión encuentra efecto plantilla

La batería completa ya está materializada, pero fue generada desde ocho escenarios-base para poder
controlar variables. Esto facilita comparación causal, aunque puede producir dependencia del
escenario. Antes de una campaña definitiva hay que decidir, con la revisión semántica, si hacen
falta escenarios-base adicionales por dominio.

Si se agregan, deben conservarse familias controladas y un split sin paraphrase leakage. No se debe
inflar cantidad mediante reformulaciones casi idénticas.

### 3. Mejorar el análisis automático por familia

El reporte ya conserva las respuestas, pero falta incorporar directamente el registry de hipótesis
para que cada familia muestre:

- problema original de Infinidev;
- pregunta de investigación;
- significado que debía permanecer estable;
- significado que debía cambiar;
- diferencias concretas entre reconstrucciones;
- omisiones e inferencias no soportadas;
- posible intervención, marcada siempre como hipótesis.

No hay que convertir ese análisis en un único ranking de “inteligencia” o “calidad”.

### 4. Congelar manifests de ejecución por proveedor y modelo

Se necesitan manifests inmutables para cada ruta real:

- GPT-5.6 Sol;
- GPT-5.6 Terra;
- GPT-5.6 Luna;
- Anthropic;
- Kimi;
- MiniMax;
- cualquier modelo futuro.

Cada manifest debe registrar provider, modelo/revisión, hash del dataset, temperatura, token budget,
condición, pacing y política de error.

La primera campaña debería usar **sólo la condición raw**, sin system prompt que indique cómo
comportarse. Las condiciones behavior y behavior+execution pertenecen a experimentos posteriores,
una vez establecido el baseline de comprensión cruda.

### 5. Revisar coste y número exacto de llamadas antes de ejecutar

Con la batería actual:

- raw completo: 672 llamadas por modelo;
- raw sólo validation: 336 llamadas por modelo;
- raw completo para Sol/Terra/Luna: 2.016 llamadas.

Antes de cualquier llamada se debe calcular coste/API o impacto sobre subscription, confirmar rate
limits actuales y obtener autorización explícita. No debe paralelizarse para acelerar.

### 6. Ejecutar las campañas de forma segura

Requisitos ya acordados:

- una pregunta por vez;
- cero paralelismo;
- una conversación nueva y sin historial por pregunta;
- raw sin system prompt;
- pacing inicial de 1 segundo, nunca menor a 0,75 sin evidencia del límite;
- detenerse ante el primer 429 o error del proveedor;
- no retry automático;
- persistir cada respuesta inmediatamente;
- no mezclar modelos, revisiones o hashes en un mismo reporte.

### 7. Construir mapas cualitativos por modelo y categoría

Después de ejecutar, producir un dossier separado para cada modelo que describa, con ejemplos
concretos:

- interpretaciones estables;
- sensibilidad a formato o registro;
- errores de alcance, negación, cuantificadores y excepciones;
- resolución de contradicciones;
- tendencia a inventar autoridad u objetivos;
- tratamiento de decisiones del usuario;
- diferencias por dominio;
- coste, latencia y errores de parseo como datos operativos secundarios.

Los números ayudan a localizar patrones, pero el insumo para escribir prompts son las respuestas
reales y sus diferencias.

### 8. Derivar candidatos pequeños y tipados

Cada hallazgo debe decidir primero qué componente corresponde modificar:

- behavior;
- execution policy;
- objective template;
- context layout;
- evaluator;
- routing;
- ninguno.

No compilar automáticamente toda divergencia en el system prompt. Un candidato debe ser pequeño,
citar familias repetidas y preservar conductas que ya funcionan.

### 9. Validar en comprensión held-out y luego en tareas reales

Un candidato no se promueve porque mejore las respuestas de esta batería. Debe:

1. mejorar familias de comprensión completamente held-out;
2. no producir regresiones semánticas en otras categorías;
3. mejorar tareas reales de agente con herramientas y costes observables;
4. sobrevivir repeticiones e idealmente revisión independiente;
5. seguir condicionado por las preferencias del usuario cuando no existe una política universal.

Sólo entonces puede considerarse un perfil runtime. “La mejor modificación es ninguna” sigue siendo
un resultado válido.

## Orden recomendado de continuación

1. Implementar el workflow de revisión ciega para las 224 familias.
2. Revisar y corregir las familias; producir un dataset aprobado nuevo y hash-bound.
3. Extender el reporte para consumir el registry de investigación.
4. Crear manifests raw-only para los modelos seleccionados.
5. Calcular llamadas, tiempo y coste; presentar el plan exacto.
6. Ejecutar primero un shard pequeño de calibración para falsificar el instrumento.
7. Si el instrumento funciona, ejecutar calibración completa secuencialmente.
8. Mantener validation cerrada hasta congelar las hipótesis e intervenciones.
9. Ejecutar validation y construir mapas cualitativos por modelo/categoría.
10. Proponer candidatos tipados y probarlos en tareas reales held-out.

## Comandos actuales

Regenerar:

```bash
uv run python -m bench.generate_prompt_comprehension_battery \
  bench/prompt_comprehension_battery.draft.jsonl \
  bench/prompt_comprehension_family_registry.json
```

Auditar:

```bash
uv run python -m bench.prompt_comprehension_battery_audit \
  bench/prompt_comprehension_battery.draft.jsonl \
  bench/prompt_comprehension_family_registry.json \
  bench/prompt_comprehension_battery.audit.json
```

Ejecutar sólo después de aprobación explícita y con un dataset aprobado:

```bash
uv run python -m bench.run_prompt_comprehension \
  APPROVED_CASES.jsonl RAW_ROUTE.json OUTPUT.observations.jsonl \
  --split calibration
```

Generar reporte:

```bash
uv run python -m bench.prompt_comprehension_report \
  APPROVED_CASES.jsonl OUTPUT.observations.jsonl \
  OUTPUT.report.json OUTPUT.report.md \
  --registry bench/prompt_comprehension_family_registry.json
```

## Condiciones de finalización del proyecto

La batería no estará terminada como instrumento de calibración sólo porque existan 672 prompts. Se
considerará lista cuando:

- las familias tengan revisión semántica independiente completa;
- exista un dataset aprobado e inmutable;
- no haya leakage indebido entre calibración y validación;
- los manifests raw-only estén ligados al hash aprobado;
- el runner y los reportes estén probados con un shard pequeño;
- las respuestas se conserven completas y separadas por modelo/categoría;
- las conclusiones estén ligadas a problemas reales de Infinidev;
- cualquier cambio de prompt sea validado después en tareas reales held-out.

Hasta entonces, el estado correcto es: **batería completa en cantidad y cobertura estructural,
todavía draft y pendiente de validación semántica y ejecución**.

## Auditoría de puertas de cierre (estado observado)

Fecha de auditoría: 2026-08-03. Estados cerrados: **APROBADA**, **FALLIDA** o
**NO EJECUTABLE**. «No ejecutable» significa que falta una intervención humana, autorización o
acceso externo que el repositorio no puede fabricar; no equivale a aprobación. La evidencia del
código corresponde al árbol de trabajo observado y a las pruebas enlazadas abajo. Los archivos de
prompt comprehension, incluidos los dos artefactos selectivamente exceptuados en
[`.gitignore`](.gitignore), son visibles para Git pero todavía no forman un commit limpio y
dedicado; por ello esta auditoría no atribuye los cambios al `HEAD` actual.

### Hito A — instrumento auditable

| Puerta | Estado | Evidencia y motivo |
|---|---|---|
| Cantidad, unicidad, balance y cobertura estructural | **APROBADA** | La [auditoría estructural](bench/prompt_comprehension_battery.audit.json) y su [implementación](bench/prompt_comprehension_battery_audit.py) cubren conteos, IDs/prompts únicos, familias, splits, claves y registry; la prueba está en [tests/test_prompt_comprehension_battery.py](tests/test_prompt_comprehension_battery.py). |
| Regeneración estructural determinista | **APROBADA** | El generador está en [bench/generate_prompt_comprehension_battery.py](bench/generate_prompt_comprehension_battery.py) y la prueba compara el resultado materializado en [tests/test_prompt_comprehension_battery.py](tests/test_prompt_comprehension_battery.py). Esto no demuestra todavía identidad canónica de todos los derivados. |
| Revisión semántica independiente y ciega de las 224 familias | **NO EJECUTABLE** | El packet ciego y el flujo de dossier/adjudicación ya existen, pero todavía no hay revisiones independientes ni decisiones aplicadas; la necesidad está documentada en [«Completar la revisión semántica humana»](#1-completar-la-revisión-semántica-humana). El autor o el modelo que genera la batería no puede suplir esa revisión. |
| Dataset nuevo, aprobado, inmutable y ligado por hash | **FALLIDA** | El único dataset principal es [prompt_comprehension_battery.draft.jsonl](bench/prompt_comprehension_battery.draft.jsonl) y permanece `draft`; no existe un dataset aprobado derivado de revisiones humanas. |
| Ausencia de leakage semántico entre calibración y validación | **NO EJECUTABLE** | La atomicidad estructural por familia sí se audita, pero el leakage conceptual requiere la revisión humana pendiente; 336/336 casos y splits balanceados no prueban independencia semántica. |
| Diversidad y representatividad suficiente por dominio | **FALLIDA** | Los 672 casos siguen reutilizando ocho escenarios-base por razones de control causal y el posible efecto plantilla continúa abierto en [«Aumentar diversidad»](#2-aumentar-diversidad-dentro-de-cada-dominio-si-la-revisión-encuentra-efecto-plantilla). El tamaño y balance de la muestra no demuestran representatividad. |
| Fiabilidad del instrumento | **NO EJECUTABLE** | No hay campaña repetida, acuerdo entre revisores ni resultados held-out. Por tanto, los 672 casos no justifican afirmar fiabilidad. |

**Cierre del hito A:** **FALLIDO/RESTRINGIDO**. La estructura está materializada y su auditoría
pasa, pero el instrumento no está aprobado ni validado semánticamente. No se declara
representatividad ni fiabilidad a partir del número de casos.

### Hito B — baseline raw trazable

| Puerta | Estado | Evidencia y motivo |
|---|---|---|
| Preflight que acepte exactamente `raw`, dataset aprobado, modelo/revisión explícitos y manifest nuevo | **APROBADA** | El [runner baseline](bench/run_prompt_comprehension.py) falla antes de llamar al proveedor salvo que la condición sea exactamente `raw`, los bytes coincidan con el SHA-256 aprobado, todos los casos estén aprobados, la ruta de modelo y revisión sea explícita y el manifest pueda reclamarse de forma exclusiva; los rechazos y la novedad se prueban en [tests/test_run_prompt_comprehension.py](tests/test_run_prompt_comprehension.py). |
| Manifest inmutable ligado a bytes del dataset y a modelo/revisión | **APROBADA** | El preflight calcula los hashes desde los bytes, persiste una reclamación exclusiva y exige la misma identidad de manifest, dataset, ledger y ruta al reanudar. El [manifest de ejemplo](bench/prompt_comprehension_run.provider.example.json) documenta el contrato, pero no representa autorización para una campaña real; la inmutabilidad está cubierta en [tests/test_run_prompt_comprehension.py](tests/test_run_prompt_comprehension.py). |
| Persistencia de cada intento antes de detener o reanudar | **APROBADA** | El [ledger durable](bench/run_prompt_comprehension.py) escribe una fila terminal canónica con error `provider_error` o `parse_error`, hace `flush` y `fsync` antes de continuar o parar y, al reanudar, rechaza ledgers truncados/mezclados y omite tuplas ya terminales. La parada por 429 y reanudación sin repetición se prueban en [tests/test_run_prompt_comprehension.py](tests/test_run_prompt_comprehension.py). |
| Auditoría desde bytes reales: alteración, truncado, duplicados, extras, mezcla y sustitución | **APROBADA** | El [auditor de campaña](bench/prompt_comprehension_campaign_audit.py) recalcula hashes desde los bytes y verifica el producto exacto caso × modelo. [tests/test_prompt_comprehension_campaign_audit.py](tests/test_prompt_comprehension_campaign_audit.py) prueba rechazos independientes de alteración, truncado, duplicado, extra, mezcla de revisión/manifest y sustitución de dataset/manifest. |
| Derivación reproducible con JSON canónico y Markdown idénticos | **APROBADA** | El [reporte](bench/prompt_comprehension_report.py) serializa JSON canónico y ordena todas las agrupaciones; [tests/test_prompt_comprehension_report.py](tests/test_prompt_comprehension_report.py) ejecuta dos derivaciones independientes desde los mismos brutos y compara JSON y Markdown byte a byte. |
| Análisis sin score global, con errores tipados y agrupación familia/escenario | **APROBADA** | El [reporte](bench/prompt_comprehension_report.py) no emite score global, separa `provider_error` y `parse_error`, conserva metadata del registry y registros por familia y escenario y marca grupos inconclusos como `evidencia insuficiente`; el contrato se prueba en [tests/test_prompt_comprehension_report.py](tests/test_prompt_comprehension_report.py). |
| Coste, rate limits y autorización explícita de la campaña | **NO EJECUTABLE** | No hay autorización ni presupuesto registrados. La inspección del entorno no encontró nombres de variables de credenciales de proveedor; no se inspeccionaron ni se exponen valores secretos. |
| Campaña cerrada con cada tupla planificada como éxito o fallo tipado | **NO EJECUTABLE** | No existe una campaña activa: el cierre administrativo del antiguo draft de 432 casos fue archivado como histórico porque contenía cero llamadas y no debe congelar el dataset durante authoring. |

**Cierre del hito B:** **NO EJECUTABLE/RESTRINGIDO**. Las garantías técnicas del baseline,
persistencia, provenance, auditoría y derivación están implementadas y aprobadas por las pruebas
enlazadas, pero no se ha ejecutado ninguna llamada de baseline. Siguen faltando un dataset aprobado
por revisión semántica independiente, la autorización explícita, el presupuesto y el acceso externo.
El cierre administrativo anterior se conserva sólo como evidencia histórica en
`bench/runs/prompt-comprehension-closure/`; sus 1.296 filas registran cero llamadas externas y no son
parte del instrumento activo. Hasta cerrar el hito A y obtener las autorizaciones externas,
cualquier resultado de esta línea se clasificará explícitamente como **exploratorio/restringido**.
