# Spec-Elaboration Loop — diseño

## El problema que resuelve

Un modelo SOTA, ante un requirement vago, corre un loop **interno** y silencioso:
analiza lo pedido → encuentra gaps → los completa → critica su idea → verifica →
descarta → converge a un spec más completo que lo que se pidió literalmente, y
**recién entonces** desarrolla.

Un modelo chico **se saltea ese loop**: toma el requirement genérico tal cual y
planifica sobre él, sin considerar consecuencias, estado actual del proyecto,
teoría ni dificultad. No es (solo) falta de conocimiento — es la **ausencia del
loop metacognitivo**.

La capa de contención (workflow 1) y la de capacidad (workflow 2, Recipe Bank +
tier router) NO atacan esto: resuelven "el modelo no *sabe* buenos diseños". El
spec-elaboration loop ataca lo **upstream**: "el modelo no *elabora* el
requirement vago en un spec fundamentado". Va **antes** de elegir el diseño, y
mejora todo lo que viene después (un spec rico = mejor retrieval de recetas,
mejor handoff al planner, mejor descomposición en steps).

### Principio rector

> **El harness es dueño del loop; el modelo solo llena cada casilla.**

Cada pasada es una tarea acotada de slot-filling que un 7B sí puede resolver. La
inteligencia no está en pedirle al modelo "pensá mejor" (no funciona), sino en
que la *estructura del loop* viva en el engine y cada gap se resuelva **contra
evidencia, no contra la imaginación del modelo**.

---

## Mapeo directo al loop del SOTA

| Lo que el SOTA hace internamente | Pasada del loop (externalizada) |
|----------------------------------|---------------------------------|
| "analiza muy bien lo pedido"     | **P0 — Restatement & scope split** |
| "busca gaps"                     | **P1 — Gap enumeration (tagged)** |
| "los completa"                   | **P2 — Resolución contra evidencia** + **P3 — Escalación de gaps de producto** |
| "critica su idea, verifica, descarta" | **P4 — Autocrítica & descarte** |
| "viene con un diseño más completo" | **P5 — Converge → GroundedSpec** |
| "desarrolla eso"                 | planner + LoopEngine consumen el GroundedSpec |

---

## La frontera crítica: gap técnico vs gap de producto

Esto es lo que separa **"completar el spec como un SOTA"** (bueno) de
**"el modelo se inventó tu producto"** (prohibido por el principio del proyecto:
*el modelo nunca decide producto, el producto es del usuario*).

Cada gap detectado en P1 se **etiqueta** por tipo, y el tipo decide quién lo resuelve:

| Tipo de gap | Ejemplo ("agregá rate-limiting al API") | Quién lo resuelve |
|-------------|------------------------------------------|-------------------|
| `technical` | ¿ya hay middleware? ¿el endpoint es sync/async? ¿qué storage? | **El loop, autónomo, contra el código** (read_file, code_search, Recipe Bank) |
| `theory`    | ¿token bucket vs sliding window? ¿jitter/cap? | **El loop, contra evidencia externa** (retrieval/web o tier fuerte), con cita |
| `product_intent` | ¿per-user o global? ¿qué límite? ¿bloquea o encola? | **Se ESCALA al usuario** — nunca se inventa |

El loop rellena lo técnico/teórico autónomamente; **junta** los gaps de producto
y los devuelve al usuario en **una sola ronda** (no goteo). Default razonable +
assumption explícita cuando el gap de producto NO cambia el diseño (así no se
vuelve molesto); escala solo cuando un gap de producto *cambiaría* el diseño.

---

## Las 6 pasadas en detalle

Cada pasada tiene salida estructurada (schema FC) y un presupuesto de tokens acotado.

### P0 — Restatement & scope split  *(barato, local-OK)*
- **Input**: `user_request` (verbatim) + `understanding` (del chat agent).
- **Output**: `{ in_scope[], out_of_scope[], deliverable }`.
- **Por qué**: el `out_of_scope[]` es la palanca anti-sobre-generalización — obliga
  a nombrar lo que NO se pide, justo lo que el chico nunca delimita.

### P1 — Gap enumeration  *(el corazón)*
- **Output**: `gaps[]`, cada uno `{ question, kind: technical|theory|product_intent, why_it_matters }`.
- **Por qué**: es el paso que el SOTA hace y el chico se saltea. El etiquetado
  materializa la frontera técnico/producto.

### P2 — Resolución contra evidencia  *(grounding)*
El lift de esta pasada NO viene del modelo — viene del **entorno**. El mismo
modelo configurado lee evidencia real y la incorpora; eso inyecta conocimiento
que no está en sus pesos sin necesitar otro modelo.
- Para cada gap `technical`: resolver **leyendo el código** (code_search / read_file /
  code-intel) y/o **Recipe Bank**. Salida: `{ question, answer, evidence: "file:line", confidence }`.
  Si no se resuelve con evidencia → se degrada a **assumption UNVERIFIED explícita**
  (nunca se inventa en silencio).
- Para cada gap `theory`: resolver por **retrieval/web** (capa Inteligencia) — el
  conocimiento sale de la fuente externa, el modelo solo la lee y la cita. Sin
  evidencia encontrada → flag explícito (no se fabrica).
- Para cada gap `product_intent`: **NO resolver**; encolar en `clarifications_needed[]`.

### P3 — Escalación de gaps de producto  *(la frontera del usuario)*
- Si `clarifications_needed[]` tiene gaps **materiales** (que cambian el diseño) →
  **escalar al usuario**: presentar las preguntas en una sola ronda y pausar.
- Gaps de producto NO materiales → default sensato + assumption surfaced (no bloquea).
- Reusa el seam `escalate`/`respond` que ya devuelve control al usuario.

### P4 — Autocrítica & descarte  *(la pasada que el chico omite — pero el descarte lo hace la ejecución, no el modelo)*
Un modelo chico no se autocritica bien (arXiv:2404.17140: necesita un *verificador
fuerte* externo). Acá el verificador fuerte es **determinístico**, no un segundo
LLM: ejecución de tests, existencia de símbolos/archivos vía code-index,
type-check. El modelo **propone**; el **entorno descarta**.
- **Generar** (mismo modelo): muestrear N direcciones de diseño candidatas
  (best-of-N / PLANSEARCH sobre el spec ya grounded).
- **Descartar (determinístico)**: cada candidata se filtra por checks ejecutables
  baratos —¿los archivos/símbolos que asume existen en el code-index? ¿el
  `verification_command` propuesto es real? ¿un dry-run rompe algo?— y por
  **self-consistency** (clusterizar las N por embedding, quedarse con el consenso).
- **Output**: `risks[]` (de lo que los checks revelaron) + `alternatives_rejected[]`
  (las candidatas que fallaron un check, con el check que las mató — descarte
  *demostrable*, no opinión del modelo).
- Funciona con **cualquier** modelo único: la generación es del modelo configurado,
  el descarte es código.

### P5 — Converge → GroundedSpec
- Emite el `GroundedSpec` final (ver schema). Se cuelga de `escalation.grounded_spec`
  y se renderiza en el handoff del planner. El planner ahora descompone un spec
  **completo**, no el requirement crudo.

---

## Artefacto: `GroundedSpec`

```python
@dataclass(frozen=True)
class GroundedSpec:
    restatement: dict          # {in_scope[], out_of_scope[], deliverable}
    resolved_facts: list       # [{question, answer, evidence, confidence}]
    assumptions: list          # [{statement, why_no_evidence, reversible: bool}]
    clarifications: list        # [{question, answer}]  (si hubo ronda con el usuario)
    design_direction: str      # 1 párrafo: enfoque elegido
    alternatives_rejected: list # [{alternative, why_rejected}]
    risks: list                # [str]
    open_questions: list       # residual honesto
    signature_text: str        # clave de retrieval para Recipe Bank (rica, no el request crudo)
    evidence_count: int        # cuántos facts tienen evidencia real (vs assumptions) — señal de grounding
```

---

## Integración en el pipeline

```
run_chat_agent → escalate
      ↓ EscalationPacket
_run_elaboration_phase  ← NUEVO (engine/analysis/spec_elaborator.py)
      ↓ grounded_spec colgado del packet  (dataclasses.replace, packet es frozen)
      │  └─ si P3 produce clarifications materiales → respond al usuario y fin de turno
_run_council_phase  (opcional, ahora delibera sobre un spec grounded)
      ↓
run_planner  ← _render_handoff renderiza el GroundedSpec (como hoy hace con design_brief)
      ↓
LoopEngine.execute  →  Review
```

**Seams concretos (archivos reales):**

| Componente | Archivo | Cambio |
|------------|---------|--------|
| `GroundedSpec` dataclass | `engine/analysis/grounded_spec.py` (nuevo) | — |
| Campo en el packet | `engine/orchestration/escalation_packet.py` | `grounded_spec: Any \| None = None` (espejo de `design_brief`) |
| Loop | `engine/analysis/spec_elaborator.py` (nuevo) | `elaborate(escalation, *, strong_tier, ...) -> GroundedSpec \| ClarificationRequest` |
| Fase en el pipeline | `engine/orchestration/pipeline.py` | `_run_elaboration_phase` entre escalación y `_run_council_phase` (~L511) |
| Render en handoff | `engine/analysis/planner.py` `_render_handoff` (~L202) | bloque `GroundedSpec` (reusa el patrón `brief.render_for_planner()`) |
| Verificador determinístico (P4) | `engine/analysis/verification_engine.py` + code-index | checks ejecutables que descartan candidatas — NO un segundo LLM |
| Clave de retrieval | Recipe Bank (workflow 2) | `signature_text` del GroundedSpec en vez del request crudo |

**Todas las llamadas LLM del loop usan `get_litellm_params()` — el único modelo
configurado.** No hay slot de modelo aparte, no hay routing por capacidad.

---

## De dónde sale el lift  (un solo modelo, cualquiera)

**Restricción dura: toda la solución corre sobre el único modelo configurado**
(z.ai, deepseek, minimax, Ollama, lo que sea). No hay dos LLMs, no hay tier
router. El loop es **idéntico** para todos los providers. Entonces el lift NO
puede venir de "un modelo más capaz" — viene de cuatro fuentes que funcionan con
cualquier modelo:

| Fuente | Cómo sube el techo | Pasadas |
|--------|--------------------|---------|
| **Estructura** | Descomponer en pasadas de slot-filling acotado da mejor resultado que one-shot, *con el mismo modelo* (refinamiento secuencial / least-to-most). | P0–P5 |
| **Evidencia** | El conocimiento sale del *entorno* (código real, retrieval, web), no de los pesos. Cualquier modelo puede leer y citar. | P2 |
| **Verificación determinística** | El descarte lo hace la ejecución/code-index, no la opinión del modelo. El "verificador fuerte" que el modelo necesita es código. | P4 |
| **Memoria** | Recipe Bank: diseños pasados ya verificados por tests. Conocimiento acumulado, no del modelo. | P2, P5 |

El modelo configurado puede ser muy capaz (deepseek, z.ai) o débil (7B local): el
loop ayuda **más** cuando el modelo es débil, pero **nunca depende** de que haya
uno fuerte disponible. Mismo código para todos.

---

## Control de costo (compone con la capa de contención)

- **Once per task**, no per step — amortizado.
- **Gateado por complejidad**: requests triviales (claros, single-file, baja ambigüedad)
  **saltean** la elaboración (clasificador barato, reusa la heurística del trigger del
  council). Solo "underspecified / multi-file / científico" paga el costo.
- **ContextGovernor** (contención) mantiene cada pasada barata e inyectable; el
  GroundedSpec se poda al renderizar en el handoff.
- **Recipe Bank short-circuit**: si ya existe un GroundedSpec casi-idéntico, se adapta
  en vez de re-elaborar.
- Latencia: se paga en la **fase de diseño** (antes de la primera línea de código),
  que es exactamente donde el usuario tolera que el sistema "piense".

---

## Validación (experimentos falsables)

1. **Calidad de plan** sobre un benchmark de requirements deliberadamente vagos:
   ¿el plan emitido aborda los gaps? — elaboración ON vs OFF.
2. **Precisión de clarificación**: % de preguntas de producto que el usuario contesta
   como materiales (que no sea molesto). Objetivo: alta precisión, pocas preguntas.
3. **Grounding rate**: % de `resolved_facts` con evidencia real (file:line) vs
   assumptions UNVERIFIED. Mide que no se inventa.
4. **Rework rate** downstream: % de steps que fallan/se rehacen, ON vs OFF.
5. **Mejora de retrieval**: hit rate del Recipe Bank con `signature_text` del
   GroundedSpec vs el request crudo.

---

## Riesgos / honestidad

- **Clarification = suspend/resume.** v1 simple: P3 hace ask-and-end-turn (rida el
  turn-loop existente); el GroundedSpec parcial se persiste (session/DB) y el próximo
  turno lo retoma con las respuestas. v2 (más rico): suspend/resume in-pipeline — más
  trabajo, diferir.
- **El techo real es el del modelo configurado + la evidencia disponible.** Un gap
  que no está en los pesos del modelo, NI es resoluble contra el código/repo, NI se
  encuentra en la web, NI existe en el Recipe Bank → queda genuinamente sin resolver.
  El loop NO lo fabrica: lo surfacea como assumption UNVERIFIED o lo escala. Esa es la
  frontera honesta — con un solo modelo no hay forma de inventar capacidad que no esté
  en ninguna de las cuatro fuentes.
- **P4 vale lo que valgan los checks ejecutables.** El descarte determinístico solo
  atrapa lo que un check barato puede detectar (símbolo inexistente, archivo ausente,
  dry-run roto). Errores de diseño sutiles que pasan los checks no se descartan. Es
  mejor que la autocrítica del modelo (que no funciona), pero no es omnisciente.
- **Gating mal calibrado**: clasificar un request vago como trivial se saltea la
  elaboración (falso negativo). Tunear conservador (ante la duda, elaborar).
- **Costo de la primera respuesta**: agrega latencia antes del primer código. Mitigado
  por el gating + once-per-task, pero es un trade-off consciente.
