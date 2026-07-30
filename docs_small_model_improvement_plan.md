Las referencias clave coinciden con el código real (max_iterations=6, max_exploration_calls=4, OLLAMA_NUM_CTX=16384, emit_plan_tool.py 79 líneas, engine.py 1832, pipeline.py 659). Procedo a escribir el informe.

# INFORME FINAL — Infinidev: cuatro problemas de modelos chicos, un plan integrado

## 1. Resumen ejecutivo

Los cuatro problemas (contexto, sobreconfianza, diseño, inteligencia) tienen la **misma causa raíz arquitectónica**: el control crítico está delegado en la disciplina del modelo chico (que obedezca avisos, tome notas, pida ayuda, investigue) justo donde el modelo chico falla. La tesis correctiva, validada en los cuatro planes, es una sola: **sacar la decisión del modelo y ponerla en código determinista engine-side, siempre activo para `ctx.is_small`**. La buena noticia es que convergen en 5 puntos del código (`planner.py`, `emit_plan_tool.py`, el bloque `step_complete` de `engine.py`, el render de `context.py`, el trigger de council en `pipeline.py`); el riesgo es que, implementados en aislamiento, **tripliquen el costo de retry del planner y de tokens del prompt**. La regla integradora es: un único punto de gate componible, un único refactor de schema, y todo bloque nuevo gobernado por un ContextGovernor. Orden recomendado: **1-Contexto(A) → 2-Sobreconfianza → 3-Diseño → 4-Inteligencia**, porque cada nivel construye la infraestructura que el siguiente reusa.

---

## 2. Problema por problema

### 2.1 Contexto — deriva y pérdida con ventana grande/ruidosa en modelos chicos

**Causa raíz (verificada en código).** El "smart context" sabe qué *inyectar* (ContextRank, opened_files, notes) pero le falta la mitad de *control de tamaño*. Concretamente:
- `build_iteration_prompt` (`context.py:618`) une bloques con strings, sin medir ni recortar tokens antes de enviar.
- El único mecanismo de budget, `<context-budget>` (`context.py:865-895`), es **advisory** ("You MUST wrap up") y depende de que el modelo obedezca — lo que un modelo chico bajo deriva no hace.
- El budget mide contra el `context_length` de entrenamiento de `/api/show` (`model_context.py:50`, vía `_get_model_max_context` en `engine.py:767`) e **ignora `OLLAMA_NUM_CTX=16384`** (`settings.py:92`, ya cableado a Ollama). Un modelo entrenado a 128K corrido a `num_ctx=16384` cree tener 128K → **Ollama trunca en silencio**, sin error → deriva.
- `<context-budget>` usa `state.last_prompt_tokens` (`llm_caller.py:522`): los tokens de la iteración **anterior**, siempre un paso atrasado.
- El único recorte intra-step, `compact_for_small` (`context_manager.py:55-89`, gated por `is_small` en `engine.py:1669`), trunca ciego a `content[:200]`, **destruyendo las líneas de error de pytest/tracebacks** que van al final. Modelos no-chicos no tienen tope por-tool-result.
- `opened_files` puede crecer a 10 archivos × 32000 chars (~80K tokens) y los pinned están **exentos de eviction LRU** (`loop_state.py:107-112`).
- La anti-amnesia descansa en que el modelo llame `add_note`; `_auto_enhance_record` (`step_manager.py:26-67`) ya captura hechos reales pero es small-model-only (`engine.py:197`) y débil.

Resultado: el modelo chico no "deriva por ruido" tanto como **(a)** se queda sin ventana porque nada poda y Ollama trunca en silencio, o **(b)** sufre amnesia porque no tomó notas y el reset por-step descartó todo.

**Enfoque elegido.** Combinación de las tres propuestas en **3 increments por riesgo**. Columna vertebral: un **ContextGovernor determinista** que recorta de verdad pre-envío (no advisory). Principio rector: el engine garantiza que el prompt cabe y captura los hechos automáticamente; el modelo nunca decide cuánto contexto cabe.
- **INCREMENT A (bajo riesgo, captura la mayor parte del impacto):** budget honesto + cap universal de tool-results + ledger determinista.
- **INCREMENT B (riesgoso, transformador):** ContextGovernor con poda priorizada detrás de golden-file tests.
- **INCREMENT C (opt-in, medido):** colapso de opened_files a outline + `/compact` forzado como último recurso.

**Alternativas descartadas.**
- `<context-budget>` advisory como control → degradado a info pura. Pedirle a un modelo chico bajo deriva que haga `step_complete` es justo lo que ignora.
- `SmartContextSummarizer` (`summarizer.py`): 100% regex-en-inglés, mide budget en chars no tokens (línea 173), y se **desactiva justo para modelos chicos**. Reemplazado por el ledger determinista.
- Inventar `LLM_NUM_CTX` / `INFINIBAY_LLM_NUM_CTX`: **`OLLAMA_NUM_CTX` ya existe** y está cableado. Reusar, no duplicar.
- Truncado ciego `content[:200]` / `[:80]`: destruye errores. Reemplazado por head+tail con regex de errores (lost-in-the-middle, Liu 2023).
- Bloque `<observed-facts>` separado: añade otro bloque que compite por los mismos tokens. La captura vive dentro del ledger/ActionRecord, que ya se renderiza.
- Colapso de pinned recién editados → diferido a opt-in: son los archivos que el modelo **acaba de escribir**; colapsarlos fuerza re-lecturas que gastan la ventana que se quiere proteger.

**Pasos concretos (archivos).**
- **A1 — Budget honesto** (`model_context.py` `_get_model_max_context`, `engine.py:767`, `settings.py`): `max_context_tokens = min(context_length, OLLAMA_NUM_CTX) − headroom`. Headroom conservador **25-30%** (nuevo `CONTEXT_BUDGET_HEADROOM_PCT=0.25`) porque `litellm.token_counter` es impreciso para GGUF. Si Ollama no expone `context_length`, caer a `OLLAMA_NUM_CTX`.
- **A2 — Medición real** (nuevo `context_governor.py: count_tokens`, `context.py:865-895`): `litellm.token_counter` con fallback `chars/4`. `<context-budget>` deja de usar `last_prompt_tokens` y mide el prompt que **se va a enviar**; pasa a info pura (se elimina "You MUST wrap up").
- **A3 — Cap universal de tool-results** (`context_manager.py: cap_tool_result`, quitar gate `is_small` en `engine.py:1669`, `settings.py: LOOP_TOOL_RESULT_CAP≈2500`): head N líneas + tail M líneas + inyección de líneas que matchean `Error|Traceback|FAILED|Exception|^E\s|assert`. Aplicado en `_execute_regular_tools` para **todos** los modelos.
- **A4 — Ledger determinista siempre-activo** (`step_manager.py: _auto_enhance_record`, `action_record.py`, `behavior_rules.py`, render en `context.py`): quitar gate `is_small` (`:197`) y guarda `and not record.X`. Mapa per-tool **explícito**: `read_file`→path+rango; `edit_symbol/add_symbol`→símbolo (¡usa arg de símbolo, NO `file_path`!); `execute_command`→comando+exit_code+últimas 2 líneas; error→primera línea con `Error/Traceback`. Tope por recencia (~30 entradas con dedup). Render tabular estable.
- **B1 — Refactor a `list[PromptBlock]`** (`context.py:618`, `context_governor.py`, golden-file tests): **primero** escribir golden-file tests del prompt actual; luego `build_iteration_prompt` devuelve bloques con prioridad fija (P0 inmutable=`task`/`current-action`/`expected-output`/`plan`; P1=notes/ledger/context-rank; P2=opened-files; P3=previous-actions antiguas). Preservar `CACHE_BREAKPOINT_MARKER`.
- **B2 — `fit_to_budget`** (`context_governor.py`, `engine.py` entre ensamblado y `llm_caller`): suma P0 siempre; agrega P1..P3 mientras quepan; shrink escalonado (previous-actions → StreamingLLM anchor+evict-middle, no FIFO ciego). **Assert duro: nunca se envía prompt > max_tokens.**
- **C1 — Colapso opened_files a outline** (`loop_state.py`, `opened_file.py`, `context.py`, flag `CONTEXT_COLLAPSE_OPENED_FILES=False`): solo archivos unpinned viejos **Y** pinned no tocados en los últimos K tool calls **Y** no referenciados por el step activo. Regla dura.
- **C2 — `/compact` forzado** (`engine.py`, `context_governor.py`): solo si tras todo el shrink P0 ya excede budget. Loguear antes/después; nunca darlo como tool al modelo en este increment.

**Métricas de éxito.**
- Crashes `ContextWindowExceeded` / truncado silencioso en suite de tasks largas (30-50 steps): de N>0 a **0** tras A+B.
- p95 de `prompt_tokens` **siempre** < `OLLAMA_NUM_CTX*(1−headroom)` (assert duro).
- Re-lecturas redundantes (mismo path 2+ veces sin edit): debe **bajar** con el ledger.
- Preservación de error: % de runs donde la línea de error sobrevive al cap, de ~0% a **>95%**.
- Cobertura del mapa per-tool del ledger: **100%** en test, incluyendo `edit_symbol` (arg símbolo) y `execute_command` (sin `file_path`).
- Golden-file: **0 diffs** no intencionales tras B1.

**Preguntas abiertas.**
- Precisión de `litellm.token_counter` para GGUF arbitrarios (puede errar 20-40%): ¿el headroom de 25-30% lo absorbe, o hay que usar el tokenizer de Ollama? Medir antes de fijar.
- ¿`num_ctx` efectivo si el usuario corre Ollama con otro valor? Por ahora se confía en `OLLAMA_NUM_CTX` como contrato; documentar la suposición.
- Tope del ledger (~30) es adivinado; tuning empírico.
- Fidelidad del `/compact` forzado en modelos chicos: el mismo modelo que escribe summaries pobres compacta pobre. Solo anti-crash hasta medir.
- Interacción Governor ↔ ContextRank: ContextRank **agrega** tokens, el Governor **poda**. Definir prioridad de los bloques de context-rank (P1) vs opened_files (P2).

---

### 2.2 Sobreconfianza — modelos chicos sobreestiman su capacidad de diseñar/construir

**Causa raíz.** La defensa existe pero está desconectada o fuera del hardware objetivo:
- `critic_dispatcher.py` (lentes `FALSE_CONFIDENCE`/`PREMATURE_CLOSURE`) y `reasoning_pattern_detector.py` (`victory_lap`) están implementados y testeados pero **nunca se cablean en `engine.py`** (grep da 0 call sites de producción).
- El único revisor semántico vivo, `AssistantCritic`, está **OFF por defecto** (`ASSISTANT_LLM_ENABLED=False`) y asume una segunda GPU — justo lo que el target single-GPU no tiene; sería "ciego juzgando a ciego".
- Lo único activo son guardas deterministas (`LoopGuard`) que reaccionan a síntomas **mecánicos** (repetición, loops de test) **después** de atascarse, y a veces "resuelven" forzando `step_complete` (convirtiendo "atascado por sobre-alcance" en falso "done").
- **No existe ninguna compuerta de alcance/complejidad:** el planner acepta cualquier plan sin tope de steps; `_parse_emitted_plan` solo valida overview/steps no-vacíos. El caso clásico —**fix confiadamente erróneo con output limpio**— pasa todas las compuertas porque ninguna exige **evidencia positiva** de verificación; solo reaccionan a la presencia de un error.

**Enfoque elegido.** Combinación determinista en tres capas (aritmética/lookup/exit-codes, 0 GPU extra, default ON para small models):
- **(A) EVIDENCE GATE** en `step_complete`: invariante estructural anclado en `tracker.files_edited` (NO en regex de frases). Si el step editó archivos y cierra `done` **sin** evidencia ejecutada *después del último edit* (un `execute_command`/`code_interpreter`, un test, o un `read_file` del path editado) dentro de la ventana del step, se bloquea **una vez** reusando `_overwrite_step_complete_tool_result`+`continue` (el mismo mecanismo del critic y el note-gate). **Invierte la compuerta: la ausencia de prueba bloquea.**
- **(B) SCOPE GATE plan-time** duro pero acotado: cap de número de steps (única defensa robusta porque no depende de auto-reporte) + blocklist léxica de verbos grandilocuentes cuando `steps≤2`; rechaza-con-reparación vía 1 retry, luego degrada a fallback.
- **(C) SCOPE BUDGET runtime:** nudge cuando un step edita > N archivos distintos reales (sobre `tracker.files_edited`).

**Alternativas descartadas.**
- Cablear `select_checks` de `critic_dispatcher.py` como "selector": verificado en `critic_dispatcher.py:152-154`, agrega `PREMATURE_CLOSURE`+`FALSE_CONFIDENCE` **incondicionalmente** en cualquier `step_complete` → trigger siempre-verdadero, sin selectividad. La lógica útil (invariante edit→evidencia) se escribe de cero.
- Evidence gate por regex de frases de éxito ("should work", "this fixes"): frágil, evadible, y **muchos modelos chicos no verbalizan confianza** (summary telegráfico ~50 tokens). Deja escapar el cierre silencioso-pero-erróneo.
- Auto-crítica forzada del plan: Huang et al. 2024 — un modelo chico criticando su propio plan tiende a confirmarlo. Añade latencia LLM por valor dudoso.
- `kind='design_decision'` → escalada a Council: depende de auto-etiquetado honesto (la metacognición que falta) **y** mecánicamente inexistente (Council corre **upstream** del planner, sin back-edge).
- Cap por `target_paths` declarados → **degradado a WARNING** (no rechazo): auto-reportado por el modelo mal calibrado (declara 2, toca 9).
- `ReasoningPatternDetector` (`victory_lap`): depende de tabla DB que puede estar vacía y solo sugiere al critic (apagado). Fuera del MVP.

**Pasos concretos (archivos).**
1. Crear módulo puro `engine/loop/scope_gate.py` (testeable sin GPU, estilo `test_critic_dispatcher.py`).
2. **EVIDENCE GATE** en `engine.py` (bloque `step_complete` ~1118-1273, antes de `_parse_step_complete_args` en 1273): si `tracker.files_edited` no vacío Y `not step_has_post_edit_evidence(...)` Y `not self._evidence_gated` → bloquear con mensaje accionable, `continue`. Steps de pura lectura exentos.
3. **Parseo robusto de evidencia** (`engine.py`, `scope_gate.step_has_post_edit_evidence`): leer la secuencia tipada de tool calls desde `step_messages_start`, identificar índice del último edit, verificar evidencia posterior **por nombres de tool y paths estructurados, NO regex sobre texto renderizado**. `execute_command` con exit≠0 cuenta como intento (el error visible lo cubre el breaker existente).
4. **SCOPE GATE plan-time** (`planner.py` `_parse_emitted_plan:239-264` + `_run_llm_loop:157-159`): `len(steps) > SCOPE_MAX_STEPS(6)` → rechazar; blocklist léxica cuando `steps≤2` → rechazar. Reinyectar 1 mensaje concreto, permitir 1 retry, luego `_fallback_plan(steps=[])`.
5. **WARNING plan-time** por `count_targets > SCOPE_MAX_TARGETS_PER_STEP(3)`: anota el detail, **no rechaza** (advisory).
6. **SCOPE BUDGET runtime** (`engine.py` `_finalize_inner_loop` ~1308): nudge-once sobre `count_distinct(tracker.files_edited) > 3`. **No fuerza `step_complete`.**
7. **Settings** (`config/settings.py`): `SCOPE_GATE_ENABLED`, `SCOPE_EVIDENCE_GATE_ENABLED`, `SCOPE_MAX_STEPS=6`, `SCOPE_MAX_TARGETS_PER_STEP=3` (prefijo `INFINIBAY_`, default ON, gateado por `ctx.is_small`).
8. Alinear prompt del planner (`planner_prompt.py`): el límite de 6 steps es **duro**, no sugerencia.
9. Prompt del developer (`prompts/shared.py` `LOOP_PROTOCOL`): "Si un step modificó archivos, no lo cierres sin ejecutar/releer/testear lo editado" (restricción concreta y negativa).

**Métricas de éxito.**
- % de `step_complete done` sobre steps que editaron archivos **sin** evidencia post-edit: caer a **~0**.
- Falsos positivos del evidence gate: % de bloqueos donde la verificación era innecesaria; vigilar `<X%` de steps legítimos bloqueados.
- Distribución de `len(steps)`: cola larga (>6 y monolíticos 1-2 grandilocuentes) **eliminada**, mediana sin cambios.
- Caso "fix limpio pero erróneo": batería con fix plausible-incorrecto; detección **sube** vs baseline.
- Tasa de `_fallback_plan(steps=[])`: vigilar que **no aumente** (no convertir planes válidos en no-plan).

**Preguntas abiertas.**
- Umbrales mágicos (`SCOPE_MAX_STEPS=6`, `MAX_TARGETS=3`) sin validación: un refactor multi-archivo legítimo puede chocarlos. Calibrar sobre tareas reales.
- **Gaming de la evidencia:** el gate exige evidencia-de-*ejecución*, no de-*corrección*. Un modelo puede satisfacerlo con `echo ok && exit 0` o un reread trivial. El determinismo compra ejecución, no corrección — límite honesto del enfoque.
- ¿Reread cuenta como evidencia suficiente o solo `command`/`test`? Reread ≠ verify.
- **Interacción con LoopGuard:** un breaker que fuerza `step_complete` por estancamiento podría convertir un step bloqueado-por-falta-de-evidencia en falso `done`. Definir precedencia (el evidence gate debe correr **antes** del forzado).
- Plan-gate con solo 2 defensas duras (cap de steps + blocklist), ambas evadibles reformulando. ¿Resolver paths del detail contra el índice tree-sitter para estimar alcance real?

---

### 2.3 Diseño — planes malos/incompletos por planner de un solo pase

**Causa raíz.** El diseño se produce en un único pase de un solo agente, sin gate de calidad ni obligación de fundamentar:
- En `planner.py` el primer `emit_plan` termina el turno (`L158-159`); el único check (`_parse_emitted_plan`, `L258-263`) es overview/steps no-vacíos.
- El schema `EmitPlanInput` (`emit_plan_tool.py`, 79 líneas) **no tiene** campos de assumptions/alternatives/rationale/risks → el modelo **nunca** está obligado a externalizar su razonamiento (el agujero exacto del exceso de confianza).
- El loop optimiza **costo** ("emit the plan NOW with whatever you have", `L176-186`), lo opuesto a forzar exploración.
- El único mecanismo riguroso (Council) está gated detrás de `escalation.council_requested`, bandera que pone el mismo modelo sobreconfiado (default False).
- `_looks_concrete` de `plan_tools.py` es "a nudge, not a gate" (comentario `L18`), solo sintáctico, y ni se aplica al analista.

**Enfoque elegido.** Generator-verifier con el crítico **fuera del LLM** (código determinista), citando Huang et al. ICLR 2024 ("LLMs cannot self-correct without external feedback"): un modelo chico no se autoevalúa pero **sí** rellena slots de schema y reacciona a una lista concreta de fallos. Cuatro piezas ortogonales:
1. **SCHEMA-AS-CONTRACT:** `EmitPlanInput` gana campos **obligatorios**: `assumptions:list[{statement,evidence}]`, `alternatives_considered:list[{approach,why_rejected}]` (≥1), `rationale:str`, `open_risks:list[str]`. Reusar el dataclass `Alternative` ya existente en `council/brief.py`.
2. **GATE DETERMINISTICO INLINE** sobre la única llamada `emit_plan` (NO segundo terminador): evidence vacío/`UNVERIFIED` → fallo; <1 alternativa sustantiva → fallo; cada step title pasa `_looks_concrete` (ahora aplicado como **gate** en el analista); `expected_output` con token verificable.
3. **UNA SOLA RONDA DE CRÍTICA** acotada (Reflexion/Self-Refine): si el gate falla, inyectar mensaje con fallos concretos + 1 reintento (`REVISION_BUDGET=1`, **desacoplado** del budget de exploración). Segundo emit: fail-open con warning, nunca cuelga el turno.
4. **TRIGGER OBJETIVO DEL COUNCIL:** en `pipeline.py`, `_should_convene_council(escalation)` por señales deterministas (nº opened_files/scope > N, keywords de diseño abierto ES/EN/PT). Elimina la dependencia del autodiagnóstico.

**Alternativas descartadas.**
- Trigger "gate-fail 2× → escalar a council" (propuesta 1): **fallo de feasibility verificado** — `pipeline.py` corre `_run_council_phase` (`L521`) **estrictamente antes** de `run_planner` (`L531`) sin loop de retorno. Cuando el planner gatea, la fase council ya terminó.
- Dos terminadores (`emit_plan` draft + `verify_plan`): duplica la superficie de confusión para modelos chicos que ya luchan por llamar UNO; re-emitir el plan completo es justo donde los 7B truncan JSON. Checks **inline** sobre la única llamada.
- Assumption-coverage como check load-bearing inmediato: el loop solo incrementa `exploration_calls` (`L163`), **nunca registra qué paths se abrieron**. El accumulator de `read_paths` no existe → diferido a fase 2.
- Subir budget de exploración a 6 + max_iterations a 9 agresivo: choca con `max_iterations=6`/4-call y puede caer al `_fallback_plan` (peor que el plan imperfecto). Solo subir `max_iterations` lo justo para absorber 1 revisión.
- "assumptions ya probado por DesignBrief": **verificado en `brief.py:87-96`** — DesignBrief tiene `alternatives_considered`/`open_risks`/`rationale` pero **NO** `assumptions`. Es campo nuevo.
- Check de cohesión por substring: trivialmente gameable o frágil ante paráfrasis. Omitido.
- **Límite honesto:** el gate mide **forma, no verdad**. Eleva el piso (no más planes vacíos/sin alternativas) pero no garantiza soundness. El mecanismo load-bearing real es "ancla cada supuesto a leer el archivo que lo prueba".

**Pasos concretos (archivos).**
1. `tools/planner/emit_plan_tool.py`: pydantic `Assumption{statement,evidence}` y `Alternative{approach,why_rejected}`; extender `EmitPlanInput`. Obligatorios en FC-mode, **tolerantes en el parser** para manual-TC.
2. `tools/meta/_plan_checks.py` (nuevo): promover `_looks_concrete` + `_has_verifiable_token` desde `plan_tools.py` (developer conserva el nudge).
3. `engine/analysis/plan_gate.py` (nuevo): `check_plan(args, *, opened_files, read_paths=None, workspace_path) -> list[str]`. Exigir alternativas solo cuando `steps>2` (evita falsos positivos en edits triviales).
4. `engine/analysis/planner.py`: cablear `check_plan` inline en `_run_llm_loop` al **primer** `emit_plan` (en vez del `return` en `L158-159`); si falla y `REVISION_BUDGET` no consumido, inyectar crítica estilo CoVe + `continue`; segundo emit fail-open. Subir `max_iterations` a ~8 **solo** para la revisión (`max_exploration_calls=4` intacto).
5. `engine/analysis/plan.py` + handoff: `Plan`/`PlanStepSpec` ganan `assumptions`/`open_risks`; el handoff expone solo los `UNVERIFIED` + risks **una vez** (no en el `<plan-overview>` reconstruido cada iteración — anti-bloat de CLAUDE.md).
6. `prompts/analyst/planner_prompt.py`: quitar "emit NOW with whatever you have"; añadir "## Fundamentación obligatoria" con protocolo numerado de rellenar-casillas (supuestos+archivo que los prueba, ≥1 alternativa, riesgos, luego emit).
7. `engine/orchestration/pipeline.py`: `_should_convene_council(escalation)` por señales objetivas; `_run_council_phase` deja de exigir solo `council_requested`. **Drop** del sub-trigger gate-fail-2x.
8. *(Fase 2, diferida)* Accumulator de `read_paths` en `_run_llm_loop` (parsear arg de path por-tool) + assumption-coverage contra paths realmente abiertos.

**Métricas de éxito.**
- `emit_plan` con assumptions no-vacías y ≥1 alternativa: de **0%** a **≥80%** con qwen2.5-coder:7b.
- % de planes rechazados al 1er intento que **pasan tras 1 revisión** (mide que la crítica es accionable).
- Tasa de `_fallback_plan(steps=[])` **no debe aumentar** (guardarraíl anti-contraproducente).
- Emit-parse failures en manual-TC con schema enriquecido: **no debe subir** significativamente.
- Latencia: nº medio de llamadas LLM por planning **≤ +1.5** (1 revisión capeada).
- Tokens del `<plan-overview>` por iteración: **sin aumento** (fundamentación una sola vez).

**Preguntas abiertas.**
- El gate mide forma, no verdad: un 7B puede rellenar `evidence='src/auth.py'` (archivo real que nunca abrió). La fase-2 cierra "archivo nunca abierto" pero NO "archivo abierto, supuesto falso sobre su contenido".
- Umbral N de archivos para el trigger del council: demasiado bajo convoca en tareas medianas (caro en single-GPU); demasiado alto nunca dispara.
- Keywords idioma-dependientes (repo multi-idioma): falsos negativos sistemáticos aun con ES/EN/PT. ¿El trigger por nº-archivos solo es más robusto?
- ¿El schema enriquecido (6 campos) consume tanto del `max_tokens=3000` que degrada la **calidad de los steps** en sí? Riesgo de techo, no de piso. Medir.
- CoVe/Self-Refine son empíricos para QA factual, **no** para justificación de planes multi-step. Transferencia plausible pero no demostrada.

---

### 2.4 Inteligencia — modelos chicos no investigan ni fundamentan en evidencia externa

**Causa raíz.** La recolección de evidencia externa es **100% opt-in y nunca gateada**. Las tools capaces (`web_search`, `web_fetch`, `code_search_web`, `doc_flow`) ya están bound al chat_agent/planner/council como read-only (`tools/__init__.py:145-161`), pero **ningún punto del pipeline obliga, cuenta ni verifica** una llamada de investigación antes de aceptar un plan. Todo el grounding forzado apunta al **código local** (`evidence_summary` en `step_complete`), nunca al mundo externo. El planner —choke point del enfoque de una tarea científica— no menciona web/research/docs y su nudge empuja a "emit the plan NOW" tras 4 archivos (`planner.py:176-186`). El único camino con sesgo a research (council) depende del opt-in del propio modelo que "no sabe que no sabe". Además: `FindDocumentationTool` **no declara `is_read_only`** (`find_documentation_tool.py:19`) y queda fuera del toolset del planner; `search_ddg` devuelve `[]` silencioso en error/timeout/vacío (`backends.py:54,57,66`); context7 (MCP) no está cableado; y el loop descarta el output crudo dejando solo summaries de ~50 tokens, así que **aun investigando, la evidencia no sobrevive entre iteraciones**.

**Enfoque elegido.** Convertir la investigación de opt-in a **precondición estructural** del planner vía gate determinista, con tres endurecimientos:
1. **GATE SOFT no hard:** rechaza+reinyecta el nudge **una vez**; tras `MAX_RESEARCH` rechazos acepta el plan pero lo **tagea UNGROUNDED** y lo expone al usuario (evita deadlock por fragilidad de DuckDuckGo).
2. **RELEVANCIA, no solo conteo:** cada claim externo debe tener fuente con cosine MiniLM(claim, snippet) sobre umbral, reusando `_cosine_similarity`/`_get_embed_fn` de `dedup.py:45,88`, y `web_fetch` real del URL antes de aceptar (cierra el gate ritual que se satura con SEO-spam).
3. **RETRIEVER DETERMINISTA + CONTEXT7 PRIMARIO:** cablear el MCP context7 como wrapper read-only priorizado sobre DDG para docs-de-librería; endurecer `backends.py` para reintentar y devolver `NO_EVIDENCE_FOUND` explícito.

La detección `needs_external_research` usa heurística conservadora complementada con un `knowledge_check` binario en `emit_plan` (el modelo declara y la heurística puede forzar el gate aunque diga no). Off-by-default tras `RESEARCH_GATE_ENABLED`. La evidencia se persiste como `EvidencePack` adjunto al Plan y al `EscalationPacket` (patrón loose-typed de `design_brief`, `escalation_packet.py:65`) y se renderiza como bloque inmutable `<evidence>` en el prompt base del developer.

**Alternativas descartadas.**
- HARD GATE / `EVIDENCE_FLOOR` bloqueante: convierte cada turno flagueado en `MAX_RESEARCH` round-trips desperdiciados ante DDG flaky. Reemplazado por gate SOFT.
- Gate que cuenta filas (`evidence_count>=2`): gate ritual; el modelo junta 2 filas de SEO-spam. Reemplazado por relevancia vía cosine.
- Step de investigación **dentro** del developer loop: reintroduce el problema que el diseño evita. La investigación se queda **solo en el planner**.
- Tabla SQLite de evidence por session_id: redundante; la evidencia ya queda en el Plan.
- Reutilizar GATHER: `GATHER_ENABLED=False` y orientado al codebase/ticket, no a literatura externa.

**Pasos concretos (archivos).**
1. Endurecer `search_ddg` (`tools/web/backends.py:30/54/57/66`): 1 retry tras backoff + `NO_EVIDENCE` explícito distinguible de cero-resultados.
2. `tools/web/context7_tool.py` (nuevo) + `tools/web/__init__.py` + `tools/__init__.py:56`: wrapper read-only del MCP context7 (resolve-library-id + query-docs), primario para docs-de-librería; degrada limpio si el MCP no está.
3. `tools/docs/find_documentation_tool.py:19`: `is_read_only=True` (no tocar Update/Delete).
4. `engine/research/evidence_pack.py` (nuevo): `EvidencePack` (items claim/url/title/snippet/relevance, render con citas `[n]`, cap top-K); adjuntar a `escalation_packet.py:65` y `plan.py`.
5. `engine/research/evidence_gate.py` (nuevo) + `settings.py`: `needs_external_research` (dispara si `opened_files` vacío AND keywords científicas). `RESEARCH_GATE_ENABLED=False`, `EVIDENCE_RELEVANCE_THRESHOLD`, `MAX_RESEARCH_CALLS=3`, `RESEARCH_MAX_EXPLORATION=8`.
6. `tools/planner/emit_plan_tool.py`: `knowledge_check:bool` (obligatorio) + `external_claims:list` (opcional); refs obligatorias **por código**, no por schema (no recargar FC).
7. `engine/research/evidence_gate.py` + `tools/research/emit_evidence_tool.py` (nuevo): mini-loop acotado `run_evidence_collection` (solo context7/find_documentation/web_search/web_fetch/code_search_web + terminador `emit_evidence`); queries semilla deterministas; bundling search-fetch-extract en código.
8. Validación de relevancia (`evidence_gate.py` reusando `dedup.py:45,88`): por claim, `web_fetch` real + cosine MiniLM, aceptar solo sobre `EVIDENCE_RELEVANCE_THRESHOLD`.
9. **GATE SOFT** en `planner.py:239-264`: si flagueado sin claims relevantes → rechazar+reinyectar nudge; `research_calls` **separado** del budget de exploración; tras `MAX` → aceptar+tag `UNGROUNDED`. Nudge "emit NOW" (`176-186`) condicional al modo research.
10. Inyección persistente (`engine/loop/engine.py` + `prompts/analyst/planner_prompt.py`): render `<evidence>` inmutable en el prompt base (no en summaries); reescribir el prompt con protocolo de 2 fases y definición operacional de claim externo (ref buena vs mala).

**Métricas de éxito.**
- % de tareas flagueadas con ≥1 `external_claim` sobre threshold: **>70%** antes de degradar a UNGROUNDED.
- Precisión/recall del gate: falsos positivos **<15%**, falsos negativos **<25%** en tareas científicas.
- Separación de cosine claim-vs-snippet (relevante vs SEO-spam).
- Supervivencia: bloque `<evidence>` en **100%** de iteraciones del developer en tareas flagueadas.
- Latencia añadida en modo research: **<2×** el turno base.
- Tasa de UNGROUNDED (si alta, el cuello de botella es el retriever, no el gate).

**Preguntas abiertas.**
- `all-MiniLM-L6-v2` es débil para relevancia semántica fina: calibrar el threshold; riesgo de grounding **parcialmente decorativo**.
- ¿Derivar queries semilla deterministas del `user_request` (las del modelo chico son pobres)?
- context7 brilla en docs-de-framework, pero diseño de LLMs es ciencia pura → quizá haga falta backend académico (arXiv/Semantic Scholar), fuera de scope.
- La heurística necesita lista estática de keywords (la doctrina del repo prefiere evitarlo). ¿Señal más dinámica (code index no matchea ningún sustantivo técnico)?
- Medir **primero** la tasa real de `knowledge_check=sí` en tareas científicas antes de invertir en budget=8 y reintentos.
- Costo de `web_fetch` real por claim en hardware de consumo con conectividad intermitente; ¿cachear en `web_cache` mitiga?

---

## 3. Roadmap integrado

### 3.1 Orden recomendado (no es arbitrario: cada nivel construye infra que el siguiente reusa)

| # | Problema | Por qué en este orden |
|---|----------|----------------------|
| **1** | **Contexto (INCREMENT A)** | Fundacional. Sin budget honesto y cap de tool-results, **cualquier bloque nuevo** de los otros 3 planes empuja a Ollama a truncar en silencio. El ledger (A4) construye el índice de tool_calls que el evidence gate de Sobreconfianza necesita. El cap (A3) hace que los tests que Sobreconfianza fuerza no se trunquen. |
| **2** | **Sobreconfianza** | El evidence gate post-edit es el **mayor ROI / menor riesgo** y se monta **gratis** sobre el ledger A4 (mismo `tracker.files_edited`). El scope gate plan-time **fija el patrón gate-inline-con-1-retry** que Diseño e Inteligencia reusarán. |
| **3** | **Diseño** | Depende del patrón de gate-inline (Sobreconfianza) y del Governor/PromptBlock (Contexto-B) para registrar el bloque risks/assumptions sin romper la ventana. Comparte el refactor de schema `emit_plan` con Inteligencia. |
| **4** | **Inteligencia** | El más caro (web_fetch + cosine + context7 MCP + mini-loop) y el más frágil (DDG flaky, MCP puede no estar). Off-by-default → seguro dejarlo último. Hereda: schema enriquecido de Diseño, patrón gate-SOFT-con-tag, render de bloque inmutable de Contexto, trigger-por-keywords casi-idéntico al de Diseño. |

### 3.2 Sinergias (reforzar, no duplicar)

- **Schema-as-contract compartido:** Diseño (assumptions/alternatives/rationale/risks) e Inteligencia (knowledge_check/external_claims) son **UN solo refactor** de `EmitPlanInput` (79 líneas) con un solo set de tests. Coordinar en **una sola PR** evita dos round-trips de FC-schema sobre el mismo terminador frágil.
- **Gate inline componible:** Sobreconfianza (`enforce_scope`), Diseño (`check_plan`) e Inteligencia (gate SOFT) deben **fusionarse en un solo punto** (lista de validadores → lista de fallos → **1 reinyección combinada**), no tres ramas que cada una consume un retry.
- **`_looks_concrete` compartido** (`tools/meta/_plan_checks.py`): sirve al gate del planner (Diseño/Sobreconfianza) y al nudge del developer. Una implementación, sin drift.
- **Ledger A4 → evidence gate gratis:** ambos leen `tracker.files_edited` + secuencia tipada del step. Construir el ledger primero le da el evidence gate gratis.
- **Cap A3 → verificación útil:** sin el cap head+tail, forzar tests post-edit (Sobreconfianza) produce output que se trunca y pierde la línea de error que prueba la corrección.
- **Bloque inmutable compartido:** `<evidence>` (Inteligencia) y assumptions-UNVERIFIED+risks (Diseño) usan el **mismo mecanismo**: bandera inmutable en el prompt base que sobrevive al ciclo plan-execute-summarize, fuera de los summaries. Mismo render path en `context.py`, misma disciplina anti-bloat (una vez, no por iteración).
- **Trigger/heurística único:** el trigger del council (Diseño, keywords + nº opened_files) y `needs_external_research` (Inteligencia, opened_files vacío + keywords científicas) comparten heurística casi idéntica sobre el `EscalationPacket`. **Una sola función de clasificación** alimenta ambos.
- **PromptBlock + count_tokens del Governor** es la infra que mide y poda; los bloques nuevos de Diseño/Inteligencia **deben registrarse como PromptBlock con prioridad** para que el Governor los gobierne.

### 3.3 Conflictos (deben resolverse antes de implementar en paralelo)

1. **Presupuesto del loop del planner (el más duro).** `planner.py` tiene `max_iterations=6`, `max_exploration_calls=4`. Sobreconfianza pide +1 retry, Diseño pide +1, Inteligencia pide mini-loop separado + subir exploración a 8. Si cada plan reclama su retry, un plan flagueado por los tres gasta **3 emit extra** y cae en `_fallback_plan` (steps vacíos) — **el peor resultado**. → **Un solo presupuesto de revisión: 1 reinyección que combina los tres tipos de fallo.**
2. **max_iterations=8 vs presupuesto de contexto.** Más iteraciones = más lecturas acumuladas en el contexto del planner, que Contexto poda. → Subir iteraciones **solo para absorber la revisión, NO para más exploración** (`max_exploration_calls=4` intacto).
3. **Inflado del schema emit_plan.** Diseño +4 campos (anidados), Inteligencia +2 = **6 campos nuevos** sobre un terminador que los 7B ya truncan, con `max_tokens=3000`. → Campos obligatorios en FC-mode pero **tolerantes en el parser**, y medir longitud/concreción de steps antes/después.
4. **Competencia por tokens (bloques nuevos vs ContextRank).** ContextRank agrega tokens mientras el Governor poda; Diseño agrega risks/assumptions, Inteligencia agrega `<evidence>`. → **El Governor (Contexto-B) DEBE existir y gobernar esos bloques con prioridades fijas (P0..P3) antes de habilitarlos por defecto.**
5. **Evidence gate vs LoopGuard.** Un breaker que fuerza `step_complete` podría convertir un step bloqueado-por-falta-de-evidencia en falso `done`. → Definir precedencia: el evidence gate corre **antes** del forzado.
6. **`/compact` forzado (Contexto-C2) vs todo lo demás.** Usa el mismo modelo chico para compactar; podría borrar silenciosamente assumptions/evidence/ledger. → C2 debe respetar los bloques **P0 inmutables** y nunca compactarlos.
7. **Latencia agregada en single-GPU.** Inteligencia (web_fetch + cosine + mini-loop) + Diseño/Sobreconfianza (+1 retry c/u) + `count_tokens` por iteración. Cada plan justifica su costo aislado, pero **el agregado no está presupuestado por nadie**. Medir el turno de planning completo.

### 3.4 Quick wins (alto ROI, casi cero riesgo — empezar por acá)

- **Contexto A1:** anclar `max_context_tokens` a `min(context_length, OLLAMA_NUM_CTX)` con headroom 25%. ~1 línea en `model_context.py` reusando un setting que **ya existe**; elimina el truncado silencioso, la causa raíz más insidiosa.
- **Contexto A3:** `cap_tool_result` head+tail con regex de errores, quitando el gate `is_small`. Convierte el corte ciego `[:200]` en preservación de errores universal.
- **Sobreconfianza (evidence gate, pasos 2-3):** reusa `_overwrite_step_complete_tool_result` y `tracker.files_edited`, ya disponibles. Código puro testeable sin GPU; ataca el caso clásico "fix limpio pero erróneo".
- **Inteligencia (pasos 1-3):** `search_ddg` con retry + `NO_EVIDENCE_FOUND`, y `FindDocumentationTool` gana `is_read_only=True` (1 atributo). Triviales, sin dependencias.
- **Diseño (paso 2):** promover `_looks_concrete` a módulo compartido. Refactor mecánico sin cambio de comportamiento que desbloquea el gate y elimina drift.
- **Sobreconfianza (`SCOPE_MAX_STEPS`):** rechazar planes de >6 steps. Pocas líneas en `_parse_emitted_plan`, alto valor contra el step monolítico "build the whole system".

### 3.5 Big bets (transformadores pero caros/riesgosos — medir antes de confiar)

- **Contexto INCREMENT B (ContextGovernor):** el refactor más riesgoso (toca `build_iteration_prompt`, el historial que comparte el critic, el prompt caching/`CACHE_BREAKPOINT_MARKER`); exige golden-file tests primero. Pero es la **única infra** que convierte el budget de advisory a enforcement garantizado y evita que los bloques de Diseño/Inteligencia desborden la ventana. Sin esto, los otros planes son parches sobre una ventana que sigue reventando.
- **Inteligencia completo:** context7 MCP primario + web_fetch real + cosine MiniLM + mini-loop + EvidencePack persistente. Caro (<2× latencia) y frágil en hardware de consumo. Es lo único que ataca "el modelo no sabe que no sabe" con evidencia externa. **Alto riesgo de grounding decorativo si el cosine no separa bien; off-by-default y medir.**
- **Schema-as-contract enriquecido (Diseño + Inteligencia):** 6 campos que materializan la cadena de razonamiento. Transformador (de 0% a ≥80% con assumptions/alternativas) **pero** arriesga el techo de calidad de los steps en `max_tokens=3000`. Hipótesis a validar empíricamente.

---

## 4. Regla integradora (una frase)

**Un solo punto de gate componible en el planner** (lista de validadores → 1 reinyección combinada → 1 retry compartido → fail-open), **un solo refactor de schema `emit_plan`**, y **todo bloque nuevo registrado como PromptBlock bajo el Governor** con prioridad fija. El presupuesto de contexto agregado es el guardarraíl dominante: Diseño e Inteligencia quieren **agregar** tokens exactamente en la ventana que Contexto **protege**, así que el Governor debe existir antes de habilitar esos bloques por defecto.

**Archivos clave (verificados):** `src/infinidev/engine/analysis/planner.py` (`_DEFAULT_MAX_ITERATIONS=6`, `_DEFAULT_MAX_EXPLORATION_CALLS=4` en L33-34, emit L158-159, parse L239-263), `src/infinidev/config/settings.py` (`OLLAMA_NUM_CTX=16384` en L92, ya cableado), `src/infinidev/tools/planner/emit_plan_tool.py` (79 líneas, schema compartido), `src/infinidev/engine/loop/engine.py` (1832 líneas, step_complete + render), `src/infinidev/engine/orchestration/pipeline.py` (659 líneas, trigger council).