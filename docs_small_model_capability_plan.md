Los touchpoints citados están verificados (planner.py:202/224, llm.py:225, settings.py:182, review_engine.py automated_checks/verification_passed, findings con embedding BLOB+FTS5, SANDBOX_ENABLED=False con subprocess.run pelado). Escribo el informe.

# INFORME FINAL — Capa de Capacidad (subir el techo) de Infinidev

*Para el dueño de Infinidev. Esta capa va ARRIBA de la capa de contención ya diseñada. Honesto, concreto, y explícito sobre qué requiere un modelo grande remoto.*

---

## 1. Resumen ejecutivo y la tesis

### La tesis: "subir el techo" significa inyectar capacidad DESDE FUERA de los pesos del 7B

La contención (capa 0, ya diseñada) **tapa el piso**: impide que el modelo chico actúe más capaz de lo que es —que edite sin verificar, que alucine confiadamente, que reviente el contexto. Pero la contención **no agrega ni una gota de capacidad nueva**. Un 7B contenido sigue sin saber diseñar un token-bucket con lock, sin conocer la ecuación gobernante de una PDE, sin poder descomponer una tarea cross-cutting correctamente. Forzarlo a "externalizar" su debilidad no la convierte en fortaleza.

**Subir el techo = hacer que el sistema PRODUZCA artefactos correctos que el `pass@1` del 7B no alcanza.** Y la causa raíz, confirmada por los cuatro problemas de forma independiente, es brutal: **la capacidad no está en los pesos del 7B, y ninguna búsqueda en tiempo de inferencia la fabrica de la nada.** Por lo tanto, sólo hay dos fuentes reales de techo:

1. **MEMORIA recuperable verificada** (warm path) — una receta/procedimiento que YA pasó tests reales en una tarea similar. El 7B *elige y adapta* en vez de *inventar*. **Es lo único que sube el techo offline/local-puro, una vez sembrada.**
2. **TIER ROUTER a modelo grande remoto** (cold path) — cuando no hay receta cercana, un modelo grande (z.ai/glm, minimax, deepseek) **autora** la arquitectura/esqueleto y el 7B sólo lo **aterriza y ejecuta**. Es la única fuente de techo en una tarea genuinamente novedosa.

Estas dos fuentes se complementan exactamente: el oráculo remoto es el *maestro* del cold path, la memoria es el *alumno* del warm path. Cada esqueleto que el oráculo produce y que pasa verificación se **cosecha** hacia la memoria → la dependencia remota **amortiza a cero** en encuentros futuros. Es transferencia grande→chico vía artefactos, sin fine-tuning, medible.

### Honestidad central, que debe llegar al usuario textualmente

> **En local-puro + cold-start + sin recetas sembradas + sin red, el techo NO sube en ninguno de los cuatro problemas.** El sistema degrada limpiamente al comportamiento de hoy (sin breakage, pero sin ganancia). No hay forma seria de subir el techo de diseño/ciencia/conocimiento de un 7B sin (a) una receta verificada cercana, o (b) un modelo grande remoto en algún momento. Este informe lo confirma cuatro veces.

### Capacidad vs. plomería (la distinción que ordena todo)

- **Capacidad (sube el techo):** la memoria de recetas, el tier router de autoría remota, el selector por ejecución (BSVS). Producen un artefacto que el 7B no podía producir.
- **Plomería (necesaria, no sube el techo):** el gate NLI que evita envenenar la librería, el `executable_check` contra el code-index, el aislamiento git-worktree, la poda de contexto. Sin ellas la capacidad se corrompe o no cabe, pero por sí solas no agregan inteligencia.

Lo digo sin vueltas a lo largo del informe: **dónde el techo sólo sube con un modelo grande remoto, lo marco explícitamente.**

---

## 2. Por problema

> Nota de integración leída por adelantado: los cuatro problemas **comparten** una sola tabla `recipes`, un solo retriever, un solo tier router y un solo harvest gate. Lo describo problema por problema, pero **NO se construyen cuatro de cada cosa** (ver §3).

---

### 2.1 DESIGN — Subir el techo de la CALIDAD de diseño

**Arquitectura elegida:** dos fuentes externas de capacidad.
- **WARM (retrieval) — Recipe Bank:** biblioteca recuperable de `DesignBrief` verificados por ejecución. El 7B selecciona+adapta un diseño que YA pasó tests reales. Totalmente local. Es el ancla (feasibility=5, único que sube el techo sin red).
- **COLD (routing) — Oracle-Authored Plan Skeleton:** en cold miss con keys, un modelo grande remoto autora el esqueleto (overview + por-step `design_rationale`/`assumptions`/`alternatives`/`verification_command`) vía el seam `emit_plan`; el 7B sólo lo aterriza.
- **OPCIONAL (search) — PlanForge degradado a re-ranker:** no es generador. Sólo desempata cuando hay ≥2 candidatos diversos, vía self-consistency + verificador ejecutable. OFF por defecto en local.

**Qué produce el chico que hoy no puede.** Hoy, en "agrega rate-limiting al API", el 7B inventa un diseño plausible-pero-roto: dict global sin TTL ni lock, o un `bool deleted` que rompe el unique index. Con el Recipe Bank, recupera un `DesignBrief` que ya pasó tests reales (token-bucket por-key con lock; `deleted_at` + partial index + migración) con la alternativa rechazada **vetada** y los archivos exactos. Convierte "generar arquitectura" (debilidad verificada, arXiv:2404.17140) en "elegir entre 3 diseños vetados y mapear archivos" — una **operación de ejecutor**. En tarea novedosa, el oráculo aporta lo que el 7B no tiene (jitter/cap/idempotency, circuit-breaker como alternativa explícita).

**Grounding (citas reales):**
- **Voyager** — Wang et al., 2023, arXiv:2305.16291 (verificado). Biblioteca de skills/recetas ejecutables indexadas por embedding; seleccionar+adaptar en vez de inventar; lazo write→retrieve→re-verify. Patrón exacto del Recipe Bank.
- **FrugalGPT** — Chen, Zaharia, Zou, 2023, arXiv:2305.05176 (verificado). Cascade chico→grande; sustenta el split planner/executor del Oracle Skeleton.
- **Small Language Models Need Strong Verifiers to Self-Correct Reasoning** — arXiv:2404.17140. Prueba que el 7B NO puede auto-juzgar ni auto-reparar diseño → justifica poner TODA selección/autoría FUERA del modelo.
- **ExpeL** — Zhao et al., AAAI 2024, arXiv:2308.10144. Extracción offline de insights como exemplars in-context sin tocar pesos; sustenta el seed pack.
- **PLANSEARCH** (arXiv:2409.03733) + **Self-Consistency** (Wang et al., ICLR 2023, arXiv:2203.11171). Sustentan la capa de selección OPCIONAL.
- *ADaPT (arXiv:2311.05772) citada de conocimiento previo, NO re-verificada esta sesión — tratar como no-confirmada.*

**Diseño concreto con touchpoints (verificados en el código):**
- Tabla `recipes` clonada del patrón `findings` (ya tiene `embedding BLOB` + FTS5, verificado en `db/service.py:60-83`). Dedup con `find_semantic_duplicate` (0.82).
- `engine/analysis/recipe_bank.py` (nuevo): `harvest_recipe()` y `retrieve_recipes(k=3, hard_filter=lang_tags)`. Ranking = `sim * (success+1)/(success+fail+2)` [Laplace]. **Filtro DURO por lang/stack ANTES del kNN** (obligatorio: strings cortos hacen leakage cross-familia).
- RETRIEVE pre-planner en `pipeline.py`; inyección en `_render_handoff` (verificado `planner.py:202`, y ya lee `escalation.design_brief` en `:224`).
- `get_litellm_params_for_oracle()` clonado de `get_litellm_params_for_behavior` (verificado `llm.py:225`).
- **Lock challengeable del skeleton (enforcement NUEVO, no reuse gratis):** `apply_operations` sólo protege el `StepOperation` del LoopEngine, NO el re-emit del planner → requiere un **field-diff check nuevo** en el grounding pass. Si un assumption no se puede bindear → `grounding_conflict` → re-spin con oracle, NO lo reescribe el 7B (evita *capability laundering*).
- **HARVEST gateado en señal determinista:** cosechar SÓLO si `automated_checks.verification_passed==True` (verificado `review_engine.py:168-185`), NUNCA por `verdict==APPROVED` (en local es el mismo 7B débil).

**Local vs. remoto.** WARM funciona idéntica y es el mayor ROI: cero modelo/GPU/red extra; sólo el embedder MiniLM ya presente. COLD: el oráculo es remoto **aunque el dev model sea Ollama local** — sin 2ª GPU, ese es el punto. Pure-offline sin keys: cold miss cae al `run_planner` single-model de hoy (gain perdido, sin breakage). **Honestidad: local-puro + cold-start + sin recetas → el techo NO sube.**

**Experimento.** 20 tareas en 2 familias sobre `examples/`, cada una con un **grader oculto** (test held-out que codifica el buen diseño). Dev model fijo `qwen2.5-coder:7b`. Arms: (A) control, (B) Recipe Bank sembrada, (C) B + Oracle. **Primaria:** Review pass@1 con `verification_passed==True` en el grader oculto (no el rubric del skeleton). **Falsación pre-registrada clave:** (i) **ablación de barajado como gate primero** — asignar receta de familia EQUIVOCADA debe DESTRUIR la ganancia; si no, el retrieval no se usa → no se envía. (ii) Si la ganancia de B no correlaciona con la similitud receta-tarea → el efecto no viene del retrieval. (iii) Control A' = best-of-1 con el mismo presupuesto total de tokens; si A' iguala a C, el gain fue compute, no capacidad.

**Costo.** WARM: despreciable (+1 embed ~11-115ms, +1 INSERT post-Review, ~1.5KB/receta). COLD: 1 completion remota sub-centavo, sólo en cold miss con keys. PlanForge OFF en local. **Hardware nuevo: ninguno.**

---

### 2.2 INTELLIGENCE — Inyectar conocimiento científico que el modelo NO tiene embebido

**Arquitectura elegida.**
- **NÚCLEO — "Recetario":** librería SQLite de PROCEDIMIENTOS científicos verificados (plan-skeletons parametrizables, autorados una vez por el tier grande, anclados por entailment) que el chico recupera-y-adapta. Es lo único que sobrevive offline.
- **GENERACIÓN-EN-MISS via tier routing:** cuando `rerank < tau`, la autoría se escala al modelo grande remoto. **Esto ya es shippable hoy:** verificado que `COUNCIL_MODEL` existe (`settings.py:182`), el `DesignBrief` se produce en council, y `planner.py:224` ya lee `escalation.design_brief`. **La novedad no es el cascade (ya existe), es CRISTALIZAR su salida en la librería reutilizable.**
- **GATE subordinado (plomería necesaria):** cross-encoder NLI/entailment local en el commit-to-library. Filtra recetas malas para que la librería no se envenene. **No es headline, es plomería.**

**Qué produce el chico que hoy no puede.** Hoy en "elige el test estadístico válido para este diseño" o "monta un esquema de diferencias finitas estable", el planner es el chico (`planner.py:125` usa `get_litellm_params_for_behavior`, verificado) y emite un plan plausible pero **metodológicamente incorrecto**: ecuación gobernante equivocada, precondiciones del algoritmo ignoradas. Con el Recetario, en el SEGUNDO encuentro de esa familia, recupera un skeleton autorado por el teacher (orden de pasos correcto + passage de la ecuación verificado por entailment) y sólo bindea los números del usuario en los `parameter_slots`. Produce un método científico multi-paso correcto **offline**, porque la capacidad vive en SQLite, no en los pesos.

**Honestidad sobre el límite.** El techo sube **al del modelo grande que autoró la receta, no al infinito.** Si el teacher se equivoca, la receta es mala (mitigado por NLI gate + decay, no eliminado). **En despliegue 100% local-only y SIN seed corpus, NO hay forma seria de subir el techo científico** — el chico no tiene el conocimiento y nadie se lo da.

**Grounding (citas reales):**
- **Voyager** — arXiv:2305.16291 (verificado). La capacidad se acumula en la librería, no en los pesos; self-verification antes de commit.
- **FrugalGPT** — arXiv:2305.05176 (verificado). Sustenta la autoría-en-miss.
- **Let's Verify Step by Step** — Lightman, Kosaraju, Burda et al., 2023, arXiv:2305.20050 (verificado, 78% MATH, PRM800K). Supervisión por proceso > outcome; sustenta scoring por step si se añade PRM.
- **SCoTD** — Li, Hessel et al., 2023, arXiv:2306.14050 (**ID sin verificar**; claim núcleo —estudiantes sub-1.3B ganan de rationales del teacher— es real). El teacher autora el rationale, cacheado como exemplar sin fine-tuning.
- **ALCE** — Gao, Yen et al., EMNLP 2023, arXiv:2305.14627 (**ID sin verificar**) + "Correctness is not Faithfulness in RAG Attributions" arXiv:2412.18004 (**sin verificar**). NLI cross-encoder chequea que el passage IMPLIQUE el claim.
- **Inference Scaling Flaws** — arXiv:2411.17501 (**sin verificar**). Verificadores imperfectos imponen un piso irreducible de falsos positivos → tau es señal, no binario duro.
- **Self-Consistency** — arXiv:2203.11171 (**ID sin verificar**). Sólo para la mejora OPCIONAL best-of-N.

**Diseño concreto.** Tabla `recipes` (reusa BLOB-embedding + FTS5). `recipe_retriever.py` con recall **bi-encoder + FTS5/BM25, SIN cross-encoder en hot path local** (verificado: no hay cross-encoder en `src`). Inyección en `_render_handoff`. Autoría-en-miss reusa `COUNCIL_MODEL` + `DesignBrief` ya existentes. `nli_gate.py` **commit-time SOLAMENTE** — modelo NUEVO (~80-100MB ONNX-CPU, DeBERTa/MiniLM-NLI), corre sólo en autoría remota, nunca en retrieval local. **Provenance check:** el passage debe ser substring verbatim de un doc realmente recuperado (`tools/web`), no de la memoria del teacher — sin esto el grande alucina citas y el NLI verifica ficción contra ficción.

**Local vs. remoto.** Recuperación local sin red, sin modelo nuevo, sin cross-encoder. Tras la primera autoría (o seed corpus), el chico es autosuficiente offline. El NLI gate sólo corre en autoría → en local puro no se paga. **El salto de techo real requiere que la receta haya sido autorada en algún momento por un modelo grande. Sin esa semilla, esto degrada a contención + recuperación de nada.**

**Experimento.** 40 tareas, ~8 familias científicas, rúbrica graduada de CORRECCIÓN DE MÉTODO, juez independiente (claude-opus ciego o humano). Arms A (baseline), B (librería caliente), C (librería fría que se calienta). **Primaria:** score de corrección-de-método + % con ecuación correcta citada Y pasando entailment. **Falsable:** (1) B>A por ≥15 pts; (2) **C converge hacia B al repetirse las familias** (prueba el compounding — la capacidad vive en la librería, no en el cascade); (3) B con ≥90% menos llamadas al teacher que always-escalate. **Refutado si** C no converge → es sólo cascade, no librería. NLI held-out distinto del commit-gate para evitar leakage.

**Costo.** Hot path local: cientos de ms, cero GPU/red/modelo nuevo. Autoría-en-miss: 1 llamada remota + NLI cross-encoder (~80-100MB, **único modelo nuevo del sistema entero**). **Cold-start obligatorio: shippear seed corpus**, si no el claim de capacidad-durable no se demuestra limpio.

---

### 2.3 CONFIDENCE — De contener la sobreconfianza a CALIBRARLA y rutear

**Arquitectura elegida: SCALE + el leg asimétrico de PlanArena**, fusionados en una fase "plan-search-then-route" en `run_planner()`. **PlanBank fue DESCARTADA** (`grounding_is_real=FALSE`): su claim de que la entropía sobre k DOCUMENTOS recuperados es "semantic-entropy calibration" es falso — la entropía semántica (Kuhn/Farquhar) mide acuerdo entre muestras INDEPENDIENTES de la distribución generativa, no entre documentos correlacionados por retrieval.

**La objeción fatal que la fusión resuelve.** La entropía/vote-margin de un 7B **NO discrimina**, porque su modo de fallo dominante es **confident-and-consistent confabulation**: los k samples comparten el mismo sesgo en los pesos (no en la temperatura) → BAJA entropía, NUNCA escala, justo el fallo que se quiere atrapar. Por eso: (1) vote-margin es **ahorrador de costo, no gate de capacidad**; (2) trigger ortogonal barato = `executable_check` determinista contra el code-index, que fuerza escalada aunque la entropía sea baja; (3) PRM remoto-grande como JUEZ — único leg que atrapa el sesgo lógico sistemático.

**Qué produce que hoy no puede.** Hoy `run_planner` hace UNA llamada greedy (temp 0.1) y acepta el primer `emit_plan`; si funda un step en una alucinación (editar archivo inexistente, asumir API que no existe), el error se propaga intacto al LoopEngine. Con esto, en "add a provider-aware streaming token budget across Ollama y litellm", el 7B hoy apunta al archivo equivocado y omite la abstracción de provider; tras este plan, el archivo correcto (`config/llm.py`, donde viven los `get_litellm_params_*`) y la abstracción vienen del modelo grande remoto. En tareas fáciles los borradores convergen, el chequeo ejecutable pasa, y no hay llamada remota.

**Grounding (citas reales):**
- **Self-Consistency** — Wang et al., ICLR 2023, arXiv:2203.11171 (GSM8K +17.9%). SÓLO como pre-filtro/ahorrador.
- **FrugalGPT** (arXiv:2305.05176) + **RouteLLM** — Ong et al., 2024, arXiv:2406.18665 (~85% recorte de costo a 95% de GPT-4). El leg que de verdad sube el techo.
- **Let's Verify Step by Step** (arXiv:2305.20050) + **ThinkPRM** — Khalifa et al., 2025, arXiv:2504.16828. PRM generativo; verificar << generar.
- **Semantic Uncertainty** — Kuhn, Gal & Farquhar, ICLR 2023, arXiv:2302.09664 (+ Nature 2024). Vote-margin como señal SUAVE ortogonal, calibrada per-model-id, NUNCA sola fuente.

**Diseño concreto.** `plan_arena.py` (nuevo): `sample_plans` (N-1 re-rolls del mismo prompt a temps 0.2/0.5/0.7/0.9), `cluster_plans` sobre `{target-files + step-intent verbs}` ordenados (NO prosa de títulos, para no ser ciego al DAG) via MiniLM. `executable_check(plan)->penalty` model-free contra el code-index (step que edita archivo inexistente → score 0 antes de gastar PRM; `penalty>0` fuerza escalada). Trigger compuesto: `(vote_margin bajo) OR (penalty>0) OR (pre-signal de complejidad)`. `get_litellm_params_for_strong_planner()` espejando `llm.py:171-276`. `run_prm_verifier` en rol **adversarial** ('nombra un archivo que falta'), grande=JUEZ si hay red, chico=señal suave si no. Persistencia de calibración en `artifacts` keyed por `model_id` — **NO usar vote_margin como confianza hasta demostrar AUROC>0.65.**

**Local vs. remoto.** Local: N=4 borradores SERIALIZAN (~4x latencia, ~24s, fase corta). Clustering + executable_check en CPU. El tier fuerte es REMOTO — ese es el punto, sube el techo sin 2ª GPU. Sin red: degrada a majority-vote medoid local + executable_check, **la UI debe surfacear el tier degradado.** Remoto: los N borradores PARALELIZAN.

**Experimento.** 40 tareas (20 fáciles, 20 difíciles), gold Plan a mano. Arms A (greedy), B (N=4 medoid + executable_check sin remoto), C (completo). **Primaria:** PLAN-GROUNDEDNESS (% steps con refs válidas). **Falsable clave (correr PRIMERO):** medir DIVERSIDAD — mean pairwise cosine de los N=4 samples; **si colapsan a casi-idénticos, best-of-N está muerto para ese modelo** y C debe ganar SÓLO por executable_check + PRM remoto. Refutado si C no supera A en difíciles, o el PRM no supera al vote-only, o AUROC<0.65.

**Costo.** Cero hardware nuevo. La GPU paga N-1 generaciones SÓLO en tareas que pasan el gate de complejidad. **Backstop obligatorio día-1: cap de presupuesto remoto por-sesión** para que un 7B crónicamente inconsistente no infle el costo.

**Honestidad.** La parte que de VERDAD sube el techo en la fracción difícil ES el modelo grande remoto. Sin provider remoto, lo que se sube es modesto (medoid-vote + filtrado ejecutable: gana groundedness, no inteligencia de descomposición).

---

### 2.4 CONTEXT — De resumir mejor a una arquitectura de memoria

**El eje real NO es el que sugiere el título.** El problema de contexto/memoria **no se resuelve resumiendo ni paginando mejor** (MemGPT): el problema no es *recordar*, es **no-saber**. El eje es externalizar capacidad a (a) un autor remoto y (b) un selector por ejecución.

**Arquitectura elegida:** "Tier-Escalated Authorship + Reflexion local" envuelto sobre un núcleo de **Execution-Verified Best-of-N** (BSVS) como selector ground-truth. La memoria persistente se DEGRADA a una versión honesta: skills SEMILLA deterministas descubiertas del repo, NO una librería de código-LLM persistido.

**Qué murió en el debate (y por qué):**
- De Skill Authoring sobrevive el routing de autoría + Reflexion. **MUERE "invoke free forever / compounding"**: verificado que `SANDBOX_ENABLED=False` (`settings.py:70`) y el shell es `subprocess.run` pelado (`execute_command_tool.py:184`, `code_interpreter_tool.py:229`) sin firejail/docker/seccomp → **persistir y re-ejecutar código escrito por un LLM remoto es un agujero RCE peor que el baseline.** El compounding cross-task de skills repo-específicas es especulativo (su propia kill-condition).
- De Recipe Forge sobrevive UNA idea robusta: skills semilla hand-written descubiertas deterministamente del repo (`pytest.ini`, `alembic.ini`, `Makefile`) al indexar → valor día-uno sin esperar librería ni remoto. **MUERE la captura automática del "ledger"** (el raw tool output se descarta en summarize, no existe tal ledger).
- De BSVS sobrevive el núcleo. **MUERE "reusa el sub-loop sin reimplementar nada":** el sub-loop muta estado global entrelazado (`total_tool_calls`, `opened_files`, índice tree-sitter, `messages`) y **no hay git stash/worktree en el codebase.** Fix adoptada: worktree por candidato + targeted-test obligatorio.

**Qué produce que hoy no puede.** (1) Un STEP VERDE cuando el parche correcto existe en la cola de su distribución (lo acierta 1 de N): BSVS muestrea N=3 a temps escalonadas y selecciona **por ejecución real, no por auto-juicio** (single-pass ~40% → BSVS@3 ~78%, @5 ~92% **si oracle@N lo contiene**). (2) Un artefacto correcto para un sub-problema que supera su techo: se escala la AUTORÍA (no la tarea entera) a un grande remoto; el MISMO verificador decide. (3) Procedimientos project-specific via skills semilla descubiertas del repo.

**Grounding (citas reales):**
- **Self-Consistency** — arXiv:2203.11171 (ICLR 2023). Núcleo de BSVS con verificador de ejecución (ground-truth, no voto).
- **FrugalGPT** — arXiv:2305.05176 (TMLR 2024). Generaliza el terminador `escalate` (verificado `escalation_packet.py`) al nivel de step.
- **Voyager** — arXiv:2305.16291 (NeurIPS 2023). Self-verification como portero de skills — SÓLO para skills semilla con re-verificación lazy, NO para persistir código-LLM.
- **Reflexion** — Shinn et al., 2023, arXiv:2303.11366 (NeurIPS 2023). Stack-trace concreta → memoria que condiciona el reintento local barato antes de pagar la escalada.
- **RAG** (Lewis et al., 2020, arXiv:2005.11401) + **RepoCoder** (Zhang et al., EMNLP 2023, arXiv:2303.12570). Retrieval iterativo de skills semilla y EscalationPacket.

**Diseño concreto.** `engine/loop/step_search.py::BranchSampleVerifySelect`: para step flagged-hard, snapshot via **git worktree** (net-new), N=3 candidatos cada uno en su propio `LoopState`/`FileChangeTracker`/índice FRESCO (aislamiento de proceso, NO rollback in-place), verificar por ejecución, seleccionar ground-truth, re-aplicar sólo el ganador. **VerificationEngine targeted-test (extensión obligatoria):** hoy `verify()` corre la SUITE COMPLETA; extender a sólo los nodeids afectados (`pytest -k`, `cargo test <name>`), distinguiendo test-failure de infra-failure (timeout != assert) para no envenenar el selector. Escalada de autoría a tier remoto vía `get_litellm_params_for_search()`. Reflexion local entre rondas. Skills semilla en `artifacts` (`type='skill'`), tool `invoke_skill`.

**Local vs. remoto.** Local: BSVS serializa (latencia, no dinero); N=3 SÓLO en steps flagged-hard (<30%), targeted-test obligatorio (sin él, 3-6 min/step es prohibitivo). Degradación graciosa: sin `SEARCH_LLM_*` o sin red → Reflexion local → bloquea como hoy, **nunca peor que baseline.** Remoto: BSVS paraleliza, autoría = 1-2 candidatos extra del grande que pasan los mismos tests.

**Experimento.** 40 tareas bug-fix/feature (subconjunto SWE-bench-lite + sintéticas con tests oracle), ejecutor `qwen2.5-coder:7b`. Arms A (single-pass), B (BSVS@3 + Reflexion), C (BSVS@5), D (BSVS@3 + 1-2 remoto). **Diagnóstica CLAVE: oracle@N vs selected@N** — si oracle alto pero selected bajo, el bug está en el SELECTOR (timeouts/infra-failure mal clasificado), no en el muestreo. **Refutado si** B no sube ≥10 pts sobre A → el parche correcto NO está en N muestras → se rechaza BSVS, queda sólo la escalada.

**Costo.** Sin 2ª GPU. BSVS@3 = ~3x latencia + 3x test TARGETED. Cuello real: latencia serial de N muestras + N tests en local → por eso N=3, gating conservador y targeted-test son no-negociables.

**Honestidad final.** Para steps donde el parche correcto NO está en ninguna de N muestras (conocimiento que el modelo no tiene), **no hay forma seria de subir el techo sin un modelo grande.** BSVS sólo captura lo que ya existe en la cola de la distribución del chico. La memoria/paginación no resuelve esto.

---

## 3. Roadmap

### 3.1 Infraestructura compartida — UNA de cada cosa, no cuatro

El peor error de integración sería construir 4 tablas, 4 retrievers, 4 slots de config, 4 routers. Los cuatro planes convergen en **dos piezas reusadas**:

| Pieza única | Qué es | Reusa (verificado) | Sirve a |
|---|---|---|---|
| **Tabla `recipes`** | Memoria polimórfica por `kind`: `design_brief` / `science_skeleton`+evidence / `step_patch_recipe`. Campos: `signature_text, embedding BLOB, payload_json, evidence_pack_json, provider_provenance, lang_tags, verification_passed_at_harvest, success_count, fail_count` | patrón `findings`/`artifacts` con `embedding BLOB` + FTS5 + triggers (`db/service.py:60-144`) | design, intelligence, context |
| **`recipe_retriever.py`** | Recall en 2 etapas: cosine (MiniLM `dedup._get_embed_fn`) + FTS5/BM25, **filtro DURO por lang/stack ANTES del kNN**. Tool read-only `search_recipes` al tope de `run_planner` | embedder MiniLM ya shippado (~11ms MNN/~115ms ONNX) | los 3, sólo cambia el filtro `kind` |
| **Tier router** `get_litellm_params_for_strong()` | UN slot `INFINIBAY_STRONG_LLM_*` con fallback a `LLM_MODEL`. Trigger compuesto compartido decide cuándo escalar | clone de `get_litellm_params_for_behavior` (`llm.py:225`); **`COUNCIL_MODEL` ya existe (`settings.py:182`) → REUSARLO, no duplicar** | los 4 |
| **Harvest gate único** | Cosecha post-Review SÓLO sobre `automated_checks.verification_passed==True`, nunca por `verdict==APPROVED` | señal determinista verificada (`review_engine.py:168-185`) | los 4 |
| **VerificationEngine + targeted-test + git-worktree** | targeted-test (nodeids afectados) y aislamiento worktree, net-new pero compartidos | `verification_engine.py` `verify()` | BSVS (context), selector PlanForge (design), harvest (todos) |
| **Embedder MiniLM** | 384-dim para retrieval, dedup, clustering, vote-margin | `dedup._get_embed_fn` ya presente | los 4. **Cero modelo nuevo aquí** |

> **Único modelo nuevo en todo el sistema integrado: el cross-encoder NLI ~80-100MB ONNX-CPU (problema intelligence), y SÓLO en commit-time de autoría remota. CERO segunda GPU en ningún plan.**

### 3.2 Composición con la contención: contención = PISO, capacidad = TECHO

No son alternativas: **la capacidad se APILA sobre la contención** y depende de ella en cuatro contratos verificados.

**ABAJO (contención, capa 0):**
- (a) **ContextGovernor + budget honesto provider-aware + cap head+tail** → mantiene barata e inyectable cada llamada (handoff k=3 ~1-2k tokens, digest compacto al oráculo, plan remoto verboso podado). Sin esto, inyectar skeleton+evidence satura la ventana del 7B y degrada el propio retrieval.
- (b) **evidence-gate post-edit** → PORTERO DE ADMISIÓN de TODO output de capacidad: candidato BSVS, receta mal-aterrizada, plan del oráculo. Defensa final cuando el plan remoto es correcto-pero-el-7B-lo-aterriza-mal.
- (c) **gate generator-verifier + señal `verification_passed`** → EXACTAMENTE la señal en que gatea HARVEST. Sin ella la memoria se envenena con recetas cuya única credencial es que un 7B débil dijo OK.
- (d) **critic_dispatcher FALSE_CONFIDENCE + victory_lap detector + cap de steps** → el GATILLO determinista que marca step flagged-hard / dispara BSVS / fuerza escalada.

**Regla de oro:** la contención impide que el 7B actúe más capaz de lo que es (tapa el piso); la capacidad inyecta capacidad real DESDE FUERA del modelo (sube el techo). **Quitar la contención no baja el techo — lo HACE PELIGROSO:** "plan confiadamente-equivocado con prosa grado-glm" ejecutado sin freno.

**Flujo apilado por turno:** contención poda contexto → RETRIEVE recipes (local, gratis) → si hit: planner ADAPTA en vez de inventar → si miss y hay strong tier: oráculo AUTORA skeleton → 7B ATERRIZA (locked, field-diff check) → LoopEngine ejecuta, BSVS en steps flagged-hard → evidence-gate admite/rechaza cada parche → Review con señal determinista → HARVEST cristaliza a la memoria.

### 3.3 Orden recomendado

1. **Infra compartida (recipe memory + retriever + harvest gate).** Mayor ROI, menor riesgo: clona patrones existentes, reusa MiniLM, costo despreciable. Único componente que sube el techo en local-puro (warm cache). Sin esto, el oráculo no tiene dónde cristalizar → la dependencia remota nunca amortiza a cero.
2. **DESIGN.** Materialización más directa de la infra, feasibility=5. Reusa `_render_handoff` (`:202`) y el harvest gate. Valor warm-path inmediato en las tareas más comunes (rate-limit, soft-delete, pagination). El strong-tier oráculo se introduce aquí y queda reutilizable.
3. **CONTEXT (BSVS).** Segundo mayor salto de pass@1 y NO depende de corpus sembrado (día-uno en cualquier repo con suite verde). PERO requiere net-new arriesgado (worktree del sub-loop stateful + targeted-test). Va tercero por su riesgo de implementación.
4. **INTELLIGENCE.** Mayor techo potencial pero mayor dependencia externa: requiere SEED CORPUS + el único modelo nuevo (NLI) + provenance-check sobre web_search real. En local-puro sin seed corpus el techo NO sube. Va último porque reusa toda la infra previa; sólo añade NLI gate + seed corpus.
5. **CONFIDENCE.** Valioso pero en gran parte RECOMBINACIÓN de piezas ya entregadas (strong tier de prio 2, executable_check, clustering MiniLM). Su leg que sube el techo ES el mismo strong tier. Su mayor incógnita (diversidad real de N samples del 7B) debe medirse antes de invertir; si los samples colapsan, gran parte muere y queda sólo el remoto que ya tenemos.

### 3.4 Biggest bet vs. cheapest high-impact

**Biggest bet: el TIER ROUTER** (modelo grande remoto autora/decide, 7B local aterriza/ejecuta) como fuente única de techo en tarea novedosa. Es la apuesta más grande porque (1) los 4 planes convergen en que sin un modelo grande en algún momento, el techo de un 7B NO sube; (2) introduce dependencia de red + costo + el **riesgo central no resuelto: el GROUNDING del 7B.** Bindear un skeleton abstracto autorado por glm al control-flow real del repo ES juicio de diseño disfrazado — el cuello de botella verificado. **Un skeleton bueno-pero-mal-aterrizado es peor que un plan mediocre coherente.** La apuesta sólo paga si (a) el field-diff lock + `grounding_conflict` detectan el binding-que-contradice-un-assumption (NO probado que el 7B sepa detectarlo), y (b) el evidence-gate atrapa lo que se cuele. Es la palanca de mayor techo Y de mayor riesgo de *capability-laundering*.

**Cheapest high-impact: el Recipe Bank WARM-PATH de DESIGN** (prio 2 sobre la infra de prio 1). Costo casi-cero verificado: cero modelo/GPU/red extra, reusa MiniLM (+1 embed ~11-115ms), +1 INSERT post-Review, ~1.5KB/receta, handoff +1-2k tokens una vez. Impacto alto y DURABLE: convierte "generar arquitectura" (debilidad verificada, arXiv:2404.17140) en "elegir entre 3 diseños ya-vetados-por-tests-reales y mapear archivos". **Es lo único que sube el techo offline/local-puro una vez sembrado, y compone a través de sesiones — algo que ningún gate de contención hace.** Es el ancla del sistema: el strong tier amortiza a cero porque sus skeletons aceptados se cosechan aquí.

### 3.5 Costo agregado honesto en hardware de consumo

**Sé brutalmente honesto: el techo real lo pone el modelo grande remoto, y eso tiene costo agregado no-trivial.**

- **LATENCIA (el peor enemigo en local, 1 GPU, serial):** BSVS@3 serializa N forward passes + N tests → SIN targeted-test (net-new obligatorio) es 3-6 min/step, prohibitivo; con él baja a decenas de seg. Plan-search N=4 = ~4x latencia del planner (~24s). **Estas SE APILAN:** una tarea dura podría pagar plan-search (24s) + BSVS en varios steps (varios min) + N llamadas remotas. **Mitigación no-negociable: gating de complejidad agresivo (N>1 SÓLO en <30% de steps flagged-hard)**, si no el sistema se vuelve inusable en local.
- **TOKENS:** handoff k=3 +1-2k tokens (barato, una vez). Plan-search remoto = Nx (paraleliza). Oracle skeleton ~3k out, sub-centavo. NLI sobre N claims×passages en CPU = segundos (subestimado por el propio plan — **medir antes de comprometer**).
- **RED/$:** dependencia remota OPCIONAL con fallback gracioso a 100% local en los 4. **Backstop obligatorio día-1 (no opcional): cap de presupuesto remoto por-sesión compartido** por los 4 consumidores del strong tier.
- **MODELOS NUEVOS EN DISCO:** uno solo — el NLI ~80-100MB (intelligence), ONNX-CPU. **CERO segunda GPU.** Todo lo demás reusa MiniLM.
- **RIESGO COMPUESTO MÁS PELIGROSO: la falacia "compute = capacidad".** El control A' pre-registrado (best-of-1 con el mismo presupuesto de tokens en una call más larga) **DEBE correrse:** si A' iguala a la versión con oráculo/BSVS, el gain fue compute, no capacidad inyectada, y todo el costo agregado fue desperdiciado.

---

## Cierre: tres verdades que ningún plan individual ve completas

1. **Los 4 planes son UN solo sistema con dos piezas reusadas:** una MEMORIA recuperable (tabla `recipes` polimórfica por `kind`) y un TIER ROUTER (strong remoto autora, 7B aterriza). Design = warm-path; intelligence = memoria científica + NLI; context = memoria de step-patches + selector ejecutable; confidence = el router como capa de decisión. **Una tabla, un retriever, un router, un harvest gate, un VerificationEngine.**

2. **El eje no es el que el título de cada problema sugiere.** "Contexto" no se resuelve resumiendo (el problema es no-saber, no recordar); "confianza" no se resuelve con entropía (el 7B falla por confident-and-consistent confabulation). En los 4, el eje real es **externalizar capacidad a memoria verificada o modelo grande** — coherente con la causa raíz: la capacidad no está en los pesos del 7B.

3. **Honestidad que debe llegar al usuario explícitamente:** en local-puro + cold-start + sin recetas sembradas + sin red, el techo NO sube en NINGUNO de los 4 — degrada limpiamente al comportamiento de hoy (sin breakage, sin ganancia). La memoria es el mecanismo que hace que UNA autoría remota (o un seed corpus shippeado) sirva infinitas veces offline. El compounding cross-task es ESPECULATIVO y es la kill-condition probable de varias piezas — por eso context degrada Recipe Forge a skills semilla deterministas y **CORTA persistir código-LLM** (`SANDBOX_ENABLED=False`, shell `subprocess.run` pelado → agujero RCE peor que el baseline). Cada plan debe shippear con su falsación pre-registrada activa, especialmente: (i) ablación-de-barajado, (ii) oracle@N vs selected@N en BSVS, (iii) el control de compute A'. **Si esas falsaciones disparan, la pieza se mata, no se parchea.**