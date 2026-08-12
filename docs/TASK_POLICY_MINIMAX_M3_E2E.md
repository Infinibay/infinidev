# Conditional Task Policies: MiniMax M3 E2E

Fecha: 2026-08-11

> Evaluación histórica anterior al compositor prompt-only y al rollout por
> fragmento. La evidencia actual está en
> [`TASK_POLICY_IMPROVEMENT_E2E.md`](TASK_POLICY_IMPROVEMENT_E2E.md); estas
> cifras se conservan porque motivaron el diseño evidence-gated.

## Resultado

Las políticas condicionales se probaron con el loop real de Infinidev,
MiniMax M3, herramientas reales y workspaces aislados. Baseline y candidate
pasaron los dos verificadores deterministas y respetaron el scope. La candidate
no mejoró el pass rate y consumió más contexto, output y herramientas.

Por eso el rollout queda en **shadow mode**: se calcula y registra el perfil,
pero no se inyectan políticas en prompts de producción por defecto.

## Diseño de la comparación

- modelo observado: `minimax:MiniMax-M3:api-observed-2026-08-11`;
- dos fixtures aprobados: corrección con selección de tests y code review
  read-only;
- workspace, sesión y agente nuevos por ejecución;
- baseline sin `TaskProfile` versus candidate con la política determinista;
- semantic routing y fallback LLM desactivados para aislar el tratamiento;
- verificación externa después del loop;
- sin retries automáticos ni paralelismo.

Los artefactos completos están bajo
`bench/runs/20260811-task-policy-minimax-m3-e2e-retry1/`. La primera carpeta
sin sufijo contiene intentos fallidos por aislamiento de red y no cuenta como
evidencia del modelo.

## Métricas

| Métrica | Baseline | Candidate | Cambio |
| --- | ---: | ---: | ---: |
| tareas con verificador verde | 2/2 | 2/2 | igual |
| prompt tokens | 245.658 | 300.083 | +22,2 % |
| completion tokens | 12.355 | 16.052 | +29,9 % |
| tool calls | 25 | 30 | +20,0 % |
| latencia total | 287,72 s | 289,68 s | +0,7 % |
| cambios fuera de scope | 0 | 0 | igual |

Cuatro ejecuciones no permiten estimar calidad general. Sí permiten rechazar
la hipótesis más fuerte de que esta versión ya ofrece una mejora sin coste.

## Observaciones por tarea

### Corrección y selección de tests

Ambas condiciones corrigieron únicamente `src/tags.py` y el verificador
externo completo pasó. El baseline usó `casefold()` y ejecutó también el
contrato consumidor. La candidate usó `lower()` y ejecutó tres tests enfocados,
pero no el test consumidor durante su propio loop. El verificador externo
evitó un falso positivo de completitud.

La candidate necesitó 160.556 prompt tokens, 5.303 completion tokens y 18
herramientas, frente a 114.061, 3.144 y 14 en baseline. No hay evidencia de
mejor assurance; sí hay una regresión clara de coste.

### Code review read-only

Ambas condiciones crearon solamente `REVIEW.md` y pasaron el verificador. La
primera candidate produjo un informe más largo y elevó hipótesis sobre callers
no visibles a blockers. Una política corregida añadió explícitamente la
separación entre hechos, supuestos y preguntas abiertas; el rerun terminó en
104,50 s, 117.941 prompt tokens, 6.607 completion tokens y 13 herramientas,
pero todavía presentó como explotables varias inferencias no demostradas.

La instrucción correcta no garantizó el juicio correcto. Este caso necesita
un rubric humano o verificadores de claims, no más texto prescriptivo.

### Política de bugfix expandida

También se probó una variante más detallada de `bugfix.root_cause`. El modelo
entró en exploración excesiva y repetición, sin completar un artefacto
evaluable; la ejecución se interrumpió para limitar coste. Esa variante se
revirtió. Un run interrumpido no se mezcla con los pares completos, pero sí
funciona como evidencia de riesgo ante prompt inflation.

## Routing semántico local

El tratamiento E2E anterior aísla políticas deterministas. Por separado, el
router semántico se validó sobre 112 paráfrasis que evitan el vocabulario
literal de operaciones:

| Métrica | Recuperación contrastiva |
| --- | ---: |
| coverage | 58,9 % |
| precisión entre selecciones | 100 % |
| exact match total | 73,2 % |
| falsas activaciones neutrales | 0 % |
| falsa autoridad de escritura | 0 % |

El backend está fijado a `ken/static-qwen3-r512-v2`, registra el `space_id`,
usa positivos y negativos difíciles, exige margen sobre el runner-up y
abstiene ante ambigüedad. No concede autoridad.

Un head lineal experimental de 23.646 bytes elevó coverage a 78,6 % en un
holdout separado, con 96,6 % de precisión selectiva. Sus tres errores fueron
casos españoles de bugfix clasificados como feature. Como no alcanza el gate
de precisión y ese holdout ya fue observado, el head no se despliega.

## Reproducción

Routing determinista y semántico:

```bash
uv run python -m bench.task_policy_eval
uv run python -m bench.task_policy_semantic_eval
uv run python -m bench.task_policy_linear_head \
  --artifact /tmp/infinidev-task-policy-head.npz
```

El runner E2E usa:

- `bench/task_policy_e2e.minimax-m3.json`;
- `bench/task_policy_e2e.conditions.json`;
- `bench/agent_task_run.py`.

Repetir las llamadas al proveedor requiere una API key y constituye una nueva
evaluación; los artefactos existentes conservan la evidencia de esta corrida.

## Decisión

1. Mantener authority, negación, comillas y acciones externas en una capa
   literal fail-closed.
2. Activar el clasificador semántico local únicamente en shadow mode.
3. Mantener el fallback LLM apagado.
4. No desplegar todavía el head lineal ni la inyección de políticas.
5. Recopilar requests naturales etiquetados y congelar un nuevo holdout antes
   de reconsiderar el head.
6. Evaluar `Task.kind` producido por el planner como evidencia estructurada,
   sin añadir otra llamada LLM.
