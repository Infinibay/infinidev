# Bugfix task policy: model-specific E2E

Fecha: 2026-08-11

## Resultado

`bugfix.developer@3` es un fragmento genérico y outcome-focused. Ganó un par
prompt-only en GPT-5.6 Terra y queda aprobado solamente para esa ruta. En
MiniMax M3 no superó el gate; la detección sigue activa, pero el fragmento no
se inyecta.

La composición admite además fragmentos aditivos filtrados por ruta de modelo.
Cada fragmento genérico o específico conserva su propio ID, versión, hash,
presupuesto y aprobación E2E. Un ajuste MiniMax experimental se evaluó y se
eliminó porque empeoró el resultado.

## Detección frente a tratamiento

El caso E2E se clasificó como `bugfix` por la señal literal. La cabeza lineal
también colocó `bugfix.root_cause` primero, con score 0,287, pero su margen de
0,058 era menor que el umbral global de 0,150.

El router permite ahora que una categoría literal resuelva ese empate si la
primera candidata de la cabeza coincide y supera el threshold. Esto no agrega
una operación ni concede autoridad: solamente convierte evidencia
`literal-signal` en el acuerdo auditable `mini-head+literal`. Para requests sin
categoría literal, el margen estricto y el acuerdo contrastivo siguen siendo
obligatorios.

También se amplió la autoridad literal para verbos operativos naturales como
`correct`, `restore`, `restablecer` y `rétablir`. Una clasificación semántica
nunca concede escritura por sí sola.

## Prompt nuevo

El prompt anterior dictaba cuándo explorar, cuántas veces probar y cuándo
terminar. Eso interfería con el protocolo del loop y aumentaba trabajo en
MiniMax. La versión 3 expresa solamente resultado, scope y evidencia:

> Repair the narrowest demonstrated contract violation. Keep unrelated
> behavior unchanged and validate the reproduced failure plus any directly
> affected contract.

Esto sigue la recomendación de usar prompts compactos y outcome-focused para
GPT-5.6: el modelo ya infiere intención y no necesita que se le prescriba cada
paso. Véase la [guía oficial de prompting para GPT-5.6][gpt56-prompting].

[gpt56-prompting]: https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6

## GPT-5.6 Terra

Ambas condiciones cambiaron solamente `src/pagination.py`, terminaron `done` y
pasaron los cuatro tests y el verificador externo. Baseline y candidate
recibieron el mismo user prompt y schemas; la diferencia fue únicamente
`bugfix.developer@3`.

| Métrica | Baseline | Candidate | Reducción |
| --- | ---: | ---: | ---: |
| prompt tokens | 171.887 | 143.813 | 16,3 % |
| completion tokens | 1.947 | 1.697 | 12,8 % |
| tool calls | 17 | 16 | 5,9 % |
| latencia | 171,02 s | 127,07 s | 25,7 % |

Los nueve gates pasaron. El rollout exacto es
`openai_subscription:gpt-5.6-terra -> bugfix.developer@3`; Sol y Luna no
heredan esa aprobación.

## MiniMax M3

El primer candidate combinó el núcleo genérico con un ajuste MiniMax que pedía
no ampliar discovery. Aunque terminó correctamente, empeoró frente al baseline:

| Métrica | Baseline | Generic + MiniMax | Cambio |
| --- | ---: | ---: | ---: |
| prompt tokens | 93.990 | 136.246 | +45,0 % |
| completion tokens | 1.513 | 3.681 | +143,3 % |
| tool calls | 11 | 13 | +18,2 % |
| latencia | 42,05 s | 61,03 s | +45,1 % |

El ajuste específico fue eliminado. Un rerun candidate-only con el núcleo
genérico terminó verde y produjo 93.168 prompt tokens, 1.662 completion tokens,
10 tools y 41,17 s. Es un resultado mixto: -0,9 % prompt, +9,8 % completion,
-9,1 % tools y -2,1 % latencia. Por tanto, MiniMax conserva únicamente la
aprobación previa de `refactor.developer@1`.

## Límite del mini-modelo

La cabeza v2 acierta la primera clase en este caso, pero su margen muestra que
todavía separa mal algunos bugfix de refactor, review y performance. Se probó
agregar más síntomas y negativos sintéticos al fit; la precisión selectiva del
holdout empeoró y una calibración fail-closed colapsó la cobertura. El
experimento se revirtió y el artefacto empaquetado sigue siendo v2.

El siguiente intento debe usar requests naturales revisados, splits nuevos por
familia y calibración por clase o una cabeza pequeña no lineal. No se relajará
el gate semántico global usando este caso ya observado.

## Reproducción

```bash
uv run python -m bench.task_policy_improvement_eval \
  bench/task_policy_bugfix_terra.gates.json \
  /path/to/terra-observations.jsonl
```

Una sola familia sirve para aprobar o rechazar este fragmento en esta ruta; no
demuestra mejora general de bugfix ni permite transferir la aprobación a otros
modelos.
