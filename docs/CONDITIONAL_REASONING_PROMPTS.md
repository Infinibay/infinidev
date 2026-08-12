# Conditional Reasoning Prompts

Fecha: 2026-08-11

Estado: integrado. La clasificación usa el embedding
`ken/static-qwen3-r512-v2` y una cabeza lineal empaquetada. Las intervenciones
están habilitadas, pero necesitan además evidencia observable y son acotadas
por tarea.

## Flujo real

```text
respuesta del proveedor
  -> texto de reasoning explícitamente expuesto por la API
  -> ventanas de hasta 1.600 caracteres, máximo 4
  -> embedding ken/static-qwen3-r512-v2 (1024 dimensiones)
  + 9 señales observables del loop
  -> mini-head lineal de 28 KB
  -> categoría o abstención
  -> veto observable independiente
  -> un prompt correctivo en el siguiente turno
```

El mini-modelo sí participa en la decisión. Las reglas no sustituyen la
clasificación; sirven como veto para impedir que una semejanza semántica aislada
interrumpa un run saludable.

## Qué entrega cada proveedor

| Proveedor | Forma nativa relevante | Forma normalizada en Infinidev |
| --- | --- | --- |
| OpenAI Responses | items/eventos de reasoning, texto o summary según modelo y configuración | `reasoning_content` cuando LiteLLM lo expone |
| Anthropic | bloques `thinking`, `signature` y posiblemente `redacted_thinking` | texto visible en `reasoning_content`; bloques completos en `thinking_blocks` |
| Gemini | parts con `thought: true`; `thoughtSignature` opaca para continuidad | texto en `reasoning_content`/`thinking_blocks`; firmas en `provider_specific_fields` |
| MiniMax | `<think>` en `content`, o `reasoning_details` con `reasoning_split=true` | `reasoning_content` en la ruta LiteLLM actual; fallback para `reasoning_details` y tags |
| Z.AI/GLM | `delta.reasoning_content`, incluido reasoning intercalado con tools | `reasoning_content` |
| Ollama/OpenAI-compatible | `thinking`, `reasoning_content` o tags `<think>` según servidor | `reasoning_content` o promoción local de tags |

Referencias oficiales:

- [OpenAI Responses streaming reasoning events](https://platform.openai.com/docs/api-reference/responses-streaming/response/content_part)
- [Anthropic thinking and tool workflows](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- [Gemini thinking summaries and signatures](https://ai.google.dev/gemini-api/docs/generate-content/thinking)
- [MiniMax OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api)
- [Z.AI thinking mode](https://docs.z.ai/guides/capabilities/thinking-mode)

No se asume que este texto sea chain-of-thought privado o crudo. OpenAI,
Anthropic y Gemini pueden entregar resúmenes. Infinidev sólo analiza texto que
el proveedor decidió exponer. Nunca convierte firmas, payloads cifrados ni
`redacted_thinking` en texto para el embedding.

Las firmas y bloques opacos sí se devuelven intactos en el historial cuando el
protocolo los necesita. La intervención se agrega únicamente después de cerrar
el bloque `assistant -> tool results`; insertarla antes invalidaría function
calling en varios proveedores.

## Categorías del mini-modelo

| Categoría | Prompt dinámico | Veto observable requerido |
| --- | --- | --- |
| `excessive_exploration` | actuar sobre la evidencia cargada y probar el cambio mínimo | tarea de modificación, al menos 3/4 de presión de discovery, sin edit ni test |
| `retry_loop` | cambiar cwd, input, argumentos o hipótesis | fallos observados y operación equivalente repetida |
| `premature_completion` | continuar el requisito abierto | `step_complete` y trabajo requerido pendiente |
| `speculative_claim` | mantener la explicación como hipótesis y buscar una prueba | sin evidencia y score extraordinariamente por encima del umbral |
| `verification_gap` | ejecutar el check mínimo que cubre el cambio | edit observado, ningún test y tentativa de completar |
| `healthy_progress` | ninguno | no aplica |
| `uncategorized` o abstención | ninguno | no aplica |

Una clasificación puede registrarse sin producir un prompt. Por ejemplo, el
modelo puede reconocer lenguaje de verificación incompleta mientras todavía se
está implementando; sólo interviene si además existe un edit y el modelo intenta
cerrar sin test.

## Modelo y dataset

El artefacto está en
`src/infinidev/engine/behavior/artifacts/reasoning_pattern_head_v1.npz`.
Contiene pesos `float32`, orden de clases, features, umbrales, hashes de corpus
y la identidad exacta del espacio de embedding. La carga falla cerrada si el
espacio Qwen, el shape o el schema no coinciden.

El corpus inicial tiene 61 ejemplos de calibration, 25 de validation y 25 de
holdout, con inglés, español, portugués, francés, alemán e italiano. Incluye
negativos difíciles que expresan hipótesis con cautela, tests diagnósticos,
planes normales y tareas read-only. El holdout sintético alcanzó:

- 96% de cobertura;
- 100% de precisión selectiva;
- 0 activaciones inseguras.

Esto valida el contrato del artefacto, no demuestra generalización productiva.
Los umbrales deben evolucionar con thinking natural revisado y splits por
proveedor/modelo.

## E2E MiniMax M3

La primera ejecución real candidate usó una copia aislada de jsmn:

- MiniMax entregó 15 bloques mediante `reasoning_content`;
- el mini-modelo produjo eventos auditables con score, umbral, features y
  origen del campo;
- `excessive_exploration` superó el veto después de tres lecturas/discovery sin
  edit ni test;
- el run terminó `done`, cambió sólo `jsmn.h` y `test/tests.c`;
- el verificador C pasó en normal, strict, parent-links y strict+parent-links;
- se usaron 21 tool calls.

La misma trayectoria reveló que `speculative_claim` era demasiado sensible al
primer pensamiento del run. Su gate se endureció de +0,08 a +0,30 sobre el
umbral: una hipótesis inicial normal ya no basta para intervenir.

## Configuración

```text
INFINIDEV_ADAPTIVE_RUNTIME_REASONING_ENABLED=true
INFINIDEV_ADAPTIVE_RUNTIME_REASONING_SHADOW_MODE=false
INFINIDEV_ADAPTIVE_RUNTIME_MAX_INTERVENTIONS=2
```

Shadow mode sigue ejecutando y registrando el mini-modelo, pero no modifica el
prompt. Una intervención nunca amplía permisos, scope, herramientas, autoridad
de escritura, commit, push o publicación.

## Siguiente dataset

Las siguientes campañas deben capturar, con opt-in y sanitización, únicamente
la ventana expuesta que produjo desacuerdo, abstención o intervención. Las
prioridades son:

- hipótesis iniciales normales como negativos de `speculative_claim`;
- retries que cambian materialmente de estrategia;
- razonamiento intercalado antes y después de tool results;
- summaries de OpenAI/Gemini frente a thinking más literal de MiniMax/GLM;
- trayectorias saludables largas para medir falsos bloqueos;
- evaluación baseline/candidate pareada, no sólo éxito de una candidate.
