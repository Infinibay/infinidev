# Modelos y controles de razonamiento

Contratos revisados el 6 de septiembre de 2026. Los catálogos en línea siguen siendo
la fuente de disponibilidad de cada cuenta; el catálogo incorporado permite selección
sin descubrimiento. Agregar un modelo no cambia el modelo guardado por el usuario.

## Uso

`/effort` muestra los valores de la combinación **proveedor + modelo** seleccionada,
marca el valor efectivo e identifica el mecanismo. `/effort <valor>` valida ese conjunto
y guarda la preferencia. Elegir un nivel activo también activa `THINKING_ENABLED`:
antes era posible guardar `high` y seguir enviando thinking desactivado.

La CLI clásica y la TUI utilizan el mismo catálogo y representación. Si se cambia de
modelo, una preferencia guardada que no existe en el nuevo modelo se traduce a un nivel
admitido y el listado muestra la traducción. Por ejemplo, `high` pasa a `xhigh` en Qwen
3.8; `medium` pasa a `high` en GLM 5.3. El comando rechaza valores nuevos inválidos.

`custom` identifica un presupuesto de tokens, configurado mediante
`INFINIDEV_THINKING_BUDGET_TOKENS`. `off`/`none` se ofrecen solamente cuando el contrato
lo permite. Un límite de salida `max_tokens` por sí solo no se anuncia como esfuerzo.

## OpenAI

Se agregó `gpt-6-astra`, con contexto de 1.050.000 tokens, a la API pública. Infinidev
usa Responses para Astra y GPT-5.5/5.6; conserva las credenciales y el endpoint propio
de Codex cuando se usa la suscripción. Astra exige Responses para function calling y
admite `low`, `medium`, `high`, `xhigh`, `max`; no admite `none` ni `minimal`.
[Guía de Astra](https://developers.openai.com/api/docs/guides/latest-model),
[modelo](https://developers.openai.com/api/docs/models/gpt-6-astra).

| Modelo de API | Valores |
| --- | --- |
| GPT-6 Astra | low, medium, high, xhigh, max |
| GPT-5.6 Sol / Terra / Luna / alias 5.6 | none, low, medium, high, xhigh, max |
| GPT-5.5, 5.4, 5.4-mini/nano, 5.2 | none, low, medium, high, xhigh |
| GPT-5.5 Pro | medium, high, xhigh |
| GPT-5.1 | none, low, medium, high |
| GPT-5, mini, nano | minimal, low, medium, high |

Fuentes de los contratos de API:
[Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
[Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra),
[Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna),
[5.5](https://developers.openai.com/api/docs/models/gpt-5.5),
[5.5 Pro](https://developers.openai.com/api/docs/models/gpt-5.5-pro),
[5.1](https://developers.openai.com/api/docs/models/gpt-5.1).

En **Codex**, los valores y ventanas vienen de `~/.codex/models_cache.json`, incluidos
niveles adicionales publicados por ese catálogo. Sin archivo se ofrecen los tres niveles
conservadores existentes; no se copian las ventanas ni los permisos de la API pública.

El SDK bloqueado descarta niveles de Responses que desconoce, incluido `max`, al recibir
un string. El adaptador usa su camino de diccionario `{"effort": "max"}` para conservar
el valor en el JSON HTTP. Las pruebas ejecutan esa transformación real sin llamar modelos.
También se eliminan parámetros de sampling incompatibles y se adapta la retención de
caché antigua de Astra en la frontera final, cubriendo llamadas auxiliares directas.

## Anthropic

Se agregó `claude-fable-5-1` con contexto de un millón de tokens. Su thinking adaptativo
es obligatorio: no se envían presupuestos manuales ni `thinking.type=disabled`.
Con thinking activo se usa tool choice automático. También se omiten overrides de
sampling que los modelos nuevos rechazan, incluso con thinking desactivado.
[Migración de Fable 5.1](https://platform.claude.com/docs/en/models/fable-5-1/migration-guide),
[parámetros retirados](https://platform.claude.com/docs/en/about-claude/model-deprecations).

| Modelo | Control |
| --- | --- |
| Fable 5/5.1, Mythos 5/5.1 | output_config.effort: low, medium, high, xhigh, max; thinking obligatorio |
| Opus 5, 4.8, 4.7; Sonnet 5 | los cinco esfuerzos y opción off |
| Opus 4.6; Sonnet 4.6 | low, medium, high, max y opción off; sin xhigh |
| Opus 4.5 | low, medium, high mediante output_config, con thinking manual; opción off |
| Haiku/Sonnet 4.5 y modelos de thinking manual anteriores | off o presupuestos low, medium, high, custom |

Los nombres de esfuerzo son señales de comportamiento, no cantidades equivalentes de
tokens entre modelos. Los presets manuales reservan espacio de salida por encima del
presupuesto de thinking. [Referencia de effort](https://platform.claude.com/docs/en/build-with-claude/effort).
Mythos se reconoce si se configura explícitamente; no se ofrece como acceso general.

## GLM y Qwen

Se agregaron `glm-5.3` y `glm-5.3-flash` para API y Coding Plan, ambos con contexto de
un millón de tokens. Ambos requieren thinking y ofrecen `low`, `high`, `max`.
Flash incorpora entrada visual. Se corrigió también el contexto de GLM 5.2 a un millón.
[GLM 5.3](https://docs.z.ai/guides/llm/glm-5.3),
[GLM 5.3 Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash),
[GLM 5.2](https://docs.z.ai/guides/llm/glm-5.2).

Qwen Cloud incorpora `qwen3.8-max`, `qwen3.8-flash`, `qwen3.8-27b` y
`qwen3.8-2.4t-a95b`. Token Plan incorpora Max estable y Flash; conserva su endpoint
de suscripción. Los modelos 3.8 ofrecen `low`, `medium`, `xhigh`; el adaptador envía
`enable_thinking` y `reasoning_effort` en el cuerpo extendido, sin combinar esfuerzo
con `thinking_budget`. El modelo 2.4T requiere thinking; los otros permiten `off`.
[API de Qwen](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions),
[thinking por modelo](https://www.alibabacloud.com/help/en/model-studio/deep-thinking),
[Token Plan](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/token-plan-personal-overview).

## Otros proveedores

| Ruta | Mecanismo |
| --- | --- |
| OpenRouter | catálogo público por modelo; reasoning.effort, enabled o max_tokens |
| Gemini 3 | thinkingLevel, con niveles distintos entre Pro y Flash |
| Gemini 2.5 | thinkingBudget; presets de tokens |
| Ollama GPT-OSS | think: low, medium, high |
| Ollama con thinking booleano | think: true/false |
| llama.cpp / vLLM, modelos identificados | chat_template_kwargs del modelo |
| DeepSeek V4 | thinking.type y reasoning_effort low/high/max |
| Mistral Small / Medium 3.5 con esfuerzo | reasoning_effort none/high |
| Modelo o API sin contrato conocido | no se ofrecen valores inventados |

OpenRouter distingue modelos con razonamiento obligatorio, esfuerzos enumerados y
presupuestos. El catálogo se consulta sin credenciales y se conserva cinco minutos;
si no está disponible, el comando lo indica. Los routers dinámicos no reciben una lista
de esfuerzos de un modelo fijo.
[OpenRouter](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens),
[Ollama](https://docs.ollama.com/capabilities/thinking),
[Gemini](https://ai.google.dev/gemini-api/docs/thinking),
[DeepSeek](https://api-docs.deepseek.com/guides/thinking_mode/),
[Mistral](https://docs.mistral.ai/studio/conversations/reasoning).

La configuración no convierte servidores OpenAI-compatible arbitrarios en OpenAI:
sin contrato identificado se conserva su compatibilidad anterior. Estas verificaciones
cubren selección, parámetros y transformaciones del SDK. La disponibilidad de cuenta,
cuotas y aceptación HTTP en cada despliegue requieren una llamada real a ese despliegue.
