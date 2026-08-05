# Capacidad de generación de imágenes

> Revisión de evidencia: **2026-08-03**. Las capacidades de productos alojados pueden
> cambiar sin aviso; los enlaces a repositorios fijan el commit revisado cuando es
> posible. Un dato sin endpoint, autorización o estabilidad verificables se registra
> como evidencia, nunca como una ruta ejecutable.

Esta nota separa cuatro cosas que suelen confundirse:

1. **Producto Codex/ChatGPT**: una función operada por OpenAI dentro de sus propias
   aplicaciones y límites de suscripción.
2. **API pública de OpenAI**: una superficie para integradores, autenticada con API
   key y facturada a la organización/proyecto de API.
3. **Visión o adjuntos**: capacidad de recibir y comprender una imagen; no crea una.
4. **Extensiones de terceros**: código, MCP o plugins que un agente puede cargar; no
   convierten esa función en soporte nativo del agente ni transfieren credenciales.

## Conclusión operativa

Codex sí genera imágenes en superficies oficiales. La documentación de OpenAI ofrece
la función en la app, ChatGPT web, CLI e IDE; en CLI se puede invocar explícitamente
con `$imagegen`. La ruta integrada usa `gpt-image-2` y consume límites generales de
Codex. Esto demuestra una **capacidad del producto Codex**, no un contrato público
para que Infinidev reutilice el OAuth o la sesión de ChatGPT.

Para integradores, OpenAI documenta dos mecanismos distintos: Image API
(`POST /v1/images/generations`, modelos `gpt-image-*`) y la herramienta
`image_generation` de Responses API. La guía de Codex dirige el uso programático a
esas APIs y, para lotes, pide `OPENAI_API_KEY` y avisa que aplica el precio de API.
Por tanto:

- un token OAuth de ChatGPT/Codex **no es una credencial válida documentada para
  OpenAI Images**;
- la suscripción de ChatGPT/Codex y la API pública tienen identidad, autorización,
  cuota y facturación diferentes;
- Infinidev solo puede publicar `generate_image` para un perfil API explícito,
  revisado y autenticado con API key;
- cualquier señal del catálogo Codex, incluso `supports_image_generation: true`, es
  diagnóstica y no invocable.

## Matriz de evidencia por superficie

`Sí` significa que la fuente citada demuestra esa propiedad en la superficie exacta.
`No verificado` significa que no se encontró un contrato estable suficiente; no
significa que la función sea técnicamente imposible.

| Agente / superficie revisada | Generación real | Comprensión / adjuntos | Mecanismo observado | Credencial y cuota | Clasificación para Infinidev | Evidencia |
| --- | --- | --- | --- | --- | --- | --- |
| **Codex / ChatGPT oficiales** (app, web, CLI, IDE) | **Sí**, integrada; `$imagegen` en CLI y herramienta interna `image_gen` | **Sí**, admite referencias y edición | Función alojada por OpenAI; el código usa el proveedor y autorización activos de Codex. Endpoint, estabilidad y autorización para terceros: **no verificados como API pública** | Inicio de sesión/plan de ChatGPT; consume límites o créditos generales de Codex, aproximadamente 3–5× más rápido. No es una API key transferible | `ADVERTISED_UNVERIFIED`: evidencia positiva del producto, sin perfil ejecutable para Infinidev | [Guía oficial de Codex](https://developers.openai.com/codex/image-generation), [precios y límites](https://developers.openai.com/codex/pricing), [skill fijada](https://github.com/openai/codex/blob/8922a784fe6aa80683fe97c2dcdfdc361478aa7f/codex-rs/skills/src/assets/samples/imagegen/SKILL.md), [wiring de autorización fijado](https://github.com/openai/codex/blob/8922a784fe6aa80683fe97c2dcdfdc361478aa7f/codex-rs/ext/image-generation/src/extension.rs) |
| **OpenAI API — Image API** | **Sí** | Edición y referencias, separadas de visión del chat | `POST /v1/images/generations` / `images.generate`, con modelo `gpt-image-*` | `OPENAI_API_KEY`; organización/proyecto, cuota y precios de API | `SUPPORTED` solo con perfil exacto y credencial presente | [Guía Image API](https://developers.openai.com/api/docs/guides/image-generation), [quickstart y API key](https://developers.openai.com/api/docs/quickstart), [cuota y facturación de organización](https://developers.openai.com/api/docs/guides/production-best-practices) |
| **OpenAI API — Responses** | **Sí**, como herramienta incorporada | **Sí** | `responses.create(..., tools=[{"type": "image_generation"}])`; el modelo principal llama a GPT Image | API key y facturación de API; suma uso del modelo principal y coste de imagen | Mecanismo distinto; no debe confundirse con Image API ni habilitar su adaptador | [Guía oficial, “Responses API”](https://developers.openai.com/api/docs/guides/image-generation#overview) |
| **OpenCode** (CLI/desktop, commit `89130db`) | **No nativa en el registro revisado** | **Sí**, normaliza adjuntos PNG/JPEG/WebP | Built-ins: shell, read, glob, grep, edit, write, task, fetch, todo, search, skill y patch. Puede añadir custom tools o MCP | Depende del proveedor o de la extensión instalada; cuota de generación: **no verificada en core** | Un plugin/MCP puede generar, pero no prueba soporte nativo ni una ruta OAuth de OpenAI | [registro fijado](https://github.com/anomalyco/opencode/blob/89130db6b0060a345548d870c51132ee71d6a828/packages/opencode/src/tool/registry.ts), [documentación de tools](https://github.com/anomalyco/opencode/blob/89130db6b0060a345548d870c51132ee71d6a828/packages/web/src/content/docs/tools.mdx), [pruebas de adjuntos](https://github.com/anomalyco/opencode/blob/89130db6b0060a345548d870c51132ee71d6a828/packages/opencode/test/image/image.test.ts) |
| **Pi coding agent** (commit `305c014`) | **No como herramienta built-in del coding agent** | **Sí**, pegado, arrastre y archivos de imagen | Core expone `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`; extensiones pueden registrar herramientas | `/login` admite suscripciones para modelos de chat; eso no demuestra autorización de Images. Cuota de generación nativa: **no verificada** | Visión y extensibilidad no equivalen a generación nativa | [README fijado](https://github.com/badlogic/pi-mono/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/packages/coding-agent/README.md), [tools fijadas](https://github.com/badlogic/pi-mono/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/packages/coding-agent/src/core/tools/index.ts) |
| **Pi AI SDK / OpenRouter Images** (misma monorepo) | **Sí en una biblioteca inferior**, no publicada automáticamente al coding agent | Acepta texto e imágenes según modelo | `generateImages` con proveedor `openrouter-images` | `OPENROUTER_API_KEY` y cuota de OpenRouter | Demuestra que se puede construir una extensión explícita; no demuestra OpenAI Subscription | [API de imágenes fijada](https://github.com/badlogic/pi-mono/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/packages/ai/src/images.ts), [proveedor fijado](https://github.com/badlogic/pi-mono/blob/305c014dcccfe97ebd3f4057ac16c436f1e2c71e/packages/ai/src/providers/openrouter-images.ts) |
| **Gemini CLI** (comparador, commit `f47d6c6`) | **No nativa según el README revisado**; menciona generación multimedia mediante un servidor MCP externo de Vertex AI Creative Studio | **Sí**, entrada multimodal desde PDFs, imágenes o bocetos | Built-ins para búsqueda, archivos, shell y web; Imagen/Veo/Lyria mediante MCP | Login de Google o API/Vertex para el chat; credencial y cuota del MCP multimedia son una ruta aparte | Ejemplo explícito de extensión de terceros, no capability nativa | [README fijado](https://github.com/google-gemini/gemini-cli/blob/f47d6c6f7a1308d81f9f57acf7d279f0928c5249/README.md), [MCP de generación enlazado](https://github.com/GoogleCloudPlatform/vertex-ai-creative-studio/tree/main/experiments/mcp-genmedia) |

### Qué no se puede inferir de la matriz

- Que un agente muestre imágenes no significa que pueda crearlas.
- Que un modelo anuncie output de imagen no significa que el host le haya publicado
  una herramienta o endpoint.
- Que exista una extensión comunitaria no significa que venga instalada, auditada o
  autenticada por el agente.
- Que Codex use internamente la autorización de la cuenta no concede a Infinidev un
  endpoint público, scopes OAuth, términos de uso ni política de cuota para repetir
  esa llamada.
- El código abierto de un cliente permite observar una implementación, pero no vuelve
  público ni estable el servicio alojado que hay detrás.

## Estados de capacidad

La resolución usa cuatro estados y falla de forma cerrada:

| Estado | Significado | ¿Publica `generate_image`? |
| --- | --- | --- |
| `SUPPORTED` | Hay un perfil exacto, revisado y ejecutable para la identidad, endpoint, transporte, modelo, operación y revisión actuales | **Sí** |
| `ADVERTISED_UNVERIFIED` | Una fuente o catálogo anuncia la función, pero falta un contrato invocable verificable | **No** |
| `UNSUPPORTED` | La combinación exacta fue evaluada y no está admitida | **No** |
| `UNKNOWN` | Evidencia ausente, inválida, obsoleta o insuficiente; también es el estado seguro ante errores del resolvedor | **No** |

`ADVERTISED_UNVERIFIED` es únicamente diagnóstico. No contiene un perfil de
generación, no se promociona por nombre de modelo y no autoriza una prueba en vivo.
Para el catálogo Codex, `true` produce este estado; `false`, ausente o un valor
inválido tampoco pueden producir `SUPPORTED`. En todos esos casos se realizan **cero**
llamadas a `litellm.image_generation`.

La generación y la entrada de imágenes son capacidades independientes:
`CapabilitySnapshot.image_input` controla si el modelo puede recibir adjuntos;
`CapabilitySnapshot.image_generation` controla si existe una ruta de generación.
Una nunca se deriva de la otra.

## Perfil API explícito de Infinidev

La única ruta habilitable es una configuración independiente de la ruta de chat. Debe
resolver, como una unidad inmutable, todos estos campos:

- **identidad** de cuenta/proyecto de API;
- **endpoint** y proveedor exactos;
- **transporte/adaptador** revisado;
- **mecanismo** (`openai_images_api`, no Codex Subscription ni visión);
- **modelo** permitido `gpt-image-*`;
- **operación** de generación compatible;
- **clase de credencial** `api_key` y API key no vacía;
- **revisión** del perfil/contrato.

Un adaptador desconocido, una configuración parcial, un modelo de chat, una
credencial OAuth de suscripción o un snapshot cuya revisión cambió se rechazan antes
de contactar al proveedor. La API key no se deriva del login del chat ni se registra
en el snapshot, conversación o ledger; solo se conserva una identidad no secreta de
la ruta.

Publicación y ejecución deben usar **el mismo snapshot**. La herramienta no vuelve a
resolver configuración al ejecutarse, y el adaptador no relee settings globales para
cambiar credencial, endpoint o modelo después de que el esquema fue publicado.

## Invalidación y aislamiento

La clave de resolución e invalidación incluye identidad, endpoint, transporte,
adaptador, mecanismo, modelo, operación y revisión. Se invalida el perfil cuando
cambia cualquiera de ellos, cuando rota o desaparece la credencial, cuando cambia el
snapshot, o cuando una revisión de evidencia retira soporte.

Los resultados de una cuenta o ruta no se convierten en soporte global:

- un `401`/`403` afecta a esa identidad y credencial;
- un `404` afecta a ese endpoint/modelo/operación;
- un `429` o `5xx` afecta a esa ruta y conserva su información de reintento;
- un timeout o desconexión deja el resultado incierto de esa operación;
- ninguna denegación o fallo de una cuenta deshabilita otra cuenta, endpoint o
  proyecto que tenga una clave de resolución diferente.

El `operation_id` queda ligado a la solicitud y a la identidad completa de la ruta.
Un resultado incierto no se reintenta automáticamente con el mismo identificador.
Las respuestas vacías o artefactos malformados se diferencian de fallos de
autorización, cuota y transporte; si una respuesta parcial contiene artefactos
válidos, estos se conservan y los elementos restantes mantienen su error individual.

## Requisitos para una futura integración oficial de suscripción

Infinidev podría convertir la evidencia de Codex en `SUPPORTED` solo si OpenAI publica
y mantiene para integradores, como mínimo:

1. una superficie y endpoint documentados para clientes de terceros;
2. scopes y flujo de autorización explícitos para generación de imágenes;
3. autorización contractual para reutilizar esa credencial fuera de Codex;
4. modelos, operaciones, formatos, límites y semántica de errores versionados;
5. reglas de cuota/facturación y aislamiento por cuenta;
6. idempotencia o reconciliación de resultados inciertos;
7. una señal de disponibilidad por cuenta/ruta que no dependa de inferir nombres o
   leer metadata informal del catálogo.

Hasta entonces, copiar endpoints observados, reutilizar tokens de ChatGPT o emular
headers privados de Codex queda fuera del contrato. La alternativa soportada es que
el usuario configure una API key de OpenAI y acepte la facturación independiente de
la API.

## Referencias internas

- Resolución: `src/infinidev/config/model_capabilities.py`.
- Publicación: `src/infinidev/tools/__init__.py`.
- Herramienta: `src/infinidev/tools/image_generation.py`.
- Adaptador y clasificación de resultados:
  `src/infinidev/engine/image_generation.py`.
- Idempotencia y persistencia durable: `src/infinidev/engine/image_ledger.py`.
- Assets privados: `src/infinidev/engine/assets.py`.

Las pruebas relevantes viven en `tests/test_openai_subscription.py`,
`tests/test_image_generation.py` y `tests/test_image_attachments.py`.
