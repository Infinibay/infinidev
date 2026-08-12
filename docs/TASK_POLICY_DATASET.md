# Task Policy Dataset

Estado: contrato y pool sintético draft; no aprobado para entrenar release

## Objetivo

El clasificador debe distinguir métodos de trabajo y `uncategorized`. Más
ejemplos solo ayudan cuando agregan diversidad real y no filtran la misma frase
entre calibration, validation y holdout.

`bench/task_policy_project_corpus.py` crea un primer pool de 672 requests:

- 12 proyectos ficticios y 15 lenguajes de programación;
- inglés, español, portugués y francés balanceados;
- 12 operaciones con 48 ejemplos cada una;
- 96 negativos explícitos;
- procedencia OSS y licencia registradas por arquetipo;
- nombres, componentes y requests originales, sin copiar issues o código.

Las referencias se usan solo como inspiración de dominio: [Ruff](https://github.com/astral-sh/ruff),
[FastAPI](https://github.com/fastapi/fastapi), [Bun](https://github.com/oven-sh/bun),
[Helix](https://github.com/helix-editor/helix), [Neovim](https://github.com/neovim/neovim),
[Loki](https://github.com/grafana/loki), [Rails](https://github.com/rails/rails),
[Elasticsearch Java Client](https://github.com/elastic/elasticsearch-java),
[ASP.NET Core](https://github.com/dotnet/aspnetcore),
[Laravel](https://github.com/laravel/framework),
[Phoenix](https://github.com/phoenixframework/phoenix) y
[Ktor](https://github.com/ktorio/ktor). Esto no convierte ejemplos sintéticos
en datos naturales ni en evidencia de generalización.

## Taxonomía

La expansión cubre `bugfix`, `feature`, `refactor`, `research`, `review`,
`planning`, `performance`, `security`, `migration`, `docs`, `test`,
`configuration` y `uncategorized`.

Una categoría solo se promueve a política si requiere un método o verificación
diferente. “Database”, “editor” o “web” son contexto, no labels de operación.

## Uncategorized

Runtime representa `uncategorized` como ausencia de labels sobre threshold, no
como un softmax que siempre elige. El dataset conserva una razón explícita:

- conversación, status o aprobación;
- explicación de una acción citada en un log;
- pregunta informativa sin método especializado;
- contenido fuera de dominio;
- request técnico aún no soportado;
- input ambiguo o insuficiente.

Los negativos deben contener términos de acción. “El log dice ‘implementa la
migración’; ¿qué significa?” prueba más que una frase obviamente no técnica.

## Contrato de cada fila

Cada ejemplo registra id/texto, proyecto, lenguajes de implementación, idioma
natural, labels o razón `uncategorized`, familia de proyecto, familia de frase,
split, fuente, URL, `review_status`, reviewer y rationale.

Solo filas `approved` con reviewer y rationale pueden entrar a un artifact de
release. Los drafts sirven para revisión y hard-negative mining.

## Leakage

Separar únicamente repositorios no alcanza. Deben aislarse familias de
proyecto, paráfrasis, issue/conversación, root cause compartido, traducciones y
transformaciones con negación o texto citado.

El pool actual aísla proyectos y no tiene duplicados, pero reutiliza familias
de plantillas entre splits. El auditor lo reporta como `phrase_split_leakage` y
mantiene `release_ready=false`. El siguiente paso es reescribir las familias
por split y revisarlas; no ocultar leakage cambiando ids.

## Datos naturales y aprendizaje activo

El corpus final combina requests naturales anonimizados, ejemplos OSS
inspirados revisados, negativos difíciles, abstenciones/disagreements de shadow
mode y muestras aleatorias de tráfico sano. Paths personales, secrets, nombres
y contenido propietario se redactan antes de persistir.

## Gates

- cero duplicados y leakage de familias;
- balance por label, idioma, dominio y split;
- `uncategorized` diverso;
- aprobación humana y rationale;
- hashes de dataset/artifact y `space_id` de `ken/static-qwen3-r512-v2`;
- thresholds elegidos sin observar holdout;
- precisión selectiva alta y falsos positivos cercanos a cero;
- mejora E2E, no solo clasificación correcta.

## Comandos

```bash
uv run python -m bench.task_policy_project_corpus
uv run python -m bench.task_policy_project_corpus \
  --output /tmp/task-policy-project-drafts.jsonl
```
