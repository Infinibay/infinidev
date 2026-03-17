"""Standalone mini-flow for fetching and generating library documentation.

Makes its own LLM calls via litellm — does not depend on the loop engine.
"""

import json
import logging
import sqlite3
from urllib.parse import urlparse

import litellm

from infinidev.config.llm import get_litellm_params
from infinidev.db.service import execute_with_retry
from infinidev.tools.base.dedup import find_semantic_duplicate
from infinidev.tools.base.embeddings import compute_embedding
from infinidev.tools.web.backends import search_ddg, fetch_with_trafilatura

logger = logging.getLogger(__name__)

_DEFAULT_SECTIONS = [
    "Overview",
    "Installation",
    "Quick Start",
    "API Reference",
    "Configuration",
    "Examples",
]

_MAX_CONTENT_CHARS = 100_000  # ~100KB cap on scraped content

# Domains that never contain programming library documentation
_JUNK_DOMAINS = {
    "brainly.com", "brainly.ph", "brainly.in", "brainly.co.id",
    "quora.com", "answers.com", "chegg.com", "coursehero.com",
    "studocu.com", "bartleby.com", "wikipedia.org",
    "youtube.com", "reddit.com", "facebook.com", "twitter.com",
    "amazon.com", "ebay.com",
}

# Domains that are strong signals of real documentation
_DOC_DOMAINS = {
    "readthedocs.io", "readthedocs.org", "docs.rs", "pkg.go.dev",
    "pypi.org", "npmjs.com", "crates.io", "rubygems.org",
    "docs.python.org", "developer.mozilla.org", "devdocs.io",
}


def _is_junk_url(url: str) -> bool:
    """Return True if URL is from a domain that doesn't host library docs."""
    try:
        host = urlparse(url).hostname or ""
        # Strip www.
        if host.startswith("www."):
            host = host[4:]
        return host in _JUNK_DOMAINS
    except Exception:
        return False


def _is_doc_domain(url: str) -> bool:
    """Return True if URL looks like a documentation site."""
    try:
        host = urlparse(url).hostname or ""
        if host.startswith("www."):
            host = host[4:]
        # Exact match or subdomain match (e.g., flask.readthedocs.io)
        for d in _DOC_DOMAINS:
            if host == d or host.endswith(f".{d}"):
                return True
        # Heuristic: URL contains "docs" or "documentation" or "api"
        path = urlparse(url).path.lower()
        return any(kw in host or kw in path for kw in ("docs", "documentation", "api-reference", "reference"))
    except Exception:
        return False


class DocFlow:
    """Web search → fetch → LLM plan → LLM generate → DB store."""

    def execute(self, library_name: str, language: str, version: str) -> dict:
        # Step 1: discover source URLs
        urls = self._discover_sources(library_name, language, version)
        if not urls:
            raise RuntimeError(f"Web search returned no results for {library_name}")

        # Step 2: fetch content from URLs
        raw_content = self._fetch_content(urls)
        if not raw_content:
            raise RuntimeError(f"Could not fetch content from any URL for {library_name}")

        # Step 3: plan sections via LLM
        sections = self._plan_sections(library_name, language, raw_content)

        # Step 4: generate each section via LLM
        generated = self._generate_sections(library_name, language, sections, raw_content)

        # Step 5: store in DB
        stored_count = self._store_sections(library_name, language, version, generated, urls)

        return {
            "library": library_name,
            "language": language,
            "version": version,
            "sections_stored": stored_count,
            "section_titles": [s["title"] for s in generated],
        }

    def _discover_sources(self, library_name: str, language: str, version: str) -> list[str]:
        """Search for documentation URLs, prioritizing official docs and API references."""
        lang_label = language if language != "unknown" else ""
        ver_label = f" {version}" if version != "latest" else ""

        # Targeted queries — include "library", "package", or "module" to disambiguate
        queries = [
            f"{library_name} {lang_label} library official documentation{ver_label}".strip(),
            f"{library_name} {lang_label} package API reference".strip(),
            f"site:readthedocs.io OR site:github.com {library_name} {lang_label}".strip(),
            f"{library_name} {lang_label} tutorial getting started examples".strip(),
        ]

        urls: list[str] = []
        seen: set[str] = set()
        for q in queries:
            try:
                results = search_ddg(q, num_results=8)
            except Exception:
                continue
            for r in results:
                url = r.get("url", "")
                if url and url not in seen and not _is_junk_url(url):
                    seen.add(url)
                    urls.append(url)

        # Sort: doc domains first, then rest
        doc_urls = [u for u in urls if _is_doc_domain(u)]
        other_urls = [u for u in urls if not _is_doc_domain(u)]
        return (doc_urls + other_urls)[:10]

    def _fetch_content(self, urls: list[str]) -> str:
        parts: list[str] = []
        total_len = 0
        for url in urls:
            if total_len >= _MAX_CONTENT_CHARS:
                break
            try:
                content = fetch_with_trafilatura(url)
            except Exception:
                continue
            if content and len(content) > 100:  # Skip near-empty pages
                parts.append(f"--- Source: {url} ---\n{content}")
                total_len += len(content)
        return "\n\n".join(parts)[:_MAX_CONTENT_CHARS]

    def _plan_sections(self, library_name: str, language: str, raw_content: str) -> list[dict]:
        preview = raw_content[:20000]
        params = get_litellm_params()
        prompt = f"""\
You are a technical documentation architect. You are analyzing scraped web content \
about the **{language} programming library "{library_name}"**.

Your task: produce a JSON array of documentation sections that would be most useful \
for a developer using this library. Each object must have "title" (string) and "order" (int starting at 0).

Required sections (include ALL of these, plus any others that make sense):
- "Overview" — what the library does, key features, when to use it
- "Installation" — how to install (pip, npm, cargo, etc.) and basic setup
- "Core Concepts" — main abstractions, architecture, how things fit together
- "API Reference" — classes, functions, methods, their signatures and parameters
- "Configuration" — settings, options, environment variables
- "Examples" — practical code examples showing common use cases
- "Error Handling" — common errors, debugging tips, troubleshooting

Add more sections if the source material covers topics like:
- CLI usage, plugins, middleware, events, testing, deployment, migration guides

IMPORTANT: These sections are about the SOFTWARE LIBRARY "{library_name}", not about \
the general meaning of the word. If the source material is not about the software library, \
use the default section list above.

Return ONLY the JSON array. No other text.

Source material:
{preview}"""

        try:
            response = litellm.completion(
                **params,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            text = response.choices[0].message.content.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                sections = json.loads(text[start:end])
                if isinstance(sections, list) and len(sections) >= 2:
                    return sections
        except Exception as e:
            logger.warning("LLM section planning failed: %s", e)

        return [{"title": t, "order": i} for i, t in enumerate(_DEFAULT_SECTIONS)]

    def _generate_sections(
        self, library_name: str, language: str, sections: list[dict], raw_content: str
    ) -> list[dict]:
        params = get_litellm_params()
        generated: list[dict] = []

        for sec in sections:
            title = sec.get("title", "Untitled")
            order = sec.get("order", 0)
            # Give each section a generous chunk of source material
            chunk = raw_content[:30000]

            prompt = f"""\
You are writing developer documentation for the **{language} library "{library_name}"**.

Write the **"{title}"** section. This must be practical, accurate documentation that \
a developer can reference while coding.

RULES:
- Write ONLY about the software library "{library_name}", not about the general meaning of the word.
- ONLY use information present in the source material below. Do NOT invent or guess.
- If the source material does not contain relevant information for this section, \
  respond with exactly: SKIP_SECTION
- Use markdown formatting with proper headers (##, ###).
- ALWAYS include code examples in {language} using fenced code blocks (```{language}).
- For "API Reference" sections: list classes, methods, functions with their signatures, \
  parameter types, return types, and a one-line description. Format as a reference, not prose.
- For "Examples" sections: show practical, runnable code examples from simple to advanced.
- For "Installation" sections: show exact install commands and any prerequisites.
- For "Overview" sections: explain what problem the library solves, key features as bullet points, \
  and a minimal "hello world" example.
- Be specific and concrete. Show actual function/class names, not generic descriptions.

Source material (extract relevant information for "{title}"):
{chunk}"""

            try:
                response = litellm.completion(
                    **params,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("LLM generation failed for section '%s': %s", title, e)
                continue

            if "SKIP_SECTION" in content:
                logger.info("Skipping section '%s' — no relevant source material", title)
                continue

            generated.append({"title": title, "order": order, "content": content})

        return generated

    def _deduplicate_sections(self, sections: list[dict], library_name: str, language: str, version: str) -> list[dict]:
        """Remove sections whose title or content is semantically duplicate of an existing section."""
        # Load existing sections for this library
        def _load_existing(conn: sqlite3.Connection):
            return conn.execute(
                """\
                SELECT id, section_title, content
                FROM library_docs
                WHERE library_name = ? AND language = ? AND version = ?
                """,
                (library_name, language, version),
            ).fetchall()

        existing_rows = execute_with_retry(_load_existing) or []
        existing_titles = [{"id": r["id"], "title": r["section_title"]} for r in existing_rows]
        existing_contents = [{"id": r["id"], "title": r["content"][:500]} for r in existing_rows]

        # Also deduplicate within the new batch itself
        accepted: list[dict] = []
        for sec in sections:
            # Check title against existing DB sections
            title_dup = find_semantic_duplicate(sec["title"], existing_titles, threshold=0.85)
            if title_dup:
                logger.info(
                    "Skipping section '%s' — similar title to existing '%s' (%.2f)",
                    sec["title"], title_dup["title"], title_dup["similarity"],
                )
                continue

            # Check content against existing DB sections
            content_preview = sec["content"][:500]
            content_dup = find_semantic_duplicate(content_preview, existing_contents, threshold=0.85)
            if content_dup:
                logger.info(
                    "Skipping section '%s' — similar content to existing section (%.2f)",
                    sec["title"], content_dup["similarity"],
                )
                continue

            # Check title against already-accepted new sections
            accepted_titles = [{"id": i, "title": a["title"]} for i, a in enumerate(accepted)]
            batch_title_dup = find_semantic_duplicate(sec["title"], accepted_titles, threshold=0.85)
            if batch_title_dup:
                logger.info(
                    "Skipping section '%s' — similar title to new section '%s' (%.2f)",
                    sec["title"], batch_title_dup["title"], batch_title_dup["similarity"],
                )
                continue

            # Check content against already-accepted new sections
            accepted_contents = [{"id": i, "title": a["content"][:500]} for i, a in enumerate(accepted)]
            batch_content_dup = find_semantic_duplicate(content_preview, accepted_contents, threshold=0.85)
            if batch_content_dup:
                logger.info(
                    "Skipping section '%s' — similar content to new section '%s' (%.2f)",
                    sec["title"], accepted[batch_content_dup["id"]]["title"], batch_content_dup["similarity"],
                )
                continue

            accepted.append(sec)

        return accepted

    def _store_sections(
        self,
        library_name: str,
        language: str,
        version: str,
        sections: list[dict],
        source_urls: list[str],
    ) -> int:
        # Deduplicate before storing
        sections = self._deduplicate_sections(sections, library_name, language, version)

        urls_json = json.dumps(source_urls)
        stored = 0

        def _upsert(conn: sqlite3.Connection):
            nonlocal stored
            for sec in sections:
                emb = compute_embedding(f"{sec['title']} {sec['content'][:500]}")
                conn.execute(
                    """\
                    INSERT INTO library_docs
                        (library_name, language, version, section_title, section_order,
                         content, embedding, source_urls)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(library_name, language, version, section_title) DO UPDATE SET
                        section_order = excluded.section_order,
                        content       = excluded.content,
                        embedding     = excluded.embedding,
                        source_urls   = excluded.source_urls,
                        updated_at    = CURRENT_TIMESTAMP
                    """,
                    (
                        library_name, language, version,
                        sec["title"], sec["order"],
                        sec["content"], emb, urls_json,
                    ),
                )
                stored += 1
            conn.commit()

        execute_with_retry(_upsert)
        return stored
