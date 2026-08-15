"""Acquire diverse public pull requests as real code-review task candidates.

The upstream title and body remain unchanged inside a short review instruction.
That instruction supplies the user intent which a pull-request description alone
normally leaves implicit.  Repository metadata is used only for sampling and
provenance, never as a task-policy label.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable


GITHUB_API_VERSION = "2022-11-28"
DEFAULT_OUTPUT = Path(".infinidev/external-data/github-pr-reviews/candidates.jsonl")
DEFAULT_LANGUAGES = (
    "Python", "TypeScript", "JavaScript", "Java", "Go", "Rust",
    "C++", "C#", "Ruby", "PHP", "Kotlin", "Swift", "Shell",
)
_DATE_WINDOWS = (
    "created:2018-01-01..2021-12-31",
    "created:2022-01-01..2024-12-31",
    "created:>=2025-01-01",
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_SECRET = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)
_REVIEW_INSTRUCTIONS = (
    "Review this pull request. Report concrete findings and do not modify the code.",
    "Inspect this proposed change for correctness and regressions; return a read-only review.",
    "Perform a code review of this pull request and prioritize actionable defects.",
    "Audit the following pull request. Explain any blocking problems without editing files.",
    "Revisa este pull request y reporta hallazgos concretos sin modificar el código.",
    "Haz una revisión de código de este cambio; prioriza defectos y regresiones verificables.",
    "Revise ce pull request et signale les défauts concrets sans modifier le code.",
    "Revise este pull request e relate problemas concretos sem alterar o código.",
)


GRAPHQL_QUERY = """
query($searchQuery: String!, $cursor: String) {
  search(query: $searchQuery, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      __typename
      ... on PullRequest {
        id number title body url state isDraft createdAt updatedAt closedAt mergedAt
        author { __typename login }
        labels(first: 30) { nodes { name description } }
        repository {
          nameWithOwner url isPrivate isArchived isFork stargazerCount
          primaryLanguage { name }
          licenseInfo { spdxId name }
        }
      }
    }
  }
  rateLimit { cost remaining resetAt }
}
"""


def _length_bucket(words: int) -> str:
    if words < 80:
        return "short"
    if words < 300:
        return "medium"
    return "long"


def _review_instruction(candidate_id: str) -> str:
    digest = hashlib.sha256(candidate_id.encode()).digest()
    return _REVIEW_INSTRUCTIONS[digest[0] % len(_REVIEW_INSTRUCTIONS)]


def normalize_pull_request(
    node: dict[str, Any], *, query_hint: str,
) -> dict[str, Any] | None:
    """Normalize one public non-bot pull request into a review request."""
    if node.get("__typename") != "PullRequest" or node.get("isDraft"):
        return None
    repository = node.get("repository")
    author = node.get("author")
    if not isinstance(repository, dict) or not isinstance(author, dict):
        return None
    if repository.get("isPrivate") or repository.get("isArchived") or repository.get("isFork"):
        return None
    license_info = repository.get("licenseInfo")
    spdx_id = str(license_info.get("spdxId", "")) if isinstance(license_info, dict) else ""
    if not spdx_id or spdx_id in {"NOASSERTION", "OTHER"}:
        return None
    login = str(author.get("login", ""))
    if author.get("__typename") == "Bot" or login.casefold().endswith("[bot]"):
        return None

    title = str(node.get("title") or "").strip()
    body = _HTML_COMMENT.sub("", str(node.get("body") or "")).strip()
    source_text = f"{title}\n\n{body}".strip()
    if not title or _SECRET.search(source_text):
        return None
    source_words = len(source_text.split())
    if not 12 <= source_words <= 1500:
        return None
    repo = str(repository.get("nameWithOwner", ""))
    number = node.get("number")
    if not repo or not isinstance(number, int):
        return None
    candidate_id = f"github-pr-review:{repo.casefold()}:{number}"
    issue_text = (
        f"{_review_instruction(candidate_id)}\n\n"
        f"Pull request title: {title}\n\n{body}"
    ).strip()
    language = repository.get("primaryLanguage")
    language_name = language.get("name") if isinstance(language, dict) else None
    labels = node.get("labels")
    label_nodes = labels.get("nodes", []) if isinstance(labels, dict) else []
    return {
        "candidate_id": candidate_id,
        "source": {
            "provider": "github",
            "source_kind": "pull_request",
            "node_id": node.get("id"),
            "repo": repo,
            "repo_url": repository.get("url"),
            "repo_license_spdx": spdx_id,
            "repo_license_name": license_info.get("name"),
            "repo_stars": int(repository.get("stargazerCount") or 0),
            "programming_language": language_name,
            "pull_number": number,
            "pull_url": node.get("url"),
            "pull_state": str(node.get("state", "")).casefold(),
            "pull_created_at": node.get("createdAt"),
            "pull_updated_at": node.get("updatedAt"),
            "pull_closed_at": node.get("closedAt"),
            "pull_merged_at": node.get("mergedAt"),
            "author_login": login,
            "upstream_label_hints": sorted({
                str(item.get("name", "")).strip()
                for item in label_nodes
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            }),
            "selection_query_hint": query_hint,
            "source_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
            "task_transform": "explicit-read-only-review-wrapper-v1",
            "length_bucket": _length_bucket(len(issue_text.split())),
        },
        "issue_text": issue_text,
        "manual_review": {
            "status": "unreviewed", "include": None, "policies": None,
            "uncategorized_reason": None, "notes": None,
        },
    }


def query_github(
    query: str,
    *,
    pages: int,
    max_attempts: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read bounded GraphQL search pages through the authenticated GitHub CLI."""
    if pages < 1 or pages > 10:
        raise ValueError("pages must be in [1, 10]")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    nodes: list[dict[str, Any]] = []
    cursor: str | None = None
    issue_count = 0
    rate_limit: dict[str, Any] = {}
    for _ in range(pages):
        command = [
            "gh", "api", "graphql",
            "-H", f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
            "-f", f"query={GRAPHQL_QUERY}",
            "-F", f"searchQuery={query}",
        ]
        if cursor is not None:
            command.extend(["-F", f"cursor={cursor}"])
        completed: subprocess.CompletedProcess[str] | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                completed = subprocess.run(
                    command, check=True, capture_output=True, text=True,
                )
            except subprocess.CalledProcessError as exc:
                if attempt == max_attempts:
                    detail = (exc.stderr or exc.stdout or str(exc)).strip()
                    raise RuntimeError(
                        f"GitHub query failed after {max_attempts} attempts: {detail}"
                    ) from exc
                time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)))
            else:
                break
        if completed is None:
            raise RuntimeError("GitHub query completed without a result")
        payload = json.loads(completed.stdout)["data"]
        search = payload["search"]
        issue_count = int(search["issueCount"])
        rate_limit = payload["rateLimit"]
        nodes.extend(item for item in search["nodes"] if isinstance(item, dict))
        if not search["pageInfo"]["hasNextPage"]:
            break
        cursor = str(search["pageInfo"]["endCursor"])
    return nodes, {
        "query": query, "matching_pull_requests": issue_count,
        "fetched_nodes": len(nodes), "rate_limit_after": rate_limit,
    }


def select_candidates(
    nodes_by_query: Iterable[tuple[str, Iterable[dict[str, Any]]]],
    *,
    limit: int,
    max_per_repo: int,
    min_repo_stars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic mix of languages, lengths, queries, and repositories."""
    if limit < 1 or max_per_repo < 1 or min_repo_stars < 0:
        raise ValueError("invalid selection bounds")
    rejected: Counter[str] = Counter()
    unique: dict[str, dict[str, Any]] = {}
    for query_hint, nodes in nodes_by_query:
        for node in nodes:
            candidate = normalize_pull_request(node, query_hint=query_hint)
            if candidate is None:
                rejected["invalid_or_filtered"] += 1
                continue
            if int(candidate["source"]["repo_stars"]) < min_repo_stars:
                rejected["repository_below_star_floor"] += 1
                continue
            unique.setdefault(str(candidate["candidate_id"]), candidate)

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in unique.values():
        source = candidate["source"]
        key = (str(source.get("programming_language") or "unknown"), source["length_bucket"])
        buckets[key].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: hashlib.sha256(
            f"{seed}:{row['candidate_id']}".encode()
        ).digest())

    selected: list[dict[str, Any]] = []
    positions: Counter[tuple[str, str]] = Counter()
    repo_counts: Counter[str] = Counter()
    keys = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            bucket = buckets[key]
            while positions[key] < len(bucket):
                candidate = bucket[positions[key]]
                positions[key] += 1
                repo = str(candidate["source"]["repo"]).casefold()
                if repo_counts[repo] >= max_per_repo:
                    continue
                selected.append(candidate)
                repo_counts[repo] += 1
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    selected.sort(key=lambda row: str(row["candidate_id"]))
    return selected, {
        "normalized_unique_candidates": len(unique),
        "selected": len(selected),
        "selected_repositories": len(repo_counts),
        "languages": dict(sorted(Counter(
            str(row["source"].get("programming_language") or "unknown") for row in selected
        ).items())),
        "length_buckets": dict(sorted(Counter(
            str(row["source"]["length_bucket"]) for row in selected
        ).items())),
        "rejected": dict(sorted(rejected.items())),
    }


def write_queue(
    output: Path,
    candidates: list[dict[str, Any]],
    *,
    queries: list[str],
    query_reports: list[dict[str, Any]],
    report: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    """Write the ignored candidate queue and a complete provenance manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates
    )
    output.write_text(payload, encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".provenance.json")
    manifest = {
        "artifact": {
            "path": output.name, "rows": len(candidates),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "distribution_notice": (
                "Raw pull-request text is an ignored external artifact. Repository licenses "
                "are provenance metadata, not a claim over contributor-authored text."
            ),
        },
        "source": {
            "provider": "GitHub GraphQL API", "api_version": GITHUB_API_VERSION,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "queries": queries, "query_reports": query_reports,
        },
        "selection": {
            "pages_per_query": args.pages_per_query, "selection_limit": args.limit,
            "max_per_repo": args.max_per_repo, "min_repo_stars": args.min_repo_stars,
            "seed": args.seed, "report": report,
        },
        "review_contract": {
            "source_is_real_pull_request": True,
            "review_intent_is_explicit_wrapper": True,
            "individual_model_or_human_review_required": True,
            "source_text_remains_external": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--language", action="append", default=[])
    parser.add_argument("--pages-per-query", type=int, default=3)
    parser.add_argument("--limit", type=int, default=900)
    parser.add_argument("--max-per-repo", type=int, default=2)
    parser.add_argument("--min-repo-stars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    languages = args.language or list(DEFAULT_LANGUAGES)
    queries = [
        f"is:pr archived:false {window} language:{language} sort:updated-desc"
        for language in languages
        for window in _DATE_WINDOWS
    ]
    queried = [(query, query_github(query, pages=args.pages_per_query)) for query in queries]
    candidates, report = select_candidates(
        ((query, result[0]) for query, result in queried),
        limit=args.limit, max_per_repo=args.max_per_repo,
        min_repo_stars=args.min_repo_stars, seed=args.seed,
    )
    manifest = write_queue(
        args.output, candidates, queries=queries,
        query_reports=[result[1] for _, result in queried], report=report, args=args,
    )
    print(json.dumps({
        **report, "output": str(args.output), "provenance_manifest": str(manifest),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["normalize_pull_request", "select_candidates"]
