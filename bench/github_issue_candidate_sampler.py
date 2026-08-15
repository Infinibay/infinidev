"""Acquire diverse public GitHub issues for individual task-policy review.

GitHub labels are sampling hints, never Infinidev policy labels. Raw issue text
is an ignored external artifact; committed review ledgers contain annotations
and stable candidate identifiers, not copied issue bodies.
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
from typing import Any, Iterable


GITHUB_API_VERSION = "2022-11-28"
DEFAULT_OUTPUT = Path(".infinidev/external-data/github-issues/candidates.jsonl")
_DATE_WINDOWS = (
    "created:2018-01-01..2021-12-31",
    "created:2022-01-01..2024-12-31",
    "created:>=2025-01-01",
)
_LABEL_HINTS = ("bug", "enhancement", "performance", "refactor", "investigation")
DEFAULT_QUERIES = tuple(
    f"is:issue archived:false {window} label:{label} sort:updated-desc"
    for label in _LABEL_HINTS
    for window in _DATE_WINDOWS
)
_HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_SECRET = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


GRAPHQL_QUERY = """
query($searchQuery: String!, $cursor: String) {
  search(query: $searchQuery, type: ISSUE, first: 100, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      __typename
      ... on Issue {
        id number title body url state createdAt updatedAt closedAt
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


def normalize_issue(node: dict[str, Any], *, query_hint: str) -> dict[str, Any] | None:
    """Normalize a GraphQL Issue node, rejecting unsafe or unsuitable rows."""
    if node.get("__typename") != "Issue":
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
    issue_text = f"{title}\n\n{body}".strip()
    word_count = len(issue_text.split())
    if not title or not 12 <= word_count <= 1500 or _SECRET.search(issue_text):
        return None
    repo = str(repository.get("nameWithOwner", ""))
    number = node.get("number")
    if not repo or not isinstance(number, int):
        return None
    labels = node.get("labels")
    label_nodes = labels.get("nodes", []) if isinstance(labels, dict) else []
    label_names = sorted({
        str(label.get("name", "")).strip()
        for label in label_nodes
        if isinstance(label, dict) and str(label.get("name", "")).strip()
    })
    language = repository.get("primaryLanguage")
    return {
        "candidate_id": f"github-issue:{repo.casefold()}:{number}",
        "source": {
            "provider": "github",
            "node_id": node.get("id"),
            "repo": repo,
            "repo_url": repository.get("url"),
            "repo_license_spdx": spdx_id,
            "repo_license_name": license_info.get("name"),
            "repo_stars": int(repository.get("stargazerCount") or 0),
            "programming_language": (
                language.get("name") if isinstance(language, dict) else None
            ),
            "issue_number": number,
            "issue_url": node.get("url"),
            "issue_state": str(node.get("state", "")).casefold(),
            "issue_created_at": node.get("createdAt"),
            "issue_updated_at": node.get("updatedAt"),
            "issue_closed_at": node.get("closedAt"),
            "author_login": login,
            "upstream_label_hints": label_names,
            "selection_query_hint": query_hint,
            "issue_text_sha256": hashlib.sha256(issue_text.encode()).hexdigest(),
        },
        "issue_text": issue_text,
        "manual_review": {
            "status": "unreviewed",
            "include": None,
            "policies": None,
            "uncategorized_reason": None,
            "notes": None,
        },
    }


def query_github(query: str, *, pages: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read bounded GraphQL search pages through the authenticated GitHub CLI."""
    if pages < 1 or pages > 10:
        raise ValueError("pages must be in [1, 10]; GitHub search exposes at most 1,000 rows")
    nodes: list[dict[str, Any]] = []
    cursor: str | None = None
    issue_count = 0
    last_rate_limit: dict[str, Any] = {}
    for _ in range(pages):
        command = [
            "gh", "api", "graphql",
            "-H", f"X-GitHub-Api-Version: {GITHUB_API_VERSION}",
            "-f", f"query={GRAPHQL_QUERY}",
            "-F", f"searchQuery={query}",
        ]
        if cursor is not None:
            command.extend(["-F", f"cursor={cursor}"])
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout)
        data = payload["data"]
        search = data["search"]
        issue_count = int(search["issueCount"])
        last_rate_limit = data["rateLimit"]
        nodes.extend(node for node in search["nodes"] if isinstance(node, dict))
        page_info = search["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = str(page_info["endCursor"])
    return nodes, {
        "query": query,
        "matching_issues": issue_count,
        "fetched_nodes": len(nodes),
        "rate_limit_after": last_rate_limit,
    }


def exclusion_sets(paths: Iterable[Path]) -> tuple[frozenset[str], frozenset[str]]:
    """Return candidate identifiers and repositories present in prior queues."""
    identifiers: set[str] = set()
    repositories: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                identifiers.add(str(row["candidate_id"]))
                repositories.add(str(row["source"]["repo"]).casefold())
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid candidate") from exc
    return frozenset(identifiers), frozenset(repositories)


def select_candidates(
    nodes_by_query: Iterable[tuple[str, Iterable[dict[str, Any]]]],
    *,
    limit: int,
    max_per_repo: int,
    min_repo_stars: int,
    seed: int,
    excluded_ids: frozenset[str] = frozenset(),
    excluded_repositories: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic query-balanced and repository-diverse queue."""
    if limit < 1 or max_per_repo < 1 or min_repo_stars < 0:
        raise ValueError("invalid selection bounds")
    rejected: Counter[str] = Counter()
    candidates: dict[str, dict[str, Any]] = {}
    candidate_hints: dict[str, set[str]] = defaultdict(set)
    fetched = 0
    for query_hint, nodes in nodes_by_query:
        for node in nodes:
            fetched += 1
            candidate = normalize_issue(node, query_hint=query_hint)
            if candidate is None:
                rejected["invalid_or_filtered"] += 1
                continue
            candidate_id = str(candidate["candidate_id"])
            repo = str(candidate["source"]["repo"]).casefold()
            if candidate_id in excluded_ids:
                rejected["excluded_candidate_id"] += 1
                continue
            if repo in excluded_repositories:
                rejected["excluded_repository"] += 1
                continue
            if int(candidate["source"]["repo_stars"]) < min_repo_stars:
                rejected["repository_below_star_floor"] += 1
                continue
            if candidate_id in candidates:
                rejected["duplicate_across_queries"] += 1
            else:
                candidates[candidate_id] = candidate
            candidate_hints[candidate_id].add(query_hint)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_id, candidate in candidates.items():
        hints = sorted(candidate_hints[candidate_id])
        candidate["source"]["selection_query_hints"] = hints
        candidate["source"].pop("selection_query_hint", None)
        bucket = hashlib.sha256(f"{seed}:{candidate_id}".encode()).hexdigest()
        candidate["_selection_rank"] = bucket
        buckets[hints[0]].append(candidate)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: str(row["_selection_rank"]))

    selected: list[dict[str, Any]] = []
    repo_counts: Counter[str] = Counter()
    positions: Counter[str] = Counter()
    names = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for name in names:
            bucket = buckets[name]
            while positions[name] < len(bucket):
                candidate = bucket[positions[name]]
                positions[name] += 1
                repo = str(candidate["source"]["repo"]).casefold()
                if repo_counts[repo] >= max_per_repo:
                    continue
                candidate.pop("_selection_rank", None)
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
        "fetched_nodes": fetched,
        "normalized_unique_candidates": len(candidates),
        "selected": len(selected),
        "selected_repositories": len(repo_counts),
        "selected_query_hints": dict(sorted(Counter(
            str(candidate["source"]["selection_query_hints"][0])
            for candidate in selected
        ).items())),
        "rejected": dict(sorted(rejected.items())),
        "interpretation": (
            "GitHub labels and search queries only diversify manual review; they are not policy "
            "labels. Repository licenses describe repositories, not third-party issue text."
        ),
    }


def write_queue(
    output: Path,
    candidates: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    queries: list[str],
    query_reports: list[dict[str, Any]],
    pages_per_query: int,
    selection_limit: int,
    max_per_repo: int,
    min_repo_stars: int,
    seed: int,
) -> Path:
    """Write the ignored external queue and its complete acquisition manifest."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n"
        for candidate in candidates
    )
    output.write_text(payload, encoding="utf-8")
    manifest_path = output.with_suffix(output.suffix + ".provenance.json")
    manifest = {
        "artifact": {
            "path": output.name,
            "rows": len(candidates),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "distribution_notice": (
                "Raw issue text is an ignored external artifact. GitHub users retain rights in "
                "their content; a repository SPDX license is contextual metadata, not a claim "
                "that the issue body uses that license."
            ),
        },
        "source": {
            "provider": "GitHub GraphQL API",
            "api_version": GITHUB_API_VERSION,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "queries": queries,
            "query_reports": query_reports,
        },
        "selection": {
            "pages_per_query": pages_per_query,
            "selection_limit": selection_limit,
            "max_per_repo": max_per_repo,
            "min_repo_stars": min_repo_stars,
            "seed": seed,
            "report": report,
        },
        "filters": {
            "public_only": True,
            "issues_only_no_pull_requests": True,
            "non_archived_non_fork_repositories": True,
            "identified_spdx_repository_license": True,
            "bots_rejected": True,
            "obvious_secrets_rejected": True,
        },
        "review_contract": {
            "github_labels_are_policy_labels": False,
            "individual_manual_review_required": True,
            "source_text_remains_external": True,
            "reviewed_annotations_path": "data/task-policy-reviews/github-issues",
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
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--pages-per-query", type=int, default=5)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--max-per-repo", type=int, default=2)
    parser.add_argument("--min-repo-stars", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--exclude-candidates", type=Path, action="append", default=[])
    args = parser.parse_args()
    queries = args.query or list(DEFAULT_QUERIES)
    excluded_ids, excluded_repositories = exclusion_sets(args.exclude_candidates)
    query_results = []
    query_reports = []
    for query in queries:
        nodes, query_report = query_github(query, pages=args.pages_per_query)
        query_results.append((query, nodes))
        query_reports.append(query_report)
    candidates, report = select_candidates(
        query_results,
        limit=args.limit,
        max_per_repo=args.max_per_repo,
        min_repo_stars=args.min_repo_stars,
        seed=args.seed,
        excluded_ids=excluded_ids,
        excluded_repositories=excluded_repositories,
    )
    manifest = write_queue(
        args.output,
        candidates,
        report,
        queries=queries,
        query_reports=query_reports,
        pages_per_query=args.pages_per_query,
        selection_limit=args.limit,
        max_per_repo=args.max_per_repo,
        min_repo_stars=args.min_repo_stars,
        seed=args.seed,
    )
    report.update({"output": str(args.output), "provenance_manifest": str(manifest)})
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_QUERIES", "exclusion_sets", "normalize_issue", "select_candidates"]
