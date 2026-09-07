"""Provider quota and billing reports, separate from local token totals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx

from infinidev.config.providers import PROVIDERS
from infinidev.config.settings import settings
from infinidev.engine.usage import usage_ledger


@dataclass(frozen=True)
class UsageSelection:
    """Capture the selected connection before a background query starts."""

    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str
    admin_key: str = field(default="", repr=False)

    @classmethod
    def current(cls) -> UsageSelection:
        provider = settings.LLM_PROVIDER
        admin = getattr(settings, f"USAGE_{provider.upper()}_ADMIN_KEY", "")
        return cls(provider, settings.LLM_MODEL, settings.LLM_API_KEY,
                   settings.LLM_BASE_URL, admin)

    def request_params(self) -> dict:
        provider = PROVIDERS.get(self.provider)
        model = self.model.rsplit("/", 1)[-1]
        prefix = provider.prefix if provider else ""
        base = self.base_url
        if provider and (provider.is_native or self.provider.endswith("_subscription")):
            base = provider.default_base_url
        params = {"model": prefix + model, "api_base": base, "api_key": self.api_key}
        if self.provider == "openai_subscription":
            from infinidev.config.openai_oauth import load_credentials

            credentials = load_credentials()
            params["extra_headers"] = {"ChatGPT-Account-ID": credentials.account_id}
        return params


def _reset_time(value) -> str:
    try:
        return datetime.fromtimestamp(float(value), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError, OverflowError, OSError):
        return "unknown"


def format_codex_limits(data: dict) -> str:
    """Render all returned buckets and their actual quota window durations."""
    buckets = data.get("rateLimitsByLimitId") or {}
    if not buckets and data.get("rateLimits"):
        bucket = data["rateLimits"]
        buckets = {bucket.get("limitId") or "Codex": bucket}
    lines = []
    for key, bucket in buckets.items():
        name = bucket.get("limitName") or key
        for window_name in ("primary", "secondary"):
            window = bucket.get(window_name)
            if not window or window.get("usedPercent") is None:
                continue
            used = float(window["usedPercent"])
            duration = window.get("windowDurationMins")
            label = window_name
            if isinstance(duration, (int, float)) and duration > 0:
                if duration % 1440 == 0:
                    label = f"{duration / 1440:g}d"
                elif duration % 60 == 0:
                    label = f"{duration / 60:g}h"
                else:
                    label = f"{duration:g}m"
            remaining = max(0, min(100, 100 - used))
            lines.append(f"{name} · {label}: {remaining:g}% remaining ({used:g}% used)"
                         f" · resets {_reset_time(window.get('resetsAt'))}")
        credits = bucket.get("credits")
        if isinstance(credits, dict):
            if credits.get("unlimited"):
                lines.append(f"{name} · credits: unlimited")
            elif credits.get("balance") is not None:
                lines.append(f"{name} · credit balance: {credits['balance']}")
    return "\n".join(lines) or "Subscription quota unavailable in the Codex response."


def _admin_rows(url: str, headers: dict, params: dict) -> list[dict]:
    result = []
    query = dict(params)
    seen_pages = set()
    for _ in range(3):
        response = httpx.get(url, headers=headers, params=query, timeout=8,
                             follow_redirects=False)
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}; check the reporting key's access.")
        data = response.json()
        for bucket in data["data"]:
            result.extend(bucket["results"])
        if not data.get("has_more"):
            return result
        page = data.get("next_page")
        if not page or page in seen_pages:
            break
        seen_pages.add(page)
        query["page"] = page
    raise RuntimeError("The provider returned an incomplete reporting page.")


def _admin_report(selection: UsageSelection) -> list[str]:
    provider = selection.provider
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    headers = {"User-Agent": "Infinidev/usage"}
    if provider == "openai":
        base = "https://api.openai.com/v1/organization/"
        headers["Authorization"] = f"Bearer {selection.admin_key}"
        params = {"start_time": int(today.timestamp()), "end_time": int(tomorrow.timestamp()),
                  "bucket_width": "1d", "limit": 1}
        usage_path, cost_path = "usage/completions", "costs"
    else:
        base = "https://api.anthropic.com/v1/organizations/"
        headers.update({"x-api-key": selection.admin_key, "anthropic-version": "2023-06-01"})
        params = {"starting_at": today.isoformat(), "ending_at": tomorrow.isoformat(),
                  "bucket_width": "1d", "limit": 1}
        usage_path, cost_path = "usage_report/messages", "cost_report"
    lines = [f"Organization · {today:%Y-%m-%d} UTC · all models and projects (may lag)"]
    for kind, path in (("Tokens", usage_path), ("Costs", cost_path)):
        try:
            rows = _admin_rows(base + path, headers, params)
            if kind == "Tokens":
                output = sum(int(row.get("output_tokens", 0)) for row in rows)
                if provider == "openai":
                    total_input = sum(int(row.get("input_tokens", 0)) for row in rows)
                else:
                    total_input = sum(
                        int(row.get("uncached_input_tokens", 0))
                        + int(row.get("cache_read_input_tokens", 0))
                        + sum(int(v) for v in (row.get("cache_creation") or {}).values())
                        for row in rows
                    )
                lines.append(f"{total_input:,} input · {output:,} output tokens")
            else:
                total = Decimal(0)
                for row in rows:
                    amount = row.get("amount")
                    currency = (amount.get("currency") if isinstance(amount, dict)
                                else row.get("currency"))
                    if str(currency).lower() != "usd":
                        raise ValueError("Unrecognized billing currency")
                    total += (Decimal(str(amount["value"])) if isinstance(amount, dict)
                              else Decimal(str(amount)) / 100)
                lines.append(f"Reported cost: ${total:.4f} USD")
                if provider == "anthropic":
                    lines.append("Priority Tier costs are excluded by Anthropic's cost endpoint.")
        except RuntimeError as exc:
            lines.append(f"{kind} unavailable: {exc}")
        except (httpx.HTTPError, ValueError, TypeError, KeyError, ArithmeticError):
            lines.append(f"{kind} unavailable: network error or invalid reporting response.")
    return lines


def _rate_limit_lines(snapshot: dict, provider: str) -> list[str]:
    headers = snapshot["headers"]
    lines = []
    for resource in ("requests", "tokens", "input-tokens", "output-tokens"):
        if provider == "anthropic":
            prefix = f"anthropic-ratelimit-{resource}-"
            remaining, limit, reset = (headers.get(prefix + part)
                                       for part in ("remaining", "limit", "reset"))
        else:
            remaining, limit, reset = (headers.get(f"x-ratelimit-{part}-{resource}")
                                       for part in ("remaining", "limit", "reset"))
        if remaining is not None:
            lines.append(f"{resource}: {remaining} remaining / {limit or '?'}"
                         + (f" · reset {reset}" if reset else ""))
    if lines:
        lines.insert(0, "API rate-limit snapshot · observed " + _reset_time(snapshot["observed_at"]))
    return lines


def render_usage(selection: UsageSelection | None = None) -> str:
    """Query only the selected provider; missing capabilities remain explicit."""
    selection = selection or UsageSelection.current()
    provider = PROVIDERS.get(selection.provider)
    title = provider.display_name if provider else selection.provider
    lines = [f"Usage · {title} · {selection.model}", ""]
    try:
        snapshot = usage_ledger.snapshot(selection.request_params())
        lines.append("Observed in this Infinidev process · selected model and connection")
        if snapshot["requests"]:
            lines.append(f"{snapshot['requests']:,} requests · {snapshot['input_tokens']:,} input"
                         f" · {snapshot['output_tokens']:,} output"
                         f" · {snapshot['cached_tokens']:,} cached input tokens")
            if snapshot["missing_usage"]:
                lines.append(f"Token totals incomplete: {snapshot['missing_usage']} calls omitted usage.")
            if snapshot["priced_requests"] and not selection.provider.endswith("_subscription"):
                lines.append(f"SDK-reported cost: ${snapshot['cost']:.4f} USD"
                             f" ({snapshot['priced_requests']}/{snapshot['requests']} calls priced)")
        else:
            lines.append("No completed calls observed for this connection yet.")
        lines.extend(_rate_limit_lines(snapshot, selection.provider))
    except (RuntimeError, ValueError, OSError):
        lines.append("Local usage unavailable: connection credentials could not be resolved.")
    lines.append("")
    if selection.provider == "openai_subscription":
        from infinidev.config.codex_usage import read_codex_limits

        try:
            lines.append(format_codex_limits(read_codex_limits()))
        except (RuntimeError, OSError) as exc:
            lines.append(f"Subscription quota unavailable: {exc}")
        except (ValueError, TypeError, KeyError, AttributeError):
            lines.append("Subscription quota unavailable: unrecognized Codex response.")
    elif selection.provider in {"openai", "anthropic"}:
        if selection.admin_key:
            lines.extend(_admin_report(selection))
        else:
            field_name = f"INFINIDEV_USAGE_{selection.provider.upper()}_ADMIN_KEY"
            lines.append(f"Organization billing requires a reporting/admin key: {field_name}.")
        lines.append("API billing and rate limits are separate from Codex/Claude subscription quotas.")
    elif selection.provider == "ollama":
        lines.append("Local provider: no hosted subscription quota or billing balance.")
    else:
        lines.append("Account quota/balance lookup is not integrated for this provider. "
                     "Use its dashboard; local usage above remains available.")
    return "\n".join(lines)
