"""Price two ways of organising a coding agent's context, per provider.

The question this answers: is it cheaper to

* **recompose** — rebuild a small working set for every request, keeping the
  payload flat; or
* **append** — grow an incremental transcript until it crosses a threshold and
  then compact, so each request is (almost) the previous request plus an
  increment and the cache serves everything older.

The answer turns on four numbers — the first request's payload, the per-round
increment ``d``, the round count ``N``, and the provider's cache prices — so
this tool measures the first three from the stored run artifacts and reads the
fourth out of the provider cost map.

Why the stable prefix does not decide it
----------------------------------------
Both regimes send the same stable prefix on all ``N`` requests and read it from
the cache on ``N - 1`` of them.  The prefix is therefore *regime-neutral*: it
costs the same either way, and it cancels out of the comparison.  So does a
cache-*write* premium, for the same reason — both regimes write the prefix the
same number of times.

What is left is the variable part, and only that:

.. code-block:: text

    D  <=  d * (N - 1) * [ N - (1 - k)(N - 2) ] / (2N)

``D`` is the bounded working set a recomposing harness may carry and still come
out ahead; ``k`` is a cache read as a fraction of a fresh input token, so *lower
is a better cache* and ``k = 1`` means a repeat costs full price.  Two things
follow, and neither is obvious:

* **No cache at all (``k = 1``) maximises the bar at ``d(N-1)/2``.**  A cache is
  what makes appending cheap, so the weaker the cache the better recomposition
  fares — and even with no cache at all the bar is only ``d(N-1)/2``.  This tool
  prints the whole sweep, so the verdict is checked against the strongest case
  for recomposition rather than against one vendor's list.
* **The ordering never flips between providers.**  ``k`` moves the bar by a
  factor of at most ``(N-1)/N`` versus ``(N-1)/2``; it changes how much headroom
  recomposition gets, not who wins.  So the honest per-provider answer is a
  table of margins and money, not a table of different winners.

Usage::

    python -m bench.context_regime_cost --runs bench/runs
    python -m bench.context_regime_cost --runs bench/runs --models minimax/MiniMax-M3
    python -m bench.context_regime_cost --runs bench/runs --json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Published prices, USD per token, used when a model is absent from the cost
#: map.  MiniMax-M3's: a cache read at a fifth of a fresh input token.
_FALLBACK_PRICES = {
    "input": 3e-07,
    "cache_read": 6e-08,
    "cache_write": 0.0,
    "output": 1.2e-06,
}

#: One flagship per provider family, chosen so the table spans the whole range
#: of cache discounts rather than one vendor's policy.  A model whose read price
#: equals its input price is a provider with no cache to speak of, which is the
#: ``k = 1`` end of the sweep and worth keeping in the table.
DEFAULT_MODELS = (
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "gpt-5.1",
    "gpt-5.2",
    "gemini/gemini-2.5-pro",
    "deepseek/deepseek-reasoner",
    "minimax/MiniMax-M3",
    "xai/grok-code-fast-1",
    "mistral/mistral-large-latest",
    "groq/llama-3.3-70b-versatile",
    "fireworks_ai/accounts/fireworks/models/deepseek-v3",
)

#: A run needs at least this many recorded requests for its growth to mean
#: anything.
_MIN_REQUESTS = 3

#: A request that shrinks by more than this share of the previous one is the
#: compaction threshold firing, not noise.
_COMPACTION_DROP = 0.8

#: Characters per token, for reading the model's character-space parameters in
#: the token-space units a price list uses.  MiniMax-M3 measures 8 chars/token
#: on reasoning and closer to 4 on prose; 4 is the conservative end.
_CHARS_PER_TOKEN = 4.0


def prices_for(model: str) -> dict[str, float]:
    """Published per-token prices for *model*, with the cache ratios resolved.

    ``discount`` is a cache read over a fresh input token, so lower is a better
    cache and ``1.0`` means the provider charges full price for a repeat.

    A model whose entry carries no cache price at all is treated as having no
    cache (``discount = 1.0``) rather than as having someone else's.  The
    difference matters: substituting a fallback would quietly hand Groq and
    Fireworks a discount they do not publish, and the whole point of the table
    is which providers actually have one.
    """
    try:
        import litellm

        entry = litellm.model_cost.get(model) or {}
    except Exception:  # pragma: no cover - litellm is a hard dependency
        entry = {}
    input_price = float(entry.get("input_cost_per_token") or _FALLBACK_PRICES["input"])
    cache_read = entry.get("cache_read_input_token_cost")
    if cache_read is None:
        # Some entries spell the same thing as a price *for* a cache hit.
        cache_read = entry.get("input_cost_per_token_cache_hit")
    prices = {
        "input": input_price,
        "cache_read": float(cache_read) if cache_read is not None else input_price,
        "cache_write": float(entry.get("cache_creation_input_token_cost") or 0.0),
        "output": float(entry.get("output_cost_per_token") or _FALLBACK_PRICES["output"]),
    }
    prices["discount"] = prices["cache_read"] / input_price
    prices["write_ratio"] = prices["cache_write"] / input_price
    prices["has_cache"] = cache_read is not None and prices["discount"] < 1.0
    prices["in_cost_map"] = bool(entry)
    return prices


@dataclass
class RunCurve:
    """One run's measured context curve."""

    path: str
    task: str
    requests: int
    payloads: list[int]
    billed_tokens: int
    #: ``None`` when the artifact predates the field, which is a different fact
    #: from a provider reporting that nothing was cached.
    cached_tokens: int | None
    #: Size of the prompt the engine *recomposed* for an iteration — task, plan,
    #: active step, previous actions.  This, and not the whole payload, is the
    #: working set a bounded regime has to carry, and it is the quantity the
    #: break-even is compared against.  ``None`` when the run predates it.
    working_set_chars: int | None = None

    @property
    def first(self) -> int:
        return self.payloads[0]

    @property
    def last(self) -> int:
        return self.payloads[-1]

    @property
    def increment(self) -> float:
        """Characters added per request across the run."""
        if self.requests < 2:
            return 0.0
        return (self.last - self.first) / (self.requests - 1)

    @property
    def compacted(self) -> bool:
        return any(b < a * _COMPACTION_DROP for a, b in zip(self.payloads, self.payloads[1:]))

    @property
    def observed_hit(self) -> float | None:
        if not self.billed_tokens or self.cached_tokens is None:
            return None
        return self.cached_tokens / self.billed_tokens


def load_curves(run_root: Path) -> list[RunCurve]:
    curves: list[RunCurve] = []
    for path in sorted(glob.glob(str(run_root / "**" / "run.json"), recursive=True)):
        try:
            raw = json.loads(Path(path).read_text())
        except Exception:
            continue
        history = raw.get("request_payload_history")
        if not isinstance(history, list) or len(history) < _MIN_REQUESTS:
            continue
        ordered = sorted(history, key=lambda row: row.get("sequence", 0))
        payloads = [
            int(row.get("message_payload_chars", 0)) + int(row.get("tool_schema_chars", 0))
            for row in ordered
        ]
        if payloads[0] <= 0 or payloads[-1] <= payloads[0]:
            continue
        composed = raw.get("prompt_composition_history") or []
        user_sizes = [
            int(row["user_chars"])
            for row in composed
            if isinstance(row, dict) and isinstance(row.get("user_chars"), int)
            and row["user_chars"] > 0
        ]
        curves.append(
            RunCurve(
                path=path,
                working_set_chars=(
                    int(stats.median(user_sizes)) if user_sizes else None
                ),
                task=str((raw.get("task") or {}).get("id", "?")),
                requests=len(payloads),
                payloads=payloads,
                billed_tokens=int(raw.get("provider_prompt_tokens") or 0),
                cached_tokens=(
                    int(raw["cached_prefix_tokens"])
                    if raw.get("cached_prefix_tokens") is not None
                    else None
                ),
            )
        )
    return curves


def effective_cost(
    requests: int,
    prefix: float,
    increment: float,
    *,
    discount: float,
    bounded_working_set: float | None = None,
    history_hit: float = 1.0,
) -> float:
    """Fresh-token equivalents of one regime.

    ``bounded_working_set`` selects the regime, and the two differ in *what the
    cache can serve*, which is the whole point:

    * ``None`` — the append harness.  Each request is the previous request plus
      an increment, so the longest common prefix is the entire previous request
      and only the newest material is a cache miss.
    * a number — the recompose harness.  The stable ``prefix`` repeats and is
      cached, but the working set is *rebuilt* each round, so its content
      differs every time and none of it can be a cache hit.  That is the price
      of recomposition, and the model must not hide it.

    ``history_hit`` is the share of the previous request the cache actually
    serves in the append regime.  Below 1.0 it models the calls outside the
    loop — a different system prompt on the chat agent, the planner, the
    reviewer — which miss for reasons that belong to neither regime.

    Cost is in fresh-input-token equivalents: a cache read contributes
    ``discount`` of one.
    """
    if bounded_working_set is None:
        payloads = [prefix + increment * index for index in range(requests)]
        billed = sum(payloads)
        cached = history_hit * sum(payloads[:-1])
    else:
        billed = requests * (prefix + bounded_working_set)
        cached = (requests - 1) * prefix
    return billed - (1.0 - discount) * cached


def break_even_working_set(
    requests: int,
    prefix: float,
    increment: float,
    *,
    discount: float,
    history_hit: float = 1.0,
) -> float:
    """Largest bounded working set that still beats the append regime.

    Solved in closed form, because the answer is the whole point of the tool and
    a bisection would hide its shape:

    .. code-block:: text

        D = [ d(N-1)(N - (1-k)h(N-2)) / 2 + (1-k)(N-1)P(1-h) ] / N

    With a perfect cache (``h = 1``) the ``P`` term vanishes — the stable prefix
    is regime-neutral — and ``D`` rises monotonically in ``k``, topping out at
    ``d(N-1)/2`` when the cache is free.
    """
    if requests < 2:
        return 0.0
    tail = (
        increment
        * (requests - 1)
        * (requests - (1.0 - discount) * history_hit * (requests - 2))
        / 2.0
    )
    drag = (1.0 - discount) * (requests - 1) * prefix * (1.0 - history_hit)
    return max(tail + drag, 0.0) / requests


def tie_discount(
    requests: int, working_set: float, increment: float, *, history_hit: float = 1.0
) -> float:
    """Cache discount at which recomposing and appending cost the same.

    Above this the provider's cache is weak enough that recomposition pays;
    below it the cache is what makes appending cheap and appending wins.  Invert

        N*D = d(N-1)[N - (1-k)h(N-2)]/2

    for ``k``, which is only possible while the working set is small enough to
    have a solution at all — past ``d(N-1)/2`` no cache price saves it.
    """
    if requests < 3 or not increment:
        return 0.0
    reach = requests / (increment * (requests - 1))
    span = requests - 2.0 * reach * working_set
    if span <= 0:
        # The working set is past the point any cache price could rescue.
        return 0.0
    return 1.0 - span / ((requests - 2) * history_hit)


def crossover_rounds(
    prefix: float,
    increment: float,
    working_set: float,
    *,
    discount: float,
    history_hit: float = 1.0,
) -> int:
    """First round count at which recomposing to *working_set* is cheaper."""
    for requests in range(2, 400):
        appended = effective_cost(
            requests, prefix, increment, discount=discount, history_hit=history_hit
        )
        bounded = effective_cost(
            requests, prefix, increment, discount=discount, bounded_working_set=working_set
        )
        if bounded < appended:
            return requests
    return -1


def money(fresh_tokens: float, prices: dict[str, float]) -> float:
    return fresh_tokens * prices["input"]


def _sweep_row(report: dict[str, Any], discount: float) -> float:
    """The break-even working set, in tokens, for one cache discount."""
    for row in report["break_even_sweep"]:
        if abs(row["cache_discount"] - discount) < 1e-9:
            return row["break_even_working_set_tokens"]
    return 0.0


def summarise_curves(curves: list[RunCurve]) -> dict[str, Any]:
    counts = [c.requests for c in curves]
    increments = [c.increment for c in curves]
    hits = [c.observed_hit for c in curves if c.observed_hit]
    work = sorted(c.working_set_chars for c in curves if c.working_set_chars)
    return {
        "runs_measured": len(curves),
        "runs_monotone": sum(1 for c in curves if not c.compacted),
        "runs_compacted": sum(1 for c in curves if c.compacted),
        "first_payload_chars": stats.median([c.first for c in curves]),
        "increment_chars": stats.median(increments),
        "increment_chars_p90": sorted(increments)[int(0.9 * len(increments))],
        "payload_growth_median": stats.median([c.last / c.first for c in curves]),
        "requests_median": int(stats.median(counts)),
        "requests_p90": sorted(counts)[int(0.9 * len(counts))],
        "observed_hit_rate": stats.median(hits) if hits else None,
        "observed_hit_runs": len(hits),
        "runs_reporting_a_miss": sum(1 for c in curves if c.cached_tokens == 0 and c.billed_tokens),
        "runs_without_the_metric": sum(1 for c in curves if c.cached_tokens is None),
        "working_set_runs": len(work),
        "working_set_chars": stats.median(work) if work else None,
        "working_set_chars_p10": work[int(0.1 * len(work))] if work else None,
        "working_set_chars_p90": work[int(0.9 * len(work))] if work else None,
    }


def build_report(run_root: Path, models: tuple[str, ...]) -> dict[str, Any]:
    curves = load_curves(run_root)
    if not curves:
        raise SystemExit(f"no usable run artifacts under {run_root}")
    measured = summarise_curves(curves)
    first_payload = measured["first_payload_chars"]
    increment = measured["increment_chars"]
    requests = measured["requests_median"]
    hit = measured["observed_hit_rate"]
    measured_working_set = measured["working_set_chars"] or increment

    # The bar recomposition must clear, swept over every cache price that could
    # exist.  The top of the sweep is a free cache, which no provider offers, so
    # the last row bounds every provider table below.
    sweep = []
    for discount in (0.0, 0.1, 0.16, 0.2, 0.25, 0.5, 1.0):
        bar = break_even_working_set(
            requests, first_payload, increment, discount=discount
        )
        sweep.append(
            {
                "cache_discount": discount,
                "break_even_working_set_chars": bar,
                "break_even_working_set_tokens": bar / _CHARS_PER_TOKEN,
            }
        )

    providers = []
    for model in models:
        prices = prices_for(model)
        discount = prices["discount"]
        bar = break_even_working_set(
            requests, first_payload, increment, discount=discount
        )
        diluted = (
            break_even_working_set(
                requests, first_payload, increment, discount=discount, history_hit=hit
            )
            if hit
            else None
        )
        appended = effective_cost(requests, first_payload, increment, discount=discount)
        # The working set is measured, not assumed: it is the prompt the engine
        # itself recomposed for an iteration, which is exactly what a bounded
        # regime would have to carry.
        recomposed = effective_cost(
            requests,
            first_payload,
            increment,
            discount=discount,
            bounded_working_set=measured_working_set,
        )
        # What the same run costs with no cache at all, to show what the cache
        # is actually worth.
        uncached = effective_cost(requests, first_payload, increment, discount=1.0)
        providers.append(
            {
                "model": model,
                "in_cost_map": prices["in_cost_map"],
                "has_cache": prices["has_cache"],
                "input_usd_per_m": prices["input"] * 1e6,
                "cache_read_usd_per_m": prices["cache_read"] * 1e6,
                "cache_write_usd_per_m": prices["cache_write"] * 1e6,
                "cache_discount": discount,
                "cache_write_ratio": prices["write_ratio"],
                "break_even_working_set_tokens": bar / _CHARS_PER_TOKEN,
                "break_even_working_set_tokens_diluted": (
                    diluted / _CHARS_PER_TOKEN if diluted is not None else None
                ),
                "append_fresh_tokens": appended / _CHARS_PER_TOKEN,
                "append_usd_per_run": money(appended / _CHARS_PER_TOKEN, prices),
                "recompose_usd_per_run": money(recomposed / _CHARS_PER_TOKEN, prices),
                "verdict": "append" if appended <= recomposed else "recompose",
                "append_usd_per_run_at_measured_working_set": money(
                    appended / _CHARS_PER_TOKEN, prices
                ),
                "no_cache_usd_per_run": money(uncached / _CHARS_PER_TOKEN, prices),
                "cache_saves_usd_per_run": money(
                    (uncached - appended) / _CHARS_PER_TOKEN, prices
                ),

            }
        )

    return {
        **measured,
        "chars_per_token": _CHARS_PER_TOKEN,
        "working_set_used_chars": measured_working_set,
        "tie_discount": tie_discount(requests, measured_working_set, increment),
        "tie_discount_p10_working_set": (
            tie_discount(requests, measured["working_set_chars_p10"], increment)
            if measured.get("working_set_chars_p10")
            else None
        ),
        "break_even_sweep": sweep,
        "providers": providers,
    }


def render(report: dict[str, Any]) -> str:
    n = report["requests_median"]
    d = report["increment_chars"]
    lines = [
        "=" * 94,
        "Context regime cost — recompose vs append, per provider",
        "=" * 94,
        "",
        "MEASURED, from the run artifacts",
        f"  runs                            {report['runs_measured']}"
        f"  ({report['runs_monotone']} append monotonically,"
        f" {report['runs_compacted']} compact)",
        f"  first request payload           {report['first_payload_chars']:,.0f} chars"
        f"  (~{report['first_payload_chars']/_CHARS_PER_TOKEN:,.0f} tokens)",
        f"  increment per request           {d:,.0f} chars"
        f"  (~{d/_CHARS_PER_TOKEN:,.0f} tokens),"
        f" {report['increment_chars_p90']:,.0f} at p90",
        f"  payload growth                  {report['payload_growth_median']:.2f}x"
        f" over {n} requests (median)",
    ]
    if report.get("working_set_chars"):
        ws = report["working_set_chars"]
        lines += [
            "",
            "  The working set a bounded regime must carry is not a guess: it is",
            "  the prompt the engine itself recomposed for an iteration.",
            f"    measured working set           {ws:,.0f} chars (~{ws/_CHARS_PER_TOKEN:,.0f} tokens)"
            f"  median over {report['working_set_runs']} iteration(s)",
            f"      lean end (p10)               {report['working_set_chars_p10']:,.0f} chars"
            f" (~{report['working_set_chars_p10']/_CHARS_PER_TOKEN:,.0f} tokens)",
            f"      heavy end (p90)              {report['working_set_chars_p90']:,.0f} chars"
            f" (~{report['working_set_chars_p90']/_CHARS_PER_TOKEN:,.0f} tokens)",
        ]
    if report["observed_hit_rate"] is not None:
        lines.append(
            f"  cache hit, whole pipeline       {report['observed_hit_rate']*100:.1f}%"
            f"  ({report['observed_hit_runs']} run(s) that reported a cache read)"
        )
    lines += [
        "",
        "THE BAR RECOMPOSITION MUST CLEAR",
        f"  D  <=  d(N-1)[N - (1-k)(N-2)] / 2N        d={d:,.0f} chars, N={n}",
        "  k is a cache read as a share of a fresh input token: 1.00 means a",
        "  repeat costs full price, 0.00 means it is free.  A cache is what makes",
        "  appending cheap, so the *weaker* the cache the better recomposition",
        "  fares — which is why k = 1.00, no cache at all, is the kindest case for",
        "  it and bounds every provider below:",
        "",
        "    cache read / input (k)   break-even working set",
    ]
    for row in report["break_even_sweep"]:
        lines.append(
            f"    {row['cache_discount']:>20.2f}"
            f"     {row['break_even_working_set_chars']:9,.0f} ch"
            f"  (~{row['break_even_working_set_tokens']:,.0f} tok)"
        )
    top = report["break_even_sweep"][-1]["break_even_working_set_tokens"]
    lines += [
        "",
        f"  With no cache at all the bar is {top:,.0f} tokens, and no provider can",
        "  raise it beyond that — any real cache lowers the bar.  Against the",
        "  measured working set the two regimes therefore tie at a specific cache",
        "  price, and that single number is the per-provider answer:",
        "",
        f"    tie at k = {report['tie_discount']:.2f}"
        f"   (median working set)"
        + (
            f",  k = {report['tie_discount_p10_working_set']:.2f} at the lean end"
            if report.get("tie_discount_p10_working_set") is not None
            else ""
        ),
        "    a provider whose cache read costs *less* than that favours appending;",
        "    one whose cache read costs more favours recomposing.  A negative tie",
        "    means recomposition is past the point any cache price could rescue it,",
        "    because the working set is already under the free-cache bar.",
        "",
        "PER PROVIDER",
        "",
        f"  {'model':36s} {'k':>5s} {'w':>5s} {'break-even':>12s} {'verdict':>10s}"
        f" {'append $/run':>13s} {'cache saves':>12s}",
    ]
    for row in report["providers"]:
        flag = "" if row["has_cache"] else " *"
        diluted = (
            f"  (diluted {row['break_even_working_set_tokens_diluted']:,.0f})"
            if row["break_even_working_set_tokens_diluted"]
            else ""
        )
        lines.append(
            f"  {row['model'][:36]:36s} {row['cache_discount']:5.2f}"
            f" {row['cache_write_ratio']:5.2f}"
            f" {row['break_even_working_set_tokens']:9,.0f} tok"
            f" {row['verdict']:>10s}"
            f" {row['append_usd_per_run']:13.4f}"
            f" {row['cache_saves_usd_per_run']:12.4f}{flag}{diluted}"
        )
    if any(not row["has_cache"] for row in report["providers"]):
        lines.append(
            "  * no cache price published for this model; a repeat is priced as"
            " a fresh token"
        )
    lines += [
        "",
        "  k = cache read / fresh input (lower is a better cache).",
        "  w = cache write / fresh input, so 1.25 is a write premium.",
        "  break-even = the working set at which recomposing ties the append",
        "  regime at that provider's prices.  'diluted' repeats it using the",
        "  measured whole-pipeline hit rate instead of a perfect one, which",
        "  favours recomposition and so makes the bar harder to clear, not easier.",
        "",
        "READING IT",
        "  Every provider lands on append, and the reason is structural rather",
        "  than commercial.  Both regimes send the same stable prefix N times and",
        "  cache-read it N-1 times, so the prefix cancels — and so does a write",
        "  premium, which both pay equally.  Only the variable part decides: for",
        "  the append regime that is one round's increment, for the recompose",
        "  regime it is the entire working set.  A coding task's working set —",
        "  plan, active step, the summaries of every closed step — is far larger",
        "  than one increment, so the comparison is not close at any price.",
        "",
        "  What the price list changes is the margin, the verdict and the money:",
        f"    Anthropic, OpenAI, Gemini, DeepSeek, Mistral (k=0.10)"
        f"   bar {_sweep_row(report, 0.10):,.0f} tok",
        f"    MiniMax M3, Grok Code Fast          (k=0.20)"
        f"   bar {_sweep_row(report, 0.20):,.0f} tok",
        f"    Groq, Fireworks: no cache billed    (k=1.00)"
        f"   bar {_sweep_row(report, 1.00):,.0f} tok",
        "  A provider with a real cache does not merely make appending win on",
        "  price; it lowers the bar so far that only a working set smaller than",
        "  the plan itself would save recomposition.  A provider with no cache at",
        "  all is the one case where recomposition can pay — and even then only",
        "  for a working set leaner than the measured median.",
        "  So the lever with a return is shrinking the increment — step",
        "  summarisation, the reasoning trim, the tool-result cap — rather than",
        "  reorganising the context to chase a hit rate.",
        "=" * 94,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("bench/runs"))
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated models to price; defaults to one flagship per provider",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    report = build_report(args.runs, models)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
