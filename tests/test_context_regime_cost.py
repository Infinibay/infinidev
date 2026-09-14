"""The recompose-vs-append cost model, and what a provider's cache price does.

The engine's context is not bounded: it appends and compacts. Whether that is
the right choice depends on a cache read costing a fraction of a fresh input
token, and the fraction differs by provider. These tests pin the arithmetic that
decides it, including the two results that are easy to get backwards: the
stable prefix cancels out of the comparison, and a *weaker* cache is what makes
recomposition look better.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.context_regime_cost import (
    break_even_working_set,
    build_report,
    effective_cost,
    load_curves,
    prices_for,
    tie_discount,
)


def test_a_cache_read_is_priced_relative_to_the_input_token() -> None:
    prices = prices_for("minimax/MiniMax-M3")

    assert prices["input"] == 3e-07
    assert prices["cache_read"] == 6e-08
    assert prices["discount"] == pytest.approx(0.2)
    assert prices["has_cache"] is True
    # MiniMax publishes no write surcharge.
    assert prices["write_ratio"] == 0.0


def test_a_model_with_no_cache_price_is_not_given_someone_elses() -> None:
    """An absent cache price means a repeat costs full, not a fallback rate.

    Substituting MiniMax's 0.06/M would have handed Groq and Fireworks a
    discount they do not publish, and which providers actually have a cache is
    the entire subject of the table.
    """
    prices = prices_for("groq/llama-3.3-70b-versatile")

    assert prices["in_cost_map"] is True
    assert prices["discount"] == 1.0
    assert prices["has_cache"] is False


def test_an_anthropic_write_premium_is_preserved() -> None:
    prices = prices_for("claude-sonnet-4-5")

    assert prices["discount"] == pytest.approx(0.1)
    # 3.75/M written against 3.00/M read: a write costs a quarter more than a
    # fresh token, which both regimes pay equally and so neither is penalised.
    assert prices["write_ratio"] == pytest.approx(1.25)


def test_a_weaker_cache_favours_recomposition() -> None:
    """No cache is the kindest case for recomposing, not a free one.

    A cache is what makes appending cheap, so ``k = 1`` — a repeat at full
    price — maximises the working set recomposition may carry. Getting this
    backwards inverts the whole conclusion.
    """
    prefix, increment, requests = 30_000, 2_400, 10

    bars = [
        break_even_working_set(requests, prefix, increment, discount=k)
        for k in (0.0, 0.1, 0.2, 0.5, 1.0)
    ]

    assert bars == sorted(bars), "the bar must rise as the cache gets weaker"
    assert bars[-1] > bars[0]
    # With no cache at all the bar is exactly half the history the append
    # regime would have accumulated.
    assert abs(bars[-1] - increment * (requests - 1) / 2) < 1e-9


def test_a_free_cache_makes_appending_almost_unbeatable() -> None:
    prefix, increment, requests = 30_000, 2_400, 10

    free = break_even_working_set(requests, prefix, increment, discount=0.0)
    uncached = break_even_working_set(requests, prefix, increment, discount=1.0)

    assert abs(free - increment * (requests - 1) / requests) < 1e-9
    assert free < uncached / 4


def test_the_stable_prefix_cancels_out_of_the_comparison() -> None:
    """Both regimes send it N times and cache-read it N-1 times, so it is free.

    It is why this question is about cache pricing and increment size at all,
    and not about how big the system prompt happens to be.
    """
    requests, increment, working_set = 10, 2_400, 3_000
    costs = []
    for prefix in (10_000, 60_000, 200_000):
        appended = effective_cost(requests, prefix, increment, discount=0.2)
        bounded = effective_cost(
            requests,
            prefix,
            increment,
            discount=0.2,
            bounded_working_set=working_set,
        )
        costs.append(round(appended - bounded))

    assert len(set(costs)) == 1


def test_the_break_even_is_the_point_where_the_two_regimes_tie() -> None:
    requests, prefix, increment = 10, 30_000, 2_400
    for discount in (0.1, 0.2, 1.0):
        bar = break_even_working_set(requests, prefix, increment, discount=discount)
        appended = effective_cost(requests, prefix, increment, discount=discount)

        at_bar = effective_cost(
            requests, prefix, increment, discount=discount, bounded_working_set=bar
        )
        assert abs(at_bar - appended) < 1.0
        # Just under it recomposing wins; just over it loses.
        assert (
            effective_cost(
                requests,
                prefix,
                increment,
                discount=discount,
                bounded_working_set=bar + 500,
            )
            > appended
        )
        assert (
            effective_cost(
                requests,
                prefix,
                increment,
                discount=discount,
                bounded_working_set=bar - 500,
            )
            < appended
        )


def test_the_tie_discount_inverts_the_break_even() -> None:
    requests, prefix, increment = 10, 30_000, 2_400
    working_set = 3_000

    k = tie_discount(requests, working_set, increment)

    for discount, expect_recompose in ((k - 0.05, False), (k + 0.05, True)):
        appended = effective_cost(requests, prefix, increment, discount=discount)
        bounded = effective_cost(
            requests,
            prefix,
            increment,
            discount=discount,
            bounded_working_set=working_set,
        )
        assert (bounded < appended) is expect_recompose


def test_a_working_set_too_large_for_any_cache_price_has_no_tie() -> None:
    """Past ``d(N-1)/2`` no provider's price list can save recomposition."""
    requests, increment = 10, 2_400
    hopeless = increment * (requests - 1) / 2 + 1_000

    assert tie_discount(requests, hopeless, increment) == 0.0


def _artifact(
    tmp_path: Path,
    name: str,
    payloads: list[int],
    *,
    user_chars: int | None = None,
    **extra: object,
) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True)
    body: dict[str, object] = {
        "task": {"id": name},
        "request_payload_history": [
            {"sequence": index, "message_payload_chars": value, "tool_schema_chars": 0}
            for index, value in enumerate(payloads)
        ],
        **extra,
    }
    if user_chars is not None:
        body["prompt_composition_history"] = [{"iteration": 0, "user_chars": user_chars}]
    (directory / "run.json").write_text(json.dumps(body), encoding="utf-8")
    return directory / "run.json"


def test_curves_separate_an_absent_cache_field_from_a_reported_miss(tmp_path: Path) -> None:
    _artifact(tmp_path, "absent", [10_000, 12_000, 14_000], provider_prompt_tokens=5_000)
    _artifact(
        tmp_path,
        "measured",
        [10_000, 12_000, 14_000],
        provider_prompt_tokens=5_000,
        cached_prefix_tokens=2_000,
    )
    _artifact(
        tmp_path,
        "miss",
        [10_000, 12_000, 14_000],
        provider_prompt_tokens=5_000,
        cached_prefix_tokens=0,
    )

    curves = {curve.task: curve for curve in load_curves(tmp_path)}

    # Absent: the instrument was not watching, so there is no hit rate at all.
    assert curves["absent"].observed_hit is None
    assert curves["measured"].observed_hit == 0.4
    # Present and zero is a real miss, and a different fact from absent.
    assert curves["miss"].observed_hit == 0.0


def test_the_working_set_is_taken_from_the_recomposed_prompt(tmp_path: Path) -> None:
    """Not assumed: it is the prompt the engine itself built for an iteration."""
    for index in range(3):
        _artifact(
            tmp_path,
            f"run{index}",
            [40_000, 42_400, 44_800],
            user_chars=5_000 + index,
        )

    report = build_report(tmp_path, ("minimax/MiniMax-M3",))

    assert report["working_set_chars"] == 5_001
    assert report["working_set_used_chars"] == 5_001
    assert report["runs_measured"] == 3


def test_a_real_cache_keeps_the_append_regime_and_a_missing_one_can_flip_it(
    tmp_path: Path,
) -> None:
    """The per-provider answer, on the same measured working set.

    A 5 000-character working set sits between the bar a good cache allows
    (~2 900 characters) and the bar no cache at all allows (~10 200), so the
    provider's price list decides — which is the whole point of the table.
    """
    for index in range(3):
        _artifact(
            tmp_path,
            f"run{index}",
            [29_600, 31_900, 34_200, 36_500, 38_800, 41_100, 43_400, 45_700, 48_000, 50_300],
            user_chars=5_000,
        )

    report = build_report(
        tmp_path, ("minimax/MiniMax-M3", "groq/llama-3.3-70b-versatile")
    )
    by_model = {row["model"]: row for row in report["providers"]}

    assert by_model["minimax/MiniMax-M3"]["cache_discount"] == pytest.approx(0.2)
    assert by_model["minimax/MiniMax-M3"]["verdict"] == "append"
    # No cache published: appending pays for its whole transcript every round.
    assert by_model["groq/llama-3.3-70b-versatile"]["cache_discount"] == 1.0
    assert by_model["groq/llama-3.3-70b-versatile"]["verdict"] == "recompose"
    assert by_model["groq/llama-3.3-70b-versatile"]["cache_saves_usd_per_run"] == 0.0
    # And the tie point sits between the two cache prices, as it must.
    assert (
        by_model["minimax/MiniMax-M3"]["cache_discount"]
        < report["tie_discount"]
        < by_model["groq/llama-3.3-70b-versatile"]["cache_discount"]
    )


def test_compaction_is_visible_as_a_sawtooth(tmp_path: Path) -> None:
    _artifact(tmp_path, "grows", [10_000, 20_000, 30_000])
    _artifact(tmp_path, "compacts", [10_000, 30_000, 12_000, 24_000])

    curves = {curve.task: curve for curve in load_curves(tmp_path)}

    assert curves["grows"].compacted is False
    assert curves["compacts"].compacted is True
