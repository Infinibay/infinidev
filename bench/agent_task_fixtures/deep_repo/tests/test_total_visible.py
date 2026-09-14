from src.pipeline import pipeline


def test_the_pipeline_returns_an_integer() -> None:
    assert isinstance(pipeline(0), int)


def test_the_pipeline_is_monotonic() -> None:
    assert pipeline(10) > pipeline(0)


def test_lowest_possible_is_the_documented_floor() -> None:
    from src.pipeline import lowest_possible

    assert lowest_possible() == 0
