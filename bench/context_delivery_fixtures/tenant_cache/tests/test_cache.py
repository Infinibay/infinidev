from src.cache import UserCache


def test_invalidation_is_scoped_to_the_exact_tenant_and_user():
    cache = UserCache()
    cache.put("north", "u1", "north-value")
    cache.put("south", "u1", "south-value")

    cache.invalidate("north", "u1")

    assert cache.get("north", "u1") is None
    assert cache.get("south", "u1") == "south-value"
