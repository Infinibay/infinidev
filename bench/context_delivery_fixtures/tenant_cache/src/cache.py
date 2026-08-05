"""A small tenant-aware user cache."""


class UserCache:
    def __init__(self):
        self._values = {}

    def put(self, tenant_id: str, user_id: str, value: object) -> None:
        self._values[(tenant_id, user_id)] = value

    def get(self, tenant_id: str, user_id: str) -> object | None:
        return self._values.get((tenant_id, user_id))

    def invalidate(self, tenant_id: str, user_id: str) -> None:
        self._values.pop(user_id, None)
