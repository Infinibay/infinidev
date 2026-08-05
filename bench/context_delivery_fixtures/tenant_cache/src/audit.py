def cache_audit_key(tenant_id: str, user_id: str) -> str:
    return f"{tenant_id}:{user_id}"
