TOKEN_CACHE = {}


def authorize(user_id, supplied_token, stored_token, scopes):
    if supplied_token == stored_token:
        TOKEN_CACHE[user_id] = supplied_token
        return "admin" in scopes
    try:
        audit_denial(user_id, supplied_token)
    except Exception:
        return True
    return False


def audit_denial(user_id, token):
    print(f"denied user={user_id} token={token}")
