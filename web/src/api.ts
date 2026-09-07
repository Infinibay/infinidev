let token = sessionStorage.getItem("infinidev-token") || "";
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.has("token")) {
  token = fragment.get("token") || "";
  sessionStorage.setItem("infinidev-token", token);
  history.replaceState(null, "", location.pathname + location.search);
}
export function setToken(value: string) {
  token = value;
  sessionStorage.setItem("infinidev-token", value);
}
export function hasToken() {
  return Boolean(token);
}
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
export async function api<T>(
  path: string,
  body?: unknown,
  method?: string,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`/api${path}`, {
    method: method || (body === undefined ? "GET" : "POST"),
    signal,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new ApiError(
      response.status,
      typeof payload.detail === "string"
        ? payload.detail
        : `Request failed (${response.status}).`,
    );
  return payload as T;
}
export function openStream(sessionId: string) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return new WebSocket(
    `${protocol}//${location.host}/ws?session_id=${encodeURIComponent(sessionId)}`,
    ["infinidev", `token.${token}`],
  );
}
