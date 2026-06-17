/**
 * Platform bridge — the same build runs in a browser (`npm run dev`) and in
 * the Tauri desktop shell. Native calls are guarded by a runtime check and a
 * lazy import, with a web fallback on the other branch.
 */

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

async function invokeTauri<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<T>(command, args);
}

/** Public invoke wrapper for the embedded Rust engine commands. */
export function invoke<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  return invokeTauri<T>(command, args);
}

/** Subscribe to the engine's event stream; returns an unlisten function. */
export async function onEngineEvent(cb: (payload: unknown) => void): Promise<() => void> {
  const { listen } = await import("@tauri-apps/api/event");
  return listen("engine://event", (e) => cb(e.payload));
}

/** Subscribe to native-menu actions (`menu://action`); returns an unlisten fn. */
export async function onMenuAction(cb: (action: string) => void): Promise<() => void> {
  const { listen } = await import("@tauri-apps/api/event");
  return listen<string>("menu://action", (e) => cb(e.payload));
}

/** Subscribe to engine host questions (`engine://ask`); returns an unlisten fn. */
export async function onAskPrompt(cb: (payload: unknown) => void): Promise<() => void> {
  const { listen } = await import("@tauri-apps/api/event");
  return listen("engine://ask", (e) => cb(e.payload));
}

/** Open the native folder picker; returns the chosen path or null if cancelled. */
export async function pickFolder(): Promise<string | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const picked = await open({ directory: true, multiple: false, title: "Open Folder" });
  return typeof picked === "string" ? picked : null;
}

const DEFAULT_SERVER_URL = "http://127.0.0.1:8765";

let cachedServerUrl: string | null = null;

/**
 * Resolve the backend base URL.
 *  - In Tauri, ask the Rust shell which port it launched the sidecar on
 *    (`get_server_url` command).
 *  - In a browser, use VITE_SERVER_URL or the conventional default.
 */
export async function getServerUrl(): Promise<string> {
  if (cachedServerUrl) return cachedServerUrl;
  const envUrl = import.meta.env.VITE_SERVER_URL as string | undefined;
  if (envUrl) {
    cachedServerUrl = envUrl.replace(/\/$/, "");
    return cachedServerUrl;
  }
  if (isTauri()) {
    try {
      const url = await invokeTauri<string>("get_server_url");
      if (url) {
        cachedServerUrl = url.replace(/\/$/, "");
        return cachedServerUrl;
      }
    } catch {
      // Fall through to the default below.
    }
  }
  cachedServerUrl = DEFAULT_SERVER_URL;
  return cachedServerUrl;
}

/** Derive the WebSocket URL (ws[s]://…/ws) from the HTTP base URL. */
export function toWebSocketUrl(httpUrl: string): string {
  return httpUrl.replace(/^http/, "ws") + "/ws";
}

/** Open a URL in the system browser (native) or a new tab (web). */
export async function openExternalUrl(url: string): Promise<void> {
  if (!/^https?:\/\//i.test(url)) return;
  if (isTauri()) {
    try {
      await invokeTauri<void>("open_external_url", { url });
      return;
    } catch {
      // fall through
    }
  }
  window.open(url, "_blank", "noopener,noreferrer");
}
