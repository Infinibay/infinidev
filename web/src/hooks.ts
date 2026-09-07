import { useCallback, useEffect, useRef, useState } from "react";
import { api, openStream } from "./api";
import { reduceSession } from "./state";
import type { Session } from "./types";
export function useResource<T>(path: string | null, interval = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(Boolean(path));
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision((v) => v + 1), []);
  useEffect(() => {
    if (!path) {
      setData(null);
      setLoading(false);
      return;
    }
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    setData(null);
    setLoading(true);
    setError("");
    async function load() {
      try {
        const value = await api<T>(
          path!,
          undefined,
          undefined,
          controller.signal,
        );
        if (active) {
          setData(value);
          setError("");
        }
      } catch (e) {
        if (active) setError((e as Error).message);
      } finally {
        if (active) {
          setLoading(false);
          if (interval) timer = setTimeout(load, interval);
        }
      }
    }
    void load();
    return () => {
      active = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [path, interval, revision]);
  return { data, error, loading, reload, setData };
}
export function useSession(id: string | null) {
  const [session, setSession] = useState<Session | null>(null);
  const [connection, setConnection] = useState<
    "connecting" | "live" | "offline"
  >("connecting");
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    setSession(null);
    if (!id) return;
    let closed = false;
    let socket: WebSocket;
    let timeout: ReturnType<typeof setTimeout>;
    let attempts = 0;
    function connect() {
      setConnection("connecting");
      socket = openStream(id!);
      socket.onmessage = (message) => {
        const event = JSON.parse(message.data);
        if (event.type === "snapshot") {
          attempts = 0;
          setConnection("live");
        }
        setSession((previous) => reduceSession(previous, event));
        if (event.type === "refresh") setRefresh((r) => r + 1);
      };
      socket.onclose = () => {
        if (closed) return;
        setConnection("offline");
        timeout = setTimeout(connect, Math.min(1000 * 2 ** attempts++, 15000));
      };
    }
    connect();
    return () => {
      closed = true;
      clearTimeout(timeout);
      socket?.close();
    };
  }, [id]);
  return { session, connection, refresh };
}
export function useFollow(dependency: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  useEffect(() => {
    if (following.current && ref.current)
      ref.current.scrollTop = ref.current.scrollHeight;
  }, [dependency]);
  return {
    ref,
    onScroll: () => {
      const el = ref.current;
      if (el)
        following.current =
          el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    },
  };
}

export function useTeam(sessionId: string | null) {
  const [team, setTeam] = useState<import("./types").Team | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    setTeam(null);
    setError("");
    if (!sessionId) return;
    let stopped = false;
    let cursor = 0;
    let events: import("./types").TeamEvent[] = [];
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    async function poll() {
      try {
        const result = await api<import("./types").Team>(
          `/sessions/${sessionId}/team?after=${cursor}`,
          undefined,
          undefined,
          controller.signal,
        );
        if (stopped) return;
        cursor = result.next;
        events = [...events, ...result.events].slice(-1000);
        const superseded = new Set(
          events.map(
            (e) => (e as unknown as { supersedes?: number }).supersedes,
          ),
        );
        setTeam({
          ...result,
          events: events.filter((e) => !superseded.has(e.id)),
        });
        setError("");
        timer = setTimeout(poll, result.has_more ? 50 : 2000);
      } catch (e) {
        if (!stopped) {
          setError((e as Error).message);
          timer = setTimeout(poll, 5000);
        }
      }
    }
    void poll();
    return () => {
      stopped = true;
      controller.abort();
      clearTimeout(timer);
    };
  }, [sessionId]);
  return { team, error };
}
