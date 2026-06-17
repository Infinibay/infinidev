import { useEffect, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { TextField } from "@harbor/components/inputs/TextField";
import { Button } from "@harbor/components/buttons/Button";
import { Badge } from "@harbor/components/display/Badge";
import { useToast } from "@harbor/components/feedback/Toast";
import { useEngine } from "@/state/store";
import type { DocHit, DocLibrary, DocSection } from "@/api/engineEvent";

interface Props {
  open: boolean;
  onClose: () => void;
}

/** Browser for the local documentation cache (`library_docs`). Mirrors the
 *  TUI's three-panel docs browser: libraries → sections → content, plus a
 *  cross-library search and a form to fetch + cache a new docs URL. */
export function DocsModal({ open, onClose }: Props) {
  const { commands, state } = useEngine();
  const { push } = useToast();
  const [libs, setLibs] = useState<DocLibrary[]>([]);
  const [lib, setLib] = useState<string | null>(null);
  const [sections, setSections] = useState<DocSection[]>([]);
  const [section, setSection] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<DocHit[] | null>(null);

  const [adding, setAdding] = useState(false);
  const [fLib, setFLib] = useState("");
  const [fUrl, setFUrl] = useState("");
  const [fetching, setFetching] = useState(false);

  const loadLibs = () =>
    commands
      .docsLibraries()
      .then((l) => {
        setLibs(l);
        setLib((cur) => cur ?? l[0]?.library ?? null);
      })
      .catch(() => setLibs([]));

  useEffect(() => {
    if (!open) return;
    loadLibs();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, state.projectNonce]);

  // Load sections when the selected library changes.
  useEffect(() => {
    if (!open || !lib) {
      setSections([]);
      return;
    }
    commands
      .docsSections(lib)
      .then((s) => {
        setSections(s);
        setSection(s[0]?.section_title ?? null);
      })
      .catch(() => setSections([]));
  }, [open, lib, commands]);

  // Load content when the selected section changes.
  useEffect(() => {
    if (!open || !lib || !section) {
      setContent("");
      return;
    }
    commands.docsRead(lib, section).then(setContent).catch(() => setContent(""));
  }, [open, lib, section, commands]);

  const runSearch = (q: string) => {
    setQuery(q);
    if (!q.trim()) {
      setHits(null);
      return;
    }
    commands.docsSearch(q.trim()).then(setHits).catch(() => setHits([]));
  };

  const fetchDocs = async () => {
    if (!fLib.trim() || !fUrl.trim()) return;
    setFetching(true);
    try {
      const n = await commands.docsFetch(fLib.trim(), fUrl.trim());
      push({ title: `Cached ${n} section${n === 1 ? "" : "s"}`, tone: "success" });
      setFLib("");
      setFUrl("");
      setAdding(false);
      await loadLibs();
      setLib(fLib.trim());
    } catch (e) {
      push({ title: "Fetch failed", description: String((e as Error).message), tone: "danger" });
    } finally {
      setFetching(false);
    }
  };

  const removeLib = async (name: string) => {
    if (!window.confirm(`Delete cached docs for ${name}?`)) return;
    await commands.docsDelete(name);
    setLib(null);
    await loadLibs();
  };

  return (
    <Dialog open={open} onClose={onClose} size="lg" title="Documentation cache">
      <div className="flex items-center gap-2">
        <div className="flex-1">
          <TextField
            placeholder="Search all cached docs…"
            value={query}
            onChange={(e) => runSearch(e.target.value)}
          />
        </div>
        <Button variant={adding ? "ghost" : "secondary"} size="sm" onClick={() => setAdding((a) => !a)}>
          {adding ? "Cancel" : "Fetch docs"}
        </Button>
      </div>

      {adding && (
        <div className="mt-2 flex items-end gap-2 rounded-md border border-fg/10 bg-surface-2/40 p-2.5">
          <div className="w-40">
            <TextField placeholder="Library (e.g. tokio)" value={fLib} onChange={(e) => setFLib(e.target.value)} />
          </div>
          <div className="flex-1">
            <TextField placeholder="https://docs…/page" value={fUrl} onChange={(e) => setFUrl(e.target.value)} />
          </div>
          <Button variant="primary" size="sm" disabled={!fLib.trim() || !fUrl.trim() || fetching} onClick={() => void fetchDocs()}>
            {fetching ? "Fetching…" : "Fetch"}
          </Button>
        </div>
      )}

      {hits ? (
        <div className="mt-2 max-h-[55vh] space-y-1 overflow-y-auto">
          <div className="text-[10px] text-fg-subtle">
            {hits.length} match{hits.length === 1 ? "" : "es"} for “{query}”
          </div>
          {hits.map((h, i) => (
            <button
              key={i}
              onClick={() => {
                setHits(null);
                setQuery("");
                setLib(h.library);
                setSection(h.section_title);
              }}
              className="block w-full rounded px-2 py-1.5 text-left hover:bg-surface-2/60"
            >
              <span className="text-[12px] text-fg">
                {h.library} · {h.section_title}
              </span>
              <span className="block truncate text-[11px] text-fg-subtle">{h.snippet}</span>
            </button>
          ))}
        </div>
      ) : (
        <div className="mt-2 grid h-[55vh] grid-cols-[minmax(120px,22%)_minmax(120px,26%)_1fr] gap-3">
          {/* Libraries */}
          <div className="space-y-0.5 overflow-y-auto border-r border-fg/8 pr-2">
            {libs.length === 0 && (
              <div className="px-1 py-3 text-[11px] text-fg-subtle">
                No cached docs. Use <span className="mono">Fetch docs</span> or the agent's
                <span className="mono"> fetch_documentation</span> tool.
              </div>
            )}
            {libs.map((l) => (
              <button
                key={l.library}
                onClick={() => setLib(l.library)}
                className={`group flex w-full items-center justify-between rounded px-2 py-1.5 text-left ${
                  lib === l.library ? "bg-accent/15" : "hover:bg-surface-2/60"
                }`}
              >
                <span className="truncate text-[12px] text-fg">{l.library}</span>
                <span className="flex items-center gap-1">
                  <Badge tone="neutral">{l.sections}</Badge>
                  <span
                    onClick={(e) => {
                      e.stopPropagation();
                      void removeLib(l.library);
                    }}
                    className="hidden text-fg-subtle hover:text-danger group-hover:inline"
                    title="Delete"
                  >
                    🗑
                  </span>
                </span>
              </button>
            ))}
          </div>
          {/* Sections */}
          <div className="space-y-0.5 overflow-y-auto border-r border-fg/8 pr-2">
            {sections.map((s) => (
              <button
                key={s.section_title}
                onClick={() => setSection(s.section_title)}
                className={`block w-full truncate rounded px-2 py-1.5 text-left text-[12px] ${
                  section === s.section_title ? "bg-accent/15 text-fg" : "text-fg-muted hover:bg-surface-2/60"
                }`}
              >
                {s.section_title}
              </button>
            ))}
          </div>
          {/* Content */}
          <div className="overflow-y-auto">
            {content ? (
              <p className="whitespace-pre-wrap text-[13px] leading-relaxed text-fg-muted">{content}</p>
            ) : (
              <div className="grid h-full place-items-center text-[12px] text-fg-subtle">
                {lib ? "Select a section." : "Select a library."}
              </div>
            )}
          </div>
        </div>
      )}
    </Dialog>
  );
}
