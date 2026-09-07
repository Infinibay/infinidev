import { useEffect, useRef, useState } from "react";
import { Tabs } from "radix-ui";
import { Copy, Download, Play, Square, Terminal, X } from "lucide-react";
import { api } from "./api";
import { Badge, Empty, ErrorNotice, PageTitle, Tip } from "./components";
import { useFollow, useResource } from "./hooks";
import type { Process, ProcessOutput } from "./types";

export function ProcessesView() {
  const resource = useResource<{ processes: Process[] }>("/processes", 1500);
  const [tabs, setTabs] = useState<string[]>([]);
  const [active, setActive] = useState("");
  const [error, setError] = useState("");
  const initialized = useRef(false);
  const output = useResource<ProcessOutput>(
    active ? `/processes/${active}` : null,
    750,
  );
  const follow = useFollow(output.data?.output);
  const processes = resource.data?.processes || [];
  const task = processes.find((p) => p.id === active);
  useEffect(() => {
    if (processes.length && !initialized.current) {
      initialized.current = true;
      setTabs([processes[0].id]);
      setActive(processes[0].id);
    }
  }, [resource.data]);
  function open(id: string) {
    setTabs((values) => (values.includes(id) ? values : [...values, id]));
    setActive(id);
  }
  function close(id: string) {
    const next = tabs.filter((t) => t !== id);
    setTabs(next);
    if (active === id) setActive(next.at(-1) || "");
  }
  async function stop() {
    try {
      await api(`/processes/${active}/stop`, {});
      resource.reload();
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function download() {
    const url = URL.createObjectURL(
      new Blob([output.data?.output || ""], { type: "text/plain" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${active}.log`;
    anchor.click();
    URL.revokeObjectURL(url);
  }
  return (
    <div className="page-scroll processes-page">
      <PageTitle
        eyebrow="Workspace processes"
        title="Keep an eye on the long runs."
        actions={
          <Badge
            status={
              processes.some((p) => p.status === "running") ? "running" : "idle"
            }
          >
            {processes.filter((p) => p.status === "running").length} running
          </Badge>
        }
      >
        Tests, servers and experiments. Open a process to follow its output as
        it happens.
      </PageTitle>
      <ErrorNotice error={resource.error || error} retry={resource.reload} />
      {!processes.length ? (
        <Empty
          icon={<Terminal size={28} />}
          title="No background processes yet"
        >
          Processes started by the agent will appear here. They belong to this
          server instance and stay active when you close a browser tab.
        </Empty>
      ) : (
        <div className="process-layout">
          <div className="process-list">
            {processes.map((process) => (
              <button
                className={`process-card ${active === process.id ? "selected" : ""}`}
                key={process.id}
                onClick={() => open(process.id)}
              >
                <div>
                  <Terminal size={16} />
                  <Badge status={process.status} />
                </div>
                <h3>{process.description || process.id}</h3>
                <code>{process.command}</code>
                <footer>
                  <span>{process.id}</span>
                  <span>
                    {Math.floor(process.runtime_seconds / 60)}m{" "}
                    {Math.floor(process.runtime_seconds % 60)}s
                  </span>
                </footer>
              </button>
            ))}
          </div>
          <div className="terminal-panel">
            <Tabs.Root value={active} onValueChange={setActive}>
              <Tabs.List
                className="terminal-tabs"
                aria-label="Process output tabs"
              >
                {tabs.map((id) => (
                  <div className="terminal-tab" key={id}>
                    <Tabs.Trigger value={id}>
                      <Terminal size={13} />
                      {id}
                    </Tabs.Trigger>
                    <button
                      aria-label={`Close ${id} output`}
                      onClick={() => close(id)}
                    >
                      <X size={12} />
                    </button>
                  </div>
                ))}
              </Tabs.List>
            </Tabs.Root>
            {active ? (
              <>
                <div className="terminal-toolbar">
                  <div>
                    <span className="live-dot" />
                    {task?.status === "running"
                      ? "Live output · updates every 750 ms"
                      : `Process ${output.data?.status || "output"}`}
                  </div>
                  <div>
                    <Tip label="Copy output">
                      <button
                        className="icon-button"
                        aria-label="Copy process output"
                        onClick={() =>
                          void navigator.clipboard.writeText(
                            output.data?.output || "",
                          )
                        }
                      >
                        <Copy size={14} />
                      </button>
                    </Tip>
                    <Tip label="Download log">
                      <button
                        className="icon-button"
                        aria-label="Download process log"
                        onClick={download}
                      >
                        <Download size={14} />
                      </button>
                    </Tip>
                    {task?.status === "running" && (
                      <button
                        className="button danger small"
                        onClick={() => void stop()}
                      >
                        <Square size={12} />
                        Stop
                      </button>
                    )}
                  </div>
                </div>
                <ErrorNotice error={output.error} />
                {output.data &&
                  (output.data.discarded > 0 || output.data.truncated) && (
                    <div className="terminal-truncated">
                      Showing the retained tail of the output; earlier output
                      was truncated.
                    </div>
                  )}
                <div
                  className="terminal-output"
                  ref={follow.ref}
                  onScroll={follow.onScroll}
                >
                  <div className="terminal-command">
                    <span>$</span> {task?.command}
                  </div>
                  <pre>{output.data?.output || "Waiting for output…"}</pre>
                </div>
                <div className="terminal-footer">
                  <span>{task?.cwd}</span>
                  <button
                    onClick={() => {
                      if (follow.ref.current)
                        follow.ref.current.scrollTop =
                          follow.ref.current.scrollHeight;
                    }}
                  >
                    <Play size={10} />
                    Jump to latest
                  </button>
                </div>
              </>
            ) : (
              <Empty icon={<Terminal size={24} />} title="Open a process">
                Select a process to inspect its output.
              </Empty>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
