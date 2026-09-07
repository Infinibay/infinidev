import { useEffect, useState } from "react";
import { Tabs } from "radix-ui";
import {
  ChevronRight,
  File,
  FileCode,
  Folder,
  GitBranch,
  GitCompareArrows,
  RotateCw,
} from "lucide-react";
import { api } from "./api";
import {
  Empty,
  ErrorNotice,
  Loading,
  PageTitle,
  SaveButton,
} from "./components";
import { useResource } from "./hooks";
import type { FileData, FileEntry } from "./types";

export function FilesView({ onDirty }: { onDirty: (dirty: boolean) => void }) {
  const changes = useResource<{
    files: { path: string; status: string }[];
    branch: string;
  }>("/changes", 5000);
  const [directory, setDirectory] = useState("");
  const tree = useResource<{ path: string; entries: FileEntry[] }>(
    `/files/tree?path=${encodeURIComponent(directory)}`,
  );
  const [path, setPath] = useState("");
  const source = useResource<FileData>(
    path ? `/files/read?path=${encodeURIComponent(path)}` : null,
  );
  const diff = useResource<{ staged: string; unstaged: string }>(
    path ? `/changes/diff?path=${encodeURIComponent(path)}` : null,
  );
  const [mode, setMode] = useState("diff");
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    onDirty(Boolean(source.data && text !== source.data.text));
  }, [text, source.data, onDirty]);
  useEffect(() => () => onDirty(false), [onDirty]);
  useEffect(() => {
    if (source.data) {
      setText(source.data.text);
      setSaved(false);
    }
  }, [source.data]);
  function select(next: string) {
    if (
      source.data &&
      text !== source.data.text &&
      !window.confirm("Discard your unsaved edits?")
    )
      return;
    setPath(next);
    setError("");
  }
  async function save() {
    setSaving(true);
    setError("");
    try {
      const updated = await api<FileData>(
        "/files/write",
        { path, text, revision: source.data?.revision },
        "PUT",
      );
      source.setData(updated);
      setSaved(true);
      changes.reload();
      diff.reload();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSaving(false);
    }
  }
  function reloadFile() {
    if (
      source.data &&
      text !== source.data.text &&
      !window.confirm("Discard your edits and reload the file from disk?")
    )
      return;
    setError("");
    source.reload();
    diff.reload();
  }
  useEffect(() => {
    const before = (event: BeforeUnloadEvent) => {
      if (source.data && text !== source.data.text) event.preventDefault();
    };
    window.addEventListener("beforeunload", before);
    return () => window.removeEventListener("beforeunload", before);
  }, [text, source.data]);
  const patches = [
    diff.data?.staged ? `# Staged changes\n${diff.data.staged}` : "",
    diff.data?.unstaged ? `# Working changes\n${diff.data.unstaged}` : "",
  ]
    .filter(Boolean)
    .join("\n");
  return (
    <div className="page-scroll files-page">
      <PageTitle
        eyebrow="Inspect the work"
        title="From idea to actual changes."
        actions={
          <button
            className="button"
            onClick={() => {
              changes.reload();
              tree.reload();
              diff.reload();
            }}
          >
            <RotateCw size={14} />
            Refresh
          </button>
        }
      >
        Review diffs, explore the project and make deliberate edits.
      </PageTitle>
      <div className="files-layout">
        <aside className="file-browser">
          <Tabs.Root defaultValue="changes">
            <Tabs.List className="tabs-list" aria-label="File navigation">
              <Tabs.Trigger value="changes">
                Changes <span>{changes.data?.files.length || 0}</span>
              </Tabs.Trigger>
              <Tabs.Trigger value="files">Files</Tabs.Trigger>
            </Tabs.List>
            <Tabs.Content value="changes">
              <div className="branch-label">
                <GitBranch size={14} />
                {changes.data?.branch || "Working tree"}
              </div>
              <ErrorNotice error={changes.error} />
              {changes.data?.files.map((file) => (
                <button
                  key={file.path}
                  className={`file-row ${path === file.path ? "selected" : ""}`}
                  onClick={() => {
                    select(file.path);
                    setMode("diff");
                  }}
                >
                  <FileCode size={15} />
                  <span>{file.path}</span>
                  <code>{file.status.trim()}</code>
                </button>
              ))}
              {changes.data && !changes.data.files.length && (
                <p className="column-empty">Working tree is clean.</p>
              )}
            </Tabs.Content>
            <Tabs.Content value="files">
              <div className="directory-bar">
                <button
                  onClick={() =>
                    setDirectory(directory.split("/").slice(0, -1).join("/"))
                  }
                  disabled={!directory}
                >
                  ↑
                </button>
                <span>{directory || "/ workspace"}</span>
              </div>
              <ErrorNotice error={tree.error} />
              {tree.data?.entries.map((file) => (
                <button
                  className={`file-row ${path === file.path ? "selected" : ""}`}
                  key={file.path}
                  onClick={() => {
                    if (file.is_dir) setDirectory(file.path);
                    else {
                      select(file.path);
                      setMode("source");
                    }
                  }}
                >
                  {file.is_dir ? <Folder size={15} /> : <File size={15} />}
                  <span>{file.name}</span>
                  {file.is_dir && <ChevronRight size={12} />}
                </button>
              ))}
            </Tabs.Content>
          </Tabs.Root>
        </aside>
        <div className="file-preview">
          {path ? (
            <>
              <div className="file-preview-header">
                <span>
                  <FileCode size={15} />
                  {path}
                </span>
                <button
                  className="icon-button"
                  aria-label="Reload file from disk"
                  onClick={reloadFile}
                >
                  <RotateCw size={14} />
                </button>
                <Tabs.Root value={mode} onValueChange={setMode}>
                  <Tabs.List className="segmented" aria-label="File display">
                    <Tabs.Trigger value="diff">Diff</Tabs.Trigger>
                    <Tabs.Trigger value="source">Source</Tabs.Trigger>
                  </Tabs.List>
                </Tabs.Root>
              </div>
              <ErrorNotice
                error={error || (mode === "source" ? source.error : diff.error)}
              />
              {mode === "diff" ? (
                <div className="diff-view">
                  {diff.loading ? (
                    <Loading />
                  ) : patches ? (
                    <pre>
                      {patches.split("\n").map((line, index) => (
                        <div
                          key={index}
                          className={
                            line.startsWith("+")
                              ? "diff-added"
                              : line.startsWith("-")
                                ? "diff-removed"
                                : line.startsWith("@@")
                                  ? "diff-hunk"
                                  : ""
                          }
                        >
                          {line || " "}
                        </div>
                      ))}
                    </pre>
                  ) : (
                    <Empty
                      icon={<GitCompareArrows size={24} />}
                      title="No tracked diff for this file"
                    >
                      Untracked files can be inspected in Source.
                    </Empty>
                  )}
                </div>
              ) : source.loading ? (
                <Loading />
              ) : source.data?.binary ? (
                <Empty icon={<File size={24} />} title="Binary file">
                  This file cannot be edited as text.
                </Empty>
              ) : (
                <>
                  <textarea
                    className="source-editor"
                    aria-label={`Edit ${path}`}
                    spellCheck={false}
                    value={text}
                    onChange={(e) => {
                      setText(e.target.value);
                      setSaved(false);
                    }}
                  />
                  <div className="editor-footer">
                    <span>
                      {source.data && text !== source.data.text
                        ? "Unsaved changes"
                        : "Matches the file on disk"}{" "}
                      · UTF-8
                    </span>
                    <SaveButton
                      busy={saving}
                      saved={saved}
                      disabled={!source.data || text === source.data.text}
                      onClick={() => void save()}
                    />
                  </div>
                </>
              )}
            </>
          ) : (
            <Empty
              icon={<FileCode size={28} />}
              title="Look closer at the work"
            >
              Choose a changed file or explore the workspace.
            </Empty>
          )}
        </div>
      </div>
    </div>
  );
}
