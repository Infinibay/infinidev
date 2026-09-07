import { useState } from "react";
import {
  BookOpen,
  ChevronRight,
  CircleGauge,
  Plug,
  RefreshCw,
  Search,
  Shield,
  Wrench,
} from "lucide-react";
import {
  Badge,
  Empty,
  ErrorNotice,
  Loading,
  Markdown,
  Modal,
  PageTitle,
} from "./components";
import { useResource } from "./hooks";

interface Finding {
  id: number;
  topic: string;
  content: string;
  finding_type: string;
  confidence: number;
  status: string;
  created_at: string;
}
export function KnowledgeView() {
  const resource = useResource<{ findings: Finding[] }>(
    "/findings?limit=1000",
    15000,
  );
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Finding | null>(null);
  const findings =
    resource.data?.findings.filter((f) =>
      `${f.topic} ${f.content} ${f.finding_type}`
        .toLowerCase()
        .includes(search.toLowerCase()),
    ) || [];
  return (
    <div className="page-scroll">
      <PageTitle
        eyebrow="Research memory"
        title="Good work builds on what’s known."
        actions={
          <button className="button" onClick={resource.reload}>
            <RefreshCw size={14} />
            Refresh
          </button>
        }
      >
        Browse the project’s saved findings, constraints and accumulated
        evidence.
      </PageTitle>
      <ErrorNotice error={resource.error} retry={resource.reload} />
      <div className="search-row">
        <label className="search-field">
          <Search size={16} />
          <input
            aria-label="Search findings"
            placeholder="Search topics, findings and evidence…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
        <span>{findings.length} findings</span>
      </div>
      {resource.loading ? (
        <Loading />
      ) : findings.length ? (
        <div className="findings-list">
          {findings.map((finding) => (
            <button
              className="finding-row"
              key={finding.id}
              onClick={() => setSelected(finding)}
            >
              <span className="finding-icon">
                <BookOpen size={17} />
              </span>
              <div>
                <div className="finding-meta">
                  <span>
                    {finding.finding_type?.replaceAll("_", " ") || "finding"}
                  </span>
                  <span>#{finding.id}</span>
                </div>
                <h3>{finding.topic}</h3>
                <p>{finding.content}</p>
              </div>
              <div className="finding-aside">
                <Badge status={finding.status || "saved"} />
                <ChevronRight size={17} />
              </div>
            </button>
          ))}
        </div>
      ) : (
        <Empty
          icon={<BookOpen size={28} />}
          title={
            search ? "No matching findings" : "Knowledge grows as you work"
          }
        >
          {search
            ? "Try a different topic or keyword."
            : "Saved project findings will appear here. Ken’s external index remains available to agents through its tools."}
        </Empty>
      )}
      <Modal
        open={!!selected}
        onOpenChange={(open) => {
          if (!open) setSelected(null);
        }}
        title={selected?.topic || "Finding"}
        description={
          selected
            ? `Finding #${selected.id} · ${selected.finding_type} · ${selected.status}`
            : undefined
        }
      >
        {selected && <Markdown text={selected.content} />}
      </Modal>
    </div>
  );
}

export function ToolsView() {
  const resource = useResource<{
    tools: { name: string; description: string; read_only: boolean }[];
  }>("/tools");
  const [search, setSearch] = useState("");
  const rows =
    resource.data?.tools.filter((t) =>
      `${t.name} ${t.description}`.toLowerCase().includes(search.toLowerCase()),
    ) || [];
  return (
    <div className="page-scroll">
      <PageTitle
        eyebrow="Harness capabilities"
        title="The tools behind the work."
        actions={
          <button className="button" onClick={resource.reload}>
            <RefreshCw size={14} />
            Refresh discovery
          </button>
        }
      >
        Inspect the tool catalog, including discovered MCP tools. Each
        specialist gets the tools selected by the orchestrator.
      </PageTitle>
      <ErrorNotice error={resource.error} retry={resource.reload} />
      <label className="search-field">
        <Search size={16} />
        <input
          aria-label="Search tools"
          placeholder="Find a tool…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </label>
      {resource.loading ? (
        <Loading />
      ) : (
        <div className="tools-grid">
          {rows.map((tool) => (
            <article key={tool.name} className="tool-card">
              <div>
                <Wrench size={17} />
                <span>
                  {tool.read_only ? (
                    <>
                      <Shield size={12} />
                      Read only
                    </>
                  ) : (
                    <>
                      <Plug size={12} />
                      Tool
                    </>
                  )}
                </span>
              </div>
              <h3>{tool.name}</h3>
              <p>{tool.description}</p>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

export function UsageView() {
  const resource = useResource<{ report: string }>("/usage");
  return (
    <div className="page-scroll">
      <PageTitle
        eyebrow="Consumption & limits"
        title="Understand what your work uses."
        actions={
          <button
            className="button"
            disabled={resource.loading}
            onClick={resource.reload}
          >
            <RefreshCw size={14} className={resource.loading ? "spin" : ""} />
            Refresh usage
          </button>
        }
      >
        The selected provider’s available quota and billing information,
        alongside locally observed consumption.
      </PageTitle>
      <ErrorNotice error={resource.error} retry={resource.reload} />
      {resource.loading ? (
        <Loading />
      ) : resource.data ? (
        <article className="usage-report">
          <div>
            <CircleGauge size={22} />
            <h3>Provider usage report</h3>
            <Badge status="saved">Latest query</Badge>
          </div>
          <pre>{resource.data.report}</pre>
        </article>
      ) : (
        <Empty icon={<CircleGauge size={28} />} title="Usage is unavailable">
          Check the connection and retry.
        </Empty>
      )}
      <div className="information-note">
        <Shield size={17} />
        <p>
          Providers expose different information. Missing quota or billing data
          is shown as unavailable, without estimates presented as account
          limits. Administrative billing credentials can be configured through
          the existing Infinidev settings.
        </p>
      </div>
    </div>
  );
}
