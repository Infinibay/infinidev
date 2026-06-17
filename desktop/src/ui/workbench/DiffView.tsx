/** Renders a unified diff with per-line coloring (added / removed / hunk /
 *  file headers), mirroring the TUI's file_diff control. Line numbers track the
 *  old and new sides as the hunks advance. */
export function DiffView({ diff }: { diff: string }) {
  const lines = diff.replace(/\n$/, "").split("\n");
  if (!diff.trim()) {
    return <div className="p-4 text-xs text-fg-subtle">No changes vs HEAD.</div>;
  }

  let oldNo = 0;
  let newNo = 0;
  return (
    <div className="mono min-h-full overflow-auto bg-surface p-0 text-[11px] leading-relaxed">
      {lines.map((line, i) => {
        let cls = "text-fg-muted";
        let oldLabel = "";
        let newLabel = "";

        if (line.startsWith("@@")) {
          // @@ -a,b +c,d @@ — reset both counters to the hunk start.
          const m = /@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
          if (m) {
            oldNo = parseInt(m[1], 10);
            newNo = parseInt(m[2], 10);
          }
          cls = "bg-accent/10 text-accent";
        } else if (line.startsWith("+++") || line.startsWith("---") || line.startsWith("diff ") || line.startsWith("index ")) {
          cls = "text-fg-subtle";
        } else if (line.startsWith("+")) {
          cls = "bg-success/10 text-success";
          newLabel = String(newNo++);
        } else if (line.startsWith("-")) {
          cls = "bg-danger/10 text-danger";
          oldLabel = String(oldNo++);
        } else {
          oldLabel = String(oldNo++);
          newLabel = String(newNo++);
        }

        return (
          <div key={i} className={`flex ${cls}`}>
            <span className="w-10 shrink-0 select-none px-1 text-right text-fg-subtle/60">{oldLabel}</span>
            <span className="w-10 shrink-0 select-none px-1 text-right text-fg-subtle/60">{newLabel}</span>
            <span className="whitespace-pre px-2">{line || " "}</span>
          </div>
        );
      })}
    </div>
  );
}
