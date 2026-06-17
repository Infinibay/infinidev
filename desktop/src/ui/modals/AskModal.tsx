import { useEffect, useState } from "react";
import { Dialog } from "@harbor/components/overlays/Dialog";
import { Button } from "@harbor/components/buttons/Button";
import { TextField } from "@harbor/components/inputs/TextField";
import { useEngine } from "@/state/store";

/** Interactive host prompt the engine blocks on — a command-confirmation gate
 *  ("permission") or a free-text question ("input", e.g. an interactive stdin
 *  prompt). Closing/Denying answers negatively so the engine never hangs. */
export function AskModal() {
  const { state, answerPrompt } = useEngine();
  const ask = state.ask;
  const [value, setValue] = useState("");

  useEffect(() => {
    setValue("");
  }, [ask?.id]);

  if (!ask) return null;
  const isPermission = ask.kind === "permission";
  // The first line reads as a title; the rest (e.g. the command) is the detail.
  const [title, ...rest] = ask.prompt.split("\n");
  const detail = rest.join("\n").trim();

  return (
    <Dialog
      open
      onClose={() => answerPrompt(isPermission ? "deny" : "")}
      size="sm"
      title={isPermission ? "Confirm action" : "Infinidev needs input"}
      footer={
        isPermission ? (
          <div className="ml-auto flex gap-2">
            <Button variant="ghost" onClick={() => answerPrompt("deny")}>
              Deny
            </Button>
            <Button variant="primary" onClick={() => answerPrompt("allow")}>
              Allow
            </Button>
          </div>
        ) : (
          <div className="ml-auto flex gap-2">
            <Button variant="ghost" onClick={() => answerPrompt("")}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => answerPrompt(value)}>
              Send
            </Button>
          </div>
        )
      }
    >
      <div className="space-y-3 text-sm">
        <p className="text-fg">{title}</p>
        {detail && (
          <pre className="mono max-h-48 overflow-auto whitespace-pre-wrap rounded-md border border-fg/10 bg-surface-2/50 p-2.5 text-[12px] text-fg-muted">
            {detail}
          </pre>
        )}
        {!isPermission && (
          <TextField
            autoFocus
            placeholder="Type your answer…"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") answerPrompt(value);
            }}
          />
        )}
      </div>
    </Dialog>
  );
}
