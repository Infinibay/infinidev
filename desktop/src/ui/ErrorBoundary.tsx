import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}
interface State {
  error: Error | null;
}

/**
 * Catches render crashes so a desktop binary (no URL bar to reload) shows a
 * recovery UI instead of a white window.
 */
export class AppErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // eslint-disable-next-line no-console
    console.error("Render error:", error, info.componentStack);
  }

  render(): ReactNode {
    if (this.state.error) {
      return (
        <div className="fixed inset-0 flex flex-col items-center justify-center gap-4 bg-surface p-8 text-center text-fg">
          <h1 className="text-lg font-semibold text-danger">Something broke</h1>
          <pre className="max-w-2xl overflow-auto rounded-md bg-surface-2 p-4 text-left text-xs text-fg-muted">
            {this.state.error.message}
          </pre>
          <button
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white"
            onClick={() => this.setState({ error: null })}
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
