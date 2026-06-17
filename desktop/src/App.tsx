import { CursorProvider } from "@harbor/lib/cursor";
import { ToastProvider } from "@harbor/components/feedback/Toast";
import { AppProvider } from "@/state/store";
import { AppShell } from "@/ui/AppShell";
import { AppErrorBoundary } from "@/ui/ErrorBoundary";

export function App() {
  return (
    <CursorProvider>
      <ToastProvider>
        <AppProvider>
          <AppErrorBoundary>
            <AppShell />
          </AppErrorBoundary>
        </AppProvider>
      </ToastProvider>
    </CursorProvider>
  );
}
