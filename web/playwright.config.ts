import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  use: {
    baseURL: "http://127.0.0.1:8765",
    viewport: { width: 1440, height: 1000 },
    trace: "retain-on-failure",
  },
  webServer: {
    command: "uv run --locked --project .. --extra web python e2e/server.py",
    url: "http://127.0.0.1:8765/",
    reuseExistingServer: false,
    timeout: 60000,
  },
});
