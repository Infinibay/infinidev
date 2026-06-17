import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Must mirror vite.config.ts aliases or tests can't resolve @harbor/* on a
// fresh checkout.
const harborSrc = path.resolve(__dirname, "external/infinibay_ui/src");

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@harbor": harborSrc,
    },
    dedupe: ["react", "react-dom", "framer-motion"],
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    globals: false,
  },
});
