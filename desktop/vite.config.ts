import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Harbor UI lives as a pinned git submodule under external/ so the repo is
// self-contained (a fresh clone --recursive + npm ci builds with no sibling
// checkout). Bump it with: git -C external/infinibay_ui checkout <tag>.
const harborSrc = path.resolve(__dirname, "external/infinibay_ui/src");

// Tauri expects a fixed dev-server port (see src-tauri/tauri.conf.json devUrl).
export default defineConfig({
  plugins: [react()],
  // Don't let Vite clear the terminal — Tauri shares it for the Rust logs.
  clearScreen: false,
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@harbor": harborSrc,
    },
    // Harbor reads React/framer context; deduping prevents "two Reacts" /
    // "two framer-motion" bugs when importing source from outside node_modules.
    dedupe: ["react", "react-dom", "framer-motion"],
  },
  server: {
    port: 5173,
    strictPort: true,
    fs: {
      // Vite blocks serving files outside the project root by default. Harbor's
      // source is under external/, so allow it (and the parent for submodules).
      allow: [path.resolve(__dirname, ".."), harborSrc],
    },
  },
});
