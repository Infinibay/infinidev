import path from "node:path";

// Harbor UI is a pinned git submodule (see vite.config.ts). Tailwind scans its
// classes for the JIT, so this must track the submodule.
const harborSrc = path.resolve(process.cwd(), "external/infinibay_ui/src");

/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    // Harbor components live outside our project root; include their sources
    // so Tailwind picks up the utility classes they use.
    `${harborSrc}/**/*.{js,ts,jsx,tsx}`,
  ],
  theme: {
    extend: {
      colors: {
        accent: "rgb(var(--harbor-accent) / <alpha-value>)",
        "accent-2": "rgb(var(--harbor-accent-2) / <alpha-value>)",
        "accent-3": "rgb(var(--harbor-accent-3) / <alpha-value>)",
        success: "rgb(var(--harbor-success) / <alpha-value>)",
        warning: "rgb(var(--harbor-warning) / <alpha-value>)",
        danger: "rgb(var(--harbor-danger) / <alpha-value>)",
        info: "rgb(var(--harbor-info) / <alpha-value>)",
        surface: {
          DEFAULT: "rgb(var(--harbor-bg) / <alpha-value>)",
          1: "rgb(var(--harbor-bg-elev-1) / <alpha-value>)",
          2: "rgb(var(--harbor-bg-elev-2) / <alpha-value>)",
          3: "rgb(var(--harbor-bg-elev-3) / <alpha-value>)",
        },
        fg: {
          DEFAULT: "rgb(var(--harbor-text) / <alpha-value>)",
          muted: "rgb(var(--harbor-text-muted) / <alpha-value>)",
          subtle: "rgb(var(--harbor-text-subtle) / <alpha-value>)",
        },
      },
      borderRadius: {
        sm: "var(--harbor-radius-sm)",
        md: "var(--harbor-radius-md)",
        lg: "var(--harbor-radius-lg)",
        xl: "var(--harbor-radius-xl)",
        "2xl": "var(--harbor-radius-2xl)",
      },
      boxShadow: {
        "harbor-sm": "var(--harbor-shadow-sm)",
        "harbor-md": "var(--harbor-shadow-md)",
        "harbor-lg": "var(--harbor-shadow-lg)",
        "harbor-glow": "var(--harbor-shadow-glow)",
      },
      transitionDuration: {
        instant: "var(--harbor-dur-instant)",
        fast: "var(--harbor-dur-fast)",
        base: "var(--harbor-dur-base)",
        slow: "var(--harbor-dur-slow)",
        slower: "var(--harbor-dur-slower)",
      },
      transitionTimingFunction: {
        out: "var(--harbor-ease-out)",
        "in-out": "var(--harbor-ease-in-out)",
        spring: "var(--harbor-ease-spring)",
      },
      fontFamily: {
        sans: ["InterVariable", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
    },
  },
  plugins: [],
};
