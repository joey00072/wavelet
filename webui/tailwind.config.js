/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        surface: "var(--surface-1)",
        plane: "var(--plane)",
        raised: "var(--surface-2)",
        ink: "var(--ink-1)",
        ink2: "var(--ink-2)",
        muted: "var(--ink-muted)",
        hair: "var(--grid)",
        edge: "var(--border)",
        accent: "var(--series-1)",
        good: "var(--status-good)",
        warn: "var(--status-warning)",
        serious: "var(--status-serious)",
        critical: "var(--status-critical)",
      },
    },
  },
  plugins: [],
};
