import { useEffect, useState } from "react";

export type Theme = "dark" | "light";
const KEY = "wavelet.theme";

export function initialTheme(): Theme {
  const stored = window.localStorage.getItem(KEY);
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function useTheme(): [Theme, (theme: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => {
    window.localStorage.setItem(KEY, theme);
    const root = document.documentElement;
    root.classList.add("theme-transition");
    root.classList.toggle("dark", theme === "dark");
    const timer = window.setTimeout(() => root.classList.remove("theme-transition"), 400);
    return () => window.clearTimeout(timer);
  }, [theme]);
  return [theme, setTheme];
}

const SERIES_VARS = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => `var(--series-${i})`);

export function seriesColor(index: number): string {
  // Fixed order, never cycled: past slot 8 series fold into the de-emphasis gray.
  return index < SERIES_VARS.length ? SERIES_VARS[index] : "var(--deemph)";
}
