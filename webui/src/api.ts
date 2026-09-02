import type { Theme } from "./types";

const DEFAULT_API_BASE = "http://127.0.0.1:8765";

export const EVENT_LIMIT = 2000;
export const EVAL_METRIC_LIMIT = 2000;
export const METRIC_LIMIT = 200;
export const POLL_MS = 2000;
export const ROLLOUT_INSPECT_POLL_MS = 5000;

export function initialApiBase(): string {
  const params = new URLSearchParams(window.location.search);
  const query = params.get("api");
  if (query) {
    return normalizeApiBase(query);
  }
  const stored = window.localStorage.getItem("wavelet.apiBase");
  if (stored) {
    return normalizeApiBase(stored);
  }
  const envBase = import.meta.env.VITE_WAVELET_API_BASE as string | undefined;
  return normalizeApiBase(envBase || DEFAULT_API_BASE);
}

export function initialTheme(): Theme {
  const stored = window.localStorage.getItem("wavelet.theme");
  return stored === "light" ? "light" : "dark";
}

export function normalizeApiBase(value: string): string {
  return value.trim().replace(/\/$/, "");
}

export async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal, cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}
