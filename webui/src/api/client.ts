import { useEffect, useRef, useState } from "react";

const STORAGE_KEY = "wavelet.apiBase";

/** Candidate API bases in priority order: explicit query, remembered, build default, same origin. */
function apiBaseCandidates(): string[] {
  const params = new URLSearchParams(window.location.search);
  const candidates: string[] = [];
  if (params.has("api")) {
    const query = params.get("api") ?? "";
    candidates.push(query === "" || query === "same" ? "" : normalizeApiBase(query));
  }
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored !== null) candidates.push(normalizeApiBase(stored));
  const envBase = import.meta.env.VITE_WAVELET_API_BASE as string | undefined;
  if (envBase) candidates.push(normalizeApiBase(envBase));
  if (window.location.port === "5173") {
    // Vite dev server: the standalone dashboard usually runs beside it on the same host.
    candidates.push(`${window.location.protocol}//${window.location.hostname}:8766`);
  }
  candidates.push("");
  return [...new Set(candidates)];
}

export async function probeApiBase(base: string, timeoutMs = 2500): Promise<boolean> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${base}/api/health`, { signal: controller.signal, cache: "no-store" });
    if (!response.ok) return false;
    // Dev servers answer unknown paths with index.html; insist on the JSON contract.
    const body = (await response.json()) as { ok?: boolean };
    return body?.ok === true;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

/** Resolve the first reachable API base; falls back to the first candidate so errors stay visible. */
export async function resolveApiBase(): Promise<string> {
  const candidates = apiBaseCandidates();
  for (const candidate of candidates) {
    if (await probeApiBase(candidate)) return candidate;
  }
  return candidates[0] ?? "";
}

export function persistApiBase(value: string): void {
  window.localStorage.setItem(STORAGE_KEY, value);
}

export function normalizeApiBase(value: string): string {
  return value.trim().replace(/\/$/, "");
}

export async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal, cache: "no-store" });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // keep status text
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

export type PollState<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  refetching: boolean;
  updatedAt: string | null;
};

type PollOptions = {
  /**
   * Stable identity for the resource behind a URL. Query-only changes with the
   * same key retain the current payload while the replacement is fetched.
   */
  resourceKey?: string | null;
};

/**
 * Polls a URL. Pass a stable `resourceKey` to retain the previous payload
 * across query-only URL changes, or pass `null` as the URL to disable.
 */
export function usePoll<T>(url: string | null, intervalMs: number, options: PollOptions = {}): PollState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(url !== null);
  const [refetching, setRefetching] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const lastUrl = useRef<string | null>(null);
  const lastResourceKey = useRef<string | null>(null);
  const resourceKey = options.resourceKey === undefined ? url : options.resourceKey;

  useEffect(() => {
    if (url === null) {
      setData(null);
      setLoading(false);
      setRefetching(false);
      setError(null);
      setUpdatedAt(null);
      lastUrl.current = null;
      lastResourceKey.current = null;
      return;
    }
    let cancelled = false;
    let timer: number | undefined;
    const controller = new AbortController();
    const changedUrl = lastUrl.current !== url;
    const changedResource = lastResourceKey.current !== resourceKey;
    lastUrl.current = url;
    lastResourceKey.current = resourceKey;
    if (changedUrl) setError(null);
    if (changedResource) {
      // A different run/source must never briefly display the previous
      // resource's payload while its first request is in flight.
      setData(null);
      setUpdatedAt(null);
      setLoading(true);
    }
    const run = async () => {
      setRefetching(true);
      try {
        const next = await fetchJson<T>(url, controller.signal);
        if (!cancelled) {
          setData(next);
          setError(null);
          setUpdatedAt(new Date().toISOString());
        }
      } catch (caught) {
        if (!cancelled && !(caught instanceof DOMException && caught.name === "AbortError")) {
          setError(caught instanceof Error ? caught.message : String(caught));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
          setRefetching(false);
          if (intervalMs > 0) timer = window.setTimeout(run, intervalMs);
        }
      }
    };
    run();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, intervalMs, resourceKey]);

  return { data, error, loading, refetching, updatedAt };
}

export function runUrl(apiBase: string, runId: string, path: string): string {
  return `${apiBase}/api/runs/${encodeURIComponent(runId)}${path}`;
}
