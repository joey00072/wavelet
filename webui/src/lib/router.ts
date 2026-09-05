import { useEffect, useState } from "react";

type Route = {
  page: "runs" | "run" | "compare";
  runId: string | null;
  view: string;
  params: URLSearchParams;
};

const RUN_VIEWS = [
  "overview",
  "training",
  "rollouts",
  "inspector",
  "evals",
  "pipeline",
  "infra",
  "config",
] as const;
export type RunView = (typeof RUN_VIEWS)[number];

function parseRoute(hash: string): Route {
  const cleaned = hash.replace(/^#\/?/, "");
  const [pathPart, queryPart = ""] = cleaned.split("?");
  const params = new URLSearchParams(queryPart);
  const segments = pathPart.split("/").filter(Boolean);
  if (segments[0] === "run" && segments[1]) {
    const view = segments[2] && (RUN_VIEWS as readonly string[]).includes(segments[2]) ? segments[2] : "overview";
    return { page: "run", runId: decodeURIComponent(segments[1]), view, params };
  }
  if (segments[0] === "compare") {
    return { page: "compare", runId: null, view: "compare", params };
  }
  if (segments[0] === "runs") {
    return { page: "runs", runId: null, view: "runs", params };
  }
  // Default landing: the run that is active right now, resolved server-side.
  return { page: "run", runId: CURRENT_RUN, view: "overview", params };
}

/** Alias the API resolves to the live (or most recent) run. */
export const CURRENT_RUN = "current";

export function routeHref(route: Partial<Route> & { page: Route["page"] }): string {
  const query = route.params && [...route.params.keys()].length > 0 ? `?${route.params.toString()}` : "";
  if (route.page === "run" && route.runId) {
    return `#/run/${encodeURIComponent(route.runId)}/${route.view ?? "overview"}${query}`;
  }
  if (route.page === "compare") return `#/compare${query}`;
  return `#/runs${query}`;
}

export function navigate(route: Partial<Route> & { page: Route["page"] }): void {
  window.location.hash = routeHref(route);
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function updateParams(mutate: (params: URLSearchParams) => void): void {
  const route = parseRoute(window.location.hash);
  const params = new URLSearchParams(route.params);
  mutate(params);
  window.history.replaceState(null, "", routeHref({ ...route, params }));
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}
