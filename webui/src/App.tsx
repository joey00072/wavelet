import { useEffect, useMemo, useState, type ReactElement } from "react";
import { Activity, BarChart3, Cpu, FileJson, FlaskConical, GitCompare, History, Moon, Radio, Search, Sun, Waves, Workflow } from "lucide-react";

import { normalizeApiBase, persistApiBase, probeApiBase, resolveApiBase } from "./api/client";
import { StatusBadge } from "./components/Badge";
import { Popover } from "./components/Controls";
import { CURRENT_RUN, navigate, routeHref, useRoute, type RunView } from "./lib/router";
import { useTheme } from "./lib/theme";
import { CompareView } from "./views/CompareView";
import { ConfigView } from "./views/ConfigView";
import { EvalsView } from "./views/EvalsView";
import { InfraView } from "./views/InfraView";
import { InspectorView } from "./views/InspectorView";
import { OverviewView } from "./views/OverviewView";
import { PipelineView } from "./views/PipelineView";
import { RunsView, useRuns } from "./views/RunsView";
import { RolloutMetricsView, TrainingView } from "./views/TrainingView";
import { useSummary } from "./views/useRunData";

const NAV: Array<{ view: RunView; label: string; icon: ReactElement; hint: string }> = [
  { view: "overview", label: "Overview", icon: <Activity className="h-3.5 w-3.5" />, hint: "health, reward, eval, key trainer signals" },
  { view: "training", label: "Trainer", icon: <BarChart3 className="h-3.5 w-3.5" />, hint: "every trainer metric" },
  { view: "rollouts", label: "Generation", icon: <Waves className="h-3.5 w-3.5" />, hint: "rollout generation metrics" },
  { view: "inspector", label: "Inspector", icon: <Search className="h-3.5 w-3.5" />, hint: "browse, sort, filter rollouts" },
  { view: "evals", label: "Evals", icon: <FlaskConical className="h-3.5 w-3.5" />, hint: "fixed-policy evaluation" },
  { view: "pipeline", label: "Pipeline", icon: <Workflow className="h-3.5 w-3.5" />, hint: "queue, policy, lifecycle" },
  { view: "infra", label: "Infra", icon: <Cpu className="h-3.5 w-3.5" />, hint: "GPU, disk, inference load, logs" },
  { view: "config", label: "Config", icon: <FileJson className="h-3.5 w-3.5" />, hint: "resolved config and diff" },
];

export function App() {
  const [apiBase, setApiBase] = useState<string | null>(null);
  const [apiInput, setApiInput] = useState("");
  const [theme, setTheme] = useTheme();
  const route = useRoute();
  const runs = useRuns(apiBase ?? "");
  const runId = route.runId;
  const summary = useSummary(apiBase ?? "", apiBase === null ? null : runId, 3000);
  const current = runId ? summary.data : null;
  const runIsLive = current?.status === "running";
  const runIds = useMemo(() => (runs.data ?? []).map((r) => r.id), [runs.data]);
  const currentRun = useMemo(() => (runs.data ?? []).find((r) => r.is_current) ?? null, [runs.data]);
  const viewingCurrent = runId === CURRENT_RUN || (currentRun !== null && runId === currentRun.id);
  const primaryRunLabel = currentRun?.status === "running" ? "Current" : "Recent";

  useEffect(() => {
    let cancelled = false;
    resolveApiBase().then((base) => {
      if (cancelled) return;
      setApiBase(base);
      setApiInput(base);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (apiBase !== null) persistApiBase(apiBase);
  }, [apiBase]);

  // Self-heal: if the chosen API stops answering but another candidate works, move over.
  useEffect(() => {
    if (!runs.error || apiBase === null) return;
    let cancelled = false;
    (async () => {
      for (const candidate of apiBase === "" ? [] : [""]) {
        if (await probeApiBase(candidate)) {
          if (!cancelled) {
            setApiBase(candidate);
            setApiInput(candidate);
          }
          return;
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runs.error, apiBase]);

  useEffect(() => {
    const shown = current?.id ?? runId;
    document.title = shown ? `${shown} · Wavelet` : "Wavelet Dashboard";
  }, [runId, current?.id]);

  const applyApi = () => setApiBase(normalizeApiBase(apiInput));
  const useSameOrigin = () => {
    setApiBase("");
    setApiInput("");
    const url = new URL(window.location.href);
    url.searchParams.delete("api");
    window.history.replaceState(null, "", url.toString());
  };
  const showEvals = current ? current.eval_envs.length > 0 || Boolean(current.latest.eval) : true;
  const navItems = NAV.filter((item) => item.view !== "evals" || showEvals);
  const toggleTheme = () => setTheme(theme === "dark" ? "light" : "dark");
  const themeIcon = theme === "dark" ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />;
  const apiPanel = (note: ReactElement) => (
    <div className="space-y-2">
      <div className="eyebrow">API base</div>
      <div className="flex items-center gap-1">
        <input className="input min-w-0 flex-1" value={apiInput} onChange={(e) => setApiInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && applyApi()} placeholder="same origin" aria-label="API base URL" />
        <button type="button" className="btn btn-active" onClick={applyApi}>Go</button>
      </div>
      {note}
      {runs.error && <div className="text-[11px] text-critical">{runs.error}</div>}
    </div>
  );

  return (
    <div className="flex min-h-screen">
      <a className="skip-link" href="#main-content">Skip to dashboard content</a>
      <aside className="hidden w-56 shrink-0 flex-col lg:flex">
        <div className="sticky top-0 flex h-screen flex-col gap-6 px-4 py-5">
          <a href={routeHref({ page: "runs" })} className="flex items-center gap-2 px-1">
            <span className="flex h-7 w-7 items-center justify-center rounded-xl bg-ink text-surface"><Waves className="h-4 w-4" /></span>
            <span className="text-sm font-semibold tracking-tight">Wavelet</span>
          </a>
          <div>
            <div className="eyebrow mb-1 px-3">{primaryRunLabel} run</div>
            <a href={routeHref({ page: "run", runId: CURRENT_RUN, view: "overview" })} className={`block px-3 py-1 transition-colors ${viewingCurrent ? "" : "opacity-80 hover:opacity-100"}`}>
              <div className="flex items-center gap-1.5 text-[13px] font-semibold text-ink">
                <Radio className={`h-3.5 w-3.5 shrink-0 ${currentRun?.status === "running" ? "live-dot text-good" : "text-muted"}`} />
                <span className="line-clamp-2 break-all leading-tight" title={currentRun?.id}>{currentRun?.id ?? (runs.data ? "no runs found" : "…")}</span>
              </div>
              {currentRun && (
                <div className="tabular mt-1 flex items-center justify-between text-[11px] text-muted">
                  <StatusBadge status={currentRun.status} reason={currentRun.status_reason} />
                  <span>step {currentRun.trainer_step === null ? "–" : currentRun.trainer_step + 1}{currentRun.target_step ? ` / ${currentRun.target_step}` : ""}</span>
                </div>
              )}
            </a>
          </div>
          {runId && (
            <div className="min-h-0 flex-1">
              {!viewingCurrent && current && (
                <div className="mb-3 px-3 text-[11px] text-ink2">
                  <div className="flex items-center gap-1.5 font-medium text-ink"><History className="h-3 w-3" /> older run</div>
                  <div className="truncate" title={current.id}>{current.id}</div>
                </div>
              )}
              <nav className="flex flex-col gap-0.5" aria-label="Run views">
                {navItems.map((item) => (
                  <NavLink key={item.view} active={route.view === item.view} href={routeHref({ page: "run", runId, view: item.view })} icon={item.icon} label={item.label} hint={item.hint} />
                ))}
              </nav>
            </div>
          )}
          <nav className="flex flex-col gap-0.5 pt-1" aria-label="Global">
            <NavLink active={route.page === "runs"} href={routeHref({ page: "runs" })} icon={<History className="h-3.5 w-3.5" />} label={`Runs${runs.data?.length ? ` (${runs.data.length})` : ""}`} />
            <NavLink active={route.page === "compare"} href={routeHref({ page: "compare", params: route.page === "compare" ? route.params : undefined })} icon={<GitCompare className="h-3.5 w-3.5" />} label="Compare" />
          </nav>
          <div className="mt-auto flex items-center justify-between">
            <Popover
              align="left"
              placement="top"
              width={300}
              trigger={(open) => (
                <button type="button" className={`btn !px-2 ${open ? "btn-active" : ""}`} title="Connection">
                  {runs.error ? <span className="inline-block h-2 w-2 rounded-full bg-critical" /> : <span className="live-dot inline-block h-2 w-2 rounded-full bg-good" />}
                  <span className="text-[11px]">{runs.error ? "offline" : `${runs.data?.length ?? 0} runs`}</span>
                </button>
              )}
            >
              {apiPanel(<p className="text-[10.5px] leading-relaxed text-muted">{apiBase ? `Reading from ${apiBase}.` : "Reading from the server that served this page."} Leave blank to use this server; the address is remembered by the browser.</p>)}
            </Popover>
            <button type="button" className="btn !px-1.5 !py-1" onClick={toggleTheme} aria-label="Toggle theme">{themeIcon}</button>
          </div>
        </div>
      </aside>

      <main id="main-content" className="min-w-0 flex-1 px-4 py-4 sm:px-5 lg:px-10 lg:py-6">
        <div className="sticky top-0 z-30 -mx-4 -mt-4 mb-5 border-b border-edge bg-surface/95 px-4 py-2.5 backdrop-blur lg:hidden">
          <div className="flex items-center gap-1">
            <a href={routeHref({ page: "runs" })} className="mr-auto flex items-center gap-2 py-1 text-sm font-semibold" aria-label="Wavelet runs">
              <span className="flex h-7 w-7 items-center justify-center rounded-xl bg-ink text-surface"><Waves className="h-4 w-4" /></span>
              <span className="hidden min-[360px]:inline">Wavelet</span>
            </a>
            <a href={routeHref({ page: "run", runId: CURRENT_RUN, view: "overview" })} className="btn" aria-label={`${primaryRunLabel} run`}><Radio className="h-3.5 w-3.5" /><span className="hidden min-[360px]:inline">{primaryRunLabel}</span></a>
            <a href={routeHref({ page: "runs" })} className="btn" aria-label="All runs"><History className="h-3.5 w-3.5" /><span className="hidden min-[360px]:inline">Runs</span></a>
            <a href={routeHref({ page: "compare", params: route.page === "compare" ? route.params : undefined })} className="btn !px-2" aria-label="Compare runs" title="Compare runs"><GitCompare className="h-3.5 w-3.5" /></a>
            <button type="button" className="btn !px-2" onClick={toggleTheme} aria-label={`Use ${theme === "dark" ? "light" : "dark"} theme`}>{themeIcon}</button>
          </div>
          {runId && (
            <div className="mt-2 flex items-center gap-2">
              <select className="select min-w-0 flex-1" aria-label="Run view" value={route.view} onChange={(e) => navigate({ page: "run", runId, view: e.target.value })}>
                {navItems.map((item) => <option key={item.view} value={item.view}>{item.label}</option>)}
              </select>
              <Popover
                width={300}
                trigger={(open) => (
                  <button type="button" className={`btn !px-2 ${open ? "btn-active" : ""}`} aria-label="API connection" title="API connection">
                    <span className={`inline-block h-2 w-2 rounded-full ${runs.error ? "bg-critical" : "bg-good"}`} />
                  </button>
                )}
              >
                {apiPanel(<p className="text-[11px] leading-relaxed text-muted">{apiBase ? `Reading from ${apiBase}.` : "Reading from this server."}</p>)}
              </Popover>
            </div>
          )}
        </div>
        {apiBase === null && <div className="animate-enter text-xs text-muted">Connecting…</div>}
        <div key={`${route.page}:${runId ?? ""}:${route.view}`} className="animate-enter" data-page={route.page} data-view={route.view}>
        {apiBase !== null && runs.error && (
          <div className="mb-6 max-w-2xl space-y-2 text-sm">
            <div className="font-semibold text-critical">Cannot reach the API at {apiBase || window.location.origin}</div>
            <div className="text-xs text-muted">{runs.error}. Start <code>wavelet dashboard</code> on the training host, or point the sidebar field at a running state server.</div>
            {apiBase !== "" && <button type="button" className="btn btn-active" onClick={useSameOrigin}>Use this server's API ({window.location.origin})</button>}
          </div>
        )}
        {apiBase !== null && route.page === "runs" && <RunsView apiBase={apiBase} runs={runs.data} error={runs.error} />}
        {apiBase !== null && route.page === "compare" && <CompareView apiBase={apiBase} runs={runs.data ?? []} params={route.params} />}
        {apiBase !== null && route.page === "run" && runId && (
          <>
            {summary.error && (
              <div className="mb-3 rounded-md px-4 py-2.5 text-xs text-critical" style={{ background: "color-mix(in srgb, var(--status-critical) 12%, transparent)" }}>
                {runId === CURRENT_RUN && summary.error.startsWith("Unknown run") ? "No run directories found under the configured roots." : summary.error}{" "}
                <a className="underline" href={routeHref({ page: "runs" })}>All runs</a>
              </div>
            )}
            {!viewingCurrent && current && currentRun && (
              <div className="mb-3 flex flex-wrap items-center gap-2 rounded-md px-4 py-2.5 text-xs text-ink2" style={{ background: "color-mix(in srgb, var(--status-warning) 14%, transparent)" }}>
                <History className="h-3.5 w-3.5 text-warn" />
                Viewing older run <span className="font-medium text-ink">{current.id}</span>.
                <a className="underline" href={routeHref({ page: "run", runId: CURRENT_RUN, view: route.view })}>Go to {primaryRunLabel.toLowerCase()} run {currentRun.id}</a>
              </div>
            )}
            {route.view === "overview" && <OverviewView apiBase={apiBase} runId={runId} summary={current} />}
            {route.view === "training" && <TrainingView apiBase={apiBase} runId={runId} params={route.params} live={runIsLive} />}
            {route.view === "rollouts" && <RolloutMetricsView apiBase={apiBase} runId={runId} params={route.params} live={runIsLive} />}
            {route.view === "inspector" && <InspectorView apiBase={apiBase} runId={runId} params={route.params} live={runIsLive} />}
            {route.view === "evals" && <EvalsView apiBase={apiBase} runId={runId} params={route.params} trainerStep={current?.trainer_step ?? null} live={runIsLive} />}
            {route.view === "pipeline" && <PipelineView apiBase={apiBase} runId={runId} summary={current} />}
            {route.view === "infra" && <InfraView apiBase={apiBase} runId={runId} summary={current} />}
            {route.view === "config" && <ConfigView apiBase={apiBase} runId={runId} otherRuns={runIds.filter((id) => id !== current?.id)} />}
          </>
        )}
        </div>
      </main>
    </div>
  );
}

function NavLink({ active, href, icon, label, hint }: { active: boolean; href: string; icon: ReactElement; label: string; hint?: string }) {
  return (
    <a href={href} title={hint} className={`nav-link ${active ? "nav-link-active" : ""}`} aria-current={active ? "page" : undefined}>
      {icon}
      {label}
    </a>
  );
}
