import { cloneElement, isValidElement, useEffect, useRef, useState, type ReactNode } from "react";

import { LineChart } from "./LineChart";
import { Maximize2, Settings2, Table2, LineChart as LineIcon } from "lucide-react";

import { Popover, Segmented, Slider, smoothingLabel } from "../components/Controls";
import { Modal } from "../components/Modal";
import type { SmoothingType } from "../lib/series";

type ChartPrefs = {
  smoothing: number;
  smoothingType: SmoothingType;
  yMode: "auto" | "zero" | "fixed";
  yMin: number | null;
  yMax: number | null;
  xLog: boolean;
  yLog: boolean;
};
type ChartOptions = ChartPrefs & { height: number; expanded: boolean };

/** Props to spread onto a LineChart from a panel's options. */
export function chartProps(o: ChartOptions) {
  return { height: o.height, smoothing: o.smoothing, smoothingType: o.smoothingType, logScale: o.yLog, xLog: o.xLog, yMode: o.yMode, yMin: o.yMin, yMax: o.yMax };
}

const SMOOTHING_TYPES: Array<{ value: SmoothingType; label: string }> = [
  { value: "tema", label: "Time-weighted EMA" },
  { value: "ema", label: "EMA" },
  { value: "running", label: "Running average" },
  { value: "gaussian", label: "Gaussian" },
];

/** Bounded or non-negative metric families read better from zero than auto-fitted. */
function defaultYModeFor(title: string): ChartPrefs["yMode"] {
  return /reward|rate|ratio|frac|is_truncated|avg@|pass@|pass\^|mfu|accuracy|solve|admission|usage|kv_cache|entropy|loss|grad_norm|tokens|len|seconds|time\//i.test(title) ? "zero" : "auto";
}

/** "from zero" is meaningless on a log axis, so it is hidden and reads as auto. */
function yModeOptions(yLog: boolean, [auto, zero, fixed]: [string, string, string]): Array<{ value: ChartPrefs["yMode"]; label: string }> {
  return [{ value: "auto", label: auto }, ...(yLog ? [] : [{ value: "zero" as const, label: zero }]), { value: "fixed", label: fixed }];
}

type Children = ReactNode | ((options: ChartOptions) => ReactNode);

/**
 * W&B-style panel: quiet title, a toolbar that appears on hover (expand,
 * settings, table), and click-to-expand into a full-size view. Smoothing and
 * log scale live behind the gear and are remembered per chart.
 */
export function ChartCard({
  title,
  subtitle,
  value,
  table,
  children,
  className = "",
  refetching = false,
  smoothingKey,
  defaultSmoothing = 0,
  smoothable = true,
  height = 180,
  defaultLogScale = false,
}: {
  title: string;
  subtitle?: string;
  value?: ReactNode;
  table?: ReactNode;
  children: Children;
  className?: string;
  refetching?: boolean;
  smoothingKey?: string;
  defaultSmoothing?: number;
  smoothable?: boolean;
  height?: number;
  defaultLogScale?: boolean;
}) {
  const storageKey = `wavelet.chart.${smoothingKey ?? title}`;
  const defaults: ChartPrefs = { smoothing: defaultSmoothing, smoothingType: "tema", yMode: defaultYModeFor(title), yMin: null, yMax: null, xLog: false, yLog: defaultLogScale };
  const [stored, setStored] = useState<Partial<ChartPrefs> | null>(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      return raw ? (JSON.parse(raw) as Partial<ChartPrefs>) : null;
    } catch {
      return null;
    }
  });
  const prefs: ChartPrefs = { ...defaults, ...(stored ?? {}) };
  const { smoothing } = prefs;
  const yModeValue = prefs.yLog && prefs.yMode === "zero" ? "auto" : prefs.yMode;
  const update = (patch: Partial<ChartPrefs>) => {
    const next: Partial<ChartPrefs> = { ...(stored ?? {}), ...patch };
    for (const key of Object.keys(next) as Array<keyof ChartPrefs>) {
      if (next[key] === defaults[key]) delete next[key];
    }
    if (Object.keys(next).length === 0) {
      window.localStorage.removeItem(storageKey);
      setStored(null);
    } else {
      window.localStorage.setItem(storageKey, JSON.stringify(next));
      setStored(next);
    }
  };
  const prefsActive = stored !== null;
  const [showTable, setShowTable] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const down = useRef<{ x: number; y: number } | null>(null);
  useEffect(() => {
    if (!expanded) setShowTable(false);
  }, [expanded]);

  const render = (options: ChartOptions) => {
    if (typeof children === "function") return children(options);
    // Plain <LineChart/> children pick up the panel's smoothing, scale, and expanded height.
    if (isValidElement(children) && children.type === LineChart) {
      const own = (children.props as { height?: number }).height ?? height;
      return cloneElement(children as React.ReactElement<Record<string, unknown>>, { ...chartProps(options), height: options.expanded ? options.height : own });
    }
    return children;
  };
  const [prefix, name] = splitTitle(title);
  const logToggle = (axis: "xLog" | "yLog", label: string) => (
    <button type="button" className={`btn !py-0.5 ${prefs[axis] ? "btn-active" : ""}`} onClick={() => update({ [axis]: !prefs[axis] })}>{label}</button>
  );
  const settings = (
    <Popover
      width={280}
      trigger={(open) => (
        <button type="button" className={`btn !px-1.5 !py-1 ${open || prefsActive ? "text-ink" : ""}`} title="Chart settings">
          <Settings2 className="h-3.5 w-3.5" />
        </button>
      )}
    >
      <div className="space-y-4" onClick={(e) => e.stopPropagation()}>
        {smoothable && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-[11px] text-muted">
              <span>Smoothing</span>
              <select className="select !py-0.5 !text-[11px]" value={prefs.smoothingType} onChange={(e) => update({ smoothingType: e.target.value as SmoothingType })}>
                {SMOOTHING_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>{t.label}</option>
                ))}
              </select>
            </div>
            <Slider min={0} max={0.99} step={0.01} value={smoothing} onChange={(v) => update({ smoothing: v })} format={smoothingLabel} />
          </div>
        )}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] text-muted">
            <span>Y range</span>
            <Segmented value={yModeValue} onChange={(v) => update({ yMode: v })} size="xs" options={yModeOptions(prefs.yLog, ["fit data", "from zero", "custom"])} />
          </div>
          {prefs.yMode === "fixed" && (
            <div className="flex items-center gap-2 text-[11px] text-muted">
              <input className="input w-24 !py-1" aria-label="Minimum y value" inputMode="decimal" placeholder="min" value={prefs.yMin ?? ""} onChange={(e) => update({ yMin: e.target.value === "" ? null : Number(e.target.value) })} />
              <span>to</span>
              <input className="input w-24 !py-1" aria-label="Maximum y value" inputMode="decimal" placeholder="max" value={prefs.yMax ?? ""} onChange={(e) => update({ yMax: e.target.value === "" ? null : Number(e.target.value) })} />
            </div>
          )}
        </div>
        <div className="flex items-center justify-between text-[11px] text-muted">
          <span>Log scale</span>
          <div className="flex items-center gap-1">
            {logToggle("xLog", "x")}
            {logToggle("yLog", "y")}
          </div>
        </div>
        <div className="flex items-center justify-between">
          <p className="text-[10.5px] leading-relaxed text-muted">Drag to zoom the x range; double-click resets. Click the chart to expand.</p>
          {prefsActive && <button type="button" className="btn !py-0.5 text-[10.5px]" onClick={() => update(defaults)}>reset</button>}
        </div>
      </div>
    </Popover>
  );

  return (
    <>
      <figure className={`group/card flex min-w-0 flex-col ${className}`}>
        <figcaption className="relative z-20 mb-1.5 flex items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-[12.5px] font-medium text-ink" title={title}>
              {prefix && <span className="text-muted">{prefix}/</span>}
              <span className="font-semibold">{name}</span>
            </div>
            {(subtitle || value !== undefined) && (
              <div className="truncate text-[11px] text-muted">
                {value !== undefined && <span className="tabular text-ink2">{value}</span>}
                {value !== undefined && subtitle && " · "}
                {subtitle}
              </div>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <div className="chart-actions absolute right-0 top-1/2 flex -translate-y-1/2 items-center rounded-md bg-surface pl-2 opacity-0 transition-opacity duration-150 group-hover/card:opacity-100 focus-within:opacity-100" style={{ boxShadow: "-12px 0 12px -6px var(--surface-1)" }}>
              {settings}
              {table && (
                <button type="button" className="btn !px-1.5 !py-1" onClick={() => setShowTable((v) => !v)} aria-pressed={showTable} title={showTable ? "Show chart" : "Show table"}>
                  {showTable ? <LineIcon className="h-3.5 w-3.5" /> : <Table2 className="h-3.5 w-3.5" />}
                </button>
              )}
              <button type="button" className="btn !px-1.5 !py-1" onClick={() => setExpanded(true)} title="Expand">
                <Maximize2 className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </figcaption>
        <div
          className={`relative z-0 min-h-0 flex-1 rounded-md transition-colors ${refetching ? "refetching" : ""} ${showTable ? "" : "cursor-pointer hover:bg-raised/40"}`}
          role={showTable ? undefined : "button"}
          tabIndex={showTable ? undefined : 0}
          aria-label={showTable ? undefined : `Expand ${title} chart`}
          onKeyDown={(event) => {
            if (event.target === event.currentTarget && !showTable && (event.key === "Enter" || event.key === " ")) {
              event.preventDefault();
              setExpanded(true);
            }
          }}
          onPointerDown={(e) => {
            if ((e.target as Element).closest("button, input, select, a")) return;
            down.current = { x: e.clientX, y: e.clientY };
          }}
          onPointerUp={(e) => {
            const start = down.current;
            down.current = null;
            if ((e.target as Element).closest("button, input, select, a")) return;
            if (showTable || !start) return;
            if (Math.hypot(e.clientX - start.x, e.clientY - start.y) < 5) setExpanded(true);
          }}
        >
          {showTable && table ? table : render({ ...prefs, height, expanded: false })}
        </div>
      </figure>
      <Modal
        open={expanded}
        onClose={() => setExpanded(false)}
        title={title}
        subtitle={subtitle}
        actions={
          <>
            {smoothable && (
              <div className="w-full sm:w-56">
                <Slider label="Smoothing" min={0} max={0.99} step={0.01} value={smoothing} onChange={(v) => update({ smoothing: v })} format={smoothingLabel} />
              </div>
            )}
            <Segmented value={yModeValue} onChange={(v) => update({ yMode: v })} size="xs" options={yModeOptions(prefs.yLog, ["fit", "from 0", "custom"])} />
            {logToggle("xLog", "log x")}
            {logToggle("yLog", "log y")}
            {settings}
            {table && (
              <button type="button" className={`btn !px-1.5 ${showTable ? "btn-active" : ""}`} onClick={() => setShowTable((v) => !v)} title="Table view">
                <Table2 className="h-3.5 w-3.5" />
              </button>
            )}
          </>
        }
      >
        {showTable && table ? <div className="max-h-[70vh] overflow-auto">{table}</div> : render({ ...prefs, height: Math.max(360, Math.min(640, window.innerHeight - 260)), expanded: true })}
        {value !== undefined && <div className="mt-3 text-xs text-muted">latest <span className="tabular font-semibold text-ink">{value}</span></div>}
      </Modal>
    </>
  );
}

function splitTitle(title: string): [string | null, string] {
  const slash = title.lastIndexOf("/");
  if (slash <= 0 || title.includes(" ")) return [null, title];
  return [title.slice(0, slash), title.slice(slash + 1)];
}
