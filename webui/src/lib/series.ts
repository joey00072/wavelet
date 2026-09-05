import type { MetricRow, Series } from "../api/types";

export type Point = { x: number; y: number };
export type LineSeries = { id: string; label: string; points: Point[]; envelope?: { min: Point[]; max: Point[] }; colorIndex?: number; deemphasize?: boolean };

export function seriesToLines(
  series: Series | null,
  keys: string[],
  options: { xAxis?: "step" | "time"; smoothing?: number; labels?: Record<string, string> } = {},
): LineSeries[] {
  if (!series) return [];
  const xAxis = options.xAxis ?? "step";
  const start = firstTime(series.timestamps);
  return keys.map((key, index) => {
    const values = series.series[key] ?? [];
    const pointsFor = (column: Array<number | null>): Point[] => {
      const points: Point[] = [];
      column.forEach((value, i) => {
        if (value === null || value === undefined || !Number.isFinite(value)) return;
        const x = xAxis === "time" ? elapsedMinutes(series.timestamps[i], start) : series.steps[i];
        if (x === null || x === undefined || !Number.isFinite(x)) return;
        points.push({ x, y: value });
      });
      points.sort((a, b) => a.x - b.x);
      return points;
    };
    const raw = pointsFor(values);
    const bounds = series.envelope?.[key];
    const envelope = bounds ? { min: pointsFor(bounds.min), max: pointsFor(bounds.max) } : undefined;
    return {
      id: key,
      label: options.labels?.[key] ?? key,
      points: options.smoothing && options.smoothing > 0 ? ema(raw, options.smoothing) : raw,
      envelope,
      colorIndex: index,
    };
  });
}

function ema(points: Point[], alphaComplement: number): Point[] {
  // alphaComplement in [0, 1): 0 = no smoothing, 0.9 = heavy.
  if (points.length === 0 || alphaComplement <= 0) return points;
  const out: Point[] = [];
  let acc = points[0].y;
  for (const point of points) {
    acc = alphaComplement * acc + (1 - alphaComplement) * point.y;
    out.push({ x: point.x, y: acc });
  }
  return out;
}

export function firstTime(timestamps: Array<string | null>): number | null {
  for (const stamp of timestamps) {
    if (!stamp) continue;
    const time = new Date(stamp).getTime();
    if (!Number.isNaN(time)) return time;
  }
  return null;
}

function elapsedMinutes(stamp: string | null | undefined, start: number | null): number | null {
  if (!stamp || start === null) return null;
  const time = new Date(stamp).getTime();
  if (Number.isNaN(time)) return null;
  return (time - start) / 60000;
}

export function lastFinite(values: Array<number | null> | undefined): number | null {
  if (!values) return null;
  for (let i = values.length - 1; i >= 0; i -= 1) {
    const value = values[i];
    if (value !== null && Number.isFinite(value)) return value;
  }
  return null;
}

export function trailingMean(
  values: Array<number | null> | undefined,
  window = 5,
): number | null {
  if (!values || window < 1) return null;
  const finite = values.filter(
    (value): value is number => value !== null && Number.isFinite(value),
  );
  const tail = finite.slice(-window);
  if (tail.length === 0) return null;
  return tail.reduce((sum, value) => sum + value, 0) / tail.length;
}

export function num(row: MetricRow | null | undefined, key: string): number | null {
  const value = row?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function windowDelta(values: Array<number | null> | undefined, window = 5): number | null {
  if (!values) return null;
  const finite = values.filter((v): v is number => v !== null && Number.isFinite(v));
  if (finite.length < 2) return null;
  const size = Math.min(window, Math.floor(finite.length / 2));
  if (size < 1) return null;
  const head = finite.slice(0, size);
  const tail = finite.slice(-size);
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  return mean(tail) - mean(head);
}

export function groupKeys(keys: string[]): Array<{ group: string; keys: string[] }> {
  const groups = new Map<string, string[]>();
  for (const key of keys) {
    const slash = key.indexOf("/");
    const group = slash > 0 ? key.slice(0, slash) : "other";
    groups.set(group, [...(groups.get(group) ?? []), key]);
  }
  return [...groups.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([group, list]) => ({ group, keys: list.sort() }));
}

export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const span = max - min;
  const rough = span / Math.max(count - 1, 1);
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const residual = rough / magnitude;
  const step = (residual >= 5 ? 5 : residual >= 2 ? 2 : 1) * magnitude;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-6; v += step) {
    ticks.push(Number(v.toFixed(12)));
  }
  return ticks;
}

export function extent(values: number[]): [number, number] | null {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return min <= max ? [min, max] : null;
}

/** First finite value among alternative metric keys (runs log different names across versions). */
export function numAny(row: MetricRow | null | undefined, keys: string[]): number | null {
  for (const key of keys) {
    const value = num(row, key);
    if (value !== null) return value;
  }
  return null;
}

export const TOKENS_PER_SECOND_KEYS = ["perf/tokens_per_second", "perf/train_tokens_per_second", "perf/throughput", "perf/step_tokens_per_second"];
export const STEP_SECONDS_KEYS = ["time/step", "perf/step_seconds"];
export const TRAIN_SECONDS_KEYS = ["time/train_until", "perf/train_seconds"];

export type SmoothingType = "ema" | "tema" | "running" | "gaussian";

/**
 * W&B-style smoothing family. `amount` is 0..1 from the slider.
 * ema: exponential moving average with weight `amount`.
 * tema: time-weighted EMA; the weight decays with x spacing so uneven steps are treated fairly.
 * running: centered running mean over a window that grows with `amount`.
 * gaussian: gaussian kernel whose std grows with `amount`.
 */
export function smooth(points: Point[], type: SmoothingType, amount: number): Point[] {
  if (points.length < 2 || amount <= 0) return points;
  switch (type) {
    case "ema":
      return ema(points, amount);
    case "tema":
      return timeWeightedEma(points, amount);
    case "running":
      return runningMean(points, Math.max(1, Math.round(1 + amount * (points.length / 8))));
    case "gaussian":
      return gaussian(points, Math.max(0.5, amount * (points.length / 12)));
    default:
      return points;
  }
}

function timeWeightedEma(points: Point[], amount: number): Point[] {
  const xs = points.map((p) => p.x);
  const span = Math.max(1e-9, xs[xs.length - 1] - xs[0]);
  const meanGap = span / Math.max(1, points.length - 1);
  const out: Point[] = [];
  let last: Point | null = null;
  let acc = 0;
  let debias = 0;
  for (const p of points) {
    const gap = last ? Math.max(0, p.x - last.x) / meanGap : 1;
    const w = amount ** gap;
    acc = acc * w + p.y * (1 - w);
    debias = debias * w + (1 - w);
    out.push({ x: p.x, y: acc / debias });
    last = p;
  }
  return out;
}

function runningMean(points: Point[], window: number): Point[] {
  const half = Math.floor(window / 2);
  return points.map((p, i) => {
    const lo = Math.max(0, i - half);
    const hi = Math.min(points.length - 1, i + half);
    let sum = 0;
    for (let j = lo; j <= hi; j += 1) sum += points[j].y;
    return { x: p.x, y: sum / (hi - lo + 1) };
  });
}

function gaussian(points: Point[], sigma: number): Point[] {
  const radius = Math.ceil(sigma * 3);
  return points.map((p, i) => {
    let weight = 0;
    let sum = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(points.length - 1, i + radius); j += 1) {
      const w = Math.exp(-0.5 * ((j - i) / sigma) ** 2);
      weight += w;
      sum += w * points[j].y;
    }
    return { x: p.x, y: sum / weight };
  });
}

/** Log-axis ticks that always label: 1-2-5 mantissas within a few decades, linear ticks inside one decade. */
export function logTicks(min: number, max: number, target = 5): number[] {
  const lo = Math.max(min, 1e-12);
  const hi = Math.max(max, lo * 1.0001);
  const decades = Math.log10(hi) - Math.log10(lo);
  if (decades < 1) return niceTicks(lo, hi, target).filter((t) => t > 0);
  const perDecade = target / Math.max(decades, 1);
  const mantissas = perDecade >= 7 ? [1, 2, 3, 4, 5, 6, 7, 8, 9] : perDecade >= 4 ? [1, 2, 3, 5, 7] : perDecade >= 2 ? [1, 2, 5] : [1];
  const ticks: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e += 1) {
    for (const m of mantissas) {
      const v = m * 10 ** e;
      if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v);
    }
  }
  if (ticks.length > target * 1.6) {
    const every = Math.ceil(ticks.length / target);
    return ticks.filter((_, i) => i % every === 0);
  }
  return ticks.length >= 2 ? ticks : niceTicks(lo, hi, target).filter((t) => t > 0);
}

/** Format a set of axis ticks with one consistent precision so neighbours stay distinguishable. */
export function tickFormatter(ticks: number[]): (v: number) => string {
  const finite = ticks.filter((t) => Number.isFinite(t));
  const maxAbs = Math.max(...finite.map((t) => Math.abs(t)), 0);
  let gap = Number.POSITIVE_INFINITY;
  const sorted = [...finite].sort((a, b) => a - b);
  for (let i = 1; i < sorted.length; i += 1) gap = Math.min(gap, Math.abs(sorted[i] - sorted[i - 1]));
  if (!Number.isFinite(gap) || gap === 0) gap = maxAbs || 1;
  if (maxAbs >= 1e6 || (maxAbs > 0 && maxAbs < 1e-3)) return (v) => (v === 0 ? "0" : v.toExponential(1).replace("e+", "e"));
  if (maxAbs >= 1e4) return (v) => `${(v / 1e3).toFixed(gap >= 1e3 ? 0 : 1)}k`;
  const decimals = Math.min(6, Math.max(0, Math.ceil(-Math.log10(gap)) + (gap / 10 ** Math.floor(Math.log10(gap)) < 2 ? 1 : 0)));
  return (v) => v.toFixed(decimals);
}
