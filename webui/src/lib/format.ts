export function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e4) return `${(value / 1e3).toFixed(1)}K`;
  if (abs >= 100) return value.toFixed(Math.max(0, digits - 2));
  if (abs >= 1) return value.toFixed(digits);
  if (abs === 0) return "0";
  if (abs < 1e-4) return value.toExponential(2);
  return value.toFixed(digits + 1);
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return Math.round(value).toLocaleString();
}

export function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let index = 0;
  let scaled = value;
  while (scaled >= 1024 && index < units.length - 1) {
    scaled /= 1024;
    index += 1;
  }
  return `${scaled.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function fmtSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (value < 1) return `${(value * 1000).toFixed(0)} ms`;
  if (value < 90) return `${value.toFixed(1)} s`;
  if (value < 5400) return `${(value / 60).toFixed(1)} min`;
  return `${(value / 3600).toFixed(2)} h`;
}

export function fmtTime(value: string | null | undefined): string {
  if (!value) return "–";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function fmtDateTime(value: string | null | undefined): string {
  if (!value) return "–";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function fmtAge(value: string | null | undefined, now = Date.now()): string {
  if (!value) return "–";
  const time = new Date(value).getTime();
  if (Number.isNaN(time)) return "–";
  const seconds = Math.max(0, (now - time) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h ago`;
  return `${(seconds / 86400).toFixed(1)}d ago`;
}

export function shortId(value: string | null | undefined, max = 28): string {
  if (!value) return "–";
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

/** Compact tick labels: at most three significant digits. */
export function fmtAxis(value: number): string {
  if (!Number.isFinite(value)) return "–";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e4) return fmt(value, 3);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 1) return Number(value.toPrecision(3)).toString();
  return Number(value.toPrecision(2)).toString();
}

/** Human model label from a name or a Hugging Face cache snapshot path. */
export function modelLabel(name: string | null | undefined): string {
  if (!name) return "–";
  const cache = name.match(/models--([^/]+)--([^/]+)/);
  if (cache) return `${cache[1]}/${cache[2]}`;
  const parts = name.replace(/\/+$/, "").split("/");
  return parts.length >= 2 ? parts.slice(-2).join("/") : name;
}
