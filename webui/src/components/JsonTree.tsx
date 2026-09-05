import { useMemo, useState } from "react";

type Diff = "added" | "removed" | "changed" | null;

export function JsonTree({ value, other, filter = "", depth = 0, path = "", comparing }: { value: unknown; other?: unknown; filter?: string; depth?: number; path?: string; comparing?: boolean }) {
  if (!isObject(value) && !Array.isArray(value)) {
    return <Leaf value={value} other={other} path={path} />;
  }
  const diffEnabled = comparing ?? other !== undefined;
  const valueRecord = value as Record<string, unknown>;
  const otherRecord = isObject(other) || Array.isArray(other) ? other as Record<string, unknown> : {};
  const keys = new Set(Object.keys(valueRecord));
  if (diffEnabled) Object.keys(otherRecord).forEach((key) => keys.add(key));
  const entries = [...keys].sort((a, b) => a.localeCompare(b));
  return (
    <ul className={depth === 0 ? "space-y-0.5" : "ml-3 space-y-0.5 border-l border-edge pl-2"}>
      {entries.map((key) => {
        const valuePresent = Object.prototype.hasOwnProperty.call(valueRecord, key);
        const otherPresent = diffEnabled && Object.prototype.hasOwnProperty.call(otherRecord, key);
        const child = valueRecord[key];
        const childPath = path ? `${path}.${key}` : key;
        const otherChild = otherRecord[key];
        const searchable = `${JSON.stringify(child) ?? ""} ${JSON.stringify(otherChild) ?? ""}`.toLowerCase();
        if (filter && !childPath.toLowerCase().includes(filter.toLowerCase()) && !searchable.includes(filter.toLowerCase())) {
          return null;
        }
        return <Node key={key} name={key} value={child} valuePresent={valuePresent} other={otherChild} otherPresent={otherPresent} comparing={diffEnabled} filter={filter} depth={depth + 1} path={childPath} />;
      })}
    </ul>
  );
}

function Node({ name, value, valuePresent, other, otherPresent, comparing, filter, depth, path }: { name: string; value: unknown; valuePresent: boolean; other: unknown; otherPresent: boolean; comparing: boolean; filter: string; depth: number; path: string }) {
  const shownValue = valuePresent ? value : other;
  const nested = isObject(shownValue) || Array.isArray(shownValue);
  const [open, setOpen] = useState(depth <= 1 || Boolean(filter));
  const shownOpen = open || Boolean(filter);
  const diff: Diff = useMemo(() => {
    if (!comparing) return null;
    if (!valuePresent && otherPresent) return "removed";
    if (valuePresent && !otherPresent) return "added";
    if (JSON.stringify(other) !== JSON.stringify(value)) return "changed";
    return null;
  }, [comparing, other, otherPresent, value, valuePresent]);
  const diffClass = diff === "added" ? "text-good" : diff === "changed" ? "text-warn" : diff === "removed" ? "text-critical" : "text-ink2";
  return (
    <li>
      <div className="flex items-baseline gap-1.5 text-xs">
        {nested ? (
          <button type="button" className="tabular w-3 text-muted" onClick={() => setOpen((v) => !v)} aria-label={`${shownOpen ? "Collapse" : "Expand"} ${path}`} aria-expanded={shownOpen}>
            {shownOpen ? "−" : "+"}
          </button>
        ) : (
          <span className="w-3" />
        )}
        <span className={`font-medium ${diffClass}`}>{name}</span>
        {!nested && <Leaf value={shownValue} other={diff === "changed" ? other : undefined} path={path} inline diff={diff} />}
        {nested && !shownOpen && <span className="text-muted">{Array.isArray(shownValue) ? `[${shownValue.length}]` : `{${Object.keys(shownValue as object).length}}`}</span>}
      </div>
      {nested && shownOpen && <JsonTree value={valuePresent ? value : Array.isArray(shownValue) ? [] : {}} other={otherPresent ? other : undefined} comparing={comparing} filter={filter} depth={depth} path={path} />}
    </li>
  );
}

function Leaf({ value, other, inline = false, diff = null }: { value: unknown; other?: unknown; path?: string; inline?: boolean; diff?: Diff }) {
  const text = value === null ? "null" : typeof value === "string" ? `"${value}"` : String(value);
  const changed = other !== undefined && JSON.stringify(other) !== JSON.stringify(value);
  return (
    <span className={`${inline ? "" : "text-xs"} tabular break-all font-mono ${diff === "removed" ? "text-critical line-through" : "text-ink"}`}>
      {text}
      {changed && (
        <span className="ml-2 text-muted line-through">{other === null ? "null" : typeof other === "string" ? `"${other}"` : String(other)}</span>
      )}
    </span>
  );
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function flattenJson(value: unknown, prefix = ""): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (isObject(value)) {
    for (const [key, child] of Object.entries(value)) Object.assign(out, flattenJson(child, prefix ? `${prefix}.${key}` : key));
  } else if (Array.isArray(value)) {
    out[prefix] = JSON.stringify(value);
  } else {
    out[prefix] = value;
  }
  return out;
}
