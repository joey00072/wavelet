import { useEffect, useRef, useState, type ReactNode } from "react";

export function Segmented<T extends string>({ value, options, onChange, size = "sm" }: { value: T; options: Array<{ value: T; label: ReactNode; title?: string }>; onChange: (value: T) => void; size?: "sm" | "xs" }) {
  return (
    <div className="inline-flex gap-0.5 rounded-md bg-raised p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          title={option.title}
          className={`${size === "xs" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs"} rounded transition-colors ${value === option.value ? "bg-surface font-medium text-ink shadow-[var(--shadow)]" : "text-ink2 hover:text-ink"}`}
          onClick={() => onChange(option.value)}
          aria-pressed={value === option.value}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Toolbar({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-center gap-2">{children}</div>;
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex items-center gap-1.5 text-[11px] text-muted">
      {label}
      {children}
    </label>
  );
}


export function Slider({ value, min, max, step, onChange, format, label }: { value: number; min: number; max: number; step: number; onChange: (value: number) => void; format?: (v: number) => string; label?: string }) {
  return (
    <label className="flex items-center gap-3 text-[11px] text-muted">
      {label && <span className="w-20 shrink-0">{label}</span>}
      <input type="range" className="slider flex-1" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="tabular w-10 text-right text-ink2">{format ? format(value) : value}</span>
    </label>
  );
}

/** Lightweight popover anchored to its trigger; closes on outside click or Escape. */
export function Popover({ trigger, children, align = "right", width = 300, placement = "bottom" }: { trigger: (open: boolean) => ReactNode; children: ReactNode; align?: "left" | "right"; width?: number; placement?: "bottom" | "top" }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <div onClick={() => setOpen((v) => !v)}>{trigger(open)}</div>
      {open && (
        <div className={`animate-fade absolute z-30 rounded-lg bg-surface p-4 ${placement === "top" ? "bottom-full mb-2" : "mt-2"} ${align === "right" ? "right-0" : "left-0"}`} style={{ width, maxWidth: "calc(100vw - 2rem)", boxShadow: "0 20px 50px -12px rgba(0,0,0,.6), 0 0 0 1px var(--border)" }}>
          {children}
        </div>
      )}
    </div>
  );
}
