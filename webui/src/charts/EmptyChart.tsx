import type { Ref } from "react";

/** Placeholder a chart renders while it has nothing to plot; keeps the measured box so layout does not jump. */
export function EmptyChart({ height, text = "No data yet", ref }: { height: number; text?: string; ref?: Ref<HTMLDivElement> }) {
  return (
    <div ref={ref} className="flex items-center justify-center text-xs text-muted" style={{ height }}>
      {text}
    </div>
  );
}
