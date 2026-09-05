import { useEffect, useRef, useState } from "react";

/** Tweens between successive numeric values so live tiles glide instead of jumping. */
export function AnimatedNumber({ value, format, duration = 500 }: { value: number | null | undefined; format: (v: number) => string; duration?: number }) {
  const [shown, setShown] = useState<number | null>(value ?? null);
  const from = useRef<number | null>(value ?? null);
  useEffect(() => {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      setShown(null);
      from.current = null;
      return;
    }
    const start = from.current;
    if (start === null || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      setShown(value);
      from.current = value;
      return;
    }
    const t0 = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / duration);
      const eased = 1 - (1 - t) ** 3;
      setShown(start + (value - start) * eased);
      if (t < 1) frame = requestAnimationFrame(tick);
      else from.current = value;
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, duration]);
  return <>{shown === null ? "–" : format(shown)}</>;
}
