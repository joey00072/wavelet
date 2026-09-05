import type { Point } from "../lib/series";
import { extent } from "../lib/series";

export function Sparkline({ points, width = 96, height = 26, colorVar = "var(--series-1)" }: { points: Point[]; width?: number; height?: number; colorVar?: string }) {
  const finite = points.filter((p) => Number.isFinite(p.y));
  if (finite.length < 2) return <svg width={width} height={height} />;
  const xe = extent(finite.map((p) => p.x))!;
  const ye = extent(finite.map((p) => p.y))!;
  const sx = (x: number) => (xe[0] === xe[1] ? width / 2 : ((x - xe[0]) / (xe[1] - xe[0])) * (width - 4) + 2);
  const sy = (y: number) => (ye[0] === ye[1] ? height / 2 : height - 3 - ((y - ye[0]) / (ye[1] - ye[0])) * (height - 6));
  const path = finite.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
  const last = finite[finite.length - 1];
  return (
    <svg width={width} height={height} aria-hidden>
      <path d={path} fill="none" stroke="var(--deemph)" strokeWidth={1.5} strokeLinejoin="round" />
      <circle cx={sx(last.x)} cy={sy(last.y)} r={3} fill={colorVar} stroke="var(--surface-1)" strokeWidth={1.5} />
    </svg>
  );
}
