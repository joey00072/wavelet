import type { ReactNode } from "react";
import { Database, Pause, PackageCheck, Play, RefreshCw, Save, ScanSearch } from "lucide-react";

import { LaneTable } from "../components/LaneTable";
import { RolloutInspector } from "../components/RolloutInspector";
import type { PipelineInventory, RolloutEvent, RolloutInspection, RolloutSnapshot, RunState } from "../types";
import { formatNumber, formatTime } from "../utils/format";

export function RolloutsView({
  state,
  events,
  counts,
  inventory,
  inspection,
  liveInspection,
  inspectionError,
  inspectionUpdatedAt,
  autoFollow,
  bufferedSnapshots,
  savedSnapshots,
  selectedSnapshotId,
  onSetAutoFollow,
  onRefreshInspection,
  onSelectSnapshot,
  onSaveSnapshot,
}: {
  state: RunState | null;
  events: RolloutEvent[];
  counts: { submitted: number; completed: number; published: number };
  inventory: PipelineInventory;
  inspection: RolloutInspection | null;
  liveInspection: RolloutInspection | null;
  inspectionError: string | null;
  inspectionUpdatedAt: string | null;
  autoFollow: boolean;
  bufferedSnapshots: RolloutSnapshot[];
  savedSnapshots: RolloutSnapshot[];
  selectedSnapshotId: string | null;
  onSetAutoFollow: (follow: boolean) => void;
  onRefreshInspection: () => void;
  onSelectSnapshot: (snapshotId: string) => void;
  onSaveSnapshot: () => void;
}) {
  return (
    <div className="space-y-12 pb-12">
      {/* ── Queue stats ── */}
      <section className="grid grid-cols-2 gap-8 sm:grid-cols-4">
        <QueueStat
          icon={<ScanSearch className="h-4 w-4" />}
          label={autoFollow ? "Live batch" : "Reading batch"}
          value={inspection?.queue_step ?? "–"}
          sub={`${formatNumber(inspection?.scanned_rows, 0)} scanned`}
        />
        <QueueStat
          icon={<PackageCheck className="h-4 w-4" />}
          label="Published"
          value={formatNumber(counts.published, 0)}
          sub={`watermark ${formatNumber(inventory.publishedWatermark, 0)}`}
        />
        <QueueStat
          icon={<Database className="h-4 w-4" />}
          label="Ready"
          value={formatNumber(inventory.readyForTrainer, 0)}
          sub={`${formatNumber(inventory.readyForTrainer * inventory.rolloutsPerChunk, 0)} rollouts`}
        />
        <QueueStat
          icon={<Database className="h-4 w-4" />}
          label="Pending"
          value={formatNumber(state?.rollouts.pending_count ?? 0, 0)}
          sub={`${formatNumber(counts.submitted, 0)} submitted`}
        />
      </section>

      <div className="h-px w-full bg-slate-200 dark:bg-white/[0.04]" />

      {/* ── Inspector Controls ── */}
      <section>
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className="mb-1 text-sm font-semibold text-slate-900 dark:text-slate-100">
              Reading Mode
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {autoFollow
                ? "Following the newest stable batch as polling continues."
                : "Frozen on a selected snapshot."}
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <ActionButton
              onClick={() => onSetAutoFollow(!autoFollow)}
              active={autoFollow}
            >
              {autoFollow ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
              {autoFollow ? "Pause" : "Follow latest"}
            </ActionButton>
            <ActionButton onClick={onSaveSnapshot} disabled={!inspection?.available}>
              <Save className="h-3.5 w-3.5" />
              Save snapshot
            </ActionButton>
          </div>
        </div>

        {/* Snapshot timeline */}
        {(bufferedSnapshots.length > 0 || savedSnapshots.length > 0) && (
          <div className="mt-8 grid gap-8 border-t border-slate-100 pt-8 dark:border-white/[0.04] sm:grid-cols-2">
            <SnapshotList
              title="Recent buffer"
              snapshots={bufferedSnapshots}
              selectedSnapshotId={selectedSnapshotId}
              liveBatchKey={liveInspection?.available ? liveInspection.path : null}
              onSelectSnapshot={onSelectSnapshot}
            />
            <SnapshotList
              title="Saved snapshots"
              snapshots={savedSnapshots}
              selectedSnapshotId={selectedSnapshotId}
              liveBatchKey={null}
              onSelectSnapshot={onSelectSnapshot}
            />
          </div>
        )}
      </section>

      <div className="h-px w-full bg-slate-200 dark:bg-white/[0.04]" />

      {/* ── Inspector ── */}
      <RolloutInspector
        title={autoFollow ? "Live rollout inspector" : "Snapshot"}
        inspection={inspection}
        error={inspectionError}
        updatedAt={inspectionUpdatedAt}
        onRefresh={onRefreshInspection}
      />

      <div className="h-px w-full bg-slate-200 dark:bg-white/[0.04]" />

      {/* ── Lane table ── */}
      <section>
        <LaneTable events={events} />
      </section>
    </div>
  );
}

function QueueStat({ icon, label, value, sub }: { icon: ReactNode; label: string; value: ReactNode; sub: string }) {
  return (
    <div className="flex flex-col justify-center">
      <div className="flex items-center gap-1.5 text-sm font-medium text-slate-500 dark:text-slate-400">
        {icon}
        {label}
      </div>
      <div className="mt-1 font-mono text-3xl font-semibold tracking-tight text-slate-900 dark:text-slate-50">{value}</div>
      <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{sub}</div>
    </div>
  );
}

function ActionButton({
  children,
  onClick,
  active,
  disabled,
}: {
  children: ReactNode;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center gap-1.5 rounded-lg px-4 py-2 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
        active
          ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
          : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-white/10 dark:bg-transparent dark:text-slate-300 dark:hover:bg-white/5"
      }`}
    >
      {children}
    </button>
  );
}

function SnapshotList({
  title,
  snapshots,
  selectedSnapshotId,
  liveBatchKey,
  onSelectSnapshot,
}: {
  title: string;
  snapshots: RolloutSnapshot[];
  selectedSnapshotId: string | null;
  liveBatchKey: string | null;
  onSelectSnapshot: (id: string) => void;
}) {
  if (snapshots.length === 0) return null;
  return (
    <div>
      <div className="mb-3 text-xs font-medium uppercase tracking-widest text-slate-500 dark:text-slate-400">{title}</div>
      <div className="flex gap-3 overflow-x-auto pb-2">
        {snapshots.map((s) => {
          const isSelected = selectedSnapshotId === s.id;
          const isLive = liveBatchKey !== null && s.inspection.path === liveBatchKey;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => onSelectSnapshot(s.id)}
              className={`shrink-0 rounded-lg border px-3 py-2 text-left text-xs transition-colors ${
                isSelected
                  ? "border-slate-900 bg-slate-900 text-white shadow-sm dark:border-white dark:bg-white dark:text-slate-900"
                  : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-white/10 dark:bg-transparent dark:text-slate-400 dark:hover:border-white/20"
              }`}
            >
              <div className="font-mono font-medium">step {s.inspection.queue_step ?? "–"}</div>
              <div className={`mt-1 text-[10px] ${isSelected ? "opacity-75" : "text-slate-500"}`}>
                {formatTime(s.captured_at)}{isLive ? " · live" : ""}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
