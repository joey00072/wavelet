import type { RolloutInspection, RolloutSnapshot } from "../types";

export const ROLLOUT_BUFFER_LIMIT = 12;
export const SAVED_ROLLOUT_LIMIT = 8;

export function rolloutBatchKey(inspection: RolloutInspection): string {
  return [
    inspection.queue_step ?? "none",
    inspection.path ?? "no-path",
    inspection.manifest?.created_at ?? "no-created-at",
  ].join(":");
}

export function makeRolloutSnapshot(
  inspection: RolloutInspection,
  capturedAt: string,
  source: RolloutSnapshot["source"],
): RolloutSnapshot {
  const batchKey = rolloutBatchKey(inspection);
  return {
    id: `${source}:${batchKey}:${capturedAt}`,
    batch_key: batchKey,
    captured_at: capturedAt,
    inspection,
    source,
  };
}

export function appendBufferedSnapshot(
  snapshots: RolloutSnapshot[],
  inspection: RolloutInspection,
  capturedAt: string,
): RolloutSnapshot[] {
  if (!inspection.available) {
    return snapshots;
  }
  const batchKey = rolloutBatchKey(inspection);
  if (snapshots.some((snapshot) => snapshot.batch_key === batchKey)) {
    return snapshots;
  }
  return [makeRolloutSnapshot(inspection, capturedAt, "buffer"), ...snapshots].slice(0, ROLLOUT_BUFFER_LIMIT);
}

export function prependSnapshot(
  snapshots: RolloutSnapshot[],
  snapshot: RolloutSnapshot,
  limit: number,
): RolloutSnapshot[] {
  return [snapshot, ...snapshots.filter((item) => item.id !== snapshot.id)].slice(0, limit);
}
