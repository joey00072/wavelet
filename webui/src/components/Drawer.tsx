import { useId, type ReactNode } from "react";
import { X } from "lucide-react";

import { useDialogLifecycle } from "./focus";

export function Drawer({ open, onClose, title, subtitle, children, width = 720 }: { open: boolean; onClose: () => void; title: ReactNode; subtitle?: ReactNode; children: ReactNode; width?: number }) {
  const titleId = useId();
  const { closeRef, panelRef } = useDialogLifecycle<HTMLElement>(open, onClose);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-40 flex">
      <div className="flex-1 bg-black/40 backdrop-blur-[2px]" onClick={onClose} aria-hidden />
      <aside ref={panelRef} className="flex h-full flex-col bg-surface shadow-2xl" style={{ width: `min(${width}px, 100vw)` }} role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <header className="flex items-start justify-between gap-3 px-5 py-4">
          <div className="min-w-0">
            <div id={titleId} className="truncate text-sm font-semibold text-ink">{title}</div>
            {subtitle && <div className="mt-0.5 text-[11px] text-muted">{subtitle}</div>}
          </div>
          <button ref={closeRef} type="button" className="btn !px-1.5" onClick={onClose} aria-label="Close drawer">
            <X className="h-3.5 w-3.5" />
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-auto px-5 pb-5">{children}</div>
      </aside>
    </div>
  );
}
