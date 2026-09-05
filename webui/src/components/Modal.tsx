import { useId, type ReactNode } from "react";
import { X } from "lucide-react";

import { useDialogLifecycle } from "./focus";

export function Modal({ open, onClose, title, subtitle, children, actions }: { open: boolean; onClose: () => void; title: ReactNode; subtitle?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  const titleId = useId();
  const { closeRef, panelRef } = useDialogLifecycle<HTMLDivElement>(open, onClose);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <div className="animate-fade absolute inset-0 bg-black/50 backdrop-blur-[3px]" onClick={onClose} aria-hidden />
      <div ref={panelRef} className="animate-enter relative flex max-h-full w-full max-w-6xl flex-col rounded-lg bg-surface p-4 shadow-2xl sm:p-6" style={{ boxShadow: "0 30px 80px -20px rgba(0,0,0,.7), 0 0 0 1px var(--border)" }}>
        <header className="mb-4 flex flex-col items-start justify-between gap-3 sm:flex-row sm:gap-4">
          <div className="min-w-0">
            <div id={titleId} className="truncate text-base font-semibold tracking-tight text-ink">{title}</div>
            {subtitle && <div className="mt-0.5 text-xs text-muted">{subtitle}</div>}
          </div>
          <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto sm:shrink-0 sm:justify-end">
            {actions}
            <button ref={closeRef} type="button" className="btn !px-1.5" onClick={onClose} aria-label="Close dialog">
              <X className="h-4 w-4" />
            </button>
          </div>
        </header>
        <div className="min-h-0 flex-1 overflow-auto">{children}</div>
      </div>
    </div>
  );
}
