import { useEffect, useState, type ReactNode } from "react";
import { ChevronRight } from "lucide-react";

/**
 * Collapsible section that remembers its state per id. Secondary detail lives
 * behind it so a page opens on its primary signals only.
 */
export function Disclosure({ id, title, summary, defaultOpen = false, children, className = "" }: { id: string; title: string; summary?: ReactNode; defaultOpen?: boolean; children: ReactNode; className?: string }) {
  const key = `wavelet.disclosure.${id}`;
  const [open, setOpen] = useState<boolean>(() => {
    const stored = window.localStorage.getItem(key);
    return stored === null ? defaultOpen : stored === "1";
  });
  useEffect(() => {
    window.localStorage.setItem(key, open ? "1" : "0");
  }, [key, open]);
  return (
    <section className={className}>
      <button type="button" className="group flex w-full items-center gap-2 py-1 text-left" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <ChevronRight className={`h-3.5 w-3.5 text-muted transition-transform duration-200 ${open ? "rotate-90" : ""}`} />
        <span className="eyebrow group-hover:text-ink transition-colors">{title}</span>
        {!open && summary && <span className="ml-2 truncate text-[11px] text-muted">{summary}</span>}
      </button>
      <div className="grid transition-[grid-template-rows,opacity] duration-300 ease-out" style={{ gridTemplateRows: open ? "1fr" : "0fr", opacity: open ? 1 : 0 }}>
        <div className="min-h-0 overflow-hidden">
          <div className="pt-4">{open && children}</div>
        </div>
      </div>
    </section>
  );
}
