import type { ReactNode } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

export type Column<T> = {
  key: string;
  label: string;
  render: (row: T) => ReactNode;
  sortable?: boolean;
  align?: "left" | "right";
  width?: string;
  title?: string;
};

export type Sort = { key: string; desc: boolean };

/** Clicking the active column flips direction; a new column starts descending. */
export function toggleSort(sort: Sort, key: string): Sort {
  return sort.key === key ? { key, desc: !sort.desc } : { key, desc: true };
}

export function DataTable<T>({
  rows,
  columns,
  rowKey,
  sort,
  onSort,
  onRowClick,
  empty = "No rows",
  dense = false,
  maxHeight,
  label = "Data table",
}: {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  sort?: Sort | null;
  onSort?: (key: string) => void;
  onRowClick?: (row: T) => void;
  empty?: string;
  dense?: boolean;
  maxHeight?: number | string;
  label?: string;
}) {
  return (
    <div className="data-table-scroll overflow-auto" style={{ maxHeight }} role="region" aria-label={label} tabIndex={0}>
      <table className="w-full border-collapse">
        <thead className="sticky top-0 z-[1] bg-surface">
          <tr>
            {columns.map((col) => {
              const active = sort?.key === col.key;
              const sortable = col.sortable && onSort;
              return (
                <th
                  key={col.key}
                  className={`th ${col.align === "right" ? "text-right" : ""}`}
                  style={{ width: col.width }}
                  title={col.title}
                  aria-sort={active ? (sort?.desc ? "descending" : "ascending") : undefined}
                >
                  {sortable ? (
                    <button type="button" className={`inline-flex w-full items-center gap-0.5 hover:text-ink ${col.align === "right" ? "justify-end" : ""}`} onClick={() => onSort(col.key)}>
                      {col.label}
                      {active && (sort?.desc ? <ChevronDown className="h-3 w-3" /> : <ChevronUp className="h-3 w-3" />)}
                    </button>
                  ) : (
                    <span className="inline-flex items-center gap-0.5">{col.label}</span>
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td className="td py-6 text-center text-muted" colSpan={columns.length}>
                {empty}
              </td>
            </tr>
          )}
          {rows.map((row) => (
            <tr
              key={rowKey(row)}
              className={`border-b border-edge last:border-0 ${onRowClick ? "tr-hover cursor-pointer" : ""}`}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              onKeyDown={onRowClick ? (event) => {
                if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) {
                  event.preventDefault();
                  onRowClick(row);
                }
              } : undefined}
              tabIndex={onRowClick ? 0 : undefined}
            >
              {columns.map((col) => (
                <td key={col.key} className={`td ${dense ? "!py-1" : ""} ${col.align === "right" ? "text-right tabular" : ""}`}>
                  {col.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Pager({ offset, limit, total, onChange }: { offset: number; limit: number; total: number; onChange: (offset: number) => void }) {
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  return (
    <div className="flex items-center justify-between gap-2 text-[11px] text-muted">
      <span className="tabular">
        {total === 0 ? "0" : `${offset + 1}–${Math.min(offset + limit, total)}`} of {total}
      </span>
      <div className="flex items-center gap-1">
        <button type="button" className="btn !py-0.5" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>
          Prev
        </button>
        <span className="tabular px-1">
          {page}/{pages}
        </span>
        <button type="button" className="btn !py-0.5" disabled={offset + limit >= total} onClick={() => onChange(offset + limit)}>
          Next
        </button>
      </div>
    </div>
  );
}
