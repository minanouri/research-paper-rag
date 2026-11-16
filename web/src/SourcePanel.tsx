import { useEffect } from "react";
import type { Source } from "./api";
import { pdfUrl, sectionLabel } from "./api";

export function SourcePanel({ source, onClose }: { source: Source; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const retrieved = source.retrieved_id ?? source.id;
  return (
    <aside className="fixed inset-y-0 right-0 z-20 flex w-full flex-col border-l border-stone-200 bg-white shadow-xl sm:w-[28rem] lg:static lg:z-auto lg:w-[30rem] lg:shrink-0 lg:shadow-none">
      <header className="flex items-start gap-3 border-b border-stone-200 p-4">
        <span className="mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded bg-indigo-600 px-1.5 text-sm font-semibold text-white">
          {source.id}
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="leading-snug font-semibold">{source.title ?? source.source ?? "Unknown title"}</h2>
          <p className="mt-1 text-sm text-stone-600">
            {source.authors?.split(";").map((a) => a.trim()).join(", ") || "Unknown authors"}
            {source.year ? ` · ${source.year}` : ""}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close source"
          className="-m-1 rounded p-1 text-stone-400 hover:bg-stone-100 hover:text-stone-700"
        >
          <svg viewBox="0 0 20 20" className="size-5" fill="currentColor" aria-hidden="true">
            <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
          </svg>
        </button>
      </header>

      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 border-b border-stone-200 px-4 py-3 text-sm">
        <dt className="text-stone-500">Section</dt>
        <dd>{sectionLabel(source)}</dd>
        <dt className="text-stone-500">Page</dt>
        <dd>{source.page ?? "unknown"}</dd>
        <dt className="text-stone-500">Retrieved</dt>
        <dd>
          excerpt #{retrieved}
          {source.score != null && <span className="text-stone-500"> · rerank score {source.score.toFixed(3)}</span>}
        </dd>
      </dl>

      <div className="flex-1 overflow-y-auto px-4 py-3">
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-stone-500 uppercase">Excerpt</h3>
        <p className="text-sm leading-relaxed whitespace-pre-wrap text-stone-800">{source.excerpt}</p>
      </div>

      <footer className="border-t border-stone-200 p-4">
        <a
          href={pdfUrl(source.paper_id, source.page)}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md bg-stone-900 px-3 py-2 text-sm font-medium text-white hover:bg-stone-700"
        >
          Open PDF{source.page ? ` at page ${source.page}` : ""}
          <svg viewBox="0 0 20 20" className="size-4" fill="currentColor" aria-hidden="true">
            <path d="M11 3a1 1 0 1 0 0 2h2.59l-6.3 6.3a1 1 0 1 0 1.42 1.4L15 6.42V9a1 1 0 1 0 2 0V4a1 1 0 0 0-1-1h-5Z" />
            <path d="M5 5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-3a1 1 0 1 0-2 0v3H5V7h3a1 1 0 0 0 0-2H5Z" />
          </svg>
        </a>
      </footer>
    </aside>
  );
}
