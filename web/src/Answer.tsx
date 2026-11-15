import type { ReactNode } from "react";
import type { Source, Stage } from "./api";
import { citeLabel, sectionLabel } from "./api";

export type Turn = {
  id: number;
  question: string;
  stage: Stage | "done" | "stopped" | "error";
  sources: Source[];
  text: string;
  cited: Source[] | null;
  error?: string;
};

const STAGE_TEXT: Record<Stage, string> = {
  searching: "Searching papers…",
  reranking: "Reranking excerpts…",
  writing: "Writing answer…",
};

const CITATION_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

function AnswerText({ text, cited }: { text: string; cited: Source[] | null }) {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(CITATION_RE)) {
    parts.push(text.slice(last, m.index));
    const ids = m[1].split(",").map((v) => Number(v.trim()));
    const sources = ids.map((id) => cited?.find((s) => s.id === id));
    if (cited && sources.some((s) => !s)) {
      // Citation the backend could not map
      parts.push(m[0]);
    } else {
      parts.push(
        <span key={m.index} className="whitespace-nowrap">
          {ids.map((id, i) => (
            <CitationChip key={id} id={id} source={sources[i]} />
          ))}
        </span>,
      );
    }
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  return <div className="whitespace-pre-wrap leading-relaxed">{parts}</div>;
}

function CitationChip({ id, source }: { id: number; source?: Source }) {
  const base = "mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded px-1 align-baseline text-xs font-semibold";
  if (!source) {
    return <span className={`${base} bg-stone-200 text-stone-500`}>{id}</span>;
  }
  return (
    <span
      title={`${citeLabel(source.authors, source.year)} · ${sectionLabel(source)}${source.page ? ` · p. ${source.page}` : ""}`}
      className={`${base} cursor-help bg-indigo-100 text-indigo-700`}
    >
      {id}
    </span>
  );
}

function SourceRow({ source }: { source: Source }) {
  return (
    <li className="flex gap-3 px-2 py-1.5 text-sm">
        <span className="mt-0.5 inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded bg-indigo-100 px-1 text-xs font-semibold text-indigo-700">
          {source.id}
        </span>
        <span className="min-w-0">
          <span className="font-medium">{citeLabel(source.authors, source.year)}</span>
          <span className="text-stone-500"> · {sectionLabel(source)}{source.page ? ` · p. ${source.page}` : ""}</span>
          <span className="block truncate text-stone-500">{source.title ?? source.source}</span>
      </span>
    </li>
  );
}

export function TurnView({ turn }: { turn: Turn }) {
  const busy = turn.stage === "searching" || turn.stage === "reranking" || turn.stage === "writing";
  return (
    <article className="space-y-4">
      <h2 className="text-lg font-semibold text-stone-900">{turn.question}</h2>

      {busy && !turn.text && (
        <p className="flex items-center gap-2 text-sm text-stone-500">
          <span className="size-2 animate-pulse rounded-full bg-indigo-500" />
          {STAGE_TEXT[turn.stage as Stage]}
        </p>
      )}

      {turn.text && (
        <div className="text-[15px] text-stone-800">
          <AnswerText text={turn.text} cited={turn.cited} />
          {busy && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-stone-400 align-text-bottom" />}
        </div>
      )}

      {turn.stage === "stopped" && <p className="text-sm text-stone-500">Answer stopped.</p>}
      {turn.error && (
        <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{turn.error}</p>
      )}

      {turn.cited && turn.sources.length > 0 && (
        <section>
          <h3 className="mb-1 text-xs font-semibold tracking-wide text-stone-500 uppercase">Cited sources</h3>
          {turn.cited.length ? (
            <ul>
              {turn.cited.map((s) => (
                <SourceRow key={s.id} source={s} />
              ))}
            </ul>
          ) : (
            <p className="text-sm text-stone-500">No valid inline citations were found in the answer.</p>
          )}
        </section>
      )}

      {turn.sources.length > 0 && !busy && (
        <details className="text-sm">
          <summary className="cursor-pointer text-stone-500 select-none hover:text-stone-700">
            All {turn.sources.length} retrieved excerpts
          </summary>
          <ul className="mt-1">
            {turn.sources.map((s) => (
              <SourceRow key={s.id} source={s} />
            ))}
          </ul>
        </details>
      )}
    </article>
  );
}
