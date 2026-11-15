import { useEffect, useRef, useState } from "react";
import type { AnswerEvent, AskFilters, Paper } from "./api";
import { ask, citeLabel, fetchPapers } from "./api";
import { TurnView, type Turn } from "./Answer";

const sectionName = (s: string) => s.charAt(0).toUpperCase() + s.slice(1).replaceAll("_", " ");
const selectClass =
  "min-w-0 rounded-md border border-stone-300 bg-white px-2 py-1.5 text-sm focus:border-indigo-500 focus:outline-none";

function applyEvent(turn: Turn, e: AnswerEvent): Turn {
  switch (e.type) {
    case "status":
      return { ...turn, stage: e.stage };
    case "sources":
      return { ...turn, sources: e.sources };
    case "token":
      return { ...turn, text: turn.text + e.text };
    case "done":
      return { ...turn, cited: e.cited, stage: "done" };
    case "error":
      return { ...turn, error: e.message, stage: "error" };
  }
}

export default function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [sectionTypes, setSectionTypes] = useState<string[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [paper, setPaper] = useState("");
  const [sectionType, setSectionType] = useState("");
  const [yearMin, setYearMin] = useState("");

  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);

  const controller = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  useEffect(() => {
    fetchPapers()
      .then((d) => {
        setPapers(d.papers);
        setSectionTypes(d.section_types);
      })
      .catch((e) => setLoadError(`${e.message}. Is the backend running (python -m paper_rag serve)?`));
  }, []);

  // Follow the streaming answer (unless the reader has scrolled up)
  useEffect(() => {
    const el = scroller.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight;
  }, [turns]);

  async function submit() {
    const q = question.trim();
    if (!q || busy) return;
    const id = nextId.current++;
    const update = (f: (t: Turn) => Turn) => setTurns((ts) => ts.map((t) => (t.id === id ? f(t) : t)));
    setTurns((ts) => [...ts, { id, question: q, stage: "searching", sources: [], text: "", cited: null }]);
    setQuestion("");
    setBusy(true);
    requestAnimationFrame(() => scroller.current?.scrollTo({ top: scroller.current.scrollHeight }));

    const filters: AskFilters = {};
    if (paper) filters.paper = paper;
    if (sectionType) filters.section_type = sectionType;
    if (yearMin) filters.year_min = Number(yearMin);

    const ctrl = new AbortController();
    controller.current = ctrl;
    try {
      await ask(q, filters, (e) => update((t) => applyEvent(t, e)), ctrl.signal);
      update((t) =>
        t.stage === "done" || t.stage === "error"
          ? t
          : { ...t, stage: "error", error: "The answer stream ended unexpectedly." },
      );
    } catch (err) {
      if (ctrl.signal.aborted) update((t) => ({ ...t, stage: "stopped" }));
      else update((t) => ({ ...t, stage: "error", error: (err as Error).message }));
    } finally {
      controller.current = null;
      setBusy(false);
    }
  }

  return (
    <div className="flex h-dvh">
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="border-b border-stone-200 bg-white px-4 py-3">
          <div className="mx-auto flex max-w-3xl flex-wrap items-center gap-x-4 gap-y-2">
            <h1 className="mr-auto font-semibold">Paper RAG</h1>
            <select aria-label="Paper" value={paper} onChange={(e) => setPaper(e.target.value)} className={`${selectClass} max-w-56`}>
              <option value="">All papers</option>
              {papers.map((p) => (
                <option key={p.paper_id} value={p.paper_id}>
                  {citeLabel(p.authors ?? null, p.year ?? null)} — {p.title ?? p.source}
                </option>
              ))}
            </select>
            <select aria-label="Section type" value={sectionType} onChange={(e) => setSectionType(e.target.value)} className={selectClass}>
              <option value="">Any section</option>
              {sectionTypes.map((s) => (
                <option key={s} value={s}>
                  {sectionName(s)}
                </option>
              ))}
            </select>
            <label className="flex items-center gap-1.5 text-sm text-stone-600">
              From
              <input
                type="number"
                inputMode="numeric"
                placeholder="year"
                value={yearMin}
                onChange={(e) => setYearMin(e.target.value)}
                className={`${selectClass} w-20`}
              />
            </label>
          </div>
        </header>

        <main ref={scroller} className="flex-1 overflow-y-auto px-4">
          <div className="mx-auto max-w-3xl space-y-10 py-8">
            {loadError && (
              <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{loadError}</p>
            )}
            {turns.length === 0 && !loadError && (
              <div className="space-y-3 pt-8">
                <h2 className="text-2xl font-semibold">Ask your papers a question</h2>
                <p className="text-stone-600">
                  Answers cite the excerpts they use. Hover over a citation to see its source.
                </p>
                {papers.length > 0 && (
                  <ul className="space-y-1 pt-2 text-sm text-stone-600">
                    {papers.map((p) => (
                      <li key={p.paper_id}>
                        <span className="font-medium text-stone-800">{citeLabel(p.authors ?? null, p.year ?? null)}</span>{" "}
                        — {p.title ?? p.source} <span className="text-stone-400">({p.chunks} chunks)</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            {turns.map((t) => (
              <TurnView key={t.id} turn={t} />
            ))}
          </div>
        </main>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
          className="border-t border-stone-200 bg-white px-4 py-3"
        >
          <div className="mx-auto flex max-w-3xl items-end gap-2">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  submit();
                }
              }}
              rows={1}
              placeholder="Ask a question about the papers…"
              aria-label="Question"
              className="field-sizing-content max-h-40 min-h-10 flex-1 resize-none rounded-md border border-stone-300 px-3 py-2 text-[15px] focus:border-indigo-500 focus:outline-none"
            />
            {busy ? (
              <button
                type="button"
                onClick={() => controller.current?.abort()}
                className="h-10 rounded-md border border-stone-300 px-4 text-sm font-medium hover:bg-stone-100"
              >
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!question.trim()}
                className="h-10 rounded-md bg-indigo-600 px-4 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-40"
              >
                Ask
              </button>
            )}
          </div>
        </form>
      </div>

    </div>
  );
}
