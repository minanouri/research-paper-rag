// Client for the FastAPI backend

export type Paper = {
  paper_id: string;
  title?: string;
  authors?: string;
  year?: number;
  source: string;
  chunks: number;
};

export type Source = {
  id: number; // number in the answer (renumbered for cited sources)
  retrieved_id?: number; // number among the retrieved excerpts, for cited sources
  paper_id: string;
  title: string | null;
  authors: string | null;
  year: number | null;
  section: string | null;
  subsection: string | null;
  page: number | null;
  source: string | null;
  excerpt: string;
  score: number | null;
};

export type Stage = "searching" | "reranking" | "writing";

export type AnswerEvent =
  | { type: "status"; stage: Stage }
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; cited: Source[] }
  | { type: "error"; message: string };

export type AskFilters = {
  paper?: string;
  section_type?: string;
  year_min?: number;
};

export async function fetchPapers(): Promise<{ papers: Paper[]; section_types: string[] }> {
  const res = await fetch("/papers");
  if (!res.ok) throw new Error(`Could not load papers (HTTP ${res.status})`);
  return res.json();
}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    // fall through
  }
  return `Request failed (HTTP ${res.status})`;
}

/** POST /ask and call onEvent for each server-sent event, until the stream ends or is aborted. */
export async function ask(
  question: string,
  filters: AskFilters,
  onEvent: (event: AnswerEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, ...filters }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorMessage(res));

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // Events are separated by a blank line.
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

export function pdfUrl(paperId: string, page: number | null): string {
  const path = paperId.split("/").map(encodeURIComponent).join("/");
  return `/pdf/${path}${page ? `#page=${page}` : ""}`;
}

export function citeLabel(authors: string | null, year: number | null): string {
  const names = (authors ?? "").split(";").map((a) => a.trim()).filter(Boolean);
  const last = (n: string) => (n.includes(",") ? n.split(",")[0] : n.split(/\s+/).at(-1)!);
  let label = "Unknown";
  if (names.length === 1) label = last(names[0]);
  else if (names.length === 2) label = `${last(names[0])} & ${last(names[1])}`;
  else if (names.length > 2) label = `${last(names[0])} et al.`;
  return year ? `${label}, ${year}` : label;
}

export function sectionLabel(s: Source): string {
  return [s.section, s.subsection].filter(Boolean).join(" → ") || "Unknown section";
}
