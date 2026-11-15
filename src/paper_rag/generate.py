from llama_index.core.llms import ChatMessage
import re

from .metadata import get_llm
from .retrieve import hybrid_search, reranker
from . import config


SYSTEM = """You answer questions about research papers using ONLY the numbered excerpts provided in the user's message.
Follow these rules:
- Use only the supplied excerpts to answer the question.
- Cite supporting claims using the exact numeric IDs provided, such as [1] or [2, 3].
- Never invent citation IDs or cite sources that do not support the claim.
- Place citations immediately after the claims they support.
- If the excerpts do not provide enough evidence, state that clearly.
- If the excerpts conflict, explain the disagreement and cite both sides."""


def cite_label(authors: str, year: int) -> str:
    names = [a.strip() for a in authors.split(";") if a.strip()]
    last = lambda n: n.split(",")[0] if "," in n else n.split()[-1]
    if not names:
        label = "Unknown"
    elif len(names) == 1:
        label = last(names[0])
    elif len(names) == 2:
        label = f"{last(names[0])} & {last(names[1])}"
    else:
        label = f"{last(names[0])} et al."
    return f"{label}, {year}" if year else label


def source_line(meta: dict) -> str:
    citation = cite_label(str(meta.get("authors") or ""), meta.get("year") or 0)
    title = meta.get("title") or meta.get("source") or "Unknown title"

    sections = [value for value in (meta.get("section"), meta.get("subsection")) if value]
    section = " -> ".join(sections) or "Unknown section"

    page = meta.get("page")
    page_text = f"p. {page}" if page is not None else "page unknown"

    return f'{citation} | "{title}" | {section} | {page_text}'


CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# The end of streamed text that may still become a citation, e.g. "[2" or "[1, "
PARTIAL_CITATION_RE = re.compile(r"\[[\d,\s]*$")


def extract_citations(answer: str) -> list[int]:
    found = CITATION_RE.findall(answer)
    citation_ids = []
    seen = set()

    for group in found:
        for value in group.split(","):
            citation_id = int(value.strip())
            if citation_id not in seen:
                seen.add(citation_id)
                citation_ids.append(citation_id)

    return citation_ids


class CitationRenumberer:
    """Renumber citations while the answer streams in.

    Cited excerpts that share a source (same paper, section and page) get one number, and numbers
    follow the order of first citation. Text after an unclosed "[" is held back until it can be decided.
    """

    def __init__(self, sources):
        self.sources = sources
        self.unique_sources = []
        self._new_ids = {}
        self._key_ids = {}
        self._pending = ""

    def feed(self, text: str) -> str:
        """Add streamed text and return the part that is ready to print."""
        self._pending += text
        partial = PARTIAL_CITATION_RE.search(self._pending)
        cut = partial.start() if partial else len(self._pending)
        ready, self._pending = self._pending[:cut], self._pending[cut:]
        return CITATION_RE.sub(self._replace, ready)

    def flush(self) -> str:
        """Return the text still held back, once the stream has ended."""
        ready, self._pending = self._pending, ""
        return CITATION_RE.sub(self._replace, ready)

    def _replace(self, match) -> str:
        new_ids = []
        for value in match.group(1).split(","):
            old_id = int(value)
            if 1 <= old_id <= len(self.sources):
                new_id = self._new_id(old_id)
                if new_id not in new_ids:
                    new_ids.append(new_id)
        # Citations to excerpts that do not exist are left unchanged
        return "[" + ", ".join(map(str, new_ids)) + "]" if new_ids else match.group(0)

    def _new_id(self, old_id: int) -> int:
        if old_id not in self._new_ids:
            node = self.sources[old_id - 1]
            meta = node.node.metadata
            key = tuple(str(meta.get(field) or "").strip().casefold() for field in ("authors", "year", "title", "source", "section", "subsection", "page"))
            if key not in self._key_ids:
                self.unique_sources.append(node)
                self._key_ids[key] = len(self.unique_sources)
            self._new_ids[old_id] = self._key_ids[key]
        return self._new_ids[old_id]


def build_context(nodes) -> str:
    context = "\n\n".join(f"[{i}] {source_line(n.node.metadata)}\n{n.node.get_content()}" for i, n in enumerate(nodes, 1))
    return context


SOURCE_FIELDS = ("paper_id", "title", "authors", "year", "section", "subsection", "page", "source")


def source_info(node, source_id: int) -> dict:
    """A retrieved node as a plain dict, numbered as in the prompt."""
    meta = node.node.metadata
    return {"id": source_id, **{f: meta.get(f) for f in SOURCE_FIELDS},
            "excerpt": node.node.get_content(), "score": None if node.score is None else float(node.score)}


def answer_events(query: str, paper_id: str | None = None, section_type: str | None = None,
                  year_min: int | None = None, renumber: bool = True):
    """Answer a question as a stream of events (dicts with a "type" key), in this order:

    - status:  {"stage": "searching" | "reranking" | "writing"}
    - sources: {"sources": [...]}, the reranked excerpts numbered 1..n as given to the model
    - token:   {"text": ...}, answer text; citations are renumbered unless renumber is False
    - done:    {"cited": [...]}, the cited sources. With renumber, "id" is the new number and
               "retrieved_id" the number in the sources event; without it, "id" is the model's number.

    If nothing matches, sources is empty and a single token carries the message.
    """
    yield {"type": "status", "stage": "searching"}
    nodes = hybrid_search(query, paper_id, section_type, year_min)
    if nodes:
        yield {"type": "status", "stage": "reranking"}
        nodes = reranker().postprocess_nodes(nodes, query_str=query)
    sources = [source_info(n, i) for i, n in enumerate(nodes, 1)]
    yield {"type": "sources", "sources": sources}
    if not nodes:
        yield {"type": "token", "text": "No indexed chunks matched."}
        yield {"type": "done", "cited": []}
        return

    yield {"type": "status", "stage": "writing"}
    tokens = answer_from_nodes(query, nodes)
    if not renumber:
        answer = ""
        for token in tokens:
            answer += token
            yield {"type": "token", "text": token}
        cited = [sources[i - 1] for i in extract_citations(answer) if 1 <= i <= len(sources)]
        yield {"type": "done", "cited": cited}
        return

    renumberer = CitationRenumberer(nodes)
    for token in tokens:
        if text := renumberer.feed(token):
            yield {"type": "token", "text": text}
    if text := renumberer.flush():
        yield {"type": "token", "text": text}
    retrieved_id = {id(n): i for i, n in enumerate(nodes, 1)}
    cited = [{**sources[retrieved_id[id(n)] - 1], "id": new_id, "retrieved_id": retrieved_id[id(n)]}
             for new_id, n in enumerate(renumberer.unique_sources, 1)]
    yield {"type": "done", "cited": cited}


def answer_from_nodes(query: str, nodes):
    no_think_suffix = " /no_think" if config.LLM_MODEL.startswith("qwen3") else ""
    messages = [ChatMessage(role="system", content=SYSTEM), 
                ChatMessage(role="user", content=f"Excerpts:\n{build_context(nodes)}\n\nQuestion: {query}{no_think_suffix}")]
    
    stream = get_llm().stream_chat(messages)
    
    return (chunk.delta for chunk in stream if chunk.delta)
