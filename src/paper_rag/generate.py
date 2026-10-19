from llama_index.core.llms import ChatMessage
import re

from .metadata import get_llm
from .retrieve import retrieve
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


def extract_citations(answer: str) -> list[int]:
    found = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer)
    citation_ids = []
    seen = set()

    for group in found:
        for value in group.split(","):
            citation_id = int(value.strip())
            if citation_id not in seen:
                seen.add(citation_id)
                citation_ids.append(citation_id)

    return citation_ids


def deduplicate_citations(sources, cited_ids: list[int]):
    unique_sources = []
    citation_map = {}
    metadata_keys = {}

    for old_id in cited_ids:
        node = sources[old_id - 1]
        meta = node.node.metadata

        key = tuple(str(meta.get(field) or "").strip().casefold() for field in ("authors", "year", "title", "source", "section", "subsection", "page"))

        if key not in metadata_keys:
            new_id = len(unique_sources) + 1
            metadata_keys[key] = new_id
            unique_sources.append(node)

        citation_map[old_id] = metadata_keys[key]

    return unique_sources, citation_map


def remap_citations(answer: str, citation_map: dict[int, int]) -> str:
    def replace(match):
        old_ids = [int(value.strip()) for value in match.group(1).split(",")]

        new_ids = []
        for old_id in old_ids:
            new_id = citation_map.get(old_id)
            if new_id is not None and new_id not in new_ids:
                new_ids.append(new_id)

        if not new_ids:
            return match.group(0)

        return "[" + ", ".join(map(str, new_ids)) + "]"

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", replace, answer)


def build_context(nodes) -> str:
    context = "\n\n".join(f"[{i}] {source_line(n.node.metadata)}\n{n.node.get_content()}" for i, n in enumerate(nodes, 1))
    return context


def ask(query: str, paper_id: str | None = None, section_type: str | None = None, year_min: int | None = None):
    nodes = retrieve(query, paper_id, section_type, year_min)
    if not nodes:
        return iter(["No indexed chunks matched."]), []

    no_think_suffix = " /no_think" if config.LLM_MODEL.startswith("qwen3") else ""
    messages = [ChatMessage(role="system", content=SYSTEM), 
                ChatMessage(role="user", content=f"Excerpts:\n{build_context(nodes)}\n\nQuestion: {query}{no_think_suffix}")]
    
    stream = get_llm().stream_chat(messages)
    
    return (chunk.delta for chunk in stream if chunk.delta), nodes
