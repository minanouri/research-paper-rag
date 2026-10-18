from llama_index.core.llms import ChatMessage

from metadata import get_llm, cite_label
from retrieve import retrieve
import config


SYSTEM = """You answer questions about research papers using ONLY the numbered excerpts provided in the user's message."""


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
    sec = meta.get("section", "")
    if meta.get("subsection"):
        sec += f" > {meta['subsection']}"
    source = f"{cite_label(meta.get('authors', ''), meta.get('year', 0))} | \"{meta.get('title') or meta.get('source')}\" | {sec} | p.{meta.get('page')}"
    return source


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
