from llama_index.core.llms import ChatMessage

from metadata import get_llm, cite_label
from retrieve import retrieve
import config


SYSTEM = """You answer questions about research papers using ONLY the numbered excerpts provided in the user's message."""


def build_context(nodes) -> str:
    context = "\n\n".join(f"[{i}] {n.node.get_content()}" for i, n in enumerate(nodes, 1))
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
