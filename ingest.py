from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader

import config
from metadata import PaperMeta, resolve_metadata
from sections import SectionTracker, build_casing_vocab


EMBED_KEYS = {"title", "section", "subsection"}


def _first_page(dl_meta: dict) -> int:
    pages = [p["page_no"] for item in dl_meta.get("doc_items", []) for p in item.get("prov", []) if "page_no" in p]
    return min(pages) if pages else 0


def parse_pdf(path: Path, refresh_meta: bool = False) -> tuple[PaperMeta, list]:
    meta = resolve_metadata(path, refresh=refresh_meta)

    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    raw = DoclingNodeParser().get_nodes_from_documents(reader.load_data(str(path)))

    items = []
    for node in raw:
        dl = node.metadata
        items.append((node, dl.get("headings") or [], node.get_content(), _first_page(dl)))

    nested = any(len(headings) >= 2 for _, headings, _, _ in items)

    # Chunk-level metadata from the document structure
    vocab = build_casing_vocab(text for _, _, text, _ in items)
    tracker = SectionTracker(nested=nested, vocab=vocab)
    kept = []
    for node, headings, text, page in items:
        info = tracker.update(headings, text, page)
        if info.skip:
            continue
        node.metadata = {"section": info.section, 
                         "subsection": info.subsection, 
                         "section_type": info.section_type, 
                         "page": page}
        kept.append(node)

    # Split oversized chunks by sentence
    nodes = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)(kept)

    # Add paper-level metadata
    paper_fields = {"paper_id": meta.paper_id, "title": meta.title, "authors": meta.authors, "year": meta.year, 
                    "doi": meta.doi, "source": meta.source, "content_hash": meta.content_hash}
    for i, node in enumerate(nodes):
        node.metadata.update(paper_fields)
        node.metadata["chunk_index"] = i
        keys = list(node.metadata)
        node.excluded_embed_metadata_keys = [k for k in keys if k not in EMBED_KEYS]
        node.excluded_llm_metadata_keys = keys
    
    return meta, nodes
