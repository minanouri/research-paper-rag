import argparse
import sys
from pathlib import Path

from . import config
from .sections import SECTION_TYPES


def cmd_ingest(args):
    from .index import add_paper
    from .ingest import parse_pdf

    target = Path(args.path)
    pdfs = sorted(target.glob("*.pdf")) if target.is_dir() else [target]
    if not pdfs:
        sys.exit(f"No PDFs found in {target}")
    failed = []
    for pdf in pdfs:
        print(f"Parsing {pdf.name} ...")
        try:
            meta, nodes = parse_pdf(pdf, refresh_meta=args.refresh_meta)
            src = meta.field_sources
            print(f"  id:      {meta.paper_id}")
            print(f"  title:   {meta.title or '(none found)'}  [{src.get('title', '-')}]")
            print(f"  authors: {meta.authors or '(none found)'}  [{src.get('authors', '-')}]")
            print(f"  year:    {meta.year or '?'}  [{src.get('year', '-')}]   doi: {meta.doi or '-'}")
            if src.get("title") in ("layout", None):
                print("  WARNING: title is extracted from layout or missing; consider adding a .meta.json file next to the PDF to override")
            print(f"  indexing {len(nodes)} chunks ...")
            add_paper(meta, nodes)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(pdf.name)
    if failed:
        print(f"\n{len(failed)} file(s) failed: {', '.join(failed)}")
        return 1
    print("Paper(s) ingested and indexed successfully.")


def cmd_ask(args):
    from .generate import ask, source_line, extract_citations, deduplicate_citations, remap_citations
    from .metadata import find_paper

    try:
        paper_id = find_paper(args.paper) if args.paper else None
    except ValueError as e:
        sys.exit(str(e))

    tokens, sources = ask(args.question, paper_id, args.section_type, args.year_min)

    answer = "".join(tokens for tokens in tokens)
    cited_ids = extract_citations(answer)
    valid_ids = [i for i in cited_ids if 1 <= i <= len(sources)]

    if args.show_retrieved:
        print(answer)
        print("\nRetrieved sources:")

        for i, node in enumerate(sources, 1):
            status = " (cited)" if i in valid_ids else " (not cited)"
            print(f"  [{i}] {source_line(node.node.metadata)}{status}")

    else:
        unique_sources, citation_map = deduplicate_citations(sources, valid_ids)
        answer = remap_citations(answer, citation_map)

        print(answer)
        print("\nCited sources:")

        if not unique_sources:
            print("  No valid inline citations were found in the answer.")
            return

        for new_id, node in enumerate(unique_sources, 1):
            print(f"  [{new_id}] {source_line(node.node.metadata)}")


def cmd_list(_):
    from .index import list_papers

    for m in sorted(list_papers(), key=lambda m: (m.get("year", 0), m["paper_id"])):
        print(f"{m['paper_id']}  {m.get('year') or '?'}  {m.get('title') or m['source']}  ({m['chunks']} chunks)")


def main():
    p = argparse.ArgumentParser(prog="paper-rag")
    sub = p.add_subparsers(required=True)

    a = sub.add_parser("ingest", help="Ingest a PDF or a folder of PDFs")
    a.add_argument("path", nargs="?", default=str(config.PDF_DIR))
    a.add_argument("--refresh-meta", action="store_true", help="Re-run Crossref/LLM metadata extraction")
    a.set_defaults(fn=cmd_ingest)

    q = sub.add_parser("ask", help="Ask a question")
    q.add_argument("question")
    q.add_argument("--paper", help="paper_id, PDF filename, or unique title substring")
    q.add_argument("--section-type", choices=SECTION_TYPES, help="Restrict to a section type, e.g. methods")
    q.add_argument("--year-min", type=int, help="Only papers published in or after this year")
    q.add_argument("--show-retrieved", action="store_true", help="Show all retrieved sources, not just cited ones")
    q.set_defaults(fn=cmd_ask)

    l = sub.add_parser("list", help="List indexed papers")
    l.set_defaults(fn=cmd_list)

    args = p.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
