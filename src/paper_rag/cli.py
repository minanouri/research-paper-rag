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


def _resolve_paper(ref: str | None) -> str | None:
    from .metadata import find_paper

    try:
        return find_paper(ref) if ref else None
    except ValueError as e:
        sys.exit(str(e))


def answer_question(question: str, paper_id: str | None, args) -> None:
    from .generate import ask, source_line, extract_citations, deduplicate_citations, remap_citations

    tokens, sources = ask(question, paper_id, args.section_type, args.year_min)

    answer = "".join(tokens)
    if not sources:
        print(answer)
        return
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


def cmd_ask(args):
    answer_question(args.question, _resolve_paper(args.paper), args)


def cmd_chat(args):
    from .index import embed_model
    from .retrieve import reranker

    paper_id = _resolve_paper(args.paper)
    print("Loading models ...")
    embed_model()
    reranker()
    print("Ready. Ask a question, or type 'exit' to quit (Ctrl-C stops an answer).")

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        try:
            answer_question(question, paper_id, args)
        except KeyboardInterrupt:
            print("\n(answer stopped)")
        except Exception as e:
            print(f"Error: {e}")


def cmd_eval(args):
    from .evaluate import check_quotes, evaluate, load_questions, print_report, save_result

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError) as e:
        sys.exit(str(e))
    warnings = check_quotes(questions)
    for w in warnings:
        print(f"WARNING: {w}")
    if warnings and not args.ignore_missing:
        sys.exit("Fix the quotes above (or pass --ignore-missing); they can never be matched.")

    print(f"Evaluating {len(questions)} questions{' with answers' if args.answers else ''} ...")
    result = evaluate(questions, with_answers=args.answers)
    print_report(result)
    if not args.no_save:
        print(f"\nSaved to {save_result(result)}")


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

    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("--paper", help="paper_id, PDF filename, or unique title substring")
    filters.add_argument("--section-type", choices=SECTION_TYPES, help="Restrict to a section type, e.g. methods")
    filters.add_argument("--year-min", type=int, help="Only papers published in or after this year")
    filters.add_argument("--show-retrieved", action="store_true", help="Show all retrieved sources, not just cited ones")

    q = sub.add_parser("ask", parents=[filters], help="Ask a question")
    q.add_argument("question")
    q.set_defaults(fn=cmd_ask)

    c = sub.add_parser("chat", parents=[filters], help="Ask several questions, loading the models only once")
    c.set_defaults(fn=cmd_chat)

    e = sub.add_parser("eval", help="Evaluate retrieval against a labeled question set")
    e.add_argument("--questions", default=str(config.ROOT / "eval" / "questions.jsonl"), help="Question file (JSONL)")
    e.add_argument("--answers", action="store_true", help="Also generate answers and score citations")
    e.add_argument("--ignore-missing", action="store_true", help="Run even if some evidence quotes are not found")
    e.add_argument("--no-save", action="store_true", help="Do not save results to eval/results/")
    e.set_defaults(fn=cmd_eval)

    l = sub.add_parser("list", help="List indexed papers")
    l.set_defaults(fn=cmd_list)

    args = p.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
