import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

from . import config

QUESTIONS_PATH = config.ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = config.ROOT / "eval" / "results"
CONFIG_KEYS = ("EMBED_MODEL", "RERANK_MODEL", "LLM_MODEL", "USE_FP16", "CHUNK_SIZE", "CHUNK_OVERLAP",
               "TOP_K_DENSE", "TOP_K_BM25", "FUSION_TOP_K", "RERANK_TOP_N")

# Phrases an answer uses when the excerpts do not contain the answer
ABSTAIN_RE = re.compile(r"(do(es)? not (provide|contain|mention|include|specify|state|report)"
                        r"|not (mentioned|specified|provided|stated|reported|available)"
                        r"|no (information|mention|details?)|insufficient|not enough (information|evidence)"
                        r"|cannot be (determined|answered))", re.I)


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


def load_questions(path: Path) -> list[dict]:
    from .metadata import find_paper

    questions = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
        if not q.get("q"):
            raise ValueError(f"{path}:{line_no}: missing 'q'")
        q.setdefault("id", f"line{line_no}")
        q.setdefault("type", "other")
        q.setdefault("filters", {})
        q["evidence"] = [{"paper": find_paper(e["paper"]), "quote": e["quote"]} for e in q.get("evidence", [])]
        if q["filters"].get("paper"):
            q["filters"]["paper_id"] = find_paper(q["filters"].pop("paper"))
        questions.append(q)
    return questions


def _quotes(e: dict) -> list[str]:
    return e["quote"] if isinstance(e["quote"], list) else [e["quote"]]


def _matched(node, evidence: list[dict]) -> set[int]:
    meta, text = node.metadata, _norm(node.get_content())
    return {i for i, e in enumerate(evidence)
            if meta.get("paper_id") == e["paper"] and any(_norm(q) in text for q in _quotes(e))}


def check_quotes(questions: list[dict]) -> list[str]:
    from .index import load_docstore

    docs = list(load_docstore().docs.values())
    warnings = []
    for q in questions:
        for e in q["evidence"]:
            for quote in _quotes(e):
                if not any(_matched(n, [{"paper": e["paper"], "quote": quote}]) for n in docs):
                    warnings.append(f"{q['id']}: quote not found in {e['paper']}: {quote!r}")
    return warnings


def _stage(nodes, evidence: list[dict], k: int) -> dict:
    found, first_rank = set(), None
    for rank, n in enumerate(nodes[:k], 1):
        hits = _matched(n.node, evidence)
        if hits and first_rank is None:
            first_rank = rank
        found |= hits
    return {"recall": len(found) / len(evidence), "first_rank": first_rank}


def _evaluate_answer(q: dict, nodes) -> dict:
    from .generate import answer_from_nodes, extract_citations

    text = "".join(answer_from_nodes(q["q"], nodes)) if nodes else ""
    cited = [i for i in extract_citations(text) if 1 <= i <= len(nodes)]
    relevant_cited = [i for i in cited if _matched(nodes[i - 1].node, q["evidence"])]
    return {"answer": text,
            "cited": cited,
            "citation_precision": len(relevant_cited) / len(cited) if cited else None,
            "cites_evidence": bool(relevant_cited),
            "abstained": bool(ABSTAIN_RE.search(text))}


def evaluate(questions: list[dict], with_answers: bool = False, progress=print) -> dict:
    from .retrieve import build_retrievers, reranker

    stage_k = {"dense": config.TOP_K_DENSE, "bm25": config.TOP_K_BM25,
               "fusion": config.FUSION_TOP_K, "rerank": config.RERANK_TOP_N}
    rows = []
    for n, q in enumerate(questions, 1):
        progress(f"  [{n}/{len(questions)}] {q['id']}")
        f = q["filters"]
        retrievers = build_retrievers(f.get("paper_id"), f.get("section_type"), f.get("year_min"))
        row = {"id": q["id"], "q": q["q"], "type": q["type"], "stages": {}}
        reranked = []
        if retrievers:
            dense, bm25, fusion = retrievers
            fused = fusion.retrieve(q["q"])
            results = {"dense": dense.retrieve(q["q"]), "bm25": bm25.retrieve(q["q"]), "fusion": fused}
            reranked = reranker().postprocess_nodes(fused, query_str=q["q"])
            results["rerank"] = reranked
            if q["evidence"]:
                row["stages"] = {s: _stage(results[s], q["evidence"], stage_k[s]) for s in stage_k}
        if with_answers:
            row.update(_evaluate_answer(q, reranked))
        rows.append(row)
    return {"time": datetime.now().isoformat(timespec="seconds"),
            "config": {k: getattr(config, k) for k in CONFIG_KEYS},
            "stage_k": stage_k,
            "rows": rows}


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(v) -> str:
    return "  -  " if v is None else f"{v:.2f}"


def print_report(result: dict) -> None:
    rows, stage_k = result["rows"], result["stage_k"]
    answerable = [r for r in rows if r["stages"]]
    print(f"\nRetrieval: {len(answerable)} answerable questions")
    print(f"  {'stage':8s} {'k':>3s}  {'recall':>6s}  {'MRR':>5s}")
    for s, k in stage_k.items():
        recall = _mean(r["stages"][s]["recall"] for r in answerable)
        mrr = _mean(1 / r["stages"][s]["first_rank"] if r["stages"][s]["first_rank"] else 0 for r in answerable)
        print(f"  {s:8s} {k:3d}  {_fmt(recall):>6s}  {_fmt(mrr):>5s}")

    types = sorted({r["type"] for r in answerable})
    print(f"\nRecall by question type ({' / '.join(stage_k)})")
    for t in types:
        group = [r for r in answerable if r["type"] == t]
        scores = "  ".join(_fmt(_mean(r["stages"][s]["recall"] for r in group)) for s in stage_k)
        print(f"  {t:14s} n={len(group):<3d} {scores}")

    misses = [r for r in answerable if r["stages"]["rerank"]["recall"] < 1]
    print(f"\nQuestions missing evidence after reranking: {len(misses)}")
    for r in misses:
        flags = " ".join(f"{s}:{r['stages'][s]['recall']:.0%}" for s in stage_k)
        print(f"  {r['id']:16s} {flags}  | {r['q'][:70]}")

    if "answer" in rows[0]:
        print("\nAnswers")
        print(f"  citation precision (answerable): {_fmt(_mean(r['citation_precision'] for r in answerable))}")
        print(f"  cites the evidence (answerable): {_fmt(_mean(float(r['cites_evidence']) for r in answerable))}")
        unanswerable = [r for r in rows if not r["stages"]]
        if unanswerable:
            declined = _mean(float(r["abstained"]) for r in unanswerable)
            print(f"  declined unanswerable (heuristic): {_fmt(declined)}  (n={len(unanswerable)})")


def save_result(result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
