from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field, fields
from datetime import date
from pathlib import Path

from functools import lru_cache
from llama_index.llms.ollama import Ollama

import config

DOI_RE = re.compile(r"(?<!\d)(10\.\s*(?:\d\s*){4,9}\s*/\s*[^\s<>\"']+)", re.I)
ARXIV_RE = re.compile(r"arXiv:\s?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_Y = r"(?<!\d)((?:19|20)\d{2})(?!\d)"
YEAR_PATTERNS = (rf"date\s*of\s*publication[^.]{{0,40}}?{_Y}", 
                 rf"published[:\s][^.\n]{{0,40}}?{_Y}", 
                 rf"©\s*{_Y}")
_BANNER_WORDS = ("research article", "ieee", "arxiv", "preprint", "proceedings", "journal", "conference", 
                 "transactions", "elsevier", "springer", "open access", "manuscript")


@dataclass
class PaperMeta:
    paper_id: str
    title: str = ""
    authors: str = ""
    year: int = 0
    doi: str = ""
    arxiv_id: str = ""
    source: str = ""
    content_hash: str = ""
    field_sources: dict[str, str] = field(default_factory=dict)


def _meta_from_dict(d: dict) -> PaperMeta:
    known = {f.name for f in fields(PaperMeta)}
    return PaperMeta(**{k: v for k, v in d.items() if k in known})


def clean_doi(raw: str) -> str:
    raw = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", raw.strip(), flags=re.I)
    raw = re.sub(r"\s+", "", raw)
    raw = raw.rstrip(".,;:)]}>").lower()
    return raw


def clean_arxiv_id(raw: str) -> str:
    raw = re.sub(r"^(https?://arxiv\.org/(abs|pdf)/|arxiv:\s*)", "", raw.strip(), flags=re.I)
    raw = re.sub(r"\.pdf$", "", raw, flags=re.I)
    return re.sub(r"v\d+$", "", raw)


def find_doi(text: str) -> str:
    match = DOI_RE.search(text)
    return clean_doi(match.group(1)) if match else ""


def find_arxiv_id(text: str) -> str:
    match = ARXIV_RE.search(text)
    return match.group(1) if match else ""


def find_year(text: str) -> int:
    for pat in YEAR_PATTERNS:
        if match := re.search(pat, text, re.I):
            y = int(match.group(1))
            if 1900 <= y <= date.today().year + 1:
                return y
    return 0


def plausible_title(text: str, strict: bool = False) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    words = t.split()
    if not (10 <= len(t) <= 300 and 3 <= len(words) <= 40):
        return ""
    if sum(c.isalpha() for c in t) / len(t) < 0.6:
        return ""
    if strict and len(words) <= 6 and any(b in t.lower() for b in _BANNER_WORDS):
        return ""
    return t


def make_paper_id(doi: str, arxiv_id: str, content_hash: str) -> str:
    if doi:
        return f"doi:{doi}"
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    return f"sha:{content_hash[:16]}"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def _norm_ws(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _first_page(path: Path) -> tuple[str, str]:
    try:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            if doc.page_count == 0:
                return "", ""
            page = doc[0]
            text = page.get_text("text")
            height = page.rect.height
            spans = [(s["size"], s["bbox"][1], s["bbox"][0], s["text"].strip()) 
                     for b in page.get_text("dict")["blocks"] 
                     for ln in b.get("lines", []) 
                     for s in ln["spans"] 
                     if s["text"].strip() and s["bbox"][1] < 0.55 * height]
    except Exception as e:
        print(f"  warning: could not read first page of {Path(path).name}: {e}")
        return "", ""
    title = ""
    if spans:
        top = max(sz for sz, *_ in spans)
        lines = sorted((s for s in spans if s[0] >= top - 0.5), key=lambda s: (round(s[1] / 3), s[2]))
        title = " ".join(s[3] for s in lines)
    return text, title


# External metadata lookups
def _http_get(url: str, timeout: int = 10) -> bytes:
    ua = "paper-rag/0.1" + (f" (mailto:{config.CONTACT_EMAIL})" if config.CONTACT_EMAIL else "")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=timeout) as r:
        return r.read()


def _crossref(doi: str) -> dict:
    try:
        msg = json.loads(_http_get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")))["message"]
        authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) or a.get("name", "") for a in msg.get("author", [])]
        year = 0
        for key in ("issued", "published", "published-print", "published-online"):
            parts = (msg.get(key) or {}).get("date-parts") or [[None]]
            if parts[0] and parts[0][0]:
                year = int(parts[0][0])
                break
        return {"title": _strip_tags((msg.get("title") or [""])[0]), "authors": "; ".join(a for a in authors if a), "year": year}
    except Exception:
        return {}


def _arxiv(arxiv_id: str) -> dict:
    try:
        root = ET.fromstring(_http_get(f"http://export.arxiv.org/api/query?id_list={arxiv_id}"))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        return {"title": re.sub(r"\s+", " ", entry.findtext("a:title", "", ns)).strip(), 
                "authors": "; ".join(a.findtext("a:name", "", ns) for a in entry.findall("a:author", ns)), 
                "year": int(entry.findtext("a:published", "0000", ns)[:4])}
    except Exception:
        return {}


@lru_cache
def get_llm():
    return Ollama(model=config.LLM_MODEL, request_timeout=300.0, context_window=config.CONTEXT_WINDOW, temperature=0.1)


def _llm_extract(first_page: str) -> dict:
    no_think_suffix = " /no_think" if config.LLM_MODEL.startswith("qwen3") else ""

    prompt = ("Extract bibliographic metadata from the first page of a research paper.\n" 
              'Return ONLY JSON: {"title": str, "authors": [str], "year": int}\n' 
              "- authors: people only (no affiliations), as 'First Last' in Title Case even if printed in ALL CAPS.\n" 
              "- title: the paper title only, in normal title case.\n" 
              "- year: the publication year (not received/accepted dates unless nothing else is given).\n\n" 
              f"FIRST PAGE:\n{first_page[:3500]}{no_think_suffix}")
    try:
        raw = re.sub(r"<think>.*?</think>", "", str(get_llm().complete(prompt)), flags=re.S)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception:
        return {}
    return _validate(data)


def _validate(data: dict) -> dict:
    out = {}
    if title := plausible_title(str(data.get("title") or "")):
        out["title"] = title
    authors = data.get("authors") or []
    if isinstance(authors, list):
        authors = [a.strip() for a in authors if isinstance(a, str) and 2 < len(a.strip()) < 60]
        if authors and not any(w in " ".join(authors).lower() for w in ("university", "institute", "department")):
            out["authors"] = "; ".join(authors)
    try:
        y = int(data.get("year") or 0)
        if 1900 <= y <= date.today().year + 1:
            out["year"] = y
    except (TypeError, ValueError):
        pass
    return out


# Extract metadata
def _extract(path: Path, content_hash: str) -> PaperMeta:
    text, layout_title = _first_page(path)
    doi, arxiv_id = find_doi(text), find_arxiv_id(text)
    values: dict = {"title": "", "authors": "", "year": 0}
    sources: dict = {}
    if doi:
        sources["doi"] = "pdf"
    if arxiv_id:
        sources["arxiv_id"] = "pdf"

    def fill(src: str, found: dict):
        for k in values:
            if not values[k] and found.get(k):
                values[k], sources[k] = found[k], src

    if config.ALLOW_NETWORK:
        if doi:
            fill("crossref", _crossref(doi))
        if arxiv_id and not all(values.values()):
            fill("arxiv", _arxiv(arxiv_id))
    if config.USE_LLM_META and not all(values.values()):
        fill("llm", _llm_extract(text))
    fill("layout", {"title": plausible_title(layout_title, strict=True)})
    fill("regex", {"year": find_year(text)})

    return PaperMeta(paper_id=make_paper_id(doi, arxiv_id, content_hash), 
                     doi=doi, arxiv_id=arxiv_id, source=path.name, content_hash=content_hash, 
                     field_sources=sources, **values)


# Override metadata with sidecar file if present
def _apply_sidecar(meta: PaperMeta, path: Path) -> PaperMeta:
    side = path.with_suffix(".meta.json")
    if not side.exists():
        return meta
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {side.name}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{side.name} must contain a JSON object")

    def put(name: str, value):
        setattr(meta, name, value)
        meta.field_sources[name] = "sidecar"

    if data.get("title"):
        put("title", _norm_ws(data["title"]))
    if data.get("doi"):
        put("doi", clean_doi(str(data["doi"])))
    if data.get("arxiv_id"):
        put("arxiv_id", clean_arxiv_id(str(data["arxiv_id"])))
    if data.get("authors"):
        a = data["authors"]
        names = a if isinstance(a, list) else str(a).split(";")
        put("authors", "; ".join(n for n in (_norm_ws(x) for x in names) if n))
    if data.get("year"):
        try:
            y = int(data["year"])
        except (TypeError, ValueError):
            raise ValueError(f"{side.name}: year must be an integer, got {data['year']!r}") from None
        if not 1900 <= y <= date.today().year + 1:
            raise ValueError(f"{side.name}: year {y} is out of range")
        put("year", y)
    meta.paper_id = make_paper_id(meta.doi, meta.arxiv_id, meta.content_hash)
    return meta


# Save/load metadata
def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _save_final(meta: PaperMeta) -> None:
    reg = {k: v for k, v in _read_json(config.REGISTRY_PATH).items() if v.get("content_hash") != meta.content_hash}
    reg[meta.paper_id] = asdict(meta)
    _write_json(config.REGISTRY_PATH, reg)


def resolve_metadata(path: Path, refresh: bool = False) -> PaperMeta:
    path = Path(path)
    content_hash = file_sha256(path)
    cache = _read_json(config.META_CACHE_PATH)
    if content_hash in cache and not refresh:
        meta = _meta_from_dict(cache[content_hash])
    else:
        meta = _extract(path, content_hash)
        cache[content_hash] = asdict(meta)
        _write_json(config.META_CACHE_PATH, cache)
    meta.source = path.name
    meta = _apply_sidecar(meta, path)
    _save_final(meta)
    return meta


def find_paper(ref: str) -> str:
    reg = _read_json(config.REGISTRY_PATH)
    if ref in reg:
        return ref
    low = ref.lower()
    hits = [pid for pid, m in reg.items() if low in (m["source"].lower(), Path(m["source"]).stem.lower())]
    hits = hits or [pid for pid, m in reg.items() if low in m.get("title", "").lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"No paper matches {ref!r}. Run `list` to see indexed papers.")
    raise ValueError(f"{ref!r} is ambiguous: {', '.join(hits)}")
