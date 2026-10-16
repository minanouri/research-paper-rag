from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import config


SECTION_TYPES = ("front_matter", "abstract", "introduction", "related_work", "background", 
                 "methods", "experiments", "discussion", "conclusion", "appendix", "other")
FRONT_MATTER = "Front matter"

_ROMAN = r"(?=[IVX])X{0,3}(?:IX|IV|V?I{0,3})"
RE_ROMAN = re.compile(rf"^\s*({_ROMAN})\.\s+(\S.*)$")
RE_LETTER = re.compile(r"^\s*([A-Z])\.\s+(\S.*)$")
RE_LETTER_DECIMAL = re.compile(r"^\s*([A-Z](?:\.\d+)+)\.?\s+(\S.*)$")
RE_DECIMAL = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
RE_PAREN = re.compile(r"^\s*(\d+)\)\s+(\S.*)$")
RE_APPENDIX = re.compile(r"^\s*(?:[A-Z]\.?\s+)?(?:appendix|appendices|supplementary|supplemental)\b", re.I)

TOP_LEVEL_NAMES = ("abstract", "introduction", "motivation", "background", "preliminaries", "related work", 
                   "related works", "literature review", "methods", "method", "methodology", "materials and methods", 
                   "approach", "experiments", "experimental setup", "experimental evaluation", "evaluation", 
                   "results", "results and discussion", "discussion", "limitations", "conclusion", "conclusions", 
                   "future work", "acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements", 
                   "references", "bibliography", "appendix", "appendices", "supplementary material", 
                   "supplementary materials", "declaration of competing interest", "declarations of competing interest", 
                   "conflict of interest", "conflicts of interest", "credit authorship contribution statement", 
                   "author contributions", "data availability", "funding")

_TYPE_KEYWORDS = (("abstract", ("abstract",)), 
                  ("appendix", ("appendix", "appendices", "supplementary", "supplemental")),
                  ("introduction", ("introduction", "motivation")), 
                  ("related_work", ("related work", "literature", "prior work", "previous work", "state of the art")), 
                  ("background", ("preliminar", "background", "foundation", "notation")), 
                  ("conclusion", ("conclusion", "concluding", "future work", "future research", "future direction", "summary")), 
                  ("experiments", ("experiment", "evaluation", "implementation", "result", "empirical", "case study", 
                                   "dataset", "performance", "benchmark", "ablation")), 
                  ("discussion", ("discussion", "limitation", "implication", "threats to validity")), 
                  ("methods", ("method", "approach", "model", "proposed", "framework", "architecture", "algorithm", 
                               "formulation", "design", "materials")))


def normalize_name(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def _squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


_NORM_TOP = tuple(normalize_name(n) for n in TOP_LEVEL_NAMES)
_SQUASHED_NAMES = {_squash(n): n for n in (*TOP_LEVEL_NAMES, "case study", "problem formulation")}

_SMALL = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "via", "vs", "with"}
_TOKEN = re.compile(r"[^\W\d_][^\W_]*(?:-[^\W_]+)*")


def build_casing_vocab(texts: Iterable[str]) -> dict[str, str]:
    forms: dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        for line in clean_chunk_text(text).splitlines():
            letters = [c for c in line if c.isalpha()]
            if len(letters) >= 4 and sum(c.isupper() for c in letters) / len(letters) >= 0.8:
                continue
            for tok in _TOKEN.findall(line):
                forms[tok.lower()][tok] += 1
    vocab: dict[str, str] = {}
    for low, counter in forms.items():
        form, n = counter.most_common(1)[0]
        has_inner_capital = any(c.isupper() for c in form[1:])
        is_term = has_inner_capital and (form != form.upper() or n >= 2 or len(form) <= 4)
        vocab[low] = form if is_term else ""
    return vocab


def _recase(core: str, vocab: dict[str, str] | None, part: bool = False) -> str:
    low = core.lower()
    if vocab and low in vocab:
        return vocab[low] or "-".join(part.capitalize() for part in core.split("-"))
    if "-" in core:
        return "-".join(_recase(p, vocab, part=True) for p in core.split("-"))
    if part or len(core) <= 3 or any(c.isdigit() for c in core):
        return core
    return core.capitalize()


def pretty_label(label: str, vocab: dict[str, str] | None = None) -> str:
    label = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", label)).strip().rstrip(".:")
    letters = [c for c in label if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) < 0.8:
        return label
    out = []
    for i, w in enumerate(label.split()):
        core = w.strip("()[],:;")
        if not core:
            out.append(w)
            continue
        if core != core.upper():
            new = core
        elif core.lower() in _SMALL and i > 0:
            new = core.lower()
        else:
            new = _recase(core, vocab)
        out.append(w.replace(core, new, 1))
    return " ".join(out)


def restore_spacing(label: str) -> str:
    raw = label.strip()
    name = _SQUASHED_NAMES.get(_squash(raw))
    if name and " " in name and not re.search(r"\s", raw):
        return pretty_label(name.upper())
    return raw


def roman_to_int(value: str) -> int:
    value = value.upper()
    if not re.fullmatch(_ROMAN, value):
        raise ValueError(f"unsupported Roman numeral: {value!r}")
    vals = {"I": 1, "V": 5, "X": 10}
    total = 0
    for i, ch in enumerate(value):
        total += -vals[ch] if i + 1 < len(value) and vals[ch] < vals[value[i + 1]] else vals[ch]
    return total


def _is_known_top_level(text: str) -> bool:
    norm = normalize_name(text)
    if any(norm == n or norm.startswith(n + " and ") for n in _NORM_TOP):
        return True
    head = re.split(r"[:\u2013\u2014.]", text, maxsplit=1)[0]
    return head != text and normalize_name(head) in _NORM_TOP


def classify_heading(text: str, roman_idx: int = 0) -> tuple[int, str, int]:
    raw = text.strip()
    if not raw:
        return 0, "", 0
    if m := RE_PAREN.match(raw):
        return 3, m.group(2).strip(), 0
    if m := RE_DECIMAL.match(raw):
        return min(m.group(1).count(".") + 1, 3), m.group(2).strip(), 0
    if m := RE_ROMAN.match(raw):
        numeral, label = m.groups()
        value = roman_to_int(numeral)
        if len(numeral) > 1 or value > roman_idx or _is_known_top_level(label):
            return 1, label.strip(), value
    if m := RE_LETTER_DECIMAL.match(raw):
        return min(m.group(1).count(".") + 1, 3), m.group(2).strip(), 0
    if RE_APPENDIX.match(raw):
        return 1, raw, 0
    if m := RE_LETTER.match(raw):
        return 2, m.group(2).strip(), 0
    if _is_known_top_level(raw):
        return 1, raw, 0
    return 0, raw, 0


def section_type_of(*labels: str) -> str:
    for label in labels:
        low = label.lower()
        if not low.strip():
            continue
        squashed = re.sub(r"\s+", "", low)
        for stype, keys in _TYPE_KEYWORDS:
            if any(re.search(rf"\b{re.escape(k)}", low) or (" " in k and k.replace(" ", "") in squashed) for k in keys):
                return stype
    return "other"


def clean_chunk_text(text: str) -> str:
    text = (text or "").replace("<!-- formula-not-decoded -->", " ").replace("\u00a0", " ")
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _should_skip(label: str) -> bool:
    norm = normalize_name(label)
    for term in getattr(config, "SKIP_SECTIONS", ()):
        t = normalize_name(term) if isinstance(term, str) else ""
        if t and (norm == t or norm.startswith(t + " ")):
            return True
    return False


@dataclass
class SectionInfo:
    section: str
    subsection: str
    section_type: str
    skip: bool


class SectionTracker:
    def __init__(self, nested: bool | None = None, vocab: dict[str, str] | None = None):
        self.nested = nested
        self.vocab = vocab
        self.section = FRONT_MATTER
        self.sub = ""
        self.sub2 = ""
        self.roman = 0
        self.skipping = False
        self.last: list[str] = []

    def _fallback_level(self, index: int, nested: bool, page: int | None) -> int:
        if self.section == FRONT_MATTER:
            return 1 if (page or 0) >= 2 else 0
        return min(index + 1, 3) if nested else 2

    def _apply(self, heading: str, index: int, nested: bool, page: int | None) -> None:
        level, label, roman_value = classify_heading(heading, self.roman)
        label = pretty_label(restore_spacing(label), self.vocab)
        if not label:
            return
        if level == 0:
            level = self._fallback_level(index, nested, page)
            if level == 0:
                return
        if level == 1:
            self.section, self.sub, self.sub2 = label, "", ""
            self.roman = max(self.roman, roman_value)
            self.skipping = _should_skip(label)
        elif level == 2:
            self.sub, self.sub2 = label, ""
        else:
            self.sub2 = label

    def update(self, headings, text: str = "", page: int | None = None) -> SectionInfo:
        path = [h.strip() for h in (headings or []) if isinstance(h, str) and h.strip()]
        if self.nested is None and len(path) >= 2:
            self.nested = True
        nested = bool(self.nested)

        if path:
            common = 0
            while common < min(len(path), len(self.last)) and path[common] == self.last[common]:
                common += 1
            for index in range(common, len(path)):
                self._apply(path[index], index, nested, page)
            if nested and common == len(path) < len(self.last):
                if len(path) <= 1:
                    self.sub = self.sub2 = ""
                elif len(path) == 2:
                    self.sub2 = ""
            self.last = path
        return self._info(text)

    def _info(self, text: str) -> SectionInfo:
        if self.section == FRONT_MATTER:
            if re.match(r"\W*abstract\b", clean_chunk_text(text)[:60], re.I):
                return SectionInfo("Abstract", "", "abstract", False)
            return SectionInfo(FRONT_MATTER, "", "front_matter", False)
        sub = " > ".join(p for p in (self.sub, self.sub2) if p)
        return SectionInfo(self.section, sub, section_type_of(self.section, self.sub, self.sub2), self.skipping)
