"""Chunking + BM25 keyword retrieval over the scraped gnani.ai pages.

Why BM25 and not embeddings? The corpus is one company's website (~1,300 pages), its
vocabulary is very specific (product names such as Evon, Prisma, Timbre, Plexus), and
BM25 needs no model download, no vector DB and no second API key. It is pure Python,
deterministic and builds in seconds at startup. The LLM does the semantic work on top
of the retrieved passages.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher, get_close_matches
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import snowballstemmer

from app import config

STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself
    yourselves tell know please give get let us me show explain describe list also much many
    something anything thing things way ways""".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+")
TAG_LINE_RE = re.compile(r"^\s*(<[^>]+>\s*)+$")


_STEMMER = snowballstemmer.stemmer("english")  # 'pricing'/'prices' -> 'price', 'agents' -> 'agent'


@lru_cache(maxsize=200_000)
def _stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# How visitors phrase company questions vs. how the site words it ("Who founded Gnani?"
# vs. "Co-Founder & CEO"). Added to the query only, at reduced weight.
_RAW_SYNONYMS = {
    "founded": "founder", "founder": "founded", "started": "founder founded",
    "ceo": "founder leadership", "cto": "founder leadership", "leader": "leadership founder",
    "owner": "founder", "headquarters": "office", "located": "office", "location": "office",
    "contact": "email phone demo", "price": "pricing cost plan", "cost": "pricing price",
    "job": "careers hiring", "hiring": "careers", "vacancy": "careers",
}
QUERY_SYNONYMS = {_stem(k): tokenize(v) for k, v in _RAW_SYNONYMS.items()}
SYNONYM_WEIGHT = 0.6


@dataclass(frozen=True)
class Chunk:
    id: int
    url: str
    title: str
    heading: str
    text: str
    section: str


def _lines(text: str, max_words: int) -> list[str]:
    """Page lines, with any paragraph longer than max_words split at sentence ends."""
    out = []
    for line in text.splitlines():
        if TAG_LINE_RE.match(line):  # stray markup such as "</svg>" from page embeds
            continue
        if len(line.split()) <= max_words:
            out.append(line)
            continue
        piece: list[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if piece and len(" ".join(piece + [sent]).split()) > max_words:
                out.append(" ".join(piece))
                piece = []
            piece.append(sent)
        if piece:
            out.append(" ".join(piece))
    return out


def _split_page(page: dict, max_words: int, overlap: int) -> list[tuple[str, str]]:
    """Split one page into (heading, text) chunks of about max_words words.

    A new chunk starts at a major heading (## / ###) once the current chunk has some
    substance, or when the size limit is reached. In the latter case the last ~overlap
    words are carried over so a fact that straddles the boundary stays retrievable.
    """
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_words = 0
    heading = buf_heading = ""

    for line in _lines(page["text"], max_words):
        n = len(line.split())
        if line.startswith("#"):
            heading = line.lstrip("# ").strip()
            if line.startswith(("## ", "### ")) and buf_words >= max_words // 2:
                chunks.append((buf_heading, "\n".join(buf)))
                buf, buf_words = [], 0
            if not buf:
                buf_heading = heading
        elif buf and buf_words + n > max_words:
            chunks.append((buf_heading, "\n".join(buf)))
            tail, w = [], 0
            for prev in reversed(buf):
                if w >= overlap:
                    break
                tail.insert(0, prev)
                w += len(prev.split())
            buf, buf_words = (tail, w) if w <= max_words // 2 else ([], 0)
            buf_heading = heading
        buf.append(line)
        buf_words += n
    if buf:
        chunks.append((buf_heading, "\n".join(buf)))
    return [(h, t) for h, t in chunks if len(t.split()) >= 8]


class Retriever:
    def __init__(self, pages: list[dict], max_words: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP_WORDS):
        self.pages = pages
        self.chunks: list[Chunk] = []
        self.token_sets: list[frozenset[str]] = []  # for near-duplicate suppression
        for page in pages:
            # The site menu is kept whole: it is the company's table of contents.
            splits = [("", page["text"])] if page["section"] == "menu" else _split_page(page, max_words, overlap)
            for heading, text in splits:
                self.chunks.append(
                    Chunk(len(self.chunks), page["url"], page["title"], heading, text, page["section"])
                )

        # BM25 inverted index. URL path words, title and heading are indexed with the body
        # so that "what industries…" matches /industries/* and "what is Prisma" matches the
        # Prisma pages strongly.
        self.k1, self.b = 1.5, 0.75
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for c in self.chunks:
            slug = urlparse(c.url).path.replace("/", " ").replace("-", " ")
            toks = tokenize(f"{slug} {c.title} {c.heading} {c.text}")
            self.doc_len.append(len(toks))
            self.token_sets.append(frozenset(tokenize(c.text)))
            for term, tf in Counter(toks).items():
                self.postings[term].append((c.id, tf))
        # Broad questions ("which industries do you serve?") have no single passage that
        # answers them, but the menu lists every product, model, industry and use case.
        # It is attached to every question as one extra excerpt (~350 tokens).
        self.overview = [c for c in self.chunks if c.section == "menu"]
        n = max(len(self.chunks), 1)
        self.avg_len = sum(self.doc_len) / n
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }
        # Site vocabulary (real words, not stems) with frequencies, grouped by length,
        # for fast spelling correction of query words.
        self.word_freq: Counter[str] = Counter()
        for c in self.chunks:
            self.word_freq.update(
                w for w in TOKEN_RE.findall(f"{c.title} {c.text}".lower())
                if w.isalpha() and len(w) >= 4 and w not in STOPWORDS
            )
        self.vocab_by_len: dict[int, list[str]] = defaultdict(list)
        for w in self.word_freq:
            self.vocab_by_len[len(w)].append(w)

    @classmethod
    def from_file(cls, path: Path = config.PAGES_FILE) -> "Retriever":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m app.scraper` first.")
        with open(path, encoding="utf-8") as f:
            pages = [json.loads(line) for line in f if line.strip()]
        return cls(pages)

    def correct(self, word: str) -> str | None:
        """Closest site word for a query word that doesn't occur on the site
        ("prisam" -> "prisma", "gnnai" -> "gnani"), or None if nothing is close enough.
        Candidates are scored by similarity plus a small bonus for how common the word is
        on the site, so "gnnai" becomes "gnani" (very common), not "gnaniai" (rare)."""
        if not word.isalpha() or len(word) < 4 or _stem(word) in self.postings:
            return None
        candidates = [w for d in (-2, -1, 0, 1, 2) for w in self.vocab_by_len.get(len(word) + d, ())]
        matches = get_close_matches(word, candidates, n=10, cutoff=0.8)
        if not matches:
            return None
        return max(
            matches,
            key=lambda m: SequenceMatcher(None, word, m).ratio() + 0.03 * math.log10(self.word_freq[m] + 1),
        )

    def query_terms(self, query: str) -> dict[str, float]:
        """Search terms with weights: the query's words (typos corrected), plus synonyms."""
        terms: dict[str, float] = {}
        for word in TOKEN_RE.findall(query.lower()):
            if word in STOPWORDS:
                continue
            fixed = self.correct(word)
            terms[_stem(fixed or word)] = 1.0
        for t in list(terms):
            for syn in QUERY_SYNONYMS.get(t, ()):
                terms.setdefault(syn, SYNONYM_WEIGHT)
        return terms

    def _scores(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        terms = self.query_terms(query)
        for term, weight in terms.items():
            for doc_id, tf in self.postings.get(term, ()):
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                scores[doc_id] += weight * self.idf[term] * tf * (self.k1 + 1) / denom
        # Core product/company pages are the most authoritative source; give them a
        # small edge over the ~1,100 blog/glossary articles.
        for doc_id in scores:
            if self.chunks[doc_id].section not in {"resources", "glossary"}:
                scores[doc_id] *= 1.15
        return scores

    def search(self, query: str, k: int = config.TOP_K, per_page: int = config.MAX_CHUNKS_PER_PAGE) -> list[tuple[Chunk, float]]:
        """Top-k chunks for the query, for source diversity at most `per_page` per URL and
        no near-duplicates (the site repeats the same FAQ on e.g. every
        /speech-to-text/<language> page).

        Returns [] when no query term occurs anywhere on the site.
        """
        scores = self._scores(query)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out, per_url = [], Counter()
        for doc_id, s in ranked:
            c = self.chunks[doc_id]
            if per_url[c.url] >= per_page or self._is_near_duplicate(doc_id, out):
                continue
            per_url[c.url] += 1
            out.append((c, s))
            if len(out) >= k:
                break
        return out

    def _is_near_duplicate(self, doc_id: int, selected: list[tuple[Chunk, float]], threshold: float = 0.8) -> bool:
        a = self.token_sets[doc_id]
        for c, _ in selected:
            b = self.token_sets[c.id]
            if a and b and len(a & b) / len(a | b) >= threshold:
                return True
        return False

    def search_with_history(self, question: str, previous_question: str | None, k: int = config.TOP_K) -> list[Chunk]:
        """Retrieve for a possibly follow-up question ("what languages does it support?").

        Results for the question alone and for (previous question + question) are merged
        with reciprocal rank fusion, so a follow-up still finds the right topic.
        """
        lists = [self.search(question, k)]
        if previous_question:
            lists.append(self.search(f"{previous_question} {question}", k))
        fused: dict[int, float] = defaultdict(float)
        by_id: dict[int, Chunk] = {}
        for results in lists:
            for rank, (c, _) in enumerate(results):
                fused[c.id] += 1.0 / (60 + rank)
                by_id[c.id] = c
        top = [by_id[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]
        if not top:  # nothing on the site matches: let the caller refuse
            return []
        return top + [c for c in self.overview if c not in top]
