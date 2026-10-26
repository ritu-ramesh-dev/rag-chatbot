"""Core RAG pipeline: load -> chunk -> embed/store -> retrieve -> answer."""
import os
import re
import sys
import uuid
from collections import deque
from html import unescape
from html.parser import HTMLParser

try:  # some hosts (e.g. Streamlit Cloud) ship an old sqlite that Chroma rejects; use the bundled one if present
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from pypdf import PdfReader

CHUNK_WORDS = 300
OVERLAP_WORDS = 50
LABEL_TABLES = False  # rewrite HTML table rows as labelled sentences (see _TextExtractor); evals: it hurt, see README
ROW_CHUNKS = False  # give each labelled table row its own chunk (experimental, see README)
TOP_K = 6 # raised from 4: retrieval hit rate 75% -> 88% in evals/run_evals.py
MODEL = "claude-sonnet-5-5"
GEMINI_MODEL = "gemini-3.1-flash-lite"  # free tier; change in the sidebar if Google retires it

MAX_SESSION_CHUNKS = 4000  # per visitor: keeps one person from filling the host's memory
MAX_LIVE_SESSIONS = 25  # oldest visitor spaces are dropped past this many

# Chroma embeds text for us with a small built-in local model (no API key needed).
_persistent_client = None
_live_sessions: deque = deque()


def get_collection(name: str = "docs"):
    """The saved on-disk collection used by the command-line evals. The web app never touches it."""
    global _persistent_client
    if _persistent_client is None:
        _persistent_client = chromadb.PersistentClient(path="chroma_db")
    return _persistent_client.get_or_create_collection(name)


def new_session_collection():
    """A private, in-memory collection for one visitor. Nothing is written to disk and nothing is shared."""
    client = chromadb.EphemeralClient()
    col = client.create_collection(f"session_{uuid.uuid4().hex}")
    _live_sessions.append((client, col.name))
    while len(_live_sessions) > MAX_LIVE_SESSIONS:  # free the memory of the longest-idle visitor spaces
        old_client, old_name = _live_sessions.popleft()
        try:
            old_client.delete_collection(old_name)
        except Exception:
            pass
    return col


def drop_session_collection(col) -> None:
    for item in list(_live_sessions):
        if item[1] == col.name:
            try:
                item[0].delete_collection(col.name)
            except Exception:
                pass
            _live_sessions.remove(item)


def load_pdf(file) -> list[tuple[str, str]]:
    """Return [("p.1", text), ...] for a PDF path or file-like object."""
    reader = PdfReader(file)
    return [(f"p.{i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]


NUM_RE = re.compile(r"^\(?\$?\s*[\d,]*\.?\d+\)?%?$|^[—–-]%?$")  # 1,234 (4)% 10% 7.5 and dashes
YEAR_RE = re.compile(r"\b(19|20)\d\d\b")
TABLE_TAG = "[table] "


class _TextExtractor(HTMLParser):
    """Turn HTML into plain text, keeping line breaks readable.

    With label_tables=True each numeric table row is rewritten as a self-describing line, e.g.
      "[table] The following table shows net sales by reportable segment ... - Europe: 2025: 111,032; Change: 10%; ..."
    so the number carries its row label, column headings and table title inside one chunk.
    """

    SKIP = {"script", "style", "ix:header"}  # ix:header = hidden XBRL metadata
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self, label_tables: bool = True):
        super().__init__()
        self.parts: list[str] = []
        self.label_tables = label_tables
        self._skip_depth = 0
        self._table_depth = 0
        self._rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._header: list[str] | None = None  # most recent column-heading row

    def _in_table(self) -> bool:
        return self.label_tables and self._table_depth > 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif self.label_tables and tag == "table":
            if self._table_depth == 0:
                self._rows = []
                self.parts.append("\n")
            self._table_depth += 1
        elif self._in_table():
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._cell = []
            elif tag == "br" and self._cell is not None:
                self._cell.append(" ")
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif self._in_table():
            if tag in ("td", "th") and self._cell is not None and self._row is not None:
                self._row.append(" ".join("".join(self._cell).split()))
                self._cell = None
            elif tag == "tr" and self._row is not None:
                self._rows.append(self._row)
                self._row = None
            elif tag == "table":
                self._table_depth -= 1
                if self._table_depth == 0:
                    self.parts.append("\n" + self._render_table() + "\n")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_table():
            if self._cell is not None:
                self._cell.append(data)
        else:
            self.parts.append(data)

    # -- table rendering ---------------------------------------------------
    def _title(self) -> str:
        """The last line of normal text before this table (usually 'The following table shows ...')."""
        text = "".join(self.parts[-80:])
        for line in reversed(text.splitlines()):
            line = line.strip()
            if line and not line.startswith(TABLE_TAG.strip()):
                return line[:200]
        return ""

    @staticmethod
    def _clean(raw: list[str]) -> list[str]:
        """Drop empty/'$' cells and glue stray ')' and '%' cells onto the number before them."""
        out: list[str] = []
        for c in raw:
            if not c or c in ("$", "|"):
                continue
            if c in (")", "%", ")%") and out:
                out[-1] += c
                continue
            out.append(c)
        return out

    def _render_table(self) -> str:
        title = self._title()
        lines: list[str] = []
        for raw in self._rows:
            cells = self._clean(raw)
            if not cells:
                continue
            nums = [c for c in cells if NUM_RE.match(c)]
            texts = [c for c in cells if not NUM_RE.match(c)]
            years = [c for c in cells if YEAR_RE.search(c)]
            plain_nums = [c for c in nums if not YEAR_RE.fullmatch(c)]

            if len(years) >= 2 and not plain_nums:  # a column-heading row like 2025 | Change | 2024 ...
                self._header = cells
                continue
            if len(nums) < 2:  # a text-only row: keep as is
                lines.append(" | ".join(cells))
                continue

            label = " ".join(texts) or (title if len(title) < 70 else "")
            if self._header and len(self._header) == len(nums):
                values = "; ".join(f"{h}: {v}" for h, v in zip(self._header, nums))
            else:
                values = "; ".join(nums)
            lines.append(f"{TABLE_TAG}{title} - {label}: {values}" if title else f"{TABLE_TAG}{label}: {values}")
        return "\n".join(lines)


ITEM_RE = re.compile(r"^(Item\s+\d+[A-C]?\.?.{0,120})$", re.IGNORECASE)


def load_html(file) -> list[tuple[str, str]]:
    """Return [(section_title, text), ...] for an HTML filing (e.g. an SEC 10-K).

    Sections are split on lines like "Item 1A. Risk Factors" so citations point to
    a section instead of a page number.
    """
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    parser = _TextExtractor(label_tables=LABEL_TABLES)
    parser.feed(raw.decode("utf-8", errors="ignore"))
    text = unescape("".join(parser.parts)).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]

    sections: list[tuple[str, list[str]]] = [("Front matter", [])]
    for ln in lines:
        if not ln or ln == "|":
            continue
        if ITEM_RE.match(ln):
            sections.append((ln, []))
        else:
            sections[-1][1].append(ln)
    # Drop tiny sections (e.g. table-of-contents entries) - they're just noise.
    return [(t, "\n".join(b)) for t, b in sections if len(" ".join(b).split()) >= 30]


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping windows of words so context isn't cut mid-idea."""
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [
        " ".join(words[i : i + CHUNK_WORDS])
        for i in range(0, max(len(words), 1), step)
        if words[i : i + CHUNK_WORDS]
    ]


def chunk_section(text: str) -> list[str]:
    """Chunk a section. With ROW_CHUNKS, every labelled table row becomes its own small chunk
    (so 'Europe: 111,032' is not blended with five neighbouring rows) and prose is chunked as usual."""
    if not (ROW_CHUNKS and TABLE_TAG in text):
        return chunk_text(text)
    chunks, prose = [], []
    for line in text.splitlines():
        if line.startswith(TABLE_TAG):
            if prose:
                chunks += chunk_text("\n".join(prose))
                prose = []
            chunks.append(line)
        else:
            prose.append(line)
    if prose:
        chunks += chunk_text("\n".join(prose))
    return chunks


def ingest(file, source_name: str, col=None, max_chunks: int | None = None) -> int:
    """Chunk a PDF or HTML file and store it in the vector DB. Returns chunks added.

    col: the collection to store into (the web app passes the visitor's private one).
    max_chunks: refuse (ValueError) if the collection would end up bigger than this.
    """
    is_html = source_name.lower().endswith((".htm", ".html"))
    pieces = load_html(file) if is_html else load_pdf(file)

    col = get_collection() if col is None else col
    docs, metas, ids = [], [], []
    for location, text in pieces:
        for chunk in chunk_section(text):
            docs.append(chunk)
            metas.append({"source": source_name, "location": location})
            ids.append(str(uuid.uuid4()))
    if not docs:
        raise ValueError("No readable text found in this file (scanned PDFs need OCR first).")
    if max_chunks is not None and col.count() + len(docs) > max_chunks:
        raise ValueError(
            f"This would exceed the limit of {max_chunks} text chunks per session "
            f"({col.count()} already stored + {len(docs)} new). Clear your documents or upload a smaller file."
        )
    for i in range(0, len(docs), 500):  # add in batches to stay under Chroma's limit
        col.add(documents=docs[i : i + 500], metadatas=metas[i : i + 500], ids=ids[i : i + 500])
    return len(docs)


RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"  # small cross-encoder (~80 MB, runs locally)
RERANK_POOL = 25  # candidates taken from EACH of vector and BM25 before reranking
_reranker = None
BM25_WEIGHT = 1.0  # how much the keyword ranking counts vs the vector ranking in hybrid mode
STOPWORDS = set(
    "a an the of in to and or is are was were be been did do does what how s for on by at "
    "as it its that this with from have had has".split()
)
_bm25_cache: dict = {}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps numbers like 166,000 and words like full-time whole."""
    toks = re.findall(r"\w+(?:[,.\-]\w+)*", text.lower())
    return [t for t in toks if t not in STOPWORDS]


def _bm25_index(col):
    """Build (and cache) a BM25 keyword index over every chunk in the collection."""
    key = (str(col.id), col.count())  # the collection's own id: python's id() can be reused after garbage collection
    if key not in _bm25_cache:
        if len(_bm25_cache) > 40:  # many visitors: don't let the cache grow without bound
            _bm25_cache.clear()
        data = col.get(include=["documents", "metadatas"])
        docs_tokens = [tokenize(d) for d in data["documents"]]
        df: dict[str, int] = {}
        for toks in docs_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        avgdl = sum(len(t) for t in docs_tokens) / max(len(docs_tokens), 1)
        _bm25_cache[key] = (data, docs_tokens, df, avgdl)
    return _bm25_cache[key]


def bm25_rank(col, question: str, n: int) -> list[str]:
    """Return ids of the n best keyword matches (BM25: rare words count more)."""
    import math

    data, docs_tokens, df, avgdl = _bm25_index(col)
    k1, b, N = 1.5, 0.75, len(docs_tokens)
    q = set(tokenize(question))
    scores = []
    for toks in docs_tokens:
        s = 0.0
        for term in q:
            f = toks.count(term)
            if f:
                idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks) / avgdl))
        scores.append(s)
    order = sorted(range(N), key=lambda i: scores[i], reverse=True)
    return [data["ids"][i] for i in order[:n] if scores[i] > 0]


def _rerank(col, question: str, k: int, vec_ids: list[str], bm_ids: list[str]) -> list[dict]:
    """Two-stage retrieval: gather candidates from vector + BM25, then re-score with a cross-encoder.

    A bi-encoder (the embedding model) turns question and chunk into vectors separately; a
    cross-encoder reads them together, so it is slower but far better at judging relevance.
    """
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)

    cand = list(dict.fromkeys(vec_ids + bm_ids))  # union, order kept, no duplicates
    got = col.get(ids=cand, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    scores = list(_reranker.rerank(question, [by_id[i][0] for i in cand]))
    top = sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)[:k]
    return [
        {
            "text": by_id[i][0],
            "source": by_id[i][1]["source"],
            "location": by_id[i][1]["location"],
            "distance": None,
            "score": float(s),
        }
        for i, s in top
    ]


def retrieve(
    question: str, k: int = TOP_K, mode: str = "vector", bm25_weight: float = None, col=None
) -> list[dict]:
    """Find the k best chunks. mode: "vector" (meaning), "bm25" (keywords) or "hybrid" (both).

    Default is "vector": in evals/ it beat BM25 and hybrid overall because real questions are
    paraphrased (see README, "Hybrid search experiment").

    Hybrid merges the two ranked lists with Reciprocal Rank Fusion: each list gives a chunk
    1/(60 + rank) points, so chunks ranked well by *either* method rise to the top.
    """
    col = get_collection() if col is None else col
    total = col.count()
    if total == 0:
        return []
    pool = min(RERANK_POOL if mode == "rerank" else 30, total)

    vec_ids, vec_dist = [], {}
    if mode in ("vector", "hybrid", "rerank"):
        res = col.query(query_texts=[question], n_results=pool)
        vec_ids = res["ids"][0]
        vec_dist = dict(zip(vec_ids, res["distances"][0]))
    bm_ids = bm25_rank(col, question, pool) if mode in ("bm25", "hybrid", "rerank") else []

    if mode == "rerank":
        return _rerank(col, question, k, vec_ids, bm_ids)

    w = BM25_WEIGHT if bm25_weight is None else bm25_weight
    scores: dict[str, float] = {}
    for ranked, weight in ((vec_ids, 1.0), (bm_ids, w)):
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + weight / (60 + rank + 1)
    top = sorted(scores, key=scores.get, reverse=True)[:k]

    got = col.get(ids=top, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}
    return [
        {
            "text": by_id[i][0],
            "source": by_id[i][1]["source"],
            "location": by_id[i][1]["location"],
            "distance": vec_dist.get(i),
            "score": scores[i],
        }
        for i in top
    ]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] ({c['source']}, {c['location']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the context below. "
        "Cite sources like [1], [2]. If the context does not contain the answer, "
        'say "I don\'t know based on the provided documents."\n\n'
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def generate(prompt: str, provider: str, api_key: str | None, model: str | None) -> str:
    """Send a prompt to the chosen LLM provider and return its text reply."""
    if provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
            http_options=types.HttpOptions(timeout=45_000),  # ms: fail fast instead of hanging
        )
        resp = client.models.generate_content(model=model or GEMINI_MODEL, contents=prompt)
        return resp.text or ""

    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model or MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def answer(
    question: str,
    api_key: str | None = None,
    provider: str = "gemini",
    model: str | None = None,
    mode: str = "vector",
    col=None,
) -> tuple[str, list[dict]]:
    """Full RAG: retrieve chunks, ask the LLM, return (answer, chunks used)."""
    chunks = retrieve(question, mode=mode, col=col)
    if not chunks:
        return "No documents loaded yet. Upload a PDF or HTML file first.", []
    return generate(build_prompt(question, chunks), provider, api_key, model), chunks
