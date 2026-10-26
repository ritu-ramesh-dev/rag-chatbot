# Chat with SEC filings (RAG)

Ask questions about a company's 10-K and get answers with citations to the section they came from.
Built with Python, Streamlit, ChromaDB, and a swappable LLM (Gemini free tier or Claude).

**Live demo: <https://chat-with-your-documents-rag.streamlit.app/>** (bring your own free Gemini key; the Apple 10-K
sample loads in one click). It runs on a free tier that sleeps when idle, so the first load can take about a minute.

Demo corpus: Apple's FY2025 Form 10-K (filed 2025-10-31).

## How it works

```
INGEST  filing (HTML/PDF) -> text -> split by "Item N." sections -> 300-word overlapping chunks
        -> local embeddings (MiniLM via Chroma) -> vector DB with {source, section} metadata

QUERY   question -> embed -> top-6 nearest chunks -> prompt ("answer ONLY from context, cite [n],
        say I don't know otherwise") -> LLM -> answer + source chunks shown in the UI
```

Files: `rag.py` (pipeline, no UI), `app.py` (Streamlit), `evals/` (evaluation harness).

## Look and feel

The UI is a dark, editorial "space" theme: near-black panels with hairline borders, bold white type, a yellow
accent, and a halftone grain over the picture. It is plain CSS on top of Streamlit (`style.css`,
`.streamlit/config.toml`, the header markup in `app.py`).

### The live moon

The background is a **real-time, rotating Moon built from NASA data**, drawn in the browser by a small WebGL2 shader
(`static/moon.js`, no libraries). It wraps NASA's LRO colour map around a sphere, lights it with a normal map computed
from laser-altimeter heights (so the craters, mountains and basins are the real ones), and uses the Lommel-Seeliger law,
which is how a dusty airless world actually scatters sunlight. It turns slowly (about one revolution every 8 minutes),
the stars twinkle and drift, and it eases a little toward the mouse. It honours `prefers-reduced-motion`, pauses when the
tab is hidden, and falls back to a still image (`assets/background.jpg`) when WebGL2 is unavailable.

How it is wired: Streamlit serves `static/` (`enableStaticServing` in `.streamlit/config.toml`), and a zero-height
component in `app.py` loads `moon.js` into the page, where its canvas sits fixed behind the app.

| File | What it is |
|---|---|
| `static/moon.js` | the live renderer |
| `static/moon_color_{8k,4k}.jpg`, `static/moon_normal_{4k,2k}.jpg` | web textures (the larger ones for big screens, the smaller for phones and weak GPUs) |
| `tools/make_moon_textures.py` | builds those textures from NASA's raw maps |
| `tools/make_fallback.py` | renders the still `assets/background.jpg` from the same data and maths |

To rebuild the textures, download `lroc_color_poles_8k.tif` (48 MB) and `ldem_16.tif` (63 MB) from
<https://svs.gsfc.nasa.gov/4720/> into `tools/_nasa_raw/` (git-ignored), then run
`.venv\Scripts\python.exe -I tools\make_moon_textures.py` and `.venv\Scripts\python.exe -I tools\make_fallback.py`.

Moon imagery: NASA / GSFC / Arizona State University (LROC colour) and NASA / GSFC / MIT (LOLA elevation), via the
NASA Scientific Visualization Studio "Moon CGI Kit". NASA media is public domain; the credit is shown in the sidebar.

The font (Manrope) loads from Google Fonts and falls back to the system font offline.

## Use it online (for anyone)

The app is built to be hosted publicly. Each visitor gets a **private, in-memory** document space (nothing is shared
between visitors or written to disk), and brings their **own API key** (a free Gemini key works), so the host's
quota and key are never used.

| Protection | How |
|---|---|
| Documents are private per visitor | Each browser session gets its own Chroma collection; idle sessions are dropped after 25 live ones |
| No shared or stored keys | The key lives only in that visitor's session; the server never reads one from the environment |
| Upload limits | 20 MB per file, 4,000 chunks per session, empty / scanned files rejected with a clear message |
| No duplicates | A file already indexed in the session is skipped |
| One-click try-out | "Try the Apple 10-K sample" button, no upload needed |

### Deploy on Streamlit Community Cloud (free)

1. Push this repo to GitHub (public repos deploy free).
2. Go to <https://share.streamlit.io>, sign in with GitHub and choose **Create app**.
3. Pick the repo and branch, set **Main file path** to `app.py`, and under *Advanced settings* choose Python 3.12.
4. Click **Deploy**. The first start takes a few minutes while dependencies install; the first document a visitor
   indexes also downloads a small (~80 MB) embedding model.
5. Share the `https://<your-app>.streamlit.app` link.

Notes: the free tier has about 1 GB of memory. If the app crashes at startup or when ranking, delete the
`fastembed` line from `requirements.txt` (this only removes the optional "Rerank results" checkbox, which hides itself).
Do **not** add a real API key to the repo or to the host's secrets for this app: visitors enter their own.

## Run it on your own computer

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Paste an API key in the sidebar (never commit keys), click **Try the Apple 10-K sample** or upload your own file, click
**Index documents**, then ask questions.

## Evaluation

28 questions with known answers taken from the filing (`evals/cases.json`). Set A (19): 12 numeric facts,
4 text facts and 3 unanswerable questions that should get "I don't know". Set B (9): the same kinds of
facts asked in paraphrased wording. The answer-accuracy numbers below are from Set A only.
Two separate scores, because failures happen in two places: finding the right text (retrieval)
and writing the answer (LLM).

```bash
.venv\Scripts\python.exe evals\run_evals.py                 # retrieval only, no API key needed
.venv\Scripts\python.exe evals\run_evals.py --answers       # also grades LLM answers (needs key)
```

### Headline results (300-word chunks, top-6, `gemini-3.1-flash-lite`)

| Metric | Score |
|---|---|
| Retrieval hit rate (16 answerable questions) | **14/16 (88%)** |
| Answer accuracy (all 19 questions) | **17/19 (89%)** |
| Unanswerable questions correctly refused ("I don't know") | **3/3** |

Every question whose answer text was retrieved was answered correctly (14/14), and both wrong
answers are retrieval failures (employee count, Europe net sales), not LLM mistakes. So the LLM step
is not the bottleneck here; finding the right text is. The model also never invented an answer to
the three out-of-scope questions.

### Retrieval hit rate (does the top-k contain the expected answer text?)

| Chunk words / overlap | k=4 | k=6 |
|---|---|---|
| 60 / 12 | 38% | 56% |
| 100 / 20 | 50% | 56% |
| 150 / 30 | 81% | 81% |
| 200 / 40 | 75% | 81% |
| **300 / 50 (used)** | 75% | **88%** |
| 400 / 67 | 69% | 69% |
| 500 / 83 | 81% | 88% |

What I learned:
- Retrieving more chunks helped more than tuning chunk size: k=4 -> 6 took the default from 75% to 88%.
- Very small chunks were clearly worse (38% at 60 words): they lose the context the embedding needs.
- Between 150 and 500 words the results are within noise. With 16 scorable questions, one question is 6 points, so I did not over-read small differences.
- Two questions fail at every setting, even at k=12 (employee count, Europe net sales). The facts are in the
  index, but the correct chunk ranks 42nd and 28th. The employee count is one sentence inside a 300-word
  chunk about human capital, and the Europe figure sits in a number-dense table row. Pure embedding
  search is weak for exact figures.

I then tried hybrid search to fix those two (next section).

### Hybrid search experiment (BM25 keywords + vectors)

I implemented BM25 from scratch in `rag.py` and merged it with the vector ranking using Reciprocal Rank
Fusion. Pure keyword search scored 100% on my first 16 questions, which looked like a win until I
realised I had written those questions using the filing's own words. So I added **Set B: 9
paraphrased questions** ("How big is Apple's workforce?" instead of "full-time equivalent employees").

| Retrieval (k=6) | Set A: filing's wording (16) | Set B: paraphrased (9) | Overall (25) |
|---|---|---|---|
| **Vector only (used)** | 88% | **89%** | **88%** |
| BM25 only | **100%** | 33% | 76% |
| Hybrid, equal weights | 88% | 67% | 80% |
| Hybrid, BM25 weight 5 | 100% | 56% | 84% |

Result: **hybrid did not beat plain vector search overall**, so vector stays the default
(`retrieve(..., mode="hybrid")` is available). Keyword matching is excellent when the question reuses
the document's exact words and poor when it doesn't, and in fusion its noisy matches on generic words
("company", "sales", "year") pushed good vector hits out of the top 6.

Caveats: 9 paraphrased questions is a small sample, and I should not tune stopwords or weights on the same
questions I evaluate on.

### Reranking experiment

Next I tried two-stage retrieval: collect the top 25 chunks from vector search plus the top 25 from BM25
(up to 50 candidates), then re-score each with a cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, ~80 MB,
runs locally). A cross-encoder reads the question and chunk together, so it judges relevance better than
comparing two separately computed vectors.

| Retrieval | k=4 | k=6 | Latency / query |
|---|---|---|---|
| Vector only (default) | 76% | 88% (22/25) | ~0.2 s |
| Vector + BM25 candidates, reranked | 84% | **92% (23/25)** | ~3 s |

It fixed the employee-count question, but two questions still fail, for different reasons:
- **"Net sales in Europe"**: the correct table chunks were found by BM25 (vector search missed them) but the
  reranker placed them 11th and 19th. The model was trained on prose, and number-heavy table rows score poorly.
- **"How big is Apple's workforce?"**: the correct chunk was in *neither* candidate pool, because the question
  shares no words with "166,000 full-time equivalent employees". A reranker can only re-order candidates it is given.

The gain is one question out of 25 for roughly 15x the latency, so reranking is an opt-in checkbox in the
app rather than the default.

### Table parsing experiment (an idea that did not work)

The Europe failure looked like a table problem: the 10-K's HTML tables were flattened to
`| Europe | 111,032 | | | 10 | % |`, so the number was separated from its row label, column heading and title.
I rewrote the HTML parser (`LABEL_TABLES`) to rebuild every numeric row as a self-describing line, e.g.
`[table] ...net sales by reportable segment for 2025, 2024 and 2023 (dollars in millions): - Europe: 2025: 111,032; Change: 10%; 2024: 101,328 ...`
(343 rows parsed), then tried giving each row its own chunk (`ROW_CHUNKS`).

| Table handling (same 28 questions) | Chunks | Vector k=6 | Rerank k=6 | Europe fixed? |
|---|---|---|---|---|
| **Flat text (used)** | 158 | **88%** | **92%** | no |
| Labelled rows inside 300-word chunks | 160 | 76% | 88% | no (rank 28 -> 13) |
| Labelled rows, one chunk per row | 523 | 56% | 72% | only at k=8 |

Why it backfired: with labelled rows inside normal chunks, each chunk repeats the same long title six times,
so all the segment tables look alike to the embedding model. With one chunk per row, 343 near-identical
rows crowd the top results and push out the chunks other questions need (reranking at k=8 recovers to 84%,
still below the 92% baseline). Fixing one question cost several others.

I left `LABEL_TABLES = False` and kept the code as an experiment. The remaining failures would need a different
approach: store numbers in a structured side table and answer numeric questions with a lookup (or let the LLM
call a "get financial statement value" tool), rather than hoping text similarity finds the right row.

## Limitations

- Financial tables are flattened to text (`| Europe | 111,032 |`), so numeric lookups are the weakest area (see above).
- No conversation memory: each question is answered independently, so follow-ups like "what about last year?" fail.
- Duplicate files are skipped by file name only, so the same text under a different name is indexed twice. The saved index used by the command-line evals (`chroma_db/`) can still be double-filled by running the old indexing path twice; delete the folder to rebuild it.
- A shared host has limited memory and no rate limiting, so a very busy public deployment would need a paid tier or a request limiter.
- Chunks never cross section or page boundaries, so an idea spanning two sections is split.
- The eval set is small (28 questions, only 9 paraphrased) and grading is substring match, so it can miss correct rephrasings.
- Text inside a document could try to instruct the model (prompt injection); the "answer only from context" rule reduces but does not remove this.
- On Gemini's free tier, Google may use prompts for product improvement, so use public documents only.

## License and credits

The code is released under the [MIT License](LICENSE). Third-party material keeps its own terms:

- **Moon imagery** (`static/moon_*.jpg`, `assets/background.jpg`): NASA / GSFC / Arizona State University (LROC colour)
  and NASA / GSFC / MIT (LOLA elevation), via NASA's Scientific Visualization Studio. NASA media is public domain.
- **Sample document** (`data/apple-10k-2025.htm`): Apple Inc.'s annual report as filed with the U.S. SEC, a public record.
- **Manrope** font: SIL Open Font License, loaded from Google Fonts at runtime (not bundled).
