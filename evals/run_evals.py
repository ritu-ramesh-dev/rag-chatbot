"""Evaluate the RAG pipeline against questions with known answers from the Apple FY2025 10-K.

Two scores, because failures come from two different places:
  1. Retrieval hit rate - did the top-k chunks contain the expected answer text?
     (Needs no API key. If this is low, no LLM can save the answer.)
  2. Answer accuracy   - did the LLM's final answer contain the expected text?
     (Needs an API key. Unanswerable questions must say "I don't know".)

Run from the project root:
    .venv\\Scripts\\python.exe evals\\run_evals.py                  # retrieval only
    set GEMINI_API_KEY=...  &&  .venv\\Scripts\\python.exe evals\\run_evals.py --answers
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag  # noqa: E402

HERE = Path(__file__).resolve().parent
IDK = re.compile(r"don'?t know|do not know|does not (contain|mention|provide|identify)|not (contain|mention|provide)|cannot (find|answer)", re.I)


def norm(s: str) -> str:
    return s.lower().replace("’", "'")


def contains_any(text: str, options: list[str]) -> bool:
    t = norm(text)
    return any(norm(o) in t for o in options)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=rag.TOP_K, help="chunks to retrieve")
    ap.add_argument("--answers", action="store_true", help="also grade LLM answers (needs API key)")
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between LLM calls (free-tier limits)")
    ap.add_argument("--mode", default="vector", choices=["vector", "bm25", "hybrid", "rerank"], help="retrieval method")
    ap.add_argument("--bm25-weight", type=float, default=None, help="hybrid fusion weight for BM25 (default: rag.BM25_WEIGHT)")
    ap.add_argument("--row-chunks", action="store_true", help="with --chunk-words: one chunk per table row")
    ap.add_argument("--plain-tables", action="store_true", help="with --chunk-words: use the old flat table text")
    ap.add_argument("--chunk-words", type=int, help="re-chunk the 10-K in memory at this size (experiment)")
    ap.add_argument("--overlap", type=int, default=50, help="overlap words when using --chunk-words")
    ap.add_argument("--source", default=str(HERE.parent / "data" / "apple-10k-2025.htm"))
    args = ap.parse_args()

    if args.chunk_words:
        # Build a throwaway in-memory index so experiments never touch the app's database.
        import chromadb

        rag.CHUNK_WORDS, rag.OVERLAP_WORDS = args.chunk_words, args.overlap
        rag.LABEL_TABLES = not args.plain_tables
        rag.ROW_CHUNKS = args.row_chunks
        temp = chromadb.EphemeralClient().get_or_create_collection("eval_tmp")
        rag.get_collection = lambda name="docs": temp
        with open(args.source, "rb") as f:
            rag.ingest(f, Path(args.source).name)
        print(f"[experiment] chunk={args.chunk_words} overlap={args.overlap} -> {temp.count()} chunks")

    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    sample = HERE.parent / "data" / "apple-10k-2025.htm"
    if rag.get_collection().count() == 0 and sample.exists():
        print("[evals] saved index is empty, indexing the bundled Apple 10-K sample first...")
        with open(sample, "rb") as f:
            rag.ingest(f, sample.name)
    n_chunks = rag.get_collection().count()
    if n_chunks == 0:
        sys.exit("No chunks in the database and no data/apple-10k-2025.htm to index.")

    rows, ret_hits, ret_total, ans_hits, ans_total, errors = [], 0, 0, 0, 0, 0
    by_set: dict[str, list[int]] = {}  # set name -> [retrieval hits, retrieval total]
    for c in cases:
        chunks = rag.retrieve(c["question"], k=args.k, mode=args.mode, bm25_weight=args.bm25_weight)
        joined = "\n".join(ch["text"] for ch in chunks)
        ret = None
        if c["type"] != "unanswerable":
            ret = contains_any(joined, c["any_of"])
            ret_total += 1
            ret_hits += ret
            s = by_set.setdefault(c.get("set", "A"), [0, 0])
            s[0] += ret
            s[1] += 1

        ans = reply = None
        if args.answers:
            try:
                reply = rag.generate(rag.build_prompt(c["question"], chunks), args.provider, None, args.model)
                ans = bool(IDK.search(reply)) if c["type"] == "unanswerable" else contains_any(reply, c["any_of"])
            except Exception as e:  # keep going: one failed call shouldn't kill the run
                reply = f"ERROR: {e}"
                ans = False
            ans_total += 1
            ans_hits += ans
            if reply.startswith("ERROR"):
                print(f"   {reply[:300]}")
                errors += 1
                if errors >= 3:
                    sys.exit("\nStopping: 3 LLM calls failed in a row. Fix the error above, then rerun.")
            else:
                errors = 0
            time.sleep(args.delay)

        rows.append((c, ret, ans, reply, chunks))
        mark = lambda v: "-" if v is None else ("PASS" if v else "FAIL")
        print(f"{c['id']:>2} {c['type']:<12} retrieval={mark(ret):<4} answer={mark(ans):<4} {c['question']}")

    print(f"\nRetrieval hit rate @k={args.k}: {ret_hits}/{ret_total} = {ret_hits / ret_total:.0%}")
    for name, (h, t) in sorted(by_set.items()):
        print(f"   set {name} ({'paraphrased' if name == 'B' else 'filing wording'}): {h}/{t} = {h / t:.0%}")
    if args.answers:
        print(f"Answer accuracy: {ans_hits}/{ans_total} = {ans_hits / ans_total:.0%}")

    # Write a report you can paste in the README.
    lines = [f"# Eval results\n", f"- Chunks in index: {n_chunks}", f"- Retrieval mode: {args.mode}", f"- Top-k: {args.k}",
             f"- Retrieval hit rate: **{ret_hits}/{ret_total} ({ret_hits / ret_total:.0%})**"]
    if args.answers:
        lines.append(f"- Answer accuracy ({args.provider}, {args.model or 'default model'}): "
                     f"**{ans_hits}/{ans_total} ({ans_hits / ans_total:.0%})**")
    lines += ["", "| # | Type | Retrieval | Answer | Question |", "|---|---|---|---|---|"]
    for c, ret, ans, _, _ in rows:
        f = lambda v: "-" if v is None else ("pass" if v else "FAIL")
        lines.append(f"| {c['id']} | {c['type']} | {f(ret)} | {f(ans)} | {c['question']} |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {HERE / 'results.md'}")


if __name__ == "__main__":
    main()
