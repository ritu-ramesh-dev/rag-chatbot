import base64
import importlib.util
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import rag

ROOT = Path(__file__).parent
DOC_ICON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="#ffc61a" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
    '<path d="M14 3v5h5M9 13h6M9 17h4"/></svg>'
)
DOC_ICON_DARK = DOC_ICON.replace("#ffc61a", "#111111")


def load_style() -> None:
    """Inject style.css, with a still of the moon (assets/background.jpg) behind the page until the live moon starts."""
    still = ROOT / "assets" / "background.jpg"
    url = "data:image/jpeg;base64," + base64.b64encode(still.read_bytes()).decode() if still.exists() else ""
    css = (ROOT / "style.css").read_text(encoding="utf-8").replace("__BG__", url)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def load_live_moon() -> None:
    """Start the real-time moon (static/moon.js) in the page itself.

    Streamlit can only run scripts inside a small iframe, so that iframe loads moon.js into the parent page, where
    the canvas sits fixed behind the app. It runs once per browser tab; if the script or WebGL is missing, the
    static picture from load_style() simply stays.
    """
    if not (ROOT / "static" / "moon.js").exists():
        return
    components.html(
        """<script>
        (function () {
          var P = window.parent;
          if (!P || P.__moonRequested) return;
          P.__moonRequested = true;
          var base = P.location.pathname.replace(/\\/+$/, "") + "/app/static/";
          var s = P.document.createElement("script");
          s.src = base + "moon.js";
          s.dataset.base = base;
          P.document.head.appendChild(s);
        })();
        </script>""",
        height=0,
    )


st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")
load_style()
load_live_moon()

st.markdown(
    f'<div class="hero"><div class="hero-icon">{DOC_ICON}</div><div>'
    '<div class="hero-title" role="heading" aria-level="1">Chat with your documents<i>.</i></div>'
    "<p>Upload your files and get instant, accurate answers</p></div></div>",
    unsafe_allow_html=True,
)

SAMPLE = ROOT / "data" / "apple-10k-2025.htm"
HAS_RERANKER = importlib.util.find_spec("fastembed") is not None


def ensure_session() -> None:
    """Every visitor gets a private in-memory collection; recreate it if the host dropped it."""
    col = st.session_state.get("col")
    try:
        if col is not None:
            col.count()
            return
    except Exception:
        st.session_state.notice = "Your documents expired on the server, so please upload them again."
    st.session_state.col = rag.new_session_collection()
    st.session_state.indexed = {}


def index_file(file, name: str) -> None:
    if name in st.session_state.indexed:
        st.info(f"{name} is already indexed.")
        return
    with st.spinner(f"Reading {name}..."):
        try:
            n = rag.ingest(file, name, col=st.session_state.col, max_chunks=rag.MAX_SESSION_CHUNKS)
        except Exception as e:
            st.error(f"{name}: {e}")
            return
    st.session_state.indexed[name] = n


def clear_documents() -> None:
    rag.drop_session_collection(st.session_state.col)
    st.session_state.col = rag.new_session_collection()
    st.session_state.indexed = {}
    st.session_state.history = []


ensure_session()
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown(
        f'<div class="brand"><div class="brand-icon">{DOC_ICON_DARK}</div>'
        '<div class="brand-text"><b>Chat with</b>your documents</div><span class="brand-spark">✦</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="setup-title">Setup</div>', unsafe_allow_html=True)
    provider = st.selectbox("LLM provider", ["gemini", "anthropic"])
    api_key = st.text_input(
        f"{provider.capitalize()} API key",
        type="password",
        help="Used only for your own requests. It is never stored or shared. Get a free Gemini key at aistudio.google.com.",
    )
    model = st.text_input("Model", value=rag.GEMINI_MODEL if provider == "gemini" else rag.MODEL)
    use_rerank = False
    if HAS_RERANKER:
        use_rerank = st.checkbox(
            "Rerank results (slower, ~3s)",
            help="Re-scores 50 candidate chunks with a cross-encoder. Eval: 88% -> 92% retrieval hit rate.",
        )
    uploads = st.file_uploader(
        "Upload PDF or HTML files", type=["pdf", "htm", "html"], accept_multiple_files=True
    )
    if st.button("Index documents   →", use_container_width=True, key="index") and uploads:
        for f in uploads:
            index_file(f, f.name)
    if SAMPLE.exists() and st.button("Try the Apple 10-K sample", use_container_width=True, key="sample"):
        with open(SAMPLE, "rb") as f:
            index_file(f, SAMPLE.name)

    if st.session_state.indexed:
        total = sum(st.session_state.indexed.values())
        st.markdown(f"**Your documents** · {total} chunks")
        for name, n in st.session_state.indexed.items():
            st.caption(f"• {name} ({n})")
        st.button("Clear my documents", use_container_width=True, key="clear", on_click=clear_documents)
    else:
        st.caption("No documents yet.")
    st.caption(
        "Your files are kept in memory for this browser session only and are not saved by this app. "
        "Your questions and the matching passages are sent to the AI provider you choose."
    )
    st.caption("Moon imagery: NASA / GSFC / Arizona State University / MIT (LRO), public domain.")

if notice := st.session_state.pop("notice", None):
    st.warning(notice)

if not st.session_state.history and not st.session_state.indexed:
    st.markdown(
        '<div class="empty"><b>Get started</b><br>1. Add your API key in the sidebar.<br>'
        "2. Upload a PDF or HTML file, or try the Apple 10-K sample.<br>3. Ask a question below.</div>",
        unsafe_allow_html=True,
    )

for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            with st.expander("Sources"):
                for i, c in enumerate(sources, 1):
                    st.markdown(f"**[{i}] {c['source']}, {c['location']}**")
                    st.caption(c["text"][:400] + "...")

if question := st.chat_input("Ask something about your documents"):
    st.session_state.history.append(("user", question, None))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        sources = []
        if not api_key:
            reply = (
                f"Please add your {provider.capitalize()} API key in the sidebar first. "
                "It stays in your browser session and is never stored."
            )
        elif not st.session_state.indexed:
            reply = "Upload a document or try the sample first, then ask your question."
        else:
            try:
                with st.spinner("Thinking..."):
                    reply, sources = rag.answer(
                        question, api_key, provider, model or None,
                        mode="rerank" if use_rerank else "vector",
                        col=st.session_state.col,
                    )
            except Exception as e:  # bad key, quota, network...
                reply, sources = f"Error: {e}", []
        st.markdown(reply)
    st.session_state.history.append(("assistant", reply, sources))
    st.rerun()
