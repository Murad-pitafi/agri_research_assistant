"""
AgriMind Research Assistant
Streamlit UI — bilingual (English / Arabic), CPU-only, free deploy
"""

import streamlit as st
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriMind · Research Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp {
    background: #0a0e14;
}

/* Main container */
.block-container {
    padding: 2rem 2.5rem;
    max-width: 1100px;
}

/* Header */
.main-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    color: #4ade80;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.main-title {
    font-size: 32px;
    font-weight: 600;
    color: #f0f6fc;
    line-height: 1.2;
    margin-bottom: 8px;
}
.main-sub {
    font-size: 14px;
    color: #6b7f91;
    line-height: 1.6;
}

/* Chat messages */
.user-msg {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #4ade80;
    border-radius: 4px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #e6edf3;
}
.bot-msg {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #0ea5e9;
    border-radius: 4px;
    padding: 16px 18px;
    margin: 10px 0;
    font-size: 14px;
    color: #e6edf3;
    line-height: 1.7;
}
.arabic-msg {
    direction: rtl;
    text-align: right;
    font-size: 16px;
    line-height: 2;
}
.citation-pill {
    display: inline-block;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    color: #8b949e;
    margin: 2px 3px;
    font-family: 'IBM Plex Mono', monospace;
}
.meta-row {
    display: flex;
    gap: 12px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #21262d;
    flex-wrap: wrap;
}
.meta-chip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #6b7f91;
    letter-spacing: 0.08em;
}
.error-msg {
    background: #1c1017;
    border: 1px solid #4a1535;
    border-left: 3px solid #ef4444;
    border-radius: 4px;
    padding: 12px 16px;
    color: #fca5a5;
    font-size: 13px;
}

/* Sidebar */
.sidebar-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.15em;
    color: #4b5563;
    text-transform: uppercase;
    margin: 16px 0 8px;
}

/* Input area */
.stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.1) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #4ade80 !important;
    color: #4ade80 !important;
    background: rgba(74,222,128,0.05) !important;
}

/* Divider */
hr {
    border-color: #21262d !important;
    margin: 1.5rem 0 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-header">Configuration</div>', unsafe_allow_html=True)

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    st.markdown('<div class="sidebar-section">// language</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Response Language",
        ["English", "Arabic"],
        label_visibility="collapsed",
    ) or "English"

    st.markdown('<div class="sidebar-section">// knowledge base</div>', unsafe_allow_html=True)

    # PDF upload OR use bundled paper
    pdf_source = st.radio(
        "Paper source",
        ["Use bundled paper", "Upload PDF"],
        label_visibility="collapsed",
    )

    uploaded_pdf = None
    if pdf_source == "Upload PDF":
        uploaded_pdf = st.file_uploader("Upload research paper PDF", type=["pdf"])

    st.markdown('<div class="sidebar-section">// retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=8, value=5)

    st.markdown("---")
    st.markdown('<div class="sidebar-section">// about</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:11px;color:#4b5563;line-height:1.8;">
Built by <span style="color:#4ade80">Muhammad Murad</span><br>
AgriEngineering 2026 · MDPI<br>
<span style="font-family:'IBM Plex Mono',monospace;font-size:9px;">
Stack: LangChain · CrewAI · FAISS<br>
Groq (Llama 3.1) · Sentence Transformers<br>
CPU-only · No GPU required
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">// research assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">AgriMind</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-sub">Multi-agent research assistant grounded in the '
    '<em>Agentic AI Framework for Smart Agriculture</em> paper (AgriEngineering, 2026). '
    'Ask in English or Arabic — answers are cited to the source.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Index initialization ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_index(pdf_bytes: bytes | None = None):
    from core.rag_engine import load_or_build_index, VectorStore, parse_pdf, chunk_pages
    from knowledge.related_works import get_knowledge_texts
    import tempfile, os

    # Determine PDF path
    if pdf_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()
        pdf_path = tmp.name
    else:
        # Look for bundled paper
        candidates = [
            Path("Agentic_AI_Framework_to_Automate_Traditional_Farming_for_Smart_Agriculture.pdf"),
            Path("paper.pdf"),
            Path("knowledge/paper.pdf"),
        ]
        pdf_path = None
        for c in candidates:
            if c.exists():
                pdf_path = str(c)
                break

    if pdf_path and os.path.exists(pdf_path):
        store = load_or_build_index(pdf_path)
    else:
        # Build index from related works only (no PDF available)
        store = VectorStore()
        rw_texts = get_knowledge_texts()
        store.build(rw_texts)

    return store


# Load index
# Detect if a new PDF was uploaded and reset index if so
current_file_name = uploaded_pdf.name if uploaded_pdf else None
if current_file_name != st.session_state.last_uploaded_file:
    st.session_state.index_ready = False
    st.session_state.last_uploaded_file = current_file_name
    # Clear the cache for init_index when a new file is uploaded
    init_index.clear()

if not st.session_state.index_ready:
    with st.spinner("⚙ Building vector index (first run ~60s on CPU)…"):
        pdf_bytes = None
        if uploaded_pdf is not None:
            pdf_bytes = uploaded_pdf.read()
        try:
            st.session_state.vector_store = init_index(pdf_bytes)
            st.session_state.index_ready = True
        except Exception as e:
            st.error(f"Index error: {e}")


# ── Example questions ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    examples = [
        "What models were used for rice disease detection and what accuracies did they achieve?",
        "How does the supervisor agent communicate with the soil and weather agents?",
        "What is the difference between LSTM and GRU in this paper?",
        "ما هي دقة نموذج MobileViT في اكتشاف أمراض الأرز؟",  # Arabic: What is MobileViT accuracy?
        "How was the Mali Robot built and what sensors does it use?",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex[:60] + ("…" if len(ex) > 60 else ""), key=f"ex_{i}"):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        is_arabic = msg.get("language") == "Arabic"
        arabic_class = "arabic-msg" if is_arabic else ""
        st.markdown(
            f'<div class="bot-msg {arabic_class}">🌾 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("citations"):
            pills = " ".join(
                f'<span class="citation-pill">📄 {c}</span>'
                for c in msg["citations"]
            )
            st.markdown(
                f'<div style="margin-top:6px;">{pills}</div>',
                unsafe_allow_html=True,
            )
        if msg.get("meta"):
            m = msg["meta"]
            st.markdown(
                f'<div class="meta-row">'
                f'<span class="meta-chip">CHUNKS: {m.get("chunks", "?")}</span>'
                f'<span class="meta-chip">REQUESTS LEFT: {m.get("remaining", "?")}</span>'
                f'<span class="meta-chip">AGENTS: 2 (RETRIEVER + ANALYST)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("---")
query = st.text_area(
    "Ask a question",
    placeholder="Ask anything about the research paper… (English or Arabic)",
    height=80,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("→ Ask", use_container_width=True)

if submit and query.strip():
    if not api_key:
        st.markdown(
            '<div class="error-msg">⚠ Add GROQ_API_KEY to .streamlit/secrets.toml or your environment.</div>',
            unsafe_allow_html=True,
        )
    elif not st.session_state.index_ready:
        st.warning("Index still building, please wait…")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})

        # Run pipeline
        with st.spinner("🤖 Agents working…"):
            from core.agents import run_query
            result = run_query(
                query=query,
                vector_store=st.session_state.vector_store,
                api_key=api_key,
                language=language,
                top_k=top_k,
                user_id="streamlit_user",
            )

        if result.get("error") == "rate_limited":
            st.markdown(
                '<div class="error-msg">⏱ Daily limit reached (10 queries/day). '
                'Please try again tomorrow.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result.get("citations", []),
                "language": language,
                "meta": {
                    "chunks": result.get("retrieved_chunks", "?"),
                    "remaining": result.get("remaining_requests", "?"),
                },
            })

        st.rerun()
