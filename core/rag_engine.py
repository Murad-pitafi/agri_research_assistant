"""
RAG Engine
- Parses the AgriEngineering PDF into chunks
- Embeds using sentence-transformers (CPU, ~90MB, no GPU needed)
- Stores in FAISS vector index
- Retrieves top-k relevant chunks with source citations
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# Lazy import to avoid slow startup when not needed
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: fast, CPU-friendly, 90MB, great for English + decent Arabic
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── Chunking ──────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> List[Tuple[str, Dict]]:
    """Extract text from PDF, return list of (text, metadata) per page."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((text, {"source": "paper", "page": i + 1}))
    return pages


def chunk_pages(
    pages: List[Tuple[str, Dict]],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Tuple[str, Dict]]:
    """Split pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for text, meta in pages:
        splits = splitter.split_text(text)
        for j, split in enumerate(splits):
            chunk_meta = {**meta, "chunk": j}
            chunks.append((split, chunk_meta))
    return chunks


# ── FAISS Index ───────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.index = None
        self.texts: List[str] = []
        self.metas: List[Dict] = []
        self.dim: int = 384  # all-MiniLM-L6-v2 output dim

    def build(self, chunks: List[Tuple[str, Dict]]):
        """Embed all chunks and build FAISS index."""
        embedder = get_embedder()
        texts = [c[0] for c in chunks]
        metas = [c[1] for c in chunks]

        print(f"  Embedding {len(texts)} chunks on CPU...")
        # Batch embedding - CPU friendly
        embeddings = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Inner product index (cosine similarity since normalized)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.texts = texts
        self.metas = metas
        print(f"  FAISS index built: {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k chunks for query."""
        embedder = get_embedder()
        q_emb = embedder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.texts[idx],
                "meta": self.metas[idx],
                "score": float(score),
            })
        return results

    def save(self, path: str):
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metas": self.metas}, f)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".pkl", "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.metas = data["metas"]


# ── Index Manager ─────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent.parent / ".cache"
INDEX_PATH = str(CACHE_DIR / "vector_index")


def _pdf_hash(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read(8192))  # hash first 8KB for speed
    return h.hexdigest()[:8]


def load_or_build_index(pdf_path: str, force_rebuild: bool = False) -> VectorStore:
    """
    Load cached index if available, else build from PDF + related works.
    Cache is invalidated if the PDF changes.
    """
    from knowledge.related_works import get_knowledge_texts

    CACHE_DIR.mkdir(exist_ok=True)
    hash_file = CACHE_DIR / "pdf_hash.txt"
    current_hash = _pdf_hash(pdf_path)

    # Check cache validity
    cached_hash = ""
    if hash_file.exists():
        cached_hash = hash_file.read_text().strip()

    cache_valid = (
        not force_rebuild
        and cached_hash == current_hash
        and Path(INDEX_PATH + ".faiss").exists()
        and Path(INDEX_PATH + ".pkl").exists()
    )

    store = VectorStore()

    if cache_valid:
        print("Loading cached FAISS index...")
        store.load(INDEX_PATH)
        print(f"  Loaded {store.index.ntotal} vectors from cache")
    else:
        print("Building vector index (first run takes ~1 min on CPU)...")
        # Parse PDF
        pages = parse_pdf(pdf_path)
        print(f"  Parsed {len(pages)} pages from PDF")
        chunks = chunk_pages(pages)
        print(f"  Created {len(chunks)} chunks from paper")

        # Add related works
        rw_texts = get_knowledge_texts()
        all_chunks = chunks + rw_texts
        print(f"  Total chunks (paper + related works): {len(all_chunks)}")

        store.build(all_chunks)
        store.save(INDEX_PATH)
        hash_file.write_text(current_hash)
        print("Index cached for future runs.")

    return store
