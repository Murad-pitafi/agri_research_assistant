# AgriMind — Multi-Agent Research Assistant

A bilingual (English / Arabic) research assistant powered by a two-agent CrewAI pipeline, LangChain RAG, and FAISS vector search. Grounded in the peer-reviewed paper:

> **Agentic AI Framework to Automate Traditional Farming for Smart Agriculture**
> *AgriEngineering, MDPI, vol. 8, no. 1, Jan 2026*

**Live demo:** *(add your Streamlit Cloud URL here)*

---

## What It Does

- **Retrieval-Augmented Generation** — answers are grounded in the actual paper, not hallucinated
- **Two CrewAI agents** collaborate on every query: a Research Retriever and a Research Analyst
- **Bilingual** — responds in English or Arabic based on user selection
- **Source citations** — every answer shows which page or related work it came from
- **Rate limited** — 10 queries/day per user to protect the free API quota
- **CPU-only** — no GPU required anywhere in the pipeline

---

## Architecture

```
User Query (English or Arabic)
        │
        ▼
LangChain → FAISS Vector Search
        │   (sentence-transformers, CPU)
        │   Knowledge: Paper PDF + 7 related work summaries
        ▼
CrewAI Agent 1 — Research Retriever
        │   Identifies relevant passages, extracts citations
        ▼
CrewAI Agent 2 — Research Analyst
        │   Synthesizes answer in chosen language
        ▼
Groq API (Llama 3.1-8b-instant, free tier)
        │
        ▼
Streamlit UI — cited, bilingual response
```

---

## Stack

| Component | Tool | Cost |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, CPU |
| Vector store | FAISS (CPU) | Free |
| LLM | Groq — Llama 3.1-8b-instant | Free tier |
| Agents | CrewAI | Open source |
| RAG | LangChain | Open source |
| UI | Streamlit | Free |
| Hosting | Streamlit Cloud | Free |

---

## Run Locally

```bash
git clone https://github.com/murad-pitafi/agrimind
cd agrimind

pip install -r requirements.txt

# Add your Groq API key (free at console.groq.com)
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml

# Put the paper PDF in the project root
cp your_paper.pdf Agentic_AI_Framework_to_Automate_Traditional_Farming_for_Smart_Agriculture.pdf

# Run
streamlit run ui/app.py
```

First run downloads the embedding model (~90MB) and indexes the PDF — takes about 60 seconds. Subsequent runs load from cache instantly.

---

## Deploy Free on Streamlit Cloud

1. Push repo to GitHub (exclude `.streamlit/secrets.toml` via `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → your repo
3. Set **Main file path** to `ui/app.py`
4. Under **Advanced settings → Secrets**, paste:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Deploy — done. Free, permanent, shareable URL.

---

## Project Structure

```
agrimind/
├── core/
│   ├── rag_engine.py     # PDF parsing, chunking, FAISS index
│   └── agents.py         # CrewAI agent definitions + rate limiter
├── knowledge/
│   └── related_works.py  # Curated related paper summaries
├── ui/
│   └── app.py            # Streamlit interface
├── .streamlit/
│   └── config.toml       # Theme config
├── requirements.txt
└── README.md
```

---

## Author

**Muhammad Murad** — AI/ML Engineer

- 📄 First Author, AgriEngineering (MDPI), vol. 8, Jan 2026
- 🏅 Global Top 5 Finalist — IEEE YESIST12, Kuala Lumpur 2025
- 🌏 International Presenter — GCIEM Global Summit, Taipei 2026
- 📧 pitafimurad99@gmail.com
- 🔗 [linkedin.com/in/murad-pitafi](https://linkedin.com/in/murad-pitafi)
