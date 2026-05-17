"""
Multi-Agent Pipeline using CrewAI
Two agents collaborate on every query:

Agent 1 — Research Retriever
  Role: Find the most relevant chunks from the vector store
  Output: Structured context with citations

Agent 2 — Research Analyst
  Role: Synthesize retrieved context into a clear, cited answer
  Handles both English and Arabic responses
  Never hallucinates — only answers from retrieved context
"""

from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from typing import List, Dict, Any, Optional
import json
import re


# ── LLM Setup ─────────────────────────────────────────────────────────────────

def get_llm(api_key: str, model: str = "llama-3.1-8b-instant"):
    """
    Groq free tier:
    - llama-3.1-8b-instant: fastest, great for CPU-gated workflows
    - llama-3.1-70b-versatile: smarter, still free but slower
    Using 8b for speed — good enough for RAG where context does the heavy lifting
    """
    return ChatGroq(
        api_key=api_key,
        model=model,
        temperature=0.2,      # low temp = more factual, less hallucination
        max_tokens=1024,
    )


# ── Rate Limiter ──────────────────────────────────────────────────────────────

import time
from collections import defaultdict

_request_log: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 10   # per IP per day
RATE_LIMIT_WINDOW = 86400  # 24 hours in seconds


def check_rate_limit(user_id: str) -> tuple[bool, int]:
    """Returns (is_allowed, remaining_requests)."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Clean old entries
    _request_log[user_id] = [t for t in _request_log[user_id] if t > window_start]
    count = len(_request_log[user_id])
    if count >= RATE_LIMIT_REQUESTS:
        return False, 0
    _request_log[user_id].append(now)
    return True, RATE_LIMIT_REQUESTS - count - 1


# ── Context Formatter ─────────────────────────────────────────────────────────

def format_context(retrieved: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into structured context string."""
    parts = []
    for i, r in enumerate(retrieved):
        meta = r["meta"]
        source = meta.get("source", "unknown")
        if source == "paper":
            citation = f"[Paper, Page {meta.get('page', '?')}]"
        else:
            citation = f"[Related Work: {meta.get('title', 'Unknown')}]"
        score = r["score"]
        parts.append(
            f"--- Context {i+1} {citation} (relevance: {score:.2f}) ---\n{r['text']}"
        )
    return "\n\n".join(parts)


# ── Agent Definitions ─────────────────────────────────────────────────────────

def build_crew(llm, context: str, query: str, language: str = "English") -> Crew:
    lang_instruction = (
        "Respond in Arabic (العربية). Use clear Modern Standard Arabic."
        if language == "Arabic"
        else "Respond in English."
    )

    retriever_agent = Agent(
        role="Research Context Analyst",
        goal=(
            "Analyze the retrieved context and identify the most relevant "
            "passages that directly answer the user's question. "
            "Extract key facts, numbers, and citations accurately."
        ),
        backstory=(
            "You are a meticulous research analyst specializing in AI and "
            "agricultural technology. You extract precise information from "
            "academic papers and never make up facts not present in the context."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    analyst_agent = Agent(
        role="Research Communication Specialist",
        goal=(
            f"Synthesize extracted research findings into a clear, well-cited "
            f"answer for the user. {lang_instruction}"
        ),
        backstory=(
            "You are an expert at communicating complex research findings "
            "clearly and accurately. You always cite your sources and clearly "
            "distinguish between what the paper says versus general knowledge. "
            "You never fabricate information."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    retriever_task = Task(
        description=(
            f"The user asked: '{query}'\n\n"
            f"Here is the retrieved context from the research paper and related works:\n\n"
            f"{context}\n\n"
            f"Your job:\n"
            f"1. Identify which context passages are most relevant to the question\n"
            f"2. Extract the key facts, model results, and citations\n"
            f"3. Note any limitations or caveats mentioned in the paper\n"
            f"4. Output a structured summary of relevant findings with citations"
        ),
        expected_output=(
            "A structured list of relevant findings extracted from the context, "
            "each tagged with its source citation (page number or related work title)."
        ),
        agent=retriever_agent,
    )

    analyst_task = Task(
        description=(
            f"Using the structured findings from the Research Context Analyst, "
            f"write a comprehensive answer to: '{query}'\n\n"
            f"Requirements:\n"
            f"- {lang_instruction}\n"
            f"- Cite sources inline e.g. (Paper, Page 5) or (Related Work: ViT)\n"
            f"- If the answer is not in the context, say so clearly\n"
            f"- Keep the answer focused and under 400 words\n"
            f"- Use bullet points for lists of results or comparisons"
        ),
        expected_output=(
            "A clear, well-cited answer to the user's question, "
            f"written in {language}, grounded only in the provided research context."
        ),
        agent=analyst_agent,
        context=[retriever_task],
    )

    return Crew(
        agents=[retriever_agent, analyst_agent],
        tasks=[retriever_task, analyst_task],
        process=Process.sequential,
        verbose=False,
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_query(
    query: str,
    vector_store,
    api_key: str,
    language: str = "English",
    top_k: int = 5,
    user_id: str = "default",
) -> Dict[str, Any]:
    """
    Full pipeline: rate check → retrieve → agent crew → response.
    Returns dict with answer, citations, and metadata.
    """
    # Rate limit check
    allowed, remaining = check_rate_limit(user_id)
    if not allowed:
        return {
            "answer": "Daily request limit reached (10/day). Please try again tomorrow.",
            "citations": [],
            "remaining_requests": 0,
            "error": "rate_limited",
        }

    # Retrieve relevant chunks
    retrieved = vector_store.search(query, k=top_k)
    if not retrieved:
        return {
            "answer": "No relevant content found in the knowledge base.",
            "citations": [],
            "remaining_requests": remaining,
            "error": "no_results",
        }

    # Format context
    context = format_context(retrieved)

    # Extract citations for display
    citations = []
    for r in retrieved:
        meta = r["meta"]
        if meta.get("source") == "paper":
            citations.append(f"Paper — Page {meta.get('page', '?')}")
        else:
            citations.append(f"Related Work: {meta.get('title', '')}")

    # Run CrewAI pipeline
    llm = get_llm(api_key)
    crew = build_crew(llm, context, query, language)

    try:
        result = crew.kickoff()
        answer = str(result).strip()
    except Exception as e:
        # Fallback: direct LLM call without crew overhead
        from groq import Groq
        client = Groq(api_key=api_key)
        lang_note = "Respond in Arabic." if language == "Arabic" else "Respond in English."
        prompt = (
            f"Based only on this research context, answer the question.\n"
            f"{lang_note}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content

    return {
        "answer": answer,
        "citations": list(dict.fromkeys(citations)),  # deduplicate
        "remaining_requests": remaining,
        "retrieved_chunks": len(retrieved),
        "error": None,
    }
