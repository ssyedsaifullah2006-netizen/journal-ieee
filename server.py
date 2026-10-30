import logging
import json
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from fetchers.base import Comment
from fetchers.platform_detector import detect_platform, get_fetcher
from analyzers.lexicon_sentiment import LexiconSentimentAnalyzer
from analyzers.engagement_diagnostics import EngagementDiagnostics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=10)

app = FastAPI(title="Comment Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Session Management ─────────────────────────────────────────
sessions: Dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if session_id not in sessions:
        sessions[session_id] = {
            'comments': [],
            'metadata': {},
            'platform': '',
            'analysis_results': {},
            'vector_store': None,
            'qa_chain': None,
            'chat_history': [],
            'rag_status': 'idle',  # idle | building | ready | error
            'rag_error': None,
        }
    return sessions[session_id]

# ─── Pydantic Models ────────────────────────────────────────────
import config

class FetchRequest(BaseModel):
    url: str
    max_comments: int = config.MAX_COMMENTS_DEFAULT
    session_id: str

class AskRequest(BaseModel):
    session_id: str
    question: str

class SessionRequest(BaseModel):
    session_id: str

# ─── Static Files ───────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ─── Gemini-powered analysis ────────────────────────────────────
# Shared Gemini client (created once, reused across all requests)
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client

# URL-based cache to avoid re-analyzing the same content
_analysis_cache: Dict[str, dict] = {}

def _run_local_sentiment_and_diagnostics(comments_data: List[dict]) -> dict:
    """
    Run 100% local, deterministic sentiment classification and engagement diagnostics.
    Zero API calls, zero credit consumption.
    """
    analyzer = LexiconSentimentAnalyzer()
    classified = analyzer.analyze(comments_data)
    raw_dist = analyzer.get_raw_distribution(classified)
    
    diag = EngagementDiagnostics(n_permutations=1000)
    report = diag.run_full_diagnostics(classified, raw_dist)
    return report

def _run_analysis_gemini(comment_texts: List[str], metadata: dict, local_report: dict) -> dict:
    """
    Analyze comments summary using Gemini.
    Informed by the deterministic local sentiment diagnostics.
    If Gemini fails or credits are exhausted, falls back to algorithmic summary.
    """
    from google.genai import types
    from google.genai.errors import APIError
    import config
    import time

    raw = local_report.get("raw_distribution", {})
    weighted = local_report.get("weighted_distribution", {})
    flip = local_report.get("majority_flip_test", {})

    sentiment_context = (
        f"Local Sentiment Diagnostics:\n"
        f"- Raw (Count-based): {raw.get('positive_pct', 0)*100:.1f}% Pos, {raw.get('negative_pct', 0)*100:.1f}% Neg, {raw.get('neutral_pct', 0)*100:.1f}% Neu\n"
        f"- Engagement-Weighted: {weighted.get('positive', 0)*100:.1f}% Pos, {weighted.get('negative', 0)*100:.1f}% Neg, {weighted.get('neutral', 0)*100:.1f}% Neu\n"
        f"- Majority Flip: {flip.get('flip_detected')} (p-value: {flip.get('p_value')})\n"
    )

    max_to_analyze = min(len(comment_texts), config.MAX_ANALYSIS_COMMENTS)
    comments_block = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(comment_texts[:max_to_analyze]))

    meta_info = f"\nSource: {metadata.get('title', '')}" if metadata and metadata.get('title') else ""

    prompt = f"""Synthesize the audience reception for these social media comments.{meta_info}

{sentiment_context}

SAMPLE COMMENTS:
{comments_block}

Return a JSON object with this EXACT structure:
{{
  "topics": [
    {{"name": "<topic name>", "count": <approx count>, "description": "<1 sentence summary>"}}
  ],
  "summary": "<A detailed 3-5 paragraph executive summary of what people are saying, addressing whether popular/liked comments diverge from casual comments>",
  "key_themes": ["<theme 1>", "<theme 2>", ...],
  "notable_opinions": ["<opinion 1>", "<opinion 2>", ...],
  "controversies": ["<controversy 1>", ...]
}}"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "topics": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "count": {"type": "INTEGER"},
                        "description": {"type": "STRING"}
                    },
                    "required": ["name", "count", "description"]
                }
            },
            "summary": {"type": "STRING"},
            "key_themes": {"type": "ARRAY", "items": {"type": "STRING"}},
            "notable_opinions": {"type": "ARRAY", "items": {"type": "STRING"}},
            "controversies": {"type": "ARRAY", "items": {"type": "STRING"}}
        },
        "required": ["topics", "summary", "key_themes", "notable_opinions", "controversies"]
    }

    try:
        client = _get_gemini_client()
        safety = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]
        gen_config = types.GenerateContentConfig(
            safety_settings=safety,
            response_mime_type="application/json",
            response_schema=schema
        )
        response = client.models.generate_content(
            model=config.GEMINI_ANALYSIS_MODEL,
            contents=prompt,
            config=gen_config
        )
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            parsed = json.loads(response.candidates[0].content.parts[0].text)
            return parsed
    except Exception as e:
        logger.warning(f"Gemini summary generation skipped/failed ({e}). Using algorithmic fallback.")

    # Algorithmic offline fallback summary
    top_stat = f"{raw.get('positive', 0)} positive, {raw.get('negative', 0)} negative, and {raw.get('neutral', 0)} neutral comments."
    flip_note = (
        f"Statistical diagnostics reveal a majority consensus shift: the raw comment volume leaned {flip.get('raw_majority')}, "
        f"but community engagement (likes & replies) significantly amplifies {flip.get('weighted_majority')} sentiment (p={flip.get('p_value')})."
        if flip.get("flip_detected") else
        f"Engagement diagnostics indicate general alignment between casual comment volume and community-endorsed opinions."
    )
    fallback_summary = (
        f"### Audience Reception Analysis\n\n"
        f"Evaluated {raw.get('total', len(comment_texts))} comments via Engagement-Calibrated Sentiment Diagnostics. "
        f"The raw unweighted distribution recorded {top_stat}\n\n"
        f"{flip_note}\n\n"
        f"The reliability score for this distribution is rated **{local_report.get('reliability_metric', {}).get('reliability_level', 'MODERATE')}** "
        f"(Gini inequality G={local_report.get('reliability_metric', {}).get('gini', 0):.2f})."
    )

    pos_count = raw.get("positive", 0)
    neu_count = raw.get("neutral", 0)
    neg_count = raw.get("negative", 0)
    total_count = raw.get("total", len(comment_texts))

    fallback_topics = [
        {"name": "Positive Reception & Appreciation", "count": pos_count, "description": "Comments expressing praise, agreement, or positive engagement"},
        {"name": "Neutral & Discussion Commentary", "count": neu_count, "description": "Factual observations, queries, and conversational remarks"},
        {"name": "Critical Feedback & Complaints", "count": neg_count, "description": "Critiques, negative reactions, or points of contention"}
    ]
    fallback_topics = [t for t in fallback_topics if t["count"] > 0]
    if not fallback_topics:
        fallback_topics = [{"name": "Audience Reception", "count": total_count, "description": "General viewer discussion"}]

    return {
        "topics": fallback_topics,
        "summary": fallback_summary,
        "key_themes": ["Audience Feedback", "Community Engagement", "Sentiment Drift"] if flip.get("flip_detected") else ["Audience Feedback", "Discussion Consensus"],
        "notable_opinions": ["Viewer opinions highlighted in engagement weighting analysis."],
        "controversies": ["Disparity between raw comment frequency and high-engagement comment reaction."] if flip.get("flip_detected") else []
    }


def _build_index(comments: list, session_id: str, session: dict):
    """Build ChromaDB vector index for RAG. Updates session status."""
    from rag.vector_store import VectorStore
    from rag.qa_chain import QAChain

    try:
        session['rag_status'] = 'building'

        collection_name = f"comments_{session_id.replace('-', '')}"
        vs = VectorStore(collection_name=collection_name)
        vs.clear()

        docs = [
            {"id": f"comment_{i}", "text": c.text, "metadata": {"author": c.author or "Unknown", "platform": c.platform or "", "likes": c.likes or 0, "timestamp": c.timestamp or ""}}
            for i, c in enumerate(comments)
        ]
        # Single call — GeminiEmbeddingFunction handles internal batching of 100 per API call
        vs.add_comments(docs)

        qa = QAChain(vector_store=vs)
        session['vector_store'] = vs
        session['qa_chain'] = qa
        session['rag_status'] = 'ready'
        session['rag_error'] = None
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Background index error: {e}")
        session['rag_status'] = 'error'
        session['rag_error'] = str(e)


# ─── Main Endpoint: Fetch + Analyze + Index ─────────────────────

@app.post("/api/fetch")
async def api_fetch(req: FetchRequest):
    try:
        # 1. Fetch comments
        fetcher = get_fetcher(req.url)
        metadata = fetcher.get_metadata(req.url)
        comments = fetcher.fetch_comments(req.url, max_comments=req.max_comments)
        platform = detect_platform(req.url)

        session = get_session(req.session_id)
        session['comments'] = comments
        session['metadata'] = metadata
        session['platform'] = platform
        session['analysis_results'] = {}
        session['vector_store'] = None
        session['qa_chain'] = None
        session['chat_history'] = []
        session['rag_status'] = 'idle'
        session['rag_error'] = None

        sample = [vars(c) for c in comments[:5]]

        # 2. Representative systematic stratified sampling across the comment stream (prevents engagement bias)
        n_total = len(comments)
        max_analyze = min(n_total, config.MAX_ANALYSIS_COMMENTS)
        if n_total <= max_analyze:
            sampled_comments = list(comments)
        else:
            # Uniformly step across the rank/arrival sequence to ensure representative discourse coverage
            step = n_total / max_analyze
            sampled_comments = [comments[int(i * step)] for i in range(max_analyze)]

        comment_texts = [c.text for c in sampled_comments]
        comments_data = [vars(c) for c in comments]

        loop = asyncio.get_running_loop()

        # 3. Step A: 100% LOCAL & DETERMINISTIC sentiment + diagnostics (zero API cost)
        local_report = await loop.run_in_executor(executor, _run_local_sentiment_and_diagnostics, comments_data)

        # 3. Step B: Generate contextual executive summary & topics
        cache_key = req.url.strip().lower()
        if cache_key in _analysis_cache:
            logger.info(f"Using cached analysis for {req.url}")
            analysis_results = _analysis_cache[cache_key]
        else:
            summary_payload = await loop.run_in_executor(
                executor, _run_analysis_gemini, comment_texts, metadata, local_report
            )
            # Combine deterministic sentiment + diagnostics with summary/topics
            raw_dist = local_report.get("raw_distribution", {})
            analysis_results = {
                "sentiment": {
                    "positive": raw_dist.get("positive", 0),
                    "negative": raw_dist.get("negative", 0),
                    "neutral": raw_dist.get("neutral", 0),
                    "total": raw_dist.get("total", len(comments)),
                    "positive_pct": raw_dist.get("positive_pct", 0),
                    "negative_pct": raw_dist.get("negative_pct", 0),
                    "neutral_pct": raw_dist.get("neutral_pct", 0),
                },
                "weighted_sentiment": local_report.get("weighted_distribution", {}),
                "sentiment_amplification_index": local_report.get("sentiment_amplification_index", {}),
                "reliability_metric": local_report.get("reliability_metric", {}),
                "majority_flip_test": local_report.get("majority_flip_test", {}),
                "robustness_analysis": local_report.get("robustness_analysis", {}),
                "topics": summary_payload.get("topics", []),
                "summary": summary_payload.get("summary", ""),
                "key_themes": summary_payload.get("key_themes", []),
                "notable_opinions": summary_payload.get("notable_opinions", []),
                "controversies": summary_payload.get("controversies", [])
            }
            _analysis_cache[cache_key] = analysis_results

        session['analysis_results'] = analysis_results

        # 4. Build index AFTER analysis completes (sequential, not parallel)
        #    This avoids hammering the API with generation + embedding simultaneously
        def _background_index():
            _build_index(sorted_comments, req.session_id, session)

        loop.run_in_executor(executor, _background_index)

        return {
            "count": len(comments),
            "metadata": metadata,
            "platform": platform,
            "sample": sample,
            "analysis": analysis_results,
            "comments_analyzed": min(len(comment_texts), config.MAX_ANALYSIS_COMMENTS),
            "rag_ready": session.get('rag_status') == 'ready',
            "suggested_questions": ["What is the general consensus?", "Are there any complaints?", "What do people praise the most?"]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fetch pipeline: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed. Please check the URL and try again.")


@app.get("/api/session/{session_id}")
async def api_get_session(session_id: str):
    if session_id not in sessions:
        return {"count": 0, "metadata": None, "platform": None, "has_analysis": False, "analysis_results": {}, "rag_ready": False, "rag_status": "idle"}
    s = sessions[session_id]
    return {
        "count": len(s.get('comments', [])),
        "metadata": s.get('metadata'),
        "platform": s.get('platform'),
        "has_analysis": bool(s.get('analysis_results')),
        "analysis_results": s.get('analysis_results', {}),
        "rag_ready": s.get('rag_status') == 'ready',
        "rag_status": s.get('rag_status', 'idle'),
    }


@app.get("/api/rag/status/{session_id}")
async def api_rag_status(session_id: str):
    """Check whether RAG indexing is complete for a session."""
    session = get_session(session_id)
    return {
        "status": session.get('rag_status', 'idle'),
        "error": session.get('rag_error'),
        "ready": session.get('rag_status') == 'ready',
    }


@app.post("/api/rag/ask")
async def api_rag_ask(req: AskRequest):
    try:
        session = get_session(req.session_id)
        qa = session.get('qa_chain')
        if not qa:
            rag_status = session.get('rag_status', 'idle')
            if rag_status == 'building':
                raise HTTPException(status_code=202, detail="Search index is still building. Please wait a moment and try again.")
            raise HTTPException(status_code=400, detail="No data loaded. Fetch comments first.")

        history = session.get('chat_history', [])
        result = qa.ask(
            question=req.question,
            history=history,
            global_context=session.get('analysis_results')
        )

        answer = result.get('answer', '')
        source_comments = result.get('source_comments', [])
        relevance_scores = result.get('relevance_scores', [])
        source_metadatas = result.get('source_metadatas', [])

        sources = []
        for i, text in enumerate(source_comments):
            score = relevance_scores[i] if i < len(relevance_scores) else 0.0
            meta = source_metadatas[i] if i < len(source_metadatas) else {}
            sources.append({
                "text": text if isinstance(text, str) else str(text),
                "author": meta.get("author", "Unknown") if isinstance(meta, dict) else "Unknown",
                "likes": meta.get("likes", 0) if isinstance(meta, dict) else 0,
                "relevance_score": round(score, 3)
            })

        history.append((req.question, answer))
        # Bound history to last 10 exchanges
        session['chat_history'] = history[-10:]

        return {"answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during Q&A: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer. Please try again.")


@app.post("/api/rag/clear")
async def api_rag_clear(req: SessionRequest):
    session = get_session(req.session_id)
    session['chat_history'] = []
    return {"status": "cleared"}


@app.get("/api/export/csv")
async def api_export_csv(session_id: str):
    try:
        session = get_session(session_id)
        comments = session.get('comments', [])
        if not comments:
            raise HTTPException(status_code=400, detail="No data to export")

        df = pd.DataFrame([vars(c) for c in comments])
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        stream.seek(0)

        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=export.csv"
        return response
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from config import SERVER_HOST, SERVER_PORT
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
