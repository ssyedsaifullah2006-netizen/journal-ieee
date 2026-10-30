# Comment Intelligence Platform

A web-based AI platform that fetches social media comments (YouTube, Reddit), runs ML-powered analysis, and provides an interactive chat interface for exploring insights.

## Features

- **Comment Fetching** — Paste a YouTube or Reddit URL to fetch up to 1000 comments via official APIs.
- **Sentiment Analysis** — Classifies every comment locally with VADER and reports engagement-weighted descriptive diagnostics.
- **Topic Summary** — Generates a structured topic summary with Gemini when configured; it falls back to a deterministic local summary when unavailable.
- **Executive Summary** — Generates a structured summary with key themes, notable opinions, and controversies via Gemini.
- **RAG Chat** — Ask natural-language questions about the comments. Uses ChromaDB vector search + Gemini to provide grounded answers with source citations.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn |
| Frontend | Vanilla HTML/CSS/JS, Plotly.js |
| LLM | Google Gemini (via `google-genai` SDK; model selected in `config.py`) |
| Sentiment | NLTK VADER |
| Topics | Gemini structured summary with deterministic fallback |
| Vector DB | ChromaDB (persistent, local) |
| Data Sources | YouTube Data API v3, Reddit (PRAW) |

## Project Structure

```
├── server.py                  # FastAPI application entry point
├── config.py                  # Centralized configuration
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
├── static/
│   ├── index.html             # Single-page application
│   ├── style.css              # Dark-mode UI theme
│   └── app.js                 # Frontend logic
├── fetchers/
│   ├── base.py                # Comment dataclass + base fetcher ABC
│   ├── platform_detector.py   # URL detection + fetcher factory
│   ├── youtube_fetcher.py     # YouTube Data API v3 integration
│   └── reddit_fetcher.py      # PRAW-based Reddit fetcher
├── analyzers/
│   ├── sentiment.py           # RoBERTa sentiment classifier
│   ├── topics.py              # BERTopic topic modeler
│   └── summarizer.py          # Gemini-powered summarizer
├── rag/
│   ├── vector_store.py        # ChromaDB wrapper
│   └── qa_chain.py            # Gemini RAG Q&A chain
└── visualizations/
    ├── charts.py              # Plotly chart generators
    └── topic_network.py       # Topic visualization
```

## Research Paper: PACE Framework (IEEE Submission)

This repository includes the complete research implementation, reproducibility benchmarks, and IEEE journal submission package for:

**"Exposure-Conditioned Engagement Adjustment for Robust Consensus Estimation in Ranked Social Discussion Streams"**  
*Author:* Sayed Saifullah (Vellore Institute of Technology, India)

### Key Contributions & Novelty
- **Preferential-Attachment Corrected Endorsement (PACE)**: Formulates expected engagement as an empirical function of display rank ($r_i$) and arrival delay ($\Delta t_i$).
- **Duan's (1983) Smearing Calibration**: Corrects for retransformation bias under Jensen's inequality in log-linear engagement models.
- **Organic Residual Gating**: Standardizes engagement residuals $\delta_i$ via Median Absolute Deviation (MAD) and applies sigmoidal bounding to curb single-comment weight domination.
- **Auditable simulation study**: Benchmarked across deterministic $M=50$ Monte Carlo trials per scenario. The included YouTube CSV is an illustrative static semantic-analysis snapshot, not a verified causal or accuracy evaluation.

### Quick Reproducibility Commands
```bash
# 1. Run all 50 Monte Carlo trials across Scenarios 1-3 & YouTube Case Study
python experiments/reproduce_all_experiments.py

# 2. Generate 300 DPI IEEE publication figures
python experiments/generate_publication_figures.py
```

### Paper & Dataset Structure
- `latex/paper.tex`: Full IEEE Transactions two-column manuscript.
- `latex/references.bib`: BibTeX references; verify bibliographic metadata against publisher records before submission.
- `latex_submission_package.zip`: Ready-to-upload Overleaf project zip.
- `figures/`: 300 DPI publication figures (`fig1_monte_carlo_mae.png`, `fig2_ablation_study.png`, `fig3_weight_distributions.png`, `fig4_youtube_case_study.png`).
- `data/monte_carlo_trial_level_results.csv`: 150 individual trial-level benchmark observations.
- `data/ablation_trial_level_results.csv`: 150 individual trial-level ablation observations.
- `data/replicated_benchmark_results.csv`: Benchmark summary table (Mean $\pm$ SE, Paired Wilcoxon $p$-values).
- `data/replicated_ablation_results.csv`: Component ablation summary table.
- `data/youtube_corpus_100_comments.csv`: Legacy illustrative YouTube snapshot. See its provenance manifest; regenerate it with `experiments/fetch_real_youtube_corpus.py` before making API-provenance claims.

---

## Setup

### 1. Clone and install dependencies

```bash
cd c:/VIT/Semester5/ai/prjt
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key

# Optional — only needed for Reddit
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

### 3. Run the server

```bash
python -m uvicorn server:app --reload
```

Open **http://localhost:8000** in your browser.

## Usage

1. **Ingest** — Paste a YouTube or Reddit URL, set max comments, click "Fetch Comments".
2. **Analyze** — Select analysis modules (Sentiment, Topics, Executive Summary), click "Run Analysis".
3. **Chat** — Switch to the Chat tab. The app automatically builds a search index and lets you ask questions about the comments with AI-generated answers and source citations.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detect` | Detect platform from URL |
| POST | `/api/fetch` | Fetch comments and store in session |
| POST | `/api/analyze` | Run selected analysis modules |
| POST | `/api/rag/index` | Build vector search index |
| GET | `/api/rag/suggestions` | Get suggested questions |
| POST | `/api/rag/ask` | Ask a question (RAG) |
| GET | `/api/export/csv` | Export comments as CSV |
