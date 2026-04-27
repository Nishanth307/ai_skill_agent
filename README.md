# 🎯 AI Skill Assessment Agent (Minimalist Edition)

A high-performance, conversational AI agent built with **LangGraph** that verifies candidate proficiency in real-time. This version has been stripped of all legacy bloat, offering a hyper-lightweight, transient-storage architecture.

### ⚡ Highlights
- **Ultra-Lightweight**: Core repository reduced from ~3.2GB to just **112 KB**.
- **Transient Architecture**: Zero permanent database (PostgreSQL removed). Uses **Redis** for stateful sessions and **ChromaDB** for temporary runtime storage.
- **Privacy-by-Design**: All session data and vector embeddings are cleared once the assessment is discarded.
- **LLM Agnostic**: Seamlessly toggle between **Google Gemini** (1.5 Flash/Pro) and **Local Ollama** (Gemma 3).

---

## 🚀 Quick Start (Docker)

The fastest way to get up and running is via Docker Compose:

1. **Prerequisites**: [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.
2. **Environment**: Add your `GOOGLE_API_KEY` to the `.env` file (or set `LLM_BACKEND=ollama`).
3. **Launch**:
   ```bash
   docker compose up -d --build
   ```
4. **Access**:
   - **Frontend (UI)**: [http://localhost:8501](http://localhost:8501)
   - **Backend (API)**: [http://localhost:8000](http://localhost:8000)
   - **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🛠️ Stack & Optimization

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph) for stateful multi-agent workflows.
- **API**: FastAPI (Python 3.10-slim base images).
- **UI**: Streamlit for a responsive, real-time chat experience.
- **Memory**: Redis (via `langgraph-checkpoint-redis`) for conversational state management.
- **Vector DB**: ChromaDB for semantic resume parsing and retrieval.

---

## 🔧 Troubleshooting & Model Fallback

If you encounter `503 UNAVAILABLE` or "High Demand" errors during your demo:
1. **Switch Model**: Open `app/core/config.py` and change `GEMINI_MODEL` to a different option in the `GoogleModel` enum (e.g., switch from `Pro` to `Flash`).
2. **Restart**: Run `docker compose restart backend` to apply the change.
3. **Local Fail-safe**: If the internet is unstable or Gemini is down, set `LLM_BACKEND=ollama` in your `.env` to run the agent entirely on your local machine.

---

### Why the minimalist setup?
By removing heavy relational databases and optimizing Docker layers, we've achieved:
- **Instant Deployments**: Rebuilds and cold starts happen in seconds.
- **Cleaner Git History**: The `minimal-clean` branch contains only the essential source code.
- **Zero-Maintenance**: No database migrations or persistent volume management required.

---

## 💡 How to Demo
1. **Target Skills**: Paste a Job Description; the **Skill Extractor** will identify core requirements.
2. **Adaptive Chat**: Answer the agent's technical questions. It uses **Gemini 1.5 Pro** (default) with 10x retries for high stability.
3. **Gap Analysis**: View your proficiency scores and a automatically generated, week-by-week **Personalized Learning Plan**.

---
*Built with ❤️ for High-Speed AI Agent Workflows.*