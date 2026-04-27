# 🎯 AI Skill Agent: Production Setup

A conversational AI agent that assesses candidate skills from a resume against a job description and generates a personalized learning plan.

## 🚀 Quick Start (Docker - Recommended)

1. **Prerequisites**: Ensure you have [Docker](https://www.docker.com/products/docker-desktop/) installed.
2. **Setup**: Run `./start.sh`.
3. **Configure**: Add your `GOOGLE_API_KEY` to the `.env` file created in the root.
4. **Access**: UI at [http://localhost:8501](http://localhost:8501).

---

## 🛠️ Manual Setup (Fallback)

If Docker builds are too slow (e.g., due to heavy ML libraries), you can run the services natively:

1. **Start Infrastructure**:
   ```bash
   # Starts only Postgres and Redis
   docker compose up -d postgres redis
   ```
2. **Setup Virtual Env**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure**:
   ```bash
   cp .env.example .env
   # Update GOOGLE_API_KEY and set:
   # DATABASE_URL=postgresql+asyncpg://skillagent:skillagent@localhost:5432/skillagentdb
   # REDIS_URL=redis://localhost:6379
   ```
4. **Run Services**:
   - **Backend**: `uvicorn main:app --reload`
   - **Frontend**: `streamlit run app/ui/streamlit_app.py`

## 🔌 Access Points

- **Frontend (UI)**: [http://localhost:8501](http://localhost:8501)
- **Backend (API)**: [http://localhost:8000](http://localhost:8000)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

## 💡 Key Notes

- **Model Compatibility**: Currently optimized for the **Gemini 2.0/3.1** series (via Google AI Studio) to provide a high-quality free tier experience.
- **Infrastructure**: The system automatically spins up **PostgreSQL** (data storage), **Redis** (session persistence), **ChromaDB** (vector search), and the dual-service application.
- **Anonymous Assessments**: You can start an assessment directly with a Job Description without registering personal details.

## 🛠️ Maintenance

- **Stop the app**: `docker compose down`
- **View logs**: `docker compose logs -f`
- **Rebuild**: `./start.sh` (handles dependency updates automatically)

---
*Built with LangGraph, FastAPI, and Streamlit.*
git@github.com:USERNAME/REPO.git