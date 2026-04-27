"""
app/ui/streamlit_app.py

Streamlit UI for the Skill Assessment Agent.

Run:
    streamlit run app/ui/streamlit_app.py

Pages:
  1. 🏠 Home          — intro and navigation
  2. 📋 New Assessment — upload resume + JD → start assessment
  3. 💬 Assessment     — conversational Q&A with the agent
  4. 📊 Results        — gap analysis + learning plan
"""

import os
import streamlit as st
import httpx
import json
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
st.set_page_config(
    page_title="Skill Assessment Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Shared HTTP client ────────────────────────────────────────────────────────
def api_post(endpoint: str, data: dict = None, files=None) -> dict | None:
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            if files:
                r = client.post(f"{API_BASE}{endpoint}", data=data, files=files)
            else:
                r = client.post(f"{API_BASE}{endpoint}", json=data)
        
        if r.status_code in (200, 201, 202):
            try:
                return r.json()
            except Exception:
                return {"detail": "Error parsing JSON response", "text": r.text}
        
        try:
            error_json = r.json()
            error_detail = error_json.get('detail', r.text)
        except Exception:
            error_detail = r.text or f"HTTP {r.status_code}"
            
        st.error(f"API error {r.status_code}: {error_detail}")
        if r.status_code == 500:
            st.expander("Show Technical Details").code(r.text)
        return None
    except httpx.ConnectError:
        st.error("❌ Cannot connect to the API. Is the server running? (`uvicorn main:app --reload`)")
        return None


def api_stream(endpoint: str, data: dict):
    try:
        with httpx.stream("POST", f"{API_BASE}{endpoint}", json=data, timeout=120.0) as r:
            if r.status_code != 200:
                try:
                    err = r.read().decode()
                    st.error(f"Streaming error {r.status_code}: {err}")
                except:
                    st.error(f"Streaming error {r.status_code}")
                return
            
            for chunk in r.iter_text():
                yield chunk
    except Exception as e:
        st.error(f"Streaming connection error: {e}")


def api_get(endpoint: str, suppress_404: bool = False) -> dict | None:
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            r = client.get(f"{API_BASE}{endpoint}")
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return {"detail": "Error parsing JSON response", "text": r.text}
        
        try:
            error_json = r.json()
            error_detail = error_json.get('detail', r.text)
        except Exception:
            error_detail = r.text or f"HTTP {r.status_code}"
            
        if suppress_404 and r.status_code == 404:
            return None
        st.error(f"API error {r.status_code}: {error_detail}")
        if r.status_code == 500:
            st.expander("Show Technical Details").code(r.text)
        return None
    except httpx.ConnectError:
        st.error("❌ Cannot connect to the API.")
        return None


# ── Sidebar navigation ────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.title("🎯 Skill Assessment Agent")
        st.markdown("---")

        page = st.radio(
            "Navigate",
            ["🏠 Home", "📋 New Assessment", "💬 Assessment Chat", "📊 Results"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Show active session info if exists
        if st.session_state.get("thread_id"):
            st.success(f"**Active Session**")
            st.code(st.session_state["thread_id"][:16] + "...", language=None)
            if st.button("🗑️ Clear Session"):
                for key in ["thread_id", "assessment_id", "candidate_id", "chat_history", "assessment_done", "last_progress"]:
                    st.session_state.pop(key, None)
                st.rerun()

        st.markdown("---")
        st.caption("Built with LangGraph + LangChain")

    return page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

def render_home():
    st.title("🎯 AI-Powered Skill Assessment Agent")
    st.markdown("""
    A resume tells you what someone *claims* to know — not how well they actually know it.

    This agent:
    1. **Extracts** required skills from a Job Description
    2. **Conversationally assesses** real proficiency on each skill
    3. **Identifies gaps** and adjacent skills you can realistically bridge
    4. **Generates a personalised learning plan** with curated resources and time estimates
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", "4", help="Skill Extractor → Assessor → Gap Analyzer → Plan Generator")
    with col2:
        st.metric("LLM Calls", "~10–20", help="Per full assessment session")
    with col3:
        st.metric("Skills Assessed", "Up to 10", help="Configurable via MAX_SKILLS_TO_ASSESS")
    with col4:
        st.metric("Questions/Skill", "1–3", help="Adaptive — stops when enough signal gathered")

    st.markdown("---")
    st.subheader("How to use")
    st.markdown("""
    1. Go to **📋 New Assessment** → paste your resume and job description
    2. The agent extracts required skills automatically
    3. Switch to **💬 Assessment Chat** → answer the agent's questions naturally
    4. Once done, go to **📊 Results** → view your gap analysis and learning plan
    """)

    # Quick API health check
    try:
        r = httpx.get("http://localhost:8000/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success(f"✅ API is running | Backend: `{data.get('llm_backend', '?').upper()}`")
        else:
            st.warning("⚠️ API returned non-200")
    except Exception:
        st.error("❌ API not reachable. Start with: `uvicorn main:app --reload`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — NEW ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

def render_new_assessment():
    st.title("📋 Start New Assessment")

    with st.form("assessment_form"):
        # Candidate details removed for anonymity
        name = "Anonymous"
        email = None

        st.subheader("2. Resume")
        resume_tab1, resume_tab2 = st.tabs(["📄 Upload File (PDF/DOCX)", "✏️ Paste Text"])

        with resume_tab1:
            uploaded_file = st.file_uploader("Upload resume", type=["pdf", "docx"])

        with resume_tab2:
            resume_text = st.text_area(
                "Paste resume text",
                height=200,
                placeholder="Paste your full resume here...",
            )

        st.subheader("3. Job Description")
        job_title = st.text_input("Job Title (optional)", placeholder="Backend Engineer")
        jd_text = st.text_area(
            "Paste the Job Description",
            height=250,
            placeholder="Paste the full job description here...",
        )

        submitted = st.form_submit_button("🚀 Start Assessment", use_container_width=True)

    if submitted:
        if not jd_text or len(jd_text.strip()) < 50:
            st.error("Please paste a job description (at least 50 characters)")
            return

        # ── Step 1: Register candidate ─────────────────────────────────────
        with st.spinner("Registering candidate..."):
            if uploaded_file:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                candidate = api_post("/candidates/upload", data={"name": name, "email": email}, files=files)
            else:
                if not resume_text or len(resume_text.strip()) < 50:
                    st.error("Please provide a resume (upload or paste)")
                    return
                candidate = api_post("/candidates", {"name": name, "email": email, "resume_text": resume_text})

        if not candidate:
            return

        st.session_state["candidate_id"] = candidate["id"]

        # ── Step 2: Start assessment ────────────────────────────────────────
        with st.spinner("🔍 Extracting skills from JD and resume... (this takes ~10s)"):
            assessment = api_post("/assessments", {
                "candidate_id": candidate["id"],
                "job_description": jd_text,
                "job_title": job_title or None,
            })

        if not assessment:
            return

        st.session_state["assessment_id"] = assessment["id"]
        st.session_state["thread_id"] = assessment["thread_id"]
        st.session_state["chat_history"] = []
        st.session_state["assessment_done"] = False

        st.success("✅ Assessment started!")
        st.info("👉 Switch to **💬 Assessment Chat** in the sidebar to begin the conversation.")
        st.json({"thread_id": assessment["thread_id"], "status": assessment["status"]})


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ASSESSMENT CHAT
# ══════════════════════════════════════════════════════════════════════════════

def render_chat():
    st.title("💬 Assessment Chat")

    if not st.session_state.get("thread_id"):
        st.warning("No active assessment session. Go to **📋 New Assessment** first.")
        return

    if st.session_state.get("assessment_done"):
        st.success("✅ Assessment complete! Go to **📊 Results** to see your learning plan.")
        return

    assessment_id = st.session_state.get("assessment_id")
    if assessment_id:
        assessment_state = api_get(f"/assessments/{assessment_id}")
        if assessment_state and assessment_state.get("status") == "failed":
            error_msg = assessment_state.get("error", "Assessment initialization failed.")
            st.error(error_msg)
            if "Gemini" in error_msg:
                st.info("💡 **Tip:** You can switch to a local model by setting `LLM_BACKEND=ollama` in your `.env` file.")
            return

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("progress"):
                    st.caption(msg["progress"])

    # Progress bar removed for direct flow

    # If no messages yet, fetch the opening question
    if not st.session_state.get("chat_history"):
        with st.spinner("Loading assessment questions..."):
            _fetch_opening_message()

    # User input
    if user_input := st.chat_input("Type your answer here..."):
        # Add user message to history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # STREAM RESPONSES IN REAL TIME
            full_response = st.write_stream(api_stream("/assessments/chat", {
                "thread_id": st.session_state["thread_id"],
                "message": user_input,
            }))
            
            # After stream, we check if complete and get metadata
            # We make a quick call to the GET /assessments/{id} endpoint
            assessment = api_get(f"/assessments/{st.session_state['assessment_id']}")
            
            is_complete = False
            progress = None
            if assessment:
                is_complete = assessment.get("status") == "completed"
                if assessment.get("progress"):
                    st.session_state["last_progress"] = assessment["progress"]
                    st.rerun()  # Rerun to update progress bar immediately
            
            agent_msg = {
                "role": "assistant",
                "content": full_response,
                "progress": None,
            }
            st.session_state["chat_history"].append(agent_msg)

            if is_complete:
                st.session_state["assessment_done"] = True
                st.success("🎉 Assessment complete!")

                # Auto-trigger plan generation
                if st.session_state.get("assessment_id"):
                    with st.spinner("Generating learning plan..."):
                        api_post(f"/plans/generate/{st.session_state['assessment_id']}", {})
                    st.rerun()


def _fetch_opening_message():
    """
    The agent sends the first message (intro + first question).
    """
    for _ in range(30):
        response = api_post("/assessments/chat", {
            "thread_id": st.session_state["thread_id"],
            "message": "__init__",
        })
        if response and response.get("message"):
            message = response["message"].strip()
            # If we get a real message (not the generic "preparing" one), show it
            if message and "preparing your" not in message.lower() and "one moment please" not in message.lower():
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": response["message"],
                    "progress": response.get("progress"),
                })
                st.session_state["last_progress"] = response.get("progress")
                st.rerun()
                return
        time.sleep(1)

    st.info("Still setting up your assessment. Please hang tight, or try refreshing the page.")


def _word_stream(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.015)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def render_results():
    st.title("📊 Assessment Results & Learning Plan")

    if not st.session_state.get("assessment_id"):
        st.warning("No active assessment. Complete an assessment first.")
        return

    assessment_id = st.session_state["assessment_id"]

    # ── Assessment summary ─────────────────────────────────────────────────────
    with st.spinner("Loading results..."):
        assessment = api_get(f"/assessments/{assessment_id}")

    if not assessment:
        return

    status = assessment.get("status", "unknown")
    if status == "analyzing":
        st.info("🔬 **The AI is currently analyzing your performance and generating your personalized learning plan.** This typically takes 10-20 seconds. This page will refresh automatically.")
        time.sleep(5)
        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        score = assessment.get("overall_score") or 0
        st.metric("Overall Score", f"{score:.0f} / 100")
    with col2:
        gaps = len(assessment.get("skill_gaps") or [])
        st.metric("Skill Gaps", gaps)
    with col3:
        st.metric("Status", status.title())

    st.markdown("---")

    # ── Skill scores ───────────────────────────────────────────────────────────
    if assessment.get("assessment_results"):
        st.subheader("📈 Skill Assessment Scores")
        results = assessment["assessment_results"]

        # Build score bars
        for r in results:
            score = r.get("score", 0)
            skill = r.get("skill", "Unknown")
            confidence = r.get("confidence", "")
            color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(score / 5, text=f"{color} {skill}")
            with col2:
                st.caption(f"{score}/5 · {confidence} confidence")

        st.markdown("---")
        
        # Build category lists (case-insensitive)
        strong = [r["skill"] for r in results if r.get("rating", "").upper() == "STRONG"]
        average = [r["skill"] for r in results if r.get("rating", "").upper() == "AVERAGE"]
        weak = [r["skill"] for r in results if r.get("rating", "").upper() == "WEAK"]
        missing = [r["skill"] for r in results if r.get("rating", "").upper() == "MISSING"]

        # Only show this summary if we don't have the detailed report yet, or keep it very brief
        st.subheader("📊 Quick Summary")
        cols = st.columns(4)
        with cols[0]: st.success(f"**Strong:** {len(strong)}")
        with cols[1]: st.info(f"**Average:** {len(average)}")
        with cols[2]: st.warning(f"**Weak:** {len(weak)}")
        with cols[3]: st.error(f"**Missing:** {len(missing)}")
        st.markdown("---")

    # ── Skill gaps ─────────────────────────────────────────────────────────────
    if assessment.get("skill_gaps"):
        st.subheader("🔴 Identified Gaps")
        for gap in assessment["skill_gaps"]:
            severity_icon = {"critical": "🔴", "moderate": "🟡", "minor": "🟢"}.get(
                gap.get("gap_severity", "moderate"), "🟡"
            )
            adjacent = " *(adjacent — buildable from your existing skills)*" if gap.get("is_adjacent") else ""
            st.markdown(
                f"{severity_icon} **{gap['skill']}** — "
                f"Score: {gap['current_score']}/5 · "
                f"Required: {gap['required_level']} · "
                f"{gap['gap_severity'].title()}{adjacent}"
            )

        st.markdown("---")

    # ── Learning plan ─────────────────────────────────────────────────────────
    st.subheader("📚 Personalised Learning Plan")

    with st.spinner("Loading learning plan..."):
        plan = api_get(f"/plans/{assessment_id}", suppress_404=True)

    if not plan:
        if assessment.get("status") == "completed":
            if st.button("🔄 Generate Learning Plan"):
                with st.spinner("Generating..."):
                    api_post(f"/plans/generate/{assessment_id}", {})
                st.rerun()
        else:
            st.info("Complete the assessment to unlock the learning plan.")
        return

    # Summary
    st.info(plan.get("summary", ""))

    # Detailed Assessment Report (Phase 7)
    if plan.get("assessment_report"):
        with st.expander("📝 Detailed Assessment Report (Phases 1-7)", expanded=True):
            st.markdown(plan["assessment_report"])

    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Hours", plan.get("total_hours", 0))
    with col2:
        st.metric("Duration", f"{plan.get('duration_weeks', 0)} weeks")
    with col3:
        st.metric("Skills to Learn", len(plan.get("skill_items", [])))

    st.markdown("---")

    # Skill items
    for item in plan.get("skill_items", []):
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(item.get("priority", "medium"), "🟡")
        with st.expander(
            f"{priority_icon} {item['skill']} — {item['estimated_hours']}h · "
            f"Score: {item['current_score']}/5 → Target: {item['target_score']}/5"
        ):
            st.markdown(f"**Why this matters:** {item.get('rationale', '')}")

            if item.get("milestones"):
                st.markdown("**Milestones:**")
                for m in item["milestones"]:
                    st.markdown(f"- {m}")

            if item.get("resources"):
                st.markdown("**Resources:**")
                for res in item["resources"]:
                    free_tag = "🆓" if res.get("is_free") else "💰"
                    type_icon = {
                        "course": "🎓", "book": "📖", "doc": "📄",
                        "video": "▶️", "practice": "💻",
                    }.get(res.get("type", ""), "📌")
                    url = res.get("url")
                    title = res.get("title", "Resource")
                    link = f"[{title}]({url})" if url else title
                    st.markdown(
                        f"{type_icon} {free_tag} {link} "
                        f"· {res.get('platform', '')} "
                        f"· ~{res.get('estimated_hours', 0)}h"
                    )

    # Weekly schedule
    if plan.get("weekly_schedule"):
        st.markdown("---")
        st.subheader("🗓️ Weekly Schedule")
        for week in plan["weekly_schedule"]:
            with st.expander(f"Week {week['week']} — {', '.join(week.get('focus_skills', []))}"):
                st.markdown(f"**Daily commitment:** {week.get('hours_per_day', 0)}h/day")
                st.markdown("**Goals:**")
                for goal in week.get("goals", []):
                    st.markdown(f"- {goal}")

    # Export and Delete
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download Plan as JSON",
            data=json.dumps(plan, indent=2, default=str),
            file_name=f"learning_plan_{assessment_id[:8]}.json",
            mime="application/json",
        )
    with col2:
        if st.button("🗑️ Discard All Session Data", type="secondary", use_container_width=True, help="Permanently deletes your assessment and personal data from the system."):
            with st.spinner("Deleting data..."):
                r = httpx.delete(f"{API_BASE}/assessments/{assessment_id}")
                if r.status_code == 200:
                    st.success("All data discarded successfully.")
                    for key in ["thread_id", "assessment_id", "candidate_id", "chat_history", "assessment_done", "last_progress"]:
                        st.session_state.pop(key, None)
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Failed to delete data. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Initialise session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    page = render_sidebar()

    if page == "🏠 Home":
        render_home()
    elif page == "📋 New Assessment":
        render_new_assessment()
    elif page == "💬 Assessment Chat":
        render_chat()
    elif page == "📊 Results":
        render_results()


if __name__ == "__main__":
    main()