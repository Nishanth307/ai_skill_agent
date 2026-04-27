"""
app/memory/storage.py

Thread-safe in-memory storage for session-based evaluation.
Replaces Postgres for this version.
"""
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import redis
from app.core.config import settings

class SimpleStorage:
    def __init__(self):
        self._candidates: Dict[uuid.UUID, Dict[str, Any]] = {}
        self._assessments: Dict[uuid.UUID, Dict[str, Any]] = {}
        self._learning_plans: Dict[uuid.UUID, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
        # Redis client for clearing checkpointer data
        try:
            self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
        except Exception:
            self._redis = None

    # --- Candidates ---
    async def create_candidate(self, name: Optional[str] = None, email: Optional[str] = None, resume_text: str = "", chroma_doc_id: Optional[str] = None) -> Dict[str, Any]:
        async with self._lock:
            # If email is provided, check for uniqueness
            if email:
                for cand in self._candidates.values():
                    if cand.get("email") == email:
                        raise ValueError("Candidate with this email already exists")
            
            cand_id = uuid.uuid4()
            # If no email, use a unique placeholder
            final_email = email or f"anonymous_{cand_id}@example.com"
            final_name = name or "Anonymous"
            
            candidate = {
                "id": cand_id,
                "name": final_name,
                "email": final_email,
                "resume_text": resume_text,
                "chroma_doc_id": chroma_doc_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            self._candidates[cand_id] = candidate
            return candidate

    async def get_candidate(self, candidate_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        return self._candidates.get(candidate_id)

    async def get_candidate_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        for cand in self._candidates.values():
            if cand["email"] == email:
                return cand
        return None

    async def delete_candidate(self, candidate_id: uuid.UUID):
        async with self._lock:
            self._candidates.pop(candidate_id, None)

    # --- Assessments ---
    async def create_assessment(self, candidate_id: uuid.UUID, job_description: str, job_title: Optional[str] = None, thread_id: Optional[str] = None) -> Dict[str, Any]:
        async with self._lock:
            if candidate_id not in self._candidates:
                raise ValueError("Candidate not found")
            
            asmt_id = uuid.uuid4()
            assessment = {
                "id": asmt_id,
                "candidate_id": candidate_id,
                "job_description": job_description,
                "job_title": job_title,
                "status": "pending",
                "thread_id": thread_id or str(uuid.uuid4()),
                "required_skills": [],
                "assessment_results": [],
                "skill_gaps": [],
                "overall_score": None,
                "completed_at": None,
                "created_at": datetime.utcnow()
            }
            self._assessments[asmt_id] = assessment
            return assessment

    async def get_assessment(self, assessment_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        return self._assessments.get(assessment_id)

    async def get_assessment_by_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        for asmt in self._assessments.values():
            if asmt["thread_id"] == thread_id:
                return asmt
        return None

    async def update_assessment(self, assessment_id: uuid.UUID, **kwargs) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if assessment_id not in self._assessments:
                return None
            self._assessments[assessment_id].update(kwargs)
            return self._assessments[assessment_id]

    async def delete_assessment(self, assessment_id: uuid.UUID):
        async with self._lock:
            self._assessments.pop(assessment_id, None)
            # Find and delete associated learning plan
            plan_to_del = None
            for pid, plan in self._learning_plans.items():
                if plan["assessment_id"] == assessment_id:
                    plan_to_del = pid
                    break
            if plan_to_del:
                self._learning_plans.pop(plan_to_del, None)

    # --- Learning Plans ---
    async def create_learning_plan(self, assessment_id: uuid.UUID, plan_data: Dict[str, Any], total_hours: Optional[int] = None, duration_weeks: Optional[int] = None) -> Dict[str, Any]:
        async with self._lock:
            if assessment_id not in self._assessments:
                raise ValueError("Assessment not found")
            
            plan_id = uuid.uuid4()
            plan = {
                "id": plan_id,
                "assessment_id": assessment_id,
                "plan_data": plan_data,
                "total_hours": total_hours,
                "duration_weeks": duration_weeks,
                "created_at": datetime.utcnow()
            }
            self._learning_plans[plan_id] = plan
            # Link plan to assessment
            self._assessments[assessment_id]["learning_plan_id"] = plan_id
            return plan

    async def get_learning_plan(self, plan_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        return self._learning_plans.get(plan_id)

    async def get_learning_plan_by_assessment(self, assessment_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        for plan in self._learning_plans.values():
            if plan["assessment_id"] == assessment_id:
                return plan
        return None

    async def clear_session_data(self, thread_id: str):
        """
        Clear all data associated with a session/thread from memory/Redis.
        """
        async with self._lock:
            # 1. Find assessment by thread
            asmt_to_del = None
            cand_id = None
            for aid, asmt in self._assessments.items():
                if asmt["thread_id"] == thread_id:
                    asmt_to_del = aid
                    cand_id = asmt["candidate_id"]
                    break
            
            if asmt_to_del:
                print(f"🧹 Clearing session data for thread {thread_id}")
                
                # 2. Clear LangGraph checkpoints from Redis
                if self._redis:
                    try:
                        # LangGraph checkpoints are usually stored with keys like "checkpoint:<thread_id>:..."
                        # We delete all keys associated with this thread_id
                        keys = self._redis.keys(f"checkpoint:{thread_id}*")
                        if keys:
                            self._redis.delete(*keys)
                            print(f"   🗑️ Deleted {len(keys)} LangGraph keys from Redis")
                    except Exception as e:
                        print(f"   ⚠️ Failed to clear Redis keys: {e}")

                # 3. Delete candidate (and their chroma doc)
                if cand_id and cand_id in self._candidates:
                    chroma_doc_id = self._candidates[cand_id].get("chroma_doc_id")
                    if chroma_doc_id:
                        from app.memory.vector_store import resume_vector_store
                        await resume_vector_store.delete_resume(chroma_doc_id)
                    self._candidates.pop(cand_id, None)
                
                # 3. Delete assessment and learning plan
                self.delete_assessment(asmt_to_del)

# Singleton instance
storage = SimpleStorage()
