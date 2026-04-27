"""
app/memory/vector_store.py

ChromaDB wrapper for resume semantic search.
Resumes are embedded and stored here so we can later do:
  - "Find candidates who know Kafka + distributed systems"
  - "Which candidates best match this JD?"
"""

import uuid
from typing import Any, Optional
from app.core.config import settings


class ResumeVectorStore:
    """
    Thin async wrapper around ChromaDB + LangChain embeddings.
    Singleton pattern — use the `resume_vector_store` instance below.
    """

    def __init__(self):
        self._collection = None
        self._embeddings = None

    async def initialize(self):
        """Called once at startup to set up ChromaDB collection."""
        import chromadb
        from app.core.llm import get_embeddings

        self._embeddings = get_embeddings()

        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_PATH)
        self._collection = client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_RESUMES,
            metadata={"hnsw:space": "cosine"},   # cosine similarity for text
        )
        print(f"   ChromaDB: Collection '{settings.CHROMA_COLLECTION_RESUMES}' "
              f"loaded ({self._collection.count()} documents)")

    async def add_resume(
        self,
        candidate_email: Optional[str],
        candidate_name: Optional[str],
        resume_text: str,
    ) -> str | None:
        """
        Embed and store a resume. Returns the ChromaDB document ID or None on failure.
        """
        if not self._collection:
            print("⚠️ Vector store not initialized. Indexing skipped.")
            return None

        try:
            doc_id = str(uuid.uuid4())

            # Generate embedding (async-friendly via run_in_executor for sync libs)
            import asyncio
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._embeddings.embed_documents([resume_text])[0]
            )

            self._collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[resume_text],
                metadatas=[{
                    "email": candidate_email or "anonymous",
                    "name": candidate_name or "Anonymous",
                    "doc_id": doc_id,
                }],
            )
            return doc_id
        except Exception as e:
            print(f"❌ Failed to index resume in ChromaDB: {e}")
            return None

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Semantic search across all stored resumes.
        Returns top-k most similar candidates.
        """
        if not self._collection:
            return []

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,
                lambda: self._embeddings.embed_query(query)
            )

            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count() or 1),
                include=["documents", "metadatas", "distances"],
            )

            output = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"][0]):
                    output.append({
                        "doc_id": doc_id,
                        "name": results["metadatas"][0][i].get("name"),
                        "email": results["metadatas"][0][i].get("email"),
                        "similarity_score": round(1 - results["distances"][0][i], 4),
                        "resume_snippet": results["documents"][0][i][:300] + "...",
                    })
            return output
        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return []

    async def delete_resume(self, chroma_doc_id: str):
        """Remove a resume from the vector store."""
        if self._collection:
            self._collection.delete(ids=[chroma_doc_id])


# ── Singleton ─────────────────────────────────────────────────────────────────
resume_vector_store = ResumeVectorStore()