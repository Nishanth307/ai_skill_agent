"""
app/core/llm.py
 
LLM Factory — the single place where we decide which model to use.
Every node/chain imports get_llm() instead of instantiating models directly.
This means switching backends = changing ONE env var (LLM_BACKEND).
"""

from functools import lru_cache 
from langchain_core.language_models import BaseChatModel 
from langchain_core.embeddings import Embeddings 

from app.core.config import settings 

@lru_cache(maxsize = None)
def get_llm(
    temperature: float = None,
    max_tokens: int | None= None,
    streaming: bool = False,
    ) -> BaseChatModel:
    """
    Returns a configured LLM instance based on LLM_BACKEND env var.
 
    Args:
        temperature: Override default temperature (from settings)
        max_tokens:  Override default max_tokens (from settings)
        streaming:   Enable streaming responses (for Streamlit UI)
 
    Returns:
        A LangChain-compatible chat model (Ollama or Gemini)
    """
    _temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
    _max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS 

    if settings.LLM_BACKEND == "gemini":
        return _build_gemini(
            temperature = temperature,
            max_tokens = max_tokens,
            streaming = streaming,
        )
    elif settings.LLM_BACKEND == "ollama":
        return _build_ollama(
            temperature = temperature,
            max_tokens = max_tokens,
            streaming = streaming,
        )
    else:
        raise ValueError(f"Unknown LLM_BACKEND: '{settings.LLM_BACKEND}'. Use 'ollama' or 'gemini'.")

def _build_gemini(temperature: float, max_tokens: int, streaming: bool) -> BaseChatModel:
    """Google Gemini via langchain-google-genai."""
    try: 
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError("Run: pip install langchain-google-genai")

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in .env")

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=streaming,
        convert_system_message_to_human=True,
        max_retries=3, # Add retries for transient 429s
    )


def _build_ollama(temperature: float,max_tokens: int,streaming: bool,) -> BaseChatModel:
    """Local Ollama via langchain-ollama."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("Run: pip install langchain-ollama")
 
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=max_tokens,
        streaming=streaming,
    )

def get_embeddings() -> Embeddings:
    """
    Returns an embedding model for ChromaDB vector storage.
    Used to embed resumes for semantic retrieval.
    """
    try:
        if settings.EMBEDDING_BACKEND == "gemini":
            return _build_gemini_embeddings()
        elif settings.EMBEDDING_BACKEND == "ollama":
            return _build_ollama_embeddings()
        elif settings.EMBEDDING_BACKEND == "huggingface":
            return _build_huggingface_embeddings()
        else:
            print(f"⚠️ Unknown EMBEDDING_BACKEND: {settings.EMBEDDING_BACKEND!r}. Fallback to HuggingFace.")
            return _build_huggingface_embeddings()
    except Exception as e:
        print(f"❌ Failed to initialize {settings.EMBEDDING_BACKEND} embeddings: {e}. Falling back to HuggingFace.")
        return _build_huggingface_embeddings()


def _build_gemini_embeddings() -> Embeddings:
    """Google Gemini Embeddings."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError:
        raise ImportError("Run: pip install langchain-google-genai")

    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in .env")

    # models/embedding-001 is more widely available in the free tier
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
    )


def _build_ollama_embeddings() -> Embeddings:
    """Local Ollama Embeddings."""
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        raise ImportError("Run: pip install langchain-ollama")

    return OllamaEmbeddings(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )


def _build_huggingface_embeddings() -> Embeddings:
    """Local embeddings via sentence-transformers (CPU-friendly)."""
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError("Run: pip install sentence-transformers")
 
    return HuggingFaceEmbeddings(
        model_name=settings.HUGGINGFACE_EMBEDDING_MODEL,
        cache_folder="./huggingface_cache"
    )


# ── Specialized LLM variants ──────────────────────────────────────────────────
# Pre-configured instances for specific use cases.
# Import these directly instead of calling get_llm() with custom params.
 
def get_structured_llm(schema):
    """
    LLM with structured output — forces JSON response matching a Pydantic schema.
    Used by skill_extractor and plan_generator nodes.
 
    Usage:
        from app.schemas.skill import ExtractedSkills
        llm = get_structured_llm(ExtractedSkills)
        result: ExtractedSkills = llm.invoke(prompt)
    """
    return get_llm(temperature=0.0).with_structured_output(schema)
 
 
def get_assessor_llm() -> BaseChatModel:
    """
    LLM for conversational assessment — slightly higher temperature
    to make questions feel more natural, streaming enabled.
    """
    return get_llm(temperature=0.4, streaming=True)
 
 
def get_planning_llm() -> BaseChatModel:
    """
    LLM for learning plan generation — low temperature for
    consistent, structured output.
    """
    return get_llm(temperature=0.1, max_tokens=8192)
 
