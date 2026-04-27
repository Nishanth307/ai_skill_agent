from pydantic_settings import BaseSettings,SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from typing import Literal 
import enum

class GoogleModel(enum.Enum):
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_5_FLASH_EXP = "gemini-2.5-flash-exp"
    GEMINI_2_5_PRO_EXP = "gemini-2.5-pro-exp"
    GEMINI_2_5_FLASH_LITE_EXP = "gemini-2.5-flash-lite-exp"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite-preview"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False,
        extra = "ignore"
    )
    APP_NAME: str = "Skill Assessment Agent"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development","staging","production","test"] = "development"

    LLM_BACKEND: Literal["ollama","gemini"] = "ollama"

    GOOGLE_API_KEY: str =  ""
    GEMINI_MODEL: GoogleModel = GoogleModel.GEMINI_3_1_FLASH_LITE

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma3"

    LLM_TEMPERATURE: float = Field(default=0.2, ge=0.0, le=1.0)
    LLM_MAX_TOKENS: int = Field(default=4096, ge=256)


    EMBEDDING_BACKEND: Literal["huggingface", "ollama", "gemini", "local"] = "huggingface"
    HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    REDIS_URL: str = "redis://localhost:6379"

    CHROMA_PERSIST_PATH: str = "./chroma_db"

    CHROMA_COLLECTION_RESUMES: str = "resumes"

    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY:str = ""
    LANGCHAIN_PROJECT: str = "skill-assessment-agent"

    MAX_QUESTIONS_PER_SKILL: int = 2
    MIN_SCORE_TO_PASS: int = 3
    MAX_SKILLS_TO_ASSESS: int = 10
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:8501",
        "http://localhost:8000",
        "http://localhost:3000",
    ]

def get_settings() -> Settings:
    """
    Cached settings singleton.
    Use this everywhere instead of instantiating Settings() directly.
    lru_cache ensures .env is only read once per process.
    """
    return Settings()

settings = get_settings()