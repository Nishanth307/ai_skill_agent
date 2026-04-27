"""
app/memory/checkpointer.py

LangGraph checkpointer factory.

The checkpointer is what makes human-in-the-loop possible across HTTP requests.
After every node, LangGraph serialises the full AgentState and saves it
keyed by thread_id. When the next HTTP request arrives with the same thread_id,
LangGraph deserialises and resumes exactly where it left off.

Dev  → MemorySaver  (in-process dict, lost on restart — fine for development)
Prod → RedisSaver   (persistent, survives restarts, works across multiple workers)
"""

from app.core.config import settings


def get_checkpointer():
    """
    Returns the right checkpointer for the current environment.
    Called once at graph compile time (inside get_graph()).
    """
    if settings.ENVIRONMENT == "production":
        return _build_redis_checkpointer()
    return _build_memory_checkpointer()


def _build_memory_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver
    print("   Checkpointer: MemorySaver (in-memory — dev only)")
    return MemorySaver()


def _build_redis_checkpointer():
    """
    Redis checkpointer for production.

    Requires: pip install langgraph-checkpoint-redis
    Redis must be running and REDIS_URL set in .env.

    Falls back to MemorySaver if Redis is unavailable,
    so the app doesn't hard-crash in staging environments.
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver

        checkpointer = RedisSaver.from_conn_string(settings.REDIS_URL)
        print(f"   Checkpointer: RedisSaver → {settings.REDIS_URL}")
        return checkpointer

    except ImportError:
        print("⚠️  langgraph-checkpoint-redis not installed — falling back to MemorySaver")
        print("   Run: pip install langgraph-checkpoint-redis")
        return _build_memory_checkpointer()

    except Exception as e:
        print(f"⚠️  Redis unavailable ({e}) — falling back to MemorySaver")
        return _build_memory_checkpointer()


async def verify_redis_connection() -> bool:
    """
    Health-check helper — call this from the /health endpoint
    to confirm Redis is reachable.
    """
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        await client.ping()
        await client.aclose()
        return True
    except Exception:
        return False