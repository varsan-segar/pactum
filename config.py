import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ACTIVE_LLM_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "GROQ")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 600

ACTIVE_EMBEDDING_PROVIDER = os.getenv("ACTIVE_EMBEDDING_PROVIDER", "HUGGINGFACE")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 4

MAX_HISTORY_TOKENS = 2000

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "chroma_store"
FILE_UPLOAD_PATH = PROJECT_ROOT / "uploads"

VECTOR_DB_PATH.mkdir(exist_ok=True)
FILE_UPLOAD_PATH.mkdir(exist_ok=True)

VECTOR_DB_COLLECTION_NAME = "knowledge_base"

MAX_FILE_SIZE_MB = 20
MAX_QUERY_SIZE = 1000

def get_embedding_model():
    if ACTIVE_EMBEDDING_PROVIDER.upper() == "OPENAI":
        api_key = os.getenv("OPENAI_API_KEY")
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        return {"api_key": api_key, "model": embedding_model}

    embedding_model = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    return {"model": embedding_model}

def get_chat_model():
    if ACTIVE_LLM_PROVIDER.upper() == "OPENAI":
        api_key = os.getenv("OPENAI_API_KEY")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        return {"api_key": api_key, "model": chat_model, "temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_TOKENS, "streaming": True}
    
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("GROQ_API_KEY")
    chat_model = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")

    return {"base_url": base_url, "api_key": api_key, "model": chat_model, "temperature": LLM_TEMPERATURE, "max_tokens": LLM_MAX_TOKENS, "streaming": True}