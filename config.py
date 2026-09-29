import os

from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Reusable RAG Assistant")
APP_DESCRIPTION = os.getenv(
    "APP_DESCRIPTION",
    "Ask questions using the indexed knowledge base.",
)

DOCS_DIR = os.getenv("DOCS_DIR", "data/multy_docs")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "rag_engine")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_history")
MONGODB_FEEDBACK_COLLECTION = os.getenv(
    "MONGODB_FEEDBACK_COLLECTION",
    "feedback",
)

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "reusable-rag-engine")

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Retrieval
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "12"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "12"))
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "16"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "12"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

RRF_K = int(os.getenv("RRF_K", "60"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.55"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.45"))

# The threshold is now only a soft filter. If it removes everything,
# the strongest reranked result is kept when candidates exist.
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.30"))
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.20"))

# Multi-query / normalization
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))
QUERY_EXPANSION_ENABLED = os.getenv(
    "QUERY_EXPANSION_ENABLED",
    "true",
).lower() == "true"

# E5-large is multilingual and expects query:/passage: prefixes.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "8"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10"))
