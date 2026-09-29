from pathlib import Path

from config import CHROMA_DIR, GROQ_API_KEY, MONGODB_URI


def check_health():
    return {
        "groq_configured": bool(GROQ_API_KEY),
        "mongodb_configured": bool(MONGODB_URI),
        "chroma_exists": Path(CHROMA_DIR).exists(),
    }


if __name__ == "__main__":
    print(check_health())
