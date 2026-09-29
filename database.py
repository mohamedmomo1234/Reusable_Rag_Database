from datetime import datetime, timezone

from pymongo import MongoClient

from config import (
    MONGODB_URI,
    MONGODB_DATABASE,
    MONGODB_COLLECTION,
    MONGODB_FEEDBACK_COLLECTION,
)

_client = None


def get_database():
    global _client

    if not MONGODB_URI:
        return None

    if _client is None:
        _client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000,
        )

    return _client[MONGODB_DATABASE]


def save_chat(question, answer, sources=None, session_id=None):
    database = get_database()

    if database is None:
        return False

    database[MONGODB_COLLECTION].insert_one(
        {
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc),
        }
    )

    return True


def save_feedback(
    question,
    answer,
    feedback,
    sources=None,
    session_id=None,
):
    database = get_database()

    if database is None:
        return False

    database[MONGODB_FEEDBACK_COLLECTION].insert_one(
        {
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "sources": sources or [],
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc),
        }
    )

    return True


def get_chat_history(limit=100):
    database = get_database()

    if database is None:
        return []

    return list(
        database[MONGODB_COLLECTION]
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
