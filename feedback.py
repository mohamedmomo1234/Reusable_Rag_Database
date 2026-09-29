from database import save_feedback


def record_feedback(
    question,
    answer,
    feedback,
    sources=None,
    session_id=None,
):
    if feedback not in {"up", "down"}:
        raise ValueError("Feedback must be 'up' or 'down'.")

    return save_feedback(
        question=question,
        answer=answer,
        feedback=feedback,
        sources=sources,
        session_id=session_id,
    )
