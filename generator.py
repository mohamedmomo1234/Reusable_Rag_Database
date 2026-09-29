from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MODEL
from prompts import build_rag_prompt

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not configured.")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.1,
    streaming=True,
)


def format_history(history):
    if not history:
        return ""

    lines = []

    for item in history[-8:]:
        role = item.get("role", "user")
        content = item.get("content", "")

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines)


def call_model(question, context, history=None):
    prompt = build_rag_prompt(
        question=question,
        context=context,
        history=format_history(history or []),
    )

    response = llm.invoke(prompt)
    return response.content


def stream_model(question, context, history=None):
    prompt = build_rag_prompt(
        question=question,
        context=context,
        history=format_history(history or []),
    )

    for chunk in llm.stream(prompt):
        text = chunk.content

        if isinstance(text, str) and text:
            yield text
