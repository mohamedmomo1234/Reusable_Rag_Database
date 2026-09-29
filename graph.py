from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from generator import call_model
from output_guard import sanitize_output
from retriever import build_context, get_sources, retrieve_documents
from security import security_check


class RAGState(TypedDict, total=False):
    question: str
    history: list
    allowed: bool
    context: str
    answer: str
    sources: list
    documents: list


def security_node(state):
    allowed, message = security_check(state["question"])

    if not allowed:
        return {
            "allowed": False,
            "answer": message,
        }

    return {"allowed": True}


def security_router(state):
    return "blocked" if state.get("allowed") is False else "continue"


def retrieve_node(state):
    documents = retrieve_documents(
        question=state["question"],
        history=state.get("history", []),
    )

    return {
        "documents": documents,
        "context": build_context(documents),
        "sources": get_sources(documents),
    }


def relevance_router(state):
    return (
        "has_context"
        if state.get("context", "").strip()
        else "no_context"
    )


def no_context_node(state):
    return {
        "answer": (
            "The current knowledge base does not contain enough "
            "relevant information to answer this question."
        )
    }


def generate_node(state):
    answer = call_model(
        question=state["question"],
        context=state["context"],
        history=state.get("history", []),
    )

    return {"answer": sanitize_output(answer)}


builder = StateGraph(RAGState)

builder.add_node("security", security_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("no_context", no_context_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "security")

builder.add_conditional_edges(
    "security",
    security_router,
    {
        "continue": "retrieve",
        "blocked": END,
    },
)

builder.add_conditional_edges(
    "retrieve",
    relevance_router,
    {
        "has_context": "generate",
        "no_context": "no_context",
    },
)

builder.add_edge("no_context", END)
builder.add_edge("generate", END)

rag_graph = builder.compile()


def ask_rag_bot(question, history=None):
    result = rag_graph.invoke(
        {
            "question": question,
            "history": history or [],
        }
    )

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
    }


ask_medical_bot = ask_rag_bot
