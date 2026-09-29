from functools import lru_cache
import math
import random
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    BM25_TOP_K,
    BM25_WEIGHT,
    CHROMA_COLLECTION,
    CHROMA_DIR,
    FINAL_TOP_K,
    HYBRID_TOP_K,
    MIN_RERANK_SCORE,
    RERANKER_MODEL,
    RERANK_TOP_K,
    RELEVANCE_THRESHOLD,
    RRF_K,
    VECTOR_TOP_K,
    VECTOR_WEIGHT,
)
from embeddings import get_embeddings
from query_utils import tokenize


@lru_cache(maxsize=1)
def get_vectorstore():
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_all_documents():
    vectorstore = get_vectorstore()
    data = vectorstore.get(include=["documents", "metadatas"])

    documents = []
    for content, metadata in zip(
        data.get("documents", []),
        data.get("metadatas", []),
    ):
        if content and content.strip():
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata or {},
                )
            )

    return documents


def get_document_overview_samples(max_chunks=12):
    documents = get_all_documents()

    if not documents:
        return []

    total = len(documents)
    if total <= max_chunks:
        return documents

    return random.sample(documents, max_chunks)




# def get_document_overview_samples(max_chunks=20):

#     """
#     Instead of similarity-based retrieval, sample chunks evenly across
#     the whole document so a broad summary request gets real coverage.
#     """
#     documents = get_all_documents()


#     if not documents:
#         return []

#     total = len(documents)
#     if total <= max_chunks:
#         return documents

#     step = total / max_chunks
#     indices = [int(i * step) for i in range(max_chunks)]

#     return [documents[i] for i in indices]




@lru_cache(maxsize=1)
def get_bm25():
    documents = get_all_documents()
    tokenized = [tokenize(doc.page_content) for doc in documents]
    return BM25Okapi(tokenized)


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)


def clear_retrieval_cache():
    get_vectorstore.cache_clear()
    get_all_documents.cache_clear()
    get_bm25.cache_clear()
    get_reranker.cache_clear()


def vector_search(query: str):
    results = get_vectorstore().similarity_search_with_relevance_scores(
        query,
        k=VECTOR_TOP_K,
    )

    output = []
    for rank, (doc, score) in enumerate(results, start=1):
        doc.metadata["vector_score"] = max(
            0.0,
            min(1.0, float(score)),
        )
        doc.metadata["vector_rank"] = rank
        output.append(doc)

    return output


def bm25_search(query: str):
    documents = get_all_documents()
    if not documents:
        return []

    bm25 = get_bm25()
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )[:BM25_TOP_K]

    output = []
    for rank, (index, score) in enumerate(ranked, start=1):
        doc = documents[index]
        doc.metadata["bm25_score"] = float(score)
        doc.metadata["bm25_rank"] = rank
        output.append(doc)

    return output


def document_key(doc):
    metadata = doc.metadata
    return (
        metadata.get("source_file", ""),
        metadata.get("page_number", metadata.get("page", "")),
        metadata.get("chunk_id", ""),
        doc.page_content[:80],
    )


def hybrid_search(query: str):
    vector_docs = vector_search(query)
    bm25_docs = bm25_search(query)

    merged = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = document_key(doc)
        merged.setdefault(key, doc)
        merged[key].metadata["vector_rank"] = rank

    for rank, doc in enumerate(bm25_docs, start=1):
        key = document_key(doc)
        if key not in merged:
            merged[key] = doc
        merged[key].metadata["bm25_rank"] = rank

    scored = []

    for doc in merged.values():
        vector_rank = doc.metadata.get("vector_rank")
        bm25_rank = doc.metadata.get("bm25_rank")

        vector_rrf = 1 / (RRF_K + vector_rank) if vector_rank else 0
        bm25_rrf = 1 / (RRF_K + bm25_rank) if bm25_rank else 0

        hybrid_score = (
            VECTOR_WEIGHT * vector_rrf
            + BM25_WEIGHT * bm25_rrf
        )

        doc.metadata["hybrid_score"] = hybrid_score
        scored.append(doc)

    scored.sort(
        key=lambda doc: doc.metadata.get("hybrid_score", 0),
        reverse=True,
    )

    return scored[:HYBRID_TOP_K]


def rerank_documents(query: str, documents):
    if not documents:
        return []

    candidates = documents[:RERANK_TOP_K]
    reranker = get_reranker()

    pairs = [[query, document.page_content] for document in candidates]
    scores = reranker.predict(pairs)

    reranked = []

    for doc, raw_score in zip(candidates, scores):
        raw_score = float(raw_score)

        # Convert cross-encoder logit to a bounded confidence-like value.
        rerank_score = 1 / (1 + math.exp(-raw_score))

        doc.metadata["rerank_score"] = rerank_score
        reranked.append(doc)

    reranked.sort(
        key=lambda doc: doc.metadata.get("rerank_score", 0),
        reverse=True,
    )

    return reranked[:FINAL_TOP_K]


def _merge_multi_query_results(query_results):
    merged = {}

    for query_index, (query, docs) in enumerate(query_results):
        for rank, doc in enumerate(docs, start=1):
            key = document_key(doc)

            if key not in merged:
                merged[key] = doc

                doc.metadata["query_hits"] = 0
                doc.metadata["best_query_rank"] = rank
                doc.metadata["matched_queries"] = []

            doc= merged[key]

            doc.metadata["query_hits"] += 1
            doc.metadata["best_query_rank"] = min(
                doc.metadata.get("best_query_rank", rank),
                rank,
            )

            matched = doc.metadata.setdefault("matched_queries", [])
            if query not in matched:
                matched.append(query)

    return list(merged.values())


def retrieve_documents(question: str, history=None):
    """
    Multi-query retrieval:
      1. Normalize / lightly stem the question.
      2. Create 2-3 query variants.
      3. Run vector + BM25 for every variant.
      4. Merge results.
      5. Rerank merged candidates using the original normalized query.
      6. Apply a soft threshold instead of returning [] too aggressively.
    """
    from query_utils import build_query_variants

    queries = build_query_variants(question, history)

    if not queries:
        return []

    query_results = []

    for query in queries:
        candidates = hybrid_search(query)
        query_results.append((query, candidates))

    merged = _merge_multi_query_results(query_results)

    if not merged:
        return []

    # Rerank using the original question rather than a generated variant.
    reranked = rerank_documents(question, merged)

    if not reranked:
        return []

    filtered = [
        doc
        for doc in reranked
        if doc.metadata.get("rerank_score", 0) >= RELEVANCE_THRESHOLD
    ]

    # Soft fallback: avoid the old "Retrieved: []" problem when there is
    # a meaningful candidate but the threshold is slightly too strict.
    if not filtered:
        strongest = reranked[0]
        if strongest.metadata.get("rerank_score", 0) >= MIN_RERANK_SCORE:
            filtered = [strongest]

    for doc in filtered:
        doc.metadata["query_variants"] = queries

    return filtered


def build_context(documents):
    if not documents:
        return ""

    parts = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source_file", "Unknown")
        page = document.metadata.get("page_number")

        location = source
        if page:
            location += f", page {page}"

        parts.append(
            f"[SOURCE {index}]\n"
            f"Source: {location}\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n".join(parts)


def get_sources(documents):
    sources = []

    for document in documents:
        source = document.metadata.get("source_file", "Unknown")
        page = document.metadata.get("page_number")

        label = source
        if page:
            label += f" — page {page}"

        if label not in sources:
            sources.append(label)

    return sources


def get_retrieval_info(documents):
    if not documents:
        return {
            "query_variants": [],
            "result_count": 0,
        }

    variants = documents[0].metadata.get("query_variants", [])

    return {
        "query_variants": variants,
        "result_count": len(documents),
    }
