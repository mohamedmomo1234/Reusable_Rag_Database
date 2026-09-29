# Reusable RAG Engine — Improved Retrieval Version

This version keeps the original reusable RAG architecture and adds a simpler
retrieval strategy for the common problem where a relevant question produces
`Retrieved: []`.

## Retrieval pipeline

```text
User Question
     |
     v
Security
     |
     v
Light Normalization
     |
     v
Query Rewrite / Expansion
     |
     v
Multi-Query Retrieval
     |
     +-----------------------+
     |                       |
     v                       v
Vector Search            BM25 Search
E5 multilingual          Normalized/stemmed
     |                       |
     +-----------+-----------+
                 |
                 v
             RRF Hybrid
                 |
                 v
          Cross-Encoder Rerank
                 |
                 v
        Soft Relevance Filter
                 |
                 v
             RAG Prompt
                 |
                 v
              Groq LLM
                 |
                 v
       Answer + Collapsed Sources
```

## Main changes

### 1. `intfloat/multilingual-e5-large`

The vector embedding model is now:

```text
intfloat/multilingual-e5-large
```

The implementation correctly uses:

```text
query: ...
passage: ...
```

for E5 embeddings.

Because this is a large model, CPU inference can be slow and memory-heavy.
Set `EMBEDDING_DEVICE=cuda` if a compatible CUDA environment is available.

### 2. Normalization

`query_utils.py` performs light multilingual normalization:

- Unicode normalization
- Arabic diacritic removal
- common Arabic character normalization
- punctuation cleanup
- whitespace normalization

It does not aggressively rewrite the user's meaning.

### 3. Stemming

BM25 uses normalized tokens plus:

- English Snowball stemming
- Arabic ISRI stemming

This helps lexical retrieval when the query and document use related word forms.

### 4. Query expansion / rewriting

The engine creates a small number of deterministic variants instead of calling
the LLM again for every question.

Examples of variants:

- normalized question
- expanded question
- keyword query
- stemmed keyword query

Recent user messages can also be added when the question depends on conversation
context.

### 5. Multi-query retrieval

Each variant is sent through:

```text
E5 vector search + BM25
```

Results are merged and then reranked.

This means one exact wording is no longer the single point of failure.

### 6. Softer relevance filtering

The old version could return an empty list because the final threshold was too
strict.

The new version:

- uses a lower configurable threshold
- keeps the strongest candidate when it is above a minimum score
- avoids returning `[]` merely because one threshold was missed

The threshold is therefore a safety filter, not the main retrieval mechanism.

### 7. Progressive disclosure

The UI does not dump retrieved chunks into the main answer.

The user sees:

1. direct answer
2. collapsed `Sources & retrieval details`
3. search variants only when the user expands the section

This keeps the interface clean while still making retrieval behavior inspectable.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy:

```text
.env.example -> .env
```

Add your real `GROQ_API_KEY`.

Put trusted documents in:

```text
data/medical_docs/
```

Then rebuild Chroma because the embedding model changed:

```powershell
python ingest.py
```

Run:

```powershell
streamlit run app.py
```

FastAPI remains optional:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Important

Do not reuse the old Chroma database after changing the embedding model.
Delete/rebuild it with:

```powershell
python ingest.py
```

The ingestion script already resets the configured Chroma directory by default.

## Files

- `app.py` — Streamlit UI + progressive disclosure
- `config.py` — configuration
- `embeddings.py` — multilingual E5 wrapper
- `query_utils.py` — normalization, stemming, expansion, multi-query variants
- `retriever.py` — vector + BM25 + RRF + reranking
- `ingest.py` — document ingestion
- `graph.py` — LangGraph workflow
- `generator.py` — Groq model
- `prompts.py` — grounded prompt
- `database.py` — MongoDB storage
- `feedback.py` — feedback
- `security.py` — input protection
- `output_guard.py` — output secret redaction
- `rate_limiter.py` — basic request limiting
- `api.py` — optional FastAPI backend
