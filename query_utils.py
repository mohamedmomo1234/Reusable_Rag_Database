import re
import unicodedata

from nltk.stem import ISRIStemmer
from nltk.stem.snowball import SnowballStemmer

from config import MULTI_QUERY_COUNT, QUERY_EXPANSION_ENABLED

_EN_STEMMER = SnowballStemmer("english")
_AR_STEMMER = ISRIStemmer()

STOPWORDS = {
    "the", "a", "an", "is", "are", "what", "which", "how", "why",
    "when", "where", "who", "does", "do", "can", "of", "to", "in",
    "on", "for", "and", "or", "with", "about",
    "ما", "ماذا", "ماهو", "ماهي", "هو", "هي", "عن", "في", "من",
    "على", "هل", "كيف", "لماذا", "متى", "اين", "إيه", "ايه",
}



# Small domain-friendly expansion map. It is intentionally conservative.

EXPANSIONS = {
    "sql": "sql structured query language استعلام",
    "erd": "erd entity relationship diagram نموذج علاقات",
    "dbms": "dbms database management system نظام إدارة قواعد بيانات",
    "key": "key primary key foreign key مفتاح",
    "keys": "keys primary key foreign key مفاتيح",
    "join": "join joins inner join outer join دمج جداول",
    "index": "index indexing فهرسة",
    "transaction": "transaction transactions acid معاملة",
    "normalization": "normalization normal form تسوية تطبيع",
    "query": "query queries استعلام",
    "table": "table tables relation جدول",
    "schema": "schema مخطط",
    "constraint": "constraint constraints قيد قيود",

    "استعلام": "query sql استعلام",
    "جدول": "table relation جدول",
    "جداول": "tables relation جداول",
    "مفتاح": "key primary key foreign key مفتاح",
    "مفاتيح": "keys مفاتيح",
    "فهرسة": "index indexing فهرسة",
    "علاقة": "relationship relation علاقة",
    "علاقات": "relationships relations علاقات",
    "تطبيع": "normalization normal form تطبيع تسوية",
    "معاملة": "transaction acid معاملة",
    "قيد": "constraint constraints قيد",
    "قيود": "constraints قيود",

    "primary": "primary key المفتاح الأساسي",
    "foreign": "foreign key المفتاح الأجنبي",
    "forign": "foreign key المفتاح الأجنبي",   # تصحيح خطأ إملائي شائع
    "foriegn": "foreign key المفتاح الأجنبي",  # تصحيح خطأ إملائي شائع تاني
    "primary key": "primary key المفتاح الأساسي",
    "foreign key": "foreign key المفتاح الأجنبي",

    "create": "create command sql أنشئ إنشاء",
    "database": "database db قاعدة بيانات",
    "table": "table tables relation جدول جداول",
    "create database": "create database command أنشئ قاعدة بيانات",
    "create table": "create table command أنشئ جدول",
    "drop": "drop command حذف",
    "alter": "alter command تعديل",
    "insert": "insert command إدراج",
    "update": "update command تحديث",
    "delete": "delete command حذف",
    "select": "select command استعلام اختيار",
}


# EXPANSIONS = {
#     "symptom": "symptoms signs manifestations",
#     "symptoms": "symptoms signs manifestations",
#     "cause": "cause causes reason reasons",
#     "causes": "cause causes reason reasons",
#     "treatment": "treatment management therapy",
#     "diagnosis": "diagnosis diagnostic",
#     "risk": "risk risks risk factors",
#     "factors": "factors risk factors",
#     "اعراض": "أعراض علامات manifestations",
#     "عرض": "أعراض علامات",
#     "اسباب": "أسباب عوامل",
#     "سبب": "أسباب عوامل",
#     "علاج": "علاج معالجة therapy",
#     "تشخيص": "تشخيص diagnosis",
#     "مخاطر": "مخاطر عوامل الخطر",
# }





def normalize_text(text: str) -> str:
    """Light multilingual normalization; does not change document meaning."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()

    # Arabic diacritics / tatweel.
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("ـ", "")

    # Common Arabic letter normalization.
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ى", "ي")
    )

    # Keep Arabic/Latin letters and numbers.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def stem_tokens(tokens):
    stems = []

    for token in tokens:
        if re.search(r"[a-z]", token):
            stems.append(_EN_STEMMER.stem(token))
        elif re.search(r"[\u0600-\u06FF]", token):
            stems.append(_AR_STEMMER.stem(token))
        else:
            stems.append(token)

    return stems


def keyword_query(text: str) -> str:
    tokens = [
        token
        for token in tokenize(text)
        if token not in STOPWORDS and len(token) > 1
    ]
    return " ".join(tokens)


def stemmed_query(text: str) -> str:
    tokens = [
        token
        for token in tokenize(text)
        if token not in STOPWORDS and len(token) > 1
    ]
    return " ".join(stem_tokens(tokens))


def expand_query(text: str) -> str:
    if not QUERY_EXPANSION_ENABLED:
        return normalize_text(text)

    normalized = normalize_text(text)
    additions = []

    for token in tokenize(normalized):
        expansion = EXPANSIONS.get(token)
        if expansion:
            additions.append(expansion)

    if not additions:
        return normalized

    return f"{normalized} {' '.join(additions)}".strip()



COMPARISON_MARKERS = ["الفرق بين", "الفرق", "difference between", "vs", "compare"]


def is_comparison_query(text: str) -> bool:
    normalized = normalize_text(text)
    return any(marker in normalized for marker in COMPARISON_MARKERS)


def build_query_variants(question: str, history=None):
    """
    Simple query rewriting + multi-query retrieval.

    No second LLM call is required. This keeps the pipeline cheaper,
    faster, and easier to debug.
    """
    history = history or []
    current = normalize_text(question)

    recent_user = [
        item.get("content", "")
        for item in history[-4:]
        if item.get("role") == "user" and item.get("content")
    ]

    conversation_rewrite = current
    if recent_user:
        previous = normalize_text(" ".join(recent_user[-2:]))
        if previous and previous not in current:
            conversation_rewrite = f"{previous} {current}"


    variants = [
        current,
        expand_query(conversation_rewrite),
        keyword_query(conversation_rewrite),
        stemmed_query(conversation_rewrite),
    ]

    if is_comparison_query(current):
        variants.append(expand_query(current))

    unique = []
    seen = set()


    for variant in variants:
        variant = variant.strip()
        if variant and variant not in seen:
            seen.add(variant)
            unique.append(variant)

    return unique[:max(1, MULTI_QUERY_COUNT)]
