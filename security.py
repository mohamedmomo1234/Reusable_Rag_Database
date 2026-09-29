import re
from config import MAX_QUESTION_LENGTH

BLOCKED_PATTERNS = [
    r"(?i)api\s*key",
    r"(?i)secret\s*key",
    r"(?i)password",
    r"(?i)environment\s+variables?",
    r"(?i)\.env",
    r"(?i)system\s+prompt",
    r"(?i)developer\s+prompt",
    r"(?i)show\s+(me\s+)?the\s+prompt",
    r"(?i)print\s+(all\s+)?secrets",
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)reveal\s+(your\s+)?instructions",
    r"(?i)delete\s+(the\s+)?database",
]

SECRET_VALUE_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",
    r"gsk_[A-Za-z0-9_-]{10,}",
    r"mongodb\+srv://",
]


def security_check(question):
    if not isinstance(question, str):
        return False, "السؤال يجب أن يكون نصًا."

    q = question.strip()

    if not q:
        return False, "من فضلك اكتب سؤالًا."

    if len(q) > MAX_QUESTION_LENGTH:
        return False, "السؤال طويل جدًا. من فضلك اختصر السؤال."

    for pattern in SECRET_VALUE_PATTERNS:
        if re.search(pattern, q):
            return False, (
                "لا أستطيع عرض أو مشاركة مفاتيح API أو الأسرار "
                "أو بيانات الاعتماد."
            )

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, q):
            return False, (
                "لا أستطيع كشف مفاتيح API أو الأسرار أو system/developer "
                "prompts أو بيانات التطبيق الخاصة."
            )

    return True, ""
