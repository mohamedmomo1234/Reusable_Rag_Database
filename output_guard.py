import re

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_-]{10,}",
    r"gsk_[A-Za-z0-9_-]{10,}",
    r"mongodb\+srv://[^\s]+",
]


def sanitize_output(text):
    if not text:
        return text

    for pattern in SECRET_PATTERNS:
        text = re.sub(
            pattern,
            "[REDACTED]",
            text,
        )

    return text
