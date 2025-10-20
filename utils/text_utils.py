# utils/text_utils.py
import re

def redact_phi(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
    text = re.sub(r"(\+?\d[\d ()-]{7,}\d)", "[REDACTED_PHONE]", text)
    text = re.sub(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "[REDACTED_DATE]", text)
    text = re.sub(r"\b(\d{4}-\d{2}-\d{2})\b", "[REDACTED_DATE]", text)
    text = re.sub(r"\b([A-Z][a-z]{2,}\s[A-Z][a-z]{2,})\b", "[REDACTED_NAME]", text)
    return text
