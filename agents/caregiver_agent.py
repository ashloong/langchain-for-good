import os, re, json
from typing import Dict, Optional
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

class CaregiverCompanionAgent:
    """Summarizes and explains medical notes in layman's terms."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a compassionate, clear medical-notes translator for a non-medical caregiver. "
        "Your task: summarize, explain medical terms, and list brief actionable points. Do NOT give medical advice."
    )

    def __init__(self, groq_api_key: Optional[str] = None, model_name="openai/gpt-oss-20b", temperature=0.0):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY must be set.")
        self.client = ChatGroq(model=model_name, groq_api_key=self.api_key, temperature=temperature)

    def summarize_and_explain(self, text: str, redact_phi: bool = True) -> Dict[str, object]:
        if not text.strip():
            return {"summary": "", "explanations": [], "action_items": [], "raw_response": ""}

        text = self._redact_phi(text) if redact_phi else text
        prompt = f"""
        Given the medical notes below:
        1. Write a 2-4 sentence plain-language summary.
        2. Explain up to 8 medical terms.
        3. Provide up to 6 actionable items.
        4. Mention unclear information if any.
        Return JSON with keys: summary, explanations, action_items, unclear.
        \n\n{text}
        """
        messages = [
            SystemMessage(content=self.DEFAULT_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        response = self.client(messages)
        raw = getattr(response, "content", str(response))
        return self._parse_response(raw, text)

    @staticmethod
    def _redact_phi(text: str) -> str:
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
        text = re.sub(r"(\+?\d[\d ()-]{7,}\d)", "[REDACTED_PHONE]", text)
        text = re.sub(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "[REDACTED_DATE]", text)
        text = re.sub(r"\b([A-Z][a-z]{2,}\s[A-Z][a-z]{2,})\b", "[REDACTED_NAME]", text)
        return text

    def _parse_response(self, raw: str, original_text: str) -> Dict[str, object]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"summary": raw.strip(), "explanations": [], "action_items": [], "unclear": []}
        data["raw_response"] = raw
        data["original_text"] = original_text
        return data
