#!/usr/bin/env python
# coding: utf-8

# ======================
# 1- Setup
# ======================

# Optional: install dependencies via terminal instead of script
# pip install -r /Users/Mahsa/Downloads/code-base/requirements.txt
# pip install langchain langchain_groq python-dotenv pillow

from dotenv import load_dotenv
import sys
import os

# Local project path
PROJECT_PATH = "/Users/Mahsa/Downloads/code-base"
sys.path.append(PROJECT_PATH)

# Load environment variables from local .env file
load_dotenv(os.path.join(PROJECT_PATH, ".env"))

# ======================
# 2) Prompt Templates
# ======================
from prompts import DATE_ASSISTANT_SYSTEM_PROMPT
from datetime import datetime
from langchain.prompts import ChatPromptTemplate

# ======================
# 3) Agent Setup
# ======================
"""
CaregiverCompanionAgent
-----------------------
A Python module that implements a text-only "agent" to
summarize and translate clinical/medical notes into layman's terms.
Uses a Groq-hosted open model via LangChain's ChatGroq wrapper.
"""

from typing import Dict, List, Optional, Tuple
import re

# LangChain Groq import
try:
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage, SystemMessage
except Exception as e:
    raise ImportError(
        "This module requires langchain and its ChatGroq wrapper. "
        "Install with `pip install langchain langchain_groq`. "
        "Original error: {}".format(e)
    )


class CaregiverCompanionAgent:
    """Agent to summarize and explain medical text in layman's terms."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a compassionate, clear medical-notes translator for a non-medical "
        "caregiver. Your task: given clinical or medical text, produce (1) a short "
        "summary in simple, everyday language (2) a brief explanation of any "
        "medical terms or jargon used, and (3) very short bullets of main actionable points. "
        "Do NOT give medical advice."
    )

    def __init__(self,
                 groq_api_key: Optional[str] = None,
                 model_name: str = "openai/gpt-oss-20b",
                 temperature: float = 0.0):
        key = groq_api_key ##or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "A Groq API key is required. Pass `groq_api_key=` or set the environment variable GROQ_API_KEY."
            )

        self.client = ChatGroq(model=model_name, groq_api_key=key, temperature=temperature)
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def _redact_phi(text: str) -> str:
        """Simple PHI redaction for demo purposes."""
        t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
        t = re.sub(r"(\+?\d[\d ()-]{7,}\d)", "[REDACTED_PHONE]", t)
        t = re.sub(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", "[REDACTED_DATE]", t)
        t = re.sub(r"\b(\d{4}-\d{2}-\d{2})\b", "[REDACTED_DATE]", t)
        t = re.sub(r"\b([A-Z][a-z]{2,}\s[A-Z][a-z]{2,})\b", "[REDACTED_NAME]", t)
        return t

    def summarize_and_explain(self, text: str, redact_phi: bool = True) -> Dict[str, object]:
        """Summarize and explain medical notes in layman's terms."""
        if not text or not text.strip():
            return {"summary": "", "explanations": [], "action_items": [], "raw_response": ""}
        original_text = text
        if redact_phi:
            text = self._redact_phi(text)
        user_prompt = (
                """Below are clinical notes. Do the following:
                1) Write a 2-4 sentence summary in plain language.
                2) Explain up to 8 medical terms in plain language.
                3) Provide up to 6 bullet points of actionable items.
                4) List unclear info if any.
                Return JSON with keys: summary, explanations, action_items, unclear.
                \n\nSource text:\n"""
                + text
        )
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_prompt)]
        model_response = self.client(messages)
        raw = model_response.content if hasattr(model_response, 'content') else str(model_response)
        parsed = self._try_parse_model_output(raw)
        if parsed is None:
            return {"summary": raw.strip(), "explanations": [], "action_items": [],
                    "unclear": ["Could not parse structured response"], "raw_response": raw}
        parsed["raw_response"] = raw
        parsed["original_text"] = original_text
        return parsed

    @staticmethod
    def _try_parse_model_output(text: str) -> Optional[Dict[str, object]]:
        import json
        try:
            parsed = json.loads(text)
            return CaregiverCompanionAgent._normalize_parsed(parsed)
        except Exception:
            m = re.search(r"(\{[\s\S]*\})", text)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    return CaregiverCompanionAgent._normalize_parsed(parsed)
                except Exception:
                    pass
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if not lines:
                return None
            summary = lines[0]
            bullets = [l.lstrip('-* ').strip() for l in lines[1:] if l.startswith(('-', '*'))][:6]
            return {"summary": summary, "explanations": [], "action_items": bullets, "unclear": []}

    @staticmethod
    def _normalize_parsed(parsed: Dict) -> Dict[str, object]:
        summary = parsed.get("summary") or parsed.get("Summary") or parsed.get("summary_text") or ""
        explanations = parsed.get("explanations") or parsed.get("terms") or []
        if isinstance(explanations, dict):
            explanations = [{"term": k, "explanation": v} for k, v in explanations.items()]
        action_items = parsed.get("action_items") or parsed.get("actions") or parsed.get("questions") or []
        unclear = parsed.get("unclear") or parsed.get("unclear_points") or []
        return {"summary": summary, "explanations": explanations, "action_items": action_items, "unclear": unclear}


# ======================
# 4) Instantiate Agent and Example Run
# ======================

# Make sure GROQ_API_KEY is set in your environment
agent = CaregiverCompanionAgent(groq_api_key="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")

medical_notes = """
Patient presented with symptoms of acute rhinitis and a persistent cough.
Vital signs stable. Recommended rest and increased fluid intake.
Follow-up appointment scheduled in 7 days.
"""

output = agent.summarize_and_explain(medical_notes)

print("Summary:")
print(output['summary'])
print("\nExplanations:")
if output.get('explanations'):
    for item in output['explanations']:
        # if item is a dict
        if isinstance(item, dict):
            term = item.get('term') or item.get('topic') or 'Unknown term'
            expl = item.get('explanation') or item.get('definition') or item.get('text') or item.get('summary') or str(item)
            print(f"- {term}: {expl}")
        else:
            # if it's just a string
            print(f"- {item}")
else:
    print("None")


print("\nAction Items:")
for item in output['action_items']:
    print(f"- {item}")
if output.get('unclear'):
    print("\nUnclear/Missing Information:")
    for item in output['unclear']:
        print(f"- {item}")

# ======================
# 5) CDC QA Example
# ======================

from langchain.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Step 0: Groq API key
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise ValueError("Please set your GROQ_API_KEY environment variable")

# Load CDC pages
urls = [
    "https://www.cdc.gov/flu/symptoms/index.html",
    "https://www.cdc.gov/cancer/breast/basic_info/index.htm",
]
loader = WebBaseLoader(urls)
docs = loader.load()

# Embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# ChatGroq LLM
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0)

# Retrieval QA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What are the treatment options for breast cancer?"
response = qa_chain.run(query)
print(response)
