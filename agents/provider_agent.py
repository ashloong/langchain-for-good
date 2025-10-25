from __future__ import annotations
from typing import List, Dict, TypedDict, Optional
import os
import re
import math

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from pipelines.provider_json_retrieval import (
    scrape_json_url,
    filter_providers_by_zip,
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ANTHEM_URL = "https://www22.anthem.com/CMS/PROVIDERS_CAM.json"

SYSTEM_BASE = """You are ProviderCompanion, an empathetic healthcare provider assistant.
Your job is to find and summarize nearby healthcare providers for a user's request.
Be concise, accurate, and friendly. Never give a medical diagnosis."""

class ProviderState(TypedDict, total=False):
    user_input: str
    zip_code: str
    radius_miles: float
    procedure: str
    results: List[Dict[str, str]]

class ProviderAgent:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.2):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    # Utility methods

    def extract_zip_radius(self, text: str) -> tuple[str, float]:
        """Extracts ZIP and radius from user input (defaults to 25mi)."""
        m_zip = re.search(r"\b(\d{5})\b", text)
        zip_code = m_zip.group(1)

        m_radius = re.search(r"(\d{1,2})\s*(?:mi|miles?)", text.lower())
        radius = float(m_radius.group(1)) if m_radius else 25.0
        return zip_code, radius

    def detect_procedure(self, text: str) -> str:
        """LLM-based reasoning to detect what procedure/service the user is asking for."""
        examples = [
            ("colonoscopy", "gastroenterology, colorectal surgery, general surgery"),
            ("mri", "radiology, diagnostic imaging"),
            ("ct scan", "radiology, diagnostic imaging"),
            ("ultrasound", "radiology, diagnostic imaging"),
            ("mammogram", "radiology, breast imaging"),
            ("obgyn", "obstetrics, gynecology, women's health"),
            ("pregnancy", "obstetrics, gynecology"),
            ("physical therapy", "physical therapy, rehabilitation, orthopedics"),
            ("rehab", "physical therapy, rehabilitation"),
            ("internal medicine", "internal medicine, primary care, general practice"),
            ("pediatric", "pediatrics, family medicine"),
        ]
        text_lower = text.lower()
        for key, spec in examples:
            if key in text_lower:
                return spec

        # Fallback to LLM reasoning, but clean its output
        messages = [
            SystemMessage(content="You map a healthcare request to likely specialties or procedures."),
            HumanMessage(content=f"User asked: '{text}'. Return only the most likely specialty names, comma-separated (no explanations)."),
        ]
        result = self.llm.invoke(messages)
        raw = result.content.strip().lower()
        raw = re.sub(r"the most likely.*?is", "", raw)
        raw = re.sub(r"[^a-z, ]", "", raw)
        return raw.strip()


    # Main provider search

    def find_nearby_providers(self, user_query: str) -> str:
        """
        Main entrypoint:
        - Detects procedure and zip/radius.
        - Starts with 15-mile search radius.
        - If none found, automatically expands to 30 miles.
        - Falls back to closest 5 providers if still empty.
        - Outputs formatted list + short summary.
        """
        from pipelines.provider_json_retrieval import filter_providers_by_specialty

        # --- Extract base info ---
        zip_code, _ = self.extract_zip_radius(user_query)
        procedure = self.detect_procedure(user_query)

        initial_radius = 15.0
        expanded_radius = 30.0
        print(f"→ Searching for procedure '{procedure}' near ZIP {zip_code} within {initial_radius} miles")

        try:
            all_providers = scrape_json_url(ANTHEM_URL)
            providers = filter_providers_by_zip(all_providers, zip_code, initial_radius)
        except Exception as e:
            return f"⚠️ Failed to load provider data: {e}"

        # Primary specialty filtering
        specialty_filtered = []
        if procedure:
            specialty_filtered = filter_providers_by_specialty(providers, [procedure])

        # Fallback 1: expand radius if no matches
        if not specialty_filtered:
            print(f"⚠️ No specialty match within {initial_radius} miles. Expanding search radius to {expanded_radius} miles...")
            providers = filter_providers_by_zip(all_providers, zip_code, expanded_radius)
            if procedure:
                specialty_filtered = filter_providers_by_specialty(providers, [procedure])

        # Fallback 2: show any nearby providers if still none
        if not specialty_filtered:
            if not providers:
                return (
                    f"No providers found near {zip_code} within {expanded_radius} miles "
                    f"for '{procedure or 'general care'}'."
                )
            print(f"⚠️ Still no specialty match — showing 5 closest providers within {expanded_radius} miles.")
            specialty_filtered = providers[:5]

        # Format top 5 results
        top_providers = specialty_filtered[:5]
        lines = []
        for p in top_providers:
            parts = [
                f"**{p.get('name', 'Unknown')}**",
                p.get("specialty", ""),
                f"{p.get('address','')}, {p.get('city','')}, {p.get('state','')} {p.get('zip','')}".strip(),
                p.get("phone", ""),
                p.get("website", ""),
            ]
            lines.append(" | ".join([x for x in parts if x]))

        joined = "\n".join(lines)

        # Summary prompt for LLM
        summary_prompt = f"""User asked: "{user_query}"
        Here are nearby providers found in Anthem data:
        {joined}
        Provide a short, friendly summary (2–3 sentences) describing these options and note if the search radius was expanded.
        """
        summary = self.llm.invoke([
            SystemMessage(content=SYSTEM_BASE),
            HumanMessage(content=summary_prompt)
        ])
        return f"{joined}\n\n{summary.content.strip()}"