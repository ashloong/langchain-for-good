# pipelines/provider_json_retrieval.py
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List

import requests

# Optional zipcode DB (for geo filtering)
try:
    import zipcodes  # pip install zipcodes
except Exception:
    zipcodes = None

# Optional vector search (nice-to-have). We fall back if unavailable or rate-limited.
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# Network fetch
def fetch_text_from_url(url: str, timeout: int = 25) -> str:
    """Fetch raw text content from a URL (raises on error)."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text

# Normalization helpers
def _normalize_zip(value: str) -> str:
    """Return first 5-digit ZIP from a string, else ''."""
    if not value:
        return ""
    m = re.search(r"\b(\d{5})\b", str(value))
    return m.group(1) if m else ""


def _join_specialty(spec: Any) -> str:
    """Anthem often uses a list for 'specialty'. Normalize to a single string."""
    if not spec:
        return ""
    if isinstance(spec, list):
        return ", ".join(s.strip() for s in spec if isinstance(s, str) and s.strip())
    if isinstance(spec, str):
        return spec.strip()
    return ""

# JSON normalization (address-aware)
def scrape_json_url(url: str, timeout: int = 25) -> List[Dict[str, Any]]:
    """
    Fetch provider JSON and FLATTEN to one record per address.

    Normalized keys per record:
      name, phone, address, city, state, zip, specialty, website, _raw
    """
    raw = fetch_text_from_url(url, timeout=timeout)
    data = json.loads(raw)

    # If top-level is a dict, try to find the first list-like value
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array (or a dict containing an array) of providers.")

    normalized: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        # Name can be facility_name OR nested name {first,last}
        name = ""
        nm = item.get("name")
        if isinstance(nm, dict):
            first = (nm.get("first") or "").strip()
            last = (nm.get("last") or "").strip()
            name = " ".join([n for n in [first, last] if n]).strip()
        name = name or (item.get("facility_name") or item.get("provider_name") or item.get("name") or "").strip()
        if not name:
            name = "Unknown Provider"

        specialty = _join_specialty(item.get("specialty"))
        website = (item.get("website") or item.get("url") or "").strip()

        # Anthem-style: addresses is an array of dicts
        addrs = item.get("addresses") or []
        if not isinstance(addrs, list):
            # try a single address dict (rare)
            addrs = [addrs] if isinstance(addrs, dict) else []

        for addr in addrs:
            if not isinstance(addr, dict):
                continue

            street = (addr.get("address") or addr.get("Address1") or "").strip()
            city = (addr.get("city") or addr.get("City") or "").strip()
            state = (addr.get("state") or addr.get("State") or "").strip()
            zip5 = _normalize_zip(addr.get("zip") or addr.get("PostalCode") or "")
            phone = (addr.get("phone") or item.get("phone") or "").strip()

            # must have a usable zip for geo features
            if not zip5:
                continue

            normalized.append(
                {
                    "name": name,
                    "phone": phone,
                    "address": street,
                    "city": city,
                    "state": state,
                    "zip": zip5,          # 5-digit
                    "specialty": specialty,
                    "website": website,
                    "_raw": item,         # original provider object (for debugging)
                }
            )

    # Basic telemetry
    # print(f"Parsed {len(normalized)} provider-address rows.")
    return normalized

# Geo filtering by ZIP radius
def get_zip_codes_within_distance(target_zip: str, radius_miles: float) -> List[str]:
    """
    Returns zip codes within X miles from a target zip code using a bounding-box optimization.
    If `zipcodes` is unavailable or target_zip invalid, returns [target_zip] as a safe fallback.
    """
    tz = _normalize_zip(target_zip)
    if not tz or not zipcodes:
        return [tz] if tz else []

    info = zipcodes.matching(tz)
    if not info:
        return [tz]

    lat = float(info[0]["lat"])
    lon = float(info[0]["long"])
    lat_offset = radius_miles / 69.0
    lon_offset = radius_miles / (69.0 * math.cos(math.radians(lat)))

    min_lat, max_lat = lat - lat_offset, lat + lat_offset
    min_lon, max_lon = lon - lon_offset, lon + lon_offset

    nearby = []
    for z in zipcodes.list_all():
        try:
            zl = float(z["lat"])
            zlon = float(z["long"])
            if min_lat <= zl <= max_lat and min_lon <= zlon <= max_lon:
                z5 = _normalize_zip(z.get("zip_code", ""))
                if z5:
                    nearby.append(z5)
        except Exception:
            continue

    return sorted(set(nearby))


def filter_providers_by_zip(
    providers: List[Dict[str, Any]],
    target_zip: str,
    radius_miles: float,
) -> List[Dict[str, Any]]:
    """Return providers whose ZIP is within the radius of target_zip."""
    target_zip = _normalize_zip(target_zip)
    if not target_zip:
        return []

    allowed = set(get_zip_codes_within_distance(target_zip, radius_miles))
    out = []
    for p in providers:
        zp = _normalize_zip(p.get("zip", ""))
        if zp and zp in allowed:
            out.append(p)
    return out

# Fuzzy specialty filtering with broad synonyms
def filter_providers_by_specialty(providers: List[Dict[str, Any]], specialties: List[str]) -> List[Dict[str, Any]]:
    """
    Fuzzy specialty filter for provider lists.
    - Case-insensitive, partial-word matching
    - Splits comma-/slash-separated input
    - Adds synonyms for common procedures / domains
    """
    if not specialties:
        return providers

    # Normalize desired specialties → tokens
    tokens: set[str] = set()
    for s in specialties:
        for part in re.split(r"[,/]", s.lower()):
            part = part.strip()
            if part:
                tokens.add(part)

    # Synonyms / category expansion
    synonyms = {
        # GI & colon procedures
        "gastro": ["digestive", "colon", "rectal", "bowel", "endoscopy", "colorectal"],
        "colorectal": ["colon", "rectal", "proctology", "colon and rectal"],
        "digestive": ["gastro", "colon", "bowel"],

        # Imaging & diagnostics
        "radiology": ["imaging", "diagnostic", "mri", "ct", "scan", "x-ray", "ultrasound", "nuclear medicine", "mammogram", "breast imaging"],
        "mri": ["radiology", "imaging", "diagnostic", "magnetic resonance"],
        "x-ray": ["radiology", "imaging", "diagnostic"],
        "ultrasound": ["radiology", "imaging", "sonography"],
        "mammogram": ["radiology", "breast imaging"],

        # Cardiology & heart
        "cardiology": ["cardiac", "heart", "vascular", "echocardiography", "cardiovascular", "angiogram"],
        "cardiac": ["cardiology", "heart", "cardiothoracic"],

        # Women's health
        "obstetric": ["obstetrics", "gynecology", "women", "pregnancy", "ob/gyn", "obgyn"],
        "gynecology": ["obstetrics", "women", "ob/gyn", "obgyn", "female"],

        # Primary care
        "family": ["primary care", "general practice", "internal medicine", "pediatrics"],
        "internal": ["internal medicine", "primary care", "general practice"],
        "pediatric": ["child", "children", "pediatrics", "family"],

        # Orthopedics & physical therapy
        "orthopedic": ["sports medicine", "physical therapy", "rehabilitation", "joint", "musculoskeletal"],
        "rehabilitation": ["physical therapy", "occupational therapy", "sports medicine"],
        "therapy": ["physical therapy", "occupational therapy", "rehab", "speech therapy"],

        # Surgery
        "surgery": ["surgical", "general surgery", "orthopedic surgery", "colorectal surgery", "cardiac surgery", "plastic surgery"],
        "plastic": ["cosmetic", "reconstructive", "aesthetic surgery"],
        "urology": ["urinary", "kidney", "bladder", "prostate"],

        # Dentistry & vision
        "dental": ["dentistry", "oral", "teeth"],
        "optometry": ["eye", "vision", "ophthalmology"],
        "ophthalmology": ["optometry", "eye", "vision"],

        # Mental health
        "psychiatry": ["mental health", "psychology", "behavioral health", "therapy"],
        "psychology": ["counseling", "mental health", "behavioral health"],
    }

    for t in list(tokens):
        for base, syns in synonyms.items():
            if base in t:
                tokens.update(syns)

    out = []
    for p in providers:
        spec = str(p.get("specialty", "")).lower()
        # normalize punctuation to spaces so "OB/GYN" matches "obgyn"
        spec = re.sub(r"[^a-z0-9 ]", " ", spec)
        if not spec:
            continue
        if any(tok in spec for tok in tokens):
            out.append(p)

    # If nothing matched, return original list so caller can still show nearby options
    return out or providers


# Simple retriever (vector if available, else keyword)
def _keyword_score(text: str, query: str) -> float:
    hits = 0
    q_terms = set(re.findall(r"[A-Za-z0-9]+", query.lower()))
    for t in q_terms:
        if t and t in text.lower():
            hits += 1
    return hits / max(1, len(q_terms))


def simple_retriever(docs: List[str], query: str, k: int = 4) -> List[str]:
    """
    Retrieve top-k strings from a list of texts.

    1) If FAISS + OpenAI embeddings are available, use vector search.
    2) On any embedding API/rate/other error, fall back to keyword scoring.
    3) Otherwise always use keyword scoring.
    """
    if _HAS_FAISS:
        try:
            embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env
            vs = FAISS.from_texts(docs, embedding=embeddings)
            return [d.page_content for d in vs.similarity_search(query, k=k)]
        except Exception as e:
            # Graceful fallback
            print(f"⚠️ Embedding retrieval unavailable, falling back to keyword scoring ({type(e).__name__}: {e})")

    scored = sorted(docs, key=lambda d: _keyword_score(d, query), reverse=True)
    return scored[:k]
