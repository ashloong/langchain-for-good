# utils/parse_utils.py
import json
import re

def try_parse_model_output(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return normalize_parsed(parsed)
    except Exception:
        m = re.search(r"(\{[\s\S]*\})", text)
        if m:
            try:
                parsed = json.loads(m.group(1))
                return normalize_parsed(parsed)
            except Exception:
                pass
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        summary = lines[0] if lines else ""
        bullets = [l.lstrip('-* ').strip() for l in lines[1:] if l.startswith(('-', '*'))][:6]
        return {"summary": summary, "explanations": [], "action_items": bullets, "unclear": []}

def normalize_parsed(parsed: dict) -> dict:
    summary = parsed.get("summary") or parsed.get("Summary") or ""
    explanations = parsed.get("explanations") or parsed.get("terms") or []
    if isinstance(explanations, dict):
        explanations = [{"term": k, "explanation": v} for k, v in explanations.items()]
    action_items = parsed.get("action_items") or parsed.get("actions") or []
    unclear = parsed.get("unclear") or []
    return {"summary": summary, "explanations": explanations, "action_items": action_items, "unclear": unclear}
