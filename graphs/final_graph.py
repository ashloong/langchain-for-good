#!/usr/bin/env python
# coding: utf-8

"""
A router graph that combines:
  - Caregiver pipeline (summarize/explain medical notes)
  - Provider pipeline (find nearby providers based on a user query)

Routing:
  - If explicit `mode` is given in state, we respect it.
  - Else we auto-detect:
      * If text contains a 5-digit ZIP or provider-ish terms  -> 'provider'
      * Otherwise                                              -> 'caregiver'
"""

from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any
import re

from langgraph.graph import StateGraph, START, END
from agents.caregiver_agent import CaregiverCompanionAgent
from graphs.caregiver_graph import build_caregiver_graph
from agents.provider_agent import ProviderAgent
from graphs.provider_graph import build_provider_graph

# ----------------------------
# Combined Graph State
# ----------------------------
class CombinedState(TypedDict, total=False):
    # Inputs
    mode: str                 # 'caregiver' | 'provider' (optional; if absent we route)
    text: str                 # generic input; for caregiver: notes, for provider: query
    notes: str                # explicit caregiver input (optional)
    user_input: str           # explicit provider input (optional)

    # Internals
    routed_mode: str          # final resolved mode

    # Outputs
    response_text: str        # unified textual response
    raw_result: Dict[str, Any]  # full raw result from subgraph (for caregiver it may be a dict)

# ----------------------------
# Intent Router
# ----------------------------
PROVIDER_HINTS = [
    "find provider", "providers near", "doctor near", "mri", "x-ray",
    "xray", "scan", "colonoscopy", "obgyn", "pediatric", "cardiology",
    "gastro", "orthopedic", "therapy", "imaging",
]

ZIP_RE = re.compile(r"\b\d{5}\b")

def _auto_route(text: str) -> str:
    if not text:
        return "caregiver"
    if ZIP_RE.search(text):
        return "provider"
    t = text.lower()
    if any(h in t for h in PROVIDER_HINTS):
        return "provider"
    return "caregiver"

# ----------------------------
# Nodes
# ----------------------------
def node_route(state: CombinedState) -> CombinedState:
    # Priority: explicit mode > auto-detect
    if state.get("mode") in ("caregiver", "provider"):
        routed = state["mode"]
    else:
        # pick a text field to inspect
        txt = state.get("text") or state.get("user_input") or state.get("notes") or ""
        routed = _auto_route(txt)
    return {**state, "routed_mode": routed}

def node_run_caregiver(state: CombinedState, *, caregiver_agent: CaregiverCompanionAgent) -> CombinedState:
    """
    Delegates to the caregiver graph.
    Input precedence for notes:
      - state.notes
      - state.text
    """
    notes = (state.get("notes") or state.get("text") or "").strip()
    app = build_caregiver_graph(caregiver_agent)
    result = app.invoke({"notes": notes})  # caregiver graph convention

    # Many caregiver graphs return structured dict with keys like summary/explanations/action_items.
    # We'll generate a readable response_text here, but also return raw_result for callers.
    parts = []
    if isinstance(result, dict):
        if result.get("summary"):
            parts.append(f"=== Summary ===\n{result['summary']}")
        if result.get("explanations"):
            parts.append("=== Explanations ===\n" + "\n".join(
                f"- {e.get('term', 'Term')}: {e.get('explanation', '')}" if isinstance(e, dict) else f"- {e}"
                for e in result["explanations"]
            ))
        if result.get("action_items"):
            parts.append("=== Action Items ===\n" + "\n".join(f"- {a}" for a in result["action_items"]))
        if result.get("unclear"):
            parts.append("=== Unclear / Missing ===\n" + "\n".join(f"- {u}" for u in result["unclear"]))

    response_text = "\n\n".join(parts) if parts else str(result)
    return {**state, "raw_result": result, "response_text": response_text}


def node_run_provider(state: CombinedState, *, provider_agent: ProviderAgent) -> CombinedState:
    """
    Delegates to the provider graph/agent.
    Input precedence for query:
      - state.user_input
      - state.text
    """
    query = (state.get("user_input") or state.get("text") or "").strip()

    # We can either go through the graph or call the agent directly.
    # For consistency with your provider_graph, we use the graph:
    app = build_provider_graph(provider_agent)
    result = app.invoke({"user_input": query})
    # provider_graph returns {'response_text': "..."}
    response_text = ""
    if isinstance(result, dict):
        response_text = str(result.get("response_text", "")) or str(result)
    else:
        response_text = str(result)

    return {**state, "raw_result": result, "response_text": response_text}

# ----------------------------
# Builder
# ----------------------------
def build_final_graph(
    caregiver_agent: CaregiverCompanionAgent,
    provider_agent: ProviderAgent,
):
    """
    START -> route -> (caregiver || provider) -> END
    """
    builder = StateGraph(CombinedState)

    builder.add_node("route", node_route)
    builder.add_node("caregiver", lambda s: node_run_caregiver(s, caregiver_agent=caregiver_agent))
    builder.add_node("provider", lambda s: node_run_provider(s, provider_agent=provider_agent))

    builder.add_edge(START, "route")

    def _next(state: CombinedState):
        mode = state.get("routed_mode", "caregiver")
        return "provider" if mode == "provider" else "caregiver"

    builder.add_conditional_edges("route", _next, {"caregiver": "caregiver", "provider": "provider"})
    builder.add_edge("caregiver", END)
    builder.add_edge("provider", END)

    return builder.compile()