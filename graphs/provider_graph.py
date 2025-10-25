#!/usr/bin/env python
# coding: utf-8

"""
Minimal LangGraph wrapper around ProviderAgent.
- Takes user_input in the graph state
- Calls ProviderAgent.find_nearby_providers()
- Returns response_text
"""

from __future__ import annotations
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agents.provider_agent import ProviderAgent


# Graph State
class ProviderState(TypedDict, total=False):
    user_input: str
    response_text: str


# Node: run the agent
def node_run_agent(state: ProviderState, *, agent: ProviderAgent) -> ProviderState:
    user_input = state.get("user_input", "") or ""
    output = agent.find_nearby_providers(user_input)
    return {"response_text": output}

# Builder
def build_provider_graph(agent: ProviderAgent):
    """
    Build a minimal graph that:
      START -> run_agent -> END
    """
    builder = StateGraph(ProviderState)
    builder.add_node("run_agent", lambda s: node_run_agent(s, agent=agent))
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", END)

    app = builder.compile()
    return app