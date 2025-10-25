#!/usr/bin/env python
# coding: utf-8

"""
Graph runner for ProviderAgent
- Delegates the heavy lifting to ProviderAgent via the graph.
"""

from __future__ import annotations
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.provider_agent import ProviderAgent
from graphs.provider_graph import build_provider_graph

def run_provider_pipeline(query: str):
    """
    Build the agent + graph and invoke it on a single query.
    Returns the graph output dict (with 'response_text').
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please set it in .env.")

    agent = ProviderAgent()
    app = build_provider_graph(agent)
    return app.invoke({"user_input": query})

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        out = run_provider_pipeline(query)
        print(out.get("response_text", ""))
        return

    print("Provider Graph Runner (Ctrl+C to quit)")
    while True:
        try:
            q = input("> ").strip()
            if not q:
                continue
            out = run_provider_pipeline(q)
            print("\n" + out.get("response_text", "") + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!\n")
            break

if __name__ == "__main__":
    main()
