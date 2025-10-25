#!/usr/bin/env python
# coding: utf-8

"""
Run a final router graph that can handle:
  - Caregiver notes summarization/explanation
  - Provider lookups (nearby providers)

Usage:
  # Auto-detect mode (recommended)
  python examples/run_final_graph.py "colonoscopy in 91706"
  python examples/run_final_graph.py "Please summarize these notes: ..."

  # Force mode
  python examples/run_final_graph.py --mode provider "MRI near 91770"
  python examples/run_final_graph.py --mode caregiver "Patient has acute rhinitis..."

  # Interactive
  python examples/run_final_graph.py
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.caregiver_agent import CaregiverCompanionAgent
from agents.provider_agent import ProviderAgent
from graphs.final_graph import build_final_graph


def build_agents():
    # Caregiver can depend on GROQ or OPENAI depending on your implementation.
    # In your repo, CaregiverCompanionAgent takes GROQ_API_KEY; ProviderAgent uses OPENAI_API_KEY.
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Provider Agent requires OPENAI_API_KEY
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in .env.")

    # Caregiver agent (match your current constructor; if it needs a key, pass it)
    caregiver_agent = CaregiverCompanionAgent(groq_key) if groq_key else CaregiverCompanionAgent()

    provider_agent = ProviderAgent()
    return caregiver_agent, provider_agent


def main():
    parser = argparse.ArgumentParser(description="Run Combined Caregiver/Provider Graph")
    parser.add_argument("--mode", choices=["caregiver", "provider"], help="Force a specific mode.")
    parser.add_argument("text", nargs="*", help="Input text (notes or provider query)")
    args = parser.parse_args()

    caregiver_agent, provider_agent = build_agents()
    app = build_final_graph(caregiver_agent, provider_agent)

    if args.text:
        text = " ".join(args.text).strip()
        out = app.invoke({"mode": args.mode, "text": text})
        print(out.get("response_text", str(out)))
        return

    # Interactive
    banner = """Final Graph Runner
--------------------------------
Type a request (Ctrl+C to quit). Examples:
  - colonoscopy in 91706
  - MRI near 91770
  - Please summarize and explain these notes: ...
"""
    print(banner)
    while True:
        try:
            s = input("> ").strip()
            if not s:
                continue
            out = app.invoke({"mode": None, "text": s})
            print("\n" + out.get("response_text", str(out)) + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!\n")
            break

if __name__ == "__main__":
    main()