#!/usr/bin/env python
# coding: utf-8

"""
main.py — Unified runner using the combined final graph.

This app routes each request to either:
  - Caregiver pipeline (summarize/explain medical notes), or
  - Provider pipeline (find nearby providers)

Routing:
  - --mode caregiver|provider|auto (default: auto)
  - In auto mode, the combined graph auto-detects based on the input text.

Examples:
  python main.py --mode provider "colonoscopy in 91706"
  python main.py --mode caregiver "Patient has acute rhinitis..."
  python main.py --mode auto "MRI near 91770"
  python main.py    # interactive
"""

from __future__ import annotations
import argparse
import os
import sys

# Add repo root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Agents & combined graph
from agents.caregiver_agent import CaregiverCompanionAgent
from agents.provider_agent import ProviderAgent
from graphs.final_graph import build_final_graph


def build_app():
    """
    Instantiate both agents and compile the combined graph.
    Notes:
      - ProviderAgent expects OPENAI_API_KEY in the environment.
      - CaregiverCompanionAgent may use GROQ_API_KEY depending on your implementation.
    """
    # Provider requires OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please set it in .env.")

    # Caregiver (pass GROQ key if your class expects it; otherwise default construct)
    groq = os.getenv("GROQ_API_KEY")
    try:
        caregiver_agent = CaregiverCompanionAgent(groq) if groq else CaregiverCompanionAgent()
    except TypeError:
        # If your constructor strictly requires a key
        if not groq:
            raise ValueError("GROQ_API_KEY not found. Please set it in .env for caregiver mode.")
        caregiver_agent = CaregiverCompanionAgent(groq)

    provider_agent = ProviderAgent()
    app = build_final_graph(caregiver_agent, provider_agent)
    return app


def run_once(app, mode: str | None, text: str) -> str:
    """
    Invoke the combined graph once.
    Inputs:
      - mode: 'provider' | 'caregiver' | None (None = auto routing inside the graph)
      - text: user text (notes or provider query)
    Output:
      - response_text (str)
    """
    state = {"mode": mode, "text": text}
    result = app.invoke(state)
    if isinstance(result, dict):
        return str(result.get("response_text", result))
    return str(result)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified runner (combined graph)")
    p.add_argument("--mode", choices=["provider", "caregiver", "auto"], default="auto",
                   help="Pipeline mode. Default: auto (graph routes by itself).")
    p.add_argument("text", nargs="*", help="Input text (provider query or caregiver notes).")
    return p.parse_args()


def main():
    app = build_app()
    args = parse_args()

    # One-shot CLI
    if args.text:
        text = " ".join(args.text).strip()
        mode = None if args.mode == "auto" else args.mode
        out = run_once(app, mode, text)
        print(out)
        return

    # Interactive
    banner = """Combined Graph Runner (Provider + Caregiver)
------------------------------------------------
Type your request (Ctrl+C to quit). Examples:
  - colonoscopy in 91706
  - MRI near 94102
  - Please summarize these notes: ...
"""
    print(banner)
    while True:
        try:
            s = input("> ").strip()
            if not s:
                continue
            out = run_once(app, None, s)  # None => auto routing inside graph
            print("\n" + out + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!\n")
            break

if __name__ == "__main__":
    main()