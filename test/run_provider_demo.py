#!/usr/bin/env python
# coding: utf-8

"""
Demo for ProviderAgent
- Finds and summarizes nearby healthcare providers based on a user query.
- Uses ProviderAgent.find_nearby_providers() under the hood.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agents.provider_agent import ProviderAgent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env.")

print("Using OPENAI_API_KEY (hidden).")

# ======================
# Instantiate Agent
# ======================
agent = ProviderAgent()
print("✅ ProviderCompanionAgent initialized.\n")

# ======================
# Run Mode
# ======================
USAGE = """
Provider Finder Demo (ProviderAgent)
-----------------------------------
Examples:
  python examples/provider_demo.py "colonoscopy in 91706"
  python examples/provider_demo.py "MRI near 91770"
  python examples/provider_demo.py                 # interactive mode
"""

def main():
    # One-off CLI arg: treat remaining args as the query string
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        if not query:
            print(USAGE)
            sys.exit(0)
        print(f"=== Query: {query} ===")
        output = agent.find_nearby_providers(query)
        print(output)
        return

    # Interactive mode
    print(USAGE)
    while True:
        try:
            q = input("> ").strip()
            if not q:
                continue
            print(f"\n=== Query: {q} ===")
            output = agent.find_nearby_providers(q)
            print(output, "\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!\n")
            break

if __name__ == "__main__":
    main()
