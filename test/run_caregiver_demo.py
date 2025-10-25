#!/usr/bin/env python
# coding: utf-8

"""
Demo for CaregiverCompanionAgent
- Summarizes and explains medical notes in layman's terms.
"""

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from agents.caregiver_agent import CaregiverCompanionAgent
from utils.config import GROQ_API_KEY
# ======================
# Check API Key
# ======================
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in .env.")

# ======================
# Instantiate Agent
# ======================


print("Using GROQ_API_KEY:", GROQ_API_KEY)

agent = CaregiverCompanionAgent(groq_api_key=GROQ_API_KEY)


# ======================
# Sample Medical Notes
# ======================
medical_notes = """
Patient presented with symptoms of acute rhinitis and a persistent cough.
Vital signs stable. Recommended rest and increased fluid intake.
Follow-up appointment scheduled in 7 days.
"""

# ======================
# Summarize & Explain
# ======================
output = agent.summarize_and_explain(medical_notes)

# ======================
# Print Results
# ======================
print("=== Summary ===")
print(output['summary'])

print("\n=== Explanations ===")
if output.get('explanations'):
    for item in output['explanations']:
        if isinstance(item, dict):
            term = item.get('term') or 'Unknown term'
            expl = item.get('explanation') or str(item)
            print(f"- {term}: {expl}")
        else:
            print(f"- {item}")
else:
    print("None")

print("\n=== Action Items ===")
for item in output['action_items']:
    print(f"- {item}")

if output.get('unclear'):
    print("\n=== Unclear/Missing Information ===")
    for item in output['unclear']:
        print(f"- {item}")
