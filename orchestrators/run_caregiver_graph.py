from agents.caregiver_agent import CaregiverCompanionAgent
from graphs.caregiver_graph import build_caregiver_graph
from utils.config import GROQ_API_KEY

def run_caregiver_pipeline(notes: str):
    agent = CaregiverCompanionAgent(GROQ_API_KEY)
    app = build_caregiver_graph(agent)
    return app.invoke({"notes": notes})
