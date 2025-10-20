from agents.caregiver_agent import CaregiverCompanionAgent
from pipelines.cdc_retrieval_qa import build_cdc_qa

if __name__ == "__main__":
    agent = CaregiverCompanionAgent()
    notes = """Patient presented with acute rhinitis and persistent cough. Follow-up in 7 days."""
    result = agent.summarize_and_explain(notes)

    print("Summary:", result["summary"])
    print("Action Items:", result["action_items"])

    qa_chain = build_cdc_qa(agent.api_key)
    answer = qa_chain.run("What are the treatment options for breast cancer?")
    print("\nCDC QA Response:", answer)
