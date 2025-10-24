from agents.caregiver_agent import CaregiverCompanionAgent
from pipelines.cdc_retrieval_qa import build_cdc_qa
from orchestrators.run_caregiver_graph import run_caregiver_pipeline
from dotenv import load_dotenv
import os

# load_dotenv(dotenv_path=".env")
# print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))  # Debug: should print your key
#
#

def main():
    # Initialize agent
    # agent = CaregiverCompanionAgent(groq_api_key="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")
    agent = CaregiverCompanionAgent(groq_api_key="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")
    # Example medical notes
    notes = """Patient presented with acute rhinitis and persistent cough. Follow-up in 7 days."""

    # Summarize and explain notes
    result = agent.summarize_and_explain(notes)
    print("Summary:", result["summary"])
    print("Action Items:", result["action_items"])

    # CDC QA
    qa_chain = build_cdc_qa(agent.api_key)
    answer = qa_chain.run("What are the treatment options for breast cancer?")
    print("\nCDC QA Response:", answer)

    # Run full caregiver pipeline
    medical_notes = """
    Patient presented with symptoms of acute rhinitis and a persistent cough.
    Vital signs stable. Recommended rest and increased fluid intake.
    Follow-up appointment scheduled in 7 days.
    """

    pipeline_result = run_caregiver_pipeline(medical_notes)

    print("=== Summary ===")
    print(pipeline_result["summary"])

    print("\n=== Explanations ===")
    for e in pipeline_result.get("explanations", []):
        print(f"- {e.get('term')}: {e.get('explanation')}")

    print("\n=== Action Items ===")
    for a in pipeline_result.get("action_items", []):
        print(f"- {a}")


if __name__ == "__main__":
    main()
