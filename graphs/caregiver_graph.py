from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import List, Optional


class CaregiverState(BaseModel):
    notes: str
    summary: Optional[str] = None
    explanations: Optional[List[str]] = None
    action_items: Optional[List[str]] = None


def build_caregiver_graph(agent):
    """
    Build a LangGraph pipeline for the Caregiver Companion agent.
    The `agent` should be an instance of CaregiverCompanionAgent.
    """

    def summarize_and_explain_node(state: CaregiverState):
        result = agent.summarize_and_explain(state.notes)
        return {
            "summary": result.get("summary"),
            "explanations": result.get("explanations"),
            "action_items": result.get("action_items"),
        }

    # --- Define the graph structure ---
    graph = StateGraph(CaregiverState)
    graph.add_node("summarize_and_explain", summarize_and_explain_node)

    # Entry and exit
    graph.add_edge(START, "summarize_and_explain")
    graph.add_edge("summarize_and_explain", END)

    # Compile the graph to an executable app
    return graph.compile()
