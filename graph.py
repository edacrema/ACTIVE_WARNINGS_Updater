"""Complete LangGraph workflow for the Active Warnings pipeline.

Wires together all agents and retrievers into a StateGraph that can be
used by the Streamlit app via ``compiled_graph.stream(...)`` or
``compiled_graph.invoke(...)``.
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from state import ActiveWarningsState
from config import MAX_CORRECTION_ATTEMPTS
from agents.query_planner import run_query_planner
from agents.seerist_retriever import run_seerist_retriever
from agents.reliefweb_retriever import run_reliefweb_retriever
from agents.translator import run_translation_agent
from agents.event_extractor import run_event_extractor
from agents.trend_analyst import run_trend_analysis_agent
from agents.narrative_synthesis import run_narrative_synthesis
from agents.skeptic import run_skeptic
from agents.citation_manager import run_citation_manager
from agents.status_recommender import run_status_recommendation


def should_correct(state: ActiveWarningsState) -> Literal["correct", "continue"]:
    """Conditional edge function driving the ReAct loop.

    Checks if the skeptic found flags and if we are under the retry limit.
    """

    print("--- (Checking Skeptic results...) ---")

    flags = state.get("skeptic_flags")
    attempts = state.get("correction_attempts", 0)

    if attempts >= MAX_CORRECTION_ATTEMPTS:
        print(
            f"   > Max correction attempts ({attempts}) reached. "
            "Continuing with errors.",
        )
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(
            "Max correction attempts reached; draft may contain errors.",
        )
        return "continue"

    if not flags:
        print("   > Draft is clean. Continuing pipeline.")
        return "continue"

    print("   > Draft has errors. Looping back to synthesis.")
    return "correct"


workflow = StateGraph(ActiveWarningsState)

# 1. Add all nodes (9 agents + 2 retrievers)
workflow.add_node("planner", run_query_planner)
workflow.add_node("seerist_retriever", run_seerist_retriever)
workflow.add_node("reliefweb_retriever", run_reliefweb_retriever)
workflow.add_node("translator", run_translation_agent)
workflow.add_node("extractor", run_event_extractor)
workflow.add_node("trend_analysis", run_trend_analysis_agent)
workflow.add_node("synthesis", run_narrative_synthesis)
workflow.add_node("skeptic", run_skeptic)
workflow.add_node("citation_manager", run_citation_manager)
workflow.add_node("status_recommender", run_status_recommendation)

# 2. Entry point
workflow.set_entry_point("planner")

# 3. Data Gathering (serial execution)
workflow.add_edge("planner", "seerist_retriever")
workflow.add_edge("seerist_retriever", "reliefweb_retriever")

# 4. Processing and Analysis
workflow.add_edge("reliefweb_retriever", "translator")
workflow.add_edge("translator", "extractor")
workflow.add_edge("extractor", "trend_analysis")

# 5. The "Synthesize-and-Correct" ReAct Loop
workflow.add_edge("trend_analysis", "synthesis")
workflow.add_edge("synthesis", "skeptic")

workflow.add_conditional_edges(
    "skeptic",
    should_correct,
    {
        "correct": "synthesis",  # Loop back to synthesis to fix errors
        "continue": "citation_manager",  # Exit loop and continue to citations
    },
)

# 6. Final Steps
workflow.add_edge("citation_manager", "status_recommender")
workflow.add_edge("status_recommender", END)


compiled_graph = workflow.compile()

print("Active Warnings Graph compiled successfully!")
