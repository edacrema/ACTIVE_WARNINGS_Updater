"""Agent 9: Status Recommendation Agent.

Recommends a risk status change using the WFP Watch List 5x5 methodology.

Implements a hybrid "LLM-Score + Code-Logic" approach:
1.  **LLM-Score (Step 1):** The LLM scores the *current* Likelihood and Impact
    based on the new data and the official WFP methodology.
2.  **Code-Logic (Step 2):** Python code retrieves the *previous* scores from
    the state, calculates geometric means for both periods, and applies
    the business logic to determine the status change.
"""

import json
import math
from typing import Any, Dict, List, TypedDict

from langchain_core.prompts import ChatPromptTemplate

from state import (
    ActiveWarningsState,
    SeriousnessScores,
    StatusRecommendation,
)
from llm import llm


class CurrentScores(TypedDict):
    """Structured output for the Step 1 LLM scoring chain."""

    current_likelihood: int  # Score 1-5
    current_impact: int  # Score 1-5
    rationale: str  # Rationale for these *two* scores


SCORING_PROMPT_TEMPLATE = """
You are a senior humanitarian risk analyst. Your sole task is to score the **CURRENT** risk based on the official WFP 5x5 Watch List methodology.
You will be given the new data (events and trends) for the current 2-month period.

**Country:** {country}
**Risk Type:** {risk_type}

**WFP Scoring Methodology (MANDATORY):**
You MUST use these definitions to assign scores from 1 (Negligible) to 5 (Very High).

1.  **Likelihood (1-5):** The estimated probability of the risk occurring/escalating in the next 3-6 months.
    - 5 (Very High): 51-100%
    - 4 (High): 31-50%
    - 3 (Moderate): 16-30%
    - 2 (Low): 6-15%
    - 1 (Negligible): <5%

2.  **Impact (1-5):** The estimated number of *additional* people needing humanitarian assistance.
    - 5 (Very High): >500,000
    - 4 (High): 250,000 - 500,000
    - 3 (Moderate): 100,000 - 250,000
    - 2 (Low): 20,000 - 100,000
    - 1 (Negligible): <20,000

**Current Period Data:**

**Trend Analysis:**
```json
{trend_analysis_json}
```

**Key Events (Summary):**
```json
{events_json}
```

**Instructions:**
1. Analyze the Trend Analysis (especially the trajectory and outlook_factors).
2. Analyze the Key Events (look for figures on displacement, fatalities, inflation, etc.).
3. Based only on this data and the methodology, determine the current_likelihood score (1-5).
4. Determine the current_impact score (1-5).
5. Provide a brief rationale explaining why you chose those two scores.
6. You MUST return a single JSON object matching the CurrentScores schema.

**Output Schema:**
{{"current_likelihood": <int>, "current_impact": <int>, "rationale": "..."}}
"""


def run_status_recommendation(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function to run the Status Recommendation.

    (Corrected version using manual JSON parsing.)
    """

    print(f"--- (9) Running Status Recommendation Agent for {state['country']} ---")

    if state.get("warnings") is None:
        state["warnings"] = []

    current_events = state.get("events")
    trend_analysis = state.get("trend_analysis")
    previous_scores = state.get("previous_seriousness_scores")

    # ----- Guard Clauses -----
    if not current_events or not trend_analysis:
        msg = "StatusRecommendation skipped: Missing events or trend analysis."
        print(f"   > {msg}")
        state["warnings"].append(msg)
        state["current_step"] = "StatusRecommendationSkipped"
        return state

    if not previous_scores:
        msg = (
            "StatusRecommendation skipped: Missing *previous* L-I scores "
            "(required input)."
        )
        print(f"   > {msg}")
        state["warnings"].append(msg)
        state["error"] = "Missing previous_seriousness_scores"
        state["current_step"] = "StatusRecommendationSkipped"
        return state

    try:
        # -------------------------------------------------
        # STEP 1: LLM scores the CURRENT period
        # -------------------------------------------------
        print("   > Step 1: LLM scoring current Likelihood & Impact...")
        score_prompt = ChatPromptTemplate.from_template(SCORING_PROMPT_TEMPLATE)
        score_chain = score_prompt | llm

        event_summary = [
            (
                "Event: "
                f"{evt.get('event_type', 'N/A')} - "
                f"{evt.get('statement', 'No statement')}"
            )
            for evt in current_events[:10]
            if isinstance(evt, dict)
        ]

        score_response = score_chain.invoke(
            {
                "country": state["country"],
                "risk_type": ", ".join(state["risk_type"]),
                "trend_analysis_json": json.dumps(trend_analysis, indent=2),
                "events_json": json.dumps(event_summary, indent=2),
            }
        )
        score_raw_output = score_response.content

        # --- IMPROVED JSON PARSING ---
        try:
            json_part = score_raw_output.strip()

            if json_part.startswith("```json"):
                json_part = json_part[7:]
                if json_part.endswith("```"):
                    json_part = json_part[:-3]
            elif json_part.startswith("```"):
                json_part = json_part[3:]
                if json_part.endswith("```"):
                    json_part = json_part[:-3]

            json_part = json_part.strip()

            start_index = json_part.find("{")
            end_index = json_part.rfind("}")

            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_to_parse = json_part[start_index : end_index + 1]
                current_scores_output: CurrentScores = json.loads(json_to_parse)
            else:
                current_scores_output = json.loads(json_part)

            if not all(
                k in current_scores_output
                for k in ["current_likelihood", "current_impact", "rationale"]
            ):
                raise ValueError("Missing required keys in CurrentScores")
            if not isinstance(current_scores_output.get("current_likelihood"), int) or not isinstance(
                current_scores_output.get("current_impact"),
                int,
            ):
                raise ValueError("Likelihood or Impact is not an integer")
        except (json.JSONDecodeError, ValueError) as parse_error:
            print(
                "   ! ERROR: Failed to parse JSON for current scores: "
                f"{parse_error}",
            )
            print(f"   Raw LLM Output:\n{score_raw_output}")
            raise ValueError(
                "LLM did not return valid JSON for CurrentScores. "
                f"Raw output: {score_raw_output}"
            ) from parse_error

        print("   > LLM scoring complete.")

        # -------------------------------------------------
        # STEP 2: Python calculates and compares
        # -------------------------------------------------
        print("   > Step 2: Calculating geometric means and determining status...")

        prev_L = previous_scores.get("likelihood")
        prev_I = previous_scores.get("impact")
        if prev_L is None or prev_I is None:
            raise ValueError(
                "Previous scores dict is missing likelihood or impact.",
            )

        curr_L = current_scores_output["current_likelihood"]
        curr_I = current_scores_output["current_impact"]

        prev_L, prev_I = max(1, min(5, prev_L)), max(1, min(5, prev_I))
        curr_L, curr_I = max(1, min(5, curr_L)), max(1, min(5, curr_I))

        MODERATE_THRESHOLD = 3

        prev_seriousness_score = round(math.sqrt(prev_L * prev_I))
        curr_seriousness_score = round(math.sqrt(curr_L * curr_I))

        status_change: StatusRecommendation["status_change"]
        if (
            prev_seriousness_score < MODERATE_THRESHOLD
            and curr_seriousness_score >= MODERATE_THRESHOLD
        ):
            status_change = "Reactivated"
        elif curr_seriousness_score < MODERATE_THRESHOLD:
            status_change = "Closed"
        elif curr_seriousness_score > prev_seriousness_score:
            status_change = "Increased"
        elif curr_seriousness_score < prev_seriousness_score:
            status_change = "Decreased"
        else:
            status_change = "Remains"

        print(f"   > Status Recommendation: {status_change}")

        # -------------------------------------------------
        # STEP 3: Format and Update State
        # -------------------------------------------------
        final_rationale = (
            f"Determined status: {status_change}. "
            f"Seriousness score changed from {prev_seriousness_score} "
            f"(L{prev_L}, I{prev_I}) to {curr_seriousness_score} "
            f"(L{curr_L}, I{curr_I}). "
            "LLM Rationale for current score: "
            f"{current_scores_output.get('rationale', 'N/A')}"
        )

        current_seriousness_typed: SeriousnessScores = {
            "likelihood": curr_L,
            "impact": curr_I,
            "rationale": current_scores_output.get("rationale", "N/A"),
        }

        recommendation: StatusRecommendation = {
            "previous_seriousness": previous_scores,
            "current_seriousness": current_seriousness_typed,
            "status_change": status_change,
            "rationale": final_rationale,
        }

        state["status_recommendation"] = recommendation
        state["current_step"] = "StatusRecommendationComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Status Recommendation Agent: {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"StatusRecommendationError: {str(e)}")

    return state
