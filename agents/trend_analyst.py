"""
Agent 5: Trend Analysis Agent (Hybrid Extract-then-Compare)
Determines the risk trajectory by comparing current events to the previous warning narrative.

Implements a two-step "Extract-then-Compare" process:
1.  **Extract (Step 1):** Runs a lightweight extraction on the
    `state["previous_warning"]` text to get a list of key-value indicators.
2.  **Compare (Step 2):** Passes the extracted *previous* indicators AND
    the new `state["events"]` to a master "analyst" LLM. This LLM
    compares comparable data, flags new events, and makes a
    holistic assessment.

Reads from:
- state["previous_warning"] (for Step 1)
- state["events"] (for Step 2)
- state["country"]
- state["risk_type"]

Writes to:
- state["trend_analysis"] (TrendAnalysis object)
"""

import json
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.prompts import ChatPromptTemplate

from state import ActiveWarningsState, TrendAnalysis
from llm import llm


class IndicatorKV(TypedDict):
    """A simple key-value pair for an extracted past indicator."""

    indicator_type: str  # e.g., "Food Inflation", "Fatalities", "IDPs"
    value: str  # e.g., "10%", "approx 50", "1.2 million total"
    location: Optional[str]
    statement: str  # The sentence it was extracted from


class PreviousIndicators(TypedDict):
    """The structured output for the Step 1 extraction chain."""

    indicators: List[IndicatorKV]


EXTRACT_PREVIOUS_PROMPT = """
You are a fast, lightweight data extraction assistant.
Read the following humanitarian risk narrative. Your *only* job is to extract the main quantitative figures and key indicators mentioned.
Do not analyze or interpret, just extract.

**Risk Type to Focus On:** {risk_type}
**Narrative to Extract From:**
---
{previous_warning}
---

**Instructions:**
1.  Extract all key figures for the risk type (e.g., inflation %, currency rates, fatality counts, displacement numbers, people affected).
2.  For each figure, provide the 'value' and what 'indicator_type' it represents.
3.  Also, extract the original 'statement' (sentence) where you found the indicator.
4.  You MUST return a JSON object matching the `PreviousIndicators` schema.
5.  If no indicators are found, return {{"indicators": []}}.

**Output Schema:**
{{"indicators": [{{"indicator_type": "...", "value": "...", "location": "...", "statement": "..."}}]}}
"""


COMPARE_TRENDS_PROMPT = """
You are a senior humanitarian risk analyst. Your task is to analyze the trend of a risk by comparing sparse data from a **Previous Warning (Period 1)** with new, structured data from the **Current Period (Period 2)**.

**Country:** {country}
**Risk Type:** {risk_type}

**Period 1 (Previous):** Sparse indicators extracted from the last narrative. This list may be incomplete.
```json
{previous_indicators_json}
```

**Period 2 (Current):** Full structured event data from new sources for the past 2 months.
```json
{current_events_json}
```

**CRITICAL INSTRUCTIONS:** Your analysis must be holistic and account for incomplete past data.

1. **Compare Like-for-Like (Comparable Events):** For each indicator in previous_indicators, find its matching event(s) in current_events and describe the change (e.g., "Food inflation increased from 10% to 15%."). These are your key_changes.

2. **Identify New Developments (Non-Comparable Events):** Identify all significant events in current_events that have no equivalent in previous_indicators. These are new developments, not necessarily an escalation (e.g., "New displacement of 5,000 people reported" or "First-time report of blockade in City Y."). List the most important ones in significant_developments.

3. **Holistic Trajectory Assessment:** Based only on the comparison and new developments, determine the overall humanitarian risk trajectory:
   - **increasing**: The situation has clearly deteriorated.
   - **decreasing**: The situation has clearly improved.
   - **stable**: No significant change in the overall risk level.
   - **materializing**: A new risk is emerging or a dormant one is becoming active.

4. **Identify Drivers:** List the 2-3 main factors from Period 2 that are driving this trend (e.g., "New conflict events," "Rising fuel prices," "Currency volatility"). These are your outlook_factors.

5. **Return Output:** You MUST return a single, valid JSON object matching the TrendAnalysis schema.

**TrendAnalysis Output Schema:**
{{
    "trajectory": "one of: increasing, decreasing, stable, materializing",
    "key_changes": ["list of strings describing comparable changes"],
    "quantitative_changes": {{"indicator_name": {{"from": "value", "to": "value"}}}},
    "significant_developments": ["list of strings describing new events"],
    "outlook_factors": ["list of 2-3 main driving factors"]
}}
"""


def run_trend_analysis_agent(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function to run the hybrid Trend Analysis.

    (Corrected version using manual JSON parsing.)
    """

    print(f"--- (5) Running Trend Analysis Agent for {state['country']} ---")

    current_events = state.get("events")
    previous_warning = state.get("previous_warning")

    if not current_events:
        print("   > No current events found to analyze. Skipping.")
        state["current_step"] = "TrendAnalysisComplete"
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append("TrendAnalysis skipped: No events extracted.")
        return state

    if not previous_warning:
        print("   > No previous warning text found to compare against. Skipping.")
        state["current_step"] = "TrendAnalysisComplete"
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append("TrendAnalysis skipped: No previous warning text.")
        return state

    try:
        # -------------------------------------------------
        # STEP 1: Extract key indicators from previous warning
        # -------------------------------------------------
        print("   > Step 1: Extracting indicators from *previous* warning text...")
        extract_prompt = ChatPromptTemplate.from_template(EXTRACT_PREVIOUS_PROMPT)
        extract_chain = extract_prompt | llm

        extract_response = extract_chain.invoke(
            {
                "previous_warning": previous_warning,
                "risk_type": ", ".join(state["risk_type"]),
            }
        )
        extract_raw_output = extract_response.content

        try:
            # Clean potential markdown
            if extract_raw_output.strip().startswith("```json"):
                extract_raw_output = extract_raw_output.strip()[7:-3].strip()
            elif extract_raw_output.strip().startswith("```"):
                extract_raw_output = extract_raw_output.strip()[3:-3].strip()

            previous_indicators_data: Dict[str, Any] = json.loads(extract_raw_output)
            if "indicators" not in previous_indicators_data or not isinstance(
                previous_indicators_data["indicators"], list
            ):
                raise ValueError("Missing 'indicators' key or it's not a list")
        except (json.JSONDecodeError, ValueError) as parse_error:
            print(
                "   ! ERROR: Failed to parse JSON for previous indicators: "
                f"{parse_error}",
            )
            print(f"   Raw LLM Output:\n{extract_raw_output}")
            raise ValueError(
                "LLM did not return valid JSON for PreviousIndicators. "
                f"Raw output: {extract_raw_output}"
            ) from parse_error

        print(
            "   > Extracted "
            f"{len(previous_indicators_data.get('indicators', []))} key indicators from past.",
        )

        # -------------------------------------------------
        # STEP 2: Compare previous indicators to current events
        # -------------------------------------------------
        print(
            "   > Step 2: Comparing previous indicators vs. current events for trend...",
        )
        compare_prompt = ChatPromptTemplate.from_template(COMPARE_TRENDS_PROMPT)
        compare_chain = compare_prompt | llm

        previous_json = json.dumps(previous_indicators_data, indent=2)
        current_json = json.dumps(current_events, indent=2)

        compare_response = compare_chain.invoke(
            {
                "country": state["country"],
                "risk_type": ", ".join(state["risk_type"]),
                "previous_indicators_json": previous_json,
                "current_events_json": current_json,
            }
        )
        compare_raw_output = compare_response.content

        try:
            # Clean potential markdown
            if compare_raw_output.strip().startswith("```json"):
                compare_raw_output = compare_raw_output.strip()[7:-3].strip()
            elif compare_raw_output.strip().startswith("```"):
                compare_raw_output = compare_raw_output.strip()[3:-3].strip()

            trend_analysis_output: TrendAnalysis = json.loads(compare_raw_output)
            # Basic validation
            if "trajectory" not in trend_analysis_output:
                raise ValueError("Missing 'trajectory' key")
        except (json.JSONDecodeError, ValueError) as parse_error:
            print(
                "   ! ERROR: Failed to parse JSON for trend analysis: "
                f"{parse_error}",
            )
            print(f"   Raw LLM Output:\n{compare_raw_output}")
            raise ValueError(
                "LLM did not return valid JSON for TrendAnalysis. "
                f"Raw output: {compare_raw_output}"
            ) from parse_error

        print(
            "   > Holistic Trend Assessment: "
            f"{trend_analysis_output.get('trajectory', 'UNKNOWN').upper()}",
        )

        # -------------------------------------------------
        # STEP 3: Update State
        # -------------------------------------------------
        state["trend_analysis"] = trend_analysis_output
        state["current_step"] = "TrendAnalysisComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Trend Analysis Agent: {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"TrendAnalysisError: {str(e)}")

    return state
