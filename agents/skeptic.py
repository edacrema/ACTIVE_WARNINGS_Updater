"""
Agent 7: Skeptic (Red Team Quality Control) Agent
Challenges the draft narrative against the ground-truth event data.
"""

import json
from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate

from state import ActiveWarningsState, SkepticFlag
from llm import llm


class SkepticFlagList(TypedDict):
    """Schema for the Skeptic's output list."""

    flags: List[SkepticFlag]


SKEPTIC_PROMPT = """
You are a meticulous Skeptic Agent, a "Red Team" designed to find all errors in a draft humanitarian report.
Your job is to compare the `Draft Narrative` against the `Ground Truth` data (the JSON) and flag ALL errors.

**Ground Truth (Events for P1):**
---
{events_json}
---

**Ground Truth (Trend Analysis for P2):**
---
{trend_analysis_json}
---

**Draft Narrative (Paragraph 1 - "What Changed"):**
---
{draft_p1}
---

**Draft Narrative (Paragraph 2 - "Outlook"):**
---
{draft_p2}
---

**YOUR TASK (Perform all checks):**

1.  **Factual & Numerical Accuracy:**
    - For *every* claim in P1, find its cited source (e.g., `[Source: evt_123]`).
    - Look up `evt_123` in the `Ground Truth (Events)`.
    - Does the claim *exactly* match the data? (e.g., draft says "15%", data says "12%")
    - Flag any mismatch, exaggeration, or misstatement.

2.  **Citation Validation:**
    - Does *every* factual claim in P1 have a `[Source: ...]` citation? Flag any uncited claims.
    - Does the cited `event_id` actually exist in the `Ground Truth (Events)`? Flag any "hallucinated" citations.

3.  **Hedging Check (for P2):**
    - Does P2 use appropriate hedged forecast language (e.g., "likely", "could", "may", "risk of")?
    - Or does it make definitive, unhedged claims about the future (e.g., "This *will* happen")?
    - Flag any unhedged or overly confident future-tense claims.

4.  **Contradiction Check:**
    - Does any part of the draft contradict the ground truth data?

**OUTPUT FORMAT:**
You MUST return ONLY a valid JSON object matching the `SkepticFlagList` schema below. Do NOT add any text before or after the JSON object.
```json
{{
    "flags": [
        {{
            "claim": "The exact text snippet from the draft that is wrong.",
            "issue_type": "One of: 'numeracy', 'contradiction', 'source_mismatch', 'hedging', 'missing_citation'",
            "severity": "'high' (factual/numerical/citation errors) or 'medium' (hedging/style).",
            "details": "Explain *what* is wrong (e.g., 'Source evt_123 says 12%, not 15%').",
            "recommendation": "Tell the writer *exactly* what to do (e.g., 'Change 15% to 12% and verify source evt_123.')."
        }}
        // ... more flags if errors are found
    ]
}}
```
If no errors are found, return {{"flags": []}}.
"""


def run_skeptic(state: ActiveWarningsState) -> ActiveWarningsState:
    """Runs the Skeptic Agent to find errors in the draft.

    (Corrected version using manual JSON parsing.)
    """

    print("--- (7) Running Skeptic (Red Team) Agent ---")

    try:
        prompt = ChatPromptTemplate.from_template(SKEPTIC_PROMPT)
        chain = prompt | llm

        response_raw = chain.invoke(
            {
                "events_json": json.dumps(state.get("events", []), indent=2),
                "trend_analysis_json": json.dumps(
                    state.get("trend_analysis", {}),
                    indent=2,
                ),
                "draft_p1": state.get("narrative_paragraph_1", ""),
                "draft_p2": state.get("narrative_paragraph_2", ""),
            }
        )
        skeptic_raw_output = response_raw.content

        # --- IMPROVED JSON PARSING ---
        try:
            json_part = skeptic_raw_output.strip()

            # Clean potential markdown fences first
            if json_part.startswith("```json"):
                json_part = json_part[7:]
                if json_part.endswith("```"):
                    json_part = json_part[:-3]
            elif json_part.startswith("```"):
                json_part = json_part[3:]
                if json_part.endswith("```"):
                    json_part = json_part[:-3]

            json_part = json_part.strip()

            # Find the first '{' and the last '}' to isolate the JSON object
            start_index = json_part.find("{")
            end_index = json_part.rfind("}")

            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_to_parse = json_part[start_index : end_index + 1]
                response: SkepticFlagList = json.loads(json_to_parse)
            else:
                response = json.loads(json_part)

            # Basic validation
            if "flags" not in response or not isinstance(
                response.get("flags"),
                list,
            ):
                raise ValueError("Missing 'flags' key or it's not a list")
        except (json.JSONDecodeError, ValueError) as parse_error:
            print(
                "   ! ERROR: Failed to parse JSON from Skeptic: "
                f"{parse_error}",
            )
            print(f"   Raw LLM Output:\n{skeptic_raw_output}")
            state["skeptic_flags"] = []
            if state.get("warnings") is None:
                state["warnings"] = []
            state["warnings"].append(f"SkepticParseError: {str(parse_error)}")
            state["current_step"] = "SkepticCheckComplete"
            return state

        flags = response.get("flags", [])
        if flags:
            print(f"   > SKEPTIC FOUND {len(flags)} ERRORS. Rerouting for correction.")
            for flag in flags:
                if isinstance(flag, dict):
                    print(
                        "     - [SEV: "
                        f"{flag.get('severity', 'N/A')}] "
                        f"{flag.get('details', 'No details')}",
                    )
                else:
                    print(f"     - [SEV: UNKNOWN] Malformed flag: {flag}")
            state["skeptic_flags"] = flags
        else:
            print("   > Skeptic check passed. No errors found.")
            state["skeptic_flags"] = []

        state["current_step"] = "SkepticCheckComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Skeptic Agent (Invoke failed): {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"SkepticInvokeError: {str(e)}")
        state["skeptic_flags"] = []
        state["current_step"] = "SkepticCheckComplete"

    return state
