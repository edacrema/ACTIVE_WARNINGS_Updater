"""
Agent 6: Narrative Synthesis Agent
Generates the two-paragraph draft based on the "Two-Paragraph Contract".
This node is state-aware:
- If no skeptic flags exist, it generates a new draft.
- If skeptic flags exist, it *corrects* the previous draft.
"""

import json
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate

from state import ActiveWarningsState
from llm import llm


# ===== 1. PROMPT FOR PARAGRAPH 1 (WHAT CHANGED) =====

P1_PROMPT_TEMPLATE = """
You are a factual reporting agent. Your task is to write a single-paragraph summary (80-100 words) of the most significant recent developments, based *only* on the provided JSON event list.

**CRITICAL INSTRUCTIONS:**
1.  **Tense:** Use PAST TENSE only.
2.  **Content:** Lead with the most significant development. Include specific dates, locations, and figures.
3.  **Citations (MANDATORY):** For *every* factual claim, you MUST append an inline citation using the `event_id` from the JSON.
    Example: "...inflation reached 15% [Source: evt_123]"
    Example: "...5,000 people were displaced [Source: evt_456, evt_789]"
4.  **Word Count:** Stay within 80-100 words.
5.  **Format:** Return *only* the raw text of the paragraph.

**JSON Event Data:**
---
{events_json}
---
"""


# ===== 2. PROMPT FOR PARAGRAPH 2 (SO WHAT / OUTLOOK) =====

P2_PROMPT_TEMPLATE = """
You are a humanitarian risk analyst. Your task is to write a single-paragraph, forward-looking outlook (70-100 words) based *only* on the provided Trend Analysis JSON.

**CRITICAL INSTRUCTIONS:**
1.  **Tense:** Use HEDGED FORECAST LANGUAGE (e.g., "The risk of... is high," "This will likely lead to...," "Key factors to watch include...").
2.  **Content:** Focus on the implications for humanitarian needs and the key factors from the analysis.
3.  **Word Count:** Stay within 70-100 words.
4.  **Format:** Return *only* the raw text of the paragraph.

**JSON Trend Analysis Data:**
---
{trend_analysis_json}
---
"""


# ===== 3. PROMPT FOR RE-WRITING (THE "ReAct" STEP) =====

CORRECTION_PROMPT_TEMPLATE = """
You are a writing assistant. Your previous draft was reviewed by a Skeptic Agent and found to have critical errors.
Your task is to re-write the draft to fix *all* the flagged issues.

**Original Draft (Paragraph 1):**
{draft_p1}

**Original Draft (Paragraph 2):**
{draft_p2}

**Errors to Fix (from Skeptic):**
---
{skeptic_flags_json}
---

**Ground Truth Data (Use this to fix errors):**
Event List (for P1):
{events_json}

Trend Analysis (for P2):
{trend_analysis_json}
---

**CRITICAL INSTRUCTIONS:**
1.  Re-write *both* paragraphs to address all issues in the 'recommendation' field of the Skeptic flags.
2.  Ensure P1's claims *exactly* match the `events_json` data.
3.  Ensure P1's citations use the correct `event_id`.
4.  Ensure P2 uses proper hedged language.
5.  Adhere to the strict 80-100 (P1) and 70-100 (P2) word counts.

**Return ONLY a JSON object with two keys:**
{{"paragraph_1": "...", "paragraph_2": "..."}}
"""


def run_narrative_synthesis(state: ActiveWarningsState) -> ActiveWarningsState:
    """Runs the Narrative Synthesis, generating or correcting a draft.

    (Corrected version using manual JSON parsing for correction step.)
    """

    # Initialize correction attempts on the first run
    if "correction_attempts" not in state:
        state["correction_attempts"] = 0

    state["correction_attempts"] += 1
    print(
        f"--- (6) Running Narrative Synthesis (Attempt {state['correction_attempts']}) ---",
    )

    try:
        # Get ground truth data
        events_json = json.dumps(state.get("events", []), indent=2)
        trend_json = json.dumps(state.get("trend_analysis", {}), indent=2)

        # Check if this is a correction run
        skeptic_flags = state.get("skeptic_flags")

        if not skeptic_flags:
            # ----- FIRST RUN: Generate Draft -----
            print("   > Generating new draft...")

            # 1. Generate Paragraph 1
            p1_prompt = ChatPromptTemplate.from_template(P1_PROMPT_TEMPLATE)
            p1_chain = p1_prompt | llm
            p1_response = p1_chain.invoke({"events_json": events_json})
            state["narrative_paragraph_1"] = p1_response.content

            # 2. Generate Paragraph 2
            p2_prompt = ChatPromptTemplate.from_template(P2_PROMPT_TEMPLATE)
            p2_chain = p2_prompt | llm
            p2_response = p2_chain.invoke({"trend_analysis_json": trend_json})
            state["narrative_paragraph_2"] = p2_response.content

            print("   > Draft generated. Sending to Skeptic.")

        else:
            # ----- ReAct RUN: Correct Draft -----
            print(
                f"   > Correcting draft based on {len(skeptic_flags)} skeptic flags...",
            )

            correction_prompt = ChatPromptTemplate.from_template(
                CORRECTION_PROMPT_TEMPLATE,
            )

            class CorrectedDraft(TypedDict):
                paragraph_1: str
                paragraph_2: str

            correction_chain = correction_prompt | llm

            correction_response_raw = correction_chain.invoke(
                {
                    "draft_p1": state.get("narrative_paragraph_1", ""),
                    "draft_p2": state.get("narrative_paragraph_2", ""),
                    "skeptic_flags_json": json.dumps(skeptic_flags, indent=2),
                    "events_json": events_json,
                    "trend_analysis_json": trend_json,
                }
            )
            correction_raw_output = correction_response_raw.content

            try:
                # Clean potential markdown
                if correction_raw_output.strip().startswith("```json"):
                    correction_raw_output = correction_raw_output.strip()[7:-3].strip()
                elif correction_raw_output.strip().startswith("```"):
                    correction_raw_output = correction_raw_output.strip()[3:-3].strip()

                correction_response: CorrectedDraft = json.loads(
                    correction_raw_output,
                )
                # Basic validation
                if (
                    "paragraph_1" not in correction_response
                    or "paragraph_2" not in correction_response
                ):
                    raise ValueError(
                        "Missing 'paragraph_1' or 'paragraph_2' key",
                    )
            except (json.JSONDecodeError, ValueError) as parse_error:
                print(
                    "   ! ERROR: Failed to parse JSON for corrected draft: "
                    f"{parse_error}",
                )
                print(f"   Raw LLM Output:\n{correction_raw_output}")
                state.setdefault("warnings", []).append(
                    f"CorrectionParseError: {str(parse_error)}",
                )
                state["skeptic_flags"] = []  # Clear flags to prevent infinite loop
                return state

            # Overwrite the previous draft with the corrected version
            state["narrative_paragraph_1"] = correction_response["paragraph_1"]
            state["narrative_paragraph_2"] = correction_response["paragraph_2"]
            print("   > Draft corrected. Resending to Skeptic for verification.")

        # Clear flags for the next check (only if successful or first run)
        state["skeptic_flags"] = []
        state["current_step"] = "SynthesisComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Narrative Synthesis: {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"SynthesisError: {str(e)}")
        # If the LLM call itself fails, clear flags to avoid loop
        state["skeptic_flags"] = []

    return state
