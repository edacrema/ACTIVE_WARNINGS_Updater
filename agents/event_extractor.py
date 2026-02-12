"""
Agent 4: Event Extraction Agent (Map-Reduce)
Extracts structured, traceable events from all documents.

Implements a two-stage Map-Reduce process:
1.  **Map:** An LLM call per document to extract raw "facts".
2.  **Reduce:** A single LLM call to synthesize all facts into a
    clean, deduplicated list of structured Event objects.

Reads from:
- state["documents"]
- state["country"]
- state["risk_type"]

Writes to:
- state["events"]
"""

import json
import uuid
from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate

from state import ActiveWarningsState, Event
from llm import llm


class FactList(TypedDict):
    """Schema for the 'Map' step output."""

    facts: List[str]


class EventList(TypedDict):
    """Schema for the 'Reduce' step output."""

    events: List[Event]


# ===== 1. "MAP" PROMPT (Extracts raw facts from *one* doc) =====

MAP_PROMPT_TEMPLATE = """
You are a data extraction assistant. Your task is to read a single document and extract ALL key facts, figures, dates, and quoted statements relevant to humanitarian risks (conflict, economic, climate).

**Document ID:** {doc_id}

**Document Content:**
---
{document_content}
---

**Instructions:**
1.  Read the text and identify all facts matching the risk ontology.
2.  For *every* fact you extract, you MUST append the document ID as a tag, like this: `(doc_id: {doc_id})`.
3.  Return a simple JSON list of fact strings.
4.  If no relevant facts are found, return `{{"facts": []}}`.

**Example Output:**
{{
    "facts": [
        "Food inflation in City Y reached 15% in October. (doc_id: {doc_id})",
        "Clashes between Group A and Group B displaced 5,000 people. (doc_id: {doc_id})",
        "The Central Bank revised GDP growth down to 1.5%. (doc_id: {doc_id})"
    ]
}}
"""


# ===== 2. "REDUCE" PROMPT (Synthesizes *all* facts into Events) =====

REDUCE_PROMPT_TEMPLATE = """
You are a senior humanitarian analyst and data scientist. Your task is to read a large BATCH of raw facts extracted from multiple documents.
You must synthesize these facts into a single, clean, and deduplicated list of structured `Event` objects.

**Country of Focus:** {country}
**Primary Risk Type:** {risk_type}

**Event Ontology to Follow:**
- **Conflict Indicators:**
    - Fatalities (civilian, combatant)
    - Number of armed clashes/attacks
    - Blockades/sieges and humanitarian access incidents
    - Displacement events (new IDPs, returnees)
    - Territory changes
- **Economic Indicators:**
    - Headline and food inflation (%, YoY)
    - Currency exchange rate and volatility
    - Fuel prices and policy changes
    - GDP growth revisions
    - Market functionality index
- **Natural Hazard Indicators:**
    - Rainfall anomaly (% vs. historical average)
    - Flood extent (people affected, area)
    - Crop condition index and harvest projections
    - Temperature anomaly
    - Water reservoir levels

**Batch of Raw Facts (with source IDs):**
---
{facts_list}
---

**CRITICAL INSTRUCTIONS:**
1.  **Read All Facts:** Process the entire list of facts.
2.  **Deduplicate & Synthesize:** Identify *unique* events. If multiple facts report the same event (e.g., the same inflation figure from different sources), merge them into ONE event.
3.  **Aggregate Sources:** When merging, you MUST aggregate all `(doc_id: ...)` tags into the final `source_ids` list for the event. This is critical for traceability.
4.  **Adhere to Schema:** Fill in all fields for each event (`driver`, `event_type`, `date_start`, `actors`, `locations`, `figures`, `statement`, `certainty`, `novelty`).
5.  **`event_id`:** Generate a unique ID for each event (e.g., `evt_uuid_...`).
6.  **`statement`:** Write a 1-sentence summary of the event's key claim.
7.  **`figures`:** Extract numerical data precisely (e.g., `{{"type": "food inflation", "value": 15, "unit": "%"}}`).
8.  **Output:** Return *only* a valid JSON object matching the `EventList` schema: `{{"events": [Event, ...]}}`. If no events are synthesized, return `{{"events": []}}`.
"""


def run_event_extractor(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function to run the Map-Reduce Event Extraction.

    (Corrected version using manual JSON parsing and corrected prompts.)
    """

    print(f"--- (4) Running Event Extractor (Map-Reduce) for {state['country']} ---")

    if state.get("documents") is None or not state["documents"]:
        print("   > No documents found to extract events from. Skipping.")
        state["current_step"] = "EventExtractionComplete"
        return state

    if state.get("warnings") is None:
        state["warnings"] = []

    all_facts: List[str] = []

    try:
        # -------------------------------------------------
        # STEP 1: MAP (Extract facts from each document)
        # -------------------------------------------------
        print("   > Step 1 (Map): Extracting raw facts from documents...")
        map_prompt = ChatPromptTemplate.from_template(MAP_PROMPT_TEMPLATE)
        map_chain = map_prompt | llm

        doc_count = 0
        for doc in state["documents"]:
            if not doc.get("content"):
                continue  # Skip docs with no content

            doc_count += 1
            try:
                map_response = map_chain.invoke(
                    {
                        "doc_id": doc["doc_id"],
                        "document_content": doc["content"],
                    }
                )
                map_raw_output = map_response.content

                # --- Manual JSON Parsing ---
                try:
                    if map_raw_output.strip().startswith("```json"):
                        map_raw_output = map_raw_output.strip()[7:-3].strip()
                    elif map_raw_output.strip().startswith("```"):
                        map_raw_output = map_raw_output.strip()[3:-3].strip()

                    facts_result = json.loads(map_raw_output)
                    if "facts" not in facts_result or not isinstance(
                        facts_result["facts"], list
                    ):
                        raise ValueError("Missing 'facts' key or it's not a list")
                except (json.JSONDecodeError, ValueError) as parse_error:
                    print(
                        f"      ! WARN: Failed to parse JSON for doc {doc['doc_id']}: {parse_error}",
                    )
                    print(f"      Raw LLM Output:\n{map_raw_output}")
                    state["warnings"].append(
                        f"EventMapParseError (doc {doc['doc_id']}): {str(parse_error)}",
                    )
                    continue
                # --- END Manual Parsing ---

                all_facts.extend(facts_result["facts"])
            except Exception as e:  # noqa: BLE001
                print(
                    f"      ! WARN: Failed LLM invoke for doc {doc['doc_id']}: {e}",
                )
                state["warnings"].append(
                    f"EventMapInvokeError (doc {doc['doc_id']}): {str(e)}",
                )

        print(f"   > Mapped {len(all_facts)} facts from {doc_count} documents.")

        # -------------------------------------------------
        # STEP 2: REDUCE (Synthesize facts into Events)
        # -------------------------------------------------
        if not all_facts:
            print("   > No facts extracted. Skipping Reduce step.")
            state["events"] = []
            state["current_step"] = "EventExtractionComplete"
            return state

        print("   > Step 2 (Reduce): Synthesizing facts into unique events...")
        reduce_prompt = ChatPromptTemplate.from_template(REDUCE_PROMPT_TEMPLATE)
        reduce_chain = reduce_prompt | llm

        facts_string = "\n".join(all_facts)

        reduce_response = reduce_chain.invoke(
            {
                "facts_list": facts_string,
                "country": state["country"],
                "risk_type": ", ".join(state["risk_type"]),
            }
        )
        reduce_raw_output = reduce_response.content

        # --- Manual JSON Parsing ---
        try:
            if reduce_raw_output.strip().startswith("```json"):
                reduce_raw_output = reduce_raw_output.strip()[7:-3].strip()
            elif reduce_raw_output.strip().startswith("```"):
                reduce_raw_output = reduce_raw_output.strip()[3:-3].strip()

            synthesis_result = json.loads(reduce_raw_output)
            if "events" not in synthesis_result or not isinstance(
                synthesis_result["events"], list
            ):
                raise ValueError("Missing 'events' key or it's not a list")
        except (json.JSONDecodeError, ValueError) as parse_error:
            print(
                f"   ! ERROR: Failed to parse JSON in Reduce step: {parse_error}",
            )
            print(f"   Raw LLM Output:\n{reduce_raw_output}")
            raise ValueError(
                "LLM did not return valid JSON for EventList. "
                f"Raw output: {reduce_raw_output}"
            ) from parse_error
        # --- END Manual Parsing ---

        extracted_events = synthesis_result["events"]

        # -------------------------------------------------
        # STEP 3: Post-processing and State Update
        # -------------------------------------------------
        final_events: List[Event] = []
        for event in extracted_events:
            if isinstance(event, dict):
                event["country"] = state["country"]
                if not event.get("event_id"):
                    event["event_id"] = f"evt_{uuid.uuid4()}"
                final_events.append(event)  # type: ignore[list-item]
            else:
                print(
                    "   ! WARN: Item in 'events' list was not a dictionary: "
                    f"{event}",
                )

        state["events"] = final_events
        print(
            f"   > Reduce complete: Synthesized {len(final_events)} unique events.",
        )
        state["current_step"] = "EventExtractionComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Event Extractor: {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"EventExtractorError: {str(e)}")

    return state
