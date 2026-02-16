"""
Agent 4: Event Extraction Agent (Single-Pass)
Extracts structured, traceable events from all documents in a single LLM call.

Sends all documents (truncated) to the LLM at once, which extracts and
deduplicates structured Event objects directly. This replaces the previous
Map-Reduce approach (1 call per document) which was slow, memory-intensive,
and prone to silent JSON parsing failures.

Reads from:
- state["documents"]
- state["country"]
- state["risk_type"]

Writes to:
- state["events"]
"""

import json
import uuid
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI

from state import ActiveWarningsState, Event
from config import (
    PROJECT_ID,
    LOCATION,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_EXTRACTION_MAX_TOKENS,
)


# Maximum characters per document sent to the LLM.
# 33 docs * 3000 chars ≈ 100k chars ≈ 25k tokens — well within Gemini's 1M context.
MAX_DOC_CHARS = 3000


EXTRACTION_PROMPT_TEMPLATE = """
You are a senior humanitarian data analyst. Your task is to read ALL of the following documents about **{country}** and extract a deduplicated list of structured humanitarian events.

**Country of Focus:** {country}
**Primary Risk Types:** {risk_type}

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

**DOCUMENTS ({doc_count} total):**
---
{documents_block}
---

**CRITICAL INSTRUCTIONS:**
1.  **Read ALL documents** and identify every humanitarian event or development relevant to the risk types above.
2.  **Deduplicate:** If multiple documents report the same event (e.g., the same inflation figure, the same displacement event), merge them into ONE event entry.
3.  **source_ids (MANDATORY):** For every event, you MUST list ALL document IDs (from the headers above) that mention or support that event. This is critical for citation traceability.
4.  **Adhere to Schema:** Fill in all fields for each event:
    - `event_id`: Generate a short unique ID (e.g., `evt_001`, `evt_002`, etc.)
    - `driver`: One of "conflict", "economic", or "climate"
    - `event_type`: Specific type (e.g., "Fatalities", "Food inflation", "Displacement", "Currency depreciation")
    - `date_start`: Best available date (ISO format or descriptive like "January 2026")
    - `actors`: List of key actors involved (groups, institutions, etc.)
    - `locations`: List of location objects, e.g., [{{"name": "Kabul", "type": "city"}}]
    - `figures`: List of numerical data, e.g., [{{"type": "food inflation", "value": 15, "unit": "%"}}]
    - `statement`: A 1-sentence factual summary of the event
    - `source_ids`: List of document IDs that support this event (e.g., ["seerist_123", "reliefweb_456"])
    - `certainty`: Confidence level 0.0-1.0 based on source agreement
    - `novelty`: One of "new", "continuation", or "escalation"
5.  **Extract precise figures:** Percentages, counts, rates, dates — be as specific as the source data allows.
6.  **Output:** Return ONLY a valid JSON object: {{"events": [...]}}. No text before or after the JSON.
    If no events are found, return {{"events": []}}.
"""


def _extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output text.

    Handles: markdown fences, leading/trailing text, nested braces.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the outermost JSON object by matching braces
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    end = -1
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("Unbalanced braces in LLM output")

    return json.loads(cleaned[start : end + 1])


def _build_documents_block(documents: list) -> str:
    """Build the concatenated document text block for the prompt."""
    parts = []
    for doc in documents:
        content = doc.get("content", "")
        if not content:
            continue

        # Truncate long documents
        if len(content) > MAX_DOC_CHARS:
            content = content[:MAX_DOC_CHARS] + "... [truncated]"

        source = doc.get("source", "Unknown")
        date = doc.get("date", "Unknown")
        title = doc.get("title", "")

        header = f"=== Document {doc['doc_id']} (Source: {source}, Date: {date}) ==="
        if title:
            header += f"\nTitle: {title}"

        parts.append(f"{header}\n{content}\n=== END {doc['doc_id']} ===")

    return "\n\n".join(parts)


def run_event_extractor(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function: single-pass event extraction from all documents."""

    print(f"--- (4) Running Event Extractor (Single-Pass) for {state['country']} ---")

    if state.get("documents") is None or not state["documents"]:
        print("   > No documents found to extract events from. Skipping.")
        state["events"] = []
        state["current_step"] = "EventExtractionComplete"
        return state

    if state.get("warnings") is None:
        state["warnings"] = []

    try:
        # Filter documents with actual content
        docs_with_content = [d for d in state["documents"] if d.get("content")]
        print(f"   > Processing {len(docs_with_content)} documents with content...")

        if not docs_with_content:
            print("   > All documents have empty content. Skipping extraction.")
            state["events"] = []
            state["current_step"] = "EventExtractionComplete"
            return state

        # Build the concatenated document block
        documents_block = _build_documents_block(docs_with_content)
        print(
            f"   > Document block size: {len(documents_block):,} chars "
            f"(~{len(documents_block) // 4:,} tokens)",
        )

        # Use a dedicated LLM with higher max_tokens for extraction
        extraction_llm = ChatVertexAI(
            model=LLM_MODEL,
            project=PROJECT_ID,
            location=LOCATION,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_EXTRACTION_MAX_TOKENS,
        )

        prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT_TEMPLATE)
        chain = prompt | extraction_llm

        print("   > Calling LLM for single-pass event extraction...")
        response = chain.invoke(
            {
                "country": state["country"],
                "risk_type": ", ".join(state["risk_type"]),
                "doc_count": len(docs_with_content),
                "documents_block": documents_block,
            }
        )
        raw_output = response.content
        print(f"   > LLM response received ({len(raw_output):,} chars)")

        # Parse JSON with robust extractor
        try:
            result = _extract_json(raw_output)
            if "events" not in result or not isinstance(result["events"], list):
                raise ValueError("Missing 'events' key or it's not a list")
        except (json.JSONDecodeError, ValueError) as parse_error:
            print(f"   ! ERROR: Failed to parse JSON from LLM: {parse_error}")
            print(f"   Raw LLM Output (first 500 chars):\n{raw_output[:500]}")
            state["warnings"].append(
                f"EventExtractorParseError: {str(parse_error)}"
            )
            state["events"] = []
            state["current_step"] = "EventExtractionComplete"
            return state

        # Post-process events
        final_events: List[Event] = []
        for event in result["events"]:
            if isinstance(event, dict):
                event["country"] = state["country"]
                if not event.get("event_id"):
                    event["event_id"] = f"evt_{uuid.uuid4()}"
                final_events.append(event)  # type: ignore[list-item]
            else:
                print(
                    f"   ! WARN: Item in 'events' list was not a dict: {event}",
                )

        state["events"] = final_events
        print(f"   > Extracted {len(final_events)} unique events.")
        state["current_step"] = "EventExtractionComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Event Extractor: {e}")
        state["warnings"].append(f"EventExtractorError: {str(e)}")
        state["events"] = []

    return state
