"""Agent 8: Citation Management Agent.

Compiles the final, annotated source list (bibliography) with reliability weighting.

This agent runs *after* the Narrative-Skeptic loop is complete.

Reads from:
- state["narrative_paragraph_1"] (to find *used* event IDs)
- state["events"] (to map event IDs to doc IDs)
- state["documents"] (to get metadata for the doc IDs)

Writes to:
- state["citations"]
"""

import re
from typing import List, Set

from state import ActiveWarningsState, Citation, Document, Event


def _get_reliability_score(doc: Document) -> float:
    """Assigns a reliability score based on the source type.

    Follows the design document: UN/INGO reports highest, then
    international media, then other/local media.
    """

    source_name = doc.get("source", "").lower()

    if "reliefweb" in source_name:
        return 1.0  # UN/INGO reports
    if "seerist" in source_name:
        return 0.95  # Professional analyst reports

    # Default for other/local media
    return 0.75


def run_citation_manager(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function to compile the final annotated bibliography."""

    print("--- (8) Running Citation Management Agent ---")

    try:
        narrative_p1 = state.get("narrative_paragraph_1", "")
        events = state.get("events", [])
        documents = state.get("documents", [])

        if not narrative_p1 or not events or not documents:
            print(
                "   > No narrative, events, or documents. "
                "Skipping citation generation.",
            )
            state["citations"] = []
            return state

        # Create fast lookups
        event_map = {evt["event_id"]: evt for evt in events}
        doc_map = {doc["doc_id"]: doc for doc in documents}

        # 1. Find all *event_ids* cited in the final narrative
        cited_event_ids: Set[str] = set()
        # Find all patterns like [Source: evt_123] or [Source: evt_123, evt_456]
        matches = re.findall(r"\[Source: ([\w, _-]+)\]", narrative_p1)
        for match in matches:
            ids = [evt_id.strip() for evt_id in match.split(",")]
            cited_event_ids.update(ids)

        # 2. Find all *doc_ids* that these events are sourced from
        cited_doc_ids: Set[str] = set()
        for event_id in cited_event_ids:
            event = event_map.get(event_id)
            if event:
                cited_doc_ids.update(event.get("source_ids", []))

        # 3. Build the final Citation list from the unique doc_ids
        final_citations: List[Citation] = []
        for doc_id in cited_doc_ids:
            doc = doc_map.get(doc_id)
            if not doc:
                continue

            metadata = doc.get("metadata", {})
            orig_lang = metadata.get("original_language", doc.get("language"))
            trans_model = metadata.get("translation_model")

            citation = Citation(
                source_id=doc["doc_id"],
                title=doc.get("title", "No Title"),
                url=doc.get("url", ""),
                translation_url=None,
                reliability=_get_reliability_score(doc),
                language=orig_lang,
                translation_method=trans_model,
                summary=doc.get("title", ""),
                supports_claims=[],
            )
            final_citations.append(citation)

        print(
            f"   > Generated {len(final_citations)} citations from "
            f"{len(cited_doc_ids)} unique documents.",
        )
        state["citations"] = final_citations
        state["current_step"] = "CitationComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Citation Manager: {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"CitationManagerError: {str(e)}")

    return state
