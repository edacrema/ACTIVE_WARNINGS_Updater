"""
Active Warnings Agent - State Definition
Defines the state structure that flows through the LangGraph pipeline
"""

from typing import TypedDict, List, Dict, Optional, Literal
from datetime import datetime

# ===== Type Definitions for State =====

class SearchQuery(TypedDict):
    """Individual search query with metadata"""
    query: str
    source_type: Literal["news", "un_reports", "economic", "climate"]
    data_source: str  # e.g., "GDELT", "ReliefWeb", "IMF"
    priority: Literal["high", "medium", "low"]


class SearchPlan(TypedDict):
    """Output from Query Planning Agent"""
    queries: List[SearchQuery]
    key_themes: List[str]
    key_actors: List[str]
    rationale: str


class Document(TypedDict):
    """Retrieved document with metadata"""
    doc_id: str
    title: str
    url: str
    source: str
    date: str
    language: str
    content: str
    translated: bool
    translation_confidence: Optional[float]
    relevance_score: float
    metadata: Optional[Dict[str, any]]


class Event(TypedDict):
    """Structured event extracted from documents"""
    event_id: str
    country: str
    driver: Literal["conflict", "economic", "climate"]
    event_type: str  # e.g., "Fatalities", "Food inflation"
    date_start: str
    date_end: Optional[str]
    actors: List[str]
    locations: List[Dict[str, any]]
    figures: List[Dict[str, any]]  # e.g., {"type": "fatalities", "value": 10}
    statement: str
    source_ids: List[str]
    relevance: float
    certainty: float
    novelty: Literal["new", "continuation", "escalation"]


class TrendAnalysis(TypedDict):
    """Output from Trend Analysis Agent"""
    trajectory: Literal["increasing", "decreasing", "stable", "materializing"]
    key_changes: List[str]
    quantitative_changes: Dict[str, any]
    significant_developments: List[str]
    outlook_factors: List[str]


class SkepticFlag(TypedDict):
    """Issues flagged by Skeptic Agent"""
    claim: str
    issue_type: Literal["numeracy", "contradiction", "source_mismatch", "hedging", "temporal"]
    severity: Literal["high", "medium", "low"]
    details: str
    conflicting_source: Optional[str]
    recommendation: str


class Citation(TypedDict):
    """Source citation with reliability metadata"""
    source_id: str
    title: str
    url: str
    translation_url: Optional[str]
    reliability: float  # Based on source type
    language: str
    translation_method: Optional[str]
    summary: str
    supports_claims: List[str]


# --- NEW TYPEDICT FOR STATUS AGENT ---
class SeriousnessScores(TypedDict):
    """A container for Likelihood and Impact scores (1-5)."""
    likelihood: int
    impact: int
    rationale: str  # Rationale for these specific scores


class StatusRecommendation(TypedDict):
    """Output from Status Recommendation Agent"""
    previous_seriousness: SeriousnessScores
    current_seriousness: SeriousnessScores
    status_change: Literal["Increased", "Decreased", "Remains", "Closed", "Reactivated"]
    rationale: str  # The final rationale for the status change


# ===== Main State Object =====

class ActiveWarningsState(TypedDict):
    """
    Main state object that flows through the LangGraph pipeline.
    Each agent reads from and writes to this state.
    """

    # ===== INPUT FIELDS (Set at initialization) =====
    country: str
    risk_type: List[Literal["conflict", "economic", "natural hazard"]]
    risk_title: str
    previous_warning: str
    # --- NEW INPUT FIELD ---
    previous_seriousness_scores: Optional[SeriousnessScores]  # e.g., {"likelihood": 4, "impact": 3, "rationale": "Previous analyst score"}

    predefined_queries: List[str]
    preferred_domains: List[str]
    update_period_start: str  # ISO format date
    update_period_end: str  # ISO format date

    # ===== AGENT OUTPUTS (Populated during workflow) =====

    # 1. Query Planning Agent
    search_plan: Optional[SearchPlan]

    # 2. Data Gathering Agent
    documents: Optional[List[Document]]

    # 3. Translation & Processing Agent
    # (modifies documents in-place, no separate field)

    # 4. Event Extraction Agent
    events: Optional[List[Event]]

    # 5. Trend Analysis Agent
    trend_analysis: Optional[TrendAnalysis]

    # 6. Skeptic Agent (runs after draft)
    skeptic_flags: Optional[List[SkepticFlag]]

    # 7. Narrative Synthesis Agent
    narrative_paragraph_1: Optional[str]
    narrative_paragraph_2: Optional[str]

    # 8. Citation Management Agent
    citations: Optional[List[Citation]]

    # 9. Status Recommendation Agent
    status_recommendation: Optional[StatusRecommendation]

    # ===== CONTROL & METADATA =====
    current_step: str
    error: Optional[str]
    warnings: List[str]
    run_id: str
    timestamp: str
    correction_attempts: int
