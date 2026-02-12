"""
ReliefWeb Retriever Agent - Specialized Data Source Component
Fetches UN/INGO reports from ReliefWeb API for Active Warnings updates

ReliefWeb API Documentation: https://apidoc.reliefweb.int/
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Set

import requests

from state import ActiveWarningsState


class ReliefWebRetriever:
    """Specialized retriever for ReliefWeb API.

    Designed to work with the ActiveWarningsState pipeline.

    Key Features:
    - Searches curated UN/INGO reports from 4,000+ sources
    - Supports temporal filtering (critical for 2-month window)
    - Geographic filtering by country
    - Theme-based filtering (conflict, economic, climate)
    - Returns up to 1,000 reports per query
    - Built-in rate limiting and retry logic
    """

    BASE_URL = "https://api.reliefweb.int/v2/reports"
    MAX_RECORDS = 1000  # ReliefWeb API limit
    MAX_CALLS_PER_DAY = 1000  # API quota
    REQUEST_DELAY = 0.5  # Seconds between requests (conservative)
    MAX_RETRIES = 3

    # --- FIX: Added User-Agent to request headers ---
    # This is necessary to prevent 403 Forbidden errors
    REQUEST_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        ),
        "Content-Type": "application/json",
    }
    # --- END FIX ---

    # Theme mappings for different risk types
    # Full list: https://reliefweb.int/taxonomy-descriptions
    THEME_MAPPINGS = {
        "conflict": [
            "Protection",
            "Humanitarian Access",
            "Peacekeeping and Peacebuilding",
            "Mine Action",
            "Contributions",
        ],
        "economic": [
            "Food and Nutrition",
            "Logistics and Telecommunications",
            "Contributions",
            "Recovery and Reconstruction",
        ],
        "natural hazard": [
            "Climate Change and Environment",
            "Disaster Management",
            "Food and Nutrition",
        ],
        "climate": [
            "Climate Change and Environment",
            "Disaster Management",
            "Food and Nutrition",
            "Water Sanitation Hygiene",
        ],
    }

    # Country name normalization (ReliefWeb uses specific names)
    COUNTRY_NORMALIZATION = {
        "Democratic Republic of Congo": "Democratic Republic of the Congo",
        "DRC": "Democratic Republic of the Congo",
        "Congo DRC": "Democratic Republic of the Congo",
        "Palestine": "occupied Palestinian territory",
        "Venezuela": "Venezuela (Bolivarian Republic of)",
    }

    # --- MODIFIED __init__ METHOD ---
    def __init__(self, appname: Optional[str] = None, verbose: bool = True) -> None:
        """Initialize the ReliefWeb retriever.

        Args:
            appname: Application name for API tracking. If None,
                falls back to RELIEFWEB_APPNAME env var.
            verbose: If True, print progress information.
        """
        if appname is None:
            # Use the env var, with a final fallback
            appname = os.getenv("RELIEFWEB_APPNAME", "wfp-early-warnings")

        self.appname = appname
        self.verbose = verbose
        self.last_request_time = 0.0
        self.requests_today = 0

    # --- END MODIFICATION ---

    def _enforce_rate_limit(self) -> None:
        """Ensure minimum delay between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
        self.requests_today += 1

        if self.requests_today >= self.MAX_CALLS_PER_DAY:
            raise RuntimeError("Daily API quota exhausted (1000 calls)")

    def _normalize_country(self, country: str) -> str:
        """Normalize country name to ReliefWeb's conventions."""

        return self.COUNTRY_NORMALIZATION.get(country, country)

    def _format_datetime(self, date_str: str) -> str:
        """Convert ISO date string to ReliefWeb's ISO 8601 format."""

        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.strptime(date_str, "%Y-%m-%d")

        return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def _build_payload(
        self,
        country: str,
        start_date: str,
        end_date: str,
        keywords: Optional[str] = None,
        risk_type: Optional[str] = None,
        themes_list: Optional[List[str]] = None,  # <-- MODIFICATION: Added new param
        limit: int = 1000,
    ) -> Dict:
        """Build POST request payload for ReliefWeb API."""

        country_normalized = self._normalize_country(country)

        conditions = [
            {
                "field": "country",
                "value": country_normalized,
            },
            {
                "field": "date.created",
                "value": {
                    "from": self._format_datetime(start_date),
                    "to": self._format_datetime(end_date),
                },
            },
        ]

        # --- MODIFICATION: Prioritize themes_list if provided ---
        themes_to_use: List[str] = []
        if themes_list:
            themes_to_use = themes_list
        elif risk_type and risk_type.lower() in self.THEME_MAPPINGS:
            themes_to_use = self.THEME_MAPPINGS[risk_type.lower()]

        if themes_to_use:
            conditions.append(
                {
                    "field": "theme",
                    "value": themes_to_use,
                    "operator": "OR",
                }
            )
        # --- END MODIFICATION ---

        payload = {
            "preset": "latest",
            "profile": "list",
            "limit": min(limit, self.MAX_RECORDS),
            "filter": {
                "operator": "AND",
                "conditions": conditions,
            },
            "fields": {
                "include": [
                    "id",
                    "title",
                    "url",
                    "source.name",
                    "source.shortname",
                    "date.created",
                    "date.original",
                    "body",
                    "format.name",
                    "theme.name",
                    "disaster.name",
                    "language.name",
                ],
            },
        }

        if keywords:
            payload["query"] = {
                "value": keywords,
                "fields": ["title", "body"],
                "operator": "OR",
            }

        return payload

    def fetch(
        self,
        country: str,
        start_date: str,
        end_date: str,
        keywords: Optional[str] = None,
        risk_type: Optional[str] = None,
        themes_list: Optional[List[str]] = None,  # <-- MODIFICATION: Added new param
        max_records: int = 1000,
    ) -> List[Dict]:
        """Fetch reports from ReliefWeb matching the criteria."""

        self._enforce_rate_limit()

        if self.verbose:
            print(
                f"   > ReliefWeb: Searching '{country}' ({start_date} to {end_date})",
            )
            if keywords:
                print(f"   >   Keywords: {keywords}")
            if themes_list:
                print(f"   >   Themes: {', '.join(themes_list)}")
            elif risk_type:
                print(f"   >   Risk type: {risk_type}")

        payload = self._build_payload(
            country=country,
            start_date=start_date,
            end_date=end_date,
            keywords=keywords,
            risk_type=risk_type,
            themes_list=themes_list,  # <-- MODIFICATION: Pass param
            limit=max_records,
        )

        url = f"{self.BASE_URL}?appname={self.appname}"

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self.REQUEST_HEADERS,  # <-- FIX: Use headers with User-Agent
                    timeout=30,
                )
                response.raise_for_status()

                data = response.json()
                reports = data.get("data", [])
                total_count = data.get("totalCount", 0)

                if self.verbose:
                    print(
                        f"   > ReliefWeb: Retrieved {len(reports)} reports "
                        f"(total available: {total_count})",
                    )

                documents: List[Dict] = []
                for idx, report in enumerate(reports):
                    fields = report.get("fields", {})

                    source_info = (
                        fields.get("source", [{}])[0] if fields.get("source") else {}
                    )
                    source_name = source_info.get("name", "Unknown")

                    body = fields.get("body", "")
                    if len(body) > 10000:
                        body = body[:10000] + "... [truncated]"

                    themes = [t.get("name") for t in fields.get("theme", [])]
                    disasters = [d.get("name") for d in fields.get("disaster", [])]

                    doc = {
                        "doc_id": f"reliefweb_{report.get('id', idx)}",
                        "title": fields.get("title", ""),
                        "url": fields.get("url", ""),
                        "source": f"ReliefWeb - {source_name}",
                        "date": fields.get("date", {}).get("created", ""),
                        "language": fields.get("language", [{}])[0].get(
                            "name",
                            "English",
                        ),
                        "content": body,
                        "translated": False,
                        "translation_confidence": None,
                        "relevance_score": report.get("score", 1.0),
                        "metadata": {
                            # Populating the new state field
                            "format": fields.get("format", [{}])[0].get(
                                "name", ""
                            ),
                            "themes": themes,
                            "disasters": disasters,
                            "original_date": fields.get("date", {}).get(
                                "original", ""
                            ),
                        },
                    }
                    documents.append(doc)

                return documents

            except requests.exceptions.RequestException as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    if self.verbose:
                        print(
                            f"   ! ReliefWeb: Request failed, retrying in {wait_time}s...",
                        )
                    time.sleep(wait_time)
                else:
                    if self.verbose:
                        print(
                            f"   ! ReliefWeb: Failed after {self.MAX_RETRIES} attempts: {e}",
                        )
                    return []
        return []

    def fetch_batch(
        self,
        country: str,
        start_date: str,
        end_date: str,
        keyword_queries: List[str],
        risk_type: Optional[str] = None,
        themes_list: Optional[List[str]] = None,  # <-- MODIFICATION: Added new param
        max_per_query: int = 100,
    ) -> List[Dict]:
        """Fetch reports for multiple keyword queries and deduplicate results."""

        all_documents: List[Dict] = []
        seen_ids: Set[str] = set()

        for keywords in keyword_queries:
            documents = self.fetch(
                country=country,
                start_date=start_date,
                end_date=end_date,
                keywords=keywords,
                risk_type=risk_type,
                themes_list=themes_list,  # <-- MODIFICATION: Pass param
                max_records=max_per_query,
            )

            for doc in documents:
                if doc["doc_id"] not in seen_ids:
                    seen_ids.add(doc["doc_id"])
                    all_documents.append(doc)

        if self.verbose:
            print(
                f"   > ReliefWeb: Total unique reports: {len(all_documents)}",
            )

        return all_documents


def run_reliefweb_retriever(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function to retrieve reports from ReliefWeb.

    This function:
    1. Extracts queries from the search plan.
    2. Filters for queries targeting UN/INGO reports.
    3. Fetches reports within the 2-month update window.
    4. Appends results to state["documents"].
    """

    print(f"--- (2b) Running ReliefWeb Retriever for {state['country']} ---")

    try:
        if not state.get("search_plan"):
            state["warnings"].append(
                "No search plan found, skipping ReliefWeb retrieval",
            )
            return state

        search_plan = state["search_plan"]

        un_queries = [
            q
            for q in search_plan.get("queries", [])
            if q.get("source_type") == "un_reports"
            or q.get("data_source") == "ReliefWeb"
        ]

        print(
            f"   > Processing {len(un_queries)} ReliefWeb-specific queries",
        )
        if not un_queries:
            print(
                f"   > No specific queries, fetching all reports for {state['country']}",
            )

        retriever = ReliefWebRetriever(verbose=True)

        # --- MODIFICATION: Handle list of risk types ---
        risk_types = state.get("risk_type", [])
        all_themes: Set[str] = set()  # Use a set to avoid duplicates
        if risk_types:
            for rt in risk_types:
                themes = retriever.THEME_MAPPINGS.get(rt.lower(), [])
                all_themes.update(themes)

        risk_type_themes_list = list(all_themes)
        # --- END MODIFICATION ---

        if un_queries:
            keyword_queries = [q["query"] for q in un_queries]

            documents = retriever.fetch_batch(
                country=state["country"],
                start_date=state["update_period_start"],
                end_date=state["update_period_end"],
                keyword_queries=keyword_queries,
                risk_type=None,  # <-- MODIFICATION: Use themes list instead
                themes_list=risk_type_themes_list,  # <-- MODIFICATION: Pass list
                max_per_query=50,
            )
        else:
            documents = retriever.fetch(
                country=state["country"],
                start_date=state["update_period_start"],
                end_date=state["update_period_end"],
                risk_type=None,
                themes_list=risk_type_themes_list,  # <-- MODIFICATION: Pass list
                max_records=100,
            )

        if state.get("documents") is None:
            state["documents"] = []

        state["documents"].extend(documents)

        print(
            f"   > ReliefWeb contribution: {len(documents)} documents",
        )
        print(f"   > Total documents so far: {len(state['documents'])}")

        state["current_step"] = "ReliefWebRetrievalComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in ReliefWeb Retriever: {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"ReliefWebRetrieverError: {str(e)}")

    return state
