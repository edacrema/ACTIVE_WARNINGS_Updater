"""
Seerist Retriever Agent - Specialized Data Source Component
Fetches analyst reports from Seerist API for Active Warnings updates.

Seerist API Documentation: https://seerist.docs.apiary.io/#
Endpoint: /hyperionapi/v1/wod  (World of Data)
Auth: x-api-key header

Required: SEERIST_API_KEY environment variable (or .env file)

Required libraries:
pip install requests
"""

import re
import time
from datetime import datetime
from typing import List, Dict, Optional

import requests

from state import ActiveWarningsState
from config import SEERIST_API_KEY, SEERIST_BASE_URL, SEERIST_DEFAULT_PAGE_SIZE


class SeeristRetriever:
    """Specialized retriever for Seerist Analysis API.

    Designed to work with the ActiveWarningsState pipeline.

    Key Features:
    - Fetches curated analyst reports (not raw news)
    - Supports Lucene query syntax via the 'search' parameter
    - Geographic filtering via aoiId (ISO 2-letter country codes)
    - Topic-based filtering (travel, unrest, transportation, health,
      terrorism, conflict, disaster, crime)
    - Pagination support
    - No web scraping needed (content returned directly in API response)
    - Built-in rate limiting and retry logic

    Response schema (discovered via API testing):
    - properties.title: {"en": "..."} - dict keyed by language
    - properties.sanitizedBody: {"en": "..."} - plain text body
    - properties.body: {"en": "<html>..."} - HTML body
    - properties.sanitizedSummary: {"en": "..."} - plain text summary
    - properties.publishedDate: ISO 8601 string
    - properties.id: string ID (e.g. "1026908")
    - properties.risks: [{"name": "...", "id": "..."}]
    - properties.countries: [{"code": "BGR", "name": [...]}]
    - properties.regions: [{"code": "EURC", "name": "..."}]
    - properties.tags: [{"name": {"en": "..."}, "id": "...", "type": "..."}]
    """

    MAX_RETRIES = 3
    REQUEST_DELAY = 0.5  # Seconds between requests to avoid rate limiting

    # Map risk_type values to Seerist topic parameters
    RISK_TYPE_TO_TOPICS = {
        "conflict": ["conflict", "unrest", "terrorism", "crime"],
        "economic": ["unrest", "transportation"],
        "natural hazard": ["disaster", "health"],
    }

    # Country name -> ISO 2-letter code for Seerist aoiId parameter
    COUNTRY_CODES = {
        "Afghanistan": "AF",
        "Angola": "AO",
        "Bangladesh": "BD",
        "Benin": "BJ",
        "Bolivia": "BO",
        "Burkina Faso": "BF",
        "Burundi": "BI",
        "Cambodia": "KH",
        "Cameroon": "CM",
        "Central African Republic": "CF",
        "Chad": "TD",
        "Colombia": "CO",
        "Democratic Republic of Congo": "CD",
        "Cuba": "CU",
        "Ethiopia": "ET",
        "Guatemala": "GT",
        "Haiti": "HT",
        "Honduras": "HN",
        "Iraq": "IQ",
        "Kenya": "KE",
        "Lebanon": "LB",
        "Lesotho": "LS",
        "Madagascar": "MG",
        "Malawi": "MW",
        "Mali": "ML",
        "Mozambique": "MZ",
        "Myanmar": "MM",
        "Nepal": "NP",
        "Niger": "NE",
        "Nigeria": "NG",
        "Pakistan": "PK",
        "Palestine": "PS",
        "Somalia": "SO",
        "South Sudan": "SS",
        "Sudan": "SD",
        "Syria": "SY",
        "Uganda": "UG",
        "Venezuela": "VE",
        "Yemen": "YE",
        "Zimbabwe": "ZW",
    }

    def __init__(self, api_key: Optional[str] = None, verbose: bool = True) -> None:
        """Initialize the Seerist retriever.

        Args:
            api_key: Seerist API key. Falls back to SEERIST_API_KEY config.
            verbose: If True, print progress information.
        """
        if api_key is None:
            api_key = SEERIST_API_KEY
        if not api_key:
            raise ValueError(
                "Seerist API key not configured. "
                "Set the SEERIST_API_KEY environment variable."
            )
        self.api_key = api_key
        self.verbose = verbose
        self.base_url = SEERIST_BASE_URL
        self.last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        """Ensure minimum delay between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def _format_datetime(self, date_str: str) -> str:
        """Convert ISO date string to Seerist ISO 8601 format with Z suffix."""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def _get_country_code(self, country: str) -> Optional[str]:
        """Resolve country name to ISO 2-letter code for aoiId."""
        return self.COUNTRY_CODES.get(country, None)

    def _get_topics_for_risk_types(self, risk_types: List[str]) -> List[str]:
        """Map risk types to Seerist topic parameters."""
        topics = set()
        for rt in risk_types:
            mapped = self.RISK_TYPE_TO_TOPICS.get(rt.lower(), [])
            topics.update(mapped)
        return list(topics)

    @staticmethod
    def _strip_html(html_text: str) -> str:
        """Remove HTML tags from text content."""
        clean = re.sub(r"<[^>]+>", " ", html_text)
        # Collapse multiple whitespace
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()

    def _extract_text(self, field: dict) -> str:
        """Extract English text from a Seerist language-keyed dict.

        Seerist stores text fields as {"en": "content", "fr": "contenu", ...}.
        We prefer English, then fall back to the first available language.
        """
        if not field or not isinstance(field, dict):
            return str(field) if field else ""
        # Prefer English
        text = field.get("en", "")
        if not text:
            # Fall back to first available language
            for val in field.values():
                if val:
                    text = val
                    break
        return text

    def _map_feature_to_document(self, feature: dict, idx: int) -> Dict:
        """Map a single Seerist GeoJSON Feature to our Document format.

        Based on API response schema discovered via test_seerist_api.py:
        - title: {"en": "..."} -> language-keyed dict
        - sanitizedBody: {"en": "..."} -> plain text body (preferred)
        - body: {"en": "<html>..."} -> HTML body (fallback, needs stripping)
        - sanitizedSummary: {"en": "..."} -> plain text summary
        - publishedDate: ISO 8601 string
        - id: string ID
        - risks: [{"name": "political", "id": "..."}]
        - countries: [{"code": "BGR", ...}]
        - regions: [{"code": "EURC", "name": "..."}]
        - tags: [{"name": {"en": "..."}, ...}]
        """
        props = feature.get("properties", {})

        # Extract title
        title = self._extract_text(props.get("title", {}))

        # Extract content: prefer sanitizedBody (plain text), fallback to body (HTML)
        content = self._extract_text(props.get("sanitizedBody", {}))
        if not content:
            html_body = self._extract_text(props.get("body", {}))
            content = self._strip_html(html_body) if html_body else ""
        if not content:
            content = self._extract_text(props.get("sanitizedSummary", {}))

        # Extract date
        date = props.get("publishedDate", props.get("@timestamp", ""))

        # Extract ID
        seerist_id = str(props.get("id", idx))

        # Extract metadata
        risks = props.get("risks", [])
        risk_names = [r.get("name", "") for r in risks if isinstance(r, dict)]

        countries = props.get("countries", [])
        country_names = []
        for c in countries:
            if isinstance(c, dict):
                names = c.get("name", [])
                for n in names:
                    if isinstance(n, dict) and n.get("languageCode") == "en":
                        country_names.append(n.get("text", ""))

        regions = props.get("regions", [])
        region_names = [r.get("name", "") for r in regions if isinstance(r, dict)]

        tags = props.get("tags", [])
        tag_names = [
            self._extract_text(t.get("name", {}))
            for t in tags
            if isinstance(t, dict)
        ]

        # Seerist analysis reports don't have a severity score,
        # so we use position-based relevance (same approach as GDELT)
        # Higher index = lower relevance
        relevance = max(0.5, 1.0 - (idx * 0.05))

        # Detect language from the available text fields
        language = "en"
        title_dict = props.get("title", {})
        if isinstance(title_dict, dict) and "en" not in title_dict:
            # Content is in a non-English language
            available_langs = list(title_dict.keys())
            if available_langs:
                language = available_langs[0]

        return {
            "doc_id": f"seerist_{seerist_id}",
            "title": title,
            "url": "",  # Seerist analyst reports have no public URL
            "source": "Seerist",
            "date": date,
            "language": language,
            "content": content,
            "translated": False,
            "translation_confidence": None,
            "relevance_score": relevance,
            "metadata": {
                "seerist_id": seerist_id,
                "risks": risk_names,
                "countries": country_names,
                "regions": region_names,
                "tags": tag_names,
                "geometry": feature.get("geometry"),
            },
        }

    def fetch(
        self,
        search_query: str,
        start_date: str,
        end_date: str,
        country: Optional[str] = None,
        risk_types: Optional[List[str]] = None,
        max_records: int = 50,
    ) -> List[Dict]:
        """Fetch analyst reports from Seerist API.

        Args:
            search_query: Lucene syntax search string.
            start_date: Start date in ISO format (YYYY-MM-DD or full ISO 8601).
            end_date: End date in ISO format.
            country: Country name (resolved to aoiId).
            risk_types: List of risk types to map to topics.
            max_records: Max results per request.

        Returns:
            List of article dictionaries in standardized Document format.
        """
        self._enforce_rate_limit()

        if self.verbose:
            print(f"   > Seerist: Searching '{search_query}' ({start_date} to {end_date})")

        # Build request parameters
        params = {
            "sources": "analysis",
            "start": self._format_datetime(start_date),
            "end": self._format_datetime(end_date),
            "pageSize": min(max_records, SEERIST_DEFAULT_PAGE_SIZE),
            "pageOffset": 0,
            "sortDirection": "desc",
        }

        # Add search query if provided
        if search_query and search_query.strip():
            params["search"] = search_query.strip()

        # Add country filter
        if country:
            country_code = self._get_country_code(country)
            if country_code:
                params["aoiId"] = country_code
                if self.verbose:
                    print(f"   >   Country filter: {country} -> {country_code}")

        # Add topic filter from risk types
        if risk_types:
            topics = self._get_topics_for_risk_types(risk_types)
            if topics:
                params["topic"] = ",".join(topics)
                if self.verbose:
                    print(f"   >   Topics: {params['topic']}")

        headers = {"x-api-key": self.api_key}

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(
                    self.base_url, params=params, headers=headers, timeout=30,
                )
                response.raise_for_status()

                data = response.json()
                metadata = data.get("metadata", {})
                features = data.get("features", [])

                if self.verbose:
                    total = metadata.get("total", len(features))
                    print(
                        f"   > Seerist: Retrieved {len(features)} reports "
                        f"(total available: {total})",
                    )

                documents: List[Dict] = []
                for idx, feature in enumerate(features):
                    doc = self._map_feature_to_document(feature, idx)
                    if doc["content"]:  # Only include reports with content
                        documents.append(doc)

                return documents

            except requests.exceptions.RequestException as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    if self.verbose:
                        print(
                            f"   ! Seerist API: Request failed, retrying "
                            f"in {wait_time}s...",
                        )
                    time.sleep(wait_time)
                else:
                    if self.verbose:
                        print(
                            f"   ! Seerist API: Failed after "
                            f"{self.MAX_RETRIES} attempts: {e}",
                        )
                    return []
        return []

    def fetch_batch(
        self,
        queries: List[str],
        start_date: str,
        end_date: str,
        country: Optional[str] = None,
        risk_types: Optional[List[str]] = None,
        max_per_query: int = 20,
    ) -> List[Dict]:
        """Fetch reports for multiple queries, then deduplicate.

        Args:
            queries: List of Lucene search queries.
            start_date: Start date in ISO format.
            end_date: End date in ISO format.
            country: Country name for geographic filtering.
            risk_types: List of risk types for topic mapping.
            max_per_query: Max results per query.

        Returns:
            Deduplicated list of Document-like dicts across queries.
        """
        all_documents: List[Dict] = []
        seen_ids = set()

        for query in queries:
            documents = self.fetch(
                search_query=query,
                start_date=start_date,
                end_date=end_date,
                country=country,
                risk_types=risk_types,
                max_records=max_per_query,
            )

            for doc in documents:
                if doc["doc_id"] not in seen_ids:
                    seen_ids.add(doc["doc_id"])
                    all_documents.append(doc)

        if self.verbose:
            print(f"   > Seerist: Total unique reports: {len(all_documents)}")

        return all_documents


def run_seerist_retriever(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function to retrieve analyst reports from Seerist.

    This function:
    1. Extracts queries from the search plan.
    2. Filters for queries targeting Seerist/news sources.
    3. Fetches analyst reports within the update window.
    4. Appends results to ``state['documents']``.
    """
    print(
        f"--- (2a) Running Seerist Retriever for {state['country']} ---",
    )

    try:
        if not state.get("search_plan"):
            state["warnings"].append(
                "No search plan found, skipping Seerist retrieval",
            )
            return state

        search_plan = state["search_plan"]

        # Filter for Seerist/news queries from the search plan
        news_queries = [
            q
            for q in search_plan.get("queries", [])
            if q.get("source_type") == "news" or q.get("data_source") == "Seerist"
        ]

        if not news_queries:
            if state.get("warnings") is None:
                state["warnings"] = []
            state["warnings"].append("No Seerist queries in search plan")
            return state

        print(f"   > Processing {len(news_queries)} Seerist queries")

        retriever = SeeristRetriever(verbose=True)

        # Get base queries from the search plan
        base_queries = [q["query"] for q in news_queries]

        # Add fallback queries (Lucene syntax with AND operator)
        if base_queries:
            print("   > Adding broad fallback queries...")
            base_queries.append(f"{state['country']} AND economic")
            base_queries.append(f"{state['country']} AND political")

        # Note: preferred_domains is not used for Seerist.
        # Seerist returns curated analyst reports, not web articles.

        risk_types = state.get("risk_type", [])

        documents = retriever.fetch_batch(
            queries=base_queries,
            start_date=state["update_period_start"],
            end_date=state["update_period_end"],
            country=state["country"],
            risk_types=risk_types,
            max_per_query=20,
        )

        if state.get("documents") is None:
            state["documents"] = []

        state["documents"].extend(documents)

        print(f"   > Seerist contribution: {len(documents)} analyst reports")
        print(f"   > Total documents so far: {len(state['documents'])}")

        state["current_step"] = "SeeristRetrievalComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in Seerist Retriever: {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"SeeristRetrieverError: {str(e)}")

    return state
