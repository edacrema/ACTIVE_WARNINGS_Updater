"""
GDELT Retriever Agent - Specialized Data Source Component
Fetches news articles from GDELT DOC 2.0 API for Active Warnings updates
and scrapes the full text content from the source URL.

GDELT API Documentation: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

Required libraries:
pip install requests beautifulsoup4 lxml
"""

import requests
import time
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import quote

from bs4 import BeautifulSoup

from state import ActiveWarningsState


class GDELTRetriever:
    """Specialized retriever for GDELT DOC 2.0 API.

    Designed to work with the ActiveWarningsState pipeline.

    Key Features:
    - Searches last 3 months of global news coverage
    - Supports 65 machine-translated languages
    - Returns up to 250 articles per query
    - Handles date range filtering (critical for 2-month window)
    - **Includes an integrated web scraper to fetch full article text**
    - Built-in rate limiting and retry logic
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    MAX_RECORDS = 250  # GDELT API limit per request
    REQUEST_DELAY = 1.0  # Seconds between requests to avoid rate limiting
    MAX_RETRIES = 3

    # Headers for scraping to mimic a real browser
    SCRAPE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    # Country code mapping (same as your draft)
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

    def __init__(self, verbose: bool = True) -> None:
        """Initialize the GDELT retriever.

        Args:
            verbose: If True, print progress information.
        """
        self.verbose = verbose
        self.last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        """Ensure minimum delay between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()

    def _format_datetime(self, date_str: str) -> str:
        """Convert ISO date string to GDELT's YYYYMMDDHHMMSS format."""
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y%m%d%H%M%S")

    def _build_query_string(
        self,
        keywords: str,
        country: Optional[str] = None,
        language: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> str:
        """Build GDELT query string with proper operators.

        (Note: This helper is from your draft, though not currently used by
        fetch() below.)
        """
        query_parts = [keywords]
        if country:
            country_code = self.COUNTRY_CODES.get(country, country)
            query_parts.append(f"sourcecountry:{country_code}")
        if language:
            lang_map = {
                "English": "eng",
                "Spanish": "spa",
                "French": "fra",
                "Arabic": "ara",
                "Portuguese": "por",
                "Russian": "rus",
            }
            lang_code = lang_map.get(language, language.lower()[:3])
            query_parts.append(f"sourcelang:{lang_code}")
        if domain:
            query_parts.append(f"domain:{domain}")
        return " ".join(query_parts)

    def _scrape_article_content(self, url: str) -> str:
        """Scrape the full text content from a given URL.

        This is a basic scraper; it may fail on sites with heavy JavaScript,
        paywalls, or complex anti-scraping measures.

        Args:
            url: The URL of the article to scrape.

        Returns:
            The extracted text content, or an empty string if scraping fails.
        """
        if not url:
            return ""

        self._enforce_rate_limit()  # Also rate-limit scraping requests

        if self.verbose:
            print(f"      ... Scraping content from: {url[:70]}...")

        try:
            response = requests.get(url, headers=self.SCRAPE_HEADERS, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            paragraphs = soup.find_all("p")

            if not paragraphs:
                # Fallback: get all text, strip whitespace
                text = " ".join(soup.stripped_strings)
            else:
                text = " ".join(p.get_text() for p in paragraphs)

            if self.verbose and not text:
                print(f"      ! WARN: No content found at {url}")

            return text.strip()

        except requests.exceptions.RequestException as e:
            if self.verbose:
                print(f"      ! ERROR: Failed to scrape {url}. Reason: {e}")
            return ""
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"      ! ERROR: Failed to parse {url}. Reason: {e}")
            return ""

    def fetch(
        self,
        query: str,
        start_date: str,
        end_date: str,
        country: Optional[str] = None,
        max_records: int = 250,
        sort_by: str = "relevance",
    ) -> List[Dict]:
        """Fetch articles from GDELT and scrape their content.

        Args:
            query: Search query (can include GDELT operators).
            start_date: Start date in ISO format (YYYY-MM-DD).
            end_date: End date in ISO format (YYYY-MM-DD).
            country: Optional country filter (no longer used for sourcecountry).
            max_records: Max articles to retrieve (default: 250, max: 250).
            sort_by: Sort order - "relevance" (default), "date_desc", or
                "date_asc".

        Returns:
            List of article dictionaries in standardized Document format,
            now including the scraped ``content``.
        """
        self._enforce_rate_limit()

        if self.verbose:
            print(f"   > GDELT: Searching '{query}' ({start_date} to {end_date})")

        start_dt = self._format_datetime(start_date)
        end_dt = self._format_datetime(end_date)

        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": min(max_records, self.MAX_RECORDS),
            "format": "json",
            "startdatetime": start_dt,
            "enddatetime": end_dt,
        }

        if sort_by == "date_desc":
            params["sort"] = "datedesc"
        elif sort_by == "date_asc":
            params["sort"] = "dateasc"

        for attempt in range(self.MAX_RETRIES):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()
                articles = data.get("articles", [])

                if self.verbose:
                    print(
                        f"   > GDELT: Retrieved {len(articles)} article metadata records. "
                        "Now scraping content...",
                    )

                documents: List[Dict] = []
                for idx, article in enumerate(articles):
                    article_url = article.get("url", "")

                    # Scrape content immediately
                    article_content = self._scrape_article_content(article_url)

                    doc = {
                        "doc_id": f"gdelt_{int(time.time())}_{idx}",
                        "title": article.get("title", ""),
                        "url": article_url,
                        "source": "GDELT",  # This is the API source
                        "date": article.get("seendate", ""),
                        "language": article.get("language", ""),
                        "content": article_content,
                        "translated": False,
                        "translation_confidence": None,
                        "relevance_score": 1.0 - (idx / max(len(articles), 1)),
                    }
                    documents.append(doc)

                return documents

            except requests.exceptions.RequestException as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait_time = (attempt + 1) * 2
                    if self.verbose:
                        print(
                            f"   ! GDELT API: Request failed, retrying in {wait_time}s...",
                        )
                    time.sleep(wait_time)
                else:
                    if self.verbose:
                        print(
                            f"   ! GDELT API: Failed after {self.MAX_RETRIES} attempts: {e}",
                        )
                    return []
        return []

    def fetch_batch(
        self,
        queries: List[str],
        start_date: str,
        end_date: str,
        country: Optional[str] = None,
        max_per_query: int = 10,
    ) -> List[Dict]:
        """Fetch articles for multiple queries, scrape content, and deduplicate.

        Args:
            queries: List of search queries.
            start_date: Start date in ISO format.
            end_date: End date in ISO format.
            country: Optional country filter.
            max_per_query: Max articles per query (to avoid hitting 250 limit).

        Returns:
            Deduplicated list of articles (Document-like dicts) across queries.
        """
        all_documents: List[Dict] = []
        seen_urls = set()

        for query in queries:
            documents = self.fetch(
                query=query,
                start_date=start_date,
                end_date=end_date,
                country=country,
                max_records=max_per_query,
                sort_by="date_desc",  # newest articles first
            )

            for doc in documents:
                if doc["url"] not in seen_urls:
                    seen_urls.add(doc["url"])
                    all_documents.append(doc)

        if self.verbose:
            print(f"   > GDELT: Total unique scraped articles: {len(all_documents)}")

        return all_documents


def run_gdelt_retriever(state: ActiveWarningsState) -> ActiveWarningsState:
    """LangGraph node function to retrieve articles from GDELT.

    This function:
    1. Extracts queries from the search plan.
    2. Filters for queries targeting GDELT/news sources.
    3. Fetches articles and scrapes their content within the update window.
    4. Appends results to ``state['documents']``.
    """
    print(
        f"--- (2a) Running GDELT Retriever (with Scraper) for {state['country']} ---",
    )

    try:
        if not state.get("search_plan"):
            state["warnings"].append("No search plan found, skipping GDELT retrieval")
            return state

        search_plan = state["search_plan"]

        # Optional domain filter provided by the user
        domains = state.get("preferred_domains", [])
        domain_filter = ""
        if domains:
            # Build something like "(domain:bbc.co.uk OR domain:reuters.com)"
            domain_filter = "(" + " OR ".join(f"domain:{d}" for d in domains) + ")"

        news_queries = [
            q
            for q in search_plan.get("queries", [])
            if q.get("source_type") == "news" or q.get("data_source") == "GDELT"
        ]

        if not news_queries:
            if state.get("warnings") is None:
                state["warnings"] = []
            state["warnings"].append("No GDELT queries in search plan")
            return state

        print(f"   > Processing {len(news_queries)} GDELT queries")

        retriever = GDELTRetriever(verbose=True)

        # Get base queries
        base_queries = [q["query"] for q in news_queries]

        # Add fallback queries
        if base_queries:
            print("   > Adding broad fallback queries...")
            base_queries.append(f"{state['country']} economic")
            base_queries.append(f"{state['country']} political")

        # Apply domain filter to all queries
        if domain_filter:
            print(f"   > Applying domain filter: {domain_filter}")
            query_strings = [f"{q} {domain_filter}" for q in base_queries]
        else:
            query_strings = base_queries

        documents = retriever.fetch_batch(
            queries=query_strings,
            start_date=state["update_period_start"],
            end_date=state["update_period_end"],
            country=state["country"],
            max_per_query=10,
        )

        if state.get("documents") is None:
            state["documents"] = []

        state["documents"].extend(documents)

        print(f"   > GDELT contribution: {len(documents)} scraped documents")
        print(f"   > Total documents so far: {len(state['documents'])}")

        state["current_step"] = "GDELTRetrievalComplete"

    except Exception as e:  # noqa: BLE001
        print(f"   ! ERROR in GDELT Retriever: {e}")
        if state.get("warnings") is None:
            state["warnings"] = []
        state["warnings"].append(f"GDELTRetrieverError: {str(e)}")

    return state
