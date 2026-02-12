"""Utility functions for batch processing and data mapping.

Extracted from the Notebook_GEMINI.ipynb batch processing cell.
"""

import re

import pandas as pd


def likelihood_to_score(likelihood_text: str) -> int:
    """Convert likelihood text to numeric score (1-5)."""
    if pd.isna(likelihood_text):
        return 3  # Default to Moderate

    mapping = {
        "very high": 5,
        "high": 4,
        "moderate": 3,
        "low": 2,
        "very low": 1
    }
    return mapping.get(str(likelihood_text).lower().strip(), 3)


def impact_to_score(impact_text: str) -> int:
    """Convert impact text to numeric score (1-5)."""
    if pd.isna(impact_text):
        return 3  # Default to Moderate

    text_lower = str(impact_text).lower()

    if "very high" in text_lower or "> 500" in text_lower or ">500" in text_lower:
        return 5
    elif "high" in text_lower or "250,000" in text_lower or "250000" in text_lower:
        return 4
    elif "moderate" in text_lower or "100,000" in text_lower or "100000" in text_lower:
        return 3
    elif "low" in text_lower:
        return 2
    else:
        return 3  # Default


def parse_risk_type(risk_type_text: str) -> list:
    """Parse risk_type string to list format expected by the agent."""
    if pd.isna(risk_type_text):
        return ["conflict"]  # Default fallback

    text = str(risk_type_text).lower().strip()

    # Handle combined types
    if "/" in text:
        parts = text.split("/")
        result = []
        for part in parts:
            part = part.strip()
            if part == "climate":
                result.append("natural hazard")
            else:
                result.append(part)
        return result

    # Single type
    if text == "climate":
        return ["natural hazard"]
    elif text in ["conflict", "economic"]:
        return [text]
    else:
        return ["conflict"]  # Fallback


def get_preferred_domains(country: str) -> list:
    """Generate preferred domains based on country/region."""

    # Base international sources (always included)
    base_domains = [
        "bbc.co.uk", "bbc.com", "reuters.com", "aljazeera.com",
        "theguardian.com", "apnews.com", "france24.com", "dw.com",
        "reliefweb.int", "unocha.org", "thenewhumanitarian.org"
    ]

    # Regional domains
    africa_domains = [
        "theeastafrican.co.ke", "nation.africa", "allafrica.com",
        "africanews.com"
    ]

    middle_east_domains = [
        "middleeasteye.net", "al-monitor.com"
    ]

    latin_america_domains = [
        "elpais.com", "infobae.com"
    ]

    asia_domains = [
        "scmp.com", "asia.nikkei.com"
    ]

    # Country-specific mappings
    country_lower = country.lower()

    # Africa
    african_countries = [
        "sudan", "south sudan", "ethiopia", "somalia", "kenya", "uganda",
        "democratic republic of the congo", "drc", "chad", "mali", "niger",
        "burkina faso", "nigeria", "mozambique", "madagascar", "lesotho",
        "mauritania", "libya"
    ]

    # Middle East
    middle_east_countries = [
        "yemen", "syria", "lebanon", "palestine", "gaza", "iran", "iraq"
    ]

    # Latin America
    latin_america_countries = [
        "haiti", "cuba", "venezuela", "colombia", "ecuador", "bolivia"
    ]

    # Asia
    asia_countries = [
        "afghanistan", "myanmar", "bangladesh", "nepal", "pakistan"
    ]

    domains = base_domains.copy()

    if any(c in country_lower for c in african_countries):
        domains.extend(africa_domains)

    if any(c in country_lower for c in middle_east_countries):
        domains.extend(middle_east_domains)

    if any(c in country_lower for c in latin_america_countries):
        domains.extend(latin_america_domains)

    if any(c in country_lower for c in asia_countries):
        domains.extend(asia_domains)

    # Ukraine special case
    if "ukraine" in country_lower:
        domains.extend(["kyivindependent.com", "pravda.com.ua"])

    return list(dict.fromkeys(domains))  # Remove duplicates


def sanitize_filename(text: str, max_length: int = 50) -> str:
    """Create a safe filename from text."""
    # Remove or replace unsafe characters
    safe = re.sub(r'[^\w\s-]', '', text)
    safe = re.sub(r'\s+', '_', safe)
    return safe[:max_length].strip('_')
