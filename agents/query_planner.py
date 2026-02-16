import json

from langchain_core.prompts import ChatPromptTemplate

from state import ActiveWarningsState
from llm import llm


# ===== Prompt Template for Query Planner =====

PLANNER_PROMPT_TEMPLATE = """
You are a specialist Early Warning Analyst for a major humanitarian organization.
Your task is to generate an optimal search strategy to update an existing risk warning. [cite: 23]

You will be given the context of the previous warning, the country, the risk type, and the time period to cover.
Your goal is to generate 5-10 specific, targeted search queries to find new developments. [cite: 27]

**Input:**
- Country: {country}
- Risk Type: {risk_type}
- Update Period: {update_period}
- Previous Warning Text (for context):
---
{previous_warning}
---
- Existing Predefined Queries (to incorporate/improve): {predefined_queries}

**Instructions:**
1.  **Analyze Context:** Read the `Previous Warning Text` to identify key themes, actors, locations, and specific indicators (e.g., inflation figures, conflict events) that need updating. [cite: 25]
2.  **Determine Sources:** Based on the `Risk Type(s)`, identify the best data sources. [cite: 28]
    - 'economic': Prioritize sources like IMF, World Bank...
    - 'conflict': Prioritize sources like ACLED, ReliefWeb... [cite: 33, 34]
    - 'natural hazard': Prioritize sources like NOAA, ReliefWeb...
    Your strategy must gather information for ALL of the listed risk types.
    - For analyst report queries targeting Seerist: Use Lucene-compatible search syntax (e.g., "Bolivia AND inflation", "Sudan AND conflict AND displacement").
      Seerist supports the following topic categories: travel, unrest, transportation, health, terrorism, conflict, disaster, crime.
      Set data_source to "Seerist" for these queries.
3. **Generate Queries:** Create 5-10 queries. Most should be specific, but include 1-2 broader "fallback" queries (e.g., "Bolivia AND economic crisis", "Bolivia AND political situation") to ensure some results are returned.
4.  **Format Output:** You MUST return *only* a valid JSON object that adheres to the `SearchPlan` schema.
    - `key_themes` and `key_actors` should be extracted from the previous warning. [cite: 25, 26]
    - `rationale` should briefly explain *why* this search plan is effective.

**Output JSON Schema:**
{{
    "queries": [
        {{
            "query": "str (specific search query)",
            "source_type": "Literal['news', 'un_reports', 'economic', 'climate']",
            "data_source": "str (e.g., 'Seerist', 'ReliefWeb', 'IMF', 'ACLED')",
            "priority": "Literal['high', 'medium', 'low']"
        }}
    ],
    "key_themes": ["str (e.g., 'food inflation', 'subsidy cuts')"],
    "key_actors": ["str (e.g., 'Central Bank', 'Ministry of Trade')"],
    "rationale": "str (brief explanation of the search strategy)"
}}
"""


# ===== LangGraph Node Function =====

def run_query_planner(state: ActiveWarningsState) -> ActiveWarningsState:
    """Runs the Query Planning Agent to generate a search strategy.

    (Corrected version using manual JSON parsing.)
    """
    print(f"--- (1) Running Query Planner for {state['country']} ---")

    try:
        # 1. Get inputs from state
        inputs = {
            "country": state["country"],
            "risk_type": ", ".join(state["risk_type"]),  # Join list for prompt
            "update_period": f"{state['update_period_start']} to {state['update_period_end']}",
            "previous_warning": state["previous_warning"],
            "predefined_queries": ", ".join(state["predefined_queries"]),
        }

        # 2. Set up the LLM chain *without* structured output initially
        prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT_TEMPLATE)

        # We will parse the JSON manually
        planner_chain = prompt | llm

        # 3. Invoke the chain - get raw string output
        response = planner_chain.invoke(inputs)
        raw_output = response.content

        # 4. Manually parse the JSON output
        try:
            # Clean potential markdown code blocks
            if raw_output.strip().startswith("```json"):
                raw_output = raw_output.strip()[7:-3].strip()
            elif raw_output.strip().startswith("```"):
                raw_output = raw_output.strip()[3:-3].strip()

            search_plan = json.loads(raw_output)

            # Basic validation (check if essential keys exist)
            if "queries" not in search_plan or "rationale" not in search_plan:
                raise ValueError("Parsed JSON missing required keys ('queries', 'rationale')")

        except json.JSONDecodeError as e:
            print(f"   ! ERROR: Failed to parse JSON output from LLM: {e}")
            print(f"   Raw LLM Output:\n{raw_output}")
            raise ValueError(
                f"LLM did not return valid JSON for SearchPlan. Raw output: {raw_output}"
            ) from e
        except ValueError as e:
            print(f"   ! ERROR: Parsed JSON is not a valid SearchPlan: {e}")
            raise e

        print(f"   > Generated {len(search_plan.get('queries', []))} search queries.")

        # 5. Update the state
        state["search_plan"] = search_plan
        state["current_step"] = "QueryPlanningComplete"

    except Exception as e:
        print(f"   ! ERROR in Query Planner: {e}")
        state["error"] = f"QueryPlannerError: {str(e)}"

    return state
