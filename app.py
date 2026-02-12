"""Active Warning Updater - Streamlit Frontend.

Provides two modes:
1. Single Run Mode: Fill a form and run the pipeline for one country/risk.
2. Batch Mode: Upload an Excel file and process all risks sequentially.
"""

import io
import os
import uuid
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_UPDATE_PERIOD_DAYS,
    LLM_MODEL,
    MAX_CORRECTION_ATTEMPTS,
)
from utils.helpers import (
    get_preferred_domains,
    likelihood_to_score,
    impact_to_score,
    parse_risk_type,
    sanitize_filename,
)
from utils.markdown_output import generate_markdown_output, generate_summary_markdown


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Active Warning Updater",
    page_icon="\u26a0\ufe0f",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache the compiled graph so it isn't rebuilt on every Streamlit rerun
# ---------------------------------------------------------------------------
@st.cache_resource
def get_compiled_graph():
    from graph import compiled_graph
    return compiled_graph


# ---------------------------------------------------------------------------
# Helper: build initial state dict
# ---------------------------------------------------------------------------
def build_initial_state(
    country: str,
    risk_type: list,
    risk_title: str,
    previous_warning: str,
    previous_likelihood: int,
    previous_impact: int,
    previous_rationale: str,
    predefined_queries: list,
    preferred_domains: list,
    update_period_start: str,
    update_period_end: str,
) -> dict:
    return {
        "country": country,
        "risk_type": risk_type,
        "risk_title": risk_title,
        "previous_warning": previous_warning,
        "previous_seriousness_scores": {
            "likelihood": previous_likelihood,
            "impact": previous_impact,
            "rationale": previous_rationale,
        },
        "predefined_queries": predefined_queries,
        "preferred_domains": preferred_domains,
        "update_period_start": update_period_start,
        "update_period_end": update_period_end,
        # Initialize empty
        "search_plan": None,
        "documents": [],
        "events": [],
        "trend_analysis": None,
        "skeptic_flags": [],
        "narrative_paragraph_1": None,
        "narrative_paragraph_2": None,
        "citations": [],
        "status_recommendation": None,
        "error": None,
        "warnings": [],
        "run_id": f"{country}_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "correction_attempts": 0,
        "current_step": "initialized",
    }


# ---------------------------------------------------------------------------
# Node display names for progress panel
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "planner": "Query Planner",
    "gdelt_retriever": "GDELT Retriever",
    "reliefweb_retriever": "ReliefWeb Retriever",
    "translator": "Translator",
    "extractor": "Event Extractor",
    "trend_analysis": "Trend Analyst",
    "synthesis": "Narrative Synthesis",
    "skeptic": "Skeptic",
    "citation_manager": "Citation Manager",
    "status_recommender": "Status Recommender",
}


def _node_metrics(node_name: str, output_state: dict) -> str:
    """Return a short metric string for a completed node."""
    if node_name == "planner":
        plan = output_state.get("search_plan") or {}
        n = len(plan.get("queries", []))
        return f"{n} queries generated"
    if node_name == "gdelt_retriever":
        docs = output_state.get("documents", [])
        gdelt = sum(1 for d in docs if d.get("source") == "GDELT")
        return f"{gdelt} GDELT articles"
    if node_name == "reliefweb_retriever":
        docs = output_state.get("documents", [])
        rw = sum(1 for d in docs if "ReliefWeb" in d.get("source", ""))
        return f"{rw} ReliefWeb reports | {len(docs)} total documents"
    if node_name == "translator":
        docs = output_state.get("documents", [])
        translated = sum(1 for d in docs if d.get("translated"))
        return f"{translated} documents translated"
    if node_name == "extractor":
        evts = output_state.get("events", [])
        return f"{len(evts)} events extracted"
    if node_name == "trend_analysis":
        ta = output_state.get("trend_analysis") or {}
        return f"Trajectory: {ta.get('trajectory', 'N/A').upper()}"
    if node_name == "synthesis":
        att = output_state.get("correction_attempts", 0)
        return f"Attempt {att}"
    if node_name == "skeptic":
        flags = output_state.get("skeptic_flags", [])
        if flags:
            return f"{len(flags)} issues flagged"
        return "Draft clean"
    if node_name == "citation_manager":
        cits = output_state.get("citations", [])
        return f"{len(cits)} citations"
    if node_name == "status_recommender":
        rec = output_state.get("status_recommendation") or {}
        return rec.get("status_change", "N/A")
    return ""


# ===================================================================
# SIDEBAR
# ===================================================================
with st.sidebar:
    st.header("Mode")
    mode = st.radio(
        "Select mode",
        ["Single Run", "Batch"],
        label_visibility="collapsed",
    )

    with st.expander("Configuration", expanded=False):
        st.text_input("LLM Model", value=LLM_MODEL, disabled=True)
        max_corrections = st.number_input(
            "Max correction attempts",
            min_value=1,
            max_value=10,
            value=MAX_CORRECTION_ATTEMPTS,
        )

    st.markdown("---")
    with st.expander("About"):
        st.markdown(
            "**Active Warning Updater** transforms existing WFP risk "
            "warnings into updated 2-paragraph narratives with citations "
            "and status recommendations using a 9-agent LangGraph pipeline."
        )


# ===================================================================
# SINGLE RUN MODE
# ===================================================================
if mode == "Single Run":
    st.title("Active Warning Updater")
    st.caption("Single risk pipeline run")

    # ---- Input form ----
    with st.form("single_run_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            country = st.text_input("Country *", placeholder="e.g. Bolivia")
            risk_title = st.text_input(
                "Risk Title *",
                placeholder="e.g. Political instability aggravating economic challenges",
            )
            risk_type = st.multiselect(
                "Risk Type *",
                options=["conflict", "economic", "natural hazard"],
                default=["economic"],
            )
            previous_warning = st.text_area(
                "Previous Warning *",
                height=250,
                placeholder="Paste the previous warning narrative here...",
            )

        with col_right:
            st.markdown("**Previous Seriousness Scores**")
            prev_likelihood = st.slider("Likelihood", 1, 5, 3)
            prev_impact = st.slider("Impact", 1, 5, 3)
            prev_rationale = st.text_input(
                "Score Rationale",
                value="Previous analyst score",
            )

            st.markdown("**Search Configuration**")
            predefined_queries_text = st.text_area(
                "Predefined Queries (one per line)",
                height=80,
                placeholder="Optional additional queries...",
            )

            # Pre-fill preferred domains based on country
            default_domains = (
                "\n".join(get_preferred_domains(country))
                if country
                else ""
            )
            preferred_domains_text = st.text_area(
                "Preferred Domains (one per line)",
                value=default_domains,
                height=100,
            )

            st.markdown("**Update Period**")
            period_col1, period_col2 = st.columns(2)
            with period_col1:
                update_start = st.date_input(
                    "Start",
                    value=datetime.now() - timedelta(days=DEFAULT_UPDATE_PERIOD_DAYS),
                )
            with period_col2:
                update_end = st.date_input("End", value=datetime.now())

        submitted = st.form_submit_button(
            "Run Pipeline", type="primary", use_container_width=True,
        )

    # ---- Execution ----
    if submitted:
        # Validate required fields
        if not country or not risk_type or not risk_title or not previous_warning:
            st.error("Please fill in all required fields (marked with *).")
            st.stop()

        predefined_queries = [
            q.strip()
            for q in predefined_queries_text.strip().splitlines()
            if q.strip()
        ]
        preferred_domains = [
            d.strip()
            for d in preferred_domains_text.strip().splitlines()
            if d.strip()
        ]

        initial_state = build_initial_state(
            country=country,
            risk_type=risk_type,
            risk_title=risk_title,
            previous_warning=previous_warning,
            previous_likelihood=prev_likelihood,
            previous_impact=prev_impact,
            previous_rationale=prev_rationale,
            predefined_queries=predefined_queries,
            preferred_domains=preferred_domains,
            update_period_start=update_start.isoformat(),
            update_period_end=update_end.isoformat(),
        )

        graph = get_compiled_graph()
        final_state = {}

        try:
            with st.status("Running pipeline...", expanded=True) as status:
                for event in graph.stream(
                    initial_state, config={"recursion_limit": 50}
                ):
                    for node_name, output_state in event.items():
                        label = NODE_LABELS.get(node_name, node_name)
                        metrics = _node_metrics(node_name, output_state)
                        status.write(f"**{label}** complete  \n{metrics}")
                        final_state = output_state

                status.update(label="Pipeline complete!", state="complete")

        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            # Show whatever partial state we have
            if not final_state:
                final_state = initial_state

        # ---- Display results ----
        st.markdown("---")

        # Error banner
        if final_state.get("error"):
            st.error(f"**Error:** {final_state['error']}")

        # Narrative
        st.subheader("Narrative Update")
        p1 = final_state.get("narrative_paragraph_1")
        p2 = final_state.get("narrative_paragraph_2")
        st.markdown(p1 or "*Paragraph 1 not generated.*")
        st.markdown(p2 or "*Paragraph 2 not generated.*")

        # Status recommendation
        st.subheader("Status Recommendation")
        status_rec = final_state.get("status_recommendation")
        if status_rec:
            prev_s = status_rec.get("previous_seriousness", {})
            curr_s = status_rec.get("current_seriousness", {})
            rec_df = pd.DataFrame(
                {
                    "Metric": ["Likelihood", "Impact"],
                    "Previous": [
                        prev_s.get("likelihood", "N/A"),
                        prev_s.get("impact", "N/A"),
                    ],
                    "Current": [
                        curr_s.get("likelihood", "N/A"),
                        curr_s.get("impact", "N/A"),
                    ],
                }
            )
            st.table(rec_df)
            st.markdown(
                f"**Status Change:** {status_rec.get('status_change', 'N/A')}"
            )
            st.markdown(
                f"**Rationale:** {status_rec.get('rationale', 'N/A')}"
            )
        else:
            st.write("*Not generated.*")

        # Citations
        st.subheader("Citations")
        citations = final_state.get("citations") or []
        if citations:
            for i, cit in enumerate(citations, 1):
                title = cit.get("title", "No Title")
                url = cit.get("url", "")
                reliability = cit.get("reliability", 0.0)
                lang = cit.get("language", "")
                line = f"{i}. **{title}** (Reliability: {reliability:.2f})"
                if url:
                    line = f"{i}. [{title}]({url}) (Reliability: {reliability:.2f})"
                if lang and lang.lower() not in ("en", "english"):
                    line += f" | Original language: {lang}"
                st.markdown(line)
        else:
            st.write("*No citations.*")

        # Warnings
        warnings = final_state.get("warnings") or []
        if warnings:
            with st.expander(f"Warnings ({len(warnings)})", expanded=False):
                for w in warnings:
                    st.warning(w)

        # Download
        st.markdown("---")
        md_content = generate_markdown_output(
            country=country,
            risk_title=risk_title,
            risk_type=risk_type,
            previous_scores=(prev_likelihood, prev_impact),
            update_period=(update_start.isoformat(), update_end.isoformat()),
            paragraph_1=p1 or "",
            paragraph_2=p2 or "",
            status_recommendation=status_rec or {},
            citations=citations,
            warnings=warnings,
            run_id=initial_state["run_id"],
        )
        st.download_button(
            label="Download as Markdown",
            data=md_content,
            file_name=f"active_warning_{sanitize_filename(country)}.md",
            mime="text/markdown",
        )


# ===================================================================
# BATCH MODE
# ===================================================================
elif mode == "Batch":
    st.title("Active Warning Updater - Batch Mode")
    st.caption("Upload an Excel file to process all risks")

    uploaded_file = st.file_uploader(
        "Upload Active Warnings Excel file",
        type=["xlsx"],
    )

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        valid_rows = df[df["Country"].notna() & df["Title"].notna()].copy()

        st.subheader(f"Preview ({len(valid_rows)} risks)")

        preview_data = []
        for _, row in valid_rows.iterrows():
            preview_data.append({
                "Country": str(row.get("Country", "")),
                "Title": str(row.get("Title", ""))[:60],
                "Risk Type": str(row.get("risk_type", "")),
                "Likelihood": str(row.get("Likelihood", "")),
                "Impact": str(row.get("Impact", "")),
            })
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

        # Period settings
        col1, col2 = st.columns(2)
        with col1:
            batch_start = st.date_input(
                "Update Period Start",
                value=datetime.now() - timedelta(days=DEFAULT_UPDATE_PERIOD_DAYS),
                key="batch_start",
            )
        with col2:
            batch_end = st.date_input(
                "Update Period End",
                value=datetime.now(),
                key="batch_end",
            )

        run_batch = st.button(
            "Run All", type="primary", use_container_width=True,
        )

        if run_batch:
            graph = get_compiled_graph()
            results_summary = []
            all_markdown_files = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            total = len(valid_rows)

            for idx, (row_idx, row) in enumerate(valid_rows.iterrows()):
                country = str(row["Country"]).strip()
                risk_title = str(row["Title"]).strip()
                risk_type_list = parse_risk_type(row.get("risk_type", "conflict"))
                prev_warning = row.get("Last update (October 2025)", "")
                if pd.isna(prev_warning):
                    prev_warning = "No previous update available."
                else:
                    prev_warning = str(prev_warning).strip()

                l_score = likelihood_to_score(row.get("Likelihood", "Moderate"))
                i_score = impact_to_score(row.get("Impact", "Moderate"))

                status_text.markdown(
                    f"**Processing [{idx+1}/{total}]:** {country} - "
                    f"{risk_title[:50]}..."
                )
                progress_bar.progress((idx) / total)

                initial_state = build_initial_state(
                    country=country,
                    risk_type=risk_type_list,
                    risk_title=risk_title,
                    previous_warning=prev_warning,
                    previous_likelihood=l_score,
                    previous_impact=i_score,
                    previous_rationale="Previous score from Watch List.",
                    predefined_queries=[],
                    preferred_domains=get_preferred_domains(country),
                    update_period_start=batch_start.isoformat(),
                    update_period_end=batch_end.isoformat(),
                )

                try:
                    final_state = initial_state
                    for event in graph.stream(
                        initial_state, config={"recursion_limit": 50}
                    ):
                        for node_name, output_state in event.items():
                            final_state = output_state

                    p1 = final_state.get("narrative_paragraph_1", "*Not generated*")
                    p2 = final_state.get("narrative_paragraph_2", "*Not generated*")
                    status_rec = final_state.get("status_recommendation", {})
                    citations = final_state.get("citations", [])
                    warnings = final_state.get("warnings", [])

                    md_content = generate_markdown_output(
                        country=country,
                        risk_title=risk_title,
                        risk_type=risk_type_list,
                        previous_scores=(l_score, i_score),
                        update_period=(
                            batch_start.isoformat(),
                            batch_end.isoformat(),
                        ),
                        paragraph_1=p1,
                        paragraph_2=p2,
                        status_recommendation=status_rec,
                        citations=citations,
                        warnings=warnings,
                        run_id=initial_state["run_id"],
                    )

                    filename = (
                        f"{sanitize_filename(country)}_"
                        f"{sanitize_filename(risk_title[:30])}_"
                        f"{idx+1}.md"
                    )
                    all_markdown_files[filename] = md_content

                    results_summary.append({
                        "index": idx + 1,
                        "country": country,
                        "risk_title": risk_title,
                        "status": "SUCCESS",
                        "file": filename,
                        "recommendation": (
                            status_rec.get("status_change", "N/A")
                            if status_rec
                            else "N/A"
                        ),
                    })

                except Exception as e:
                    results_summary.append({
                        "index": idx + 1,
                        "country": country,
                        "risk_title": risk_title,
                        "status": "FAILED",
                        "file": None,
                        "error": str(e),
                    })

            progress_bar.progress(1.0)
            status_text.markdown("**Batch processing complete!**")

            # ---- Summary table ----
            st.subheader("Results Summary")
            summary_df = pd.DataFrame(results_summary)
            display_cols = [
                c for c in ["index", "country", "status", "recommendation"]
                if c in summary_df.columns
            ]
            st.dataframe(
                summary_df[display_cols] if display_cols else summary_df,
                use_container_width=True,
            )

            success_count = sum(
                1 for r in results_summary if r["status"] == "SUCCESS"
            )
            failed_count = sum(
                1 for r in results_summary if r["status"] == "FAILED"
            )
            st.markdown(
                f"**Successful:** {success_count}/{total} | "
                f"**Failed:** {failed_count}/{total}"
            )

            # ---- Zip download ----
            if all_markdown_files:
                # Add summary markdown
                summary_md = generate_summary_markdown(
                    results_summary,
                    batch_start.isoformat(),
                    batch_end.isoformat(),
                )
                all_markdown_files["_BATCH_SUMMARY.md"] = summary_md

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(
                    zip_buffer, "w", zipfile.ZIP_DEFLATED
                ) as zf:
                    for fname, content in all_markdown_files.items():
                        zf.writestr(fname, content)

                st.download_button(
                    label="Download All as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="active_warnings_batch.zip",
                    mime="application/zip",
                )
