"""Batch processing logic for Active Warnings pipeline.

Reads an Excel file with risk entries and processes each through the
LangGraph pipeline, generating markdown output files.

Extracted from Notebook_GEMINI.ipynb.
"""

import os
import uuid
from datetime import datetime, timedelta

import pandas as pd

from graph import compiled_graph
from config import DEFAULT_UPDATE_PERIOD_DAYS
from utils.helpers import (
    likelihood_to_score,
    impact_to_score,
    parse_risk_type,
    get_preferred_domains,
    sanitize_filename,
)
from utils.markdown_output import generate_markdown_output, generate_summary_markdown


def run_batch_processing(
    excel_file: str,
    output_dir: str = "./active_warnings_outputs",
    update_period_start: str = None,
    update_period_end: str = None,
    progress_callback=None,
) -> list:
    """Main function to process all risks from the Excel file.

    Args:
        excel_file: Path to the Excel file with risk entries.
        output_dir: Directory to save markdown output files.
        update_period_start: Start date (YYYY-MM-DD). Defaults to today - 60 days.
        update_period_end: End date (YYYY-MM-DD). Defaults to today.
        progress_callback: Optional callable(index, total, country, status)
            for UI progress updates.

    Returns:
        List of result summary dicts.
    """
    if update_period_end is None:
        update_period_end = datetime.now().strftime("%Y-%m-%d")
    if update_period_start is None:
        update_period_start = (
            datetime.now() - timedelta(days=DEFAULT_UPDATE_PERIOD_DAYS)
        ).strftime("%Y-%m-%d")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Read Excel file
    print(f"Reading Excel file: {excel_file}")
    df = pd.read_excel(excel_file)

    # Filter valid rows
    valid_rows = df[df['Country'].notna() & df['Title'].notna()].copy()
    total_risks = len(valid_rows)

    print(f"Found {total_risks} risks to process")
    print(f"Update period: {update_period_start} to {update_period_end}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    # Track results
    results_summary = []

    # Process each risk
    for idx, (row_idx, row) in enumerate(valid_rows.iterrows()):
        country = str(row['Country']).strip()
        risk_title = str(row['Title']).strip()
        risk_type = parse_risk_type(row.get('risk_type', 'conflict'))

        # Get previous warning text
        previous_warning = row.get('Last update (October 2025)', None)
        if pd.isna(previous_warning):
            previous_warning = "No previous update available."
        else:
            previous_warning = str(previous_warning).strip()

        # Get scores
        likelihood_score = likelihood_to_score(row.get('Likelihood', 'Moderate'))
        impact_score = impact_to_score(row.get('Impact', 'Moderate'))

        print(f"\n{'='*70}")
        print(f"PROCESSING [{idx+1}/{total_risks}]: {country}")
        print(f"   Risk: {risk_title[:60]}...")
        print(f"   Type: {risk_type}")
        print(f"   Previous Scores: L={likelihood_score}, I={impact_score}")
        print("=" * 70)

        if progress_callback:
            progress_callback(idx, total_risks, country, "processing")

        # Create input state
        run_input = {
            # Core fields
            "country": country,
            "risk_type": risk_type,
            "risk_title": risk_title,
            "previous_warning": previous_warning,

            # Scores
            "previous_seriousness_scores": {
                "likelihood": likelihood_score,
                "impact": impact_score,
                "rationale": "Previous score from Watch List.",
            },

            # Search configuration
            "predefined_queries": [],
            "preferred_domains": get_preferred_domains(country),
            "update_period_start": update_period_start,
            "update_period_end": update_period_end,

            # Initialize empty fields
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
            "run_id": f"batch_{country}_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "correction_attempts": 0,
            "current_step": "initialized",
        }

        try:
            # Run the graph
            final_state = run_input
            for event in compiled_graph.stream(
                run_input, config={"recursion_limit": 50}
            ):
                for node_name, output_state in event.items():
                    print(f"   --- {node_name} ---")
                    final_state = output_state

            # Extract results
            p1 = final_state.get("narrative_paragraph_1", "*Not generated*")
            p2 = final_state.get("narrative_paragraph_2", "*Not generated*")
            status_rec = final_state.get("status_recommendation", {})
            citations = final_state.get("citations", [])
            warnings = final_state.get("warnings", [])

            # Create markdown content
            md_content = generate_markdown_output(
                country=country,
                risk_title=risk_title,
                risk_type=risk_type,
                previous_scores=(likelihood_score, impact_score),
                update_period=(update_period_start, update_period_end),
                paragraph_1=p1,
                paragraph_2=p2,
                status_recommendation=status_rec,
                citations=citations,
                warnings=warnings,
                run_id=run_input["run_id"]
            )

            # Save to file
            filename = (
                f"{sanitize_filename(country)}_"
                f"{sanitize_filename(risk_title[:30])}_"
                f"{idx+1}.md"
            )
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)

            print(f"Saved: {filename}")

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
                "markdown": md_content,
            })

        except Exception as e:
            print(f"ERROR processing {country}: {e}")

            results_summary.append({
                "index": idx + 1,
                "country": country,
                "risk_title": risk_title,
                "status": "FAILED",
                "file": None,
                "error": str(e),
            })

        if progress_callback:
            progress_callback(
                idx + 1,
                total_risks,
                country,
                results_summary[-1]["status"],
            )

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)

    success_count = sum(1 for r in results_summary if r["status"] == "SUCCESS")
    failed_count = sum(1 for r in results_summary if r["status"] == "FAILED")

    print(f"Successful: {success_count}/{total_risks}")
    print(f"Failed: {failed_count}/{total_risks}")

    if failed_count > 0:
        print("\nFailed runs:")
        for r in results_summary:
            if r["status"] == "FAILED":
                print(f"  - {r['country']}: {r.get('error', 'Unknown error')}")

    # Save summary
    summary_path = os.path.join(output_dir, "_BATCH_SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            generate_summary_markdown(
                results_summary, update_period_start, update_period_end
            )
        )

    print(f"\nSummary saved to: {summary_path}")

    return results_summary
