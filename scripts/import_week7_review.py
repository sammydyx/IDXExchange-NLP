"""Import a human-reviewed Week 7 annotation workbook into the final CSV."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.query_intent_classifier import INTENT_LABELS


OUTPUT_PATH = ROOT / "data/processed/week7_intent_queries.csv"


def _clean(value: object) -> str:
    return str(value or "").strip()


def import_review(
    review_path: Path,
    output_path: Path = OUTPUT_PATH,
    *,
    ignore_invalid_approved_intent: bool = False,
) -> pd.DataFrame:
    """Create the final, human-reviewed CSV from the Review worksheet.

    A reviewer may leave ``Approved intent`` blank to accept a proposed label.
    Non-empty invalid values normally stop the import.  The explicit flag is
    reserved for correcting a known accidental spreadsheet entry.
    """
    review = pd.read_excel(review_path, sheet_name="Review", header=3)
    required = {"Query ID", "Query", "Proposed intent", "Approved intent", "Reviewer notes"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"Review sheet is missing columns: {sorted(missing)}")

    status_column = next(
        (column for column in review.columns if _clean(column).lower() in {"review status", "review_status", "approved"}),
        None,
    )
    if status_column is None:
        raise ValueError("Review sheet must contain a Review status column")

    review = review.loc[review["Query ID"].notna()].copy()
    review["review_status"] = review[status_column].map(_clean).str.lower()
    if not review["review_status"].eq("approved").all():
        pending = review.loc[~review["review_status"].eq("approved"), "Query ID"].tolist()
        raise ValueError(f"All rows must be approved before import; pending IDs: {pending[:10]}")

    proposed = review["Proposed intent"].map(_clean)
    approved = review["Approved intent"].map(_clean)
    invalid_approved = approved.ne("") & ~approved.isin(INTENT_LABELS)
    if invalid_approved.any() and not ignore_invalid_approved_intent:
        values = sorted(set(approved[invalid_approved]))
        raise ValueError(f"Invalid Approved intent values: {values}")

    effective_intent = approved.where(approved.isin(INTENT_LABELS), proposed)
    if not effective_intent.isin(INTENT_LABELS).all():
        invalid = sorted(set(effective_intent[~effective_intent.isin(INTENT_LABELS)]))
        raise ValueError(f"Invalid proposed intent values: {invalid}")

    notes = review["Reviewer notes"].map(_clean)
    if ignore_invalid_approved_intent:
        # The user confirmed these cells were accidentally filled with `11`.
        notes = notes.mask(notes.eq("11"), "")

    final = pd.DataFrame({
        "query_id": review["Query ID"].map(_clean),
        "query": review["Query"].map(_clean),
        "intent": effective_intent,
        "label_source": "human_reviewed",
        "review_status": "manual_verified",
        "notes": notes,
    })
    if len(final) < 200 or final["query"].nunique() != len(final):
        raise ValueError("Final review file must contain at least 200 unique queries")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Week 7 human review results.")
    parser.add_argument("review_path", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--ignore-invalid-approved-intent",
        action="store_true",
        help="Use only after confirming accidental invalid spreadsheet values should be ignored.",
    )
    args = parser.parse_args()
    final = import_review(
        args.review_path,
        args.output,
        ignore_invalid_approved_intent=args.ignore_invalid_approved_intent,
    )
    print(f"Saved {len(final)} manually verified queries to {args.output}")
    print(final["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
