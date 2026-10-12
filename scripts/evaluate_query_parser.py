from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.query_parser import QueryParser


QUERY_PATH = PROJECT_ROOT / "data/processed/sample_user_queries.csv"
REPORT_PATH = PROJECT_ROOT / "data/processed/week4_query_parser_report.md"


def main() -> None:
    dataframe = pd.read_csv(QUERY_PATH)
    parser = QueryParser()
    dataframe["predicted_categories"] = dataframe["query"].apply(
        lambda query: sorted(parser.parse(query).matched_categories)
    )
    dataframe["correct"] = dataframe.apply(
        lambda row: row["target_category"] in row["predicted_categories"], axis=1
    )
    accuracy = dataframe["correct"].mean()
    failures = dataframe.loc[~dataframe["correct"], ["query_id", "query", "target_category", "predicted_categories"]]
    lines = [
        "# Week 4 Query Parser Evaluation",
        "",
        f"- Week 1 queries evaluated: {len(dataframe)}",
        f"- Category parsing accuracy: {accuracy:.1%}",
        f"- Target: 90.0%+",
        "",
        "## Unmatched queries",
        "",
    ]
    if failures.empty:
        lines.append("All Week 1 sample queries matched their expected target category.")
    else:
        lines.extend(f"- {row.query_id}: expected `{row.target_category}`, got `{row.predicted_categories}`" for row in failures.itertuples())
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
