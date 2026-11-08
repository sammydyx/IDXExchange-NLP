"""Train and evaluate the Week 7 query-intent classifier."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.query_intent_classifier import INTENT_LABELS, QueryIntentClassifier


DATA_PATH = ROOT / "data/processed/week7_intent_queries.csv"
MODEL_PATH = ROOT / "data/models/week7_query_intent_classifier.joblib"
REPORT_PATH = ROOT / "data/processed/week7_intent_evaluation_report.md"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = {"query", "intent", "review_status"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if len(dataframe) < 200:
        raise ValueError("Week 7 requires at least 200 labeled queries")
    if set(dataframe["intent"]) != set(INTENT_LABELS):
        raise ValueError("dataset must contain all three Week 7 intent labels")
    return dataframe


def train_and_evaluate(dataframe: pd.DataFrame) -> tuple[QueryIntentClassifier, dict[str, object]]:
    train_queries, test_queries, train_labels, test_labels = train_test_split(
        dataframe["query"].tolist(),
        dataframe["intent"].tolist(),
        test_size=0.20,
        stratify=dataframe["intent"],
        random_state=42,
    )
    classifier = QueryIntentClassifier().train(train_queries, train_labels)
    predicted = [classifier.predict(query)[0] for query in test_queries]
    accuracy = float(accuracy_score(test_labels, predicted))
    return classifier, {
        "accuracy": accuracy,
        "test_size": len(test_queries),
        "train_size": len(train_queries),
        "classification_report": classification_report(test_labels, predicted, digits=3, zero_division=0),
        "confusion_matrix": confusion_matrix(test_labels, predicted, labels=list(INTENT_LABELS)).tolist(),
    }


def write_report(dataframe: pd.DataFrame, metrics: dict[str, object], path: Path = REPORT_PATH) -> None:
    review_counts = dataframe["review_status"].value_counts().to_dict()
    all_manually_verified = dataframe["review_status"].eq("manual_verified").all()
    limitation = (
        "All labels in this dataset were manually reviewed. The score still measures performance on "
        "held-out examples drawn from the same curated dataset, so future work should add more naturally "
        "written and ambiguous queries before making broader real-world claims."
        if all_manually_verified
        else
        "The initial dataset is a curated seed draft and is marked `needs_human_review`. Its score measures "
        "performance on held-out examples drawn from the same draft distribution; it must not be presented "
        "as final manually-labeled evaluation evidence until humans review the labels."
    )
    lines = [
        "# Week 7 Query Intent Classification Evaluation",
        "",
        f"- Dataset rows: {len(dataframe)}",
        f"- Train/test split: {metrics['train_size']}/{metrics['test_size']} (stratified, random_state=42)",
        f"- Held-out accuracy: {metrics['accuracy']:.1%}",
        "- Target: at least 80.0% held-out accuracy",
        f"- Review status counts: {review_counts}",
        "",
        "## Important limitation",
        "",
        limitation,
        "",
        "## Classification report",
        "",
        "```text",
        str(metrics["classification_report"]).rstrip(),
        "```",
        "",
        "## Confusion matrix",
        "",
        f"Labels in row/column order: {list(INTENT_LABELS)}",
        f"{metrics['confusion_matrix']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dataframe = load_dataset()
    classifier, metrics = train_and_evaluate(dataframe)
    # Retrain on every reviewed/draft row for the saved interactive model.
    classifier.train(dataframe["query"].tolist(), dataframe["intent"].tolist())
    classifier.save(MODEL_PATH)
    write_report(dataframe, metrics)
    print(REPORT_PATH.read_text(encoding="utf-8"))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
