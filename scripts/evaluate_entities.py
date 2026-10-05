from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = PROJECT_ROOT / "data/processed/week3_annotation_template.csv"
REPORT_PATH = PROJECT_ROOT / "data/processed/week3_evaluation_report.md"

def _as_set(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    try:
        return {str(item).lower().strip() for item in json.loads(value)}
    except json.JSONDecodeError:
        return {item.strip().lower() for item in str(value).split(";") if item.strip()}

def score_scalar(dataframe, predicted, expected):
    """Score a numeric entity with exact-value precision, recall, and F1."""
    subset = dataframe[[predicted, expected]].copy()
    prediction = pd.to_numeric(subset[predicted], errors="coerce")
    truth = pd.to_numeric(subset[expected], errors="coerce")
    exact_match = prediction.notna() & truth.notna() & prediction.eq(truth)
    true_positive = int(exact_match.sum())
    false_positive = int((prediction.notna() & ~exact_match).sum())
    false_negative = int((truth.notna() & ~exact_match).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return int(len(subset)), true_positive, false_positive, false_negative, precision, recall, f1

def score_lists(dataframe, predicted, expected):
    reviewed = dataframe[expected].notna() & dataframe[expected].astype(str).str.strip().ne("")
    tp = fp = fn = 0
    for _, row in dataframe.loc[reviewed, [predicted, expected]].iterrows():
        prediction, truth = _as_set(row[predicted]), _as_set(row[expected])
        tp += len(prediction & truth); fp += len(prediction - truth); fn += len(truth - prediction)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return int(reviewed.sum()), precision, recall, f1

def main():
    dataframe = pd.read_csv(ANNOTATION_PATH)
    if "review_status" in dataframe.columns:
        dataframe = dataframe[dataframe["review_status"].fillna("").str.lower().eq("manual_verified")].copy()
    lines = ["# Week 3 Entity Extraction Evaluation", "", "Only rows marked manual_verified are included. AI-assisted drafts are intentionally excluded.", "", "## Scalar entities", ""]
    for name, predicted, expected in [
        ("bedrooms", "bedrooms_extracted", "expected_bedrooms"),
        ("bathrooms", "bathrooms_extracted", "expected_bathrooms"),
        ("square_feet", "square_feet_extracted", "expected_square_feet"),
        ("price_mentioned", "price_mentioned_extracted", "expected_price_mentioned"),
    ]:
        reviewed, true_positive, false_positive, false_negative, precision, recall, f1 = score_scalar(dataframe, predicted, expected)
        lines.append(f"- {name}: precision {precision:.1%}, recall {recall:.1%}, F1 {f1:.1%} (TP={true_positive}, FP={false_positive}, FN={false_negative}; {reviewed} reviewed listings)")
    lines += ["", "## Feature entities", ""]
    for name, predicted, expected in [
        ("amenities", "amenities_extracted", "expected_amenities"),
        ("interior_features", "interior_features_extracted", "expected_interior_features"),
        ("exterior_features", "exterior_features_extracted", "expected_exterior_features"),
    ]:
        reviewed, precision, recall, f1 = score_lists(dataframe, predicted, expected)
        lines.append(f"- {name}: precision {precision:.1%}, recall {recall:.1%}, F1 {f1:.1%} across {reviewed} reviewed listings")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
