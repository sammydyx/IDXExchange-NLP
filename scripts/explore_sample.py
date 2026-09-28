from collections import Counter
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "listing_sample.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "profile_report.txt"


def main():
    dataframe = pd.read_csv(INPUT_PATH)
    remarks = dataframe["remarks"].fillna("").astype(str)

    all_text = " ".join(remarks.str.lower())
    tokens = re.findall(r"[a-z]+", all_text)

    stop_words = {
        "the", "and", "to", "of", "in", "a", "with", "for", "this",
        "is", "on", "at", "an", "or", "from", "as", "by", "are",
        "you", "your", "that", "one", "two",
    }

    useful_tokens = [
        token for token in tokens
        if len(token) >= 3 and token not in stop_words
    ]

    common_words = Counter(useful_tokens).most_common(30)

    report_lines = [
        f"Total listings: {len(dataframe):,}",
        f"Missing remarks: {remarks.eq('').sum():,}",
        f"Average remark length: {remarks.str.len().mean():.0f} characters",
        f"Shortest remark: {remarks.str.len().min():.0f} characters",
        f"Longest remark: {remarks.str.len().max():.0f} characters",
        f"Listings containing HTML-like tags: {remarks.str.contains(r'<[^>]+>', regex=True).sum():,}",
        "",
        "Top 30 useful words:",
    ]

    for word, count in common_words:
        report_lines.append(f"{word:<20} {count}")

    report = "\n".join(report_lines)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()