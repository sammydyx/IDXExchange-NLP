from collections import Counter
from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "listing_sample.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "taxonomy_candidates.csv"

TOP_CANDIDATES_PER_NGRAM = 500

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "he", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "to", "was", "were", "will", "with",
    "you", "your", "we", "our", "they", "their", "not", "all",
    "also", "can", "new", "one", "two", "three",
}


def normalize_text(text):
    """Normalize a few common real-estate formats before tokenization."""
    text = text.lower()
    text = re.sub(r"sq\.?\s*ft\.?", "sqft", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)  # 2,000 -> 2000
    return text


def tokenize(text):
    """Keep real-estate words and hyphenated terms; exclude numeric tokens."""
    pattern = r"\b[a-z]+(?:-[a-z]+)*\b"
    return re.findall(pattern, text)


def get_ngrams(tokens, n):
    """Return consecutive groups of n tokens."""
    return [
        tuple(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
    ]


def is_meaningful_candidate(gram):
    """Remove stopword-boundary phrases and short-token fragments."""
    if gram[0] in STOP_WORDS or gram[-1] in STOP_WORDS:
        return False

    if any(len(word) < 3 for word in gram):
        return False

    return any(word not in STOP_WORDS for word in gram)


def main():
    df = pd.read_csv(INPUT_PATH)

    all_text = " ".join(
        df["remarks"]
        .dropna()
        .astype(str)
        .map(normalize_text)
    )

    tokens = tokenize(all_text)

    rows = []

    for n in [1, 2, 3]:
        frequencies = Counter(get_ngrams(tokens, n))

        kept_count = 0

        for gram, count in frequencies.most_common():
            if not is_meaningful_candidate(gram):
                continue

            rows.append(
                {
                    "term": " ".join(gram),
                    "ngram": n,
                    "count": count,
                    "keep": "",
                    "category": "",
                }
            )

            kept_count += 1

            if kept_count == TOP_CANDIDATES_PER_NGRAM:
                break

    candidates = pd.DataFrame(rows)
    candidates.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(candidates):,} candidate terms to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()