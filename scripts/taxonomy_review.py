import pandas as pd


INPUT_PATH = "data/processed/taxonomy_candidates.csv"
OUTPUT_PATH = "data/processed/taxonomy_review.csv"

df = pd.read_csv(INPUT_PATH)

# Normalize terms for filtering.
terms = df["term"].fillna("").astype(str).str.strip().str.lower()

# Get rid of some obviously meaningless candidates.
bad_terms = {
    "the home",
    "this home",
    "one of",
    "welcome to",
    "features a",
    "home offers",
    "offers a",
    "a spacious",
    "opportunity to",
    "the property",
    "a large",
    "the perfect",
    "home is",
    "perfect for",
    "the main",
    "plenty of",
    "a rare",
    "ideal for",
    "space for",
    "the primary",
    "this property",
    "in one",
}

# Remove:
# - exact known bad phrases;
# - single numeric terms, such as "3" or "2024";
# - single one- or two-letter terms, such as "a", "x", "la".
is_bad_phrase = terms.isin(bad_terms)
is_single_number = terms.str.fullmatch(r"\d+(?:\.\d+)?")
is_short_single_word = terms.str.fullmatch(r"[a-z]{1,2}")

df = df[
    ~(is_bad_phrase | is_single_number | is_short_single_word)
].copy()

# Add manual-review columns only if they do not already exist.
for column in ["keep", "category", "notes"]:
    if column not in df.columns:
        df[column] = ""

limits = {
    1: 100,
    2: 150,
    3: 150,
}

shortlist_parts = []

for ngram_size, limit in limits.items():
    subset = df[df["ngram"] == ngram_size]
    subset = subset.sort_values("count", ascending=False).head(limit)
    shortlist_parts.append(subset)

shortlist = pd.concat(shortlist_parts, ignore_index=True)

shortlist.to_csv(
    "data/processed/taxonomy_shortlist.csv",
    index=False,
)

print(f"Saved {len(shortlist)} priority terms to data/processed/taxonomy_shortlist.csv")

df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} terms for review to {OUTPUT_PATH}")