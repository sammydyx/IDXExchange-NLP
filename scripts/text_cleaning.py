from __future__ import annotations
from collections import Counter
from html import unescape
from pathlib import Path
import re
import unicodedata
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class TextCleaner:
    def __init__(self):
        self.abbrev_map = {
            "br":"bedroom", "bdrm":"bedroom", "bd":"bedroom", "ba":"bathroom", "bth":"bathroom",
            "sqft":"square feet", "sq ft":"square feet", "sf":"square feet", "w/":"with", "w/o":"without",
            "a/c":"air conditioning", "ac":"air conditioning", "hvac":"heating ventilation and air conditioning",
            "fp":"fireplace", "gar":"garage", "pkg":"parking", "kit":"kitchen", "lr":"living room",
            "dr":"dining room", "mbr":"primary bedroom", "mb":"primary bedroom", "hoa":"homeowners association",
            "ss":"stainless steel", "app":"appliances", "upd":"updated", "renov":"renovated", "yr":"year",
            "yrs":"years", "approx.":"approximately","approx":"approximately","incl":"included", "det":"detached",
        }
    
    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKC", str(text or "")).replace("–", "-").replace("—", "-").replace("’", "'")
    
    def remove_html(self, text):
        return re.sub(r"<[^>]+>", " ", unescape(text))
    
    def normalize_prices(self, text):
        text = re.sub(r"\$?(\d+(?:\.\d+)?)\s*k\b", lambda m: str(int(float(m.group(1))*1000)), text, flags=re.I)
        return re.sub(r"\$?(\d+(?:\.\d+)?)\s*m\b", lambda m: str(int(float(m.group(1))*1000000)), text, flags=re.I)
    
    def normalize_measurements(self, text):
        return re.sub(r"\b(\d{1,3}(?:,\d{3})*|\d+)\s*(?:sq\.?\s*ft\.?|sqft|sf)\b", lambda m: m.group(1).replace(",", "")+" square feet", text, flags=re.I)
    
    def expand_abbreviations(self, text):
        for key in sorted(self.abbrev_map, key=len, reverse=True):
            pattern = re.escape(key).replace(r"\\/", r"\\s*/\\s*")
            text = re.sub(r"(?<!\w)"+pattern+r"(?!\w)", self.abbrev_map[key], text, flags=re.I)
        return text
    
    def normalize_whitespace(self, text):
        return re.sub(r"\s+", " ", text.replace("\\n", " ")).strip()
    
    def clean_text(self, text):
        text = self.normalize_unicode(text)
        text = self.remove_html(text)
        text = self.normalize_prices(text)
        text = self.normalize_measurements(text)
        text = self.expand_abbreviations(text)
        return self.normalize_whitespace(text)
    
    def profile_column(self, df, column_name):
        values = df[column_name].fillna("").astype(str)
        words = re.findall(r"\b[a-z]{2,}\b", " ".join(values.str.lower()))
        return {"null_rate": float(df[column_name].isna().mean()), "avg_length": float(values.str.len().mean()), "has_html": int(values.str.contains(r"<[^>]+>", regex=True).sum()), "price_mentions": int(values.str.contains(r"\$?\d+(?:\.\d+)?[km]\b", case=False, regex=True).sum()), "common_terms": Counter(words).most_common(20), "common_abbreviations": [key for key in self.abbrev_map if values.str.contains(re.escape(key), case=False, regex=True).any()]}

def main():
    cleaner = TextCleaner() 
    source = PROJECT_ROOT / "data/processed/listing_sample.csv"
    output = PROJECT_ROOT / "data/processed/listing_sample_cleaned.csv"
    report = PROJECT_ROOT / "data/processed/cleaning_profile_report.txt"

    df = pd.read_csv(source)
    profile = cleaner.profile_column(df, "remarks")
    df["remarks_cleaned"] = df["remarks"].apply(cleaner.clean_text)
    df.to_csv(output, index=False)
    report.write_text("\n".join([f"{k}: {v}" for k,v in profile.items()]), encoding="utf-8")
    print(f"Saved {len(df)} cleaned listings to {output}")


if __name__ == "__main__": main()
