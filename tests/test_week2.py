import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
from scripts.text_cleaning import TextCleaner

cleaner = TextCleaner()

@pytest.mark.parametrize("raw,expected", [
    ("450k", "450000"), ("$450k", "450000"), ("1.2m", "1200000"), ("$2M", "2000000"),
    ("2,000 sqft", "2000 square feet"), ("900 sq ft", "900 square feet"), ("750 SF", "750 square feet"),
    ("3 br", "3 bedroom"), ("2 ba", "2 bathroom"), ("w/ pool", "with pool"), ("w/o hoa", "without homeowners association"),
    ("A/C", "air conditioning"), ("FP", "fireplace"), ("gar", "garage"), ("kit", "kitchen"),
    ("<b>Updated</b>", "Updated"), ("Tom &amp; Ana", "Tom & Ana"), ("hello—world", "hello-world"),
    ("  many   spaces \\n here ", "many spaces here"), (None, ""), ("", ""),
])
def test_clean_text_cases(raw, expected): assert cleaner.clean_text(raw) == expected

@pytest.mark.parametrize("key", list(TextCleaner().abbrev_map)[:21])
def test_abbreviation_map_has_working_entries(key): assert TextCleaner().abbrev_map[key]

def test_price_normalization(): assert "450000" in cleaner.normalize_prices("priced at 450k")
def test_measurement_normalization(): assert "2000 square feet" == cleaner.normalize_measurements("2,000 sqft")
def test_profiling():
    profile = cleaner.profile_column(pd.DataFrame({"remarks":["<b>3 br</b> $450k", None]}), "remarks")
    assert {"null_rate", "avg_length", "has_html", "price_mentions", "common_abbreviations"}.issubset(profile)
def test_profile_detects_html(): assert cleaner.profile_column(pd.DataFrame({"remarks":["<i>x</i>"]}), "remarks")["has_html"] == 1
