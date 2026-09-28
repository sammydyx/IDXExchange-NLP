import json

import pandas as pd


def test_taxonomy_loaded():
    with open("data/processed/taxonomy.json") as file:
        taxonomy = json.load(file)

    assert len(taxonomy["terms"]) >= 200
    assert all("id" in term and "term" in term for term in taxonomy["terms"])


def test_sample_data_quality():
    dataframe = pd.read_csv("data/processed/listing_sample.csv")

    assert len(dataframe) >= 500
    assert dataframe["remarks"].str.len().min() > 50
